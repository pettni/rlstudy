mod trainer;

use self::trainer::DqnTrainer;
use crate::algorithms::{ModelFreeSolver, SolveResult, SolveStatus, validate_gamma};
use crate::environment::{Environment, Rollout, Space};
pub use crate::network::loss::Loss;
use crate::policy::{
    DiscretePolicy,
    dq::{DeepQPolicy, DeepQPolicyMode},
    uniform::UniformPolicy,
};
use crate::reporting::{AbstractReporter, Step};
pub use crate::schedule::Schedule;
use crate::transition_buffer::{Transition, TransitionBuffer};
use clap::{Args, ValueEnum};
use ndarray::Array1;
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DqnMode {
    #[default]
    /// actions from target network evaluated by target network (optimistic bias risk)
    Dqn,
    /// actions from online network evaluated by target network
    Ddqn,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum TargetUpdateConfig {
    Hard { steps: usize },
    Soft { rate: f64 },
}

impl Default for TargetUpdateConfig {
    fn default() -> Self {
        Self::Hard { steps: 500 }
    }
}

// Clap's ValueEnum only supports fieldless variants, so parse this data-carrying enum directly.
impl FromStr for TargetUpdateConfig {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "hard" => Ok(Self::default()),
            "soft" => Ok(Self::Soft { rate: 0.005 }),
            _ => Err("expected hard or soft".into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DqnConfig {
    pub mode: DqnMode,
    pub gamma: f64,
    pub learning_rate: Schedule,
    pub batch_size: usize,
    pub replay_capacity: usize,
    pub warm_up_transitions: usize,
    pub episodes: usize,
    pub episode_length: Option<usize>,
    pub epsilon_schedule: Schedule,
    pub target_update: TargetUpdateConfig,
    pub loss: Loss,
    pub gradient_clip_value: Option<f32>,
    pub hidden_dims: usize,
}

impl Default for DqnConfig {
    fn default() -> Self {
        Self {
            mode: DqnMode::default(),
            gamma: 0.9,
            learning_rate: Schedule::Constant { value: 0.001 },
            batch_size: 64,
            replay_capacity: 1_000,
            warm_up_transitions: 1_000,
            episodes: 50,
            episode_length: None,
            epsilon_schedule: Schedule::default(),
            target_update: TargetUpdateConfig::default(),
            loss: Loss::default(),
            gradient_clip_value: None,
            hidden_dims: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Args)]
pub struct DqnArgs {
    #[arg(long, value_enum, default_value_t = DqnConfig::default().mode)]
    mode: DqnMode,
    #[arg(long, default_value_t = DqnConfig::default().gamma)]
    gamma: f64,
    #[arg(long)]
    learning_rate: Option<Schedule>,
    #[arg(long, default_value_t = DqnConfig::default().batch_size)]
    batch_size: usize,
    #[arg(long, default_value_t = DqnConfig::default().replay_capacity)]
    replay_capacity: usize,
    #[arg(long, default_value_t = DqnConfig::default().warm_up_transitions)]
    warm_up_transitions: usize,
    #[arg(long, default_value_t = DqnConfig::default().episodes)]
    episodes: usize,
    #[arg(long)]
    episode_length: Option<usize>,
    #[arg(long)]
    epsilon_schedule: Option<Schedule>,
    #[arg(long)]
    target_update: Option<TargetUpdateConfig>,
    #[arg(long, value_enum, default_value_t = DqnConfig::default().loss)]
    loss: Loss,
    #[arg(long)]
    gradient_clip_value: Option<f32>,
    #[arg(long, default_value_t = DqnConfig::default().hidden_dims)]
    hidden_dims: usize,
}

impl From<DqnArgs> for DqnConfig {
    fn from(args: DqnArgs) -> Self {
        Self {
            mode: args.mode,
            gamma: args.gamma,
            learning_rate: args
                .learning_rate
                .unwrap_or(DqnConfig::default().learning_rate),
            batch_size: args.batch_size,
            replay_capacity: args.replay_capacity,
            warm_up_transitions: args.warm_up_transitions,
            episodes: args.episodes,
            episode_length: args.episode_length,
            epsilon_schedule: args.epsilon_schedule.unwrap_or_default(),
            target_update: args.target_update.unwrap_or_default(),
            loss: args.loss,
            gradient_clip_value: args.gradient_clip_value,
            hidden_dims: args.hidden_dims,
        }
    }
}

/// Deep Q learning algorithm using a Burn network.
pub struct DeepQLearning {
    config: DqnConfig,
    behavior_policy: Option<Box<dyn DiscretePolicy<Observation = Array1<f64>>>>,
}

impl DeepQLearning {
    pub fn new(config: DqnConfig) -> Result<Self, String> {
        validate_gamma(config.gamma)?;
        Ok(Self {
            config,
            behavior_policy: None,
        })
    }

    pub fn with_behavior_policy(
        mut self,
        policy: Box<dyn DiscretePolicy<Observation = Array1<f64>>>,
    ) -> Self {
        self.behavior_policy = Some(policy);
        self
    }
}

impl ModelFreeSolver for DeepQLearning {
    type Observation = Array1<f64>;

    fn solve_with_reporter(
        &self,
        env: &mut dyn Environment<Observation = Array1<f64>, Action = usize>,
        rng: &mut dyn rand::Rng,
        reporter: &mut dyn AbstractReporter<Array1<f64>>,
    ) -> Result<SolveResult<Array1<f64>>, String> {
        let Space::Discrete { size: n_a } = env.action_space() else {
            return Err("DQN requires a discrete action space".into());
        };

        let mut rollout = Rollout::new(env, rng);
        let f_dim = rollout.observation().len();
        let default_behavior_policy = UniformPolicy::new(n_a);
        let behavior_policy = self
            .behavior_policy
            .as_deref()
            .unwrap_or(&default_behavior_policy);

        // fill up initial replay buffer
        let mut buffer = TransitionBuffer::new(Some(self.config.replay_capacity));
        buffer.fill(
            &mut rollout,
            behavior_policy,
            rng,
            self.config.warm_up_transitions,
        );

        let mut trainer = DqnTrainer::new(
            f_dim,
            self.config.hidden_dims,
            n_a,
            self.config.gradient_clip_value,
            self.config.mode,
            rng.random(),
        );
        let mut iter = 0;
        for episode in 0..self.config.episodes {
            rollout.reset(rng);
            let learning_rate = self.config.learning_rate.value(episode);
            let mut episode_return = 0.0;
            let mut episode_loss = 0.0;
            let mut episode_gradient_norm = 0.0;
            let mut episode_length = 0;

            for step_index in 0.. {
                let epsilon = self.config.epsilon_schedule.value(iter);
                let a = trainer.policy_view().epsilon_greedy_action(
                    rollout.observation(),
                    epsilon,
                    rng,
                );
                let s = rollout.observation().clone();
                let step = rollout.step(a, rng);
                episode_return += step.reward;
                episode_length = step_index + 1;

                buffer.push(Transition {
                    s,
                    a,
                    r: step.reward,
                    sp: step.observation,
                    terminated: step.terminated,
                    truncated: step.truncated,
                });

                let batch = buffer.random_batch_tensor_data(self.config.batch_size, rng);
                let (new_trainer, loss, gradient_norm) =
                    trainer.update(batch, learning_rate, self.config.gamma, self.config.loss);
                trainer = new_trainer;

                match self.config.target_update {
                    TargetUpdateConfig::Hard { steps } if (iter + 1) % steps == 0 => {
                        trainer.sync_target();
                    }
                    TargetUpdateConfig::Soft { rate } => {
                        trainer.soft_update_target(rate);
                    }
                    _ => {}
                }

                iter += 1;
                episode_loss += loss;
                episode_gradient_norm += gradient_norm;
                if step.terminated
                    || step.truncated
                    || self
                        .config
                        .episode_length
                        .is_some_and(|length| step_index + 1 == length)
                {
                    break;
                }
            }
            let step = Step::Episode(episode + 1);
            reporter.record(
                step,
                &[
                    ("train/return", episode_return),
                    ("train/length", episode_length as f64),
                    ("loss", episode_loss / episode_length as f64),
                    ("grad/norm", episode_gradient_norm / episode_length as f64),
                    ("optimizer/lr", learning_rate),
                ],
            );
            let mut make_policy = || {
                Box::new(DeepQPolicy::new(
                    trainer.inference_model(),
                    DeepQPolicyMode::Greedy,
                )) as Box<dyn DiscretePolicy<Observation = Array1<f64>>>
            };
            reporter.policy(step, &mut make_policy);
        }

        Ok(SolveResult {
            policy: Box::new(trainer.into_policy()),
            status: SolveStatus::IterationLimit { iterations: iter },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_behavior_policy() {
        let solver = DeepQLearning::new(DqnConfig::default())
            .unwrap()
            .with_behavior_policy(Box::new(UniformPolicy::new(2)));

        assert!(solver.behavior_policy.is_some());
    }
}
