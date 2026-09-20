mod replay_buffer;
mod trainer;

use self::replay_buffer::{ReplayBuffer, Transition};
use self::trainer::DqnTrainer;
use crate::algorithms::{
    ModelFreeSolver, SolveResult, SolveStatus, SolverSnapshot, validate_gamma,
};
use crate::environment::{Environment, Space};
use crate::evaluation::{EvaluationConfig, EvaluationMode};
pub use crate::network::loss::Loss;
use crate::policy::{DiscretePolicy, Policy, uniform::UniformPolicy};
pub use crate::schedule::Schedule;
use clap::{Args, ValueEnum};
use ndarray::Array1;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::{io::IsTerminal, str::FromStr};

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
    pub evaluation_interval: usize,
    pub evaluation_episodes: usize,
    pub evaluation_max_steps: usize,
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
            evaluation_interval: 100,
            evaluation_episodes: 10,
            evaluation_max_steps: 1_000,
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
    #[arg(long, default_value_t = DqnConfig::default().evaluation_interval)]
    evaluation_interval: usize,
    #[arg(long, default_value_t = DqnConfig::default().evaluation_episodes)]
    evaluation_episodes: usize,
    #[arg(long, default_value_t = DqnConfig::default().evaluation_max_steps)]
    evaluation_max_steps: usize,
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
            evaluation_interval: args.evaluation_interval,
            evaluation_episodes: args.evaluation_episodes,
            evaluation_max_steps: args.evaluation_max_steps,
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

    fn solve(
        &self,
        env: &mut dyn Environment<Observation = Array1<f64>, Action = usize>,
        rng: &mut dyn rand::Rng,
    ) -> Result<SolveResult<Array1<f64>>, String> {
        let Space::Discrete { size: n_a } = env.action_space() else {
            return Err("DQN requires a discrete action space".into());
        };

        let mut s = env.reset(rng);
        let f_dim = s.len();
        let default_behavior_policy = UniformPolicy::new(n_a);
        let behavior_policy = self
            .behavior_policy
            .as_deref()
            .unwrap_or(&default_behavior_policy);

        // fill up initial replay buffer
        let mut buffer = ReplayBuffer::<Array1<f64>, usize>::new(self.config.replay_capacity);
        for _ in 0..self.config.warm_up_transitions {
            let a = behavior_policy.sample_action(&s, rng);
            let step = env.step(a, rng);
            let done = step.terminated || step.truncated;

            buffer.push(Transition {
                state: s.clone(),
                action: a,
                reward: step.reward,
                next_state: step.observation.clone(),
                terminated: step.terminated,
            });
            s = if done {
                env.reset(rng)
            } else {
                step.observation
            };
        }

        let mut trainer = DqnTrainer::new(
            f_dim,
            self.config.hidden_dims,
            n_a,
            self.config.gradient_clip_value,
            self.config.mode,
            rng.random(),
        );
        let mut pi = trainer.policy();

        let mut snapshots = Vec::new();
        let mut window_loss = 0.0;
        let mut window_updates = 0;
        let mut evaluation_rng = StdRng::seed_from_u64(rng.random());
        let evaluation = EvaluationConfig::new(
            self.config.evaluation_episodes,
            self.config.evaluation_max_steps,
            EvaluationMode::DiscountedReward {
                gamma: self.config.gamma,
            },
        )?;

        let mut iter = 0;
        let show_progress = std::io::stderr().is_terminal();
        if show_progress {
            print_episode_progress(0, self.config.episodes);
        }
        for episode in 0..self.config.episodes {
            let mut s = env.reset(rng);
            let learning_rate = self.config.learning_rate.value(episode);

            for step_index in 0.. {
                let epsilon = self.config.epsilon_schedule.value(iter);
                let a = pi.epsilon_greedy_action(&s, epsilon, rng);
                let step = env.step(a, rng);

                buffer.push(Transition {
                    state: s,
                    action: a,
                    reward: step.reward,
                    next_state: step.observation.clone(),
                    terminated: step.terminated,
                });

                s = step.observation;

                let batch = buffer.sample(self.config.batch_size, rng);
                let (new_trainer, loss) =
                    trainer.update(&batch, learning_rate, self.config.gamma, self.config.loss);
                trainer = new_trainer;
                pi = trainer.policy();

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
                window_loss += loss;
                window_updates += 1;

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

            if (episode + 1) % self.config.evaluation_interval == 0
                || episode + 1 == self.config.episodes
            {
                snapshots.push(
                    SolverSnapshot::new(iter)
                        .episode(episode + 1)
                        .learning_rate(learning_rate)
                        .average_loss(window_loss / window_updates as f64)
                        .evaluate_policy(env, &pi, evaluation, &mut evaluation_rng)?,
                );
                window_loss = 0.0;
                window_updates = 0;
            }
            if show_progress {
                print_episode_progress(episode + 1, self.config.episodes);
            }
        }

        Ok(SolveResult {
            policy: Box::new(trainer.into_policy()),
            snapshots,
            status: SolveStatus::IterationLimit { iterations: iter },
        })
    }
}

fn print_episode_progress(episode: usize, episodes: usize) {
    const WIDTH: usize = 30;
    let filled = WIDTH * episode / episodes;
    let bar = format!("{}{}", "=".repeat(filled), " ".repeat(WIDTH - filled));
    eprint!("\rEpisodes [{bar}] {episode}/{episodes}");
    if episode == episodes {
        eprintln!();
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
