pub use crate::network::loss::Loss;
pub use crate::schedule::Schedule;

use crate::algorithms::{ModelFreeSolver, SolveResult, SolveStatus, validate_gamma};
use crate::backend::{TrainingBackend, device};
use crate::environment::{Environment, Rollout, Space};
use crate::network::MlpSettings;
use crate::network::{AdamOptimizer, MlpModel, ema, gradient_norm};
use crate::policy::dq::{DeepQPolicy, DeepQPolicyMode};
use crate::policy::{DiscretePolicy, uniform::UniformPolicy};
use crate::reporting::{Reporter, Step};
use crate::transition_buffer::TransitionBuffer;
use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer, grad_clipping::GradientClippingConfig},
    tensor::{ElementConversion, backend::Backend},
};
use clap::ValueEnum;
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
    pub q_network: MlpSettings,
    pub batch_size: usize,
    pub replay_capacity: usize,
    pub warm_up_transitions: usize,
    pub episodes: usize,
    pub episode_length: Option<usize>,
    pub epsilon_schedule: Schedule,
    pub target_update: TargetUpdateConfig,
    pub loss: Loss,
    pub gradient_clip_value: Option<f32>,
}

impl Default for DqnConfig {
    fn default() -> Self {
        Self {
            mode: DqnMode::default(),
            gamma: 0.9,
            q_network: MlpSettings {
                learning_rate: Schedule::Constant { value: 0.001 },
                ..MlpSettings::default()
            },
            batch_size: 64,
            replay_capacity: 1_000,
            warm_up_transitions: 1_000,
            episodes: 50,
            episode_length: None,
            epsilon_schedule: Schedule::default(),
            target_update: TargetUpdateConfig::default(),
            loss: Loss::default(),
            gradient_clip_value: None,
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
    type Action = usize;

    fn solve_with_reporter(
        &self,
        env: &mut dyn Environment<Observation = Array1<f64>, Action = usize>,
        rng: &mut dyn rand::Rng,
        reporter: &mut Reporter<'_, Array1<f64>>,
    ) -> Result<SolveResult<Array1<f64>>, String> {
        let Space::Discrete { size: n_a } = env.action_space() else {
            return Err("DQN requires a discrete action space".into());
        };

        <TrainingBackend as Backend>::seed(device(), rng.random());

        let mut rollout = Rollout::new(env, rng);
        let f_dim = rollout.observation().len();
        let default_behavior_policy = UniformPolicy::new(n_a);
        let behavior_policy = self
            .behavior_policy
            .as_deref()
            .unwrap_or(&default_behavior_policy);

        // fill up initial replay buffer
        let mut buffer = TransitionBuffer::new(Some(self.config.replay_capacity));
        buffer.fill_n(
            self.config.warm_up_transitions,
            &mut rollout,
            behavior_policy,
            rng,
        );

        let mut q_pred = self.config.q_network.init(f_dim, n_a);
        let mut q_target = q_pred.clone();
        let mut optimizer: AdamOptimizer<MlpModel<TrainingBackend>> = AdamConfig::new()
            .with_grad_clipping(
                self.config
                    .gradient_clip_value
                    .map(GradientClippingConfig::Value),
            )
            .init();

        let mut iter = 0;
        for episode in 0..self.config.episodes {
            let learning_rate = self.config.q_network.learning_rate.value(episode);
            let mut episode_return = 0.0;
            let mut episode_loss = 0.0;
            let mut episode_gradient_norm = 0.0;
            let mut episode_length = 0;

            for step_index in 0.. {
                let epsilon = self.config.epsilon_schedule.value(iter);
                let behavior_policy =
                    DeepQPolicy::borrowed(&q_pred, DeepQPolicyMode::SoftGreedy { epsilon });
                let transition = buffer.fill_one(&mut rollout, &behavior_policy, rng);
                episode_return += transition.r;
                episode_length = step_index + 1;
                let done = transition.terminated || transition.truncated;

                let batch = buffer.random_batch_tensor_data(self.config.batch_size, rng);

                let predicted = q_pred
                    .forward(batch.s)
                    .gather(1, batch.a.reshape([-1, 1]))
                    .squeeze_dim(1);
                let actions = match self.config.mode {
                    DqnMode::Dqn => q_target.forward(batch.sp.clone()).argmax(1),
                    DqnMode::Ddqn => q_pred.forward(batch.sp.clone()).argmax(1),
                };
                let next_values = q_target.forward(batch.sp).gather(1, actions).squeeze_dim(1);
                let td_target = batch.r
                    + batch.terminated.bool_not().float()
                        * self.config.gamma
                        * next_values.detach();

                let loss = self.config.loss.forward(predicted, td_target);
                let loss_value = loss.clone().into_scalar().elem::<f64>();
                let gradients = GradientsParams::from_grads(loss.backward(), &q_pred);
                let gradient_norm = gradient_norm(&q_pred, &gradients);

                q_pred = optimizer.step(learning_rate, q_pred, gradients);

                match self.config.target_update {
                    TargetUpdateConfig::Hard { steps } if (iter + 1) % steps == 0 => {
                        q_target = q_pred.clone();
                    }
                    TargetUpdateConfig::Soft { rate } => {
                        q_target = ema::update(q_target, &q_pred, rate);
                    }
                    _ => {}
                }

                iter += 1;
                episode_loss += loss_value;
                episode_gradient_norm += gradient_norm;
                if done {
                    break;
                }
                if self.config.episode_length == Some(step_index + 1) {
                    buffer.truncate_episode(&mut rollout, rng);
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
            reporter.policy(step, || {
                DeepQPolicy::new(q_pred.valid(), DeepQPolicyMode::Greedy)
            });
        }

        Ok(SolveResult {
            policy: Box::new(DeepQPolicy::new(q_pred.valid(), DeepQPolicyMode::Greedy)),
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
