mod trainer;

use self::trainer::DqnTrainer;
use crate::algorithms::{
    ModelFreeSolver, SolveResult, SolveStatus, SolverSnapshot, validate_gamma,
};
use crate::environment::{Environment, Space};
use crate::evaluation::{EvaluationConfig, EvaluationMode};
use crate::policy::{DiscretePolicy, Policy, uniform::UniformPolicy};
use crate::replay_buffer::{ReplayBuffer, Transition};
use burn::lr_scheduler::linear::LinearLrSchedulerConfig;
use clap::{Args, ValueEnum};
use ndarray::Array1;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TargetUpdate {
    Hard { steps: usize },
    Soft { rate: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetUpdateMethod {
    Hard,
    Soft,
}

#[derive(Debug, Clone, PartialEq, Args, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DqnConfig {
    #[arg(long, default_value_t = DqnConfig::default().gamma)]
    pub gamma: f64,
    #[arg(long, default_value_t = DqnConfig::default().learning_rate)]
    pub learning_rate: f64,
    #[arg(long, default_value_t = DqnConfig::default().final_learning_rate)]
    pub final_learning_rate: f64,
    #[arg(long, default_value_t = DqnConfig::default().batch_size)]
    pub batch_size: usize,
    #[arg(long, default_value_t = DqnConfig::default().replay_capacity)]
    pub replay_capacity: usize,
    #[arg(long, default_value_t = DqnConfig::default().iterations)]
    pub iterations: usize,
    #[arg(long, value_enum, default_value_t = DqnConfig::default().target_update)]
    pub target_update: TargetUpdateMethod,
    #[arg(long, default_value_t = DqnConfig::default().target_sync_steps)]
    pub target_sync_steps: usize,
    #[arg(long, default_value_t = DqnConfig::default().target_update_rate)]
    pub target_update_rate: f64,
    #[arg(long, default_value_t = DqnConfig::default().hidden_dims)]
    pub hidden_dims: usize,
    #[arg(long, default_value_t = DqnConfig::default().evaluation_interval)]
    pub evaluation_interval: usize,
    #[arg(long, default_value_t = DqnConfig::default().evaluation_episodes)]
    pub evaluation_episodes: usize,
    #[arg(long, default_value_t = DqnConfig::default().evaluation_max_steps)]
    pub evaluation_max_steps: usize,
}

impl Default for DqnConfig {
    fn default() -> Self {
        Self {
            gamma: 0.9,
            learning_rate: 0.001,
            final_learning_rate: 0.0001,
            batch_size: 64,
            replay_capacity: 1_000,
            iterations: 100_000,
            target_update: TargetUpdateMethod::Hard,
            target_sync_steps: 500,
            target_update_rate: 0.005,
            hidden_dims: 64,
            evaluation_interval: 100,
            evaluation_episodes: 10,
            evaluation_max_steps: 1_000,
        }
    }
}

/// Deep Q learning algorithm using a Burn network.
pub struct DeepQLearning {
    config: DqnConfig,
    target_update: TargetUpdate,
    behavior_policy: Option<Box<dyn DiscretePolicy<Observation = Array1<f64>>>>,
}

impl DeepQLearning {
    pub fn new(config: DqnConfig) -> Result<Self, String> {
        validate_gamma(config.gamma)?;
        if !config.learning_rate.is_finite()
            || config.learning_rate <= 0.0
            || config.learning_rate > 1.0
        {
            return Err("learning rate must be finite, positive, and at most one".into());
        }
        if !config.final_learning_rate.is_finite()
            || config.final_learning_rate < 0.0
            || config.final_learning_rate > config.learning_rate
        {
            return Err(
                "final learning rate must be finite and between zero and the initial learning rate"
                    .into(),
            );
        }
        if config.batch_size == 0 || config.batch_size > config.replay_capacity {
            return Err("batch size must be positive and no greater than replay capacity".into());
        }
        if config.iterations == 0 || config.hidden_dims == 0 || config.evaluation_interval == 0 {
            return Err(
                "iterations, hidden dimensions, and evaluation interval must be positive".into(),
            );
        }
        let target_update = match config.target_update {
            TargetUpdateMethod::Hard => TargetUpdate::Hard {
                steps: config.target_sync_steps,
            },
            TargetUpdateMethod::Soft => TargetUpdate::Soft {
                rate: config.target_update_rate,
            },
        };
        match target_update {
            TargetUpdate::Hard { steps: 0 } => {
                return Err("target sync steps must be positive".into());
            }
            TargetUpdate::Soft { rate } if !rate.is_finite() || rate <= 0.0 || rate > 1.0 => {
                return Err("target update rate must be finite and between zero and one".into());
            }
            _ => {}
        }

        if config.evaluation_episodes == 0 || config.evaluation_max_steps == 0 {
            return Err("evaluation episodes and max steps must be positive".into());
        }
        Ok(Self {
            config,
            target_update,
            behavior_policy: None,
        })
    }

    pub fn gamma(&self) -> f64 {
        self.config.gamma
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

        // Collect the fixed off-policy dataset used for training.
        let mut buffer = ReplayBuffer::<Array1<f64>, usize>::new(self.config.replay_capacity);
        for _ in 0..buffer.capacity() {
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
            LinearLrSchedulerConfig::new(
                self.config.learning_rate,
                self.config.final_learning_rate,
                self.config.iterations.saturating_sub(1).max(1),
            )
            .init()?,
            rng.random(),
        );

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

        for iter in 0..self.config.iterations {
            let batch = buffer.sample(self.config.batch_size, rng);
            let (updated_trainer, loss, learning_rate) = trainer.update(&batch, self.config.gamma);
            trainer = updated_trainer;

            match self.target_update {
                TargetUpdate::Hard { steps } if (iter + 1) % steps == 0 => {
                    trainer.sync_target();
                }
                TargetUpdate::Soft { rate } => trainer.soft_update_target(rate),
                _ => {}
            }

            window_loss += loss;
            window_updates += 1;

            if (iter + 1) % self.config.evaluation_interval == 0
                || iter + 1 == self.config.iterations
            {
                let policy = trainer.policy();
                snapshots.push(
                    SolverSnapshot::new(iter + 1)
                        .learning_rate(learning_rate)
                        .average_loss(window_loss / window_updates as f64)
                        .evaluate_policy(env, &policy, evaluation, &mut evaluation_rng)?,
                );
                window_loss = 0.0;
                window_updates = 0;
            }
        }

        Ok(SolveResult {
            policy: Box::new(trainer.into_policy()),
            snapshots,
            status: SolveStatus::IterationLimit {
                iterations: self.config.iterations,
            },
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
