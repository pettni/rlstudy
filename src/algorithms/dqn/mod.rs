mod trainer;

use self::trainer::DqnTrainer;
use crate::algorithms::{ModelFreeSolver, SolveResult, SolveStatus};
use crate::environment::{Environment, Space};
use crate::evaluation::{EvaluationConfig, EvaluationMode, evaluate_policy};
use crate::policy::{DiscretePolicy, Policy};
use crate::replay_buffer::{ReplayBuffer, Transition};
use burn::lr_scheduler::linear::LinearLrSchedulerConfig;
use ndarray::Array1;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DqnSnapshot {
    pub iteration: usize,
    pub learning_rate: f64,
    pub average_loss: f64,
    pub evaluation_reward: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TargetUpdate {
    Hard { steps: usize },
    Soft { rate: f64 },
}

/// Deep Q learning algorithm using a Burn network.
pub struct DeepQLearning {
    learning_rate: f64,
    final_learning_rate: f64,
    batch_size: usize,
    replay_capacity: usize,
    iterations: usize,
    target_update: TargetUpdate,
    num_hidden_dims: usize,
    evaluation_interval: usize,
    evaluation: EvaluationConfig,
    behavior_policy: Box<dyn DiscretePolicy<Observation = Array1<f64>>>,
}

impl DeepQLearning {
    pub fn new(
        learning_rate: f64,
        final_learning_rate: f64,
        batch_size: usize,
        replay_capacity: usize,
        iterations: usize,
        target_update: TargetUpdate,
        num_hidden_dims: usize,
        evaluation_interval: usize,
        evaluation: EvaluationConfig,
        behavior_policy: Box<dyn DiscretePolicy<Observation = Array1<f64>>>,
    ) -> Result<Self, String> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 || learning_rate > 1.0 {
            return Err("learning rate must be finite, positive, and at most one".into());
        }
        if !final_learning_rate.is_finite()
            || final_learning_rate < 0.0
            || final_learning_rate > learning_rate
        {
            return Err(
                "final learning rate must be finite and between zero and the initial learning rate"
                    .into(),
            );
        }
        if batch_size == 0 || batch_size > replay_capacity {
            return Err("batch size must be positive and no greater than replay capacity".into());
        }
        if iterations == 0 || num_hidden_dims == 0 || evaluation_interval == 0 {
            return Err(
                "iterations, hidden dimensions, and evaluation interval must be positive".into(),
            );
        }
        match target_update {
            TargetUpdate::Hard { steps: 0 } => {
                return Err("target sync steps must be positive".into());
            }
            TargetUpdate::Soft { rate } if !rate.is_finite() || rate <= 0.0 || rate > 1.0 => {
                return Err("target update rate must be finite and between zero and one".into());
            }
            _ => {}
        }

        Ok(Self {
            learning_rate,
            final_learning_rate,
            batch_size,
            replay_capacity,
            iterations,
            target_update,
            num_hidden_dims,
            evaluation_interval,
            evaluation,
            behavior_policy,
        })
    }
}

impl ModelFreeSolver for DeepQLearning {
    type Observation = Array1<f64>;
    type Output = Vec<DqnSnapshot>;

    fn solve(
        &self,
        env: &mut dyn Environment<Observation = Array1<f64>, Action = usize>,
        gamma: f64,
        rng: &mut dyn rand::Rng,
    ) -> Result<SolveResult<Array1<f64>, Vec<DqnSnapshot>>, String> {
        let Space::Discrete { size: n_a } = env.action_space() else {
            return Err("DQN requires a discrete action space".into());
        };

        let mut s = env.reset(rng);
        let f_dim = s.len();

        // create replay buffer (s,a,r,s') from behavior policy
        let mut buffer = ReplayBuffer::<Array1<f64>, usize>::new(self.replay_capacity);
        for _ in 0..buffer.capacity() {
            let a = self.behavior_policy.sample_action(&s, rng);
            let (sp, r) = env.step(a, rng);

            buffer.push(Transition {
                state: s.clone(),
                action: a,
                reward: r,
                next_state: sp.clone(),
                terminated: false,
            });
            s = sp;
        }

        let mut trainer = DqnTrainer::new(
            f_dim,
            self.num_hidden_dims,
            n_a,
            self.batch_size,
            LinearLrSchedulerConfig::new(
                self.learning_rate,
                self.final_learning_rate,
                self.iterations.saturating_sub(1).max(1),
            )
            .init()?,
        );

        let mut snapshots = Vec::new();
        let mut window_loss = 0.0;
        let mut evaluation_rng = StdRng::seed_from_u64(rng.random());

        for iter in 0..self.iterations {
            let batch = buffer.sample(self.batch_size, rng);
            let (updated_trainer, loss, learning_rate) = trainer.update(&batch, gamma);
            trainer = updated_trainer;

            match self.target_update {
                TargetUpdate::Hard { steps } if (iter + 1) % steps == 0 => {
                    trainer.sync_target();
                }
                TargetUpdate::Soft { rate } => trainer.soft_update_target(rate),
                _ => {}
            }

            window_loss += loss;

            if (iter + 1) % self.evaluation_interval == 0 {
                let policy = trainer.policy();
                snapshots.push(DqnSnapshot {
                    iteration: iter + 1,
                    learning_rate,
                    average_loss: window_loss / self.evaluation_interval as f64,
                    evaluation_reward: evaluate_policy(
                        env,
                        &policy,
                        self.evaluation,
                        EvaluationMode::DiscountedReward { gamma },
                        &mut evaluation_rng,
                    )?,
                });
                window_loss = 0.0;
            }
        }

        Ok(SolveResult {
            policy: Box::new(trainer.into_policy()),
            output: snapshots,
            status: SolveStatus::IterationLimit {
                iterations: self.iterations,
            },
        })
    }
}
