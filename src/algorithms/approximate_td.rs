use std::collections::VecDeque;

use crate::algorithms::{ModelFreeSolver, SolveResult, SolveStatus};
use crate::backend::{TrainingBackend, device};
use crate::environment::{Environment, Space};
use crate::network::{AdamOptimizer, MlpModel, MlpSettings, gradient_norm};
use crate::policy::{
    Policy,
    dq::{DeepQPolicy, DeepQPolicyMode},
};
use crate::reporting::{Reporter, Step};
use crate::schedule::Schedule;
use burn::{
    Tensor,
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{Int, TensorData, backend::Backend},
};
use ndarray::Array1;
use rand::RngExt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ApproximateTDConfig {
    pub n: usize,
    pub gamma: f64,
    pub epsilon: f64,
    /// Scale rewards in TD targets; reported returns keep their original units.
    pub reward_scale: f64,
    pub q_network: MlpSettings,
    pub episodes: usize,
    pub episode_length: usize,
}

impl Default for ApproximateTDConfig {
    fn default() -> Self {
        Self {
            n: 1,
            gamma: 0.9,
            epsilon: 0.1,
            reward_scale: 1.0,
            q_network: MlpSettings {
                learning_rate: Schedule::Linear {
                    initial: 0.001,
                    final_value: 0.0001,
                    steps: 999,
                },
                ..MlpSettings::default()
            },
            episodes: 1_000,
            episode_length: 100,
        }
    }
}

pub struct ApproximateTD {
    config: ApproximateTDConfig,
}

impl ApproximateTD {
    pub fn new(config: ApproximateTDConfig) -> Result<Self, String> {
        if config.n == 0 {
            return Err("n must be positive".into());
        }
        if !config.gamma.is_finite() || !(0.0..=1.0).contains(&config.gamma) {
            return Err("gamma must be finite and in [0, 1]".into());
        }
        if !config.epsilon.is_finite() || !(0.0..=1.0).contains(&config.epsilon) {
            return Err("epsilon must be finite and in [0, 1]".into());
        }
        if config.q_network.hidden_dims.is_empty() || config.q_network.hidden_dims.contains(&0) {
            return Err("hidden dims must be positive".into());
        }
        if !config.reward_scale.is_finite() || config.reward_scale <= 0.0 {
            return Err("reward scale must be finite and positive".into());
        }
        if config.episodes == 0 || config.episode_length == 0 {
            return Err("episodes and episode length must be positive".into());
        }
        Ok(Self { config })
    }
}

impl ModelFreeSolver for ApproximateTD {
    type Observation = Array1<f64>;
    type Action = usize;

    fn solve_with_reporter(
        &self,
        env: &mut dyn Environment<Observation = Array1<f64>, Action = usize>,
        rng: &mut dyn rand::Rng,
        reporter: &mut Reporter<'_, Array1<f64>>,
    ) -> Result<SolveResult<Array1<f64>>, String> {
        let n_s = env
            .state_space()
            .continuous_dim()
            .ok_or("Approximate TD requires a continuous state space")?;
        let Space::Discrete { size: n_a } = env.action_space() else {
            return Err("Approximate TD requires a discrete action space".into());
        };

        <TrainingBackend as Backend>::seed(device(), rng.random());

        // q(s) = logits over actions
        let mut q = self.config.q_network.init(n_s, n_a);
        let mut optimizer: AdamOptimizer<MlpModel<TrainingBackend>> = AdamConfig::new().init();

        for episode_index in 0..self.config.episodes {
            let mut state = env.reset(rng);
            let mut action = DeepQPolicy::borrowed(
                &q,
                DeepQPolicyMode::SoftGreedy {
                    epsilon: self.config.epsilon,
                },
            )
            .sample_action(&state, rng);
            let mut episode_return = 0.0;
            let mut episode_length = 0;
            let mut episode_loss = 0.0;
            let mut episode_gradient_norm = 0.0;

            let learning_rate = self.config.q_network.learning_rate.value(episode_index);

            let mut terminated = false;

            let mut pending = VecDeque::with_capacity(self.config.n);
            let mut bootstrap = None;

            for _ in 0..self.config.episode_length + self.config.n {
                if !terminated {
                    let step = env.step(action, rng);

                    pending.push_back((
                        state.clone(),
                        action,
                        step.reward * self.config.reward_scale,
                    ));
                    episode_return += step.reward;
                    episode_length += 1;

                    if step.terminated
                        || step.truncated
                        || episode_length == self.config.episode_length
                    {
                        terminated = true;
                    }

                    if step.terminated {
                        bootstrap = None;
                    } else {
                        state = step.observation;
                        action = DeepQPolicy::borrowed(
                            &q,
                            DeepQPolicyMode::SoftGreedy {
                                epsilon: self.config.epsilon,
                            },
                        )
                        .sample_action(&state, rng);
                        bootstrap = Some((state.clone(), action));
                    }
                }

                if pending.len() >= self.config.n || (terminated && !pending.is_empty()) {
                    let (old_state, old_action, _) = pending.front().unwrap();
                    let t_s_old = Tensor::<TrainingBackend, 2>::from_data(
                        TensorData::new(old_state.to_vec(), [1, n_s]),
                        device(),
                    );
                    let t_a_old = Tensor::<TrainingBackend, 2, Int>::from_data(
                        TensorData::new(vec![*old_action as i64], [1, 1]),
                        device(),
                    );

                    // g = n-step return + bootstrap tail
                    let g: f64 = pending
                        .iter()
                        .enumerate()
                        .map(|(i, (_, _, reward))| self.config.gamma.powi(i as i32) * reward)
                        .sum();

                    // bootstrap tail
                    let bootstrap_tail = if let Some((next_state, next_action)) = &bootstrap {
                        let t_s_new = Tensor::<TrainingBackend, 2>::from_data(
                            TensorData::new(next_state.to_vec(), [1, n_s]),
                            device(),
                        );
                        let t_a_new = Tensor::<TrainingBackend, 2, Int>::from_data(
                            TensorData::new(vec![*next_action as i64], [1, 1]),
                            device(),
                        );
                        let q_bootstrap = q.forward(t_s_new).gather(1, t_a_new);

                        self.config.gamma.powi(pending.len() as i32) * q_bootstrap
                    } else {
                        Tensor::<TrainingBackend, 2>::zeros([1, 1], device())
                    };

                    // optimize td error
                    let td_target: Tensor<TrainingBackend, 2> = g + bootstrap_tail;
                    let td_error = td_target.detach() - q.forward(t_s_old).gather(1, t_a_old);

                    let loss = td_error.square();
                    episode_loss += loss
                        .clone()
                        .into_data()
                        .iter::<f64>()
                        .next()
                        .expect("loss tensor has one element");
                    let grad = GradientsParams::from_grads(loss.backward(), &q);
                    episode_gradient_norm += gradient_norm(&q, &grad);
                    q = optimizer.step(learning_rate, q, grad);
                    pending.pop_front();
                }
                if terminated && pending.is_empty() {
                    break;
                }
            }

            let step = Step::Episode(episode_index + 1);
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
                DeepQPolicy::new(q.valid(), DeepQPolicyMode::Greedy)
            });
        }

        Ok(SolveResult {
            policy: Box::new(DeepQPolicy::new(q.valid(), DeepQPolicyMode::Greedy)),
            status: SolveStatus::IterationLimit {
                iterations: self.config.episodes,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::FeatureEnvironment;
    use crate::environments::grid_world;
    use crate::reporting::{Reporter, ReporterConfig};
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn trains_when_n_exceeds_episode_length() {
        let model = grid_world(3).unwrap();
        let mut environment = FeatureEnvironment::new(&model).unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        let solver = ApproximateTD::new(ApproximateTDConfig {
            n: 3,
            episodes: 1,
            episode_length: 2,
            ..ApproximateTDConfig::default()
        })
        .unwrap();
        let mut reporter = Reporter::new(ReporterConfig::default(), 0, None);

        solver
            .solve_with_reporter(&mut environment, &mut rng, &mut reporter)
            .unwrap();

        assert_eq!(reporter.into_history().len(), 1);
    }
}
