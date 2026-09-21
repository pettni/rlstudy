use crate::algorithms::{ModelFreeSolver, SolveResult, SolveStatus};
use crate::environment::{Environment, Space};
use crate::network::mlp::MlpModel;
use crate::policy::{
    Policy,
    dq::{DeepQPolicy, DeepQPolicyMode},
};
use crate::reporting::{AbstractReporter, Step};
use crate::schedule::Schedule;
use burn::{
    Tensor,
    backend::{Autodiff, Flex, flex::FlexDevice},
    module::AutodiffModule,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Int, TensorData, activation::log_softmax, backend::Backend},
};
use clap::Args;
use ndarray::Array1;
use rand::RngExt;
use serde::{Deserialize, Serialize};

type TrainingBackend = Autodiff<Flex>;

#[derive(Debug, Clone, PartialEq, Args, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ActorCriticConfig {
    #[arg(long, default_value_t = ActorCriticConfig::default().gamma)]
    pub gamma: f64,
    #[arg(long, default_value = "linear:0.001:0.0001:999")]
    pub actor_learning_rate: Schedule,
    #[arg(long, default_value = "linear:0.001:0.0001:999")]
    pub critic_learning_rate: Schedule,
    #[arg(long, default_value_t = ActorCriticConfig::default().hidden_dims)]
    pub hidden_dims: usize,
    #[arg(long, default_value_t = ActorCriticConfig::default().episodes)]
    pub episodes: usize,
    #[arg(long, default_value_t = ActorCriticConfig::default().episode_length)]
    pub episode_length: usize,
}

impl Default for ActorCriticConfig {
    fn default() -> Self {
        Self {
            gamma: 0.9,
            actor_learning_rate: Schedule::Linear {
                initial: 0.001,
                final_value: 0.0001,
                steps: 999,
            },
            critic_learning_rate: Schedule::Linear {
                initial: 0.001,
                final_value: 0.0001,
                steps: 999,
            },
            hidden_dims: 64,
            episodes: 1_000,
            episode_length: 100,
        }
    }
}

pub struct ActorCritic {
    config: ActorCriticConfig,
}

impl ActorCritic {
    pub fn new(config: ActorCriticConfig) -> Result<Self, String> {
        if !config.gamma.is_finite() || !(0.0..=1.0).contains(&config.gamma) {
            return Err("gamma must be finite and in [0, 1]".into());
        }
        let valid_rate = |rate: f64| rate.is_finite() && rate > 0.0 && rate <= 1.0;
        for schedule in [&config.actor_learning_rate, &config.critic_learning_rate] {
            let valid = match schedule {
                Schedule::Constant { value } => valid_rate(*value),
                Schedule::Linear {
                    initial,
                    final_value,
                    steps,
                } => valid_rate(*initial) && valid_rate(*final_value) && *steps > 0,
                Schedule::Exponential {
                    initial,
                    final_value,
                    decay,
                } => valid_rate(*initial) && valid_rate(*final_value) && *decay > 0,
                Schedule::Stepwise { initial, changes } => {
                    valid_rate(*initial)
                        && changes.iter().all(|(_, rate)| valid_rate(*rate))
                        && changes.windows(2).all(|pair| pair[0].0 < pair[1].0)
                }
            };
            if !valid {
                return Err(
                    "learning rates must be finite and in (0, 1], with increasing steps".into(),
                );
            }
        }
        if config.hidden_dims == 0 {
            return Err("hidden dims must be positive".into());
        }
        if config.episodes == 0 || config.episode_length == 0 {
            return Err("episodes and episode length must be positive".into());
        }
        Ok(Self { config })
    }
}

impl ModelFreeSolver for ActorCritic {
    type Observation = Array1<f64>;

    fn solve_with_reporter(
        &self,
        env: &mut dyn Environment<Observation = Array1<f64>, Action = usize>,
        rng: &mut dyn rand::Rng,
        reporter: &mut dyn AbstractReporter<Array1<f64>>,
    ) -> Result<SolveResult<Array1<f64>>, String> {
        let Space::Continuous { dim: n_s } = env.state_space() else {
            return Err("Actor-critic requires a continuous state space".into());
        };
        let Space::Discrete { size: n_a } = env.action_space() else {
            return Err("Actor-critic requires a discrete action space".into());
        };

        let device = FlexDevice;
        <TrainingBackend as Backend>::seed(&device, rng.random());

        let mut pi_optimizer: OptimizerAdaptor<Adam, MlpModel<TrainingBackend>, TrainingBackend> =
            AdamConfig::new().init();
        let mut v_optimizer = pi_optimizer.clone();

        let mut pi = MlpModel::new(n_s, self.config.hidden_dims, n_a, &device);
        let mut v = MlpModel::new(n_s, self.config.hidden_dims, 1, &device);

        for episode_index in 0..self.config.episodes {
            let mut state = env.reset(rng);
            let mut episode_return = 0.0;
            let mut episode_length = 0;
            let mut entropy_sum = 0.0;
            let mut actor_gradient_norm_sum = 0.0;
            let mut critic_gradient_norm_sum = 0.0;
            let actor_learning_rate = self.config.actor_learning_rate.value(episode_index);
            let critic_learning_rate = self.config.critic_learning_rate.value(episode_index);
            for t in 0..self.config.episode_length {
                let action =
                    DeepQPolicy::borrowed(&pi, DeepQPolicyMode::SoftMax).sample_action(&state, rng);
                let step = env.step(action, rng);

                let st = Tensor::<TrainingBackend, 2>::from_data(
                    TensorData::new(state.to_vec(), [1, n_s]),
                    &device,
                );
                let at = Tensor::<TrainingBackend, 2, Int>::from_data(
                    TensorData::new(vec![action as i64], [1, 1]),
                    &device,
                );
                let stp = Tensor::<TrainingBackend, 2>::from_data(
                    TensorData::new(step.observation.to_vec(), [1, n_s]),
                    &device,
                );

                let live = if step.terminated { 0.0 } else { 1.0 };

                let td_target = step.reward + live * self.config.gamma * v.forward(stp).detach();
                let td_error = td_target - v.forward(st.clone());

                // critic update
                let critic_loss = td_error.clone().square();
                let critic_grad = GradientsParams::from_grads(critic_loss.backward(), &v);

                critic_gradient_norm_sum += v.gradient_norm(&critic_grad);

                v = v_optimizer.step(critic_learning_rate, v, critic_grad);

                // actor update
                let log_probs = log_softmax(pi.forward(st), 1);
                let ln_pi = log_probs.clone().gather(1, at);
                let actor_loss = (td_error.clone().detach() * ln_pi).neg();
                let actor_grad = GradientsParams::from_grads(actor_loss.backward(), &pi);

                actor_gradient_norm_sum += pi.gradient_norm(&actor_grad);

                pi = pi_optimizer.step(actor_learning_rate, pi, actor_grad);

                // metrics
                entropy_sum += (log_probs.clone().exp() * log_probs)
                    .sum()
                    .neg()
                    .into_data()
                    .iter::<f64>()
                    .next()
                    .expect("entropy tensor has one element");
                episode_return += step.reward;
                episode_length = t + 1;

                // loop state
                state = step.observation;
                if step.terminated || step.truncated {
                    break;
                }
            }

            reporter.record(
                Step::Episode(episode_index + 1),
                &[
                    ("train/return", episode_return),
                    ("train/length", episode_length as f64),
                    ("policy/entropy", entropy_sum / episode_length as f64),
                    (
                        "actor/grad/norm",
                        actor_gradient_norm_sum / episode_length as f64,
                    ),
                    (
                        "critic/grad/norm",
                        critic_gradient_norm_sum / episode_length as f64,
                    ),
                    ("optimizer/actor_lr", actor_learning_rate),
                    ("optimizer/critic_lr", critic_learning_rate),
                ],
            );
            let mut make_policy = || {
                Box::new(DeepQPolicy::new(pi.valid(), DeepQPolicyMode::SoftMax))
                    as Box<dyn crate::policy::DiscretePolicy<Observation = Array1<f64>>>
            };
            reporter.policy(Step::Episode(episode_index + 1), &mut make_policy);
            reporter.policy(Step::Iteration(episode_index + 1), &mut make_policy);
        }

        Ok(SolveResult {
            policy: Box::new(DeepQPolicy::new(pi.valid(), DeepQPolicyMode::SoftMax)),
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
    use crate::reporting::{Frequency, Reporter, ReporterConfig};
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn configuration_is_validated() {
        assert!(
            ActorCritic::new(ActorCriticConfig {
                episodes: 0,
                ..ActorCriticConfig::default()
            })
            .is_err()
        );
        assert!(
            ActorCritic::new(ActorCriticConfig {
                actor_learning_rate: Schedule::Constant { value: 0.0 },
                ..ActorCriticConfig::default()
            })
            .is_err()
        );
        assert!(ActorCritic::new(ActorCriticConfig::default()).is_ok());
        assert!(
            ActorCritic::new(ActorCriticConfig {
                gamma: 1.0,
                ..ActorCriticConfig::default()
            })
            .is_ok()
        );
    }

    #[test]
    fn trains_and_records_evaluations() {
        let model = grid_world(3).unwrap();
        let mut environment = FeatureEnvironment::new(&model).unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        let solver = ActorCritic::new(ActorCriticConfig {
            episodes: 2,
            episode_length: 2,
            ..ActorCriticConfig::default()
        })
        .unwrap();

        let mut evaluation_environment = FeatureEnvironment::new(&model).unwrap();
        let mut reporter = Reporter::new(
            ReporterConfig {
                evaluation_frequency: Some(Frequency::PerEpisode(1)),
                evaluation_episodes: 1,
                evaluation_max_steps: 2,
            },
            0,
            Some(&mut evaluation_environment),
        );
        let result = solver
            .solve_with_reporter(&mut environment, &mut rng, &mut reporter)
            .unwrap();
        let probabilities = result
            .policy
            .action_probabilities(&Array1::from_vec(vec![0.0, 0.0]));
        let history = reporter.into_history();

        assert_eq!(history.len(), 2);
        assert!(history.iter().all(|row| {
            row.metrics.contains_key("eval/return_mean")
                && row.metrics.contains_key("optimizer/actor_lr")
                && row.metrics.contains_key("policy/entropy")
        }));
        assert!((probabilities.sum() - 1.0).abs() < 1e-6);
    }
}
