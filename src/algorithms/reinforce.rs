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
#[cfg(test)]
use rand::{SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

type TrainingBackend = Autodiff<Flex>;

#[derive(Debug, Clone, PartialEq, Args, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReinforceConfig {
    #[arg(long, default_value_t = ReinforceConfig::default().gamma)]
    pub gamma: f64,
    #[arg(long, default_value = "linear:0.001:0.0001:999")]
    pub learning_rate: Schedule,
    #[arg(long, default_value_t = ReinforceConfig::default().hidden_dims)]
    pub hidden_dims: usize,
    #[arg(long, default_value_t = ReinforceConfig::default().episodes)]
    pub episodes: usize,
    #[arg(long, default_value_t = ReinforceConfig::default().episode_length)]
    pub episode_length: usize,
}

impl Default for ReinforceConfig {
    fn default() -> Self {
        Self {
            gamma: 0.9,
            learning_rate: Schedule::Linear {
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

pub struct Reinforce {
    config: ReinforceConfig,
}

impl Reinforce {
    pub fn new(config: ReinforceConfig) -> Result<Self, String> {
        if !config.gamma.is_finite() || !(0.0..=1.0).contains(&config.gamma) {
            return Err("gamma must be finite and in [0, 1]".into());
        }
        let valid_rate = |rate: f64| rate.is_finite() && rate > 0.0 && rate <= 1.0;
        let valid_schedule = match &config.learning_rate {
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
        if !valid_schedule {
            return Err(
                "learning rates must be finite and in (0, 1], with increasing steps".into(),
            );
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

impl ModelFreeSolver for Reinforce {
    type Observation = Array1<f64>;

    fn solve_with_reporter(
        &self,
        env: &mut dyn Environment<Observation = Array1<f64>, Action = usize>,
        rng: &mut dyn rand::Rng,
        reporter: &mut dyn AbstractReporter<Array1<f64>>,
    ) -> Result<SolveResult<Array1<f64>>, String> {
        let Space::Continuous { dim: n_s } = env.state_space() else {
            return Err("Reinforce requires a continuous state space".into());
        };
        let Space::Discrete { size: n_a } = env.action_space() else {
            return Err("Reinforce requires a discrete action space".into());
        };

        let device = FlexDevice;
        <TrainingBackend as Backend>::seed(&device, rng.random());

        let mut optimizer: OptimizerAdaptor<Adam, MlpModel<TrainingBackend>, TrainingBackend> =
            AdamConfig::new().init();

        let mut policy_network = MlpModel::new(n_s, self.config.hidden_dims, n_a, &device);
        for episode_index in 0..self.config.episodes {
            let policy = DeepQPolicy::borrowed(&policy_network, DeepQPolicyMode::SoftMax);
            let s0 = env.reset(rng);
            let episode = (0..self.config.episode_length)
                .scan(Some(s0), |state, _| {
                    let s = state.take()?;
                    let a = policy.sample_action(&s, rng);
                    let step = env.step(a, rng);
                    if !step.terminated && !step.truncated {
                        *state = Some(step.observation);
                    }
                    Some((s, a, step.reward))
                })
                .collect::<Vec<_>>();

            let episode_length = episode.len();
            let episode_return = episode.iter().map(|(_, _, reward)| reward).sum::<f64>();
            let mut states = Vec::with_capacity(episode_length * n_s);
            let mut actions = Vec::with_capacity(episode_length);

            for (state, action, _) in &episode {
                states.extend(state.iter().copied());
                actions.push(*action as i64);
            }

            let returns = discounted_policy_gradient_returns(
                episode.iter().map(|(_, _, reward)| *reward),
                self.config.gamma,
            );

            let st = Tensor::<TrainingBackend, 2>::from_data(
                TensorData::new(states, [episode_length, n_s]),
                &device,
            );
            let at = Tensor::<TrainingBackend, 1, Int>::from_data(
                TensorData::new(actions, [episode_length]),
                &device,
            )
            .reshape([episode_length, 1]);
            let gamma_t_qt_sa = Tensor::<TrainingBackend, 1>::from_data(
                TensorData::new(returns, [episode_length]),
                &device,
            );
            let log_probs = log_softmax(policy_network.forward(st), 1);
            let entropy = (log_probs.clone().exp() * log_probs.clone())
                .sum()
                .neg()
                .into_data()
                .iter::<f64>()
                .next()
                .expect("entropy tensor has one element")
                / episode_length as f64;
            let ln_pi_sa = log_probs.gather(1, at).squeeze_dim(1);
            let loss = (ln_pi_sa * gamma_t_qt_sa).sum().neg();
            let gradients = GradientsParams::from_grads(loss.backward(), &policy_network);
            let gradient_norm = policy_network.gradient_norm(&gradients);
            let learning_rate = self.config.learning_rate.value(episode_index);
            policy_network = optimizer.step(learning_rate, policy_network, gradients);
            let step = Step::Episode(episode_index + 1);
            reporter.record(
                step,
                &[
                    ("train/return", episode_return),
                    ("train/length", episode_length as f64),
                    ("policy/entropy", entropy),
                    ("grad/norm", gradient_norm),
                    ("optimizer/lr", learning_rate),
                ],
            );
            let mut make_policy = || {
                Box::new(DeepQPolicy::new(
                    policy_network.valid(),
                    DeepQPolicyMode::SoftMax,
                ))
                    as Box<dyn crate::policy::DiscretePolicy<Observation = Array1<f64>>>
            };
            reporter.policy(step, &mut make_policy);
            reporter.policy(Step::Iteration(episode_index + 1), &mut make_policy);
        }

        Ok(SolveResult {
            policy: Box::new(DeepQPolicy::new(
                policy_network.valid(),
                DeepQPolicyMode::SoftMax,
            )),
            status: SolveStatus::IterationLimit {
                iterations: self.config.episodes,
            },
        })
    }
}

fn discounted_policy_gradient_returns(
    rewards: impl DoubleEndedIterator<Item = f64> + ExactSizeIterator,
    gamma: f64,
) -> Vec<f64> {
    let mut returns = Vec::with_capacity(rewards.len());
    let mut return_value = 0.0;
    for reward in rewards.rev() {
        return_value = reward + gamma * return_value;
        returns.push(return_value);
    }
    returns.reverse();

    let mut gamma_t = 1.0;
    for return_value in &mut returns {
        *return_value *= gamma_t;
        gamma_t *= gamma;
    }
    returns
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::FeatureEnvironment;
    use crate::environments::grid_world;
    use crate::reporting::{Frequency, Reporter, ReporterConfig};

    #[test]
    fn policy_gradient_returns_include_absolute_discount() {
        let returns = discounted_policy_gradient_returns([1.0, 2.0, 3.0].into_iter(), 0.5);

        assert_eq!(returns, vec![2.75, 1.75, 0.75]);
    }

    #[test]
    fn configuration_is_validated() {
        assert!(
            Reinforce::new(ReinforceConfig {
                episodes: 0,
                ..ReinforceConfig::default()
            })
            .is_err()
        );
        assert!(
            Reinforce::new(ReinforceConfig {
                learning_rate: Schedule::Constant { value: 0.0 },
                ..ReinforceConfig::default()
            })
            .is_err()
        );
        assert!(Reinforce::new(ReinforceConfig::default()).is_ok());
        assert!(
            Reinforce::new(ReinforceConfig {
                gamma: 1.0,
                ..ReinforceConfig::default()
            })
            .is_ok()
        );
    }

    #[test]
    fn trains_and_records_evaluations() {
        let model = grid_world(3).unwrap();
        let mut environment = FeatureEnvironment::new(&model).unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        let solver = Reinforce::new(ReinforceConfig {
            episodes: 2,
            episode_length: 2,
            ..ReinforceConfig::default()
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
                && row.metrics.contains_key("optimizer/lr")
                && row.metrics.contains_key("policy/entropy")
        }));
        assert!((probabilities.sum() - 1.0).abs() < 1e-6);
    }
}
