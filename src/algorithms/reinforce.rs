use crate::algorithms::{
    ModelFreeSolver, SolveResult, SolveStatus, SolverSnapshot, validate_gamma,
};
use crate::environment::{Environment, Space};
use crate::evaluation::{EvaluationConfig, EvaluationMode};
use crate::network::mlp::MlpModel;
use crate::policy::{
    Policy,
    dq::{DeepQPolicy, DeepQPolicyMode},
};
use crate::schedule::Schedule;
use burn::{
    Tensor,
    backend::{Autodiff, Flex, flex::FlexDevice},
    module::AutodiffModule,
    optim::{GradientsParams, Optimizer, Sgd, SgdConfig, adaptor::OptimizerAdaptor},
    tensor::{Int, TensorData, activation::log_softmax, backend::Backend},
};
use clap::Args;
use ndarray::Array1;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

type TrainingBackend = Autodiff<Flex>;

#[derive(Debug, Clone, PartialEq, Args, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReinforceConfig {
    #[arg(long, default_value_t = ReinforceConfig::default().gamma)]
    pub gamma: f64,
    #[arg(long, default_value_t = ReinforceConfig::default().learning_rate)]
    pub learning_rate: f64,
    #[arg(long, default_value_t = ReinforceConfig::default().final_learning_rate)]
    pub final_learning_rate: f64,
    #[arg(long, default_value_t = ReinforceConfig::default().episodes)]
    pub episodes: usize,
    #[arg(long, default_value_t = ReinforceConfig::default().episode_length)]
    pub episode_length: usize,
    #[arg(long, default_value_t = ReinforceConfig::default().evaluation_interval)]
    pub evaluation_interval: usize,
    #[arg(long, default_value_t = ReinforceConfig::default().evaluation_episodes)]
    pub evaluation_episodes: usize,
    #[arg(long, default_value_t = ReinforceConfig::default().evaluation_max_steps)]
    pub evaluation_max_steps: usize,
}

impl Default for ReinforceConfig {
    fn default() -> Self {
        Self {
            gamma: 0.9,
            learning_rate: 0.001,
            final_learning_rate: 0.0001,
            episodes: 1_000,
            episode_length: 100,
            evaluation_interval: 100,
            evaluation_episodes: 10,
            evaluation_max_steps: 1_000,
        }
    }
}

pub struct Reinforce {
    config: ReinforceConfig,
}

impl Reinforce {
    pub fn new(config: ReinforceConfig) -> Result<Self, String> {
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
        if config.episodes == 0 || config.episode_length == 0 || config.evaluation_interval == 0 {
            return Err(
                "episodes, episode length, and evaluation interval must be positive".into(),
            );
        }
        if config.evaluation_episodes == 0 || config.evaluation_max_steps == 0 {
            return Err("evaluation episodes and max steps must be positive".into());
        }
        Ok(Self { config })
    }
}

impl ModelFreeSolver for Reinforce {
    type Observation = Array1<f64>;

    fn solve(
        &self,
        env: &mut dyn Environment<Observation = Array1<f64>, Action = usize>,
        rng: &mut dyn rand::Rng,
    ) -> Result<SolveResult<Array1<f64>>, String> {
        let Space::Continuous { dim: n_s } = env.state_space() else {
            return Err("Reinforce requires a continuous state space".into());
        };
        let Space::Discrete { size: n_a } = env.action_space() else {
            return Err("Reinforce requires a discrete action space".into());
        };

        let device = FlexDevice;
        <TrainingBackend as Backend>::seed(&device, rng.random());

        let mut optimizer: OptimizerAdaptor<Sgd<Flex>, MlpModel<TrainingBackend>, TrainingBackend> =
            SgdConfig::new().init();

        let learning_rate_schedule = if self.config.learning_rate == self.config.final_learning_rate
        {
            Schedule::Constant {
                value: self.config.learning_rate,
            }
        } else {
            Schedule::Linear {
                initial: self.config.learning_rate,
                final_value: self.config.final_learning_rate,
                steps: self.config.episodes.saturating_sub(1).max(1),
            }
        };

        let mut policy_network = MlpModel::new(n_s, 64, n_a, &device);
        let mut snapshots = Vec::new();
        let mut evaluation_rng = StdRng::seed_from_u64(rng.random());
        let evaluation = EvaluationConfig::new(
            self.config.evaluation_episodes,
            self.config.evaluation_max_steps,
            EvaluationMode::DiscountedReward {
                gamma: self.config.gamma,
            },
        )?;

        for episode_index in 0..self.config.episodes {
            // TODO: avoid copy?
            let policy = DeepQPolicy::new(policy_network.clone().valid(), DeepQPolicyMode::SoftMax);
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
            let ln_pi_sa = log_softmax(policy_network.forward(st), 1)
                .gather(1, at)
                .squeeze_dim(1);
            let loss = (ln_pi_sa * gamma_t_qt_sa).mean().neg();
            let gradients = GradientsParams::from_grads(loss.backward(), &policy_network);
            let learning_rate = learning_rate_schedule.value(episode_index);
            policy_network = optimizer.step(learning_rate, policy_network, gradients);

            if (episode_index + 1) % self.config.evaluation_interval == 0
                || episode_index + 1 == self.config.episodes
            {
                let policy =
                    DeepQPolicy::new(policy_network.clone().valid(), DeepQPolicyMode::SoftMax);
                snapshots.push(
                    SolverSnapshot::new(episode_index + 1)
                        .learning_rate(learning_rate)
                        .evaluate_policy(env, &policy, evaluation, &mut evaluation_rng)?,
                );
            }
        }

        Ok(SolveResult {
            policy: Box::new(DeepQPolicy::new(
                policy_network.valid(),
                DeepQPolicyMode::SoftMax,
            )),
            snapshots,
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
                learning_rate: 0.0,
                ..ReinforceConfig::default()
            })
            .is_err()
        );
        assert!(Reinforce::new(ReinforceConfig::default()).is_ok());
    }

    #[test]
    fn trains_and_records_evaluations() {
        let model = grid_world(3).unwrap();
        let mut environment = FeatureEnvironment::new(&model).unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        let solver = Reinforce::new(ReinforceConfig {
            episodes: 2,
            episode_length: 2,
            evaluation_interval: 1,
            evaluation_episodes: 1,
            evaluation_max_steps: 2,
            ..ReinforceConfig::default()
        })
        .unwrap();

        let result = solver.solve(&mut environment, &mut rng).unwrap();
        let probabilities = result
            .policy
            .action_probabilities(&Array1::from_vec(vec![0.0, 0.0]));

        assert_eq!(result.snapshots.len(), 2);
        assert!(result.snapshots.iter().all(|snapshot| {
            snapshot.evaluation_return.is_some() && snapshot.learning_rate.is_some()
        }));
        assert!((probabilities.sum() - 1.0).abs() < 1e-6);
    }
}
