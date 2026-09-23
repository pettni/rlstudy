use crate::algorithms::{ModelFreeSolver, SolveResult, SolveStatus};
use crate::environment::{Environment, Space};
use crate::network::mlp::MlpModel;
use crate::policy::{
    Policy,
    dq::{DeepQPolicy, DeepQPolicyMode},
};
use crate::reporting::{AbstractReporter, Step};
use crate::schedule::Schedule;
use burn::grad_clipping::GradientClippingConfig;
use burn::tensor::Bool;
use burn::{
    Tensor,
    backend::{Autodiff, Flex, flex::FlexDevice},
    module::AutodiffModule,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Int, TensorData, activation::log_softmax, backend::Backend},
};
use clap::{Args, ValueEnum};
use ndarray::Array1;
use rand::RngExt;
use serde::{Deserialize, Serialize};

type TrainingBackend = Autodiff<Flex>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum UpdateVariant {
    // Monte-carlo update g_t = r_{t+1} + gamma * r_{t+2} + ...
    MC,
    // Temporal difference update g_t = r_{t+1} + gamma * v_{t+1}
    TD,
}

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
    #[arg(long)]
    gradient_clip_value: Option<f32>,
    #[arg(long)]
    pub critic: Option<UpdateVariant>,
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
            gradient_clip_value: None,
            critic: None,
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
            return Err("Policy gradient requires a continuous state space".into());
        };
        let Space::Discrete { size: n_a } = env.action_space() else {
            return Err("Policy gradient requires a discrete action space".into());
        };

        let device = FlexDevice;
        <TrainingBackend as Backend>::seed(&device, rng.random());

        let mut pi_optimizer: OptimizerAdaptor<Adam, MlpModel<TrainingBackend>, TrainingBackend> =
            AdamConfig::new()
                .with_grad_clipping(
                    self.config
                        .gradient_clip_value
                        .map(GradientClippingConfig::Norm),
                )
                .init();

        let mut v_optimizer = pi_optimizer.clone();

        let mut pi = MlpModel::new(n_s, self.config.hidden_dims, n_a, &device);
        let mut v = MlpModel::new(n_s, self.config.hidden_dims, 1, &device);

        for episode_index in 0..self.config.episodes {
            let policy = DeepQPolicy::borrowed(&pi, DeepQPolicyMode::SoftMax);
            let s0 = env.reset(rng);

            let episode_data = sample_episode(
                s0,
                self.config.episode_length,
                &policy,
                self.config.gamma,
                env,
                rng,
            );
            let episode_length = episode_data.at.shape()[0];

            // ACTOR UPDATE

            // w_{t+1} += alpha * gamma^t * delta_t * grad_w ln pi(a_t|s_t, w_t)

            // estimate of q(s_t, a_t)
            let return_estimate_pi = episode_data.gt.clone();

            // advantage
            let delta_pi = if let Some(_) = self.config.critic {
                return_estimate_pi - v.forward(episode_data.st.clone()).squeeze_dim(1)
            } else {
                return_estimate_pi
            };

            let log_probs = log_softmax(pi.forward(episode_data.st.clone()), 1); // [T, A]
            let ln_pi = log_probs
                .clone()
                .gather(1, episode_data.at.reshape([-1, 1]))
                .squeeze_dim(1);
            let pi_loss = (episode_data.gammat * delta_pi.detach() * ln_pi)
                .sum()
                .neg();
            let grad_pi = GradientsParams::from_grads(pi_loss.backward(), &pi);

            // observability
            let gradient_norm = pi.gradient_norm(&grad_pi);
            let learning_rate = self.config.learning_rate.value(episode_index);

            pi = pi_optimizer.step(learning_rate, pi, grad_pi);

            // CRITIC UPDATE

            // w_{t+1} += alpha * delta_t * grad_w v(s_t|w_t)

            if let Some(method) = self.config.critic {
                let return_estimate_v = match method {
                    UpdateVariant::MC => episode_data.gt,
                    UpdateVariant::TD => {
                        let vtp = v.forward(episode_data.stp).squeeze_dim(1);
                        episode_data.rtp + self.config.gamma * vtp * episode_data.live.float()
                    }
                };

                let vt = v.forward(episode_data.st).squeeze_dim(1);
                let delta_v = return_estimate_v - vt.clone();
                let loss = (delta_v.detach() * vt).sum().neg();
                let grad_v = GradientsParams::from_grads(loss.backward(), &v);

                v = v_optimizer.step(learning_rate, v, grad_v);
            }

            reporter.record(
                Step::Episode(episode_index + 1),
                &[
                    ("train/return", episode_data.episode_return),
                    ("train/length", episode_length as f64),
                    (
                        "policy/entropy",
                        (log_probs.clone().exp() * log_probs.clone())
                            .sum()
                            .neg()
                            .into_data()
                            .iter::<f64>()
                            .next()
                            .expect("entropy tensor has one element")
                            / episode_length as f64,
                    ),
                    ("grad/norm", gradient_norm),
                    ("optimizer/lr", learning_rate),
                ],
            );
            let mut make_policy = || {
                Box::new(DeepQPolicy::new(pi.valid(), DeepQPolicyMode::SoftMax))
                    as Box<dyn crate::policy::DiscretePolicy<Observation = Array1<f64>>>
            };
            reporter.policy(Step::Episode(episode_index + 1), &mut make_policy);
        }

        Ok(SolveResult {
            policy: Box::new(DeepQPolicy::new(pi.valid(), DeepQPolicyMode::SoftMax)),
            status: SolveStatus::IterationLimit {
                iterations: self.config.episodes,
            },
        })
    }
}

struct EpisodeData {
    // [s0, ..., s_{T-1}]
    st: Tensor<TrainingBackend, 2>,
    // [a0, ..., a_{T-1}]
    at: Tensor<TrainingBackend, 1, Int>,
    // [r1, ..., r_T]
    rtp: Tensor<TrainingBackend, 1>,
    // [s1, ..., s_T]
    stp: Tensor<TrainingBackend, 2>,
    // [g0, ..., g_{T-1}] s.t. g_t = r_{t+1} + gamma r_{t+2} + ...
    gt: Tensor<TrainingBackend, 1>,
    // [1, gamma, gamma^2, ...]
    gammat: Tensor<TrainingBackend, 1>,
    // [l1, ..., l_{t}]  s.t. lt is 1. if episode is not terminated
    live: Tensor<TrainingBackend, 1, Bool>,
    // \sum_t r_t
    episode_return: f64,
}

fn sample_episode(
    s0: Array1<f64>,
    max_episode_length: usize,
    policy: &DeepQPolicy<Autodiff<Flex>, &MlpModel<Autodiff<Flex>>>,
    gamma: f64,
    env: &mut dyn Environment<Observation = Array1<f64>, Action = usize>,
    rng: &mut dyn rand::Rng,
) -> EpisodeData {
    let Space::Continuous { dim: n_s } = env.state_space() else {
        panic!("Invalid");
    };

    let mut states = Vec::with_capacity(max_episode_length * n_s);
    let mut actions = Vec::with_capacity(max_episode_length);
    let mut rewards = Vec::with_capacity(max_episode_length);
    let mut next_states = Vec::with_capacity(max_episode_length * n_s);
    let mut gammas = Vec::with_capacity(max_episode_length);
    let mut live = Vec::with_capacity(max_episode_length);

    let mut s = s0;
    for _ in 0..max_episode_length {
        let a = policy.sample_action(&s, rng);
        let step = env.step(a, rng);

        states.extend(s.iter().copied());
        actions.push(a as i64);
        rewards.push(step.reward);
        next_states.extend(step.observation.iter().copied());

        if gammas.is_empty() {
            gammas.push(1.);
        } else {
            gammas.push(gammas.last().unwrap() * gamma);
        }

        live.push(!step.terminated);

        if step.terminated || step.truncated {
            break;
        }

        s = step.observation;
    }

    let mut returns = Vec::with_capacity(max_episode_length);
    let mut gt: f64 = 0.0;
    let episode_return = rewards.iter().sum();
    for rt in rewards.iter().rev() {
        gt = rt + gamma * gt;
        returns.push(gt);
    }
    returns.reverse();

    let episode_length = actions.len();

    EpisodeData {
        st: Tensor::<TrainingBackend, 2>::from_data(
            TensorData::new(states, [episode_length, n_s]),
            &FlexDevice,
        ),
        at: Tensor::<TrainingBackend, 1, Int>::from_data(
            TensorData::new(actions, [episode_length]),
            &FlexDevice,
        ),
        rtp: Tensor::<TrainingBackend, 1>::from_data(
            TensorData::new(rewards, [episode_length]),
            &FlexDevice,
        ),
        stp: Tensor::<TrainingBackend, 2>::from_data(
            TensorData::new(next_states, [episode_length, n_s]),
            &FlexDevice,
        ),
        gt: Tensor::<TrainingBackend, 1>::from_data(
            TensorData::new(returns, [episode_length]),
            &FlexDevice,
        ),
        gammat: Tensor::<TrainingBackend, 1>::from_data(
            TensorData::new(gammas, [episode_length]),
            &FlexDevice,
        ),
        live: Tensor::<TrainingBackend, 1, Bool>::from_data(
            TensorData::new(live, [episode_length]),
            &FlexDevice,
        ),
        episode_return,
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
                evaluation_frequency: Some(1),
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
