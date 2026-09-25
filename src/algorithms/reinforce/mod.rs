use crate::algorithms::{ModelFreeSolver, SolveResult, SolveStatus};
use crate::backend::{TrainingBackend, device};
use crate::environment::{Environment, Rollout, Space};
use crate::network::{
    AdamOptimizer, MlpModel, MlpSettings,
    actor::{Actor, CategoricalActor},
    gradient_norm,
};
use crate::reporting::{Reporter, Step};
use crate::schedule::Schedule;
use crate::transition_buffer::TransitionBuffer;
use burn::grad_clipping::GradientClippingConfig;
use burn::tensor::ElementConversion;
use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::backend::Backend,
};
use clap::ValueEnum;
use ndarray::Array1;
use rand::RngExt;
use serde::{Deserialize, Serialize};

mod cli;

pub use cli::ReinforceArgs;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum UpdateVariant {
    // Monte-carlo update g_t = r_{t+1} + gamma * r_{t+2} + ...
    MC,
    // Temporal difference update g_t = r_{t+1} + gamma * v_{t+1}
    TD,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReinforceConfig {
    pub gamma: f64,
    pub pi_network: MlpSettings,
    pub v_network: MlpSettings,
    pub episodes: usize,
    pub episode_length: usize,
    gradient_clip_value: Option<f32>,
    pub critic: Option<UpdateVariant>,
}

impl Default for ReinforceConfig {
    fn default() -> Self {
        let learning_rate = Schedule::Linear {
            initial: 0.001,
            final_value: 0.0001,
            steps: 999,
        };
        Self {
            gamma: 0.9,
            pi_network: MlpSettings {
                learning_rate: learning_rate.clone(),
                ..MlpSettings::default()
            },
            v_network: MlpSettings {
                learning_rate,
                ..MlpSettings::default()
            },
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
        if [&config.pi_network, &config.v_network]
            .into_iter()
            .any(|network| network.hidden_dims.is_empty() || network.hidden_dims.contains(&0))
        {
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
            .ok_or("Policy gradient requires a continuous state space")?;
        let Space::Discrete { size: n_a } = env.action_space() else {
            return Err("Policy gradient requires a discrete action space".into());
        };

        <TrainingBackend as Backend>::seed(device(), rng.random());

        let gradient_clipping = self
            .config
            .gradient_clip_value
            .map(GradientClippingConfig::Norm);
        let mut pi = CategoricalActor::new(
            n_s,
            &self.config.pi_network.hidden_dims,
            n_a,
            self.config.pi_network.activation,
            device(),
        );
        let mut pi_optimizer: AdamOptimizer<CategoricalActor<TrainingBackend>> = AdamConfig::new()
            .with_grad_clipping(gradient_clipping.clone())
            .init();
        let mut v = self.config.v_network.init(n_s, 1);
        let mut v_optimizer: AdamOptimizer<MlpModel<TrainingBackend>> = AdamConfig::new()
            .with_grad_clipping(gradient_clipping)
            .init();
        let mut buffer = TransitionBuffer::<usize>::with_capacity(self.config.episode_length, None);
        let mut rollout = Rollout::new(env, rng);

        for episode_index in 0..self.config.episodes {
            buffer.clear();
            buffer.fill_episode_actor(&mut rollout, &pi, rng, Some(self.config.episode_length));
            let episode_length = buffer.len();
            let batch = buffer.tensor_data();
            let returns = batch.discounted_returns_no_bootstrapping(self.config.gamma);

            // ACTOR UPDATE

            // w_{t+1} += alpha * gamma^t * delta_t * grad_w ln pi(a_t|s_t, w_t)

            // advantage
            let delta_pi = if let Some(_) = self.config.critic {
                returns.clone() - v.forward(batch.s.clone()).squeeze_dim(1)
            } else {
                returns.clone()
            };
            let delta_pi = batch.discount_from_episode_start(delta_pi, self.config.gamma);

            let ln_pi = pi.log_probs(batch.s.clone(), &batch.a);
            let entropy = pi
                .entropy(batch.s.clone())
                .mean()
                .into_scalar()
                .elem::<f64>();
            let pi_loss = (delta_pi.detach() * ln_pi).sum().neg();
            let grad_pi = GradientsParams::from_grads(pi_loss.backward(), &pi);

            // observability
            let gradient_norm = gradient_norm(&pi, &grad_pi);
            let pi_learning_rate = self.config.pi_network.learning_rate.value(episode_index);
            let v_learning_rate = self.config.v_network.learning_rate.value(episode_index);

            pi = pi_optimizer.step(pi_learning_rate, pi, grad_pi);

            // CRITIC UPDATE

            // w_{t+1} += alpha * delta_t * grad_w v(s_t|w_t)

            if let Some(method) = self.config.critic {
                let return_estimate_v = match method {
                    UpdateVariant::MC => returns,
                    UpdateVariant::TD => {
                        let vtp = v.forward(batch.sp).squeeze_dim(1);
                        let live = batch.terminated.bool_not().float();
                        batch.r.clone() + self.config.gamma * vtp * live
                    }
                };

                let vt = v.forward(batch.s).squeeze_dim(1);
                let delta_v = return_estimate_v - vt.clone();
                let loss = (delta_v.detach() * vt).sum().neg();
                let grad_v = GradientsParams::from_grads(loss.backward(), &v);

                v = v_optimizer.step(v_learning_rate, v, grad_v);
            }

            reporter.record(
                Step::Episode(episode_index + 1),
                &[
                    (
                        "train/return",
                        batch
                            .r
                            .sum()
                            .into_data()
                            .iter::<f64>()
                            .next()
                            .expect("return tensor has one element"),
                    ),
                    ("train/length", episode_length as f64),
                    ("policy/entropy", entropy),
                    ("grad/norm", gradient_norm),
                    ("optimizer/pi_lr", pi_learning_rate),
                    ("optimizer/v_lr", v_learning_rate),
                ],
            );
            reporter.policy(Step::Episode(episode_index + 1), || pi.valid());
        }

        Ok(SolveResult {
            policy: Box::new(pi.valid()),
            status: SolveStatus::IterationLimit {
                iterations: self.config.episodes,
            },
        })
    }
}
