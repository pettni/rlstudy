use crate::algorithms::{ModelFreeSolver, SolveResult, SolveStatus};
use crate::backend::{TrainingBackend, device};
use crate::environment::{Environment, Rollout, Space};
use crate::network::{
    actor::{Actor, CategoricalActor},
    gradient_norm,
    mlp::{Activation, MlpModel},
};
use crate::reporting::{Reporter, Step};
use crate::schedule::Schedule;
use crate::transition_buffer::TransitionBuffer;
use burn::grad_clipping::GradientClippingConfig;
use burn::tensor::ElementConversion;
use burn::{
    module::AutodiffModule,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::backend::Backend,
};
use clap::{Args, ValueEnum};
use ndarray::Array1;
use rand::RngExt;
use serde::{Deserialize, Serialize};

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
    type Action = usize;

    fn solve_with_reporter(
        &self,
        env: &mut dyn Environment<Observation = Array1<f64>, Action = usize>,
        rng: &mut dyn rand::Rng,
        reporter: &mut Reporter<'_, Array1<f64>>,
    ) -> Result<SolveResult<Array1<f64>>, String> {
        let Space::Continuous { dim: n_s } = env.state_space() else {
            return Err("Policy gradient requires a continuous state space".into());
        };
        let Space::Discrete { size: n_a } = env.action_space() else {
            return Err("Policy gradient requires a discrete action space".into());
        };

        <TrainingBackend as Backend>::seed(device(), rng.random());

        let gradient_clipping = self
            .config
            .gradient_clip_value
            .map(GradientClippingConfig::Norm);
        let mut pi_optimizer: OptimizerAdaptor<
            Adam,
            CategoricalActor<TrainingBackend>,
            TrainingBackend,
        > = AdamConfig::new()
            .with_grad_clipping(gradient_clipping.clone())
            .init();
        let mut v_optimizer: OptimizerAdaptor<Adam, MlpModel<TrainingBackend>, TrainingBackend> =
            AdamConfig::new()
                .with_grad_clipping(gradient_clipping)
                .init();

        let mut pi = CategoricalActor::new(
            n_s,
            &[self.config.hidden_dims],
            n_a,
            Activation::Relu,
            device(),
        );
        let mut v = MlpModel::new(n_s, self.config.hidden_dims, 1, device());
        let mut buffer = TransitionBuffer::<usize>::with_capacity(self.config.episode_length, None);

        for episode_index in 0..self.config.episodes {
            buffer.clear();
            let mut rollout = Rollout::new(env, rng);
            for _ in 0..self.config.episode_length {
                let transition = buffer.fill_one(&mut rollout, &pi, rng);
                if transition.terminated || transition.truncated {
                    break;
                }
            }
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
            let learning_rate = self.config.learning_rate.value(episode_index);

            pi = pi_optimizer.step(learning_rate, pi, grad_pi);

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

                v = v_optimizer.step(learning_rate, v, grad_v);
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
                    ("optimizer/lr", learning_rate),
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
        let observation = Array1::from_vec(vec![0.0, 0.0]);
        let probabilities = Array1::from_iter(
            (0..model.actions().len())
                .map(|action| result.policy.log_prob(&observation, &action).exp()),
        );
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
