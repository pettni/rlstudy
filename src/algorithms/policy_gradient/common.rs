use super::{ppo::PpoConfig, vpg::VanillaPolicyGradientConfig};
use crate::algorithms::{ModelFreeSolver, SolveResult, SolveStatus};
use crate::environment::{Environment, Rollout, Space};
use crate::network::mlp::{Activation, MlpModel};
use crate::policy::dq::{DeepQPolicy, DeepQPolicyMode};
use crate::reporting::{AbstractReporter, MetricAverages, Step};
use crate::schedule::Schedule;
use crate::transition_buffer::TransitionBuffer;
use burn::{
    backend::{Autodiff, Flex, flex::FlexDevice},
    module::AutodiffModule,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{ElementConversion, activation::log_softmax, backend::Backend},
};
use ndarray::Array1;
use rand::RngExt;

type TrainingBackend = Autodiff<Flex>;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum PolicyObjective {
    /// Vanilla policy gradient objective
    ///
    ///  \log pi(a| s) * A(s, a)
    Vpg,
    /// PPO(epsilon) objective
    ///
    ///  min ( ( pi(a|s) / pi_k(a|s) ) * A, (1 + sign(A) eps) A )
    Ppo { epsilon: Schedule },
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct PolicyUpdateStrategy {
    /// Max update steps per loop
    pub(super) max_updates_per_epoch: usize,
    /// KL divergence threshold for early stopping
    pub(super) kl_limit: Option<f64>,
}

impl Default for PolicyUpdateStrategy {
    fn default() -> Self {
        Self {
            max_updates_per_epoch: 1,
            kl_limit: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct PolicyGradientConfig {
    pub(super) objective: PolicyObjective,
    pub(super) gamma: f64,
    pub(super) lambda: f64,
    pub(super) pi_learning_rate: Schedule,
    pub(super) v_learning_rate: Schedule,
    pub(super) hidden_dims: Vec<usize>,
    pub(super) activation: Activation,
    pub(super) epochs: usize,
    pub(super) steps_per_epoch: usize,
    pub(super) policy_update_strategy: PolicyUpdateStrategy,
    pub(super) value_updates_per_epoch: usize,
}

pub struct PolicyGradient {
    config: PolicyGradientConfig,
}

impl PolicyGradient {
    pub fn vpg(config: VanillaPolicyGradientConfig) -> Result<Self, String> {
        Self::new(config.into())
    }

    pub fn ppo(config: PpoConfig) -> Result<Self, String> {
        Self::new(config.into())
    }

    fn new(config: PolicyGradientConfig) -> Result<Self, String> {
        if !config.gamma.is_finite() || !(0.0..=1.0).contains(&config.gamma) {
            return Err("gamma must be finite and in [0, 1]".into());
        }
        if !config.lambda.is_finite() || !(0.0..=1.0).contains(&config.lambda) {
            return Err("lambda must be finite and in [0, 1]".into());
        }
        if config.hidden_dims.is_empty() || config.hidden_dims.contains(&0) {
            return Err("hidden dims must be positive".into());
        }
        if config.epochs == 0 || config.steps_per_epoch == 0 || config.value_updates_per_epoch == 0
        {
            return Err(
                "epochs, steps_per_epoch, and value_updates_per_epoch must be positive".into(),
            );
        }
        if config.policy_update_strategy.max_updates_per_epoch == 0 {
            return Err("policy updates per epoch must be positive".into());
        }
        Ok(Self { config })
    }
}

impl ModelFreeSolver for PolicyGradient {
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
            AdamConfig::new().init();
        let mut v_optimizer = pi_optimizer.clone();

        let mut pi = MlpModel::new_with_hidden_layers(
            n_s,
            &self.config.hidden_dims,
            n_a,
            self.config.activation,
            &device,
        );
        let mut v = MlpModel::new_with_hidden_layers(
            n_s,
            &self.config.hidden_dims,
            1,
            self.config.activation,
            &device,
        );

        let mut buffer =
            TransitionBuffer::<usize>::with_capacity(self.config.steps_per_epoch, None);
        let mut rollout = Rollout::new(env, rng);

        for epoch in 0..self.config.epochs {
            let step = Step::Epoch(epoch + 1);
            reporter.progress(step, self.config.epochs);
            let pi_learning_rate = self.config.pi_learning_rate.value(epoch);
            let v_learning_rate = self.config.v_learning_rate.value(epoch);

            buffer.clear();
            buffer.fill(
                &mut rollout,
                &DeepQPolicy::borrowed(&pi, DeepQPolicyMode::SoftMax),
                rng,
                self.config.steps_per_epoch,
            );

            let batch = buffer.tensor_data();
            let returns = batch
                .discounted_returns(|states| v.forward(states).squeeze_dim(1), self.config.gamma);

            let advantage = batch.gae(
                |states| v.forward(states).squeeze_dim(1),
                self.config.gamma,
                self.config.lambda,
            );

            // advantage normalization
            let advantage_mean = advantage.clone().mean();
            let advantage_std = (advantage.clone().sub(advantage_mean.clone()))
                .square()
                .mean()
                .sqrt();
            let advantage = (advantage.sub(advantage_mean)) / (advantage_std + 1e-7);
            let advantage = advantage.detach(); // stop-grad

            // policy update

            let mut update_metrics = MetricAverages::new();
            let mut policy_updates = 0;
            for update in 0..self.config.policy_update_strategy.max_updates_per_epoch {
                // current policy log-probs
                let log_probs = log_softmax(pi.forward(batch.s.clone()), 1)
                    .gather(1, batch.a.clone().reshape([-1, 1]))
                    .squeeze_dim(1);

                let kl_estimate = (batch.log_probs.clone() - log_probs.clone())
                    .mean()
                    .into_scalar()
                    .elem::<f64>();

                // check early stopping
                if update > 0 {
                    if let Some(limit) = self.config.policy_update_strategy.kl_limit {
                        // monte carlo estimate of KL(pi, pi_batch)
                        if kl_estimate > limit {
                            break;
                        }
                    }
                }

                let pi_loss = match &self.config.objective {
                    PolicyObjective::Vpg => (log_probs * advantage.clone()).mean().neg(),
                    PolicyObjective::Ppo { epsilon } => {
                        let eps = epsilon.value(epoch);
                        let ratio = (log_probs.clone() - batch.log_probs.clone()).exp();

                        let objective = ratio * advantage.clone();
                        let clipped_objective = advantage.clone() + eps * advantage.clone().abs();
                        let clip_mask = clipped_objective.clone().lower(objective.clone());

                        update_metrics.record("pi/kl", kl_estimate);
                        update_metrics.record(
                            "pi/clip_fraction",
                            clip_mask.clone().float().mean().into_scalar().elem::<f64>(),
                        );

                        objective
                            .mask_where(clip_mask, clipped_objective)
                            .mean()
                            .neg()
                    }
                };

                let pi_grad = GradientsParams::from_grads(pi_loss.backward(), &pi);
                update_metrics.record("pi/loss", pi_loss.into_scalar().elem::<f64>());
                update_metrics.record("pi/grad_norm", pi.gradient_norm(&pi_grad));
                pi = pi_optimizer.step(pi_learning_rate, pi, pi_grad);
                policy_updates += 1;
            }
            update_metrics.record("pi/updates", policy_updates as f64);

            // critic update
            for _ in 0..self.config.value_updates_per_epoch {
                let v_loss = (returns.clone() - v.forward(batch.s.clone()).squeeze_dim(1))
                    .square()
                    .mean();

                let v_grad = GradientsParams::from_grads(v_loss.backward(), &v);
                update_metrics.record("v/loss", v_loss.into_scalar().elem::<f64>());
                update_metrics.record("v/grad_norm", v.gradient_norm(&v_grad));
                v = v_optimizer.step(v_learning_rate, v, v_grad);
            }

            let log_probs_a = log_softmax(pi.forward(batch.s), 1);
            let mut metrics = vec![
                ("pi/lr", pi_learning_rate),
                (
                    "pi/entropy",
                    (log_probs_a.clone().exp() * log_probs_a.clone())
                        .neg()
                        .sum_dim(1)
                        .mean()
                        .into_scalar()
                        .elem::<f64>(),
                ),
                ("v/lr", v_learning_rate),
            ];
            metrics.extend(update_metrics.finish());
            if let Some(stats) = buffer.episode_stats() {
                metrics.push(("train/episode_return", stats.return_mean));
                metrics.push(("train/episode_length", stats.length_mean));
            }

            reporter.record(step, &metrics);
            let mut make_policy = || {
                Box::new(DeepQPolicy::new(pi.valid(), DeepQPolicyMode::SoftMax))
                    as Box<dyn crate::policy::DiscretePolicy<Observation = Array1<f64>>>
            };
            reporter.policy(step, &mut make_policy);
        }

        Ok(SolveResult {
            policy: Box::new(DeepQPolicy::new(pi.valid(), DeepQPolicyMode::SoftMax)),
            status: SolveStatus::IterationLimit {
                iterations: self.config.epochs,
            },
        })
    }
}
