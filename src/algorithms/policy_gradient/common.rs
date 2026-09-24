use super::{ppo::PpoConfig, vpg::VanillaPolicyGradientConfig};
use crate::algorithms::{ModelFreeSolver, SolveResult, SolveStatus};
use crate::backend::{TrainingBackend, device};
use crate::environment::{Environment, Rollout, Space};
use crate::network::action::ActionType;
use crate::network::actor::{Actor, CategoricalActor, GaussianActor};
use crate::network::gradient_norm;
use crate::network::mlp::{Activation, MlpModel};
use crate::reporting::{MetricAverages, Reporter, Step};
use crate::schedule::Schedule;
use crate::transition_buffer::TransitionBuffer;
use burn::{
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{ElementConversion, backend::Backend},
};
use ndarray::Array1;
use rand::RngExt;
use std::marker::PhantomData;

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

pub struct PolicyGradient<Action = usize> {
    config: PolicyGradientConfig,
    action: PhantomData<Action>,
}

impl PolicyGradient<usize> {
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
        Ok(Self {
            config,
            action: PhantomData,
        })
    }

    pub fn continuous(self) -> PolicyGradient<Array1<f64>> {
        PolicyGradient {
            config: self.config,
            action: PhantomData,
        }
    }
}

impl<Action> PolicyGradient<Action> {
    fn train<A: ActionType, P: Actor<A>>(
        &self,
        env: &mut dyn Environment<Observation = Array1<f64>, Action = A>,
        rng: &mut dyn rand::Rng,
        reporter: &mut Reporter<'_, Array1<f64>, A>,
        mut pi: P,
        n_s: usize,
    ) -> (P::InferencePolicy, SolveStatus) {
        let mut pi_optimizer: OptimizerAdaptor<Adam, P, TrainingBackend> = AdamConfig::new().init();
        let mut v_optimizer: OptimizerAdaptor<Adam, MlpModel<TrainingBackend>, TrainingBackend> =
            AdamConfig::new().init();
        let mut v = MlpModel::new_with_hidden_layers(
            n_s,
            &self.config.hidden_dims,
            1,
            self.config.activation,
            device(),
        );

        let mut buffer = TransitionBuffer::<A>::with_capacity(self.config.steps_per_epoch, None);
        let mut rollout = Rollout::new(env, rng);

        for epoch in 0..self.config.epochs {
            let step = Step::Epoch(epoch + 1);
            reporter.progress(step, self.config.epochs);
            let pi_learning_rate = self.config.pi_learning_rate.value(epoch);
            let v_learning_rate = self.config.v_learning_rate.value(epoch);

            buffer.clear();
            buffer.fill(&mut rollout, &pi, rng, self.config.steps_per_epoch);

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
                let log_probs = pi.log_probs(batch.s.clone(), &batch.a);

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
                update_metrics.record(
                    "pi/grad_norm",
                    gradient_norm::<TrainingBackend, _>(&pi, &pi_grad),
                );
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
                update_metrics.record(
                    "v/grad_norm",
                    gradient_norm::<TrainingBackend, _>(&v, &v_grad),
                );
                v = v_optimizer.step(v_learning_rate, v, v_grad);
            }

            let entropy = pi.entropy(batch.s.clone()).mean();
            let mut metrics = vec![
                ("pi/lr", pi_learning_rate),
                ("pi/entropy", entropy.mean().into_scalar().elem::<f64>()),
                ("v/lr", v_learning_rate),
            ];
            metrics.extend(update_metrics.finish());
            if let Some(stats) = buffer.episode_stats() {
                metrics.push(("train/episode_return", stats.return_mean));
                metrics.push(("train/episode_length", stats.length_mean));
            }

            reporter.record(step, &metrics);
            reporter.policy(step, || pi.valid());
        }

        (
            pi.valid(),
            SolveStatus::IterationLimit {
                iterations: self.config.epochs,
            },
        )
    }
}

impl ModelFreeSolver for PolicyGradient {
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
        let pi = CategoricalActor::new(
            n_s,
            &self.config.hidden_dims,
            n_a,
            self.config.activation,
            device(),
        );
        let (pi, status) = self.train(env, rng, reporter, pi, n_s);
        Ok(SolveResult {
            policy: Box::new(pi),
            status,
        })
    }
}

impl ModelFreeSolver for PolicyGradient<Array1<f64>> {
    type Observation = Array1<f64>;
    type Action = Array1<f64>;

    fn solve_with_reporter(
        &self,
        env: &mut dyn Environment<Observation = Array1<f64>, Action = Array1<f64>>,
        rng: &mut dyn rand::Rng,
        reporter: &mut Reporter<'_, Array1<f64>, Array1<f64>>,
    ) -> Result<SolveResult<Array1<f64>, Array1<f64>>, String> {
        let Space::Continuous { dim: n_s } = env.state_space() else {
            return Err("Policy gradient requires a continuous state space".into());
        };
        let Space::Continuous { dim: n_a } = env.action_space() else {
            return Err("Policy gradient requires a continuous action space".into());
        };

        <TrainingBackend as Backend>::seed(device(), rng.random());
        let pi = GaussianActor::new(
            n_s,
            &self.config.hidden_dims,
            n_a,
            self.config.activation,
            device(),
        );
        let (pi, status) = self.train(env, rng, reporter, pi, n_s);
        Ok(SolveResult {
            policy: Box::new(pi),
            status,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::Step as EnvironmentStep;
    use rand::{SeedableRng, rngs::StdRng};

    struct ContinuousBandit;

    impl Environment for ContinuousBandit {
        type Observation = Array1<f64>;
        type Action = Array1<f64>;

        fn state_space(&self) -> Space {
            Space::Continuous { dim: 1 }
        }

        fn action_space(&self) -> Space {
            Space::Continuous { dim: 1 }
        }

        fn reset(&mut self, _rng: &mut dyn rand::Rng) -> Self::Observation {
            Array1::zeros(1)
        }

        fn step(
            &mut self,
            action: Self::Action,
            _rng: &mut dyn rand::Rng,
        ) -> EnvironmentStep<Self::Observation> {
            EnvironmentStep {
                observation: Array1::zeros(1),
                reward: -action[0] * action[0],
                terminated: false,
                truncated: true,
            }
        }
    }

    #[test]
    fn trains_with_continuous_actions() {
        let solver = PolicyGradient::vpg(VanillaPolicyGradientConfig {
            hidden_dims: vec![4],
            epochs: 1,
            steps_per_epoch: 8,
            value_updates_per_epoch: 1,
            ..Default::default()
        })
        .unwrap()
        .continuous();
        let mut environment = ContinuousBandit;
        let mut rng = StdRng::seed_from_u64(0);

        let result = solver.solve(&mut environment, &mut rng).unwrap();
        let action = result.policy.sample_action(&Array1::zeros(1), &mut rng);

        assert_eq!(action.len(), 1);
        assert!(action[0].is_finite());
    }
}
