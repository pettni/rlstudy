use crate::algorithms::{ModelFreeSolver, SolveResult, SolveStatus};
use crate::environment::{Environment, Rollout, Space};
use crate::network::mlp::MlpModel;
use crate::policy::dq::{DeepQPolicy, DeepQPolicyMode};
use crate::reporting::{AbstractReporter, Step};
use crate::schedule::Schedule;
use crate::transition_buffer::TransitionBuffer;
use burn::{
    backend::{Autodiff, Flex, flex::FlexDevice},
    module::AutodiffModule,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{ElementConversion, activation::log_softmax, backend::Backend},
};
use clap::Args;
use ndarray::Array1;
use rand::RngExt;
use serde::{Deserialize, Serialize};

type TrainingBackend = Autodiff<Flex>;

#[derive(Debug, Clone, PartialEq, Args, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct VanillaPolicyGradientConfig {
    #[arg(long, default_value_t = VanillaPolicyGradientConfig::default().gamma)]
    pub gamma: f64,
    #[arg(long, default_value_t = VanillaPolicyGradientConfig::default().lambda)]
    pub lambda: f64,
    #[arg(long, default_value = "constant:0.0003")]
    pub pi_learning_rate: Schedule,
    #[arg(long, default_value = "constant:0.001")]
    pub v_learning_rate: Schedule,
    #[arg(long, default_value_t = VanillaPolicyGradientConfig::default().hidden_dims)]
    pub hidden_dims: usize,
    #[arg(long, default_value_t = VanillaPolicyGradientConfig::default().epochs)]
    pub epochs: usize,
    #[arg(long, default_value_t = VanillaPolicyGradientConfig::default().steps_per_epoch)]
    pub steps_per_epoch: usize,
    #[arg(long, default_value_t = VanillaPolicyGradientConfig::default().value_updates_per_epoch)]
    pub value_updates_per_epoch: usize,
}

impl Default for VanillaPolicyGradientConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.97,
            pi_learning_rate: Schedule::Constant { value: 0.0003 },
            v_learning_rate: Schedule::Constant { value: 0.001 },
            hidden_dims: 64,
            epochs: 100,
            steps_per_epoch: 4000,
            value_updates_per_epoch: 80,
        }
    }
}

pub struct VanillaPolicyGradient {
    config: VanillaPolicyGradientConfig,
}

impl VanillaPolicyGradient {
    pub fn new(config: VanillaPolicyGradientConfig) -> Result<Self, String> {
        if !config.gamma.is_finite() || !(0.0..=1.0).contains(&config.gamma) {
            return Err("gamma must be finite and in [0, 1]".into());
        }
        if !config.lambda.is_finite() || !(0.0..=1.0).contains(&config.lambda) {
            return Err("lambda must be finite and in [0, 1]".into());
        }
        if config.hidden_dims == 0 {
            return Err("hidden dims must be positive".into());
        }
        if config.epochs == 0 || config.steps_per_epoch == 0 || config.value_updates_per_epoch == 0
        {
            return Err(
                "epochs, steps_per_epoch, and value_updates_per_epoch must be positive".into(),
            );
        }
        Ok(Self { config })
    }
}

impl ModelFreeSolver for VanillaPolicyGradient {
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

        let mut pi = MlpModel::new(n_s, self.config.hidden_dims, n_a, &device);
        let mut v = MlpModel::new(n_s, self.config.hidden_dims, 1, &device);

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

            // log probabilities
            let log_probs_a = log_softmax(pi.forward(batch.s.clone()), 1); // (B, A)
            let log_probs = log_probs_a
                .clone()
                .gather(1, batch.a.reshape([-1, 1]))
                .squeeze_dim(1); // (B, )

            // policy update

            let pi_loss = (log_probs.clone() * advantage).mean().neg();

            let pi_grad = GradientsParams::from_grads(pi_loss.backward(), &pi);
            let pi_grad_norm = pi.gradient_norm(&pi_grad);
            pi = pi_optimizer.step(pi_learning_rate, pi, pi_grad);

            // critic update
            let mut v_grad_norm = 0.0;
            let mut v_loss_mean = 0.0;
            for _ in 0..self.config.value_updates_per_epoch {
                let v_loss = (returns.clone() - v.forward(batch.s.clone()).squeeze_dim(1))
                    .square()
                    .mean();

                let v_grad = GradientsParams::from_grads(v_loss.backward(), &v);
                v_grad_norm += v.gradient_norm(&v_grad);
                v_loss_mean += v_loss.into_scalar().elem::<f64>();
                v = v_optimizer.step(v_learning_rate, v, v_grad);
            }
            v_grad_norm /= self.config.value_updates_per_epoch as f64;
            v_loss_mean /= self.config.value_updates_per_epoch as f64;

            let mut metrics = vec![
                ("pi/lr", pi_learning_rate),
                ("pi/grad_norm", pi_grad_norm),
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
                ("v/loss", v_loss_mean),
                ("v/grad_norm", v_grad_norm),
            ];
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
