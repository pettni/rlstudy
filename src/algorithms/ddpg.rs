use crate::algorithms::{ModelFreeSolver, SolveResult, SolveStatus};
use crate::backend::{TrainingBackend, device};
use crate::environment::{Environment, Rollout};
use crate::network::{
    MlpSettings, ema,
    mlp::{Activation, MlpModel, OutputTransform},
};
use crate::policy::deterministic::DeterministicPolicy;
use crate::policy::uniform::UniformContinuousPolicy;
use crate::reporting::{MetricAverages, Reporter, Step};
use crate::transition_buffer::TransitionBuffer;
use burn::{
    module::AutodiffModule,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{ElementConversion, Tensor},
};
use clap::Args;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DDPGConfig {
    pub gamma: f64,
    pub batch_size: usize,
    pub replay_capacity: usize,
    pub warm_up_transitions: usize,
    pub steps_per_iteration: usize,
    pub updates_per_iteration: usize,
    pub exploration_noise: Option<f64>,
    pub iterations: usize,
    pub ema_rate: f64,
    pub q_network: MlpSettings,
    pub pi_network: MlpSettings,
}

impl Default for DDPGConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            batch_size: 64,
            replay_capacity: 1_000,
            warm_up_transitions: 64,
            steps_per_iteration: 1,
            updates_per_iteration: 1,
            exploration_noise: None,
            iterations: 1_000,
            ema_rate: 0.01,
            q_network: MlpSettings::default(),
            pi_network: MlpSettings::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Args)]
pub struct DdpgArgs {
    #[arg(long, default_value_t = DDPGConfig::default().gamma)]
    gamma: f64,
    #[arg(long, default_value_t = DDPGConfig::default().batch_size)]
    batch_size: usize,
    #[arg(long, default_value_t = DDPGConfig::default().replay_capacity)]
    replay_capacity: usize,
    #[arg(long, default_value_t = DDPGConfig::default().warm_up_transitions)]
    warm_up_transitions: usize,
    #[arg(long, default_value_t = DDPGConfig::default().steps_per_iteration)]
    steps_per_iteration: usize,
    #[arg(long, default_value_t = DDPGConfig::default().updates_per_iteration)]
    updates_per_iteration: usize,
    #[arg(long)]
    exploration_noise: Option<f64>,
    #[arg(long, default_value_t = DDPGConfig::default().iterations)]
    iterations: usize,
    #[arg(long, default_value_t = DDPGConfig::default().ema_rate)]
    ema_rate: f64,
    #[arg(long)]
    q_learning_rate: Option<crate::schedule::Schedule>,
    #[arg(long, value_delimiter = ',')]
    q_hidden_dims: Option<Vec<usize>>,
    #[arg(long, value_enum)]
    q_activation: Option<Activation>,
    #[arg(long)]
    pi_learning_rate: Option<crate::schedule::Schedule>,
    #[arg(long, value_delimiter = ',')]
    pi_hidden_dims: Option<Vec<usize>>,
    #[arg(long, value_enum)]
    pi_activation: Option<Activation>,
}

impl From<DdpgArgs> for DDPGConfig {
    fn from(args: DdpgArgs) -> Self {
        let defaults = Self::default();
        Self {
            gamma: args.gamma,
            batch_size: args.batch_size,
            replay_capacity: args.replay_capacity,
            warm_up_transitions: args.warm_up_transitions,
            steps_per_iteration: args.steps_per_iteration,
            updates_per_iteration: args.updates_per_iteration,
            exploration_noise: args.exploration_noise,
            iterations: args.iterations,
            ema_rate: args.ema_rate,
            q_network: MlpSettings {
                learning_rate: args
                    .q_learning_rate
                    .unwrap_or(defaults.q_network.learning_rate),
                hidden_dims: args.q_hidden_dims.unwrap_or(defaults.q_network.hidden_dims),
                activation: args.q_activation.unwrap_or(defaults.q_network.activation),
            },
            pi_network: MlpSettings {
                learning_rate: args
                    .pi_learning_rate
                    .unwrap_or(defaults.pi_network.learning_rate),
                hidden_dims: args
                    .pi_hidden_dims
                    .unwrap_or(defaults.pi_network.hidden_dims),
                activation: args.pi_activation.unwrap_or(defaults.pi_network.activation),
            },
        }
    }
}

pub struct DDPG {
    config: DDPGConfig,
}

impl DDPG {
    pub fn new(config: DDPGConfig) -> Result<Self, String> {
        Ok(Self { config })
    }
}

impl ModelFreeSolver for DDPG {
    type Action = Array1<f64>;
    type Observation = Array1<f64>;

    fn solve_with_reporter(
        &self,
        env: &mut dyn Environment<Observation = Self::Observation, Action = Self::Action>,
        rng: &mut dyn rand::Rng,
        reporter: &mut Reporter<'_, Self::Observation, Self::Action>,
    ) -> Result<SolveResult<Self::Observation, Self::Action>, String> {
        let n_s = env
            .state_space()
            .continuous_dim()
            .ok_or("DDPG requires a continuous state space")?;
        let (action_low, action_high) = env
            .action_space()
            .into_continuous_bounds()
            .ok_or("DDPG requires a continuous action space")?;
        let n_a = action_low.len();
        if action_high.len() != n_a
            || action_low
                .iter()
                .zip(&action_high)
                .any(|(&low, &high)| !low.is_finite() || !high.is_finite() || low >= high)
        {
            return Err("DDPG requires finite action bounds with low < high".into());
        }

        let mut pi: MlpModel<TrainingBackend> = MlpModel::new_with_hidden_layers(
            n_s,
            &self.config.pi_network.hidden_dims,
            n_a,
            self.config.pi_network.activation,
            device(),
        )
        .with_output_transform(OutputTransform::bounded_tanh(
            action_low.as_slice().expect("action bounds are contiguous"),
            action_high
                .as_slice()
                .expect("action bounds are contiguous"),
            device(),
        ));
        let mut q: MlpModel<TrainingBackend> = MlpModel::new_with_hidden_layers(
            n_s + n_a,
            &self.config.q_network.hidden_dims,
            1,
            self.config.q_network.activation,
            device(),
        );
        let mut pi_target = pi.clone();
        let mut q_target = q.clone();

        let mut pi_optimizer: OptimizerAdaptor<Adam, MlpModel<TrainingBackend>, TrainingBackend> =
            AdamConfig::new().init();
        let mut q_optimizer: OptimizerAdaptor<Adam, MlpModel<TrainingBackend>, TrainingBackend> =
            AdamConfig::new().init();

        let mut buffer = TransitionBuffer::<Self::Action>::new(Some(self.config.replay_capacity));

        let mut rollout = Rollout::new(env, rng);
        let warmup_policy = UniformContinuousPolicy::new(&action_low, &action_high);
        buffer.fill_n(
            self.config.warm_up_transitions,
            &mut rollout,
            &warmup_policy,
            rng,
        );

        for iteration in 0..self.config.iterations {
            let step = Step::Iteration(iteration + 1);
            reporter.progress(step, self.config.iterations);

            let policy = DeterministicPolicy::borrowed(
                &pi,
                &action_low,
                &action_high,
                self.config.exploration_noise,
            );
            buffer.fill_n(self.config.steps_per_iteration, &mut rollout, &policy, rng);

            let q_learning_rate = self.config.q_network.learning_rate.value(iteration);
            let pi_learning_rate = self.config.pi_network.learning_rate.value(iteration);
            let mut update_metrics = MetricAverages::new();

            for _ in 0..self.config.updates_per_iteration {
                let batch = buffer.random_batch_tensor_data(self.config.batch_size, rng);

                let ap = pi_target.forward(batch.sp.clone());

                // step q  -- objective:  q(s,a)  ~  r + gamma * q_T(sp, pi_T(sp))

                let target = batch.r
                    + self.config.gamma
                        * q_target
                            .forward(Tensor::cat(vec![batch.sp.clone(), ap], 1))
                            .squeeze_dim(1)
                        * batch.terminated.bool_not().float();

                let q_loss = (target.detach()
                    - q.forward(Tensor::cat(vec![batch.s.clone(), batch.a.clone()], 1))
                        .squeeze())
                .square()
                .mean();

                update_metrics.record("q/loss", q_loss.clone().into_scalar().elem::<f64>());
                let q_grad = GradientsParams::from_grads(q_loss.backward(), &q);

                q = q_optimizer.step(q_learning_rate, q, q_grad);

                // step pi  -- objective: maximize q(s, pi(s))

                let pi_a = pi.forward(batch.s.clone());
                let pi_loss = -q.forward(Tensor::cat(vec![batch.s, pi_a], 1)).mean();

                update_metrics.record("pi/loss", pi_loss.clone().into_scalar().elem::<f64>());
                let pi_grad = GradientsParams::from_grads(pi_loss.backward(), &pi);

                pi = pi_optimizer.step(pi_learning_rate, pi, pi_grad);

                // ema on target networks

                q_target = ema::update(q_target, &q, self.config.ema_rate);
                pi_target = ema::update(pi_target, &pi, self.config.ema_rate);
            }

            let mut metrics = vec![
                ("q/lr", q_learning_rate),
                ("pi/lr", pi_learning_rate),
                ("buffer/size", buffer.len() as f64),
            ];
            metrics.extend(update_metrics.finish());
            reporter.record(step, &metrics);
            reporter.policy(step, || {
                DeterministicPolicy::new(pi.valid(), action_low.clone(), action_high.clone(), None)
            });
        }

        Ok(SolveResult {
            policy: Box::new(DeterministicPolicy::new(
                pi.valid(),
                action_low,
                action_high,
                None,
            )),
            status: SolveStatus::IterationLimit {
                iterations: self.config.iterations,
            },
        })
    }
}
