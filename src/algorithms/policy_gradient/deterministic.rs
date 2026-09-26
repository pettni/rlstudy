use crate::algorithms::{ModelFreeSolver, SolveResult, SolveStatus};
use crate::backend::{TrainingBackend, device};
use crate::environment::{Environment, Rollout};
use crate::network::{AdamOptimizer, MlpModel, MlpSettings, OutputTransform, ema};
use crate::policy::deterministic::DeterministicPolicy;
use crate::policy::uniform::UniformContinuousPolicy;
use crate::reporting::{MetricAverages, Reporter, Step};
use crate::transition_buffer::TransitionBuffer;
use burn::tensor::s;
use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{ElementConversion, Tensor},
};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct TargetNoise {
    pub noise_level: f64,
    pub clip: f64,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DeterministicPolicyGradientConfig {
    pub gamma: f64,
    pub batch_size: usize,
    pub replay_capacity: usize,
    pub warm_up_transitions: usize,
    pub steps_per_iteration: usize,
    pub q_updates_per_iteration: usize,
    pub pi_updates_every_n_q_updates: usize,
    pub exploration_noise: Option<f64>,
    pub iterations: usize,
    pub ema_rate: f64,
    pub q_network: MlpSettings,
    pub double_q_learning: bool,
    pub pi_network: MlpSettings,
    pub target_noise: Option<TargetNoise>,
}

impl Default for DeterministicPolicyGradientConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            batch_size: 64,
            replay_capacity: 1_000,
            warm_up_transitions: 64,
            steps_per_iteration: 1,
            q_updates_per_iteration: 1,
            pi_updates_every_n_q_updates: 1,
            exploration_noise: None,
            iterations: 1_000,
            ema_rate: 0.01,
            q_network: MlpSettings::default(),
            double_q_learning: false,
            pi_network: MlpSettings::default(),
            target_noise: None,
        }
    }
}

pub struct DeterministicPolicyGradient {
    config: DeterministicPolicyGradientConfig,
}

impl DeterministicPolicyGradient {
    pub fn new(config: DeterministicPolicyGradientConfig) -> Result<Self, String> {
        Ok(Self { config })
    }
}

impl ModelFreeSolver for DeterministicPolicyGradient {
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
            .ok_or("DeterministicPolicyGradient requires a continuous state space")?;
        let (action_low, action_high) = env
            .action_space()
            .into_continuous_bounds()
            .ok_or("DeterministicPolicyGradient requires a continuous action space")?;
        let n_a = action_low.len();
        if action_high.len() != n_a
            || action_low
                .iter()
                .zip(&action_high)
                .any(|(&low, &high)| !low.is_finite() || !high.is_finite() || low >= high)
        {
            return Err(
                "DeterministicPolicyGradient requires finite action bounds with low < high".into(),
            );
        }

        let mut pi = self.config.pi_network.init(n_s, n_a).with_output_transform(
            OutputTransform::bounded_tanh(
                action_low.as_slice().expect("action bounds are contiguous"),
                action_high
                    .as_slice()
                    .expect("action bounds are contiguous"),
                device(),
            ),
        );
        let mut pi_optimizer: AdamOptimizer<MlpModel<TrainingBackend>> = AdamConfig::new().init();
        let mut pi_target = pi.clone();

        let q_model_factory = || self.config.q_network.init(n_s + n_a, 1);

        let mut q1 = Qdata::new(q_model_factory());
        let mut q2 = Qdata::new(q_model_factory());

        let mut buffer = TransitionBuffer::<Self::Action>::new(Some(self.config.replay_capacity));

        let mut rollout = Rollout::new(env, rng);
        let warmup_policy = UniformContinuousPolicy::new(&action_low, &action_high);
        buffer.fill_n(
            self.config.warm_up_transitions,
            &mut rollout,
            &warmup_policy,
            rng,
        );

        let mut q_update_iteration = 0;
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

            let q_lr = self.config.q_network.learning_rate.value(iteration);
            let pi_lr = self.config.pi_network.learning_rate.value(iteration);
            let mut update_metrics = MetricAverages::new();

            for _ in 0..self.config.q_updates_per_iteration {
                let batch = buffer.random_batch_tensor_data(self.config.batch_size, rng);

                let mut next_action = pi_target.forward(batch.sp.clone());

                // action noise, clip, clamp
                if let Some(TargetNoise { noise_level, clip }) = self.config.target_noise {
                    let mut noise = noise_level
                        * next_action
                            .clone()
                            .random_like(burn::tensor::Distribution::Normal(0., 1.));

                    noise = noise.clamp(-clip, clip);

                    next_action = next_action + noise;
                    for ai in 0..n_a {
                        let clamped_col = next_action
                            .clone()
                            .slice(s![.., ai])
                            .clamp(action_low[ai], action_high[ai]);

                        next_action = next_action.slice_assign(s![.., ai], clamped_col);
                    }
                }

                let next_value = match self.config.double_q_learning {
                    false => q1
                        .target
                        .forward(Tensor::cat(vec![batch.sp.clone(), next_action], 1))
                        .squeeze_dim(1),
                    true => {
                        let q1_val = q1
                            .target
                            .forward(Tensor::cat(vec![batch.sp.clone(), next_action.clone()], 1))
                            .squeeze_dim(1);
                        let q2_val = q2
                            .target
                            .forward(Tensor::cat(vec![batch.sp.clone(), next_action], 1))
                            .squeeze_dim(1);
                        q1_val.min_pair(q2_val)
                    }
                };

                let td_target =
                    batch.r + self.config.gamma * next_value * batch.terminated.bool_not().float();
                let td_target = td_target.detach();
                let s_a = Tensor::cat(vec![batch.s.clone(), batch.a.clone()], 1);

                // step q  -- objective:  q(s,a)  ~  r + gamma * q_T(sp, pi_T(sp))

                let (q1_updated, q1_loss) = q1.update(s_a.clone(), td_target.clone(), q_lr);
                q1 = q1_updated;
                update_metrics.record("q1/loss", q1_loss);

                if self.config.double_q_learning {
                    let (q2_updated, q2_loss) = q2.update(s_a, td_target, q_lr);
                    q2 = q2_updated;
                    update_metrics.record("q2/loss", q2_loss);
                }

                if (q_update_iteration + 1) % self.config.pi_updates_every_n_q_updates == 0 {
                    // step pi  -- objective: maximize q(s, pi(s))
                    let pi_a = pi.forward(batch.s.clone());
                    // note: inefficient because we get dpi_loss/dq1_params which we dont need..
                    let pi_loss = -q1.pred.forward(Tensor::cat(vec![batch.s, pi_a], 1)).mean();

                    update_metrics.record("pi/loss", pi_loss.clone().into_scalar().elem::<f64>());
                    let pi_grad = GradientsParams::from_grads(pi_loss.backward(), &pi);

                    pi = pi_optimizer.step(pi_lr, pi, pi_grad);

                    // ema on target networks
                    q1.target = ema::update(q1.target, &q1.pred, self.config.ema_rate);
                    q2.target = ema::update(q2.target, &q2.pred, self.config.ema_rate);
                    pi_target = ema::update(pi_target, &pi, self.config.ema_rate);
                }

                q_update_iteration += 1;
            }

            let mut metrics = vec![
                ("q/lr", q_lr),
                ("pi/lr", pi_lr),
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

struct Qdata {
    pred: MlpModel<TrainingBackend>,
    target: MlpModel<TrainingBackend>,

    optimizer: AdamOptimizer<MlpModel<TrainingBackend>>,
}

impl Qdata {
    fn new(network: MlpModel<TrainingBackend>) -> Self {
        Self {
            pred: network.clone(),
            target: network,
            optimizer: AdamConfig::new().init(),
        }
    }

    fn update(
        mut self,
        inputs: Tensor<TrainingBackend, 2>,
        td_target: Tensor<TrainingBackend, 1>,
        lr: f64,
    ) -> (Self, f64) {
        let qi_loss = (td_target - self.pred.forward(inputs).squeeze())
            .square()
            .mean();

        let loss_value = qi_loss.clone().into_scalar().elem::<f64>();
        let grad = GradientsParams::from_grads(qi_loss.backward(), &self.pred);

        // gradient descent on pred network
        self.pred = self.optimizer.step(lr, self.pred, grad);

        (self, loss_value)
    }
}
