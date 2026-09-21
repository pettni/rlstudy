use std::collections::VecDeque;

use crate::algorithms::{ModelFreeSolver, SolveResult, SolveStatus};
use crate::environment::{Environment, Space};
use crate::network::mlp::MlpModel;
use crate::policy::{
    Policy,
    dq::{DeepQPolicy, DeepQPolicyMode},
};
use crate::reporting::AbstractReporter;
use crate::schedule::Schedule;
use burn::{
    Tensor,
    backend::{Autodiff, Flex, flex::FlexDevice},
    module::AutodiffModule,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Int, TensorData, backend::Backend},
};
use clap::Args;
use ndarray::Array1;
use rand::RngExt;
use serde::{Deserialize, Serialize};

type TrainingBackend = Autodiff<Flex>;

#[derive(Debug, Clone, PartialEq, Args, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ApproximateTDConfig {
    #[arg(long, default_value_t = ApproximateTDConfig::default().n)]
    pub n: usize,
    #[arg(long, default_value_t = ApproximateTDConfig::default().gamma)]
    pub gamma: f64,
    #[arg(long, default_value_t = ApproximateTDConfig::default().epsilon)]
    pub epsilon: f64,
    #[arg(long, default_value = "linear:0.001:0.0001:999")]
    pub learning_rate: Schedule,
    #[arg(long, default_value = "linear:0.001:0.0001:999")]
    pub hidden_dims: usize,
    #[arg(long, default_value_t = ApproximateTDConfig::default().episodes)]
    pub episodes: usize,
    #[arg(long, default_value_t = ApproximateTDConfig::default().episode_length)]
    pub episode_length: usize,
}

impl Default for ApproximateTDConfig {
    fn default() -> Self {
        Self {
            n: 1,
            gamma: 0.9,
            epsilon: 0.1,
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

pub struct ApproximateTD {
    config: ApproximateTDConfig,
}

impl ApproximateTD {
    pub fn new(config: ApproximateTDConfig) -> Result<Self, String> {
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

impl ModelFreeSolver for ApproximateTD {
    type Observation = Array1<f64>;

    fn solve_with_reporter(
        &self,
        env: &mut dyn Environment<Observation = Array1<f64>, Action = usize>,
        rng: &mut dyn rand::Rng,
        reporter: &mut dyn AbstractReporter<Array1<f64>>,
    ) -> Result<SolveResult<Array1<f64>>, String> {
        let Space::Continuous { dim: n_s } = env.state_space() else {
            return Err("Actor-critic requires a continuous state space".into());
        };
        let Space::Discrete { size: n_a } = env.action_space() else {
            return Err("Actor-critic requires a discrete action space".into());
        };

        let device = FlexDevice;
        <TrainingBackend as Backend>::seed(&device, rng.random());

        let mut optimizer: OptimizerAdaptor<Adam, MlpModel<TrainingBackend>, TrainingBackend> =
            AdamConfig::new().init();

        // q(s) = logits over actions
        let mut q = MlpModel::new(n_s, self.config.hidden_dims, n_a, &device);

        for episode_index in 0..self.config.episodes {
            let mut state = env.reset(rng);
            let mut episode_return = 0.0;
            let mut episode_length = 0;
            let mut entropy_sum = 0.0;
            let mut actor_gradient_norm_sum = 0.0;
            let mut critic_gradient_norm_sum = 0.0;

            let learning_rate = self.config.learning_rate.value(episode_index);

            let mut terminated = false;

            // [st-n, ..., st]
            let mut st = VecDeque::with_capacity(self.config.n);
            // [at-n, ..., at]
            let mut at = VecDeque::with_capacity(self.config.n);
            // [rt-n+1, ..., rt+1]
            let mut rtp = VecDeque::with_capacity(self.config.n);

            st.push_back(state.clone());
            at.push_back(
                DeepQPolicy::borrowed(
                    &q,
                    DeepQPolicyMode::SoftGreedy {
                        epsilon: self.config.epsilon,
                    },
                )
                .sample_action(&state, rng),
            );

            for t in 0..self.config.episode_length {
                let policy = DeepQPolicy::borrowed(
                    &q,
                    DeepQPolicyMode::SoftGreedy {
                        epsilon: self.config.epsilon,
                    },
                );

                if !terminated {
                    let step = env.step(*at.back().unwrap(), rng);

                    rtp.push_back(step.reward);

                    if step.terminated || step.truncated {
                        terminated = true;
                    } else {
                        st.push_back(step.observation);
                        at.push_back(policy.sample_action(&st.back().unwrap(), rng));
                    }
                } else {
                    // peel off oldest ones
                    st.pop_front();
                    at.pop_front();
                    rtp.pop_front();
                }

                if t >= self.config.n {
                    let t_s_left = Tensor::<TrainingBackend, 2>::from_data(
                        TensorData::new(st.front().unwrap().to_vec(), [1, n_s]),
                        &device,
                    );
                    let t_a_left = Tensor::<TrainingBackend, 2, Int>::from_data(
                        TensorData::new(vec![*at.front().unwrap() as i64], [1, 1]),
                        &device,
                    );

                    // g = n-step return + bootstrap tail
                    let g: f64 = rtp
                        .iter()
                        .enumerate()
                        .map(|(i, rt)| self.config.gamma.powi(i as i32) * rt)
                        .sum();

                    // bootstrap tail
                    let bootstrap_tail = if !terminated {
                        let t_s_rght = Tensor::<TrainingBackend, 2>::from_data(
                            TensorData::new(st.back().unwrap().to_vec(), [1, n_s]),
                            &device,
                        );
                        let t_a_rght = Tensor::<TrainingBackend, 2, Int>::from_data(
                            TensorData::new(vec![*at.back().unwrap() as i64], [1, n_s]),
                            &device,
                        );
                        let q_bootstrap = q.forward(t_s_rght).gather(1, t_a_rght).squeeze_dim(1);

                        self.config.gamma.powi(self.config.n as i32) * q_bootstrap
                    } else {
                        Tensor::<TrainingBackend, 2>::zeros([1, 1], &device)
                    };

                    // optimize td error
                    let td_target: Tensor<TrainingBackend, 2> = g + bootstrap_tail;
                    let td_error = td_target.detach() - q.forward(t_s_left).gather(1, t_a_left);

                    let loss = td_error.square();
                    let grad = GradientsParams::from_grads(loss.backward(), &q);
                    q = optimizer.step(learning_rate, q, grad);
                }
            }
        }

        Ok(SolveResult {
            policy: Box::new(DeepQPolicy::new(q.valid(), DeepQPolicyMode::Greedy)),
            status: SolveStatus::IterationLimit {
                iterations: self.config.episodes,
            },
        })
    }
}
