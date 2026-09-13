use crate::algorithms::{ModelFreeSolver, SolveResult, SolveStatus};
use crate::environment::{Environment, Space};
use crate::network::mlp::MlpModel;
use crate::policy::{
    DiscretePolicy, Policy,
    dq::{DeepQPolicy, DeepQPolicyMode},
};
use crate::replay_buffer::{ReplayBuffer, Transition};
use burn::{
    Tensor,
    backend::{Autodiff, Flex, flex::FlexDevice},
    module::AutodiffModule,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Int, TensorData},
};
use ndarray::Array1;

type Device = FlexDevice;
type TrainingBackend = Autodiff<Flex>;

/// Deep Q learning algorithm using a Burn network.
pub struct DeepQLearning {
    learning_rate: f64,
    batch_size: usize,
    replay_capacity: usize,
    iterations: usize,
    target_sync_steps: usize,
    num_hidden_dims: usize,
    behavior_policy: Box<dyn DiscretePolicy<Observation = Array1<f64>>>,
}

impl DeepQLearning {
    pub fn new(
        learning_rate: f64,
        batch_size: usize,
        replay_capacity: usize,
        iterations: usize,
        target_sync_steps: usize,
        num_hidden_dims: usize,
        behavior_policy: Box<dyn DiscretePolicy<Observation = Array1<f64>>>,
    ) -> Result<Self, String> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning rate must be finite and positive".into());
        }
        if batch_size == 0 || batch_size > replay_capacity {
            return Err("batch size must be positive and no greater than replay capacity".into());
        }
        if iterations == 0 || target_sync_steps == 0 || num_hidden_dims == 0 {
            return Err(
                "iterations, target sync steps, and hidden dimensions must be positive".into(),
            );
        }

        Ok(Self {
            learning_rate,
            batch_size,
            replay_capacity,
            iterations,
            target_sync_steps,
            num_hidden_dims,
            behavior_policy,
        })
    }
}

impl ModelFreeSolver for DeepQLearning {
    type Observation = Array1<f64>;
    type Output = ();

    fn solve(
        &self,
        env: &mut dyn Environment<Observation = Array1<f64>, Action = usize>,
        gamma: f64,
        rng: &mut dyn rand::Rng,
    ) -> Result<SolveResult<Array1<f64>, ()>, String> {
        let Space::Discrete { size: n_a } = env.action_space() else {
            return Err("DQN requires a discrete action space".into());
        };

        let mut s = env.reset(rng);

        let pi_behavior = self.behavior_policy.as_ref();
        let f_dim = s.len();

        // create replay buffer (s,a,r,s') from behavior policy
        let mut buffer = ReplayBuffer::<Array1<f64>, usize>::new(self.replay_capacity);
        for _ in 0..buffer.capacity() {
            let a = pi_behavior.sample_action(&s, rng);
            let (sp, r) = env.step(a, rng);

            buffer.push(Transition {
                state: s,
                action: a,
                reward: r,
                next_state: sp.clone(),
                terminated: false,
            });
            s = sp;
        }

        let device = Device::default();

        let mut q_target =
            MlpModel::<TrainingBackend>::new(f_dim, self.num_hidden_dims, n_a, &device);
        let mut q_online = q_target.clone();

        let mut optimizer: OptimizerAdaptor<Adam, MlpModel<TrainingBackend>, TrainingBackend> =
            AdamConfig::new().init();

        for iter in 0..self.iterations {
            // sample mini-batch uniformly from replay buffer
            let batch = buffer.sample(self.batch_size, rng);

            // unpack into s, a, r, sp
            let mut state_data = Vec::with_capacity(self.batch_size * f_dim);
            let mut action_data = Vec::with_capacity(self.batch_size);
            let mut reward_data = Vec::with_capacity(self.batch_size);
            let mut next_state_data = Vec::with_capacity(self.batch_size * f_dim);

            for transition in &batch {
                state_data.extend(transition.state.iter().copied());
                action_data.push(transition.action as i64);
                reward_data.push(transition.reward);
                next_state_data.extend(transition.next_state.iter().copied());
            }

            let states = Tensor::<TrainingBackend, 2>::from_data(
                TensorData::new(state_data, [self.batch_size, f_dim]),
                &device,
            );
            let actions = Tensor::<TrainingBackend, 1, Int>::from_data(
                TensorData::new(action_data, [self.batch_size]),
                &device,
            );
            let rewards = Tensor::<TrainingBackend, 1>::from_data(
                TensorData::new(reward_data, [self.batch_size]),
                &device,
            );
            let next_states = Tensor::<TrainingBackend, 2>::from_data(
                TensorData::new(next_state_data, [self.batch_size, f_dim]),
                &device,
            );

            // y = q_online(s, a)
            let y = q_online
                .forward(states)
                .gather(1, actions.reshape([self.batch_size, 1]))
                .squeeze_dim(1);

            // yT = r + gamma * max_a' q_target(s', a')  [stopgrad]
            let yt = rewards
                + gamma
                    * q_target
                        .forward(next_states)
                        .max_dim(1)
                        .squeeze_dim(1)
                        .detach();

            let loss = (yt - y).powf_scalar(2.0).mean();

            let gradients = loss.backward();
            let gradients = GradientsParams::from_grads(gradients, &q_online);

            q_online = optimizer.step(self.learning_rate, q_online, gradients);

            if (iter + 1) % self.target_sync_steps == 0 {
                q_target = q_online.clone();
            }
        }

        let inference_q: MlpModel<Flex> = q_online.valid();

        Ok(SolveResult {
            policy: Box::new(DeepQPolicy::new(inference_q, DeepQPolicyMode::Greedy)),
            // TODO: loss(iter), periodic policy evals
            output: (),
            status: SolveStatus::IterationLimit {
                iterations: self.iterations,
            },
        })
    }
}
