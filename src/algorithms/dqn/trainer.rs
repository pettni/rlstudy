use crate::network::mlp::MlpModel;
use crate::policy::dq::{DeepQPolicy, DeepQPolicyMode};
use crate::replay_buffer::Transition;
use burn::{
    Tensor,
    backend::{Autodiff, Flex, flex::FlexDevice},
    lr_scheduler::{LrScheduler, linear::LinearLrScheduler},
    module::AutodiffModule,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Int, TensorData},
};
use ndarray::Array1;

type TrainingBackend = Autodiff<Flex>;

pub(super) struct DqnTrainer {
    online: MlpModel<TrainingBackend>,
    target: MlpModel<TrainingBackend>,
    optimizer: OptimizerAdaptor<Adam, MlpModel<TrainingBackend>, TrainingBackend>,
    device: FlexDevice,
    batch_size: usize,
    feature_dim: usize,
    learning_rate_scheduler: LinearLrScheduler,
}

impl DqnTrainer {
    pub(super) fn new(
        feature_dim: usize,
        hidden_dim: usize,
        num_actions: usize,
        batch_size: usize,
        learning_rate_scheduler: LinearLrScheduler,
    ) -> Self {
        let device = FlexDevice::default();
        let target = MlpModel::new(feature_dim, hidden_dim, num_actions, &device);

        Self {
            online: target.clone(),
            target,
            optimizer: AdamConfig::new().init(),
            device,
            batch_size,
            feature_dim,
            learning_rate_scheduler,
        }
    }

    pub(super) fn update(
        mut self,
        batch: &[Transition<Array1<f64>, usize>],
        gamma: f64,
    ) -> (Self, f64, f64) {
        let mut state_data = Vec::with_capacity(self.batch_size * self.feature_dim);
        let mut action_data = Vec::with_capacity(self.batch_size);
        let mut reward_data = Vec::with_capacity(self.batch_size);
        let mut next_state_data = Vec::with_capacity(self.batch_size * self.feature_dim);
        let mut bootstrap_data = Vec::with_capacity(self.batch_size);

        for transition in batch {
            state_data.extend(transition.state.iter().copied());
            action_data.push(transition.action as i64);
            reward_data.push(transition.reward);
            next_state_data.extend(transition.next_state.iter().copied());
            bootstrap_data.push(if transition.terminated { 0.0 } else { 1.0 });
        }

        let states = Tensor::<TrainingBackend, 2>::from_data(
            TensorData::new(state_data, [self.batch_size, self.feature_dim]),
            &self.device,
        );
        let actions = Tensor::<TrainingBackend, 1, Int>::from_data(
            TensorData::new(action_data, [self.batch_size]),
            &self.device,
        );
        let rewards = Tensor::<TrainingBackend, 1>::from_data(
            TensorData::new(reward_data, [self.batch_size]),
            &self.device,
        );
        let next_states = Tensor::<TrainingBackend, 2>::from_data(
            TensorData::new(next_state_data, [self.batch_size, self.feature_dim]),
            &self.device,
        );
        let bootstrap = Tensor::<TrainingBackend, 1>::from_data(
            TensorData::new(bootstrap_data, [self.batch_size]),
            &self.device,
        );

        let predicted = self
            .online
            .forward(states)
            .gather(1, actions.reshape([self.batch_size, 1]))
            .squeeze_dim(1);
        let target = rewards
            + bootstrap
                * gamma
                * self
                    .target
                    .forward(next_states)
                    .max_dim(1)
                    .squeeze_dim(1)
                    .detach();
        let loss = (target - predicted).powf_scalar(2.0).mean();
        let loss_value = loss
            .clone()
            .into_data()
            .iter::<f64>()
            .next()
            .expect("loss tensor has one element");
        let gradients = GradientsParams::from_grads(loss.backward(), &self.online);

        let learning_rate = self.learning_rate_scheduler.step();
        self.online = self.optimizer.step(learning_rate, self.online, gradients);
        (self, loss_value, learning_rate)
    }

    pub(super) fn sync_target(&mut self) {
        self.target = self.online.clone();
    }

    pub(super) fn soft_update_target(&mut self, rate: f64) {
        self.target = self.target.clone().soft_update(&self.online, rate);
    }

    pub(super) fn policy(&self) -> DeepQPolicy {
        DeepQPolicy::new(self.online.clone().valid(), DeepQPolicyMode::Greedy)
    }

    pub(super) fn into_policy(self) -> DeepQPolicy {
        DeepQPolicy::new(self.online.valid(), DeepQPolicyMode::Greedy)
    }
}
