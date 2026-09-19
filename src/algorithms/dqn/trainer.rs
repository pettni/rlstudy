use super::replay_buffer::Transition;
use crate::network::loss::Loss;
use crate::network::mlp::MlpModel;
use crate::policy::dq::{DeepQPolicy, DeepQPolicyMode};
use burn::{
    Tensor,
    backend::{Autodiff, Flex, flex::FlexDevice},
    module::AutodiffModule,
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    tensor::{Int, TensorData, backend::Backend},
};
use ndarray::Array1;

type TrainingBackend = Autodiff<Flex>;

struct TrainingBatch {
    states: Tensor<TrainingBackend, 2>,
    actions: Tensor<TrainingBackend, 1, Int>,
    rewards: Tensor<TrainingBackend, 1>,
    next_states: Tensor<TrainingBackend, 2>,
    bootstrap: Tensor<TrainingBackend, 1>,
}

impl TrainingBatch {
    fn new(
        transitions: &[Transition<Array1<f64>, usize>],
        feature_dim: usize,
        device: &FlexDevice,
    ) -> Self {
        let batch_size = transitions.len();
        let mut state_data = Vec::with_capacity(batch_size * feature_dim);
        let mut action_data = Vec::with_capacity(batch_size);
        let mut reward_data = Vec::with_capacity(batch_size);
        let mut next_state_data = Vec::with_capacity(batch_size * feature_dim);
        let mut bootstrap_data = Vec::with_capacity(batch_size);

        for transition in transitions {
            state_data.extend(transition.state.iter().copied());
            action_data.push(transition.action as i64);
            reward_data.push(transition.reward);
            next_state_data.extend(transition.next_state.iter().copied());
            bootstrap_data.push(if transition.terminated { 0.0 } else { 1.0 });
        }

        Self {
            states: Tensor::from_data(
                TensorData::new(state_data, [batch_size, feature_dim]),
                device,
            ),
            actions: Tensor::from_data(TensorData::new(action_data, [batch_size]), device),
            rewards: Tensor::from_data(TensorData::new(reward_data, [batch_size]), device),
            next_states: Tensor::from_data(
                TensorData::new(next_state_data, [batch_size, feature_dim]),
                device,
            ),
            bootstrap: Tensor::from_data(TensorData::new(bootstrap_data, [batch_size]), device),
        }
    }
}

pub(super) struct DqnTrainer {
    online: MlpModel<TrainingBackend>,
    target: MlpModel<TrainingBackend>,
    optimizer: OptimizerAdaptor<Adam, MlpModel<TrainingBackend>, TrainingBackend>,
    device: FlexDevice,
    feature_dim: usize,
}

impl DqnTrainer {
    pub(super) fn new(
        feature_dim: usize,
        hidden_dim: usize,
        num_actions: usize,
        seed: u64,
    ) -> Self {
        let device = FlexDevice;
        <TrainingBackend as Backend>::seed(&device, seed);
        let target = MlpModel::new(feature_dim, hidden_dim, num_actions, &device);

        Self {
            online: target.clone(),
            target,
            optimizer: AdamConfig::new().init(),
            device,
            feature_dim,
        }
    }

    pub(super) fn update(
        mut self,
        batch: &[Transition<Array1<f64>, usize>],
        learning_rate: f64,
        gamma: f64,
        loss: Loss,
    ) -> (Self, f64) {
        let batch_size = batch.len();
        let batch = TrainingBatch::new(batch, self.feature_dim, &self.device);

        let predicted = self
            .online
            .forward(batch.states)
            .gather(1, batch.actions.reshape([batch_size, 1]))
            .squeeze_dim(1);
        let target = batch.rewards
            + batch.bootstrap
                * gamma
                * self
                    .target
                    .forward(batch.next_states)
                    .max_dim(1)
                    .squeeze_dim(1)
                    .detach();
        let loss = loss.forward(predicted, target);
        let loss_value = loss
            .clone()
            .into_data()
            .iter::<f64>()
            .next()
            .expect("loss tensor has one element");
        let gradients = GradientsParams::from_grads(loss.backward(), &self.online);

        self.online = self.optimizer.step(learning_rate, self.online, gradients);
        (self, loss_value)
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
