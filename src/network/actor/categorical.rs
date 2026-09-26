use burn::{
    module::Module,
    tensor::{
        ElementConversion, Int, Tensor, TensorData,
        activation::{log_softmax, softmax},
        backend::Backend,
    },
};
use ndarray::Array1;

use super::Actor;
use crate::{
    backend::{Device, InferenceBackend, TrainingBackend, device},
    network::{Activation, MlpModel},
    policy::Policy,
};

#[derive(Module, Debug)]
pub struct CategoricalActor<B: Backend> {
    network: MlpModel<B>,
}

impl<B: Backend> CategoricalActor<B> {
    pub fn new(
        state_dim: usize,
        hidden_dims: &[usize],
        action_dim: usize,
        activation: Activation,
        device: &B::Device,
    ) -> Self {
        Self {
            network: MlpModel::new(state_dim, hidden_dims, action_dim, activation, device),
        }
    }

    pub fn forward(&self, states: Tensor<B, 2>) -> Tensor<B, 2> {
        self.network.forward(states)
    }
}

impl<B: Backend<Device = Device>> CategoricalActor<B> {
    fn sample_actions_impl(
        &self,
        observations: Tensor<B, 2>,
        _rng: &mut dyn rand::Rng,
    ) -> Tensor<B, 1, Int> {
        let probs = softmax(self.forward(observations), 1);
        probs.categorical(1).squeeze_dim(1)
    }
}

impl Actor<usize> for CategoricalActor<TrainingBackend> {
    type InferencePolicy = CategoricalActor<InferenceBackend>;

    fn log_probs(
        &self,
        states: Tensor<TrainingBackend, 2>,
        actions: &Tensor<TrainingBackend, 1, Int>,
    ) -> Tensor<TrainingBackend, 1> {
        log_softmax(self.forward(states), 1)
            .gather(1, actions.clone().reshape([-1, 1]))
            .squeeze_dim(1)
    }

    fn sample_actions(
        &self,
        states: Tensor<TrainingBackend, 2>,
        rng: &mut dyn rand::Rng,
    ) -> <usize as crate::network::action::ActionType>::TensorType {
        self.sample_actions_impl(states, rng)
    }

    fn entropy(&self, states: Tensor<TrainingBackend, 2>) -> Tensor<TrainingBackend, 1> {
        let log_probs = log_softmax(self.forward(states), 1);
        -(log_probs.clone().exp() * log_probs)
            .sum_dim(1)
            .squeeze_dim(1)
    }
}

impl<B: Backend<Device = Device>> Policy for CategoricalActor<B> {
    type Observation = Array1<f64>;
    type Action = usize;

    fn sample_action(&self, state: &Array1<f64>, rng: &mut dyn rand::Rng) -> Self::Action {
        let state =
            Tensor::<B, 2>::from_data(TensorData::new(state.to_vec(), [1, state.len()]), device());

        let action = self.sample_actions_impl(state, rng);

        action.into_scalar().elem::<i64>() as usize
    }
}
