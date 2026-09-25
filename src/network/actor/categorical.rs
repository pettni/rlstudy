use burn::{
    module::Module,
    tensor::{Int, Tensor, activation::log_softmax, backend::Backend},
};
use ndarray::Array1;
use rand::RngExt;

use super::Actor;
use crate::{
    backend::{Device, InferenceBackend, TrainingBackend, device},
    network::mlp::{Activation, MlpModel},
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
            network: MlpModel::new_with_hidden_layers(
                state_dim,
                hidden_dims,
                action_dim,
                activation,
                device,
            ),
        }
    }

    pub fn forward(&self, states: Tensor<B, 2>) -> Tensor<B, 2> {
        self.network.forward(states)
    }
}

impl<B> CategoricalActor<B>
where
    B: Backend<Device = Device>,
{
    fn probabilities(&self, observation: &Array1<f64>) -> Array1<f64> {
        let observation = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(observation.to_vec(), [1, observation.len()]),
            device(),
        );
        let logits = Array1::from_iter(
            self.forward(observation)
                .squeeze_dim::<1>(0)
                .into_data()
                .iter::<f64>(),
        );
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights = logits.mapv(|logit| (logit - max_logit).exp());
        &weights / weights.sum()
    }
}

impl<B> Policy for CategoricalActor<B>
where
    B: Backend<Device = Device>,
{
    type Observation = Array1<f64>;
    type Action = usize;

    fn sample_action(
        &self,
        observation: &Self::Observation,
        rng: &mut dyn rand::Rng,
    ) -> Self::Action {
        let probabilities = self.probabilities(observation);
        let sample: f64 = rng.random();
        let mut cumulative = 0.0;
        for (action, probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if sample < cumulative {
                return action;
            }
        }

        probabilities.len() - 1
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

    fn entropy(&self, states: Tensor<TrainingBackend, 2>) -> Tensor<TrainingBackend, 1> {
        let log_probs = log_softmax(self.forward(states), 1);
        -(log_probs.clone().exp() * log_probs)
            .sum_dim(1)
            .squeeze_dim(1)
    }
}
