use burn::{
    module::{Module, Param},
    tensor::{Tensor, TensorData, backend::Backend},
};
use ndarray::Array1;

use super::{Actor, LOG_TWO_PI};
use crate::backend::{InferenceBackend, TrainingBackend, device};
use crate::network::{Activation, MlpModel};
use crate::policy::Policy;

#[derive(Module, Debug)]
pub struct GaussianActor<B: Backend> {
    mu_mlp: MlpModel<B>,
    log_std: Param<Tensor<B, 1>>,
}

impl<B: Backend> GaussianActor<B> {
    pub fn new(
        state_dim: usize,
        hidden_dims: &[usize],
        action_dim: usize,
        activation: Activation,
        device: &B::Device,
    ) -> Self {
        Self {
            mu_mlp: MlpModel::new(state_dim, hidden_dims, action_dim, activation, device),
            log_std: Param::from_tensor(Tensor::full([action_dim], -0.5, device)),
        }
    }

    fn parameters(&self, states: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mu = self.mu_mlp.forward(states);
        let [batch_size, action_dim] = mu.dims();
        let log_std = self
            .log_std
            .val()
            .unsqueeze_dim::<2>(0)
            .expand([batch_size, action_dim]);
        (mu, log_std)
    }
}

impl Actor<Array1<f64>> for GaussianActor<TrainingBackend> {
    type InferencePolicy = GaussianActor<InferenceBackend>;

    fn log_probs(
        &self,
        states: Tensor<TrainingBackend, 2>,
        actions: &Tensor<TrainingBackend, 2>,
    ) -> Tensor<TrainingBackend, 1> {
        let (mu, log_std) = self.parameters(states);
        let normalized = (actions.clone() - mu) / log_std.clone().exp();

        let log_probs: Tensor<TrainingBackend, 2> =
            -0.5 * (normalized.square() + LOG_TWO_PI) - log_std;

        log_probs.sum_dim(1).squeeze_dim(1)
    }

    fn sample_actions(
        &self,
        observations: Tensor<TrainingBackend, 2>,
    ) -> (Tensor<TrainingBackend, 2>, Tensor<TrainingBackend, 1>) {
        let (mu, log_std) = self.parameters(observations);
        let noise = mu.random_like(burn::tensor::Distribution::Normal(0., 1.));
        let actions = mu + noise.clone() * log_std.clone().exp();
        let log_probs: Tensor<TrainingBackend, 2> = -0.5 * (noise.square() + LOG_TWO_PI) - log_std;

        (actions, log_probs.sum_dim(1).squeeze_dim(1))
    }

    fn entropy(&self, states: Tensor<TrainingBackend, 2>) -> Tensor<TrainingBackend, 1> {
        let (_, log_std) = self.parameters(states);

        (log_std + 0.5 * (1.0 + LOG_TWO_PI))
            .sum_dim(1)
            .squeeze_dim(1)
    }
}

impl Policy for GaussianActor<TrainingBackend> {
    type Observation = Array1<f64>;
    type Action = Array1<f64>;

    fn sample_action(&self, state: &Array1<f64>, _rng: &mut dyn rand::Rng) -> Self::Action {
        let state = Tensor::<TrainingBackend, 2>::from_data(
            TensorData::new(state.to_vec(), [1, state.len()]),
            device(),
        );
        let (action, _) = self.sample_actions(state);
        Array1::from_iter(action.into_data().iter())
    }
}

impl Policy for GaussianActor<InferenceBackend> {
    type Observation = Array1<f64>;
    type Action = Array1<f64>;

    fn sample_action(&self, state: &Array1<f64>, _rng: &mut dyn rand::Rng) -> Self::Action {
        let state = Tensor::<InferenceBackend, 2>::from_data(
            TensorData::new(state.to_vec(), [1, state.len()]),
            device(),
        );
        let (mu, _) = self.parameters(state);
        Array1::from_iter(mu.into_data().iter())
    }
}
