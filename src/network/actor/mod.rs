mod categorical;
mod gaussian;

use crate::{
    backend::{TrainingBackend, device},
    network::action::ActionType,
    policy::Policy,
};
use burn::tensor::ElementConversion;
use burn::{module::AutodiffModule, tensor::Tensor};
use ndarray::Array1;

pub use categorical::CategoricalActor;
pub use gaussian::GaussianActor;

pub trait Actor<A: ActionType>:
    AutodiffModule<TrainingBackend, InnerModule = Self::InferencePolicy>
    + Policy<Observation = Array1<f64>, Action = A>
{
    type InferencePolicy: Policy<Observation = Array1<f64>, Action = A>;

    fn log_prob(&self, state: &Array1<f64>, action: &A) -> f64 {
        let states = Tensor::<TrainingBackend, 2>::from_data(
            burn::tensor::TensorData::new(state.to_vec(), [1, state.len()]),
            device(),
        );

        let actions = A::tensor_data(&[action]);

        self.log_probs(states, &actions).into_scalar().elem::<f64>()
    }

    fn log_probs(
        &self,
        states: Tensor<TrainingBackend, 2>,
        actions: &A::TensorType,
    ) -> Tensor<TrainingBackend, 1>;

    fn entropy(&self, states: Tensor<TrainingBackend, 2>) -> Tensor<TrainingBackend, 1>;
}
