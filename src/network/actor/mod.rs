mod categorical;
mod gaussian;

use crate::{backend::TrainingBackend, network::action::ActionType, policy::Policy};
use burn::{module::AutodiffModule, tensor::Tensor};
use ndarray::Array1;

pub use categorical::CategoricalActor;
pub use gaussian::GaussianActor;

pub trait Actor<A: ActionType>:
    AutodiffModule<TrainingBackend, InnerModule = Self::InferencePolicy>
    + Policy<Observation = Array1<f64>, Action = A>
{
    type InferencePolicy: Policy<Observation = Array1<f64>, Action = A>;

    fn log_probs(
        &self,
        states: Tensor<TrainingBackend, 2>,
        actions: &A::TensorType,
    ) -> Tensor<TrainingBackend, 1>;

    fn entropy(&self, states: Tensor<TrainingBackend, 2>) -> Tensor<TrainingBackend, 1>;
}
