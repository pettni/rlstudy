mod categorical;
mod gaussian;
mod squashed_gaussian;

use crate::{
    backend::TrainingBackend,
    network::{TensorValue, state_tensor},
    policy::Policy,
};
use burn::tensor::ElementConversion;
use burn::{module::AutodiffModule, tensor::Tensor};
use ndarray::Array1;

pub use categorical::CategoricalActor;
pub use gaussian::GaussianActor;
pub use squashed_gaussian::SquashedGaussianActor;

pub(super) const LOG_TWO_PI: f64 = 1.8378770664093453;

pub trait Actor<A: TensorValue>:
    AutodiffModule<TrainingBackend, InnerModule = Self::InferencePolicy>
    + Policy<Observation = Array1<f64>, Action = A>
{
    type InferencePolicy: Policy<Observation = Array1<f64>, Action = A>;

    fn log_prob(&self, state: &Array1<f64>, action: &A) -> f64 {
        let states = state_tensor::<TrainingBackend>(state);
        let actions = A::singleton_tensor(action);

        self.log_probs(states, &actions).into_scalar().elem::<f64>()
    }

    fn sample_actions(
        &self,
        states: Tensor<TrainingBackend, 2>,
    ) -> (A::BatchTensor, Tensor<TrainingBackend, 1>);

    fn log_probs(
        &self,
        states: Tensor<TrainingBackend, 2>,
        actions: &A::BatchTensor,
    ) -> Tensor<TrainingBackend, 1>;

    fn entropy(&self, states: Tensor<TrainingBackend, 2>) -> Tensor<TrainingBackend, 1>;
}
