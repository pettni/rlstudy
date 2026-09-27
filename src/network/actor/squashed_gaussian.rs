use burn::{
    module::Module,
    tensor::{Tensor, TensorData, activation::softplus, backend::Backend},
};
use ndarray::Array1;

use super::{Actor, LOG_TWO_PI};
use crate::network::{Activation, MlpModel};
use crate::policy::Policy;
use crate::{
    backend::{Device, InferenceBackend, TrainingBackend, device},
    environment::Space,
};

#[derive(Module, Debug)]
pub struct SquashedGaussianActor<B: Backend> {
    mlp: MlpModel<B>,
    a: Tensor<B, 2>,
    b: Tensor<B, 2>,
}

impl<B: Backend> SquashedGaussianActor<B> {
    pub fn new(
        state_dim: usize,
        hidden_dims: &[usize],
        action_space: Space,
        activation: Activation,
        device: &B::Device,
    ) -> Self {
        let (low, high) = action_space
            .into_continuous_bounds()
            .expect("SquashedGaussianActor requires a continuous action space");

        let action_dim = low.len();

        let lo: Vec<f64> = low.iter().map(|x| *x).collect::<Vec<_>>();
        let hi: Vec<f64> = high.iter().map(|x| *x).collect::<Vec<_>>();

        let lo = Tensor::<B, 1>::from_data(TensorData::new(lo, [action_dim]), device);
        let hi = Tensor::<B, 1>::from_data(TensorData::new(hi, [action_dim]), device);
        Self {
            mlp: MlpModel::new(state_dim, hidden_dims, action_dim * 2, activation, device),
            a: (hi.clone() + lo.clone()).unsqueeze_dim::<2>(0) / 2.,
            b: (hi - lo).unsqueeze_dim::<2>(0) / 2.,
        }
    }

    fn parameters(&self, states: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let output = self.mlp.forward(states); // (B, N_a * 2)
        let [batch_size, output_dim] = output.dims();

        let action_dim = output_dim / 2;

        let mu = output.clone().slice([0..batch_size, 0..action_dim]);
        let log_std = output
            .clone()
            .slice([0..batch_size, action_dim..2 * action_dim])
            .clamp(-20.0, 2.);

        (mu, log_std)
    }
}

/// The action is y = a + b tanh(mu + eta * std), where eta ~ N(0, 1)
///
/// The distribution of x = mu + eta * std is N(mu, std)
///
/// The pdf of y p_Y(y) for y = a + b tanh(x) is via a change of variables
///
///   p_Y(y) = p_X(x) / (b * (1 - tanh^2(x)))
///
/// Thus, the log-density is
///
/// log p_Y(y) = log p_X(x) - log(b) - log(1 - tanh^2(x))
///
impl<B: Backend<Device = Device>> SquashedGaussianActor<B> {
    fn log_probs_helper(
        &self,
        eta: Tensor<B, 2>,
        x: Tensor<B, 2>,
        log_std: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let normal_log_probs = -0.5 * (eta.square() + LOG_TWO_PI) - log_std;
        let log_tanh_jacobian =
            (x.clone() + softplus(x * -2.0, 1.0)) * -2.0 + 2.0 * std::f64::consts::LN_2;
        let log_probs: Tensor<B, 2> = normal_log_probs - self.b.clone().log() - log_tanh_jacobian;

        log_probs.sum_dim(1).squeeze_dim(1)
    }
}

impl Actor<Array1<f64>> for SquashedGaussianActor<TrainingBackend> {
    type InferencePolicy = SquashedGaussianActor<InferenceBackend>;

    fn log_probs(
        &self,
        states: Tensor<TrainingBackend, 2>,
        actions: &Tensor<TrainingBackend, 2>,
    ) -> Tensor<TrainingBackend, 1> {
        let (mu, log_std) = self.parameters(states);

        let tanh_x =
            ((actions.clone() - self.a.clone()) / self.b.clone()).clamp(-1. + 1e-6, 1. - 1e-6);
        let x = tanh_x.atanh();
        let standardized = (x.clone() - mu) / log_std.clone().exp();

        self.log_probs_helper(standardized, x, log_std)
    }

    fn sample_actions(
        &self,
        observations: Tensor<TrainingBackend, 2>,
    ) -> (Tensor<TrainingBackend, 2>, Tensor<TrainingBackend, 1>) {
        let (mu, log_std) = self.parameters(observations);
        let eta = mu.random_like(burn::tensor::Distribution::Normal(0., 1.));
        let x = mu.clone() + eta.clone() * log_std.clone().exp();

        let actions = self.a.clone() + self.b.clone() * x.clone().tanh();
        let log_probs = self.log_probs_helper(eta, x, log_std);

        (actions, log_probs)
    }

    fn entropy(&self, states: Tensor<TrainingBackend, 2>) -> Tensor<TrainingBackend, 1> {
        let (_, log_std) = self.parameters(states);

        (log_std + 0.5 * (1.0 + LOG_TWO_PI))
            .sum_dim(1)
            .squeeze_dim(1)
    }
}

impl Policy for SquashedGaussianActor<TrainingBackend> {
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

impl Policy for SquashedGaussianActor<InferenceBackend> {
    type Observation = Array1<f64>;
    type Action = Array1<f64>;

    fn sample_action(&self, state: &Array1<f64>, _rng: &mut dyn rand::Rng) -> Self::Action {
        let state = Tensor::<InferenceBackend, 2>::from_data(
            TensorData::new(state.to_vec(), [1, state.len()]),
            device(),
        );
        let (mu, _) = self.parameters(state);
        let action = self.a.clone() + self.b.clone() * mu.tanh();
        Array1::from_iter(action.into_data().iter())
    }
}
