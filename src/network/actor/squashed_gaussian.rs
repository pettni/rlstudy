use burn::{
    module::Module,
    tensor::{Tensor, TensorData, backend::Backend},
};
use ndarray::Array1;

use super::Actor;
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

impl<B: Backend<Device = Device>> SquashedGaussianActor<B> {
    fn deterministic_actions_impl(&self, observations: Tensor<B, 2>) -> Tensor<B, 2> {
        let (mu, _) = self.parameters(observations);
        self.a.clone() + self.b.clone() * mu.tanh()
    }

    fn sample_actions_impl(
        &self,
        observations: Tensor<B, 2>,
        _rng: &mut dyn rand::Rng,
    ) -> Tensor<B, 2> {
        let (mu, log_std) = self.parameters(observations);
        let noise = mu.random_like(burn::tensor::Distribution::Normal(0., 1.));
        let normalized_actions = (mu + noise * log_std.exp()).tanh(); // [-1, 1]
        self.a.clone() + self.b.clone() * normalized_actions
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

        // x \sim N(mu, std)
        // y = a + b tanh (x)
        //
        // p_Y(y) = p_X(x) / (b * (1 - tanh^2(x)))

        let y = actions;

        let tanh_x = ((y.clone() - self.a.clone()) / self.b.clone()).clamp(-1. + 1e-6, 1. - 1e-6);
        let x = tanh_x.clone().atanh();
        let standardized = (x - mu) / log_std.clone().exp();

        let log_two_pi = (2.0 * std::f64::consts::PI).ln();

        let normal_log_probs = -0.5 * (standardized.square() + log_two_pi) - log_std;

        let tanh_jacobian = tanh_x.ones_like() - tanh_x.square();

        let log_probs: Tensor<TrainingBackend, 2> =
            normal_log_probs - self.b.clone().log() - tanh_jacobian.log();

        log_probs.sum_dim(1).squeeze_dim(1)
    }

    fn sample_actions(
        &self,
        observations: Tensor<TrainingBackend, 2>,
        rng: &mut dyn rand::Rng,
    ) -> Tensor<TrainingBackend, 2> {
        self.sample_actions_impl(observations, rng)
    }

    fn entropy(&self, states: Tensor<TrainingBackend, 2>) -> Tensor<TrainingBackend, 1> {
        let (_, log_std) = self.parameters(states);
        let log_two_pi = (2.0 * std::f64::consts::PI).ln();

        (log_std + 0.5 * (1.0 + log_two_pi))
            .sum_dim(1)
            .squeeze_dim(1)
    }
}

impl Policy for SquashedGaussianActor<TrainingBackend> {
    type Observation = Array1<f64>;
    type Action = Array1<f64>;

    fn sample_action(&self, state: &Array1<f64>, rng: &mut dyn rand::Rng) -> Self::Action {
        let state = Tensor::<TrainingBackend, 2>::from_data(
            TensorData::new(state.to_vec(), [1, state.len()]),
            device(),
        );
        let action = self.sample_actions_impl(state, rng);
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
        let action = self.deterministic_actions_impl(state);
        Array1::from_iter(action.into_data().iter())
    }
}
