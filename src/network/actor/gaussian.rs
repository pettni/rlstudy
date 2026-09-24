use burn::{
    module::{Module, Param},
    tensor::{Tensor, backend::Backend},
};
use ndarray::Array1;
use rand_distr::{Distribution, StandardNormal};

use super::Actor;
use crate::backend::{Device, InferenceBackend, TrainingBackend, device};
use crate::network::mlp::{Activation, MlpModel};
use crate::policy::{Policy, PolicySample};

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
            mu_mlp: MlpModel::new_with_hidden_layers(
                state_dim,
                hidden_dims,
                action_dim,
                activation,
                device,
            ),
            log_std: Param::from_tensor(Tensor::full([action_dim], -0.5, device)),
        }
    }

    pub fn forward(&self, states: Tensor<B, 2>) -> Tensor<B, 3> {
        let mu = self.mu_mlp.forward(states);
        let [batch_size, action_dim] = mu.dims();
        let log_std = self
            .log_std
            .val()
            .unsqueeze_dim::<2>(0)
            .expand([batch_size, action_dim]);

        Tensor::stack(vec![mu, log_std], 2)
    }

    fn parameters(&self, states: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let parameters = self.forward(states); // (B, N_a, 2)
        let b = parameters.shape()[0];
        let n_a = parameters.shape()[1];
        let mu = parameters
            .clone()
            .slice([0..b, 0..n_a, 0..1])
            .squeeze_dim::<2>(2);
        let log_std = parameters.slice([0..b, 0..n_a, 1..2]).squeeze_dim::<2>(2);

        (mu, log_std)
    }
}

impl<B> GaussianActor<B>
where
    B: Backend<Device = Device>,
{
    fn parameters_for_observation(&self, observation: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let observation = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(observation.to_vec(), [1, observation.len()]),
            device(),
        );
        let (mu, log_std) = self.parameters(observation);

        (
            Array1::from_iter(mu.into_data().iter::<f64>()),
            Array1::from_iter(log_std.into_data().iter::<f64>()),
        )
    }
}

impl<B> Policy for GaussianActor<B>
where
    B: Backend<Device = Device>,
{
    type Observation = Array1<f64>;
    type Action = Array1<f64>;

    fn log_prob(&self, observation: &Self::Observation, action: &Self::Action) -> f64 {
        let (mu, log_std) = self.parameters_for_observation(observation);
        let log_two_pi = (2.0 * std::f64::consts::PI).ln();

        action
            .iter()
            .zip(mu.iter().zip(&log_std))
            .map(|(&action, (&mu, &log_std))| {
                let normalized = (action - mu) / log_std.exp();
                -0.5 * (normalized * normalized + log_two_pi) - log_std
            })
            .sum()
    }

    fn sample_action_with_log_prob(
        &self,
        observation: &Self::Observation,
        rng: &mut dyn rand::Rng,
    ) -> PolicySample<Self::Action> {
        let (mu, log_std) = self.parameters_for_observation(observation);
        let log_two_pi = (2.0 * std::f64::consts::PI).ln();
        let mut log_prob = 0.0;
        let action = Array1::from_iter(mu.iter().zip(&log_std).map(|(&mu, &log_std)| {
            let noise: f64 = StandardNormal.sample(rng);
            log_prob += -0.5 * (noise * noise + log_two_pi) - log_std;
            mu + noise * log_std.exp()
        }));

        PolicySample { action, log_prob }
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
        let log_two_pi = (2.0 * std::f64::consts::PI).ln();

        let log_probs: Tensor<TrainingBackend, 2> =
            -0.5 * (normalized.square() + log_two_pi) - log_std;

        log_probs.sum_dim(1).squeeze_dim(1)
    }

    fn entropy(&self, states: Tensor<TrainingBackend, 2>) -> Tensor<TrainingBackend, 1> {
        let (_, log_std) = self.parameters(states);
        let log_two_pi = (2.0 * std::f64::consts::PI).ln();

        (log_std + 0.5 * (1.0 + log_two_pi))
            .sum_dim(1)
            .squeeze_dim(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::InferenceBackend;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn returns_mean_and_log_standard_deviation_per_action() {
        let actor = GaussianActor::<InferenceBackend>::new(3, &[4], 2, Activation::Tanh, device());

        let parameters = actor.forward(Tensor::zeros([5, 3], device()));

        assert_eq!(parameters.dims(), [5, 2, 2]);
        assert!(
            parameters
                .slice([0..5, 0..2, 1..2])
                .into_data()
                .iter::<f64>()
                .all(|value| value == -0.5)
        );
    }

    #[test]
    fn sampled_and_evaluated_log_probabilities_match() {
        let actor = GaussianActor::<InferenceBackend>::new(1, &[2], 2, Activation::Tanh, device());
        let observation = Array1::zeros(1);
        let mut rng = StdRng::seed_from_u64(0);

        let sample = actor.sample_action_with_log_prob(&observation, &mut rng);

        assert!((sample.log_prob - actor.log_prob(&observation, &sample.action)).abs() < 1e-12);
    }
}
