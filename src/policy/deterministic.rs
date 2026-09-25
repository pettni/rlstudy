use rand_distr::{Distribution, StandardNormal};
use std::borrow::Cow;

use burn::tensor::{Tensor, backend::Backend};
use ndarray::Array1;

use crate::{
    backend::{Device, device},
    network::mlp::MlpModel,
    policy::Policy,
};

#[derive(Debug, Clone)]
pub struct DeterministicPolicy<'a, B: Backend> {
    pi_network: Cow<'a, MlpModel<B>>, // f: s -> a
    action_low: Cow<'a, Array1<f64>>,
    action_high: Cow<'a, Array1<f64>>,
    noise_std: Option<f64>,
}

impl<B: Backend> DeterministicPolicy<'static, B> {
    pub fn new(
        pi_network: MlpModel<B>,
        action_low: Array1<f64>,
        action_high: Array1<f64>,
        noise_std: Option<f64>,
    ) -> Self {
        Self {
            pi_network: Cow::Owned(pi_network),
            action_low: Cow::Owned(action_low),
            action_high: Cow::Owned(action_high),
            noise_std,
        }
    }
}

impl<'a, B: Backend> DeterministicPolicy<'a, B> {
    pub fn borrowed(
        pi_network: &'a MlpModel<B>,
        action_low: &'a Array1<f64>,
        action_high: &'a Array1<f64>,
        noise_std: Option<f64>,
    ) -> Self {
        Self {
            pi_network: Cow::Borrowed(pi_network),
            action_low: Cow::Borrowed(action_low),
            action_high: Cow::Borrowed(action_high),
            noise_std,
        }
    }
}

impl<B> Policy for DeterministicPolicy<'_, B>
where
    B: Backend<Device = Device>,
{
    type Observation = Array1<f64>;
    type Action = Array1<f64>;

    fn sample_action(
        &self,
        observation: &Self::Observation,
        rng: &mut dyn rand::Rng,
    ) -> Self::Action {
        let observation = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(observation.to_vec(), [1, observation.len()]),
            device(),
        );

        let actions = self.pi_network.forward(observation);

        let mut actions = Array1::from_iter(actions.into_data().iter::<f64>());

        for ((action, &low), &high) in actions
            .iter_mut()
            .zip(self.action_low.iter())
            .zip(self.action_high.iter())
        {
            if let Some(noise_std) = self.noise_std {
                let noise: f64 = StandardNormal.sample(rng);
                *action += noise_std * noise;
            }
            *action = action.clamp(low, high);
        }

        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::InferenceBackend;
    use crate::network::mlp::OutputTransform;
    use burn::module::Reinitializer;
    use ndarray::array;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn scales_and_clips_actions_to_bounds() {
        let model = Reinitializer::<InferenceBackend>::new()
            .constant_float(0.0)
            .apply(MlpModel::new(1, 2, 2, device()))
            .with_output_transform(OutputTransform::bounded_tanh(
                &[-2.0, 1.0],
                &[2.0, 3.0],
                device(),
            ));
        let low = array![-2.0, 1.0];
        let high = array![2.0, 3.0];
        let mut rng = StdRng::seed_from_u64(0);

        let policy = DeterministicPolicy::borrowed(&model, &low, &high, None);
        let action = policy.sample_action(&array![0.0], &mut rng);
        assert_eq!(action, array![0.0, 2.0]);

        let noisy_policy = DeterministicPolicy::borrowed(&model, &low, &high, Some(100.0));
        for _ in 0..10 {
            let action = noisy_policy.sample_action(&array![0.0], &mut rng);
            assert!(
                action
                    .iter()
                    .zip(&low)
                    .zip(&high)
                    .all(|((&action, &low), &high)| (low..=high).contains(&action))
            );
        }
    }
}
