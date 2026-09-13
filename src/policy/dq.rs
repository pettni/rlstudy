use burn::Tensor;
use burn::backend::{Flex, flex::FlexDevice};
use burn::tensor::TensorData;
use ndarray::Array1;

use super::DiscretePolicy;
use crate::network::mlp::MlpModel;

#[derive(Debug, Clone, Copy)]
pub enum DeepQPolicyMode {
    Greedy,
    SoftGreedy { epsilon: f64 },
    SoftMax,
}

#[derive(Debug, Clone)]
pub struct DeepQPolicy {
    q_network: MlpModel<Flex>, // f(s) s.t. q(s, a) = f(s)[a]
    mode: DeepQPolicyMode,
}

impl DeepQPolicy {
    pub fn new(q_network: MlpModel<Flex>, mode: DeepQPolicyMode) -> Self {
        Self { q_network, mode }
    }
}

impl DiscretePolicy for DeepQPolicy {
    type Observation = Array1<f64>;

    fn action_probabilities(&self, observation: &Array1<f64>) -> Array1<f64> {
        let device = FlexDevice;
        let observation = Tensor::<Flex, 2>::from_data(
            TensorData::new(observation.to_vec(), [1, observation.len()]),
            &device,
        );
        let q_a: Array1<f64> = Array1::from_iter(
            self.q_network
                .forward(observation)
                .squeeze_dim::<1>(0)
                .into_data()
                .iter::<f64>(),
        );
        match self.mode {
            DeepQPolicyMode::Greedy => {
                let mut probabilities = Array1::zeros(q_a.len());
                probabilities[greedy_action(&q_a)] = 1.0;
                probabilities
            }
            DeepQPolicyMode::SoftGreedy { epsilon } => {
                let exploration = epsilon / q_a.len() as f64;
                let mut probabilities = Array1::from_elem(q_a.len(), exploration);
                probabilities[greedy_action(&q_a)] += 1.0 - epsilon;
                probabilities
            }
            DeepQPolicyMode::SoftMax => {
                let max_q = q_a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = q_a.mapv(|q| (q - max_q).exp());
                &weights / weights.sum()
            }
        }
    }
}

fn greedy_action(q_a: &Array1<f64>) -> usize {
    q_a.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(action, _)| action)
        .expect("policy has actions")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_probabilities_are_a_distribution() {
        let device = FlexDevice;
        let policy = DeepQPolicy::new(MlpModel::new(2, 4, 3, &device), DeepQPolicyMode::SoftMax);

        let probabilities = policy.action_probabilities(&Array1::from_vec(vec![1.0, 2.0]));

        assert_eq!(probabilities.len(), 3);
        assert!((probabilities.sum() - 1.0).abs() < 1e-6);
        assert!(probabilities.iter().all(|probability| *probability >= 0.0));
    }

    #[test]
    fn greedy_modes_are_distributions() {
        let device = FlexDevice;
        let network = MlpModel::new(2, 4, 3, &device);
        let observation = Array1::from_vec(vec![1.0, 2.0]);
        let greedy = DeepQPolicy::new(network.clone(), DeepQPolicyMode::Greedy)
            .action_probabilities(&observation);
        let soft = DeepQPolicy::new(network, DeepQPolicyMode::SoftGreedy { epsilon: 0.3 })
            .action_probabilities(&observation);

        assert_eq!(
            greedy
                .iter()
                .filter(|probability| **probability == 1.0)
                .count(),
            1
        );
        assert_eq!(greedy.sum(), 1.0);
        assert!((soft.iter().copied().fold(0.0, f64::max) - 0.8).abs() < 1e-6);
        assert!((soft.sum() - 1.0).abs() < 1e-6);
    }
}
