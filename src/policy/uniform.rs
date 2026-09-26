use super::{DiscretePolicy, Policy};

use ndarray::Array1;
use rand::RngExt;

pub struct UniformPolicy {
    num_actions: usize,
}

impl UniformPolicy {
    pub fn new(num_actions: usize) -> Self {
        Self { num_actions }
    }
}

impl DiscretePolicy for UniformPolicy {
    type Observation = Array1<f64>;

    fn action_probabilities(&self, _observation: &Array1<f64>) -> Array1<f64> {
        Array1::from_elem(self.num_actions, 1.0 / self.num_actions as f64)
    }
}

pub struct UniformContinuousPolicy {
    low: Array1<f64>,
    high: Array1<f64>,
}

impl UniformContinuousPolicy {
    pub fn new(low: Array1<f64>, high: Array1<f64>) -> Self {
        Self { low, high }
    }
}

impl Policy for UniformContinuousPolicy {
    type Observation = Array1<f64>;
    type Action = Array1<f64>;

    fn sample_action(
        &self,
        _observation: &Self::Observation,
        rng: &mut dyn rand::Rng,
    ) -> Self::Action {
        Array1::from_iter(
            self.low
                .iter()
                .zip(&self.high)
                .map(|(&low, high)| rng.random_range(low..*high)),
        )
    }
}
