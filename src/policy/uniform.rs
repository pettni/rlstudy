use super::DiscretePolicy;

use ndarray::Array1;

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
