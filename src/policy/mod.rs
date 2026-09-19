use ndarray::Array1;

pub mod dq;
pub mod tabular;
pub mod uniform;

pub use tabular::{PolicyObjective, TabularPolicy};

/// A policy maps observations to actions.
pub trait Policy {
    type Observation;
    type Action;

    fn sample_action(
        &self,
        observation: &Self::Observation,
        rng: &mut dyn rand::Rng,
    ) -> Self::Action;
}

pub trait DiscretePolicy {
    type Observation;

    fn action_probabilities(&self, observation: &Self::Observation) -> Array1<f64>;

    fn greedy_action(&self, observation: &Self::Observation) -> usize {
        self.action_probabilities(observation)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(action, _)| action)
            .expect("policy has actions")
    }

    fn epsilon_greedy_action(
        &self,
        observation: &Self::Observation,
        eps: f64,
        rng: &mut dyn rand::Rng,
    ) -> usize {
        use rand::RngExt;

        let probabilities = self.action_probabilities(observation);

        let sample: f64 = rng.random();
        if sample < eps {
            // return random action
            return rng.random_range(0..probabilities.len());
        }

        // return sampled action
        self.sample_action(observation, rng)
    }
}

impl<T> Policy for T
where
    T: DiscretePolicy + ?Sized,
{
    type Observation = T::Observation;
    type Action = usize;

    fn sample_action(&self, observation: &Self::Observation, rng: &mut dyn rand::Rng) -> usize {
        use rand::RngExt;

        let probabilities = &self.action_probabilities(observation);

        let sample: f64 = rng.random();
        let mut cumulative = 0.0;
        for (action, probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if sample < cumulative {
                return action;
            }
        }
        probabilities.len() - 1
    }
}
