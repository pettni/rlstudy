use ndarray::Array1;

pub mod dq;
pub mod tabular;
pub mod uniform;

pub use tabular::{PolicyObjective, TabularPolicy};

pub struct PolicySample<A> {
    /// Sampled action
    pub action: A,
    /// Log probability log pi(a|s) of sampled action
    pub log_prob: f64,
}

/// A policy maps observations to actions.
pub trait Policy {
    type Observation;
    type Action;

    fn log_prob(&self, observation: &Self::Observation, action: &Self::Action) -> f64;

    fn sample_action(
        &self,
        observation: &Self::Observation,
        rng: &mut dyn rand::Rng,
    ) -> Self::Action {
        self.sample_action_with_log_prob(observation, rng).action
    }

    fn sample_action_with_log_prob(
        &self,
        observation: &Self::Observation,
        rng: &mut dyn rand::Rng,
    ) -> PolicySample<Self::Action>;
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

    fn log_prob(&self, observation: &Self::Observation, action: &Self::Action) -> f64 {
        self.action_probabilities(observation)[*action].ln()
    }

    fn sample_action_with_log_prob(
        &self,
        observation: &Self::Observation,
        rng: &mut dyn rand::Rng,
    ) -> PolicySample<usize> {
        use rand::RngExt;

        let probabilities = &self.action_probabilities(observation);

        let sample: f64 = rng.random();
        let mut cumulative = 0.0;
        for (action, probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if sample < cumulative {
                return PolicySample {
                    action,
                    log_prob: probability.ln(),
                };
            }
        }
        PolicySample {
            action: probabilities.len() - 1,
            log_prob: probabilities.last().expect("probabilities not empty").ln(),
        }
    }
}
