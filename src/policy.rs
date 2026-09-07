use crate::environment::Space;
use ndarray::Array2;

/// A policy maps states to action probabilities.
pub trait Policy {
    type State;
    type Action;

    fn state_space(&self) -> Space;
    fn action_space(&self) -> Space;

    fn sample_action(&self, state: &Self::State, rng: &mut dyn rand::Rng) -> Self::Action;
}

#[derive(Debug, Clone, PartialEq)]
pub struct TabularPolicy {
    probabilities: Array2<f64>,
}

impl TabularPolicy {
    pub fn new(mut probabilities: Vec<Vec<f64>>) -> Result<Self, String> {
        let num_states = probabilities.len();
        let num_actions = probabilities.first().map_or(0, Vec::len);
        if num_states == 0 || num_actions == 0 {
            return Err("policy must have at least one state and action".into());
        }

        for row in &mut probabilities {
            if row.len() != num_actions {
                return Err("policy rows must have the same number of actions".into());
            }
            if !row.iter().all(|p| p.is_finite() && *p >= 0.0) {
                return Err("policy probabilities must be finite and non-negative".into());
            }
            let sum: f64 = row.iter().sum();
            if sum == 0.0 {
                return Err("each policy row must have a non-zero probability".into());
            }
            row.iter_mut().for_each(|p| *p /= sum);
        }

        Ok(Self {
            probabilities: Array2::from_shape_vec(
                (num_states, num_actions),
                probabilities.into_iter().flatten().collect(),
            )
            .expect("validated policy shape"),
        })
    }

    pub fn uniform(num_states: usize, num_actions: usize) -> Result<Self, String> {
        Self::new(vec![vec![1.0; num_actions]; num_states])
    }

    pub fn deterministic(num_actions: usize, actions: Vec<usize>) -> Result<Self, String> {
        if actions.iter().any(|action| *action >= num_actions) {
            return Err("policy action is out of bounds".into());
        }
        let probabilities = actions
            .into_iter()
            .map(|action| {
                let mut row = vec![0.0; num_actions];
                row[action] = 1.0;
                row
            })
            .collect();
        Self::new(probabilities)
    }

    pub fn num_states(&self) -> usize {
        self.probabilities.nrows()
    }

    pub fn num_actions(&self) -> usize {
        self.probabilities.ncols()
    }

    pub fn probability(&self, state: usize, action: usize) -> f64 {
        self.probabilities[[state, action]]
    }
}

impl Policy for TabularPolicy {
    type State = usize;
    type Action = usize;

    fn state_space(&self) -> Space {
        Space::Discrete {
            size: self.num_states(),
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete {
            size: self.num_actions(),
        }
    }

    fn sample_action(&self, state: &usize, rng: &mut dyn rand::Rng) -> usize {
        use rand::RngExt;

        let u: f64 = rng.random();
        let mut cumulative = 0.0;
        for action in 0..self.num_actions() {
            cumulative += self.probability(*state, action);
            if u < cumulative {
                return action;
            }
        }
        self.num_actions() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn uniform_policy_samples_valid_actions() {
        let policy = TabularPolicy::uniform(3, 5).unwrap();
        let mut rng = StdRng::seed_from_u64(0);

        assert_eq!(policy.state_space(), Space::Discrete { size: 3 });
        assert_eq!(policy.action_space(), Space::Discrete { size: 5 });
        for state in 0..3 {
            for _ in 0..100 {
                assert!(policy.sample_action(&state, &mut rng) < 5);
            }
        }
    }

    #[test]
    fn policy_normalizes_probabilities() {
        let policy = TabularPolicy::new(vec![vec![1.0, 2.0, 1.0]]).unwrap();

        assert_eq!(policy.probability(0, 0), 0.25);
        assert_eq!(policy.probability(0, 1), 0.5);
        assert_eq!(policy.probability(0, 2), 0.25);
    }

    #[test]
    fn deterministic_policy_uses_one_hot_rows() {
        let policy = TabularPolicy::deterministic(3, vec![2, 0]).unwrap();

        assert_eq!(policy.probability(0, 2), 1.0);
        assert_eq!(policy.probability(1, 0), 1.0);
        assert_eq!(policy.probability(0, 0), 0.0);
    }

    #[test]
    fn policy_rejects_invalid_rows() {
        assert!(TabularPolicy::new(vec![]).is_err());
        assert!(TabularPolicy::new(vec![vec![]]).is_err());
        assert!(TabularPolicy::new(vec![vec![1.0], vec![1.0, 1.0]]).is_err());
        assert!(TabularPolicy::new(vec![vec![f64::NAN, 1.0]]).is_err());
        assert!(TabularPolicy::new(vec![vec![0.0, 0.0]]).is_err());
        assert!(TabularPolicy::deterministic(2, vec![2]).is_err());
    }
}
