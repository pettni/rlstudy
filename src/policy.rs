use crate::environment::Space;
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};

/// A policy maps states to action probabilities.
pub trait Policy {
    type State;
    type Action;

    fn state_space(&self) -> Space;
    fn action_space(&self) -> Space;

    fn sample_action(&self, state: &Self::State, rng: &mut dyn rand::Rng) -> Self::Action;
}

pub enum PolicyObjective {
    /// Policy greedily maximizes q-value
    Deterministic,
    /// Soft version of Deterministic where 'epsilon' mass is assigned uniformly
    SoftDeterministic(f64),
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

    pub fn from_actions(
        num_actions: usize,
        actions: Vec<usize>,
        objective: PolicyObjective,
    ) -> Result<Self, String> {
        let n_a = num_actions;

        let exploration_probability: f64 = match objective {
            PolicyObjective::Deterministic => 0.,
            PolicyObjective::SoftDeterministic(epsilon) => {
                if !epsilon.is_finite() || !(0.0..=1.0).contains(&epsilon) {
                    return Err("epsilon must be finite and in [0, 1]".into());
                }
                epsilon / num_actions as f64
            }
        };

        if actions.iter().any(|action| *action >= num_actions) {
            return Err("policy action is out of bounds".into());
        }

        let probabilities = actions
            .into_iter()
            .map(|action| {
                let mut row = vec![exploration_probability; num_actions];
                row[action] = 1.0 - exploration_probability * (n_a - 1) as f64;
                row
            })
            .collect();
        Self::new(probabilities)
    }

    pub fn from_q(q_sa: &Array2<f64>, objective: PolicyObjective) -> Result<Self, String> {
        let n_a = q_sa.ncols();
        let greedy_actions = q_sa
            .axis_iter(Axis(0))
            .map(|row| {
                let mut best_a = 0;
                let mut best_q = row[0];
                for a in 1..n_a {
                    if row[a] > best_q {
                        best_q = row[a];
                        best_a = a;
                    }
                }
                best_a
            })
            .collect();
        Self::from_actions(n_a, greedy_actions, objective)
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

    pub fn probabilities(&self) -> ArrayView2<'_, f64> {
        self.probabilities.view()
    }

    pub fn state_probabilities(&self, state: usize) -> ArrayView1<'_, f64> {
        self.probabilities.row(state)
    }

    /// Update policy for a single state
    pub fn update_at(&mut self, state: usize, q_a: &ArrayView1<f64>, objective: PolicyObjective) {
        let num_actions = self.num_actions();
        let mut best_a = 0;
        let mut best_q = q_a[0];
        for a in 1..num_actions {
            if q_a[a] > best_q {
                best_q = q_a[a];
                best_a = a;
            }
        }
        let mut probabilities = self.probabilities.row_mut(state);
        match objective {
            PolicyObjective::Deterministic => {
                probabilities.fill(0.0);
                probabilities[best_a] = 1.0;
            }
            PolicyObjective::SoftDeterministic(epsilon) => {
                let exploration_probability = epsilon / num_actions as f64;
                probabilities.fill(exploration_probability);
                probabilities[best_a] = 1.0 - epsilon + exploration_probability;
            }
        }
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
    use ndarray::array;
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
        let policy =
            TabularPolicy::from_actions(3, vec![2, 0], PolicyObjective::Deterministic).unwrap();

        assert_eq!(policy.probability(0, 2), 1.0);
        assert_eq!(policy.probability(1, 0), 1.0);
        assert_eq!(policy.probability(0, 0), 0.0);
    }

    #[test]
    fn soft_deterministic_policy_is_epsilon_greedy() {
        let policy =
            TabularPolicy::from_actions(3, vec![2], PolicyObjective::SoftDeterministic(0.3))
                .unwrap();

        assert!((policy.probability(0, 0) - 0.1).abs() < 1e-12);
        assert!((policy.probability(0, 1) - 0.1).abs() < 1e-12);
        assert!((policy.probability(0, 2) - 0.8).abs() < 1e-12);
        assert!(
            TabularPolicy::from_actions(3, vec![0], PolicyObjective::SoftDeterministic(-0.1))
                .is_err()
        );
        assert!(
            TabularPolicy::from_actions(3, vec![0], PolicyObjective::SoftDeterministic(1.1))
                .is_err()
        );
    }

    #[test]
    fn policy_rejects_invalid_rows() {
        assert!(TabularPolicy::new(vec![]).is_err());
        assert!(TabularPolicy::new(vec![vec![]]).is_err());
        assert!(TabularPolicy::new(vec![vec![1.0], vec![1.0, 1.0]]).is_err());
        assert!(TabularPolicy::new(vec![vec![f64::NAN, 1.0]]).is_err());
        assert!(TabularPolicy::new(vec![vec![0.0, 0.0]]).is_err());
        assert!(TabularPolicy::from_actions(2, vec![2], PolicyObjective::Deterministic).is_err());
    }

    #[test]
    fn update_at_replaces_one_policy_row() {
        let mut policy = TabularPolicy::uniform(2, 3).unwrap();

        policy.update_at(
            1,
            &array![1.0, 3.0, 2.0].view(),
            PolicyObjective::Deterministic,
        );

        assert_eq!(policy.probability(0, 0), 1.0 / 3.0);
        assert_eq!(policy.probability(1, 0), 0.0);
        assert_eq!(policy.probability(1, 1), 1.0);
        assert_eq!(policy.probability(1, 2), 0.0);

        policy.update_at(
            0,
            &array![1.0, 2.0, 4.0].view(),
            PolicyObjective::SoftDeterministic(0.3),
        );

        assert!((policy.probability(0, 0) - 0.1).abs() < 1e-12);
        assert!((policy.probability(0, 1) - 0.1).abs() < 1e-12);
        assert!((policy.probability(0, 2) - 0.8).abs() < 1e-12);
    }
}
