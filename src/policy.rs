use crate::probability::Distribution;

/// A policy maps states to action probabilities.
pub trait Policy {
    fn num_states(&self) -> usize;
    fn num_actions(&self) -> usize;

    /// Probability of taking `action` in `state`.
    fn prob(&self, state: usize, action: usize) -> f64;
}

/// Sample an action from the policy for a given state.
pub fn sample_action(policy: &dyn Policy, state: usize) -> usize {
    let data: Vec<(f32, usize)> = (0..policy.num_actions())
        .map(|a| (policy.prob(state, a) as f32, a))
        .collect();
    let dist = Distribution::new(data);
    *dist.sample()
}

/// Deterministic policy
#[derive(Debug, Clone)]
pub struct DeterministicPolicy {
    n_actions: usize,
    a_s: Vec<usize>,
}

impl DeterministicPolicy {
    pub fn new(n_actions: usize, a_s: Vec<usize>) -> Self {
        Self { n_actions, a_s }
    }

    pub fn select_action(&self, state: usize) -> usize {
        self.a_s[state]
    }
}

impl Policy for DeterministicPolicy {
    fn num_states(&self) -> usize {
        self.a_s.len()
    }

    fn num_actions(&self) -> usize {
        self.n_actions
    }

    fn prob(&self, state: usize, action: usize) -> f64 {
        if self.a_s[state] == action { 1.0 } else { 0.0 }
    }
}

/// Stochastic policy
#[derive(Debug, Clone)]
pub struct StochasticPolicy {
    n_actions: usize,
    p_a_s: Vec<Distribution<usize>>,
}

impl StochasticPolicy {
    pub fn new(n_actions: usize, p_a_s: Vec<Distribution<usize>>) -> Self {
        Self { n_actions, p_a_s }
    }

    pub fn uniform(num_states: usize, num_actions: usize) -> Self {
        let actions = (0..num_actions).collect::<Vec<_>>();
        let p_a_s = vec![Distribution::uniform(actions); num_states];
        Self {
            n_actions: num_actions,
            p_a_s,
        }
    }

    pub fn select_action(&self, state: usize) -> usize {
        *self.p_a_s[state].sample()
    }
}

impl Policy for StochasticPolicy {
    fn num_states(&self) -> usize {
        self.p_a_s.len()
    }

    fn num_actions(&self) -> usize {
        self.n_actions
    }

    fn prob(&self, state: usize, action: usize) -> f64 {
        self.p_a_s[state][action].0 as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stochastic_uniform_sample() {
        let num_states = 3;
        let num_actions = 5;
        let policy = StochasticPolicy::uniform(num_states, num_actions);

        assert_eq!(policy.num_states(), num_states);
        assert_eq!(policy.num_actions(), num_actions);

        for state in 0..num_states {
            for _ in 0..100 {
                let action = policy.select_action(state);
                assert!(action < num_actions);
            }
        }
    }
}
