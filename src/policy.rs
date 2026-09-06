use crate::probability::Distribution;

pub trait Policy {
    /// Determine the action
    fn select_action(&self, state: usize) -> usize;

    /// Get action probability distribution
    fn prob(&self, state: usize, action: usize) -> f64;
}

/// Deterministic policy

#[derive(Debug, Clone)]
pub struct DeterministicPolicy {
    a_s: Vec<usize>,
}

impl DeterministicPolicy {
    pub fn new(a_s: Vec<usize>) -> Self {
        Self { a_s }
    }
}

impl Policy for DeterministicPolicy {
    fn select_action(&self, state: usize) -> usize {
        self.a_s[state]
    }

    fn prob(&self, state: usize, action: usize) -> f64 {
        if self.a_s[state] == action { 1. } else { 0. }
    }
}

/// Stochastic policy

#[derive(Debug, Clone)]
pub struct StochasticPolicy {
    p_a_s: Vec<Distribution<usize>>,
}

impl StochasticPolicy {
    pub fn new(p_a_s: Vec<Distribution<usize>>) -> Self {
        Self { p_a_s }
    }

    pub fn uniform(num_states: usize, num_actions: usize) -> Self {
        let actions = (0..num_actions).collect::<Vec<_>>();
        let p_a_s = vec![Distribution::uniform(actions); num_states];
        Self { p_a_s }
    }
}

impl Policy for StochasticPolicy {
    fn select_action(&self, state: usize) -> usize {
        *self.p_a_s[state].sample()
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

        for state in 0..num_states {
            for _ in 0..100 {
                let action = policy.select_action(state);
                assert!(action < num_actions);
            }
        }
    }
}
