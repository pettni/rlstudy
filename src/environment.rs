//! Common interface for environments used by reinforcement-learning algorithms.

use ndarray::Array1;

#[derive(Debug, Clone, PartialEq)]
pub enum Space {
    Discrete { size: usize },
    Continuous { low: Array1<f64>, high: Array1<f64> },
}

/// A stateful environment that can be reset and stepped.
pub trait Environment {
    type State;
    type Action;

    fn state_space(&self) -> Space;

    fn action_space(&self) -> Space;

    fn reset(&mut self, rng: &mut dyn rand::Rng) -> Self::State;

    /// Take `action` and return the next state and reward.
    fn step(&mut self, action: Self::Action, rng: &mut dyn rand::Rng) -> (Self::State, f64);
}

/// A black-box environment defined by a transition function.
pub struct BlackBox<F> {
    num_states: usize,
    num_actions: usize,
    initial_state: usize,
    state: usize,
    step_fn: F,
}

impl<F> BlackBox<F>
where
    F: FnMut(usize, usize) -> (usize, f64),
{
    /// Create a black-box environment.
    pub fn new(num_states: usize, num_actions: usize, initial_state: usize, step_fn: F) -> Self {
        Self {
            num_states,
            num_actions,
            initial_state,
            state: initial_state,
            step_fn,
        }
    }
}

impl<F> Environment for BlackBox<F>
where
    F: FnMut(usize, usize) -> (usize, f64),
{
    type State = usize;
    type Action = usize;

    fn state_space(&self) -> Space {
        Space::Discrete {
            size: self.num_states,
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete {
            size: self.num_actions,
        }
    }

    fn reset(&mut self, _rng: &mut dyn rand::Rng) -> usize {
        self.state = self.initial_state;
        self.state
    }

    fn step(&mut self, action: usize, _rng: &mut dyn rand::Rng) -> (usize, f64) {
        let (next_state, reward) = (self.step_fn)(self.state, action);
        self.state = next_state;
        (next_state, reward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn black_box_acts_as_environment() {
        let mut bb = BlackBox::new(3, 2, 0, |state: usize, action: usize| {
            let next = (state + action) % 3;
            (next, next as f64)
        });

        assert_eq!(bb.state_space(), Space::Discrete { size: 3 });
        assert_eq!(bb.action_space(), Space::Discrete { size: 2 });
        let mut rng = StdRng::seed_from_u64(0);

        assert_eq!(bb.reset(&mut rng), 0);

        let (s, r) = bb.step(1, &mut rng);
        assert_eq!(s, 1);
        assert_eq!(r, 1.0);

        let (s, r) = bb.step(1, &mut rng);
        assert_eq!(s, 2);
        assert_eq!(r, 2.0);

        assert_eq!(bb.reset(&mut rng), 0);
    }
}
