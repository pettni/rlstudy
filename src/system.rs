//! Common interface for systems that can be stepped by reinforcement-learning algorithms.

/// A system that can be queried and stepped.
pub trait System {
    /// Number of states in the system.
    fn num_states(&self) -> usize;

    /// Number of actions in the system.
    fn num_actions(&self) -> usize;

    /// Take `action` from `state` and return the next state and reward.
    fn step(&mut self, state: usize, action: usize) -> (usize, f64);
}

/// A black-box system defined by a transition function.
pub struct BlackBox<F> {
    num_states: usize,
    num_actions: usize,
    step_fn: F,
}

impl<F> BlackBox<F>
where
    F: FnMut(usize, usize) -> (usize, f64),
{
    /// Create a black-box system.
    pub fn new(num_states: usize, num_actions: usize, step_fn: F) -> Self {
        Self {
            num_states,
            num_actions,
            step_fn,
        }
    }
}

impl<F> System for BlackBox<F>
where
    F: FnMut(usize, usize) -> (usize, f64),
{
    fn num_states(&self) -> usize {
        self.num_states
    }

    fn num_actions(&self) -> usize {
        self.num_actions
    }

    fn step(&mut self, state: usize, action: usize) -> (usize, f64) {
        (self.step_fn)(state, action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examples;

    /// Helper that exercises a system without caring whether it is a Model or a
    /// BlackBox.
    fn take_step(sys: &mut dyn System, state: usize, action: usize) -> (usize, f64) {
        sys.step(state, action)
    }

    #[test]
    fn grid_world_acts_as_system() {
        let mut model = examples::grid_world(3);
        assert_eq!(model.num_states(), 9);
        assert_eq!(model.num_actions(), 5);

        // "right" action from state 0 -> state 1 (reward 0 in this custom scheme).
        let (s1, r1) = take_step(&mut model, 0, 1);
        assert_eq!(s1, 1);
        assert_eq!(r1, 0.0);

        // "down" action from state 1 -> state 4 (reward 0 in this custom scheme).
        let (s2, r2) = take_step(&mut model, 1, 2);
        assert_eq!(s2, 4);
        assert_eq!(r2, 0.0);

        // "stay" in goal state 8 -> state 8 with reward +1.0.
        let (s3, r3) = take_step(&mut model, 8, 4);
        assert_eq!(s3, 8);
        assert_eq!(r3, 1.0);
    }

    #[test]
    fn black_box_acts_as_system() {
        let mut bb = BlackBox::new(3, 2, |state: usize, action: usize| {
            let next = (state + action) % 3;
            (next, next as f64)
        });

        assert_eq!(bb.num_states(), 3);
        assert_eq!(bb.num_actions(), 2);

        let (s, r) = take_step(&mut bb, 1, 1);
        assert_eq!(s, 2);
        assert_eq!(r, 2.0);
    }
}
