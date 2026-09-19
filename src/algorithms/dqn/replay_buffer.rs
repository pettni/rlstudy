use rand;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct Transition<State, Action> {
    pub state: State,
    pub action: Action,
    pub reward: f64,
    pub next_state: State,
    pub terminated: bool,
}

pub struct ReplayBuffer<State, Action> {
    transitions: VecDeque<Transition<State, Action>>,
    capacity: usize,
}

impl<State, Action> ReplayBuffer<State, Action> {
    pub fn new(capacity: usize) -> Self {
        Self {
            transitions: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, transition: Transition<State, Action>) {
        if self.transitions.len() == self.capacity {
            self.transitions.pop_front();
        }
        self.transitions.push_back(transition)
    }

    pub fn sample(
        &self,
        batch_size: usize,
        rng: &mut dyn rand::Rng,
    ) -> Vec<Transition<State, Action>>
    where
        State: Clone,
        Action: Clone,
    {
        assert!(
            batch_size <= self.transitions.len(),
            "Not enough data for batch size {}",
            batch_size
        );

        rand::seq::index::sample(rng, self.transitions.len(), batch_size)
            .iter()
            .map(|index| self.transitions[index].clone())
            .collect()
    }
}
