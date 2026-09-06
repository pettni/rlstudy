use crate::probability;
use crate::system::System;

#[derive(Debug, Clone)]
pub struct State {
    pub name: String,
    pub idx: u64,
}

impl State {
    pub fn new(name: impl Into<String>, idx: u64) -> Self {
        Self {
            name: name.into(),
            idx,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Action {
    pub name: String,
    pub idx: u64,
}

impl Action {
    pub fn new(name: impl Into<String>, idx: u64) -> Self {
        Self {
            name: name.into(),
            idx,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    // all states
    pub states: Vec<State>,
    // all actions
    pub actions: Vec<Action>,
    // p(s'|s,a) for each state-action pair
    pub p_s_sa: Vec<Vec<probability::Distribution<u64>>>,
    // p(r|s,a) for each state-action pair
    pub p_r_sa: Vec<Vec<probability::Distribution<f64>>>,
}

impl Model {
    pub fn new(
        states: Vec<State>,
        actions: Vec<Action>,
        p_s_sa: Vec<Vec<probability::Distribution<u64>>>,
        p_r_sa: Vec<Vec<probability::Distribution<f64>>>,
    ) -> Self {
        Self {
            states,
            actions,
            p_s_sa,
            p_r_sa,
        }
    }
}

impl System for Model {
    fn num_states(&self) -> usize {
        self.states.len()
    }

    fn num_actions(&self) -> usize {
        self.actions.len()
    }

    fn step(&mut self, state: usize, action: usize) -> (usize, f64) {
        let next_state = *self.p_s_sa[state][action].sample() as usize;
        let reward = *self.p_r_sa[state][action].sample();
        (next_state, reward)
    }
}
