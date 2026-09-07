use crate::probability;
use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct State {
    pub name: String,
}

impl State {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[derive(Debug, Clone)]
pub struct Action {
    pub name: String,
}

impl Action {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[derive(Debug, Clone)]
pub struct TabularModel {
    // all states
    states: Vec<State>,
    // all actions
    actions: Vec<Action>,
    // p(s'|s,a) for each state-action pair
    transitions: Vec<probability::Distribution<usize>>,
    // r(s,a) for each state-action pair
    rewards: Array2<f64>,
}

impl TabularModel {
    /// Construct a model whose rewards are expected rewards `r(s, a)`.
    pub fn new(
        states: Vec<State>,
        actions: Vec<Action>,
        p_s_sa: Vec<Vec<probability::Distribution<usize>>>,
        r_sa: Vec<Vec<f64>>,
    ) -> Result<Self, String> {
        let num_states = states.len();
        let num_actions = actions.len();

        if num_states == 0 {
            return Err("model must have at least one state".into());
        }
        if num_actions == 0 {
            return Err("model must have at least one action".into());
        }
        if p_s_sa.len() != num_states {
            return Err(format!(
                "expected {num_states} transition rows, found {}",
                p_s_sa.len()
            ));
        }
        if r_sa.len() != num_states {
            return Err(format!(
                "expected {num_states} reward rows, found {}",
                r_sa.len()
            ));
        }

        for state in 0..num_states {
            if p_s_sa[state].len() != num_actions {
                return Err(format!(
                    "expected {num_actions} transitions for state {state}, found {}",
                    p_s_sa[state].len()
                ));
            }
            if r_sa[state].len() != num_actions {
                return Err(format!(
                    "expected {num_actions} rewards for state {state}, found {}",
                    r_sa[state].len()
                ));
            }

            for (action, transition) in p_s_sa[state].iter().enumerate() {
                for (_, next_state) in transition.iter() {
                    if *next_state >= num_states {
                        return Err(format!(
                            "transition for state {state}, action {action} targets state {next_state}, but the model has {num_states} states"
                        ));
                    }
                }
            }
        }

        let transitions = p_s_sa.into_iter().flatten().collect();
        let rewards = Array2::from_shape_vec(
            (num_states, num_actions),
            r_sa.into_iter().flatten().collect(),
        )
        .expect("validated reward shape");

        Ok(Self {
            states,
            actions,
            transitions,
            rewards,
        })
    }

    /// Construct a model from `r(s, a, s')` and reduce it to expected `r(s, a)`.
    pub fn from_transition_rewards(
        states: Vec<State>,
        actions: Vec<Action>,
        transitions: Vec<Vec<probability::Distribution<usize>>>,
        rewards: Vec<Vec<Vec<f64>>>,
    ) -> Result<Self, String> {
        let num_states = states.len();
        let num_actions = actions.len();
        if rewards.len() != num_states
            || rewards.iter().any(|row| row.len() != num_actions)
            || rewards
                .iter()
                .flatten()
                .any(|next_states| next_states.len() != num_states)
        {
            return Err("invalid transition reward shape".into());
        }

        let mut model = Self::new(
            states,
            actions,
            transitions,
            vec![vec![0.0; num_actions]; num_states],
        )?;
        model.rewards = Array2::from_shape_fn((num_states, num_actions), |(state, action)| {
            model
                .transitions(state, action)
                .iter()
                .map(|(probability, next_state)| probability * rewards[state][action][*next_state])
                .sum()
        });
        Ok(model)
    }

    pub fn states(&self) -> &[State] {
        &self.states
    }

    pub fn actions(&self) -> &[Action] {
        &self.actions
    }

    pub fn transitions(&self, state: usize, action: usize) -> &probability::Distribution<usize> {
        &self.transitions[state * self.actions.len() + action]
    }

    pub fn reward(&self, state: usize, action: usize) -> f64 {
        self.rewards[[state, action]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probability::Distribution;

    fn states(count: usize) -> Vec<State> {
        (0..count)
            .map(|state| State::new(format!("s{state}")))
            .collect()
    }

    fn actions(count: usize) -> Vec<Action> {
        (0..count)
            .map(|action| Action::new(format!("a{action}")))
            .collect()
    }

    #[test]
    fn model_requires_states_and_actions() {
        assert_eq!(
            TabularModel::new(vec![], actions(1), vec![], vec![]).unwrap_err(),
            "model must have at least one state"
        );
        assert_eq!(
            TabularModel::new(states(1), vec![], vec![vec![]], vec![vec![]]).unwrap_err(),
            "model must have at least one action"
        );
    }

    #[test]
    fn model_validates_state_row_counts() {
        assert_eq!(
            TabularModel::new(states(1), actions(1), vec![], vec![vec![0.0]]).unwrap_err(),
            "expected 1 transition rows, found 0"
        );
        assert_eq!(
            TabularModel::new(
                states(1),
                actions(1),
                vec![vec![Distribution::point_mass(0)]],
                vec![],
            )
            .unwrap_err(),
            "expected 1 reward rows, found 0"
        );
    }

    #[test]
    fn model_validates_action_row_counts() {
        assert_eq!(
            TabularModel::new(states(1), actions(1), vec![vec![]], vec![vec![0.0]]).unwrap_err(),
            "expected 1 transitions for state 0, found 0"
        );
        assert_eq!(
            TabularModel::new(
                states(1),
                actions(1),
                vec![vec![Distribution::point_mass(0)]],
                vec![vec![]],
            )
            .unwrap_err(),
            "expected 1 rewards for state 0, found 0"
        );
    }

    #[test]
    fn model_rejects_invalid_transitions() {
        assert_eq!(
            TabularModel::new(
                states(1),
                actions(1),
                vec![vec![Distribution::point_mass(1)]],
                vec![vec![0.0]],
            )
            .unwrap_err(),
            "transition for state 0, action 0 targets state 1, but the model has 1 states"
        );
    }

    #[test]
    fn transition_rewards_are_reduced_to_expectations() {
        let model = TabularModel::from_transition_rewards(
            states(2),
            actions(1),
            vec![
                vec![Distribution::new(vec![(0.25, 0), (0.75, 1)]).unwrap()],
                vec![Distribution::point_mass(1)],
            ],
            vec![vec![vec![2.0, 6.0]], vec![vec![0.0, 4.0]]],
        )
        .unwrap();

        assert_eq!(model.reward(0, 0), 5.0);
        assert_eq!(model.reward(1, 0), 4.0);
    }
}
