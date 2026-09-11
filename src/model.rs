use crate::probability;

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
    // initial state distribution
    initial_distribution: probability::Distribution<usize>,
    // all states
    states: Vec<State>,
    // all actions
    actions: Vec<Action>,
    // p(s',r|s,a) for each state-action pair
    outcomes: Vec<probability::Distribution<(usize, f64)>>,
}

impl TabularModel {
    /// Construct a model with joint transition distributions `p(s', r | s, a)`.
    pub fn new(
        initial_distribution: probability::Distribution<usize>,
        states: Vec<State>,
        actions: Vec<Action>,
        p_sr_sa: Vec<Vec<probability::Distribution<(usize, f64)>>>,
    ) -> Result<Self, String> {
        let num_states = states.len();
        let num_actions = actions.len();

        if num_states == 0 {
            return Err("model must have at least one state".into());
        }
        if num_actions == 0 {
            return Err("model must have at least one action".into());
        }
        if p_sr_sa.len() != num_states {
            return Err(format!(
                "expected {num_states} outcome rows, found {}",
                p_sr_sa.len()
            ));
        }

        for (state, outcomes_by_action) in p_sr_sa.iter().enumerate() {
            if outcomes_by_action.len() != num_actions {
                return Err(format!(
                    "expected {num_actions} outcomes for state {state}, found {}",
                    outcomes_by_action.len()
                ));
            }

            for (action, outcomes) in outcomes_by_action.iter().enumerate() {
                for (_, (next_state, _)) in outcomes.iter() {
                    if *next_state >= num_states {
                        return Err(format!(
                            "transition for state {state}, action {action} targets state {next_state}, but the model has {num_states} states"
                        ));
                    }
                }
            }
        }

        let outcomes = p_sr_sa.into_iter().flatten().collect();

        Ok(Self {
            initial_distribution,
            states,
            actions,
            outcomes,
        })
    }

    /// Construct a joint model from transition and reward functions.
    pub fn from_functions<F, G>(
        initial_distribution: probability::Distribution<usize>,
        states: Vec<State>,
        actions: Vec<Action>,
        transition_distribution: F,
        reward_distribution: G,
    ) -> Result<Self, String>
    where
        F: Fn(usize, usize) -> probability::Distribution<usize>,
        G: Fn(usize, usize, usize) -> probability::Distribution<f64>,
    {
        let n_s = states.len();
        let n_a = actions.len();

        let mut p_sr_sa = Vec::with_capacity(n_s);

        for s in 0..n_s {
            let mut p_sr_a = Vec::with_capacity(n_a);
            for a in 0..n_a {
                let mut outcomes = Vec::new();
                for (p_sp, sp) in transition_distribution(s, a).iter() {
                    if *sp >= n_s {
                        return Err(format!(
                            "transition for state {s}, action {a} targets state {sp}, but the model has {n_s} states"
                        ));
                    }
                    for (p_r, reward) in reward_distribution(s, a, *sp).iter() {
                        outcomes.push((p_sp * p_r, (*sp, *reward)));
                    }
                }
                p_sr_a.push(probability::Distribution::new(outcomes)?);
            }
            p_sr_sa.push(p_sr_a);
        }

        Self::new(initial_distribution, states, actions, p_sr_sa)
    }

    pub fn initial_distribution(&self) -> &probability::Distribution<usize> {
        &self.initial_distribution
    }

    pub fn states(&self) -> &[State] {
        &self.states
    }

    pub fn actions(&self) -> &[Action] {
        &self.actions
    }

    pub fn outcomes(
        &self,
        state: usize,
        action: usize,
    ) -> &probability::Distribution<(usize, f64)> {
        &self.outcomes[state * self.actions.len() + action]
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
            TabularModel::new(
                probability::Distribution::point_mass(0),
                vec![],
                actions(1),
                vec![]
            )
            .unwrap_err(),
            "model must have at least one state"
        );
        assert_eq!(
            TabularModel::new(
                probability::Distribution::point_mass(0),
                states(1),
                vec![],
                vec![vec![]]
            )
            .unwrap_err(),
            "model must have at least one action"
        );
    }

    #[test]
    fn model_validates_state_row_counts() {
        assert_eq!(
            TabularModel::new(
                probability::Distribution::point_mass(0),
                states(1),
                actions(1),
                vec![]
            )
            .unwrap_err(),
            "expected 1 outcome rows, found 0"
        );
    }

    #[test]
    fn model_validates_action_row_counts() {
        assert_eq!(
            TabularModel::new(
                probability::Distribution::point_mass(0),
                states(1),
                actions(1),
                vec![vec![]]
            )
            .unwrap_err(),
            "expected 1 outcomes for state 0, found 0"
        );
    }

    #[test]
    fn model_rejects_invalid_transitions() {
        assert_eq!(
            TabularModel::new(
                probability::Distribution::point_mass(0),
                states(1),
                actions(1),
                vec![vec![Distribution::point_mass((1, 0.0))]],
            )
            .unwrap_err(),
            "transition for state 0, action 0 targets state 1, but the model has 1 states"
        );
    }
}
