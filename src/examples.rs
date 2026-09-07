use crate::model::{Action, Model, State};
use crate::probability::Distribution;

/// Construct a custom deterministic `n x n` grid-world MDP.
///
/// States are indexed `0..n*n`, arranged row-major:
///
/// ```text
/// 0 1 2
/// 3 4 5
/// 6 7 8
/// ```
///
/// Actions are up, right, down, left, stay. Moving into a wall leaves the
/// agent in place. Rewards are assigned to states as follows:
pub fn grid_world(n: usize) -> Model {
    let num_states = n * n;

    const R_TARGET: f64 = 1.0;
    const R_BOUNDARY: f64 = -1.0;
    const R_FORBIDDEN: f64 = -1.0;

    let states: Vec<State> = (0..num_states)
        .map(|i| State::new(format!("s{}", i + 1), i as u64))
        .collect();

    let actions: Vec<Action> = vec![
        Action::new("up", 0),
        Action::new("right", 1),
        Action::new("down", 2),
        Action::new("left", 3),
        Action::new("stay", 4),
    ];

    let mut s_sa: Vec<Vec<Distribution<u64>>> = Vec::with_capacity(num_states);
    let mut r_sa: Vec<Vec<f64>> = Vec::with_capacity(num_states);

    for s in 0..num_states {
        let row = s / n;
        let col = s % n;

        let mut s_a: Vec<Distribution<u64>> = Vec::with_capacity(actions.len());
        let mut r_a: Vec<f64> = Vec::with_capacity(actions.len());

        for a in 0..actions.len() {
            let next = match a {
                0 if row > 0 => s - n,     // up
                1 if col + 1 < n => s + 1, // right
                2 if row + 1 < n => s + n, // down
                3 if col > 0 => s - 1,     // left
                _ => s,                    // stay / hit a wall
            };

            let reward = match (s, a, next) {
                (s0, _, s1) if s0 == s1 && a != 4 => R_BOUNDARY, // hit a wall
                (_, _, s1) if s1 == num_states - n || s1 == num_states - n - 1 => R_FORBIDDEN, // hit a wall
                (_, _, s1) if s1 == num_states - 1 => R_TARGET,
                _ => 0.,
            };

            s_a.push(Distribution::new(vec![(1.0, next as u64)]));
            r_a.push(reward);
        }

        s_sa.push(s_a);
        r_sa.push(r_a);
    }

    Model::new(states, actions, s_sa, r_sa)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_world_builds() {
        let m = grid_world(3);
        assert_eq!(m.states.len(), 9);
        assert_eq!(m.actions.len(), 5);
        assert_eq!(m.p_s_sa.len(), 9);
        assert_eq!(m.r_sa.len(), 9);
    }
}
