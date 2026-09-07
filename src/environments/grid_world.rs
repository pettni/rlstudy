use crate::model::{Action, State, TabularModel};
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
pub fn grid_world(n: usize) -> Result<TabularModel, String> {
    let n_s = n * n;

    const R_TARGET: f64 = 1.0;
    const R_BOUNDARY: f64 = -1.0;
    const R_FORBIDDEN: f64 = -1.0;

    let states: Vec<State> = (0..n_s)
        .map(|i| State::new(format!("s{}", i + 1)))
        .collect();

    let actions: Vec<Action> = vec![
        Action::new("up"),
        Action::new("right"),
        Action::new("down"),
        Action::new("left"),
        Action::new("stay"),
    ];

    TabularModel::from_functions(
        states,
        actions,
        |s, a| {
            let row = s / n;
            let col = s % n;

            let next = match a {
                0 if row > 0 => s - n,     // up
                1 if col + 1 < n => s + 1, // right
                2 if row + 1 < n => s + n, // down
                3 if col > 0 => s - 1,     // left
                _ => s,                    // stay / hit a wall
            };

            Distribution::point_mass(next)
        },
        |s, a, next_state| {
            let reward = match (s, a, next_state) {
                (s0, _, s1) if s0 == s1 && a != 4 => R_BOUNDARY,
                (_, _, s1) if s1 == n_s - n || s1 == n_s - n - 1 => R_FORBIDDEN,
                (_, _, s1) if s1 == n_s - 1 => R_TARGET,
                _ => 0.0,
            };
            Distribution::point_mass(reward)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_world_builds() {
        let m = grid_world(3).unwrap();
        assert_eq!(m.states().len(), 9);
        assert_eq!(m.actions().len(), 5);
        assert_eq!(m.states()[0].name, "s1");
        assert_eq!(m.actions()[1].name, "right");

        let outcome = m.outcomes(0, 1).iter().next().unwrap().1;
        assert_eq!(outcome, (1, 0.0));
    }
}
