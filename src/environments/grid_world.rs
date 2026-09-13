use crate::model::{Action, State, TabularModel};
use crate::probability::Distribution;
use ndarray::Array2;
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Construct a custom deterministic `n x n` grid-world MDP.
///
/// Actions are up, right, down, left, stay. Moving into a wall leaves the
/// agent in place. Rewards are assigned to states as follows:

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum InitialDistribution {
    /// Fixed starting coordinates
    Fixed { state: (usize, usize) },

    /// Uniform distribution over all states
    Uniform,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GridWorldSpec {
    /// Size of grid square
    side: usize,

    /// Initial state distribution
    initial_distribution: InitialDistribution,

    /// Forbidden states
    forbidden_states: Vec<(usize, usize)>,

    /// Target states
    goal_states: Vec<(usize, usize)>,

    /// Reward for moving into a boundary
    r_boundary: f64,

    /// Reward for moving into a forbidden state
    r_forbidden: f64,

    /// Reward for moving into a target state
    r_target: f64,

    /// Reward for moving into any other state
    r_other: f64,
}

/// Construct a pre-defined grid world with two obstacles
pub fn grid_world(n: usize) -> Result<TabularModel, String> {
    grid_world_impl(GridWorldSpec {
        side: n,
        initial_distribution: InitialDistribution::Fixed { state: (0, 0) },
        forbidden_states: vec![(n - 2, n - 1), (n - 1, 0)],
        goal_states: vec![(n - 1, n - 1)],
        r_boundary: -1.0,
        r_forbidden: -1.0,
        r_target: 1.0,
        r_other: 0.0,
    })
}

pub fn load_grid_world(path: impl AsRef<Path>) -> Result<TabularModel, String> {
    let path = path.as_ref();
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("failed to read grid world {}: {error}", path.display()))?;
    let spec = yaml_serde::from_str(&contents)
        .map_err(|error| format!("failed to parse grid world {}: {error}", path.display()))?;
    grid_world_impl(spec).map_err(|error| format!("invalid grid world {}: {error}", path.display()))
}

pub fn grid_world_impl(spec: GridWorldSpec) -> Result<TabularModel, String> {
    let n = spec.side;
    if n == 0 {
        return Err("grid side must be greater than zero".into());
    }
    let n_s = n
        .checked_mul(n)
        .ok_or_else(|| "grid side is too large".to_string())?;

    if let InitialDistribution::Fixed { state } = &spec.initial_distribution {
        validate_coordinates(*state, n, "initial state")?;
    }
    for &state in &spec.forbidden_states {
        validate_coordinates(state, n, "forbidden state")?;
    }
    for &state in &spec.goal_states {
        validate_coordinates(state, n, "goal state")?;
    }

    let states: Vec<State> = (0..n_s)
        .map(|s| {
            let (i, j) = idx_to_coords(s, n);
            State::new(format!("s{},{}", i, j))
        })
        .collect();

    let actions: Vec<Action> = vec![
        Action::new("up"),
        Action::new("right"),
        Action::new("down"),
        Action::new("left"),
        Action::new("stay"),
    ];

    let features = Array2::from_shape_fn((n_s, 2), |(s, i)| -> f64 {
        let (row, col) = idx_to_coords(s, n);
        [row as f64, col as f64][i]
    });

    let initial_distribution: Distribution<usize> = match spec.initial_distribution {
        InitialDistribution::Fixed { state } => {
            Distribution::<usize>::point_mass(coord_to_idx(state, n))
        }
        InitialDistribution::Uniform => {
            let ts = (0..n_s).collect::<Vec<_>>();
            Distribution::<usize>::uniform(ts)?
        }
    };

    let is_forbidden = |s| -> bool {
        let ij = idx_to_coords(s, n);
        spec.forbidden_states.contains(&ij)
    };

    let is_goal = |s| -> bool {
        let ij = idx_to_coords(s, n);
        spec.goal_states.contains(&ij)
    };

    TabularModel::from_functions(
        initial_distribution,
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
                (s0, _, s1) if s0 == s1 && a != 4 => spec.r_boundary,
                (_, _, s1) if is_forbidden(s1) => spec.r_forbidden,
                (_, _, s1) if is_goal(s1) => spec.r_target,
                _ => spec.r_other,
            };
            Distribution::point_mass(reward)
        },
        Some(features),
    )
}

fn validate_coordinates(
    coordinates: (usize, usize),
    side: usize,
    description: &str,
) -> Result<(), String> {
    if coordinates.0 >= side || coordinates.1 >= side {
        return Err(format!(
            "{description} ({}, {}) is outside a {side}x{side} grid",
            coordinates.0, coordinates.1
        ));
    }
    Ok(())
}

fn idx_to_coords(s: usize, n: usize) -> (usize, usize) {
    (s / n, s % n)
}

fn coord_to_idx(ij: (usize, usize), n: usize) -> usize {
    let (i, j) = ij;
    i * n + j
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_world_builds() {
        let m = grid_world(3).unwrap();
        assert_eq!(m.states().len(), 9);
        assert_eq!(m.actions().len(), 5);
        assert_eq!(m.states()[0].name, "s0,0");
        assert_eq!(m.actions()[1].name, "right");

        let outcome = m.outcomes(0, 1).iter().next().unwrap().1;
        assert_eq!(outcome, (1, 0.0));
    }

    #[test]
    fn coords_rowmajor() {
        assert_eq!(idx_to_coords(3, 5), (0, 3));
        assert_eq!(idx_to_coords(5, 5), (1, 0));
    }

    #[test]
    fn coords_roudtrip() {
        let n = 5;
        for s in 0..n * n {
            let (i, j) = idx_to_coords(s, n);
            let s2 = coord_to_idx((i, j), n);
            assert_eq!(s, s2);
        }
    }

    #[test]
    fn grid_world_yaml_supports_fixed_initial_state() {
        let spec = yaml_serde::from_str(
            r#"
side: 3
initial_distribution:
  type: fixed
  state: [1, 1]
forbidden_states:
  - [1, 2]
goal_states:
  - [2, 2]
r_boundary: -1
r_forbidden: -2
r_target: 3
r_other: 0
"#,
        )
        .unwrap();
        let model = grid_world_impl(spec).unwrap();

        assert_eq!(model.states().len(), 9);
        assert_eq!(model.initial_distribution().probability_of(&4), 1.0);
    }

    #[test]
    fn grid_world_yaml_supports_uniform_initial_state() {
        let spec = yaml_serde::from_str(
            r#"
side: 2
initial_distribution:
  type: uniform
forbidden_states: []
goal_states:
  - [1, 1]
r_boundary: -1
r_forbidden: -2
r_target: 3
r_other: 0
"#,
        )
        .unwrap();
        let model = grid_world_impl(spec).unwrap();

        for state in 0..4 {
            assert_eq!(model.initial_distribution().probability_of(&state), 0.25);
        }
    }

    #[test]
    fn grid_world_rejects_coordinates_outside_grid() {
        let result = grid_world_impl(GridWorldSpec {
            side: 3,
            initial_distribution: InitialDistribution::Fixed { state: (3, 0) },
            forbidden_states: vec![],
            goal_states: vec![(2, 2)],
            r_boundary: -1.0,
            r_forbidden: -2.0,
            r_target: 3.0,
            r_other: 0.0,
        });

        assert_eq!(
            result.unwrap_err(),
            "initial state (3, 0) is outside a 3x3 grid"
        );
    }

    #[test]
    fn grid_world_yaml_configures_rewards() {
        let spec = yaml_serde::from_str(
            r#"
side: 3
initial_distribution:
  type: fixed
  state: [0, 0]
forbidden_states:
  - [1, 2]
goal_states:
  - [2, 2]
r_boundary: -2
r_forbidden: -3
r_target: 4
r_other: -5
"#,
        )
        .unwrap();
        let model = grid_world_impl(spec).unwrap();

        assert_eq!(model.outcomes(0, 0).iter().next().unwrap().1, (0, -2.0));
        assert_eq!(model.outcomes(0, 1).iter().next().unwrap().1, (1, -5.0));
        assert_eq!(model.outcomes(2, 2).iter().next().unwrap().1, (5, -3.0));
        assert_eq!(model.outcomes(5, 2).iter().next().unwrap().1, (8, 4.0));
    }
}
