use rlstudy::examples::grid_world;
use rlstudy::solvers::ModelBasedSolver;
use rlstudy::solvers::truncated_policy_iteration::{StoppingCriterion, TruncatedPolicyIteration};

fn solver_solves_grid_world(name: &str, mut solver: TruncatedPolicyIteration) {
    let model = grid_world(3);
    let result = solver.solve(&model, 0.5);

    // The goal state (bottom-right, index 8) should have the highest value.
    let goal_value = result.value[8];
    let max_value = result
        .value
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (goal_value - max_value).abs() < 1e-6,
        "{}: goal state should have the maximum value, got {:?}",
        name,
        result.value
    );

    assert_eq!(
        result.policy.prob(0, 1),
        1.0,
        "{}: policy action at state 0 should be 1",
        name,
    );

    assert_eq!(
        result.policy.prob(1, 2),
        1.0,
        "{}: policy action at state 1 should be 2",
        name,
    );

    assert_eq!(
        result.policy.prob(2, 3),
        1.0,
        "{}: policy action at state 2 should be 3",
        name,
    );

    assert!(
        result.iterations < 1000,
        "{}: solver took too many iterations: {}",
        name,
        result.iterations
    );
}

#[test]
fn value_iteration_solves_grid_world() {
    solver_solves_grid_world(
        "value-iteration",
        TruncatedPolicyIteration::value_iteration(StoppingCriterion::Tolerance(1e-6)),
    );
}

#[test]
fn policy_iteration_solves_grid_world() {
    solver_solves_grid_world(
        "policy-iteration",
        TruncatedPolicyIteration::policy_iteration(StoppingCriterion::Tolerance(1e-6)),
    );
}

#[test]
fn truncated_policy_iteration_solves_grid_world() {
    solver_solves_grid_world(
        "truncated-policy-iteration",
        TruncatedPolicyIteration::new(
            StoppingCriterion::Tolerance(1e-6),
            StoppingCriterion::Iterations(4),
        ),
    );
}
