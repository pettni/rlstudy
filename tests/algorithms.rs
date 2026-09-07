use rlstudy::algorithms::{
    ModelBasedSolver, StoppingCriterion, TerminationReason, TruncatedPolicyIteration,
};
use rlstudy::environments::grid_world;

fn solver_solves_grid_world(name: &str, solver: TruncatedPolicyIteration) {
    let model = grid_world(3).unwrap();
    let result = solver.solve(&model, 0.5).unwrap();

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
    assert_eq!(result.termination, TerminationReason::Converged);

    assert_eq!(
        result.policy.probability(0, 1),
        1.0,
        "{}: policy action at state 0 should be 1",
        name,
    );

    assert_eq!(
        result.policy.probability(1, 2),
        1.0,
        "{}: policy action at state 1 should be 2",
        name,
    );

    assert_eq!(
        result.policy.probability(2, 3),
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
        TruncatedPolicyIteration::value_iteration(
            StoppingCriterion::convergence(1e-6, 1000).unwrap(),
        ),
    );
}

#[test]
fn policy_iteration_solves_grid_world() {
    solver_solves_grid_world(
        "policy-iteration",
        TruncatedPolicyIteration::policy_iteration(
            StoppingCriterion::convergence(1e-6, 1000).unwrap(),
        ),
    );
}

#[test]
fn truncated_policy_iteration_solves_grid_world() {
    solver_solves_grid_world(
        "truncated-policy-iteration",
        TruncatedPolicyIteration::new(
            StoppingCriterion::convergence(1e-6, 1000).unwrap(),
            StoppingCriterion::iterations(4).unwrap(),
        ),
    );
}

#[test]
fn solver_configuration_is_validated() {
    assert!(StoppingCriterion::iterations(0).is_err());
    assert!(StoppingCriterion::convergence(0.0, 10).is_err());
    assert!(StoppingCriterion::convergence(f64::NAN, 10).is_err());

    let model = grid_world(3).unwrap();
    let solver =
        TruncatedPolicyIteration::value_iteration(StoppingCriterion::iterations(1).unwrap());
    assert!(solver.solve(&model, -0.1).is_err());
    assert!(solver.solve(&model, 1.0).is_err());
}

#[test]
fn solver_reports_iteration_limit() {
    let model = grid_world(3).unwrap();
    let solver = TruncatedPolicyIteration::new(
        StoppingCriterion::iterations(1).unwrap(),
        StoppingCriterion::iterations(1).unwrap(),
    );

    let result = solver.solve(&model, 0.5).unwrap();

    assert_eq!(result.termination, TerminationReason::IterationLimit);
}
