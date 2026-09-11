use rand::{SeedableRng, rngs::StdRng};
use rlstudy::algorithms::{
    ModelBasedSolver, ModelFreeSolver, MonteCarlo, SolveResult, StoppingCriterion, TD,
    TerminationReason, TruncatedPolicyIteration,
};
use rlstudy::environment::{BlackBox, TabularEnvironment};
use rlstudy::environments::grid_world;

fn preferred_action(result: &SolveResult, state: usize) -> usize {
    (0..result.policy.num_actions())
        .max_by(|&a, &b| {
            result
                .policy
                .probability(state, a)
                .total_cmp(&result.policy.probability(state, b))
        })
        .unwrap()
}

fn assert_solves_grid_world(name: &str, result: &SolveResult) {
    assert!(
        [1, 2].contains(&preferred_action(result, 0)),
        "{name}: policy should move right or down from state 0",
    );
    assert_eq!(
        preferred_action(result, 1),
        2,
        "{name}: policy should move down from state 1",
    );
    assert_eq!(
        preferred_action(result, 2),
        3,
        "{name}: policy should move left from state 2",
    );
}

fn model_based_solver_solves_grid_world(name: &str, solver: TruncatedPolicyIteration) {
    let model = grid_world(3).unwrap();
    let result = solver.solve(&model, 0.5).unwrap();

    assert_solves_grid_world(name, &result);

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
    model_based_solver_solves_grid_world(
        "value-iteration",
        TruncatedPolicyIteration::value_iteration(
            StoppingCriterion::convergence(1e-6, 1000).unwrap(),
        ),
    );
}

#[test]
fn policy_iteration_solves_grid_world() {
    model_based_solver_solves_grid_world(
        "policy-iteration",
        TruncatedPolicyIteration::policy_iteration(
            StoppingCriterion::convergence(1e-6, 1000).unwrap(),
        ),
    );
}

#[test]
fn truncated_policy_iteration_solves_grid_world() {
    model_based_solver_solves_grid_world(
        "truncated-policy-iteration",
        TruncatedPolicyIteration::new(
            StoppingCriterion::convergence(1e-6, 1000).unwrap(),
            StoppingCriterion::iterations(4).unwrap(),
        ),
    );
}

#[test]
fn monte_carlo_solves_grid_world() {
    let model = grid_world(3).unwrap();
    let mut environment = TabularEnvironment::new(&model).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let result = MonteCarlo::new(10_000, 30, 0.1)
        .solve(&mut environment, 0.5, &mut rng)
        .unwrap();

    assert_solves_grid_world("monte-carlo", &result);
    assert_eq!(result.iterations, 10_000);
    assert_eq!(result.termination, TerminationReason::IterationLimit);
}

#[test]
fn sarsa_solves_grid_world() {
    let model = grid_world(3).unwrap();
    let mut environment = TabularEnvironment::new(&model).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let result = TD::new(10_000, 30, 0.1, 0.1)
        .unwrap()
        .solve(&mut environment, 0.5, &mut rng)
        .unwrap();

    assert_solves_grid_world("sarsa", &result);
    assert_eq!(result.iterations, 10_000);
    assert_eq!(result.termination, TerminationReason::IterationLimit);
}

#[test]
fn expected_sarsa_solves_grid_world() {
    let model = grid_world(3).unwrap();
    let mut environment = TabularEnvironment::new(&model).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let result = TD::expected(10_000, 30, 0.1, 0.1)
        .unwrap()
        .solve(&mut environment, 0.5, &mut rng)
        .unwrap();

    assert_solves_grid_world("expected-sarsa", &result);
    assert_eq!(result.iterations, 10_000);
    assert_eq!(result.termination, TerminationReason::IterationLimit);
}

#[test]
fn q_learning_solves_grid_world() {
    let model = grid_world(3).unwrap();
    let mut environment = TabularEnvironment::new(&model).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let result = TD::q_learning(10_000, 30, 0.1, 0.1)
        .unwrap()
        .solve(&mut environment, 0.5, &mut rng)
        .unwrap();

    assert_solves_grid_world("q-learning", &result);
    assert_eq!(result.iterations, 10_000);
    assert_eq!(result.termination, TerminationReason::IterationLimit);
}

#[test]
fn q_learning_uses_negative_next_state_values() {
    let mut environment = BlackBox::new(1, 2, 0, |_state, action| {
        (0, if action == 0 { -1.0 } else { -2.0 })
    });
    let mut rng = StdRng::seed_from_u64(0);
    let result = TD::q_learning(10_000, 1, 0.1, 0.1)
        .unwrap()
        .solve(&mut environment, 0.5, &mut rng)
        .unwrap();

    assert!(
        result.value[0] < -1.5,
        "Q-learning must not clamp a negative next-state value to zero"
    );
    assert_eq!(result.policy.probability(0, 0), 1.0);
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

#[test]
fn solver_trace_records_each_evaluation_sweep() {
    let model = grid_world(3).unwrap();
    let solver = TruncatedPolicyIteration::new(
        StoppingCriterion::iterations(1).unwrap(),
        StoppingCriterion::iterations(2).unwrap(),
    );

    let trace = solver.solve_with_trace(&model, 0.5).unwrap();

    assert_eq!(trace.snapshots.len(), 5);
    assert_eq!(trace.snapshots[0].evaluation_sweep, 0);
    assert_eq!(trace.snapshots[0].policy_iteration, 0);
    assert_eq!(trace.snapshots[2].policy_iteration, 0);
    assert_eq!(trace.snapshots[3].policy_iteration, 1);
    assert_eq!(trace.snapshots[4].evaluation_sweep, 4);
    assert_eq!(trace.snapshots[4].value, trace.result.value);
}
