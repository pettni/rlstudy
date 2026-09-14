use rand::{SeedableRng, rngs::StdRng};
use rlstudy::algorithms::{
    ModelBasedSolver, ModelFreeSolver, MonteCarlo, MonteCarloConfig, PolicyIterationConfig,
    SolveStatus, StoppingCriterion, TD, TDConfig, TDVariant, TabularSolveResult,
    TruncatedPolicyIteration, TruncatedPolicyIterationConfig, ValueIterationConfig,
};
use rlstudy::environment::{BlackBox, TabularEnvironment};
use rlstudy::environments::grid_world;
use rlstudy::policy::TabularPolicy;

fn preferred_action(result: &TabularSolveResult, state: usize) -> usize {
    result.policy.greedy_action(&state)
}

fn assert_solves_grid_world(name: &str, result: &TabularSolveResult) {
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
    let values = result
        .snapshots
        .last()
        .unwrap()
        .state_values
        .as_ref()
        .unwrap();
    let goal_value = values[8];
    let max_value = result
        .snapshots
        .last()
        .unwrap()
        .state_values
        .as_ref()
        .unwrap()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (goal_value - max_value).abs() < 1e-6,
        "{}: goal state should have the maximum value, got {:?}",
        name,
        values
    );
    assert!(matches!(result.status, SolveStatus::Converged { .. }));

    assert_eq!(
        preferred_action(&result, 0),
        1,
        "{}: policy action at state 0 should be 1",
        name,
    );

    assert_eq!(
        preferred_action(&result, 1),
        2,
        "{}: policy action at state 1 should be 2",
        name,
    );

    assert_eq!(
        preferred_action(&result, 2),
        3,
        "{}: policy action at state 2 should be 3",
        name,
    );

    assert!(
        result.status.iterations() < 1000,
        "{}: solver took too many iterations: {}",
        name,
        result.status.iterations()
    );
}

#[test]
fn value_iteration_solves_grid_world() {
    model_based_solver_solves_grid_world(
        "value-iteration",
        TruncatedPolicyIteration::value_iteration(ValueIterationConfig {
            tolerance: Some(1e-6),
            max_iterations: Some(1000),
        })
        .unwrap(),
    );
}

#[test]
fn policy_iteration_solves_grid_world() {
    model_based_solver_solves_grid_world(
        "policy-iteration",
        TruncatedPolicyIteration::policy_iteration(PolicyIterationConfig {
            tolerance: Some(1e-6),
            max_iterations: Some(1000),
        })
        .unwrap(),
    );
}

#[test]
fn truncated_policy_iteration_solves_grid_world() {
    model_based_solver_solves_grid_world(
        "truncated-policy-iteration",
        TruncatedPolicyIteration::new(TruncatedPolicyIterationConfig {
            outer_tolerance: Some(1e-6),
            outer_max_iterations: Some(1000),
            inner_tolerance: None,
            inner_max_iterations: Some(4),
        })
        .unwrap(),
    );
}

#[test]
fn monte_carlo_solves_grid_world() {
    let model = grid_world(3).unwrap();
    let mut environment = TabularEnvironment::new(&model).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let result = MonteCarlo::new(MonteCarloConfig::default())
        .unwrap()
        .solve(&mut environment, 0.5, &mut rng)
        .unwrap();

    assert_solves_grid_world("monte-carlo", &result);
    assert_eq!(
        result.status,
        SolveStatus::IterationLimit { iterations: 10_000 }
    );
}

#[test]
fn sarsa_solves_grid_world() {
    let model = grid_world(3).unwrap();
    let mut environment = TabularEnvironment::new(&model).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let result = TD::new(TDConfig::default())
        .unwrap()
        .solve(&mut environment, 0.5, &mut rng)
        .unwrap();

    assert_solves_grid_world("sarsa", &result);
    assert_eq!(
        result.status,
        SolveStatus::IterationLimit { iterations: 10_000 }
    );
}

#[test]
fn expected_sarsa_solves_grid_world() {
    let model = grid_world(3).unwrap();
    let mut environment = TabularEnvironment::new(&model).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let result = TD::new(TDConfig {
        variant: TDVariant::ExpectedSarsa,
        ..TDConfig::default()
    })
    .unwrap()
    .solve(&mut environment, 0.5, &mut rng)
    .unwrap();

    assert_solves_grid_world("expected-sarsa", &result);
    assert_eq!(
        result.status,
        SolveStatus::IterationLimit { iterations: 10_000 }
    );
}

#[test]
fn q_learning_solves_grid_world() {
    let model = grid_world(3).unwrap();
    let mut environment = TabularEnvironment::new(&model).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let result = TD::new(TDConfig {
        variant: TDVariant::QLearning,
        ..TDConfig::default()
    })
    .unwrap()
    .solve(&mut environment, 0.5, &mut rng)
    .unwrap();

    assert_solves_grid_world("q-learning", &result);
    assert_eq!(
        result.status,
        SolveStatus::IterationLimit { iterations: 10_000 }
    );
}

#[test]
fn q_learning_off_policy_solves_grid_world() {
    let model = grid_world(3).unwrap();
    let mut environment = TabularEnvironment::new(&model).unwrap();
    let mut rng = StdRng::seed_from_u64(0);
    let result = TD::new(TDConfig {
        epsilon: 0.0,
        variant: TDVariant::QLearning,
        ..TDConfig::default()
    })
    .unwrap()
    .with_behavior_policy(TabularPolicy::uniform(9, 5).unwrap())
    .solve(&mut environment, 0.5, &mut rng)
    .unwrap();

    assert_solves_grid_world("q-learning", &result);
    assert_eq!(
        result.status,
        SolveStatus::IterationLimit { iterations: 10_000 }
    );
}

#[test]
fn q_learning_uses_negative_next_state_values() {
    let mut environment = BlackBox::new(1, 2, 0, |_state, action| {
        (0, if action == 0 { -1.0 } else { -2.0 })
    });
    let mut rng = StdRng::seed_from_u64(0);
    let result = TD::new(TDConfig {
        episode_length: 1,
        variant: TDVariant::QLearning,
        ..TDConfig::default()
    })
    .unwrap()
    .solve(&mut environment, 0.5, &mut rng)
    .unwrap();

    assert!(
        result
            .snapshots
            .last()
            .unwrap()
            .state_values
            .as_ref()
            .unwrap()[0]
            < -1.5,
        "Q-learning must not clamp a negative next-state value to zero"
    );
    assert_eq!(preferred_action(&result, 0), 0);
}

#[test]
fn solver_configuration_is_validated() {
    assert!(StoppingCriterion::iterations(0).is_err());
    assert!(StoppingCriterion::convergence(0.0, 10).is_err());
    assert!(StoppingCriterion::convergence(f64::NAN, 10).is_err());

    let model = grid_world(3).unwrap();
    let solver = TruncatedPolicyIteration::value_iteration(ValueIterationConfig {
        tolerance: None,
        max_iterations: Some(1),
    })
    .unwrap();
    assert!(solver.solve(&model, -0.1).is_err());
    assert!(solver.solve(&model, 1.0).is_err());
}

#[test]
fn solver_reports_iteration_limit() {
    let model = grid_world(3).unwrap();
    let solver = TruncatedPolicyIteration::with_stopping_criteria(
        StoppingCriterion::iterations(1).unwrap(),
        StoppingCriterion::iterations(1).unwrap(),
    );

    let result = solver.solve(&model, 0.5).unwrap();

    assert_eq!(result.status, SolveStatus::IterationLimit { iterations: 1 });
}

#[test]
fn solver_trace_records_each_evaluation_sweep() {
    let model = grid_world(3).unwrap();
    let solver = TruncatedPolicyIteration::with_stopping_criteria(
        StoppingCriterion::iterations(1).unwrap(),
        StoppingCriterion::iterations(2).unwrap(),
    );

    let result = solver.solve(&model, 0.5).unwrap();
    let snapshots = &result.snapshots;

    assert_eq!(snapshots.len(), 5);
    assert_eq!(snapshots[0].evaluation_sweep, Some(0));
    assert_eq!(snapshots[0].policy_iteration, Some(0));
    assert_eq!(snapshots[2].policy_iteration, Some(0));
    assert_eq!(snapshots[3].policy_iteration, Some(1));
    assert_eq!(snapshots[4].evaluation_sweep, Some(4));
    assert_eq!(
        snapshots[4].state_value,
        Some(snapshots[4].state_values.as_ref().unwrap()[0])
    );
}
