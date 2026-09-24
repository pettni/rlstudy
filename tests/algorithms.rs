use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use rlstudy::algorithms::{
    ModelBasedSolver, ModelFreeSolver, MonteCarlo, MonteCarloConfig, PolicyIterationConfig,
    SolveStatus, StoppingCriterion, TD, TDConfig, TDVariant, TabularSolveResult,
    TruncatedPolicyIteration, TruncatedPolicyIterationConfig, ValueIterationConfig,
};
use rlstudy::environment::{BlackBox, TabularEnvironment};
use rlstudy::environments::grid_world;
use rlstudy::evaluation::fixed_policy_values;
use rlstudy::policy::TabularPolicy;
use rlstudy::reporting::{Reporter, ReporterConfig};

fn preferred_action(result: &TabularSolveResult, state: usize, num_actions: usize) -> usize {
    (0..num_actions)
        .max_by(|a, b| {
            result
                .policy
                .log_prob(&state, a)
                .total_cmp(&result.policy.log_prob(&state, b))
        })
        .unwrap()
}

fn assert_solves_grid_world(name: &str, result: &TabularSolveResult) {
    assert!(
        [1, 2].contains(&preferred_action(result, 0, 5)),
        "{name}: policy should move right or down from state 0",
    );
    assert_eq!(
        preferred_action(result, 1, 5),
        2,
        "{name}: policy should move down from state 1",
    );
    assert_eq!(
        preferred_action(result, 2, 5),
        3,
        "{name}: policy should move left from state 2",
    );
}

fn model_based_solver_solves_grid_world(name: &str, solver: TruncatedPolicyIteration) {
    let model = grid_world(3).unwrap();
    let result = solver.solve(&model).unwrap();

    assert_solves_grid_world(name, &result);

    // The goal state (bottom-right, index 8) should have the highest value.
    let probabilities = Array2::from_shape_fn(
        (model.states().len(), model.actions().len()),
        |(state, action)| result.policy.log_prob(&state, &action).exp(),
    );
    let values = fixed_policy_values(&model, &probabilities, 0.5).unwrap();
    let goal_value = values[8];
    let max_value = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (goal_value - max_value).abs() < 1e-6,
        "{}: goal state should have the maximum value, got {:?}",
        name,
        values
    );
    assert!(matches!(result.status, SolveStatus::Converged { .. }));

    assert_eq!(
        preferred_action(&result, 0, 5),
        1,
        "{}: policy action at state 0 should be 1",
        name,
    );

    assert_eq!(
        preferred_action(&result, 1, 5),
        2,
        "{}: policy action at state 1 should be 2",
        name,
    );

    assert_eq!(
        preferred_action(&result, 2, 5),
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
            gamma: 0.5,
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
            gamma: 0.5,
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
            gamma: 0.5,
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
    let result = MonteCarlo::new(MonteCarloConfig {
        gamma: 0.5,
        ..MonteCarloConfig::default()
    })
    .unwrap()
    .solve(&mut environment, &mut rng)
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
    let result = TD::new(TDConfig {
        gamma: 0.5,
        ..TDConfig::default()
    })
    .unwrap()
    .solve(&mut environment, &mut rng)
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
        gamma: 0.5,
        variant: TDVariant::ExpectedSarsa,
        ..TDConfig::default()
    })
    .unwrap()
    .solve(&mut environment, &mut rng)
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
        gamma: 0.5,
        variant: TDVariant::QLearning,
        ..TDConfig::default()
    })
    .unwrap()
    .solve(&mut environment, &mut rng)
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
        gamma: 0.5,
        epsilon: 0.0,
        variant: TDVariant::QLearning,
        ..TDConfig::default()
    })
    .unwrap()
    .with_behavior_policy(TabularPolicy::uniform(9, 5).unwrap())
    .solve(&mut environment, &mut rng)
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
        gamma: 0.5,
        episode_length: 1,
        variant: TDVariant::QLearning,
        ..TDConfig::default()
    })
    .unwrap()
    .solve(&mut environment, &mut rng)
    .unwrap();

    assert_eq!(preferred_action(&result, 0, 2), 0);
}

#[test]
fn solver_configuration_is_validated() {
    assert!(StoppingCriterion::iterations(0).is_err());
    assert!(StoppingCriterion::convergence(0.0, 10).is_err());
    assert!(StoppingCriterion::convergence(f64::NAN, 10).is_err());

    assert!(
        TruncatedPolicyIteration::value_iteration(ValueIterationConfig {
            gamma: -0.1,
            tolerance: None,
            max_iterations: Some(1),
        })
        .is_err()
    );
    assert!(
        TruncatedPolicyIteration::value_iteration(ValueIterationConfig {
            gamma: 1.0,
            tolerance: None,
            max_iterations: Some(1),
        })
        .is_err()
    );
}

#[test]
fn solver_reports_iteration_limit() {
    let model = grid_world(3).unwrap();
    let solver = TruncatedPolicyIteration::with_stopping_criteria(
        StoppingCriterion::iterations(1).unwrap(),
        StoppingCriterion::iterations(1).unwrap(),
        0.5,
    )
    .unwrap();

    let result = solver.solve(&model).unwrap();

    assert_eq!(result.status, SolveStatus::IterationLimit { iterations: 1 });
}

#[test]
fn solver_records_each_outer_iteration() {
    let model = grid_world(3).unwrap();
    let solver = TruncatedPolicyIteration::with_stopping_criteria(
        StoppingCriterion::iterations(1).unwrap(),
        StoppingCriterion::iterations(2).unwrap(),
        0.5,
    )
    .unwrap();

    let mut reporter = Reporter::new(ReporterConfig::default(), 0, None);
    let result = solver.solve_with_reporter(&model, &mut reporter).unwrap();
    let history = reporter.into_history();
    assert_eq!(result.status, SolveStatus::IterationLimit { iterations: 1 });
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].unit, "iteration");
    assert_eq!(history[0].step, 1);
    assert!(history[0].metrics.contains_key("value/initial"));
}
