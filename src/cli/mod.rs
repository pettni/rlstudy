mod solvers;
mod systems;

use self::solvers::{SelectedSolver, Solver, build_solver};
use self::systems::{SelectedSystem, build_system};
use clap::{Parser, Subcommand};
use ndarray::Array1;
use rand::{SeedableRng, rngs::StdRng};
use rlstudy::algorithms::{
    ModelBasedSolver, ModelFreeSolver, SolveResult, SolverSnapshot, TabularSolveResult,
};
use rlstudy::environment::{FeatureEnvironment, TabularEnvironment};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rlstudy", version, about = "Reinforcement learning study CLI")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a solver against a system.
    Solve(SolveArgs),
}

#[derive(Parser)]
struct SolveArgs {
    /// Grid-world YAML file. Omit both system options for the built-in grid world.
    #[arg(long, conflicts_with = "gym")]
    grid_world: Option<PathBuf>,

    /// Gymnasium environment ID.
    #[arg(long, conflicts_with = "grid_world")]
    gym: Option<String>,

    /// Load the selected solver's configuration from YAML.
    #[arg(long)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    solver: Solver,

    /// Write values and greedy policy actions as JSON.
    #[arg(long)]
    json: bool,

    /// Seed for environment, policy, replay sampling, and model initialization.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

pub(crate) fn run() -> Result<(), Box<dyn std::error::Error>> {
    match Args::parse().command {
        Commands::Solve(args) => solve(&args),
    }
}

fn solve(args: &SolveArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut system = build_system(args.grid_world.as_deref(), args.gym.as_deref())?;
    let (solver, parameters) = build_solver(&args.solver, args.config.as_deref())?;
    let mut rng = StdRng::seed_from_u64(args.seed);
    let result = run_solver(solver, &mut system, &mut rng)?;

    if args.json {
        write_json_result(args, &system, &result, &parameters)?;
    } else {
        print_result(args, result, &parameters);
    }
    Ok(())
}

fn run_solver(
    solver: SelectedSolver,
    system: &mut SelectedSystem,
    rng: &mut dyn rand::Rng,
) -> Result<SelectedResult, Box<dyn std::error::Error>> {
    match &solver {
        SelectedSolver::ModelBased(solver) => {
            let model = system
                .as_model()
                .ok_or("model-based solver requires a model system")?;
            Ok(SelectedResult::Tabular(solver.solve(model)?))
        }
        SelectedSolver::MonteCarlo(solver) => {
            let result = match system {
                SelectedSystem::Model(model) => {
                    solver.solve(&mut TabularEnvironment::new(model)?, rng)
                }
                SelectedSystem::Gym(_) => return Err("Monte Carlo requires tabular states".into()),
            };
            Ok(SelectedResult::Tabular(result?))
        }
        SelectedSolver::TD(solver) => {
            let result = match system {
                SelectedSystem::Model(model) => {
                    solver.solve(&mut TabularEnvironment::new(model)?, rng)
                }
                SelectedSystem::Gym(_) => return Err("TD requires tabular states".into()),
            };
            Ok(SelectedResult::Tabular(result?))
        }
        SelectedSolver::Dqn(solver) => {
            let result = match system {
                SelectedSystem::Model(model) => {
                    solver.solve(&mut FeatureEnvironment::new(model)?, rng)
                }
                SelectedSystem::Gym(environment) => solver.solve(environment, rng),
            };
            Ok(SelectedResult::Dqn(result?))
        }
        SelectedSolver::Reinforce(solver) => {
            let result = match system {
                SelectedSystem::Model(model) => {
                    solver.solve(&mut FeatureEnvironment::new(model)?, rng)
                }
                SelectedSystem::Gym(environment) => solver.solve(environment, rng),
            };
            Ok(SelectedResult::Dqn(result?))
        }
    }
}

enum SelectedResult {
    Tabular(TabularSolveResult),
    Dqn(SolveResult<Array1<f64>>),
}

#[derive(Serialize)]
struct VisualizationResult<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    grid_world_definition: Option<&'a str>,
    solver: &'a str,
    seed: u64,
    parameters: &'a serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    policy_actions: Option<Vec<usize>>,
    snapshots: &'a [SolverSnapshot],
}

fn write_json_result(
    args: &SolveArgs,
    system: &SelectedSystem,
    result: &SelectedResult,
    parameters: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = system.as_model();
    let grid_world_definition = args
        .grid_world
        .as_deref()
        .and_then(|path| path.to_str())
        .or_else(|| model.map(|_| "grid_worlds/zhao_3x3.yaml"));

    let (snapshots, policy_actions) = match result {
        SelectedResult::Tabular(result) => (
            result.snapshots.as_slice(),
            Some(
                (0..result
                    .snapshots
                    .last()
                    .and_then(|snapshot| snapshot.state_values.as_ref())
                    .map_or(0, Vec::len))
                    .map(|state| result.policy.greedy_action(&state))
                    .collect(),
            ),
        ),
        SelectedResult::Dqn(result) => (
            result.snapshots.as_slice(),
            model.map(|model| {
                (0..model.states().len())
                    .map(|state| {
                        let observation = model
                            .state_features(state)
                            .expect("DQN requires model state features")
                            .to_owned();
                        result.policy.greedy_action(&observation)
                    })
                    .collect()
            }),
        ),
    };

    serde_json::to_writer(
        std::io::stdout(),
        &VisualizationResult {
            grid_world_definition,
            solver: args.solver.as_str(),
            seed: args.seed,
            parameters,
            policy_actions,
            snapshots,
        },
    )?;
    println!();
    Ok(())
}

fn print_result(args: &SolveArgs, result: SelectedResult, parameters: &serde_json::Value) {
    let system = args
        .gym
        .as_deref()
        .or_else(|| args.grid_world.as_deref().and_then(|path| path.to_str()))
        .unwrap_or("gridworld");
    println!(
        "system={} solver={} seed={}",
        system,
        args.solver.as_str(),
        args.seed
    );
    println!("parameters={parameters}");
    match result {
        SelectedResult::Tabular(result) => {
            let values = result
                .snapshots
                .last()
                .and_then(|snapshot| snapshot.state_values.as_ref())
                .expect("tabular solvers return final state values");
            if let Some(v0) = values.first() {
                println!("value[0]={:.4}", v0);
            }
            println!("iterations={}", result.status.iterations());
            println!("status={:?}", result.status);
            println!("policy:");
            for (s, value) in values.iter().enumerate() {
                let action = result.policy.greedy_action(&s);
                println!("  state {} [value {:.3}] -> action {}", s, value, action);
            }
            print_snapshots(&result.snapshots);
        }
        SelectedResult::Dqn(result) => {
            println!("iterations={}", result.status.iterations());
            println!("status={:?}", result.status);
            print_snapshots(&result.snapshots);
        }
    }
}

fn print_snapshots(snapshots: &[SolverSnapshot]) {
    for snapshot in snapshots {
        print!("iteration={}", snapshot.iteration);
        if let Some(learning_rate) = snapshot.learning_rate {
            print!(" learning_rate={learning_rate:.6}");
        }
        if let Some(average_loss) = snapshot.average_loss {
            print!(" average_loss={average_loss:.6}");
        }
        if let Some(evaluation_return) = snapshot.evaluation_return {
            print!(" evaluation_return={evaluation_return:.6}");
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlstudy::algorithms::{
        DqnConfig, MonteCarloConfig, PolicyIterationConfig, ReinforceConfig, TDConfig,
        TruncatedPolicyIterationConfig, ValueIterationConfig,
    };

    fn solver(arguments: &[&str]) -> Solver {
        let args = std::iter::once("rlstudy")
            .chain(arguments.iter().copied())
            .collect::<Vec<_>>();
        let parsed = Args::try_parse_from(args).unwrap();
        let Commands::Solve(solve) = parsed.command;
        solve.solver
    }

    #[test]
    fn cli_and_yaml_have_the_same_solver_defaults() {
        let Solver::VI(cli) = solver(&["solve", "vi"]) else {
            unreachable!()
        };
        assert_eq!(
            cli,
            yaml_serde::from_str::<ValueIterationConfig>("{}").unwrap()
        );

        let Solver::PI(cli) = solver(&["solve", "pi"]) else {
            unreachable!()
        };
        assert_eq!(
            cli,
            yaml_serde::from_str::<PolicyIterationConfig>("{}").unwrap()
        );

        let Solver::Tpi(cli) = solver(&["solve", "tpi"]) else {
            unreachable!()
        };
        assert_eq!(
            cli,
            yaml_serde::from_str::<TruncatedPolicyIterationConfig>("{}").unwrap()
        );

        let Solver::MC(cli) = solver(&["solve", "mc"]) else {
            unreachable!()
        };
        assert_eq!(cli, yaml_serde::from_str::<MonteCarloConfig>("{}").unwrap());

        let Solver::TD(cli) = solver(&["solve", "td"]) else {
            unreachable!()
        };
        assert_eq!(cli, yaml_serde::from_str::<TDConfig>("{}").unwrap());

        let Solver::Dqn(cli) = solver(&["solve", "dqn"]) else {
            unreachable!()
        };
        let cli: DqnConfig = cli.into();
        assert_eq!(cli, yaml_serde::from_str::<DqnConfig>("{}").unwrap());

        let Solver::Reinforce(cli) = solver(&["solve", "reinforce"]) else {
            unreachable!()
        };
        assert_eq!(cli, yaml_serde::from_str::<ReinforceConfig>("{}").unwrap());
    }

    #[test]
    fn gamma_is_part_of_solver_configuration() {
        let Solver::VI(config) = solver(&["solve", "vi", "--gamma", "0.5"]) else {
            unreachable!()
        };
        assert_eq!(config.gamma, 0.5);

        let config: DqnConfig = yaml_serde::from_str("gamma: 0.5").unwrap();
        assert_eq!(config.gamma, 0.5);
    }

    #[test]
    fn solve_seed_defaults_and_parses() {
        let Commands::Solve(defaults) = Args::try_parse_from(["rlstudy", "solve", "vi"])
            .unwrap()
            .command;
        assert_eq!(defaults.seed, 42);

        let Commands::Solve(configured) =
            Args::try_parse_from(["rlstudy", "solve", "--seed", "7", "vi"])
                .unwrap()
                .command;
        assert_eq!(configured.seed, 7);
    }

    #[test]
    fn system_sources_are_mutually_exclusive() {
        assert!(
            Args::try_parse_from([
                "rlstudy",
                "solve",
                "--grid-world",
                "grid.yaml",
                "--gym",
                "CartPole-v1",
                "vi",
            ])
            .is_err()
        );
    }

    #[test]
    fn checked_in_dqn_config_parses() {
        yaml_serde::from_str::<DqnConfig>(include_str!("../../settings/dqn_for_5x5.yaml")).unwrap();
        yaml_serde::from_str::<DqnConfig>(include_str!("../../settings/dqn_for_cartpole.yaml"))
            .unwrap();
    }
}
