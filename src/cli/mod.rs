mod solvers;
mod systems;

use self::solvers::{SelectedSolver, Solver, build_solver};
use self::systems::{SelectedSystem, build_system};
use clap::{Parser, Subcommand};
use ndarray::Array1;
use rand::{SeedableRng, rngs::StdRng};
use rlstudy::algorithms::{
    DqnSnapshot, ModelBasedSolver, ModelFreeSolver, SolveResult, TabularSolveResult,
};
use rlstudy::environment::{FeatureEnvironment, TabularEnvironment};
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
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
    /// Write solver value snapshots as CSV.
    Trace(TraceArgs),
}

#[derive(Parser)]
struct SolveArgs {
    /// Built-in system name or path to a grid-world YAML file.
    #[arg(short, long, default_value = "gridworld")]
    system: String,

    #[command(subcommand)]
    solver: Solver,

    #[arg(short, long, default_value_t = 0.9)]
    gamma: f64,

    /// Write values and greedy policy actions as JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Parser)]
struct TraceArgs {
    #[command(flatten)]
    solve: SolveArgs,

    #[arg(short, long)]
    output: PathBuf,
}

pub(crate) fn run() -> Result<(), Box<dyn std::error::Error>> {
    match Args::parse().command {
        Commands::Solve(args) => solve(&args),
        Commands::Trace(args) => trace(&args),
    }
}

fn trace(args: &TraceArgs) -> Result<(), Box<dyn std::error::Error>> {
    let system = build_system(&args.solve.system)?;
    let solver = build_solver(&args.solve.solver, system.num_actions())?;
    let model = system.as_model().ok_or("tracing requires a model system")?;
    let SelectedSolver::ModelBased(solver) = solver else {
        return Err("tracing requires a model-based solver".into());
    };
    let trace = solver.solve_with_trace(model, args.solve.gamma)?;
    let name = args.solve.solver.name();
    let mut output = BufWriter::new(File::create(&args.output)?);

    writeln!(
        output,
        "solver,policy_iteration,evaluation_sweep,state,value"
    )?;
    for snapshot in trace.snapshots {
        for (state, value) in snapshot.value.into_iter().enumerate() {
            writeln!(
                output,
                "{},{},{},{},{}",
                name, snapshot.policy_iteration, snapshot.evaluation_sweep, state, value
            )?;
        }
    }
    Ok(())
}

fn solve(args: &SolveArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut system = build_system(&args.system)?;
    let solver = build_solver(&args.solver, system.num_actions())?;
    let mut rng = StdRng::seed_from_u64(0);
    let result = run_solver(solver, &mut system, args.gamma, &mut rng)?;

    if args.json {
        write_json_result(&args.system, &args.solver, &system, &result, args.gamma)?;
    } else {
        print_result(&args.system, &args.solver, result, args.gamma);
    }
    Ok(())
}

fn run_solver(
    solver: SelectedSolver,
    system: &mut SelectedSystem,
    gamma: f64,
    rng: &mut dyn rand::Rng,
) -> Result<SelectedResult, Box<dyn std::error::Error>> {
    match &solver {
        SelectedSolver::ModelBased(solver) => {
            let model = system
                .as_model()
                .ok_or("model-based solver requires a model system")?;
            Ok(SelectedResult::Tabular(solver.solve(model, gamma)?))
        }
        SelectedSolver::MonteCarlo(solver) => {
            let result = match system {
                SelectedSystem::Model(model) => {
                    solver.solve(&mut TabularEnvironment::new(model)?, gamma, rng)
                }
                SelectedSystem::BlackBox(environment) => {
                    solver.solve(environment.as_mut(), gamma, rng)
                }
            };
            Ok(SelectedResult::Tabular(result?))
        }
        SelectedSolver::TD(solver) => {
            let result = match system {
                SelectedSystem::Model(model) => {
                    solver.solve(&mut TabularEnvironment::new(model)?, gamma, rng)
                }
                SelectedSystem::BlackBox(environment) => {
                    solver.solve(environment.as_mut(), gamma, rng)
                }
            };
            Ok(SelectedResult::Tabular(result?))
        }
        SelectedSolver::DQN(solver) => {
            let model = system
                .as_model()
                .ok_or("DQN requires a model with state features")?;
            let mut environment = FeatureEnvironment::new(model)?;
            Ok(SelectedResult::DQN(solver.solve(
                &mut environment,
                gamma,
                rng,
            )?))
        }
    }
}

enum SelectedResult {
    Tabular(TabularSolveResult),
    DQN(SolveResult<Array1<f64>, Vec<DqnSnapshot>>),
}

#[derive(Serialize)]
struct VisualizationResult<'a> {
    grid_world_definition: &'a str,
    solver: &'a str,
    gamma: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    state_values: Option<&'a [f64]>,
    policy_actions: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    training: Option<&'a [DqnSnapshot]>,
}

fn write_json_result(
    system_name: &str,
    solver: &Solver,
    system: &SelectedSystem,
    result: &SelectedResult,
    gamma: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = system
        .as_model()
        .ok_or("JSON output requires a grid-world model")?;
    let grid_world_definition = match system_name {
        "gridworld" => "grid_worlds/zhao_3x3.yaml",
        path => path,
    };

    let (state_values, policy_actions, training) = match result {
        SelectedResult::Tabular(result) => (
            Some(result.output.as_slice()),
            (0..result.output.len())
                .map(|state| result.policy.greedy_action(&state))
                .collect(),
            None,
        ),
        SelectedResult::DQN(result) => (
            None,
            (0..model.states().len())
                .map(|state| {
                    let observation = model
                        .state_features(state)
                        .expect("DQN requires model state features")
                        .to_owned();
                    result.policy.greedy_action(&observation)
                })
                .collect(),
            Some(result.output.as_slice()),
        ),
    };

    serde_json::to_writer(
        std::io::stdout(),
        &VisualizationResult {
            grid_world_definition,
            solver: solver.name(),
            gamma,
            state_values,
            policy_actions,
            training,
        },
    )?;
    println!();
    Ok(())
}

fn print_result(system: &str, solver: &Solver, result: SelectedResult, gamma: f64) {
    println!(
        "system={} solver={} gamma={:.3}",
        system,
        solver.name(),
        gamma
    );
    match result {
        SelectedResult::Tabular(result) => {
            if let Some(v0) = result.output.first() {
                println!("value[0]={:.4}", v0);
            }
            println!("iterations={}", result.status.iterations());
            println!("status={:?}", result.status);
            println!("policy:");
            for s in 0..result.output.len() {
                let action = result.policy.greedy_action(&s);
                println!(
                    "  state {} [value {:.3}] -> action {}",
                    s, result.output[s], action
                );
            }
        }
        SelectedResult::DQN(result) => {
            println!("iterations={}", result.status.iterations());
            println!("status={:?}", result.status);
            for snapshot in result.output {
                println!(
                    "iteration={} learning_rate={:.6} average_loss={:.6} evaluation_reward={:.6}",
                    snapshot.iteration,
                    snapshot.learning_rate,
                    snapshot.average_loss,
                    snapshot.evaluation_reward
                );
            }
        }
    }
}
