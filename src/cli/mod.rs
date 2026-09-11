mod solvers;
mod systems;

use self::solvers::{SelectedSolver, Solver, build_solver};
use self::systems::{SelectedSystem, build_system};
use clap::{Parser, Subcommand};
use rand::{SeedableRng, rngs::StdRng};
use rlstudy::algorithms::{ModelBasedSolver, SolveResult};
use rlstudy::environment::TabularEnvironment;
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
    let solver = build_solver(&args.solve.solver)?;
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
    let solver = build_solver(&args.solver)?;
    let mut rng = StdRng::seed_from_u64(0);
    let result = run_solver(solver, &mut system, args.gamma, &mut rng)?;

    print_result(&args.system, &args.solver, &result, args.gamma);
    Ok(())
}

fn run_solver(
    solver: SelectedSolver,
    system: &mut SelectedSystem,
    gamma: f64,
    rng: &mut dyn rand::Rng,
) -> Result<SolveResult, Box<dyn std::error::Error>> {
    match &solver {
        SelectedSolver::ModelBased(solver) => {
            let model = system
                .as_model()
                .ok_or("model-based solver requires a model system")?;
            Ok(solver.solve(model, gamma)?)
        }
        SelectedSolver::ModelFree(solver) => {
            let result = match system {
                SelectedSystem::Model(model) => {
                    solver.solve(&mut TabularEnvironment::new(model)?, gamma, rng)
                }
                SelectedSystem::BlackBox(environment) => {
                    solver.solve(environment.as_mut(), gamma, rng)
                }
            };
            Ok(result?)
        }
    }
}

fn print_result(system: &str, solver: &Solver, result: &SolveResult, gamma: f64) {
    println!(
        "system={} solver={} gamma={:.3}",
        system,
        solver.name(),
        gamma
    );
    if let Some(v0) = result.value.first() {
        println!("value[0]={:.4}", v0);
    }
    println!("iterations={}", result.iterations);
    println!("termination={:?}", result.termination);
    println!("policy:");
    for s in 0..result.policy.num_states() {
        let action = (0..result.policy.num_actions())
            .max_by(|&a, &b| {
                result
                    .policy
                    .probability(s, a)
                    .total_cmp(&result.policy.probability(s, b))
            })
            .expect("policy has actions");
        println!(
            "  state {} [value {:.3}] -> action {}",
            s, result.value[s], action
        );
    }
}
