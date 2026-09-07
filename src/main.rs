use clap::{Parser, Subcommand};
use rlstudy::examples::grid_world;
use rlstudy::model::Model;
use rlstudy::solvers::truncated_policy_iteration::{StoppingCriterion, TruncatedPolicyIteration};
use rlstudy::solvers::{ModelBasedSolver, ModelFreeSolver, SolveResult};
use rlstudy::system::{BlackBox, System};

#[derive(Subcommand, Clone)]
enum Solver {
    Tpi(TpiArgs),
}

#[derive(Parser, Clone)]
struct TpiArgs {
    #[arg(short = 't', long, default_value_t = 1e-5)]
    pub tolerance: f64,

    #[arg(long, group = "stopping")]
    pub tol: Option<f64>,

    #[arg(long, group = "stopping")]
    pub max_iter: Option<u64>,

    #[arg(short = 'g', long, default_value_t = 0.95)]
    pub gamma: f64,
}

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
    #[arg(short, long, default_value = "gridworld")]
    pub system: String,

    #[command(subcommand)]
    pub solver: Solver,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.command {
        Commands::Solve(cmd_args) => solve_run(&cmd_args)?,
    }

    Ok(())
}

fn solve_run(args: &SolveArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut system = build_system(&args.system)?;
    let solver = build_solver(args)?;
    let result = run_solver(solver, &mut system)?;

    print_result(&args.system, &args.solver, &result);
    Ok(())
}

enum SelectedSystem {
    Model(Model),
    BlackBox(Box<dyn System>),
}

impl SelectedSystem {
    fn as_model(&self) -> Option<&Model> {
        match self {
            Self::Model(model) => Some(model),
            Self::BlackBox(_) => None,
        }
    }

    fn as_system_mut(&mut self) -> &mut dyn System {
        match self {
            Self::Model(model) => model,
            Self::BlackBox(system) => system.as_mut(),
        }
    }
}

#[allow(dead_code)]
enum SelectedSolver {
    ModelBased(Box<dyn ModelBasedSolver>),
    ModelFree(Box<dyn ModelFreeSolver>),
}

fn build_system(name: &str) -> Result<SelectedSystem, Box<dyn std::error::Error>> {
    match name {
        "gridworld" => Ok(SelectedSystem::Model(grid_world(3))),
        "blackbox-line" => Ok(SelectedSystem::BlackBox(Box::new(BlackBox::new(
            3,
            2,
            |state, action| ((state + action).min(2), if state == 2 { 1.0 } else { 0.0 }),
        )))),
        other => Err(format!("unknown system: {}", other).into()),
    }
}

fn build_solver(args: &SolveArgs) -> Result<SelectedSolver, Box<dyn std::error::Error>> {
    match &args.solver {
        Solver::Tpi(tpi_args) => {
            let sc = match (tpi_args.tol, tpi_args.max_iter) {
                (_, Some(n)) => StoppingCriterion::Iterations(n),
                (Some(t), None) => StoppingCriterion::Tolerance(t),
                (None, None) => StoppingCriterion::Tolerance(1e-6),
            };
            Ok(SelectedSolver::ModelBased(Box::new(
                TruncatedPolicyIteration::new(tpi_args.tolerance, sc, tpi_args.gamma),
            )))
        }
    }
}

fn run_solver(
    mut solver: SelectedSolver,
    system: &mut SelectedSystem,
) -> Result<SolveResult, Box<dyn std::error::Error>> {
    match &mut solver {
        SelectedSolver::ModelBased(solver) => {
            let model = system
                .as_model()
                .ok_or("model-based solver requires a model system")?;
            Ok(solver.solve(model))
        }
        SelectedSolver::ModelFree(solver) => Ok(solver.solve(system.as_system_mut())),
    }
}

fn print_result(system: &str, solver: &Solver, result: &SolveResult) {
    let solver_name = match solver {
        Solver::Tpi(_) => "tpi",
    };
    println!("system={} solver={}", system, solver_name);
    if let Some(v0) = result.value.first() {
        println!("value[0]={:.4}", v0);
    }
    println!("policy:");
    for s in 0..result.policy.num_states().min(10) {
        for a in 0..result.policy.num_actions() {
            if result.policy.prob(s, a) > 0.999 {
                println!(
                    "  state {} [value {:.3}] -> action {}",
                    s, result.value[s], a
                );
            }
        }
    }
}
