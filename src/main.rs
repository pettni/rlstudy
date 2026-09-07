use clap::{Parser, Subcommand};
use rlstudy::examples::grid_world;
use rlstudy::model::Model;
use rlstudy::solvers::truncated_policy_iteration::{StoppingCriterion, TruncatedPolicyIteration};
use rlstudy::solvers::{ModelBasedSolver, ModelFreeSolver, SolveResult};
use rlstudy::system::{BlackBox, System};

#[derive(Subcommand, Clone)]
enum Solver {
    VI(StoppingCriterionArgs),
    PI(StoppingCriterionArgs),
    TPI(TpiArgs),
}

#[derive(Parser, Clone)]
struct StoppingCriterionArgs {
    #[arg(long, group = "criterion")]
    pub tol: Option<f64>,
    #[arg(long, group = "criterion")]
    pub max_iter: Option<u64>,
}

#[derive(Parser, Clone)]
struct TpiArgs {
    #[arg(long = "outer-tol", group = "outer-criterion")]
    pub outer_tol: Option<f64>,
    #[arg(long = "outer-max-iter", group = "outer-criterion")]
    pub outer_max_iter: Option<u64>,

    #[arg(long = "inner-tol", group = "inner-criterion")]
    pub inner_tol: Option<f64>,
    #[arg(long = "inner-max-iter", group = "inner-criterion")]
    pub inner_max_iter: Option<u64>,
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

    #[arg(short, long, default_value_t = 0.9)]
    pub gamma: f64,
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
    let gamma = args.gamma;
    let result = run_solver(solver, &mut system, gamma)?;

    print_result(&args.system, &args.solver, &result, gamma);
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

fn build_stopping_criterion(
    args: &StoppingCriterionArgs,
) -> Result<StoppingCriterion, Box<dyn std::error::Error>> {
    Ok(stopping_criterion_from_options(args.tol, args.max_iter))
}

fn stopping_criterion_from_options(tol: Option<f64>, max_iter: Option<u64>) -> StoppingCriterion {
    match (tol, max_iter) {
        (_, Some(n)) => StoppingCriterion::Iterations(n),
        (Some(t), None) => StoppingCriterion::Tolerance(t),
        (None, None) => StoppingCriterion::Tolerance(1e-6),
    }
}

fn build_solver(args: &SolveArgs) -> Result<SelectedSolver, Box<dyn std::error::Error>> {
    match &args.solver {
        Solver::VI(sc_args) => {
            let sc = build_stopping_criterion(sc_args)?;
            Ok(SelectedSolver::ModelBased(Box::new(
                TruncatedPolicyIteration::new(sc, StoppingCriterion::Iterations(1)),
            )))
        }
        Solver::PI(sc_args) => {
            let sc = build_stopping_criterion(sc_args)?;
            Ok(SelectedSolver::ModelBased(Box::new(
                TruncatedPolicyIteration::new(sc, StoppingCriterion::Tolerance(1e-6)),
            )))
        }
        Solver::TPI(tpi_args) => {
            let outer =
                stopping_criterion_from_options(tpi_args.outer_tol, tpi_args.outer_max_iter);
            let inner =
                stopping_criterion_from_options(tpi_args.inner_tol, tpi_args.inner_max_iter);
            Ok(SelectedSolver::ModelBased(Box::new(
                TruncatedPolicyIteration::new(outer, inner),
            )))
        }
    }
}

fn run_solver(
    mut solver: SelectedSolver,
    system: &mut SelectedSystem,
    gamma: f64,
) -> Result<SolveResult, Box<dyn std::error::Error>> {
    match &mut solver {
        SelectedSolver::ModelBased(solver) => {
            let model = system
                .as_model()
                .ok_or("model-based solver requires a model system")?;
            Ok(solver.solve(model, gamma))
        }
        SelectedSolver::ModelFree(solver) => Ok(solver.solve(system.as_system_mut(), gamma)),
    }
}

fn print_result(system: &str, solver: &Solver, result: &SolveResult, gamma: f64) {
    let solver_name = match solver {
        Solver::VI(_) => "value-iteration",
        Solver::PI(_) => "policy-iteration",
        Solver::TPI(_) => "truncated-policy-iteration",
    };
    println!(
        "system={} solver={} gamma={:.3}",
        system, solver_name, gamma
    );
    if let Some(v0) = result.value.first() {
        println!("value[0]={:.4}", v0);
    }
    println!("iterations={}", result.iterations);
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
