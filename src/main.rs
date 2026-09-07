use clap::{Parser, Subcommand};
use rand::{SeedableRng, rngs::StdRng};
use rlstudy::algorithms::{
    ModelBasedSolver, ModelFreeSolver, SolveResult, StoppingCriterion, TruncatedPolicyIteration,
};
use rlstudy::environment::{BlackBox, Environment};
use rlstudy::environments::grid_world;
use rlstudy::model::TabularModel;

#[derive(Subcommand, Clone)]
enum Solver {
    VI(StoppingCriterionArgs),
    PI(StoppingCriterionArgs),
    TPI(TpiArgs),
}

#[derive(Parser, Clone)]
struct StoppingCriterionArgs {
    #[arg(long)]
    pub tol: Option<f64>,
    #[arg(long)]
    pub max_iter: Option<usize>,
}

#[derive(Parser, Clone)]
struct TpiArgs {
    #[arg(long = "outer-tol")]
    pub outer_tol: Option<f64>,
    #[arg(long = "outer-max-iter")]
    pub outer_max_iter: Option<usize>,

    #[arg(long = "inner-tol")]
    pub inner_tol: Option<f64>,
    #[arg(long = "inner-max-iter")]
    pub inner_max_iter: Option<usize>,
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
    let mut rng = StdRng::seed_from_u64(0);
    let result = run_solver(solver, &mut system, gamma, &mut rng)?;

    print_result(&args.system, &args.solver, &result, gamma);
    Ok(())
}

enum SelectedSystem {
    Model(TabularModel),
    BlackBox(Box<dyn Environment<State = usize, Action = usize>>),
}

impl SelectedSystem {
    fn as_model(&self) -> Option<&TabularModel> {
        match self {
            Self::Model(model) => Some(model),
            Self::BlackBox(_) => None,
        }
    }

    fn as_environment_mut(
        &mut self,
    ) -> Option<&mut dyn Environment<State = usize, Action = usize>> {
        match self {
            Self::Model(_) => None,
            Self::BlackBox(system) => Some(system.as_mut()),
        }
    }
}

#[allow(dead_code)]
enum SelectedSolver {
    ModelBased(TruncatedPolicyIteration),
    ModelFree(Box<dyn ModelFreeSolver>),
}

fn build_system(name: &str) -> Result<SelectedSystem, Box<dyn std::error::Error>> {
    match name {
        "gridworld" => Ok(SelectedSystem::Model(grid_world(3)?)),
        "blackbox-line" => Ok(SelectedSystem::BlackBox(Box::new(BlackBox::new(
            3,
            2,
            0,
            |state, action| ((state + action).min(2), if state == 2 { 1.0 } else { 0.0 }),
        )))),
        other => Err(format!("unknown system: {}", other).into()),
    }
}

fn build_stopping_criterion(
    args: &StoppingCriterionArgs,
) -> Result<StoppingCriterion, Box<dyn std::error::Error>> {
    Ok(stopping_criterion_from_options(args.tol, args.max_iter)?)
}

fn stopping_criterion_from_options(
    tol: Option<f64>,
    max_iter: Option<usize>,
) -> Result<StoppingCriterion, String> {
    match (tol, max_iter) {
        (Some(t), Some(n)) => StoppingCriterion::convergence(t, n),
        (Some(t), None) => StoppingCriterion::convergence(t, 10_000),
        (None, Some(n)) => StoppingCriterion::iterations(n),
        (None, None) => StoppingCriterion::convergence(1e-6, 10_000),
    }
}

fn build_solver(args: &SolveArgs) -> Result<SelectedSolver, Box<dyn std::error::Error>> {
    match &args.solver {
        Solver::VI(sc_args) => {
            let sc = build_stopping_criterion(sc_args)?;
            Ok(SelectedSolver::ModelBased(TruncatedPolicyIteration::new(
                sc,
                StoppingCriterion::iterations(1)?,
            )))
        }
        Solver::PI(sc_args) => {
            let sc = build_stopping_criterion(sc_args)?;
            Ok(SelectedSolver::ModelBased(TruncatedPolicyIteration::new(
                sc,
                StoppingCriterion::convergence(1e-6, 10_000)?,
            )))
        }
        Solver::TPI(tpi_args) => {
            let outer =
                stopping_criterion_from_options(tpi_args.outer_tol, tpi_args.outer_max_iter)?;
            let inner =
                stopping_criterion_from_options(tpi_args.inner_tol, tpi_args.inner_max_iter)?;
            Ok(SelectedSolver::ModelBased(TruncatedPolicyIteration::new(
                outer, inner,
            )))
        }
    }
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
            let environment = system
                .as_environment_mut()
                .ok_or("model-free solver requires an environment")?;
            Ok(solver.solve(environment, gamma, rng)?)
        }
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
    println!("termination={:?}", result.termination);
    println!("policy:");
    for s in 0..result.policy.num_states().min(10) {
        for a in 0..result.policy.num_actions() {
            if result.policy.probability(s, a) > 0.999 {
                println!(
                    "  state {} [value {:.3}] -> action {}",
                    s, result.value[s], a
                );
            }
        }
    }
}
