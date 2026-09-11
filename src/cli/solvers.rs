use clap::{Parser, Subcommand};
use rlstudy::algorithms::{
    ModelFreeSolver, MonteCarlo, StoppingCriterion, TD, TruncatedPolicyIteration,
};

#[derive(Subcommand, Clone)]
pub(super) enum Solver {
    VI(StoppingCriterionArgs),
    PI(StoppingCriterionArgs),
    TPI(TpiArgs),
    MC(MonteCarloArgs),
    TD(TDArgs),
}

impl Solver {
    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::VI(_) => "value-iteration",
            Self::PI(_) => "policy-iteration",
            Self::TPI(_) => "truncated-policy-iteration",
            Self::MC(_) => "monte-carlo",
            Self::TD(args) if args.q_learning => "q-learning",
            Self::TD(args) if args.expected => "expected-sarsa",
            Self::TD(_) => "sarsa",
        }
    }
}

#[derive(Parser, Clone)]
pub(super) struct StoppingCriterionArgs {
    #[arg(long)]
    tol: Option<f64>,
    #[arg(long)]
    max_iter: Option<usize>,
}

#[derive(Parser, Clone)]
pub(super) struct TpiArgs {
    #[arg(long = "outer-tol")]
    outer_tol: Option<f64>,
    #[arg(long = "outer-max-iter")]
    outer_max_iter: Option<usize>,
    #[arg(long = "inner-tol")]
    inner_tol: Option<f64>,
    #[arg(long = "inner-max-iter")]
    inner_max_iter: Option<usize>,
}

#[derive(Parser, Clone)]
pub(super) struct MonteCarloArgs {
    #[arg(long)]
    episodes: u64,
    #[arg(long)]
    episode_length: u64,
    #[arg(long)]
    epsilon: f64,
}

#[derive(Parser, Clone)]
pub(super) struct TDArgs {
    #[arg(long)]
    episodes: u64,
    #[arg(long)]
    episode_length: u64,
    #[arg(long)]
    epsilon: f64,
    #[arg(long)]
    alpha: f64,
    #[arg(long, conflicts_with = "q_learning")]
    expected: bool,
    #[arg(long, conflicts_with = "expected")]
    q_learning: bool,
}

pub(super) enum SelectedSolver {
    ModelBased(TruncatedPolicyIteration),
    ModelFree(Box<dyn ModelFreeSolver>),
}

pub(super) fn build_solver(solver: &Solver) -> Result<SelectedSolver, Box<dyn std::error::Error>> {
    match solver {
        Solver::VI(args) => Ok(SelectedSolver::ModelBased(TruncatedPolicyIteration::new(
            build_stopping_criterion(args)?,
            StoppingCriterion::iterations(1)?,
        ))),
        Solver::PI(args) => Ok(SelectedSolver::ModelBased(TruncatedPolicyIteration::new(
            build_stopping_criterion(args)?,
            StoppingCriterion::convergence(1e-6, 10_000)?,
        ))),
        Solver::TPI(args) => Ok(SelectedSolver::ModelBased(TruncatedPolicyIteration::new(
            stopping_criterion(args.outer_tol, args.outer_max_iter)?,
            stopping_criterion(args.inner_tol, args.inner_max_iter)?,
        ))),
        Solver::MC(args) => Ok(SelectedSolver::ModelFree(Box::new(MonteCarlo::new(
            args.episodes,
            args.episode_length,
            args.epsilon,
        )))),
        Solver::TD(args) => {
            let solver = if args.q_learning {
                TD::q_learning(args.episodes, args.episode_length, args.epsilon, args.alpha)?
            } else if args.expected {
                TD::expected(args.episodes, args.episode_length, args.epsilon, args.alpha)?
            } else {
                TD::new(args.episodes, args.episode_length, args.epsilon, args.alpha)?
            };
            Ok(SelectedSolver::ModelFree(Box::new(solver)))
        }
    }
}

fn build_stopping_criterion(args: &StoppingCriterionArgs) -> Result<StoppingCriterion, String> {
    stopping_criterion(args.tol, args.max_iter)
}

fn stopping_criterion(
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> Result<StoppingCriterion, String> {
    match (tolerance, max_iterations) {
        (Some(tolerance), Some(iterations)) => {
            StoppingCriterion::convergence(tolerance, iterations)
        }
        (Some(tolerance), None) => StoppingCriterion::convergence(tolerance, 10_000),
        (None, Some(iterations)) => StoppingCriterion::iterations(iterations),
        (None, None) => StoppingCriterion::convergence(1e-6, 10_000),
    }
}
