use clap::{Parser, Subcommand, ValueEnum};
use rlstudy::algorithms::{
    DeepQLearning, MonteCarlo, StoppingCriterion, TD, TargetUpdate, TruncatedPolicyIteration,
};
use rlstudy::evaluation::EvaluationConfig;
use rlstudy::policy::uniform::UniformPolicy;
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Subcommand, Clone)]
pub(super) enum Solver {
    VI(StoppingCriterionArgs),
    PI(StoppingCriterionArgs),
    TPI(TpiArgs),
    MC(MonteCarloArgs),
    TD(TDArgs),
    DQN(DqnArgs),
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
            Self::DQN(_) => "dqn",
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

#[derive(Parser, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct DqnArgs {
    /// YAML configuration file. Explicit CLI options override its values.
    #[arg(long)]
    #[serde(skip)]
    config: Option<PathBuf>,
    #[arg(long)]
    learning_rate: Option<f64>,
    #[arg(long)]
    final_learning_rate: Option<f64>,
    #[arg(long)]
    batch_size: Option<usize>,
    #[arg(long)]
    replay_capacity: Option<usize>,
    #[arg(long)]
    iterations: Option<usize>,
    #[arg(long, value_enum)]
    target_update: Option<TargetUpdateKind>,
    #[arg(long)]
    target_sync_steps: Option<usize>,
    #[arg(long)]
    target_update_rate: Option<f64>,
    #[arg(long)]
    hidden_dims: Option<usize>,
    #[arg(long)]
    evaluation_interval: Option<usize>,
    #[arg(long)]
    evaluation_episodes: Option<usize>,
    #[arg(long)]
    evaluation_max_steps: Option<usize>,
}

impl Default for DqnArgs {
    fn default() -> Self {
        Self {
            config: None,
            learning_rate: Some(0.001),
            final_learning_rate: Some(0.0001),
            batch_size: Some(64),
            replay_capacity: Some(1_000),
            iterations: Some(100_000),
            target_update: Some(TargetUpdateKind::Hard),
            target_sync_steps: Some(500),
            target_update_rate: Some(0.005),
            hidden_dims: Some(64),
            evaluation_interval: Some(100),
            evaluation_episodes: Some(10),
            evaluation_max_steps: Some(1_000),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum TargetUpdateKind {
    Hard,
    Soft,
}

pub(super) enum SelectedSolver {
    ModelBased(TruncatedPolicyIteration),
    MonteCarlo(MonteCarlo),
    TD(TD),
    DQN(DeepQLearning),
}

pub(super) fn build_solver(
    solver: &Solver,
    num_actions: usize,
) -> Result<SelectedSolver, Box<dyn std::error::Error>> {
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
        Solver::MC(args) => Ok(SelectedSolver::MonteCarlo(MonteCarlo::new(
            args.episodes,
            args.episode_length,
            args.epsilon,
        ))),
        Solver::TD(args) => {
            let solver = if args.q_learning {
                TD::q_learning(args.episodes, args.episode_length, args.epsilon, args.alpha)?
            } else if args.expected {
                TD::expected(args.episodes, args.episode_length, args.epsilon, args.alpha)?
            } else {
                TD::new(args.episodes, args.episode_length, args.epsilon, args.alpha)?
            };
            Ok(SelectedSolver::TD(solver))
        }
        Solver::DQN(args) => {
            let settings = resolved_dqn_args(args)?;
            Ok(SelectedSolver::DQN(DeepQLearning::new(
                settings.learning_rate.unwrap(),
                settings.final_learning_rate.unwrap(),
                settings.batch_size.unwrap(),
                settings.replay_capacity.unwrap(),
                settings.iterations.unwrap(),
                match settings.target_update.unwrap() {
                    TargetUpdateKind::Hard => TargetUpdate::Hard {
                        steps: settings.target_sync_steps.unwrap(),
                    },
                    TargetUpdateKind::Soft => TargetUpdate::Soft {
                        rate: settings.target_update_rate.unwrap(),
                    },
                },
                settings.hidden_dims.unwrap(),
                settings.evaluation_interval.unwrap(),
                EvaluationConfig::new(
                    settings.evaluation_episodes.unwrap(),
                    settings.evaluation_max_steps.unwrap(),
                )?,
                Box::new(UniformPolicy::new(num_actions)),
            )?))
        }
    }
}

fn resolved_dqn_args(args: &DqnArgs) -> Result<DqnArgs, Box<dyn std::error::Error>> {
    let mut settings = if let Some(path) = &args.config {
        let contents = fs::read_to_string(path)
            .map_err(|error| format!("failed to read DQN config {}: {error}", path.display()))?;
        yaml_serde::from_str(&contents)
            .map_err(|error| format!("failed to parse DQN config {}: {error}", path.display()))?
    } else {
        DqnArgs::default()
    };

    settings.learning_rate = args.learning_rate.or(settings.learning_rate);
    settings.final_learning_rate = args.final_learning_rate.or(settings.final_learning_rate);
    settings.batch_size = args.batch_size.or(settings.batch_size);
    settings.replay_capacity = args.replay_capacity.or(settings.replay_capacity);
    settings.iterations = args.iterations.or(settings.iterations);
    settings.target_update = args.target_update.or(settings.target_update);
    settings.target_sync_steps = args.target_sync_steps.or(settings.target_sync_steps);
    settings.target_update_rate = args.target_update_rate.or(settings.target_update_rate);
    settings.hidden_dims = args.hidden_dims.or(settings.hidden_dims);
    settings.evaluation_interval = args.evaluation_interval.or(settings.evaluation_interval);
    settings.evaluation_episodes = args.evaluation_episodes.or(settings.evaluation_episodes);
    settings.evaluation_max_steps = args.evaluation_max_steps.or(settings.evaluation_max_steps);
    Ok(settings)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dqn_yaml_uses_defaults_for_missing_fields() {
        let settings: DqnArgs = yaml_serde::from_str("learning_rate: 0.1").unwrap();

        assert_eq!(settings.learning_rate, Some(0.1));
        assert_eq!(settings.replay_capacity, Some(1_000));
        assert_eq!(settings.evaluation_episodes, Some(10));
        assert_eq!(settings.target_update, Some(TargetUpdateKind::Hard));
    }

    #[test]
    fn dqn_yaml_selects_soft_target_updates() {
        let settings: DqnArgs =
            yaml_serde::from_str("target_update: soft\ntarget_update_rate: 0.01").unwrap();

        assert_eq!(settings.target_update, Some(TargetUpdateKind::Soft));
        assert_eq!(settings.target_update_rate, Some(0.01));
    }
}
