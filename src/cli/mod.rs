mod solvers;
mod systems;

use self::solvers::{SelectedSolver, Solver, build_solver};
use self::systems::{SelectedSystem, build_system};
use clap::{Parser, Subcommand};
use ndarray::{Array1, Array2};
use rand::{SeedableRng, rngs::StdRng};
use rlstudy::algorithms::{ModelBasedSolver, ModelFreeSolver, SoftActorCritic, SolveResult};
use rlstudy::environment::{Environment, FeatureEnvironment, TabularEnvironment};
use rlstudy::environments::{
    ContinuousGymEnvironment, DiscreteGymEnvironment, GymEnvironment, GymOptions,
};
use rlstudy::evaluation::{EvaluationStats, fixed_policy_values};
use rlstudy::model::TabularModel;
use rlstudy::policy::Policy;
use rlstudy::reporting::{HistoryRow, Reporter, ReporterConfig};
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
    Solve(SolveArgs),
}

#[derive(Parser)]
struct SolveArgs {
    #[arg(long, conflicts_with = "gym")]
    grid_world: Option<PathBuf>,
    #[arg(long, conflicts_with = "grid_world")]
    gym: Option<String>,
    /// Override Gymnasium's episode limit (independent of evaluation-max-steps).
    #[arg(long, requires = "gym")]
    gym_max_steps: Option<usize>,
    /// Rescale bounded Gymnasium observations to [-1, 1].
    #[arg(long, requires = "gym")]
    normalize_observations: bool,
    /// Save selected evaluation episodes as MP4 files.
    #[arg(long, requires = "gym")]
    evaluation_video_dir: Option<PathBuf>,
    /// Record every Nth evaluation episode.
    #[arg(long, requires = "evaluation_video_dir")]
    evaluation_video_every: Option<usize>,
    #[arg(long)]
    config: Option<PathBuf>,
    #[command(flatten)]
    reporting: ReporterConfig,
    #[arg(long, default_value_t = 0.9)]
    value_gamma: f64,
    #[command(subcommand)]
    solver: Solver,
    #[arg(long)]
    json: bool,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

pub(crate) fn run() -> Result<(), Box<dyn std::error::Error>> {
    match Args::parse().command {
        Commands::Solve(args) => solve(&args),
    }
}

fn solve(args: &SolveArgs) -> Result<(), Box<dyn std::error::Error>> {
    args.reporting.validate()?;
    if args.evaluation_video_every == Some(0) {
        return Err("evaluation video interval must be positive".into());
    }
    if !args.value_gamma.is_finite() || !(0.0..1.0).contains(&args.value_gamma) {
        return Err("value gamma must be finite and in [0, 1)".into());
    }
    let gym_options = GymOptions {
        max_episode_steps: args.gym_max_steps,
        normalize_observations: args.normalize_observations,
        video_folder: None,
        video_every: 1,
    };
    let evaluation_gym_options = GymOptions {
        max_episode_steps: args.gym_max_steps,
        normalize_observations: args.normalize_observations,
        video_folder: args.evaluation_video_dir.clone(),
        video_every: args.evaluation_video_every.unwrap_or(1),
    };
    let mut system = build_system(
        args.grid_world.as_deref(),
        args.gym.as_deref(),
        &gym_options,
    )?;
    let (solver, parameters) = build_solver(&args.solver, args.config.as_deref())?;
    let mut rng = StdRng::seed_from_u64(args.seed);
    let (result, final_evaluation) = run_solver(
        solver,
        &mut system,
        &mut rng,
        &args.reporting,
        args.gym.as_deref(),
        &evaluation_gym_options,
        args.seed ^ 0x6a09_e667_f3bc_c909,
    )?;
    let discrete_model = system
        .as_model()
        .map(|model| build_discrete_model(model, &result, args.value_gamma, &mut rng))
        .transpose()?;

    if args.json {
        let grid_world_definition = args
            .grid_world
            .as_deref()
            .and_then(|path| path.to_str())
            .or_else(|| system.as_model().map(|_| "grid_worlds/zhao_3x3.yaml"));
        serde_json::to_writer(
            std::io::stdout(),
            &VisualizationResult {
                grid_world_definition,
                solver: args.solver.as_str(),
                seed: args.seed,
                parameters: &parameters,
                reporting: &args.reporting,
                history: result.history(),
                final_evaluation: &final_evaluation,
                discrete_model,
            },
        )?;
        println!();
    } else {
        println!(
            "solver={} seed={} status={:?}",
            args.solver.as_str(),
            args.seed,
            result.status()
        );
        println!(
            "final_return={:.3}±{:.3} final_length={:.2}",
            final_evaluation.return_mean,
            final_evaluation.return_std,
            final_evaluation.episode_length_mean
        );
        if let Some(model) = discrete_model {
            for (state, (action, value)) in model.actions.iter().zip(&model.values).enumerate() {
                println!("state {state} [value {value:.3}] -> action {action}");
            }
        }
    }
    Ok(())
}

fn run_solver(
    solver: SelectedSolver,
    system: &mut SelectedSystem,
    rng: &mut dyn rand::Rng,
    reporting: &ReporterConfig,
    gym_id: Option<&str>,
    gym_options: &GymOptions,
    evaluation_seed: u64,
) -> Result<(SelectedResult, EvaluationStats), Box<dyn std::error::Error>> {
    match solver {
        SelectedSolver::ModelBased(solver) => {
            let model = system
                .as_model()
                .ok_or("model-based solver requires a model system")?;
            let mut evaluation_environment = TabularEnvironment::new(model)?;
            let mut reporter = Reporter::new(
                reporting.clone(),
                evaluation_seed,
                Some(&mut evaluation_environment),
            );
            let result = solver.solve_with_reporter(model, &mut reporter)?;
            let stats = reporter.final_evaluation(result.policy.as_ref());
            Ok((
                SelectedResult::Tabular(result, reporter.into_history()),
                stats,
            ))
        }
        SelectedSolver::MonteCarlo(solver) => {
            let SelectedSystem::Model(model) = system else {
                return Err("Monte Carlo requires tabular states".into());
            };
            let mut environment = TabularEnvironment::new(model)?;
            let mut evaluation_environment = TabularEnvironment::new(model)?;
            let (result, history, stats) = run_model_free(
                &solver,
                &mut environment,
                &mut evaluation_environment,
                rng,
                reporting,
                evaluation_seed,
            )?;
            Ok((SelectedResult::Tabular(result, history), stats))
        }
        SelectedSolver::TD(solver) => {
            let SelectedSystem::Model(model) = system else {
                return Err("TD requires tabular states".into());
            };
            let mut environment = TabularEnvironment::new(model)?;
            let mut evaluation_environment = TabularEnvironment::new(model)?;
            let (result, history, stats) = run_model_free(
                &solver,
                &mut environment,
                &mut evaluation_environment,
                rng,
                reporting,
                evaluation_seed,
            )?;
            Ok((SelectedResult::Tabular(result, history), stats))
        }
        SelectedSolver::Dqn(solver) => {
            let (result, history, stats) = match system {
                SelectedSystem::Model(model) => {
                    let mut environment = FeatureEnvironment::new(model)?;
                    let mut evaluation_environment = FeatureEnvironment::new(model)?;
                    run_model_free(
                        &solver,
                        &mut environment,
                        &mut evaluation_environment,
                        rng,
                        reporting,
                        evaluation_seed,
                    )?
                }
                SelectedSystem::DiscreteGym(environment) => {
                    let mut evaluation_environment =
                        discrete_gym(gym_id.expect("Gym ID is present"), gym_options)?;
                    run_model_free(
                        &solver,
                        environment,
                        &mut evaluation_environment,
                        rng,
                        reporting,
                        evaluation_seed,
                    )?
                }
                SelectedSystem::ContinuousGym(_) => {
                    return Err("DQN requires a discrete action space".into());
                }
            };
            Ok((SelectedResult::Neural(result, history), stats))
        }
        SelectedSolver::Ddpg(solver) | SelectedSolver::Td3(solver) => {
            let SelectedSystem::ContinuousGym(environment) = system else {
                return Err(
                    "Deterministic policy gradient requires a continuous-action Gym environment"
                        .into(),
                );
            };
            let mut evaluation_environment =
                continuous_gym(gym_id.expect("Gym ID is present"), gym_options)?;
            let (result, history, stats) = run_model_free(
                &solver,
                environment,
                &mut evaluation_environment,
                rng,
                reporting,
                evaluation_seed,
            )?;
            Ok((SelectedResult::Continuous(result, history), stats))
        }
        SelectedSolver::Sac(config) => match system {
            SelectedSystem::Model(model) => {
                let solver = SoftActorCritic::<usize>::new(config)?;
                let mut environment = FeatureEnvironment::new(model)?;
                let mut evaluation_environment = FeatureEnvironment::new(model)?;
                let (result, history, stats) = run_model_free(
                    &solver,
                    &mut environment,
                    &mut evaluation_environment,
                    rng,
                    reporting,
                    evaluation_seed,
                )?;
                Ok((SelectedResult::Neural(result, history), stats))
            }
            SelectedSystem::DiscreteGym(environment) => {
                let solver = SoftActorCritic::<usize>::new(config)?;
                let mut evaluation_environment =
                    discrete_gym(gym_id.expect("Gym ID is present"), gym_options)?;
                let (result, history, stats) = run_model_free(
                    &solver,
                    environment,
                    &mut evaluation_environment,
                    rng,
                    reporting,
                    evaluation_seed,
                )?;
                Ok((SelectedResult::Neural(result, history), stats))
            }
            SelectedSystem::ContinuousGym(environment) => {
                let solver = SoftActorCritic::<Array1<f64>>::new(config)?;
                let mut evaluation_environment =
                    continuous_gym(gym_id.expect("Gym ID is present"), gym_options)?;
                let (result, history, stats) = run_model_free(
                    &solver,
                    environment,
                    &mut evaluation_environment,
                    rng,
                    reporting,
                    evaluation_seed,
                )?;
                Ok((SelectedResult::Continuous(result, history), stats))
            }
        },
        SelectedSolver::Reinforce(solver) => {
            let (result, history, stats) = match system {
                SelectedSystem::Model(model) => {
                    let mut environment = FeatureEnvironment::new(model)?;
                    let mut evaluation_environment = FeatureEnvironment::new(model)?;
                    run_model_free(
                        &solver,
                        &mut environment,
                        &mut evaluation_environment,
                        rng,
                        reporting,
                        evaluation_seed,
                    )?
                }
                SelectedSystem::DiscreteGym(environment) => {
                    let mut evaluation_environment =
                        discrete_gym(gym_id.expect("Gym ID is present"), gym_options)?;
                    run_model_free(
                        &solver,
                        environment,
                        &mut evaluation_environment,
                        rng,
                        reporting,
                        evaluation_seed,
                    )?
                }
                SelectedSystem::ContinuousGym(_) => {
                    return Err("REINFORCE requires a discrete action space".into());
                }
            };
            Ok((SelectedResult::Neural(result, history), stats))
        }
        SelectedSolver::StochasticPolicyGradient(solver) => match system {
            SelectedSystem::Model(model) => {
                let solver = solver.build()?;
                let mut environment = FeatureEnvironment::new(model)?;
                let mut evaluation_environment = FeatureEnvironment::new(model)?;
                let (result, history, stats) = run_model_free(
                    &solver,
                    &mut environment,
                    &mut evaluation_environment,
                    rng,
                    reporting,
                    evaluation_seed,
                )?;
                Ok((SelectedResult::Neural(result, history), stats))
            }
            SelectedSystem::DiscreteGym(environment) => {
                let solver = solver.build()?;
                let mut evaluation_environment =
                    discrete_gym(gym_id.expect("Gym ID is present"), gym_options)?;
                let (result, history, stats) = run_model_free(
                    &solver,
                    environment,
                    &mut evaluation_environment,
                    rng,
                    reporting,
                    evaluation_seed,
                )?;
                Ok((SelectedResult::Neural(result, history), stats))
            }
            SelectedSystem::ContinuousGym(environment) => {
                let solver = solver.build()?;
                let mut evaluation_environment =
                    continuous_gym(gym_id.expect("Gym ID is present"), gym_options)?;
                let (result, history, stats) = run_model_free(
                    &solver,
                    environment,
                    &mut evaluation_environment,
                    rng,
                    reporting,
                    evaluation_seed,
                )?;
                Ok((SelectedResult::Continuous(result, history), stats))
            }
        },
        SelectedSolver::ActorCritic(solver) => {
            let (result, history, stats) = match system {
                SelectedSystem::Model(model) => {
                    let mut environment = FeatureEnvironment::new(model)?;
                    let mut evaluation_environment = FeatureEnvironment::new(model)?;
                    run_model_free(
                        &solver,
                        &mut environment,
                        &mut evaluation_environment,
                        rng,
                        reporting,
                        evaluation_seed,
                    )?
                }
                SelectedSystem::DiscreteGym(environment) => {
                    let mut evaluation_environment =
                        discrete_gym(gym_id.expect("Gym ID is present"), gym_options)?;
                    run_model_free(
                        &solver,
                        environment,
                        &mut evaluation_environment,
                        rng,
                        reporting,
                        evaluation_seed,
                    )?
                }
                SelectedSystem::ContinuousGym(_) => {
                    return Err("actor-critic requires a discrete action space".into());
                }
            };
            Ok((SelectedResult::Neural(result, history), stats))
        }
        SelectedSolver::ApproximateTD(solver) => {
            let (result, history, stats) = match system {
                SelectedSystem::Model(model) => {
                    let mut environment = FeatureEnvironment::new(model)?;
                    let mut evaluation_environment = FeatureEnvironment::new(model)?;
                    run_model_free(
                        &solver,
                        &mut environment,
                        &mut evaluation_environment,
                        rng,
                        reporting,
                        evaluation_seed,
                    )?
                }
                SelectedSystem::DiscreteGym(environment) => {
                    let mut evaluation_environment =
                        discrete_gym(gym_id.expect("Gym ID is present"), gym_options)?;
                    run_model_free(
                        &solver,
                        environment,
                        &mut evaluation_environment,
                        rng,
                        reporting,
                        evaluation_seed,
                    )?
                }
                SelectedSystem::ContinuousGym(_) => {
                    return Err("approximate TD requires a discrete action space".into());
                }
            };
            Ok((SelectedResult::Neural(result, history), stats))
        }
    }
}

fn discrete_gym(
    id: &str,
    options: &GymOptions,
) -> Result<DiscreteGymEnvironment, Box<dyn std::error::Error>> {
    match GymEnvironment::new(id, options)? {
        GymEnvironment::Discrete(environment) => Ok(environment),
        GymEnvironment::Continuous(_) => Err("expected a discrete-action Gym environment".into()),
    }
}

fn continuous_gym(
    id: &str,
    options: &GymOptions,
) -> Result<ContinuousGymEnvironment, Box<dyn std::error::Error>> {
    match GymEnvironment::new(id, options)? {
        GymEnvironment::Continuous(environment) => Ok(environment),
        GymEnvironment::Discrete(_) => Err("expected a continuous-action Gym environment".into()),
    }
}

fn run_model_free<S: ModelFreeSolver>(
    solver: &S,
    environment: &mut dyn Environment<Observation = S::Observation, Action = S::Action>,
    evaluation_environment: &mut dyn Environment<Observation = S::Observation, Action = S::Action>,
    rng: &mut dyn rand::Rng,
    reporting: &ReporterConfig,
    evaluation_seed: u64,
) -> Result<
    (
        SolveResult<S::Observation, S::Action>,
        Vec<HistoryRow>,
        EvaluationStats,
    ),
    String,
> {
    let mut reporter = Reporter::new(
        reporting.clone(),
        evaluation_seed,
        Some(evaluation_environment),
    );
    let result = solver.solve_with_reporter(environment, rng, &mut reporter)?;
    let stats = reporter.final_evaluation(result.policy.as_ref());
    Ok((result, reporter.into_history(), stats))
}

enum SelectedResult {
    Tabular(SolveResult<usize>, Vec<HistoryRow>),
    Neural(SolveResult<Array1<f64>>, Vec<HistoryRow>),
    Continuous(SolveResult<Array1<f64>, Array1<f64>>, Vec<HistoryRow>),
}

impl SelectedResult {
    fn history(&self) -> &[HistoryRow] {
        match self {
            Self::Tabular(_, history) | Self::Neural(_, history) | Self::Continuous(_, history) => {
                history
            }
        }
    }

    fn status(&self) -> rlstudy::algorithms::SolveStatus {
        match self {
            Self::Tabular(result, _) => result.status,
            Self::Neural(result, _) => result.status,
            Self::Continuous(result, _) => result.status,
        }
    }
}

#[derive(Serialize)]
struct DiscreteModelResult {
    value_gamma: f64,
    actions: Vec<usize>,
    values: Vec<f64>,
}

fn build_discrete_model(
    model: &TabularModel,
    result: &SelectedResult,
    gamma: f64,
    rng: &mut dyn rand::Rng,
) -> Result<DiscreteModelResult, String> {
    let mut probabilities = Array2::zeros((model.states().len(), model.actions().len()));
    for state in 0..model.states().len() {
        let row = match result {
            SelectedResult::Tabular(result, _) => {
                action_probabilities(result.policy.as_ref(), &state, model.actions().len(), rng)
            }
            SelectedResult::Neural(result, _) => {
                let features = model
                    .state_features(state)
                    .ok_or("model state features are required for neural policies")?
                    .to_owned();
                action_probabilities(
                    result.policy.as_ref(),
                    &features,
                    model.actions().len(),
                    rng,
                )
            }
            SelectedResult::Continuous(_, _) => {
                return Err("continuous policy cannot be rendered on a discrete model".into());
            }
        };
        probabilities.row_mut(state).assign(&row);
    }
    let actions = probabilities
        .outer_iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(action, _)| action)
                .expect("model has actions")
        })
        .collect();
    let values = fixed_policy_values(model, &probabilities, gamma)?;
    Ok(DiscreteModelResult {
        value_gamma: gamma,
        actions,
        values,
    })
}

fn action_probabilities<Observation>(
    policy: &dyn Policy<Observation = Observation, Action = usize>,
    observation: &Observation,
    num_actions: usize,
    rng: &mut dyn rand::Rng,
) -> Array1<f64> {
    let mut probabilities = Array1::zeros(num_actions);
    probabilities[policy.sample_action(observation, rng)] = 1.0;
    probabilities
}

#[derive(Serialize)]
struct VisualizationResult<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    grid_world_definition: Option<&'a str>,
    solver: &'a str,
    seed: u64,
    parameters: &'a serde_json::Value,
    reporting: &'a ReporterConfig,
    history: &'a [HistoryRow],
    final_evaluation: &'a EvaluationStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    discrete_model: Option<DiscreteModelResult>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlstudy::algorithms::{
        ActorCriticConfig, ApproximateTDConfig, DdpgConfig, DqnConfig, MonteCarloConfig, PpoConfig,
        ReinforceConfig, TDConfig, ValueIterationConfig, VanillaPolicyGradientConfig,
    };

    fn solver(arguments: &[&str]) -> Solver {
        let parsed =
            Args::try_parse_from(std::iter::once("rlstudy").chain(arguments.iter().copied()))
                .unwrap();
        let Commands::Solve(args) = parsed.command;
        args.solver
    }

    #[test]
    fn cli_and_yaml_have_the_same_solver_defaults() {
        let Solver::VI(config) = solver(&["solve", "vi"]) else {
            unreachable!()
        };
        let config: ValueIterationConfig = config.into();
        assert_eq!(
            config,
            yaml_serde::from_str::<ValueIterationConfig>("{}").unwrap()
        );
        let Solver::MC(config) = solver(&["solve", "mc"]) else {
            unreachable!()
        };
        let config: MonteCarloConfig = config.into();
        assert_eq!(
            config,
            yaml_serde::from_str::<MonteCarloConfig>("{}").unwrap()
        );
        let Solver::TD(config) = solver(&["solve", "td"]) else {
            unreachable!()
        };
        let config: TDConfig = config.into();
        assert_eq!(config, yaml_serde::from_str::<TDConfig>("{}").unwrap());
        let Solver::Dqn(config) = solver(&["solve", "dqn"]) else {
            unreachable!()
        };
        let config: DqnConfig = config.into();
        assert_eq!(config, yaml_serde::from_str::<DqnConfig>("{}").unwrap());
        let Solver::Ddpg(config) = solver(&["solve", "ddpg"]) else {
            unreachable!()
        };
        let config: DdpgConfig = config.into();
        assert_eq!(config, yaml_serde::from_str::<DdpgConfig>("{}").unwrap());
        let Solver::Reinforce(config) = solver(&["solve", "reinforce"]) else {
            unreachable!()
        };
        let config: ReinforceConfig = config.into();
        assert_eq!(
            config,
            yaml_serde::from_str::<ReinforceConfig>("{}").unwrap()
        );
        let Solver::Vpg(config) = solver(&["solve", "vpg"]) else {
            unreachable!()
        };
        let config: VanillaPolicyGradientConfig = config.into();
        assert_eq!(
            config,
            yaml_serde::from_str::<VanillaPolicyGradientConfig>("{}").unwrap()
        );
        let Solver::Ppo(config) = solver(&["solve", "ppo"]) else {
            unreachable!()
        };
        let config: PpoConfig = config.into();
        assert_eq!(config, yaml_serde::from_str::<PpoConfig>("{}").unwrap());
        let Solver::ActorCritic(config) = solver(&["solve", "actor-critic"]) else {
            unreachable!()
        };
        let config: ActorCriticConfig = config.into();
        assert_eq!(
            config,
            yaml_serde::from_str::<ActorCriticConfig>("{}").unwrap()
        );
        let Solver::ApproximateTD(config) = solver(&["solve", "approximate-td"]) else {
            unreachable!()
        };
        let config: ApproximateTDConfig = config.into();
        assert_eq!(
            config,
            yaml_serde::from_str::<ApproximateTDConfig>("{}").unwrap()
        );
    }

    #[test]
    fn reporting_options_parse_at_solve_level() {
        let Commands::Solve(args) = Args::try_parse_from([
            "rlstudy",
            "solve",
            "--evaluation-frequency",
            "7",
            "--evaluation-episodes",
            "3",
            "--evaluation-max-steps",
            "20",
            "--gym",
            "CartPole-v1",
            "--evaluation-video-dir",
            "videos",
            "--evaluation-video-every",
            "3",
            "--value-gamma",
            "0.5",
            "reinforce",
        ])
        .unwrap()
        .command;
        assert_eq!(args.reporting.evaluation_frequency, Some(7));
        assert_eq!(args.reporting.evaluation_episodes, 3);
        assert_eq!(args.reporting.evaluation_max_steps, 20);
        assert_eq!(args.evaluation_video_dir, Some(PathBuf::from("videos")));
        assert_eq!(args.evaluation_video_every, Some(3));
        assert_eq!(args.value_gamma, 0.5);
    }
}
