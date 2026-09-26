mod actor_critic;
mod approximate_td;
mod ddpg;
mod dqn;
mod monte_carlo;
mod policy_iteration;
mod ppo;
mod reinforce;
mod td;
mod td3;
mod truncated_policy_iteration;
mod value_iteration;
mod vpg;

use self::{
    actor_critic::ActorCriticArgs, approximate_td::ApproximateTDArgs, ddpg::DdpgArgs, dqn::DqnArgs,
    monte_carlo::MonteCarloArgs, policy_iteration::PolicyIterationArgs, ppo::PpoArgs,
    reinforce::ReinforceArgs, td::TDArgs, td3::Td3Args,
    truncated_policy_iteration::TruncatedPolicyIterationArgs, value_iteration::ValueIterationArgs,
    vpg::VanillaPolicyGradientArgs,
};
use clap::Subcommand;
use rlstudy::algorithms::{
    ActorCritic, ActorCriticConfig, ApproximateTD, ApproximateTDConfig, DdpgConfig, DeepQLearning,
    DeterministicPolicyGradient, DqnConfig, MonteCarlo, MonteCarloConfig, PolicyIterationConfig,
    PpoConfig, Reinforce, ReinforceConfig, StochasticPolicyGradient, TD, TDConfig, Td3Config,
    TruncatedPolicyIteration, TruncatedPolicyIterationConfig, ValueIterationConfig,
    VanillaPolicyGradientConfig,
};
use serde::{Serialize, de::DeserializeOwned};
use std::{fs, path::Path};

#[derive(Subcommand, Clone)]
pub(super) enum Solver {
    VI(ValueIterationArgs),
    PI(PolicyIterationArgs),
    Tpi(TruncatedPolicyIterationArgs),
    MC(MonteCarloArgs),
    TD(TDArgs),
    Dqn(DqnArgs),
    Ddpg(DdpgArgs),
    Td3(Td3Args),
    Reinforce(ReinforceArgs),
    Vpg(VanillaPolicyGradientArgs),
    Ppo(PpoArgs),
    ActorCritic(ActorCriticArgs),
    ApproximateTD(ApproximateTDArgs),
}

impl Solver {
    pub(super) fn as_str(&self) -> &'static str {
        match self {
            Self::VI(_) => "vi",
            Self::PI(_) => "pi",
            Self::Tpi(_) => "tpi",
            Self::MC(_) => "mc",
            Self::TD(_) => "td",
            Self::Dqn(_) => "dqn",
            Self::Ddpg(_) => "ddpg",
            Self::Td3(_) => "td3",
            Self::Reinforce(_) => "reinforce",
            Self::Vpg(_) => "vpg",
            Self::Ppo(_) => "ppo",
            Self::ActorCritic(_) => "actor-critic",
            Self::ApproximateTD(_) => "approximate-td",
        }
    }
}

pub(super) enum SelectedSolver {
    ModelBased(TruncatedPolicyIteration),
    MonteCarlo(MonteCarlo),
    TD(TD),
    Dqn(DeepQLearning),
    Ddpg(DeterministicPolicyGradient),
    Td3(DeterministicPolicyGradient),
    Reinforce(Reinforce),
    StochasticPolicyGradient(SelectedStochasticPolicyGradient),
    ActorCritic(ActorCritic),
    ApproximateTD(ApproximateTD),
}

pub(super) enum SelectedStochasticPolicyGradient {
    Vpg(VanillaPolicyGradientConfig),
    Ppo(PpoConfig),
}

impl SelectedStochasticPolicyGradient {
    pub(super) fn build<Action>(self) -> Result<StochasticPolicyGradient<Action>, String> {
        match self {
            Self::Vpg(config) => StochasticPolicyGradient::vpg(config),
            Self::Ppo(config) => StochasticPolicyGradient::ppo(config),
        }
    }
}

pub(super) fn build_solver(
    solver: &Solver,
    config_path: Option<&Path>,
) -> Result<(SelectedSolver, serde_json::Value), Box<dyn std::error::Error>> {
    match solver {
        Solver::VI(args) => {
            let (config, parameters): (ValueIterationConfig, _) =
                resolve_config(args, config_path)?;
            Ok((
                SelectedSolver::ModelBased(TruncatedPolicyIteration::value_iteration(config)?),
                parameters,
            ))
        }
        Solver::PI(args) => {
            let (config, parameters): (PolicyIterationConfig, _) =
                resolve_config(args, config_path)?;
            Ok((
                SelectedSolver::ModelBased(TruncatedPolicyIteration::policy_iteration(config)?),
                parameters,
            ))
        }
        Solver::Tpi(args) => {
            let (config, parameters): (TruncatedPolicyIterationConfig, _) =
                resolve_config(args, config_path)?;
            Ok((
                SelectedSolver::ModelBased(TruncatedPolicyIteration::new(config)?),
                parameters,
            ))
        }
        Solver::MC(args) => {
            let (config, parameters): (MonteCarloConfig, _) = resolve_config(args, config_path)?;
            Ok((
                SelectedSolver::MonteCarlo(MonteCarlo::new(config)?),
                parameters,
            ))
        }
        Solver::TD(args) => {
            let (config, parameters): (TDConfig, _) = resolve_config(args, config_path)?;
            Ok((SelectedSolver::TD(TD::new(config)?), parameters))
        }
        Solver::Dqn(args) => {
            let (config, parameters): (DqnConfig, _) = resolve_config(args, config_path)?;
            Ok((SelectedSolver::Dqn(DeepQLearning::new(config)?), parameters))
        }
        Solver::Ddpg(args) => {
            let (config, parameters): (DdpgConfig, _) = resolve_config(args, config_path)?;
            Ok((
                SelectedSolver::Ddpg(DeterministicPolicyGradient::new(config.into())?),
                parameters,
            ))
        }
        Solver::Td3(args) => {
            let (config, parameters): (Td3Config, _) = resolve_config(args, config_path)?;
            Ok((
                SelectedSolver::Td3(DeterministicPolicyGradient::new(config.into())?),
                parameters,
            ))
        }
        Solver::Reinforce(args) => {
            let (config, parameters): (ReinforceConfig, _) = resolve_config(args, config_path)?;
            Ok((
                SelectedSolver::Reinforce(Reinforce::new(config)?),
                parameters,
            ))
        }
        Solver::Vpg(args) => {
            let (config, parameters): (VanillaPolicyGradientConfig, _) =
                resolve_config(args, config_path)?;
            Ok((
                SelectedSolver::StochasticPolicyGradient(SelectedStochasticPolicyGradient::Vpg(
                    config,
                )),
                parameters,
            ))
        }
        Solver::Ppo(args) => {
            let (config, parameters): (PpoConfig, _) = resolve_config(args, config_path)?;
            Ok((
                SelectedSolver::StochasticPolicyGradient(SelectedStochasticPolicyGradient::Ppo(
                    config,
                )),
                parameters,
            ))
        }
        Solver::ActorCritic(args) => {
            let (config, parameters): (ActorCriticConfig, _) = resolve_config(args, config_path)?;
            Ok((
                SelectedSolver::ActorCritic(ActorCritic::new(config)?),
                parameters,
            ))
        }
        Solver::ApproximateTD(args) => {
            let (config, parameters): (ApproximateTDConfig, _) = resolve_config(args, config_path)?;
            Ok((
                SelectedSolver::ApproximateTD(ApproximateTD::new(config)?),
                parameters,
            ))
        }
    }
}

fn resolve_config<A, T>(
    args: &A,
    path: Option<&Path>,
) -> Result<(T, serde_json::Value), Box<dyn std::error::Error>>
where
    A: Clone + Into<T>,
    T: DeserializeOwned + Serialize,
{
    let config = if let Some(path) = path {
        read_config(path)?
    } else {
        args.clone().into()
    };
    let parameters = serde_json::to_value(&config)?;
    Ok((config, parameters))
}

fn read_config<T>(path: &Path) -> Result<T, Box<dyn std::error::Error>>
where
    T: DeserializeOwned,
{
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("failed to read solver config {}: {error}", path.display()))?;
    yaml_serde::from_str(&contents).map_err(|error| {
        format!("failed to parse solver config {}: {error}", path.display()).into()
    })
}

#[cfg(test)]
mod tests {
    use rlstudy::algorithms::{
        ActorCriticConfig, ApproximateTDConfig, DdpgConfig, DqnConfig, PpoConfig, ReinforceConfig,
        StochasticPolicyGradient, VanillaPolicyGradientConfig,
    };

    #[test]
    fn dqn_yaml_uses_defaults_for_missing_fields() {
        let config: DqnConfig = yaml_serde::from_str(
            "q_network:\n  learning_rate:\n    type: constant\n    value: 0.1",
        )
        .unwrap();

        assert_eq!(
            config.q_network.learning_rate,
            rlstudy::algorithms::Schedule::Constant { value: 0.1 }
        );
        assert_eq!(config.replay_capacity, 1_000);
        assert_eq!(config.episode_length, None);
    }

    #[test]
    fn dqn_yaml_selects_soft_target_updates() {
        let config: DqnConfig =
            yaml_serde::from_str("target_update:\n  type: soft\n  rate: 0.01").unwrap();

        assert_eq!(
            config.target_update,
            rlstudy::algorithms::TargetUpdateConfig::Soft { rate: 0.01 }
        );
    }

    #[test]
    fn checked_in_dqn_configs_parse() {
        for contents in [
            include_str!("../../../settings/zhao_5x5/dqn.yaml"),
            include_str!("../../../settings/cartpole/dqn.yaml"),
        ] {
            let config: DqnConfig = yaml_serde::from_str(contents).unwrap();
            assert!(rlstudy::algorithms::DeepQLearning::new(config).is_ok());
        }
    }

    #[test]
    fn checked_in_policy_gradient_config_parses() {
        let config: ReinforceConfig =
            yaml_serde::from_str(include_str!("../../../settings/cartpole/reinforce.yaml"))
                .unwrap();

        assert!(rlstudy::algorithms::Reinforce::new(config).is_ok());
    }

    #[test]
    fn checked_in_actor_critic_config_parses() {
        let config: ActorCriticConfig =
            yaml_serde::from_str(include_str!("../../../settings/cartpole/actor_critic.yaml"))
                .unwrap();

        assert!(rlstudy::algorithms::ActorCritic::new(config).is_ok());
    }

    #[test]
    fn checked_in_vpg_config_parses() {
        let config: VanillaPolicyGradientConfig =
            yaml_serde::from_str(include_str!("../../../settings/cartpole/vpg.yaml")).unwrap();

        let solver: Result<StochasticPolicyGradient, _> = StochasticPolicyGradient::vpg(config);
        assert!(solver.is_ok());
    }

    #[test]
    fn checked_in_ppo_config_parses() {
        let config: PpoConfig =
            yaml_serde::from_str(include_str!("../../../settings/cartpole/ppo.yaml")).unwrap();
        let solver: Result<StochasticPolicyGradient, _> = StochasticPolicyGradient::ppo(config);
        assert!(solver.is_ok());
    }

    #[test]
    fn checked_in_approximate_td_config_parses() {
        let config: ApproximateTDConfig = yaml_serde::from_str(include_str!(
            "../../../settings/mountain_car/approximate_td.yaml"
        ))
        .unwrap();

        assert!(rlstudy::algorithms::ApproximateTD::new(config).is_ok());
    }

    #[test]
    fn checked_in_ddpg_config_parses() {
        let config: DdpgConfig =
            yaml_serde::from_str(include_str!("../../../settings/half_cheetah/ddpg.yaml")).unwrap();

        assert!(rlstudy::algorithms::DeterministicPolicyGradient::new(config.into()).is_ok());
    }
}
