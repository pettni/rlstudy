use clap::Subcommand;
use rlstudy::algorithms::{
    ActorCritic, ActorCriticConfig, ApproximateTD, ApproximateTDArgs, DDPG, DDPGConfig, DdpgArgs,
    DeepQLearning, DqnArgs, MonteCarlo, MonteCarloConfig, PolicyGradient, PolicyIterationConfig,
    PpoConfig, Reinforce, ReinforceConfig, TD, TDConfig, TruncatedPolicyIteration,
    TruncatedPolicyIterationConfig, ValueIterationConfig, VanillaPolicyGradientConfig,
};
use serde::{Serialize, de::DeserializeOwned};
use std::{fs, path::Path};

#[derive(Subcommand, Clone)]
pub(super) enum Solver {
    VI(ValueIterationConfig),
    PI(PolicyIterationConfig),
    Tpi(TruncatedPolicyIterationConfig),
    MC(MonteCarloConfig),
    TD(TDConfig),
    Dqn(DqnArgs),
    Ddpg(DdpgArgs),
    Reinforce(ReinforceConfig),
    Vpg(VanillaPolicyGradientConfig),
    Ppo(PpoConfig),
    ActorCritic(ActorCriticConfig),
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
    Ddpg(DDPG),
    Reinforce(Reinforce),
    Vpg(PolicyGradient),
    Ppo(PolicyGradient),
    ActorCritic(ActorCritic),
    ApproximateTD(ApproximateTD),
}

pub(super) fn build_solver(
    solver: &Solver,
    config_path: Option<&Path>,
) -> Result<(SelectedSolver, serde_json::Value), Box<dyn std::error::Error>> {
    match solver {
        Solver::VI(config) => {
            let (config, parameters) = resolve_config(config, config_path)?;
            Ok((
                SelectedSolver::ModelBased(TruncatedPolicyIteration::value_iteration(config)?),
                parameters,
            ))
        }
        Solver::PI(config) => {
            let (config, parameters) = resolve_config(config, config_path)?;
            Ok((
                SelectedSolver::ModelBased(TruncatedPolicyIteration::policy_iteration(config)?),
                parameters,
            ))
        }
        Solver::Tpi(config) => {
            let (config, parameters) = resolve_config(config, config_path)?;
            Ok((
                SelectedSolver::ModelBased(TruncatedPolicyIteration::new(config)?),
                parameters,
            ))
        }
        Solver::MC(config) => {
            let (config, parameters) = resolve_config(config, config_path)?;
            Ok((
                SelectedSolver::MonteCarlo(MonteCarlo::new(config)?),
                parameters,
            ))
        }
        Solver::TD(config) => {
            let (config, parameters) = resolve_config(config, config_path)?;
            Ok((SelectedSolver::TD(TD::new(config)?), parameters))
        }
        Solver::Dqn(args) => {
            let config = match config_path {
                Some(path) => read_config(path)?,
                None => args.clone().into(),
            };
            let parameters = serde_json::to_value(&config)?;
            Ok((SelectedSolver::Dqn(DeepQLearning::new(config)?), parameters))
        }
        Solver::Ddpg(args) => {
            let config: DDPGConfig = match config_path {
                Some(path) => read_config(path)?,
                None => args.clone().into(),
            };
            let parameters = serde_json::to_value(&config)?;
            Ok((SelectedSolver::Ddpg(DDPG::new(config)?), parameters))
        }
        Solver::Reinforce(config) => {
            let (config, parameters) = resolve_config(config, config_path)?;
            Ok((
                SelectedSolver::Reinforce(Reinforce::new(config)?),
                parameters,
            ))
        }
        Solver::Vpg(config) => {
            let (config, parameters) = resolve_config(config, config_path)?;
            Ok((
                SelectedSolver::Vpg(PolicyGradient::vpg(config)?),
                parameters,
            ))
        }
        Solver::Ppo(config) => {
            let (config, parameters) = resolve_config(config, config_path)?;
            Ok((
                SelectedSolver::Ppo(PolicyGradient::ppo(config)?),
                parameters,
            ))
        }
        Solver::ActorCritic(config) => {
            let (config, parameters) = resolve_config(config, config_path)?;
            Ok((
                SelectedSolver::ActorCritic(ActorCritic::new(config)?),
                parameters,
            ))
        }
        Solver::ApproximateTD(args) => {
            let config = match config_path {
                Some(path) => read_config(path)?,
                None => args.clone().into(),
            };
            let parameters = serde_json::to_value(&config)?;
            Ok((
                SelectedSolver::ApproximateTD(ApproximateTD::new(config)?),
                parameters,
            ))
        }
    }
}

fn resolve_config<T>(
    config: &T,
    path: Option<&Path>,
) -> Result<(T, serde_json::Value), Box<dyn std::error::Error>>
where
    T: Clone + DeserializeOwned + Serialize,
{
    let config = if let Some(path) = path {
        read_config(path)?
    } else {
        config.clone()
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
        ActorCriticConfig, ApproximateTDConfig, DDPGConfig, DqnConfig, PpoConfig, ReinforceConfig,
        VanillaPolicyGradientConfig,
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
            include_str!("../../settings/dqn_for_5x5.yaml"),
            include_str!("../../settings/dqn_for_cartpole.yaml"),
        ] {
            let config: DqnConfig = yaml_serde::from_str(contents).unwrap();
            assert!(rlstudy::algorithms::DeepQLearning::new(config).is_ok());
        }
    }

    #[test]
    fn checked_in_policy_gradient_config_parses() {
        let config: ReinforceConfig =
            yaml_serde::from_str(include_str!("../../settings/reinforce_for_cartpole.yaml"))
                .unwrap();

        assert!(rlstudy::algorithms::Reinforce::new(config).is_ok());
    }

    #[test]
    fn checked_in_actor_critic_config_parses() {
        let config: ActorCriticConfig = yaml_serde::from_str(include_str!(
            "../../settings/actor_critic_for_cartpole.yaml"
        ))
        .unwrap();

        assert!(rlstudy::algorithms::ActorCritic::new(config).is_ok());
    }

    #[test]
    fn checked_in_vpg_config_parses() {
        let config: VanillaPolicyGradientConfig =
            yaml_serde::from_str(include_str!("../../settings/vpg_for_cartpole.yaml")).unwrap();

        assert!(rlstudy::algorithms::PolicyGradient::vpg(config).is_ok());
    }

    #[test]
    fn checked_in_ppo_config_parses() {
        let config: PpoConfig =
            yaml_serde::from_str(include_str!("../../settings/ppo_for_cartpole.yaml")).unwrap();
        assert!(rlstudy::algorithms::PolicyGradient::ppo(config).is_ok());
    }

    #[test]
    fn checked_in_approximate_td_config_parses() {
        let config: ApproximateTDConfig = yaml_serde::from_str(include_str!(
            "../../settings/approximate_td_for_mountain_car.yaml"
        ))
        .unwrap();

        assert!(rlstudy::algorithms::ApproximateTD::new(config).is_ok());
    }

    #[test]
    fn checked_in_ddpg_config_parses() {
        let config: DDPGConfig =
            yaml_serde::from_str(include_str!("../../settings/ddpg_for_half_cheetah.yaml"))
                .unwrap();

        assert!(rlstudy::algorithms::DDPG::new(config).is_ok());
    }
}
