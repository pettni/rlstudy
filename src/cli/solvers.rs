use clap::Subcommand;
use rlstudy::algorithms::{
    DeepQLearning, DqnArgs, MonteCarlo, MonteCarloConfig, PolicyGradient, PolicyGradientConfig,
    PolicyIterationConfig, TD, TDConfig, TruncatedPolicyIteration, TruncatedPolicyIterationConfig,
    ValueIterationConfig,
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
    PolicyGradient(PolicyGradientConfig),
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
            Self::PolicyGradient(_) => "policy-gradient",
        }
    }
}

pub(super) enum SelectedSolver {
    ModelBased(TruncatedPolicyIteration),
    MonteCarlo(MonteCarlo),
    TD(TD),
    Dqn(DeepQLearning),
    PolicyGradient(PolicyGradient),
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
        Solver::PolicyGradient(config) => {
            let (config, parameters) = resolve_config(config, config_path)?;
            Ok((
                SelectedSolver::PolicyGradient(PolicyGradient::new(config)?),
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
    use rlstudy::algorithms::DqnConfig;
    use rlstudy::algorithms::PolicyGradientConfig;

    #[test]
    fn dqn_yaml_uses_defaults_for_missing_fields() {
        let config: DqnConfig =
            yaml_serde::from_str("learning_rate:\n  type: constant\n  value: 0.1").unwrap();

        assert_eq!(
            config.learning_rate,
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
    fn checked_in_policy_gradient_config_parses() {
        let config: PolicyGradientConfig =
            yaml_serde::from_str(include_str!("../../settings/reinforce_for_cartpole.yaml"))
                .unwrap();

        assert!(rlstudy::algorithms::PolicyGradient::new(config).is_ok());
    }
}
