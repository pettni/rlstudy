use clap::Subcommand;
use rlstudy::algorithms::{
    DeepQLearning, DqnConfig, MonteCarlo, MonteCarloConfig, PolicyIterationConfig, TD, TDConfig,
    TDVariant, TruncatedPolicyIteration, TruncatedPolicyIterationConfig, ValueIterationConfig,
};
use serde::{Serialize, de::DeserializeOwned};
use std::{fs, path::Path};

#[derive(Subcommand, Clone)]
pub(super) enum Solver {
    VI(ValueIterationConfig),
    PI(PolicyIterationConfig),
    TPI(TruncatedPolicyIterationConfig),
    MC(MonteCarloConfig),
    TD(TDConfig),
    DQN(DqnConfig),
}

impl Solver {
    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::VI(_) => "value-iteration",
            Self::PI(_) => "policy-iteration",
            Self::TPI(_) => "truncated-policy-iteration",
            Self::MC(_) => "monte-carlo",
            Self::TD(config) => match config.variant {
                TDVariant::Sarsa => "sarsa",
                TDVariant::ExpectedSarsa => "expected-sarsa",
                TDVariant::QLearning => "q-learning",
            },
            Self::DQN(_) => "dqn",
        }
    }
}

pub(super) enum SelectedSolver {
    ModelBased(TruncatedPolicyIteration),
    MonteCarlo(MonteCarlo),
    TD(TD),
    DQN(DeepQLearning),
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
        Solver::TPI(config) => {
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
        Solver::DQN(config) => {
            let (config, parameters) = resolve_config(config, config_path)?;
            Ok((SelectedSolver::DQN(DeepQLearning::new(config)?), parameters))
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
        let contents = fs::read_to_string(path)
            .map_err(|error| format!("failed to read solver config {}: {error}", path.display()))?;
        yaml_serde::from_str(&contents)
            .map_err(|error| format!("failed to parse solver config {}: {error}", path.display()))?
    } else {
        config.clone()
    };
    let parameters = serde_json::to_value(&config)?;
    Ok((config, parameters))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dqn_yaml_uses_defaults_for_missing_fields() {
        let config: DqnConfig = yaml_serde::from_str("learning_rate: 0.1").unwrap();

        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.replay_capacity, 1_000);
        assert_eq!(config.evaluation_episodes, 10);
    }

    #[test]
    fn dqn_yaml_selects_soft_target_updates() {
        let config: DqnConfig =
            yaml_serde::from_str("target_update: soft\ntarget_update_rate: 0.01").unwrap();

        assert_eq!(
            config.target_update,
            rlstudy::algorithms::TargetUpdateMethod::Soft
        );
        assert_eq!(config.target_update_rate, 0.01);
    }
}
