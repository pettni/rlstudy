use clap::Subcommand;
use rlstudy::algorithms::{
    DeepQLearning, DqnConfig, MonteCarlo, MonteCarloConfig, PolicyIterationConfig, TD, TDConfig,
    TDVariant, TruncatedPolicyIteration, TruncatedPolicyIterationConfig, ValueIterationConfig,
};
use serde::de::DeserializeOwned;
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
) -> Result<SelectedSolver, Box<dyn std::error::Error>> {
    match solver {
        Solver::VI(config) => Ok(SelectedSolver::ModelBased(
            TruncatedPolicyIteration::value_iteration(resolve_config(config, config_path)?)?,
        )),
        Solver::PI(config) => Ok(SelectedSolver::ModelBased(
            TruncatedPolicyIteration::policy_iteration(resolve_config(config, config_path)?)?,
        )),
        Solver::TPI(config) => Ok(SelectedSolver::ModelBased(TruncatedPolicyIteration::new(
            resolve_config(config, config_path)?,
        )?)),
        Solver::MC(config) => Ok(SelectedSolver::MonteCarlo(MonteCarlo::new(
            resolve_config(config, config_path)?,
        )?)),
        Solver::TD(config) => Ok(SelectedSolver::TD(TD::new(resolve_config(
            config,
            config_path,
        )?)?)),
        Solver::DQN(config) => Ok(SelectedSolver::DQN(DeepQLearning::new(resolve_config(
            config,
            config_path,
        )?)?)),
    }
}

fn resolve_config<T>(config: &T, path: Option<&Path>) -> Result<T, Box<dyn std::error::Error>>
where
    T: Clone + DeserializeOwned,
{
    let Some(path) = path else {
        return Ok(config.clone());
    };
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("failed to read solver config {}: {error}", path.display()))?;
    yaml_serde::from_str(&contents).map_err(|error| {
        format!("failed to parse solver config {}: {error}", path.display()).into()
    })
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
