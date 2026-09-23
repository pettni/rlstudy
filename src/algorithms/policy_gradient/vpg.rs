use super::common::{PolicyGradientConfig, PolicyObjective, PolicyUpdateStrategy};
use crate::network::mlp::Activation;
use crate::schedule::Schedule;
use clap::Args;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Args, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct VanillaPolicyGradientConfig {
    #[arg(long, default_value_t = VanillaPolicyGradientConfig::default().gamma)]
    pub gamma: f64,
    #[arg(long, default_value_t = VanillaPolicyGradientConfig::default().lambda)]
    pub lambda: f64,
    #[arg(long, default_value = "constant:0.0003")]
    pub pi_learning_rate: Schedule,
    #[arg(long, default_value = "constant:0.001")]
    pub v_learning_rate: Schedule,
    #[arg(long, value_delimiter = ',', default_value = "64,64")]
    pub hidden_dims: Vec<usize>,
    #[arg(long, value_enum, default_value_t = VanillaPolicyGradientConfig::default().activation)]
    pub activation: Activation,
    #[arg(long, default_value_t = VanillaPolicyGradientConfig::default().epochs)]
    pub epochs: usize,
    #[arg(long, default_value_t = VanillaPolicyGradientConfig::default().steps_per_epoch)]
    pub steps_per_epoch: usize,
    #[arg(long, default_value_t = VanillaPolicyGradientConfig::default().value_updates_per_epoch)]
    pub value_updates_per_epoch: usize,
}

impl Default for VanillaPolicyGradientConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.97,
            pi_learning_rate: Schedule::Constant { value: 0.0003 },
            v_learning_rate: Schedule::Constant { value: 0.001 },
            hidden_dims: vec![64, 64],
            activation: Activation::Tanh,
            epochs: 100,
            steps_per_epoch: 4000,
            value_updates_per_epoch: 80,
        }
    }
}

impl From<VanillaPolicyGradientConfig> for PolicyGradientConfig {
    fn from(config: VanillaPolicyGradientConfig) -> Self {
        Self {
            objective: PolicyObjective::Vpg,
            gamma: config.gamma,
            lambda: config.lambda,
            pi_learning_rate: config.pi_learning_rate,
            v_learning_rate: config.v_learning_rate,
            hidden_dims: config.hidden_dims,
            activation: config.activation,
            epochs: config.epochs,
            steps_per_epoch: config.steps_per_epoch,
            policy_update_strategy: PolicyUpdateStrategy::default(),
            value_updates_per_epoch: config.value_updates_per_epoch,
        }
    }
}
