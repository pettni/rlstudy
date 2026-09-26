use super::policy_gradient::{PolicyGradientConfig, PolicyObjective, PolicyUpdateStrategy};
use crate::network::{Activation, MlpSettings};
use crate::schedule::Schedule;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct VanillaPolicyGradientConfig {
    pub gamma: f64,
    pub lambda: f64,
    pub pi_network: MlpSettings,
    pub v_network: MlpSettings,
    pub epochs: usize,
    pub steps_per_epoch: usize,
    pub value_updates_per_epoch: usize,
}

impl Default for VanillaPolicyGradientConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.97,
            pi_network: MlpSettings {
                learning_rate: Schedule::Constant { value: 0.0003 },
                hidden_dims: vec![64, 64],
                activation: Activation::Tanh,
            },
            v_network: MlpSettings {
                learning_rate: Schedule::Constant { value: 0.001 },
                hidden_dims: vec![64, 64],
                activation: Activation::Tanh,
            },
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
            pi_network: config.pi_network,
            v_network: config.v_network,
            epochs: config.epochs,
            steps_per_epoch: config.steps_per_epoch,
            policy_update_strategy: PolicyUpdateStrategy::default(),
            value_updates_per_epoch: config.value_updates_per_epoch,
        }
    }
}
