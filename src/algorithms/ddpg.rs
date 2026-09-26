use super::policy_gradient::DeterministicPolicyGradientConfig;
use crate::network::MlpSettings;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DdpgConfig {
    pub gamma: f64,
    pub batch_size: usize,
    pub replay_capacity: usize,
    pub warm_up_transitions: usize,
    pub steps_per_iteration: usize,
    pub q_updates_per_iteration: usize,
    pub exploration_noise: Option<f64>,
    pub iterations: usize,
    pub ema_rate: f64,
    pub q_network: MlpSettings,
    pub pi_network: MlpSettings,
}

impl Default for DdpgConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            batch_size: 64,
            replay_capacity: 1_000,
            warm_up_transitions: 64,
            steps_per_iteration: 1,
            q_updates_per_iteration: 1,
            exploration_noise: None,
            iterations: 1_000,
            ema_rate: 0.01,
            q_network: MlpSettings::default(),
            pi_network: MlpSettings::default(),
        }
    }
}

impl From<DdpgConfig> for DeterministicPolicyGradientConfig {
    fn from(config: DdpgConfig) -> Self {
        Self {
            gamma: config.gamma,
            batch_size: config.batch_size,
            replay_capacity: config.replay_capacity,
            warm_up_transitions: config.warm_up_transitions,
            steps_per_iteration: config.steps_per_iteration,
            q_updates_per_iteration: config.q_updates_per_iteration,
            pi_updates_every_n_q_updates: 1,
            exploration_noise: config.exploration_noise,
            iterations: config.iterations,
            ema_rate: config.ema_rate,
            q_network: config.q_network,
            double_q_learning: false,
            pi_network: config.pi_network,
            target_noise: None,
        }
    }
}
