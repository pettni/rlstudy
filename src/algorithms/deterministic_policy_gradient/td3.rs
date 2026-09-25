use super::algorithm::DeterministicPolicyGradientConfig;
use super::algorithm::TargetNoise;
use crate::network::MlpSettings;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Td3Config {
    pub gamma: f64,
    pub batch_size: usize,
    pub replay_capacity: usize,
    pub warm_up_transitions: usize,
    pub steps_per_iteration: usize,
    pub q_updates_per_iteration: usize,
    pub pi_updates_every_n_q_updates: usize,
    pub exploration_noise: Option<f64>,
    pub iterations: usize,
    pub ema_rate: f64,
    pub q_network: MlpSettings,
    pub pi_network: MlpSettings,
    pub target_noise: f64,
    pub target_noise_clip: f64,
}

impl Default for Td3Config {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            batch_size: 64,
            replay_capacity: 1_000,
            warm_up_transitions: 64,
            steps_per_iteration: 1,
            q_updates_per_iteration: 1,
            pi_updates_every_n_q_updates: 2,
            exploration_noise: None,
            iterations: 1_000,
            ema_rate: 0.01,
            q_network: MlpSettings::default(),
            pi_network: MlpSettings::default(),
            target_noise: 0.2,
            target_noise_clip: 0.5,
        }
    }
}

impl From<Td3Config> for DeterministicPolicyGradientConfig {
    fn from(config: Td3Config) -> Self {
        Self {
            gamma: config.gamma,
            batch_size: config.batch_size,
            replay_capacity: config.replay_capacity,
            warm_up_transitions: config.warm_up_transitions,
            steps_per_iteration: config.steps_per_iteration,
            q_updates_per_iteration: config.q_updates_per_iteration,
            pi_updates_every_n_q_updates: config.pi_updates_every_n_q_updates,
            exploration_noise: config.exploration_noise,
            iterations: config.iterations,
            ema_rate: config.ema_rate,
            q_network: config.q_network,
            double_q_learning: true,
            pi_network: config.pi_network,
            target_noise: Some(TargetNoise {
                noise_level: config.target_noise,
                clip: config.target_noise_clip,
            }),
        }
    }
}
