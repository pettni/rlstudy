use super::{DqnConfig, DqnMode, TargetUpdateConfig};
use crate::network::{Activation, MlpSettings, loss::Loss};
use crate::schedule::Schedule;
use clap::Args;

#[derive(Debug, Clone, PartialEq, Args)]
pub struct DqnArgs {
    #[arg(long, value_enum, default_value_t = DqnConfig::default().mode)]
    mode: DqnMode,
    #[arg(long, default_value_t = DqnConfig::default().gamma)]
    gamma: f64,
    #[arg(long)]
    learning_rate: Option<Schedule>,
    #[arg(long, value_delimiter = ',')]
    hidden_dims: Option<Vec<usize>>,
    #[arg(long, value_enum)]
    activation: Option<Activation>,
    #[arg(long, default_value_t = DqnConfig::default().batch_size)]
    batch_size: usize,
    #[arg(long, default_value_t = DqnConfig::default().replay_capacity)]
    replay_capacity: usize,
    #[arg(long, default_value_t = DqnConfig::default().warm_up_transitions)]
    warm_up_transitions: usize,
    #[arg(long, default_value_t = DqnConfig::default().episodes)]
    episodes: usize,
    #[arg(long)]
    episode_length: Option<usize>,
    #[arg(long)]
    epsilon_schedule: Option<Schedule>,
    #[arg(long)]
    target_update: Option<TargetUpdateConfig>,
    #[arg(long, value_enum, default_value_t = DqnConfig::default().loss)]
    loss: Loss,
    #[arg(long)]
    gradient_clip_value: Option<f32>,
}

impl From<DqnArgs> for DqnConfig {
    fn from(args: DqnArgs) -> Self {
        let defaults = Self::default();
        Self {
            mode: args.mode,
            gamma: args.gamma,
            q_network: MlpSettings {
                learning_rate: args
                    .learning_rate
                    .unwrap_or(defaults.q_network.learning_rate),
                hidden_dims: args.hidden_dims.unwrap_or(defaults.q_network.hidden_dims),
                activation: args.activation.unwrap_or(defaults.q_network.activation),
            },
            batch_size: args.batch_size,
            replay_capacity: args.replay_capacity,
            warm_up_transitions: args.warm_up_transitions,
            episodes: args.episodes,
            episode_length: args.episode_length,
            epsilon_schedule: args.epsilon_schedule.unwrap_or_default(),
            target_update: args.target_update.unwrap_or_default(),
            loss: args.loss,
            gradient_clip_value: args.gradient_clip_value,
        }
    }
}
