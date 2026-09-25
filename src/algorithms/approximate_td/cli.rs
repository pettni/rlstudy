use super::ApproximateTDConfig;
use crate::network::{Activation, MlpSettings};
use crate::schedule::Schedule;
use clap::Args;

#[derive(Debug, Clone, PartialEq, Args)]
pub struct ApproximateTDArgs {
    #[arg(long, default_value_t = ApproximateTDConfig::default().n)]
    n: usize,
    #[arg(long, default_value_t = ApproximateTDConfig::default().gamma)]
    gamma: f64,
    #[arg(long, default_value_t = ApproximateTDConfig::default().epsilon)]
    epsilon: f64,
    #[arg(long, default_value_t = ApproximateTDConfig::default().reward_scale)]
    reward_scale: f64,
    #[arg(long)]
    learning_rate: Option<Schedule>,
    #[arg(long, value_delimiter = ',')]
    hidden_dims: Option<Vec<usize>>,
    #[arg(long, value_enum)]
    activation: Option<Activation>,
    #[arg(long, default_value_t = ApproximateTDConfig::default().episodes)]
    episodes: usize,
    #[arg(long, default_value_t = ApproximateTDConfig::default().episode_length)]
    episode_length: usize,
}

impl From<ApproximateTDArgs> for ApproximateTDConfig {
    fn from(args: ApproximateTDArgs) -> Self {
        let defaults = Self::default();
        Self {
            n: args.n,
            gamma: args.gamma,
            epsilon: args.epsilon,
            reward_scale: args.reward_scale,
            q_network: MlpSettings {
                learning_rate: args
                    .learning_rate
                    .unwrap_or(defaults.q_network.learning_rate),
                hidden_dims: args.hidden_dims.unwrap_or(defaults.q_network.hidden_dims),
                activation: args.activation.unwrap_or(defaults.q_network.activation),
            },
            episodes: args.episodes,
            episode_length: args.episode_length,
        }
    }
}
