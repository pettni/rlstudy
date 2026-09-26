use clap::Args;
use rlstudy::{
    algorithms::{ReinforceConfig, Schedule, UpdateVariant},
    network::{Activation, MlpSettings},
};

#[derive(Debug, Clone, PartialEq, Args)]
pub struct ReinforceArgs {
    #[arg(long, default_value_t = ReinforceConfig::default().gamma)]
    gamma: f64,
    #[arg(long)]
    learning_rate: Option<Schedule>,
    #[arg(long, value_delimiter = ',')]
    hidden_dims: Option<Vec<usize>>,
    #[arg(long, value_enum)]
    activation: Option<Activation>,
    #[arg(long, default_value_t = ReinforceConfig::default().episodes)]
    episodes: usize,
    #[arg(long, default_value_t = ReinforceConfig::default().episode_length)]
    episode_length: usize,
    #[arg(long)]
    gradient_clip_value: Option<f32>,
    #[arg(long)]
    critic: Option<UpdateVariant>,
}

impl From<ReinforceArgs> for ReinforceConfig {
    fn from(args: ReinforceArgs) -> Self {
        let defaults = Self::default();
        Self {
            gamma: args.gamma,
            pi_network: MlpSettings {
                learning_rate: args
                    .learning_rate
                    .clone()
                    .unwrap_or(defaults.pi_network.learning_rate),
                hidden_dims: args
                    .hidden_dims
                    .clone()
                    .unwrap_or(defaults.pi_network.hidden_dims),
                activation: args.activation.unwrap_or(defaults.pi_network.activation),
            },
            v_network: MlpSettings {
                learning_rate: args
                    .learning_rate
                    .unwrap_or(defaults.v_network.learning_rate),
                hidden_dims: args.hidden_dims.unwrap_or(defaults.v_network.hidden_dims),
                activation: args.activation.unwrap_or(defaults.v_network.activation),
            },
            episodes: args.episodes,
            episode_length: args.episode_length,
            gradient_clip_value: args.gradient_clip_value,
            critic: args.critic,
        }
    }
}
