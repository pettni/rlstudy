use clap::Args;
use rlstudy::algorithms::{TDConfig, TDVariant};

#[derive(Debug, Clone, PartialEq, Args)]
pub struct TDArgs {
    #[arg(long, default_value_t = TDConfig::default().gamma)]
    gamma: f64,
    #[arg(long, default_value_t = TDConfig::default().episodes)]
    episodes: u64,
    #[arg(long, default_value_t = TDConfig::default().episode_length)]
    episode_length: u64,
    #[arg(long, default_value_t = TDConfig::default().epsilon)]
    epsilon: f64,
    #[arg(long, default_value_t = TDConfig::default().alpha)]
    alpha: f64,
    #[arg(long, value_enum, default_value_t = TDConfig::default().variant)]
    variant: TDVariant,
}

impl From<TDArgs> for TDConfig {
    fn from(args: TDArgs) -> Self {
        Self {
            gamma: args.gamma,
            episodes: args.episodes,
            episode_length: args.episode_length,
            epsilon: args.epsilon,
            alpha: args.alpha,
            variant: args.variant,
        }
    }
}
