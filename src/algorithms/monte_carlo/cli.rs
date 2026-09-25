use super::MonteCarloConfig;
use clap::Args;

#[derive(Debug, Clone, PartialEq, Args)]
pub struct MonteCarloArgs {
    #[arg(long, default_value_t = MonteCarloConfig::default().gamma)]
    gamma: f64,
    #[arg(long, default_value_t = MonteCarloConfig::default().episodes)]
    episodes: usize,
    #[arg(long, default_value_t = MonteCarloConfig::default().episode_length)]
    episode_length: usize,
    #[arg(long, default_value_t = MonteCarloConfig::default().epsilon)]
    epsilon: f64,
}

impl From<MonteCarloArgs> for MonteCarloConfig {
    fn from(args: MonteCarloArgs) -> Self {
        Self {
            gamma: args.gamma,
            episodes: args.episodes,
            episode_length: args.episode_length,
            epsilon: args.epsilon,
        }
    }
}
