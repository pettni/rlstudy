use clap::Args;
use rlstudy::algorithms::Schedule;
use rlstudy::{
    algorithms::ActorCriticConfig,
    network::{Activation, MlpSettings},
};

#[derive(Debug, Clone, PartialEq, Args)]
pub struct ActorCriticArgs {
    #[arg(long, default_value_t = ActorCriticConfig::default().gamma)]
    gamma: f64,
    #[arg(long)]
    actor_learning_rate: Option<Schedule>,
    #[arg(long)]
    critic_learning_rate: Option<Schedule>,
    #[arg(long, value_delimiter = ',')]
    hidden_dims: Option<Vec<usize>>,
    #[arg(long, value_enum)]
    activation: Option<Activation>,
    #[arg(long, default_value_t = ActorCriticConfig::default().episodes)]
    episodes: usize,
    #[arg(long, default_value_t = ActorCriticConfig::default().episode_length)]
    episode_length: usize,
}

impl From<ActorCriticArgs> for ActorCriticConfig {
    fn from(args: ActorCriticArgs) -> Self {
        let defaults = Self::default();
        Self {
            gamma: args.gamma,
            pi_network: MlpSettings {
                learning_rate: args
                    .actor_learning_rate
                    .unwrap_or(defaults.pi_network.learning_rate),
                hidden_dims: args
                    .hidden_dims
                    .clone()
                    .unwrap_or(defaults.pi_network.hidden_dims),
                activation: args.activation.unwrap_or(defaults.pi_network.activation),
            },
            v_network: MlpSettings {
                learning_rate: args
                    .critic_learning_rate
                    .unwrap_or(defaults.v_network.learning_rate),
                hidden_dims: args.hidden_dims.unwrap_or(defaults.v_network.hidden_dims),
                activation: args.activation.unwrap_or(defaults.v_network.activation),
            },
            episodes: args.episodes,
            episode_length: args.episode_length,
        }
    }
}
