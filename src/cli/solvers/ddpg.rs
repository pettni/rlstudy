use clap::Args;
use rlstudy::{
    algorithms::{DdpgConfig, Schedule},
    network::{Activation, MlpSettings},
};

#[derive(Debug, Clone, PartialEq, Args)]
pub struct DdpgArgs {
    #[arg(long, default_value_t = DdpgConfig::default().gamma)]
    gamma: f64,
    #[arg(long, default_value_t = DdpgConfig::default().batch_size)]
    batch_size: usize,
    #[arg(long, default_value_t = DdpgConfig::default().replay_capacity)]
    replay_capacity: usize,
    #[arg(long, default_value_t = DdpgConfig::default().warm_up_transitions)]
    warm_up_transitions: usize,
    #[arg(long, default_value_t = DdpgConfig::default().steps_per_iteration)]
    steps_per_iteration: usize,
    #[arg(long, default_value_t = DdpgConfig::default().q_updates_per_iteration)]
    q_updates_per_iteration: usize,
    #[arg(long)]
    exploration_noise: Option<f64>,
    #[arg(long, default_value_t = DdpgConfig::default().iterations)]
    iterations: usize,
    #[arg(long, default_value_t = DdpgConfig::default().ema_rate)]
    ema_rate: f64,
    #[arg(long)]
    q_learning_rate: Option<Schedule>,
    #[arg(long, value_delimiter = ',')]
    q_hidden_dims: Option<Vec<usize>>,
    #[arg(long, value_enum)]
    q_activation: Option<Activation>,
    #[arg(long)]
    pi_learning_rate: Option<Schedule>,
    #[arg(long, value_delimiter = ',')]
    pi_hidden_dims: Option<Vec<usize>>,
    #[arg(long, value_enum)]
    pi_activation: Option<Activation>,
}

impl From<DdpgArgs> for DdpgConfig {
    fn from(args: DdpgArgs) -> Self {
        let defaults = Self::default();
        Self {
            gamma: args.gamma,
            batch_size: args.batch_size,
            replay_capacity: args.replay_capacity,
            warm_up_transitions: args.warm_up_transitions,
            steps_per_iteration: args.steps_per_iteration,
            q_updates_per_iteration: args.q_updates_per_iteration,
            exploration_noise: args.exploration_noise,
            iterations: args.iterations,
            ema_rate: args.ema_rate,
            q_network: MlpSettings {
                learning_rate: args
                    .q_learning_rate
                    .unwrap_or(defaults.q_network.learning_rate),
                hidden_dims: args.q_hidden_dims.unwrap_or(defaults.q_network.hidden_dims),
                activation: args.q_activation.unwrap_or(defaults.q_network.activation),
            },
            pi_network: MlpSettings {
                learning_rate: args
                    .pi_learning_rate
                    .unwrap_or(defaults.pi_network.learning_rate),
                hidden_dims: args
                    .pi_hidden_dims
                    .unwrap_or(defaults.pi_network.hidden_dims),
                activation: args.pi_activation.unwrap_or(defaults.pi_network.activation),
            },
        }
    }
}
