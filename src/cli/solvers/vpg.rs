use clap::Args;
use rlstudy::{
    algorithms::{Schedule, VanillaPolicyGradientConfig},
    network::{Activation, MlpSettings},
};

#[derive(Debug, Clone, PartialEq, Args)]
pub struct VanillaPolicyGradientArgs {
    #[arg(long, default_value_t = VanillaPolicyGradientConfig::default().gamma)]
    gamma: f64,
    #[arg(long, default_value_t = VanillaPolicyGradientConfig::default().lambda)]
    lambda: f64,
    #[arg(long)]
    pi_learning_rate: Option<Schedule>,
    #[arg(long)]
    v_learning_rate: Option<Schedule>,
    #[arg(long, value_delimiter = ',')]
    hidden_dims: Option<Vec<usize>>,
    #[arg(long, value_enum)]
    activation: Option<Activation>,
    #[arg(long, default_value_t = VanillaPolicyGradientConfig::default().epochs)]
    epochs: usize,
    #[arg(long, default_value_t = VanillaPolicyGradientConfig::default().steps_per_epoch)]
    steps_per_epoch: usize,
    #[arg(long, default_value_t = VanillaPolicyGradientConfig::default().value_updates_per_epoch)]
    value_updates_per_epoch: usize,
}

impl From<VanillaPolicyGradientArgs> for VanillaPolicyGradientConfig {
    fn from(args: VanillaPolicyGradientArgs) -> Self {
        let defaults = Self::default();
        Self {
            gamma: args.gamma,
            lambda: args.lambda,
            pi_network: MlpSettings {
                learning_rate: args
                    .pi_learning_rate
                    .unwrap_or(defaults.pi_network.learning_rate),
                hidden_dims: args
                    .hidden_dims
                    .clone()
                    .unwrap_or(defaults.pi_network.hidden_dims),
                activation: args.activation.unwrap_or(defaults.pi_network.activation),
            },
            v_network: MlpSettings {
                learning_rate: args
                    .v_learning_rate
                    .unwrap_or(defaults.v_network.learning_rate),
                hidden_dims: args.hidden_dims.unwrap_or(defaults.v_network.hidden_dims),
                activation: args.activation.unwrap_or(defaults.v_network.activation),
            },
            epochs: args.epochs,
            steps_per_epoch: args.steps_per_epoch,
            value_updates_per_epoch: args.value_updates_per_epoch,
        }
    }
}
