use clap::Args;
use rlstudy::algorithms::ValueIterationConfig;

#[derive(Debug, Clone, PartialEq, Args)]
pub struct ValueIterationArgs {
    #[arg(long, default_value_t = ValueIterationConfig::default().gamma)]
    gamma: f64,
    #[arg(long)]
    tolerance: Option<f64>,
    #[arg(long)]
    max_iterations: Option<usize>,
}

impl From<ValueIterationArgs> for ValueIterationConfig {
    fn from(args: ValueIterationArgs) -> Self {
        Self {
            gamma: args.gamma,
            tolerance: args.tolerance,
            max_iterations: args.max_iterations,
        }
    }
}
