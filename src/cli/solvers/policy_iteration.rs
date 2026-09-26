use clap::Args;
use rlstudy::algorithms::PolicyIterationConfig;

#[derive(Debug, Clone, PartialEq, Args)]
pub struct PolicyIterationArgs {
    #[arg(long, default_value_t = PolicyIterationConfig::default().gamma)]
    gamma: f64,
    #[arg(long)]
    tolerance: Option<f64>,
    #[arg(long)]
    max_iterations: Option<usize>,
}

impl From<PolicyIterationArgs> for PolicyIterationConfig {
    fn from(args: PolicyIterationArgs) -> Self {
        Self {
            gamma: args.gamma,
            tolerance: args.tolerance,
            max_iterations: args.max_iterations,
        }
    }
}
