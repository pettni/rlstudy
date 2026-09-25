use super::{PolicyIterationConfig, TruncatedPolicyIterationConfig, ValueIterationConfig};
use clap::Args;

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

#[derive(Debug, Clone, PartialEq, Args)]
pub struct TruncatedPolicyIterationArgs {
    #[arg(long, default_value_t = TruncatedPolicyIterationConfig::default().gamma)]
    gamma: f64,
    #[arg(long)]
    outer_tolerance: Option<f64>,
    #[arg(long)]
    outer_max_iterations: Option<usize>,
    #[arg(long)]
    inner_tolerance: Option<f64>,
    #[arg(long)]
    inner_max_iterations: Option<usize>,
}

impl From<TruncatedPolicyIterationArgs> for TruncatedPolicyIterationConfig {
    fn from(args: TruncatedPolicyIterationArgs) -> Self {
        Self {
            gamma: args.gamma,
            outer_tolerance: args.outer_tolerance,
            outer_max_iterations: args.outer_max_iterations,
            inner_tolerance: args.inner_tolerance,
            inner_max_iterations: args.inner_max_iterations,
        }
    }
}
