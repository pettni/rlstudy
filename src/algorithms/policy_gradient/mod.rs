pub(super) mod deterministic;
pub(super) mod stochastic;

pub use deterministic::DeterministicPolicyGradient;
pub use stochastic::StochasticPolicyGradient;

pub(super) use deterministic::{DeterministicPolicyGradientConfig, TargetNoise};
pub(super) use stochastic::{PolicyGradientConfig, PolicyObjective, PolicyUpdateStrategy};
