use crate::environment::Environment;
use crate::model::TabularModel;
use crate::policy::DiscretePolicy;
use crate::reporting::{AbstractReporter, Reporter, ReporterConfig};

pub mod actor_critic;
pub mod dqn;
pub mod monte_carlo;
pub mod policy_iteration;
pub mod reinforce;
pub mod td;

pub use actor_critic::{ActorCritic, ActorCriticConfig};
pub use dqn::{DeepQLearning, DqnArgs, DqnConfig, DqnMode, Loss, Schedule, TargetUpdateConfig};
pub use monte_carlo::{MonteCarlo, MonteCarloConfig};
pub use policy_iteration::{
    PolicyIterationConfig, StoppingCriterion, TruncatedPolicyIteration,
    TruncatedPolicyIterationConfig, ValueIterationConfig,
};
pub use reinforce::{Reinforce, ReinforceConfig};
pub use td::{TD, TDConfig, TDVariant};

pub trait ModelBasedSolver {
    type Observation;

    fn solve_with_reporter(
        &self,
        model: &TabularModel,
        reporter: &mut dyn AbstractReporter<Self::Observation>,
    ) -> Result<SolveResult<Self::Observation>, String>;

    fn solve(&self, model: &TabularModel) -> Result<SolveResult<Self::Observation>, String> {
        let mut reporter = Reporter::new(ReporterConfig::default(), 0, None);
        self.solve_with_reporter(model, &mut reporter)
    }
}

pub trait ModelFreeSolver {
    type Observation;

    fn solve_with_reporter(
        &self,
        env: &mut dyn Environment<Observation = Self::Observation, Action = usize>,
        rng: &mut dyn rand::Rng,
        reporter: &mut dyn AbstractReporter<Self::Observation>,
    ) -> Result<SolveResult<Self::Observation>, String>;

    fn solve(
        &self,
        env: &mut dyn Environment<Observation = Self::Observation, Action = usize>,
        rng: &mut dyn rand::Rng,
    ) -> Result<SolveResult<Self::Observation>, String> {
        let mut reporter = Reporter::new(ReporterConfig::default(), 0, None);
        self.solve_with_reporter(env, rng, &mut reporter)
    }
}

pub struct SolveResult<Observation> {
    pub policy: Box<dyn DiscretePolicy<Observation = Observation>>,
    pub status: SolveStatus,
}

pub type TabularSolveResult = SolveResult<usize>;

fn validate_gamma(gamma: f64) -> Result<(), String> {
    if !gamma.is_finite() || !(0.0..1.0).contains(&gamma) {
        return Err("gamma must be finite and in [0, 1)".into());
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    Converged { iterations: usize },
    IterationLimit { iterations: usize },
}

impl SolveStatus {
    pub fn iterations(self) -> usize {
        match self {
            Self::Converged { iterations } | Self::IterationLimit { iterations } => iterations,
        }
    }
}
