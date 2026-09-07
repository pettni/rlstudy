use crate::environment::Environment;
use crate::model::TabularModel;
use crate::policy::TabularPolicy;

pub mod policy_iteration;

pub use policy_iteration::{StoppingCriterion, TruncatedPolicyIteration};

#[derive(Debug, Clone, PartialEq)]
pub struct SolveResult {
    pub policy: TabularPolicy,
    pub value: Vec<f64>,
    pub iterations: usize,
    pub termination: TerminationReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    Converged,
    IterationLimit,
}

pub trait ModelBasedSolver {
    fn solve(&self, model: &TabularModel, gamma: f64) -> Result<SolveResult, String>;
}

pub trait ModelFreeSolver {
    fn solve(
        &self,
        environment: &mut dyn Environment<State = usize, Action = usize>,
        gamma: f64,
        rng: &mut dyn rand::Rng,
    ) -> Result<SolveResult, String>;
}
