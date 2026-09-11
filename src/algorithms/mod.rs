use crate::environment::Environment;
use crate::model::TabularModel;
use crate::policy::TabularPolicy;

pub mod monte_carlo;
pub mod policy_iteration;
pub mod sarsa;

pub use monte_carlo::MonteCarlo;
pub use policy_iteration::{StoppingCriterion, TruncatedPolicyIteration};
pub use sarsa::Sarsa;

#[derive(Debug, Clone, PartialEq)]
pub struct SolveResult {
    pub policy: TabularPolicy,
    pub value: Vec<f64>,
    pub iterations: usize,
    pub termination: TerminationReason,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValueSnapshot {
    pub policy_iteration: usize,
    pub evaluation_sweep: usize,
    pub value: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SolveTrace {
    pub result: SolveResult,
    pub snapshots: Vec<ValueSnapshot>,
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
        env: &mut dyn Environment<State = usize, Action = usize>,
        gamma: f64,
        rng: &mut dyn rand::Rng,
    ) -> Result<SolveResult, String>;
}
