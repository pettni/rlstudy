use crate::environment::Environment;
use crate::model::TabularModel;
use crate::policy::DiscretePolicy;

pub mod dqn;
pub mod monte_carlo;
pub mod policy_iteration;
pub mod td;

pub use dqn::{DeepQLearning, DqnSnapshot, TargetUpdate};
pub use monte_carlo::MonteCarlo;
pub use policy_iteration::{StoppingCriterion, TruncatedPolicyIteration};
pub use td::TD;

pub struct SolveResult<Observation, Output> {
    pub policy: Box<dyn DiscretePolicy<Observation = Observation>>,
    pub output: Output,
    pub status: SolveStatus,
}

pub type TabularSolveResult = SolveResult<usize, Vec<f64>>;

#[derive(Debug, Clone, PartialEq)]
pub struct ValueSnapshot {
    pub policy_iteration: usize,
    pub evaluation_sweep: usize,
    pub value: Vec<f64>,
}

pub struct SolveTrace {
    pub result: TabularSolveResult,
    pub snapshots: Vec<ValueSnapshot>,
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

pub trait ModelBasedSolver {
    type Observation;
    type Output;

    fn solve(
        &self,
        model: &TabularModel,
        gamma: f64,
    ) -> Result<SolveResult<Self::Observation, Self::Output>, String>;
}

pub trait ModelFreeSolver {
    type Observation;
    type Output;

    fn solve(
        &self,
        env: &mut dyn Environment<Observation = Self::Observation, Action = usize>,
        gamma: f64,
        rng: &mut dyn rand::Rng,
    ) -> Result<SolveResult<Self::Observation, Self::Output>, String>;
}
