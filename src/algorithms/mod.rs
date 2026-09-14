use crate::environment::Environment;
use crate::model::TabularModel;
use crate::policy::DiscretePolicy;

pub mod dqn;
pub mod monte_carlo;
pub mod policy_iteration;
pub mod td;

pub use dqn::{DeepQLearning, DqnConfig, TargetUpdate, TargetUpdateMethod};
pub use monte_carlo::{MonteCarlo, MonteCarloConfig};
pub use policy_iteration::{
    PolicyIterationConfig, StoppingCriterion, TruncatedPolicyIteration,
    TruncatedPolicyIterationConfig, ValueIterationConfig,
};
pub use td::{TD, TDConfig, TDVariant};

pub struct SolveResult<Observation> {
    pub policy: Box<dyn DiscretePolicy<Observation = Observation>>,
    pub snapshots: Vec<SolverSnapshot>,
    pub status: SolveStatus,
}

pub type TabularSolveResult = SolveResult<usize>;

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SolverSnapshot {
    pub iteration: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy_iteration: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation_sweep: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_values: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub average_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation_return: Option<f64>,
}

impl SolverSnapshot {
    pub fn new(iteration: usize) -> Self {
        Self {
            iteration,
            policy_iteration: None,
            evaluation_sweep: None,
            state_values: None,
            state_value: None,
            average_loss: None,
            learning_rate: None,
            evaluation_return: None,
        }
    }

    pub fn policy_iteration(mut self, policy_iteration: usize) -> Self {
        self.policy_iteration = Some(policy_iteration);
        self
    }

    pub fn evaluation_sweep(mut self, evaluation_sweep: usize) -> Self {
        self.evaluation_sweep = Some(evaluation_sweep);
        self
    }

    pub fn state_values(mut self, state_values: Vec<f64>) -> Self {
        self.state_values = Some(state_values);
        self
    }

    pub fn state_value(mut self, state_value: f64) -> Self {
        self.state_value = Some(state_value);
        self
    }

    pub fn average_loss(mut self, average_loss: f64) -> Self {
        self.average_loss = Some(average_loss);
        self
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = Some(learning_rate);
        self
    }

    pub fn evaluation_return(mut self, evaluation_return: f64) -> Self {
        self.evaluation_return = Some(evaluation_return);
        self
    }

    pub fn evaluate_policy<Observation, Action>(
        self,
        environment: &mut dyn Environment<Observation = Observation, Action = Action>,
        policy: &dyn crate::policy::Policy<Observation = Observation, Action = Action>,
        config: crate::evaluation::EvaluationConfig,
        rng: &mut dyn rand::Rng,
    ) -> Result<Self, String> {
        let evaluation_return =
            crate::evaluation::evaluate_policy(environment, policy, config, rng)?;
        let snapshot = self.evaluation_return(evaluation_return);
        Ok(match config.mode() {
            crate::evaluation::EvaluationMode::DiscountedReward { .. } => {
                snapshot.state_value(evaluation_return)
            }
            crate::evaluation::EvaluationMode::AverageReward => snapshot,
        })
    }
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

    fn solve(
        &self,
        model: &TabularModel,
        gamma: f64,
    ) -> Result<SolveResult<Self::Observation>, String>;
}

pub trait ModelFreeSolver {
    type Observation;

    fn solve(
        &self,
        env: &mut dyn Environment<Observation = Self::Observation, Action = usize>,
        gamma: f64,
        rng: &mut dyn rand::Rng,
    ) -> Result<SolveResult<Self::Observation>, String>;
}
