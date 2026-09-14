use crate::algorithms::{
    ModelBasedSolver, SolveResult, SolveStatus, SolverSnapshot, validate_gamma,
};
use crate::model::TabularModel;
use crate::policy::TabularPolicy;
use clap::Args;
use ndarray::{Array1, Array2, Array3};
use num_traits::abs;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Args, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ValueIterationConfig {
    #[arg(long, default_value_t = ValueIterationConfig::default().gamma)]
    pub gamma: f64,
    #[arg(long)]
    pub tolerance: Option<f64>,
    #[arg(long)]
    pub max_iterations: Option<usize>,
}

impl Default for ValueIterationConfig {
    fn default() -> Self {
        Self {
            gamma: 0.9,
            tolerance: None,
            max_iterations: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Args, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct PolicyIterationConfig {
    #[arg(long, default_value_t = PolicyIterationConfig::default().gamma)]
    pub gamma: f64,
    #[arg(long)]
    pub tolerance: Option<f64>,
    #[arg(long)]
    pub max_iterations: Option<usize>,
}

impl Default for PolicyIterationConfig {
    fn default() -> Self {
        Self {
            gamma: 0.9,
            tolerance: None,
            max_iterations: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Args, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct TruncatedPolicyIterationConfig {
    #[arg(long, default_value_t = TruncatedPolicyIterationConfig::default().gamma)]
    pub gamma: f64,
    #[arg(long)]
    pub outer_tolerance: Option<f64>,
    #[arg(long)]
    pub outer_max_iterations: Option<usize>,
    #[arg(long)]
    pub inner_tolerance: Option<f64>,
    #[arg(long)]
    pub inner_max_iterations: Option<usize>,
}

impl Default for TruncatedPolicyIterationConfig {
    fn default() -> Self {
        Self {
            gamma: 0.9,
            outer_tolerance: None,
            outer_max_iterations: None,
            inner_tolerance: None,
            inner_max_iterations: None,
        }
    }
}

pub struct StoppingCriterion {
    tolerance: Option<f64>,
    max_iterations: usize,
}

impl StoppingCriterion {
    pub fn iterations(max_iterations: usize) -> Result<Self, String> {
        Self::new(None, max_iterations)
    }

    pub fn convergence(tolerance: f64, max_iterations: usize) -> Result<Self, String> {
        Self::new(Some(tolerance), max_iterations)
    }

    fn new(tolerance: Option<f64>, max_iterations: usize) -> Result<Self, String> {
        if max_iterations == 0 {
            return Err("max iterations must be positive".into());
        }
        if tolerance.is_some_and(|value| !value.is_finite() || value <= 0.0) {
            return Err("tolerance must be finite and positive".into());
        }
        Ok(Self {
            tolerance,
            max_iterations,
        })
    }

    fn stop_reason(
        &self,
        iter: usize,
        previous: &Array1<f64>,
        current: &Array1<f64>,
    ) -> Option<SolveStatus> {
        if self.tolerance.is_some_and(|tolerance| {
            (previous - current)
                .mapv(abs)
                .iter()
                .fold(0.0f64, |a, &b| a.max(b))
                <= tolerance
        }) {
            Some(SolveStatus::Converged { iterations: iter })
        } else if iter >= self.max_iterations {
            Some(SolveStatus::IterationLimit { iterations: iter })
        } else {
            None
        }
    }
}

pub struct TruncatedPolicyIteration {
    gamma: f64,
    /// Outer loop convergence criterion
    outer: StoppingCriterion,
    /// Inner loop stopping crietrion
    ///  1 special case: value iteration
    ///  inf special case: policy iteration
    inner: StoppingCriterion,
}

impl TruncatedPolicyIteration {
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    pub fn new(config: TruncatedPolicyIterationConfig) -> Result<Self, String> {
        Self::with_stopping_criteria(
            stopping_criterion(config.outer_tolerance, config.outer_max_iterations)?,
            stopping_criterion(config.inner_tolerance, config.inner_max_iterations)?,
            config.gamma,
        )
    }

    pub fn with_stopping_criteria(
        outer: StoppingCriterion,
        inner: StoppingCriterion,
        gamma: f64,
    ) -> Result<Self, String> {
        validate_gamma(gamma)?;
        Ok(Self {
            gamma,
            outer,
            inner,
        })
    }

    pub fn value_iteration(config: ValueIterationConfig) -> Result<Self, String> {
        validate_gamma(config.gamma)?;
        Ok(Self {
            gamma: config.gamma,
            outer: stopping_criterion(config.tolerance, config.max_iterations)?,
            inner: StoppingCriterion::iterations(1).unwrap(),
        })
    }

    pub fn policy_iteration(config: PolicyIterationConfig) -> Result<Self, String> {
        validate_gamma(config.gamma)?;
        Ok(Self {
            gamma: config.gamma,
            outer: stopping_criterion(config.tolerance, config.max_iterations)?,
            inner: StoppingCriterion::convergence(1e-6, 10_000).unwrap(),
        })
    }
}

impl ModelBasedSolver for TruncatedPolicyIteration {
    type Observation = usize;

    fn solve(&self, model: &TabularModel) -> Result<crate::algorithms::TabularSolveResult, String> {
        let n_s = model.states().len();
        let n_a = model.actions().len();
        let mut snapshots = vec![
            SolverSnapshot::new(0)
                .policy_iteration(0)
                .evaluation_sweep(0)
                .state_values(vec![0.0; n_s])
                .state_value(0.0),
        ];

        let mut pi = TabularPolicy::uniform(n_s, n_a).expect("model has states and actions");
        let mut v_pi = policy_evaluation(
            model,
            &pi,
            &Array1::<f64>::zeros(pi.num_states()),
            self.gamma,
            &self.inner,
            0,
            &mut snapshots,
        );

        for iter in 1.. {
            let new_pi = improve_policy(model, &v_pi, self.gamma);
            let new_v_pi = policy_evaluation(
                model,
                &new_pi,
                &v_pi,
                self.gamma,
                &self.inner,
                iter,
                &mut snapshots,
            );
            let status = self.outer.stop_reason(iter, &v_pi, &new_v_pi);

            pi = new_pi;
            v_pi = new_v_pi;

            if let Some(status) = status {
                return Ok(SolveResult {
                    policy: Box::new(pi),
                    snapshots,
                    status,
                });
            }
        }

        unreachable!("TruncatedPolicyIteration: unreachable");
    }
}

fn stopping_criterion(
    tolerance: Option<f64>,
    max_iterations: Option<usize>,
) -> Result<StoppingCriterion, String> {
    match (tolerance, max_iterations) {
        (Some(tolerance), Some(iterations)) => {
            StoppingCriterion::convergence(tolerance, iterations)
        }
        (Some(tolerance), None) => StoppingCriterion::convergence(tolerance, 10_000),
        (None, Some(iterations)) => StoppingCriterion::iterations(iterations),
        (None, None) => StoppingCriterion::convergence(1e-6, 10_000),
    }
}

fn policy_evaluation(
    model: &TabularModel,
    pi: &TabularPolicy,
    v_last: &Array1<f64>,
    gamma: f64,
    sc: &StoppingCriterion,
    policy_iteration: usize,
    snapshots: &mut Vec<SolverSnapshot>,
) -> Array1<f64> {
    // R_pi(s)
    let r_pi_s = policy_reward(model, pi);
    // P_pi(s)
    let p_pi_s_s = policy_transition_matrix(model, pi);

    // Start from previous value function
    let mut v_pi = v_last.clone();
    for iter in 1.. {
        // V_pi(s) = r_pi(s) + γ * sum_{s'} P_pi(s,s') v(s')
        let new_v_pi = &r_pi_s + gamma * p_pi_s_s.dot(&v_pi);
        let evaluation_sweep = snapshots.len();
        let state_value = (0..model.states().len())
            .map(|state| model.initial_distribution().probability_of(&state) * new_v_pi[state])
            .sum();
        snapshots.push(
            SolverSnapshot::new(evaluation_sweep)
                .policy_iteration(policy_iteration)
                .evaluation_sweep(evaluation_sweep)
                .state_values(new_v_pi.to_vec())
                .state_value(state_value),
        );
        if sc.stop_reason(iter, &v_pi, &new_v_pi).is_some() {
            return new_v_pi;
        }

        v_pi = new_v_pi;
    }

    unreachable!()
}

fn improve_policy(model: &TabularModel, v_s: &Array1<f64>, gamma: f64) -> TabularPolicy {
    let n_s = model.states().len();
    let n_a = model.actions().len();

    // transition tensor P(s,a,s') and reward matrix R(s,a).
    let p_sas = transition_tensor(model);
    let r_sa = Array2::from_shape_fn((n_s, n_a), |(s, a)| expected_reward(model, s, a));

    // state-action values
    // Q_pi(s,a) = R(s,a) + γ * sum_{s'} P(s,a,s') * v(s')
    let q_sa = Array2::from_shape_fn((n_s, n_a), |(s, a)| {
        let future_value: f64 = v_s
            .iter()
            .enumerate()
            .map(|(sp, &value)| p_sas[[s, a, sp]] * value)
            .sum();
        r_sa[[s, a]] + gamma * future_value
    });

    // new policy is row-wise argmax over Q(s,a)
    TabularPolicy::from_q(&q_sa, crate::policy::PolicyObjective::Deterministic)
        .expect("model has actions")
}

// Compute P_pi(s,a) for a policy
fn policy_matrix(policy: &TabularPolicy) -> Array2<f64> {
    Array2::from_shape_fn(
        (policy.num_states(), policy.num_actions()),
        |(state, action)| policy.probability(state, action),
    )
}

// Compute P(s, a, s') for a model
fn transition_tensor(model: &TabularModel) -> Array3<f64> {
    let n_s = model.states().len();
    let n_a = model.actions().len();
    let mut tensor = Array3::zeros((n_s, n_a, n_s));

    for state in 0..n_s {
        for action in 0..n_a {
            for (probability, (next_state, _)) in model.outcomes(state, action).iter() {
                tensor[[state, action, *next_state]] += *probability;
            }
        }
    }
    tensor
}

// Compute R(s) for a model and policy
fn policy_reward(model: &TabularModel, policy: &TabularPolicy) -> Array1<f64> {
    Array1::from_shape_fn(model.states().len(), |state| {
        (0..model.actions().len())
            .map(|action| policy.probability(state, action) * expected_reward(model, state, action))
            .sum()
    })
}

fn expected_reward(model: &TabularModel, state: usize, action: usize) -> f64 {
    model
        .outcomes(state, action)
        .iter()
        .map(|(probability, (_, reward))| probability * reward)
        .sum()
}

// Compute P(s, s') for a model and policy
fn policy_transition_matrix(model: &TabularModel, policy: &TabularPolicy) -> Array2<f64> {
    let n_s = model.states().len();
    let n_a = model.actions().len();

    let p_sat = transition_tensor(model);
    let pi_sa = policy_matrix(policy);

    // einsum("sat,st->st", p_sat, pi_st)
    Array2::from_shape_fn((n_s, n_s), |(s, t)| -> f64 {
        (0..n_a).map(|a| p_sat[[s, a, t]] * pi_sa[[s, a]]).sum()
    })
}
