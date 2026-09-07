use crate::algorithms::{
    ModelBasedSolver, SolveResult, SolveTrace, TerminationReason, ValueSnapshot,
};
use crate::model::TabularModel;
use crate::policy::TabularPolicy;
use ndarray::{Array1, Array2, Array3, Axis, Ix2};
use ndarray_einsum::einsum;
use num_traits::abs;

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
    ) -> Option<TerminationReason> {
        if self.tolerance.is_some_and(|tolerance| {
            (previous - current)
                .mapv(abs)
                .iter()
                .fold(0.0f64, |a, &b| a.max(b))
                <= tolerance
        }) {
            Some(TerminationReason::Converged)
        } else if iter >= self.max_iterations {
            Some(TerminationReason::IterationLimit)
        } else {
            None
        }
    }
}

pub struct TruncatedPolicyIteration {
    /// Outer loop convergence criterion
    outer: StoppingCriterion,
    /// Inner loop stopping crietrion
    ///  1 special case: value iteration
    ///  inf special case: policy iteration
    inner: StoppingCriterion,
}

struct TraceRecorder {
    evaluation_sweep: usize,
    snapshots: Vec<ValueSnapshot>,
}

impl TraceRecorder {
    fn record(&mut self, policy_iteration: usize, value: &Array1<f64>) {
        self.evaluation_sweep += 1;
        self.snapshots.push(ValueSnapshot {
            policy_iteration,
            evaluation_sweep: self.evaluation_sweep,
            value: value.to_vec(),
        });
    }
}

impl TruncatedPolicyIteration {
    pub fn new(outer: StoppingCriterion, inner: StoppingCriterion) -> Self {
        Self { outer, inner }
    }

    pub fn value_iteration(stopping: StoppingCriterion) -> Self {
        Self {
            outer: stopping,
            inner: StoppingCriterion::iterations(1).unwrap(),
        }
    }

    pub fn policy_iteration(stopping: StoppingCriterion) -> Self {
        Self {
            outer: stopping,
            inner: StoppingCriterion::convergence(1e-6, 10_000).unwrap(),
        }
    }

    pub fn solve_with_trace(&self, model: &TabularModel, gamma: f64) -> Result<SolveTrace, String> {
        if !gamma.is_finite() || !(0.0..1.0).contains(&gamma) {
            return Err("gamma must be finite and in [0, 1)".into());
        }
        let n_s = model.states().len();
        let n_a = model.actions().len();
        let mut trace = TraceRecorder {
            evaluation_sweep: 0,
            snapshots: vec![ValueSnapshot {
                policy_iteration: 0,
                evaluation_sweep: 0,
                value: vec![0.0; n_s],
            }],
        };

        let mut pi = TabularPolicy::uniform(n_s, n_a).expect("model has states and actions");
        let mut v_pi = policy_evaluation(
            model,
            &pi,
            &Array1::<f64>::zeros(pi.num_states()),
            gamma,
            &self.inner,
            0,
            &mut trace,
        );

        for iter in 1.. {
            let new_pi = improve_policy(model, &v_pi, gamma);
            let new_v_pi =
                policy_evaluation(model, &new_pi, &v_pi, gamma, &self.inner, iter, &mut trace);
            let termination = self.outer.stop_reason(iter, &v_pi, &new_v_pi);

            pi = new_pi;
            v_pi = new_v_pi;

            if let Some(termination) = termination {
                return Ok(SolveTrace {
                    result: SolveResult {
                        policy: pi,
                        value: v_pi.to_vec(),
                        iterations: iter,
                        termination,
                    },
                    snapshots: trace.snapshots,
                });
            }
        }

        unreachable!("TruncatedPolicyIteration: unreachable");
    }
}

impl ModelBasedSolver for TruncatedPolicyIteration {
    fn solve(&self, model: &TabularModel, gamma: f64) -> Result<SolveResult, String> {
        Ok(self.solve_with_trace(model, gamma)?.result)
    }
}

fn policy_evaluation(
    model: &TabularModel,
    pi: &TabularPolicy,
    v_last: &Array1<f64>,
    gamma: f64,
    sc: &StoppingCriterion,
    policy_iteration: usize,
    trace: &mut TraceRecorder,
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
        trace.record(policy_iteration, &new_v_pi);
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
    let q_sa = &r_sa + gamma * einsum("sat,t->sa", &[&p_sas.view(), &v_s.view()]).unwrap();

    // new policy is row-wise argmax over Q(s,a)
    let actions: Vec<usize> = q_sa
        .axis_iter(Axis(0))
        .map(|row| {
            let mut best_a = 0;
            let mut best_q = row[0];
            for a in 1..n_a {
                if row[a] > best_q {
                    best_q = row[a];
                    best_a = a;
                }
            }
            best_a
        })
        .collect();

    TabularPolicy::deterministic(n_a, actions).expect("model has actions")
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
    let num_states = model.states().len();
    let num_actions = model.actions().len();
    let mut tensor = Array3::zeros((num_states, num_actions, num_states));

    for state in 0..num_states {
        for action in 0..num_actions {
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
    einsum(
        "sat,sa->st",
        &[
            &transition_tensor(model).view(),
            &policy_matrix(policy).view(),
        ],
    )
    .expect("einsum contraction failed")
    .into_dimensionality::<Ix2>()
    .expect("einsum result should be 2D")
}
