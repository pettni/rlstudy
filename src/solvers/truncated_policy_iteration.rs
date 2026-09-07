use crate::model::Model;
use crate::policy::{DeterministicPolicy, Policy, StochasticPolicy};
use crate::solvers::{ModelBasedSolver, SolveResult, get_s_r, get_s_t, get_sa_t};
use ndarray::{Array1, Array2, Axis};
use ndarray_einsum::einsum;
use num_traits::abs;

pub enum StoppingCriterion {
    Iterations(u64),
    Tolerance(f64),
}

impl StoppingCriterion {
    pub fn should_stop(&self, iter: u64, a1: &Array1<f64>, a2: &Array1<f64>) -> bool {
        match self {
            StoppingCriterion::Iterations(max_iter) => iter >= *max_iter,
            StoppingCriterion::Tolerance(tol) => {
                let max_diff = (a1 - a2).mapv(abs).iter().fold(0.0f64, |a, &b| a.max(b));
                max_diff <= *tol
            }
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

impl TruncatedPolicyIteration {
    pub fn new(outer: StoppingCriterion, inner: StoppingCriterion) -> Self {
        if let StoppingCriterion::Iterations(u64::MAX) = outer {
            panic!("Outer can't have max-iterations infty");
        }

        Self { outer, inner }
    }

    pub fn value_iteration(stopping: StoppingCriterion) -> Self {
        Self {
            outer: stopping,
            inner: StoppingCriterion::Iterations(1),
        }
    }

    pub fn policy_iteration(stopping: StoppingCriterion) -> Self {
        Self {
            outer: stopping,
            inner: StoppingCriterion::Tolerance(1e-6),
        }
    }
}

impl ModelBasedSolver for TruncatedPolicyIteration {
    fn solve(&mut self, model: &Model, gamma: f64) -> SolveResult {
        let n_s = model.states.len();
        let n_a = model.actions.len();

        let mut pi: Box<dyn Policy> = Box::new(StochasticPolicy::uniform(n_s, n_a));
        let mut v_pi = policy_evaluation(
            model,
            pi.as_ref(),
            &Array1::<f64>::zeros(pi.num_states()),
            gamma,
            &self.inner,
        );

        for iter in 1.. {
            let new_pi = improve_policy(model, &v_pi, gamma);
            let new_v_pi = policy_evaluation(model, new_pi.as_ref(), &v_pi, gamma, &self.inner);

            let should_break = self.outer.should_stop(iter, &v_pi, &new_v_pi);

            pi = new_pi;
            v_pi = new_v_pi;

            if should_break {
                return SolveResult {
                    policy: pi,
                    value: v_pi.to_vec(),
                    iterations: iter,
                };
            }
        }

        unreachable!("TruncatedPolicyIteration: unreachable");
    }
}

fn policy_evaluation(
    model: &Model,
    pi: &dyn Policy,
    v_last: &Array1<f64>,
    gamma: f64,
    sc: &StoppingCriterion,
) -> Array1<f64> {
    // R_pi(s)
    let r_pi_s = get_s_r(model, pi);
    // P_pi(s)
    let p_pi_s_s = get_s_t(model, pi);

    // Start from previous value function
    let mut v_pi = v_last.clone();
    for iter in 1.. {
        // V_pi(s) = r_pi(s) + γ * sum_{s'} P_pi(s,s') v(s')
        let new_v_pi = &r_pi_s + gamma * p_pi_s_s.dot(&v_pi);
        if sc.should_stop(iter, &v_pi, &new_v_pi) {
            return new_v_pi;
        }

        v_pi = new_v_pi;
    }

    unreachable!()
}

fn improve_policy(model: &Model, v_s: &Array1<f64>, gamma: f64) -> Box<dyn Policy> {
    let n_s = model.states.len();
    let n_a = model.actions.len();

    // transition tensor P(s,a,s') and reward matrix R(s,a).
    let p_sas = get_sa_t(model);
    let r_sa = Array2::from_shape_fn((n_s, n_a), |(s, a)| model.r_sa[s][a]);

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

    Box::new(DeterministicPolicy::new(n_a, actions))
}
