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

pub struct TruncatedPolicyIteration {
    /// Outer loop convergence criterion
    tolerance: f64,
    /// Inner loop stopping crietrion
    ///  1 special case: value iteration
    ///  inf special case: policy iteration
    stopping_criterion: StoppingCriterion,
    /// Discount factor 0 < gamma < 1
    gamma: f64,
}

impl TruncatedPolicyIteration {
    pub fn new(tolerance: f64, stopping_criterion: StoppingCriterion, gamma: f64) -> Self {
        Self {
            tolerance,
            stopping_criterion,
            gamma,
        }
    }
}

impl ModelBasedSolver for TruncatedPolicyIteration {
    fn solve(&mut self, model: &Model) -> SolveResult {
        let n_s = model.states.len();
        let n_a = model.actions.len();

        let mut pi: Box<dyn Policy> = Box::new(StochasticPolicy::uniform(n_s, n_a));
        let mut v_pi = policy_evaluation(
            model,
            pi.as_ref(),
            &Array1::<f64>::zeros(pi.num_states()),
            self.gamma,
            &self.stopping_criterion,
        );

        for _ in 0..1000 {
            let new_pi = improve_policy(model, &v_pi, self.gamma);
            let new_v_pi = policy_evaluation(
                model,
                new_pi.as_ref(),
                &v_pi,
                self.gamma,
                &self.stopping_criterion,
            );

            let should_break = (&v_pi - &new_v_pi)
                .mapv(abs)
                .iter()
                .fold(0.0f64, |a, &b| a.max(b))
                < self.tolerance;

            pi = new_pi;
            v_pi = new_v_pi;

            if should_break {
                break;
            }
        }

        SolveResult {
            policy: pi,
            value: v_pi.to_vec(),
        }
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
        let v_pi_new = &r_pi_s + gamma * p_pi_s_s.dot(&v_pi);

        match sc {
            StoppingCriterion::Iterations(max_iter) => {
                if iter >= *max_iter {
                    return v_pi_new;
                }
            }
            StoppingCriterion::Tolerance(tol) => {
                if check_tol(&v_pi_new, &v_pi, *tol) {
                    return v_pi_new;
                }
            }
        }

        v_pi = v_pi_new;
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

fn check_tol(a1: &Array1<f64>, a2: &Array1<f64>, tol: f64) -> bool {
    let max_diff = (a1 - a2).mapv(abs).iter().fold(0.0f64, |a, &b| a.max(b));
    max_diff <= tol
}
