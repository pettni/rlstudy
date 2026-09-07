use crate::model::Model;
use crate::policy::Policy;
use crate::system::System;
use ndarray::{Array1, Array2, Array3, Ix2};
use ndarray_einsum::einsum;

pub mod truncated_policy_iteration;

pub struct SolveResult {
    pub policy: Box<dyn Policy>,
    pub value: Vec<f64>,
    pub iterations: u64,
}

pub trait ModelBasedSolver {
    fn solve(&mut self, model: &Model, gamma: f64) -> SolveResult;
}

pub trait ModelFreeSolver {
    fn solve(&mut self, system: &mut dyn System, gamma: f64) -> SolveResult;
}

/// State-action probability matrix
pub fn get_s_a(policy: &dyn Policy) -> Array2<f64> {
    Array2::from_shape_fn((policy.num_states(), policy.num_actions()), |(s, a)| {
        policy.prob(s, a)
    })
}

/// State-action-state probability tensor
pub fn get_sa_t(model: &Model) -> Array3<f64> {
    let n_s = model.states.len();
    let n_a = model.actions.len();
    let mut tensor = Array3::<f64>::zeros((n_s, n_a, n_s));

    for (s, actions) in model.p_s_sa.iter().enumerate() {
        for (a, dist) in actions.iter().enumerate() {
            for (p, s_next) in dist.iter() {
                tensor[[s, a, *s_next as usize]] = *p as f64;
            }
        }
    }

    tensor
}

/// State reward under a policy
pub fn get_s_r(model: &Model, policy: &dyn Policy) -> Array1<f64> {
    let n_s = model.states.len();
    let mut s_r = Array1::<f64>::zeros(n_s);

    for s in 0..model.states.len() {
        for a in 0..model.actions.len() {
            s_r[[s]] += policy.prob(s, a) * model.r_sa[s][a];
        }
    }

    s_r
}

/// State-state probability matrix
pub fn get_s_t(model: &Model, policy: &dyn Policy) -> Array2<f64> {
    let sa_s = get_sa_t(model);
    let s_a = get_s_a(policy);

    // out[s, s'] = sum_a sa_s[s, a, t] * s_a[s, a]
    // which is einsum sat,sa->st
    einsum("sat,sa->st", &[&sa_s.view(), &s_a.view()])
        .expect("einsum contraction failed")
        .into_dimensionality::<Ix2>()
        .expect("einsum result should be 2D")
}
