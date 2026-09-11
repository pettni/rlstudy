use crate::algorithms::{ModelFreeSolver, SolveResult, TerminationReason};
use crate::environment::{Environment, Space};
use crate::policy::{Policy, PolicyObjective, TabularPolicy};
use ndarray::{Array2, ArrayView1};

/// TD-family of model-free control algorithm.
///
/// Given (s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1}) update the q estimate as
///
///  q(s_t,a_t) = q(s_t,a_t) - alpha * (q(s_t, a_t) - td_target),
///
/// where td_target =
///     r_{t+1} + gamma * q(s_{t+1}, a_{t+1})               [Sarsa]
///     r_{t+1} + gamma * E_pi q(s_{t+1}, pi(s_{t+1}))      [Expected Sarsa]
///     r_{t+1} + gamma * max_a q(s_{t+1}, a)               [Q-learning]
///

#[derive(Clone, Copy)]
pub enum TDVariant {
    /// Basic variant
    Sarsa,
    /// TD target from expected future state
    ExpectedSarsa,
    /// TD target from the greedy next-state action
    QLearning,
}

pub struct TD {
    // number of episodes
    episodes: u64,
    // episode length
    episode_length: u64,
    // soft action TODO: Support an epsilon schedule that decreases across episodes.
    epsilon: f64,
    // learning rate
    alpha: f64,
    // algorithm type
    variant: TDVariant,
}

impl TD {
    pub fn new(
        episodes: u64,
        episode_length: u64,
        epsilon: f64,
        alpha: f64,
    ) -> Result<Self, String> {
        Self::with_variant(episodes, episode_length, epsilon, alpha, TDVariant::Sarsa)
    }

    pub fn expected(
        episodes: u64,
        episode_length: u64,
        epsilon: f64,
        alpha: f64,
    ) -> Result<Self, String> {
        Self::with_variant(
            episodes,
            episode_length,
            epsilon,
            alpha,
            TDVariant::ExpectedSarsa,
        )
    }

    pub fn q_learning(
        episodes: u64,
        episode_length: u64,
        epsilon: f64,
        alpha: f64,
    ) -> Result<Self, String> {
        Self::with_variant(
            episodes,
            episode_length,
            epsilon,
            alpha,
            TDVariant::QLearning,
        )
    }

    fn with_variant(
        episodes: u64,
        episode_length: u64,
        epsilon: f64,
        alpha: f64,
        variant: TDVariant,
    ) -> Result<Self, String> {
        if episodes == 0 {
            return Err("episodes must be positive".into());
        }
        if episode_length == 0 {
            return Err("episode length must be positive".into());
        }
        if !epsilon.is_finite() || !(0.0..=1.0).contains(&epsilon) {
            return Err("epsilon must be finite and in [0, 1]".into());
        }
        if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
            return Err("alpha must be finite and in (0, 1]".into());
        }
        Ok(Self {
            episodes,
            episode_length,
            epsilon,
            alpha,
            variant,
        })
    }
}

impl ModelFreeSolver for TD {
    fn solve(
        &self,
        env: &mut dyn Environment<State = usize, Action = usize>,
        gamma: f64,
        rng: &mut dyn rand::Rng,
    ) -> Result<SolveResult, String> {
        if !gamma.is_finite() || !(0.0..1.0).contains(&gamma) {
            return Err("gamma must be finite and in [0, 1)".into());
        }
        let Space::Discrete { size: n_s } = env.state_space() else {
            return Err("TD requires a discrete state space".into());
        };
        let Space::Discrete { size: n_a } = env.action_space() else {
            return Err("TD requires a discrete action space".into());
        };

        let mut q_sa = Array2::<f64>::zeros((n_s, n_a));
        let mut pi =
            TabularPolicy::from_q(&q_sa, PolicyObjective::SoftDeterministic(self.epsilon))?;

        for _e in 0..self.episodes {
            let mut st = env.reset(rng);
            let mut at = pi.sample_action(&st, rng);

            for _ in 0..self.episode_length {
                let (stp, rt) = env.step(at, rng);
                let atp = pi.sample_action(&stp, rng);

                // sarsa is (st, at, rt, stp, atp)

                // value update
                let q_next = match self.variant {
                    TDVariant::Sarsa => q_sa[[stp, atp]],
                    TDVariant::ExpectedSarsa => {
                        let p_a = pi.state_probabilities(stp);
                        q_sa.row(stp).dot(&p_a)
                    }
                    TDVariant::QLearning => max_q(q_sa.row(stp)),
                };
                let td_target = rt + gamma * q_next;
                q_sa[[st, at]] -= self.alpha * (q_sa[[st, at]] - td_target);

                // policy improvement at st
                pi.update_at(
                    st,
                    &q_sa.row(st),
                    PolicyObjective::SoftDeterministic(self.epsilon),
                );

                st = stp;
                at = atp;
            }
        }

        let (policy, value) = match self.variant {
            TDVariant::QLearning => {
                let value = (0..n_s).map(|s| max_q(q_sa.row(s))).collect();
                let policy = TabularPolicy::from_q(&q_sa, PolicyObjective::Deterministic)?;
                (policy, value)
            }
            TDVariant::Sarsa | TDVariant::ExpectedSarsa => {
                let value = (0..n_s)
                    .map(|s| (0..n_a).map(|a| pi.probability(s, a) * q_sa[[s, a]]).sum())
                    .collect();
                (pi, value)
            }
        };

        Ok(SolveResult {
            policy,
            value,
            iterations: self
                .episodes
                .try_into()
                .map_err(|_| "episode count does not fit usize")?,
            termination: TerminationReason::IterationLimit,
        })
    }
}

fn max_q(q_a: ArrayView1<'_, f64>) -> f64 {
    q_a.iter()
        .copied()
        .reduce(f64::max)
        .expect("action space has actions")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_is_validated() {
        assert!(TD::new(0, 10, 0.1, 0.1).is_err());
        assert!(TD::new(10, 0, 0.1, 0.1).is_err());
        assert!(TD::new(10, 10, -0.1, 0.1).is_err());
        assert!(TD::new(10, 10, 0.1, 0.0).is_err());
        assert!(TD::new(10, 10, 0.1, 1.1).is_err());
        assert!(TD::new(10, 10, 0.1, 0.1).is_ok());
        assert!(TD::expected(10, 10, 0.1, 0.1).is_ok());
        assert!(TD::q_learning(10, 10, 0.1, 0.1).is_ok());
    }
}
