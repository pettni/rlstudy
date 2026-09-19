use crate::algorithms::{
    ModelFreeSolver, SolveResult, SolveStatus, SolverSnapshot, validate_gamma,
};
use crate::environment::{Environment, Space, Step};
use crate::policy::{Policy, PolicyObjective, TabularPolicy};
use clap::{Args, ValueEnum};
use ndarray::{Array2, ArrayView1};
use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TDVariant {
    /// Basic variant
    Sarsa,
    /// TD target from expected future state
    ExpectedSarsa,
    /// TD target from the greedy next-state action
    QLearning,
}

#[derive(Debug, Clone, PartialEq, Args, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct TDConfig {
    #[arg(long, default_value_t = TDConfig::default().gamma)]
    pub gamma: f64,
    #[arg(long, default_value_t = TDConfig::default().episodes)]
    pub episodes: u64,
    #[arg(long, default_value_t = TDConfig::default().episode_length)]
    pub episode_length: u64,
    #[arg(long, default_value_t = TDConfig::default().epsilon)]
    pub epsilon: f64,
    #[arg(long, default_value_t = TDConfig::default().alpha)]
    pub alpha: f64,
    #[arg(long, value_enum, default_value_t = TDConfig::default().variant)]
    pub variant: TDVariant,
}

impl Default for TDConfig {
    fn default() -> Self {
        Self {
            gamma: 0.9,
            episodes: 10_000,
            episode_length: 30,
            epsilon: 0.1,
            alpha: 0.1,
            variant: TDVariant::Sarsa,
        }
    }
}

pub struct TD {
    config: TDConfig,
    // base policy for off-policy learning (if any)
    behavior_policy: Option<TabularPolicy>,
}

impl TD {
    pub fn new(config: TDConfig) -> Result<Self, String> {
        validate_gamma(config.gamma)?;
        if config.episodes == 0 {
            return Err("episodes must be positive".into());
        }
        if config.episode_length == 0 {
            return Err("episode length must be positive".into());
        }
        if !config.epsilon.is_finite() || !(0.0..=1.0).contains(&config.epsilon) {
            return Err("epsilon must be finite and in [0, 1]".into());
        }
        if !config.alpha.is_finite() || config.alpha <= 0.0 || config.alpha > 1.0 {
            return Err("alpha must be finite and in (0, 1]".into());
        }
        Ok(Self {
            config,
            behavior_policy: None,
        })
    }

    pub fn with_behavior_policy(mut self, policy: TabularPolicy) -> Self {
        self.behavior_policy = Some(policy);
        self
    }
}

impl ModelFreeSolver for TD {
    type Observation = usize;

    fn solve(
        &self,
        env: &mut dyn Environment<Observation = usize, Action = usize>,
        rng: &mut dyn rand::Rng,
    ) -> Result<crate::algorithms::TabularSolveResult, String> {
        let Space::Discrete { size: n_s } = env.state_space() else {
            return Err("TD requires a discrete state space".into());
        };
        let Space::Discrete { size: n_a } = env.action_space() else {
            return Err("TD requires a discrete action space".into());
        };

        if let Some(policy) = &self.behavior_policy {
            if policy.num_states() != n_s || policy.num_actions() != n_a {
                return Err(format!(
                    "behavior policy has shape {}x{}, expected {n_s}x{n_a}",
                    policy.num_states(),
                    policy.num_actions()
                ));
            }
            if self.config.variant == TDVariant::Sarsa {
                return Err("fixed behavior policies are not supported for Sarsa".into());
            }
        }

        let mut q_sa = Array2::<f64>::zeros((n_s, n_a));
        let mut pi_target = TabularPolicy::uniform(n_s, n_a).unwrap();

        for _e in 0..self.config.episodes {
            let mut st = env.reset(rng);
            let mut at = self
                .behavior_policy
                .as_ref()
                .unwrap_or(&pi_target)
                .sample_action(&st, rng);

            for _ in 0..self.config.episode_length {
                let Step {
                    observation: stp,
                    reward: rt,
                    terminated,
                    truncated,
                } = env.step(at, rng);

                // value update
                let (q_next, atp) = if terminated {
                    (0.0, None)
                } else {
                    match self.config.variant {
                        TDVariant::Sarsa => {
                            let atp = self
                                .behavior_policy
                                .as_ref()
                                .unwrap_or(&pi_target)
                                .sample_action(&stp, rng);
                            (q_sa[[stp, atp]], Some(atp))
                        }
                        TDVariant::ExpectedSarsa => {
                            let p_a = pi_target.state_probabilities(stp);
                            (q_sa.row(stp).dot(&p_a), None)
                        }
                        TDVariant::QLearning => (max_q(q_sa.row(stp)), None),
                    }
                };
                let td_target = rt + self.config.gamma * q_next;
                q_sa[[st, at]] -= self.config.alpha * (q_sa[[st, at]] - td_target);

                // policy improvement at st
                pi_target.update_at(
                    st,
                    &q_sa.row(st),
                    PolicyObjective::SoftDeterministic(self.config.epsilon),
                );

                if terminated || truncated {
                    break;
                }

                st = stp;
                at = match atp {
                    Some(atp) => atp,
                    None => self
                        .behavior_policy
                        .as_ref()
                        .unwrap_or(&pi_target)
                        .sample_action(&st, rng),
                };
            }
        }

        let (policy, value) = match self.config.variant {
            TDVariant::QLearning => {
                let value = (0..n_s).map(|s| max_q(q_sa.row(s))).collect();
                let policy = TabularPolicy::from_q(&q_sa, PolicyObjective::Deterministic)?;
                (policy, value)
            }
            TDVariant::Sarsa | TDVariant::ExpectedSarsa => {
                let value = (0..n_s)
                    .map(|s| {
                        (0..n_a)
                            .map(|a| pi_target.probability(s, a) * q_sa[[s, a]])
                            .sum()
                    })
                    .collect();
                (pi_target, value)
            }
        };
        let iterations = self
            .config
            .episodes
            .try_into()
            .map_err(|_| "episode count does not fit usize")?;

        Ok(SolveResult {
            policy: Box::new(policy),
            snapshots: vec![SolverSnapshot::new(iterations).state_values(value)],
            status: SolveStatus::IterationLimit { iterations },
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
    use crate::environment::BlackBox;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn configuration_is_validated() {
        let invalid = |episodes, episode_length, epsilon, alpha| TDConfig {
            episodes,
            episode_length,
            epsilon,
            alpha,
            ..TDConfig::default()
        };
        assert!(TD::new(invalid(0, 10, 0.1, 0.1)).is_err());
        assert!(TD::new(invalid(10, 0, 0.1, 0.1)).is_err());
        assert!(TD::new(invalid(10, 10, -0.1, 0.1)).is_err());
        assert!(TD::new(invalid(10, 10, 0.1, 0.0)).is_err());
        assert!(TD::new(invalid(10, 10, 0.1, 1.1)).is_err());
        assert!(TD::new(TDConfig::default()).is_ok());
    }

    #[test]
    fn fixed_behavior_policy_is_rejected_for_sarsa() {
        let mut environment = BlackBox::new(2, 2, 0, |state, _| (state, 0.0));
        let mut rng = StdRng::seed_from_u64(0);
        let solver = TD::new(TDConfig {
            episodes: 1,
            episode_length: 1,
            ..TDConfig::default()
        })
        .unwrap()
        .with_behavior_policy(TabularPolicy::uniform(2, 2).unwrap());

        assert_eq!(
            solver.solve(&mut environment, &mut rng).err().unwrap(),
            "fixed behavior policies are not supported for Sarsa"
        );
    }

    #[test]
    fn behavior_policy_shape_must_match_environment() {
        let mut environment = BlackBox::new(2, 2, 0, |state, _| (state, 0.0));
        let mut rng = StdRng::seed_from_u64(0);
        let solver = TD::new(TDConfig {
            episodes: 1,
            episode_length: 1,
            variant: TDVariant::QLearning,
            ..TDConfig::default()
        })
        .unwrap()
        .with_behavior_policy(TabularPolicy::uniform(1, 3).unwrap());

        assert_eq!(
            solver.solve(&mut environment, &mut rng).err().unwrap(),
            "behavior policy has shape 1x3, expected 2x2"
        );
    }
}
