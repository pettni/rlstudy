use crate::algorithms::{
    ModelFreeSolver, SolveResult, SolveStatus, SolverSnapshot, validate_gamma,
};
use crate::environment::{Environment, Space};
use crate::policy::{Policy, PolicyObjective, TabularPolicy};
use clap::Args;
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Args, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MonteCarloConfig {
    #[arg(long, default_value_t = MonteCarloConfig::default().gamma)]
    pub gamma: f64,
    #[arg(long, default_value_t = MonteCarloConfig::default().episodes)]
    pub episodes: u64,
    #[arg(long, default_value_t = MonteCarloConfig::default().episode_length)]
    pub episode_length: u64,
    #[arg(long, default_value_t = MonteCarloConfig::default().epsilon)]
    pub epsilon: f64,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            gamma: 0.9,
            episodes: 10_000,
            episode_length: 30,
            epsilon: 0.1,
        }
    }
}

pub struct MonteCarlo {
    config: MonteCarloConfig,
}

/// Monte Carlo ε-greedy
///
/// Rollouts are on-policy but the policy is ε-soft.
/// Each episode contributes to the q-value estimate of each state-action pair it visits.
impl MonteCarlo {
    pub fn new(config: MonteCarloConfig) -> Result<Self, String> {
        validate_gamma(config.gamma)?;
        if config.episodes == 0 || config.episode_length == 0 {
            return Err("episodes and episode length must be positive".into());
        }
        if !config.epsilon.is_finite() || !(0.0..=1.0).contains(&config.epsilon) {
            return Err("epsilon must be finite and in [0, 1]".into());
        }
        Ok(Self { config })
    }
}

impl ModelFreeSolver for MonteCarlo {
    type Observation = usize;

    fn solve(
        &self,
        env: &mut dyn Environment<Observation = usize, Action = usize>,
        rng: &mut dyn rand::Rng,
    ) -> Result<crate::algorithms::TabularSolveResult, String> {
        let Space::Discrete { size: n_s } = env.state_space() else {
            return Err("Monte Carlo requires a discrete state space".into());
        };
        let Space::Discrete { size: n_a } = env.action_space() else {
            return Err("Monte Carlo requires a discrete action space".into());
        };

        // visits for each state
        let mut n_sa = Array2::<u64>::zeros((n_s, n_a));
        // state-action values placeholder
        let mut q_sa = Array2::<f64>::zeros((n_s, n_a));

        // policy
        let mut pi = TabularPolicy::uniform(n_s, n_a).unwrap();

        for _e in 0..self.config.episodes {
            let s0 = env.reset(rng);

            // generate [(s0, a0, r0), (s1, a1, r1), ..., (sT-1, aT-1, rT-1)]
            let episode = (0..self.config.episode_length)
                .scan(Some(s0), |state, _| {
                    let s = state.take()?;
                    let a = pi.sample_action(&s, rng);
                    let step = env.step(a, rng);
                    if !step.terminated && !step.truncated {
                        *state = Some(step.observation);
                    }
                    Some((s, a, step.reward))
                })
                .collect::<Vec<_>>();

            // update state-action rewards and counts from episode
            let returns = episode.into_iter().rev().scan(0.0, |g, (s, a, r)| {
                *g = r + self.config.gamma * *g;
                Some((s, a, *g))
            });
            for (s, a, g) in returns {
                let visits = &mut n_sa[[s, a]];
                *visits += 1;

                let estimate = &mut q_sa[[s, a]];
                *estimate += (g - *estimate) / *visits as f64;
            }

            // policy improvement
            pi = TabularPolicy::from_q(
                &q_sa,
                PolicyObjective::SoftDeterministic(self.config.epsilon),
            )?;
        }

        // value under the final epsilon-soft policy
        let value = (&pi.probabilities() * &q_sa).sum_axis(Axis(1)).to_vec();
        let iterations = self
            .config
            .episodes
            .try_into()
            .map_err(|_| "episode count does not fit usize")?;

        Ok(SolveResult {
            policy: Box::new(pi),
            snapshots: vec![SolverSnapshot::new(iterations).state_values(value)],
            status: SolveStatus::IterationLimit { iterations },
        })
    }
}
