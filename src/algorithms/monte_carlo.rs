use crate::algorithms::{ModelFreeSolver, SolveResult, TerminationReason};
use crate::environment::{Environment, Space};
use crate::policy::{Policy, PolicyObjective, TabularPolicy};
use ndarray::{Array2, Axis};

pub struct MonteCarlo {
    // number of episodes
    episodes: u64,
    // episode length
    episode_length: u64,
    // soft action TODO: Support an epsilon schedule that decreases across episodes.
    epsilon: f64,
}

/// Monte Carlo ε-greedy
///
/// Rollouts are on-policy but the policy is ε-soft.
/// Each episode contributes to the q-value estimate of each state-action pair it visits.
impl MonteCarlo {
    pub fn new(episodes: u64, episode_length: u64, epsilon: f64) -> Self {
        Self {
            episodes,
            episode_length,
            epsilon,
        }
    }
}

impl ModelFreeSolver for MonteCarlo {
    fn solve(
        &self,
        env: &mut dyn Environment<State = usize, Action = usize>,
        gamma: f64,
        rng: &mut dyn rand::Rng,
    ) -> Result<SolveResult, String> {
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

        for _e in 0..self.episodes {
            let s0 = env.reset(rng);

            // generate [(s0, a0, r0), (s1, a1, r1), ..., (sT-1, aT-1, rT-1)]
            let episode = (0..self.episode_length)
                .scan(s0, |s, _| {
                    let a = pi.sample_action(s, rng);
                    let (s_next, r) = env.step(a, rng);

                    let transition = (*s, a, r);
                    *s = s_next;

                    Some(transition)
                })
                .collect::<Vec<_>>();

            // update state-action rewards and counts from episode
            let returns = episode.into_iter().rev().scan(0.0, |g, (s, a, r)| {
                *g = r + gamma * *g;
                Some((s, a, *g))
            });
            for (s, a, g) in returns {
                let visits = &mut n_sa[[s, a]];
                *visits += 1;

                let estimate = &mut q_sa[[s, a]];
                *estimate += (g - *estimate) / *visits as f64;
            }

            // policy improvement
            pi = TabularPolicy::from_q(&q_sa, PolicyObjective::SoftDeterministic(self.epsilon))?;
        }

        // value under the final epsilon-soft policy
        let value = (&pi.probabilities() * &q_sa).sum_axis(Axis(1)).to_vec();

        Ok(SolveResult {
            policy: pi,
            value,
            iterations: self
                .episodes
                .try_into()
                .map_err(|_| "episode count does not fit usize")?,
            termination: TerminationReason::IterationLimit,
        })
    }
}
