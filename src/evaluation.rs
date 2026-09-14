use crate::environment::{Environment, Step};
use crate::policy::Policy;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvaluationMode {
    DiscountedReward { gamma: f64 },
    AverageReward,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaluationConfig {
    episodes: usize,
    max_steps: usize,
    mode: EvaluationMode,
}

impl EvaluationConfig {
    pub fn new(episodes: usize, max_steps: usize, mode: EvaluationMode) -> Result<Self, String> {
        if episodes == 0 {
            return Err("episodes must be positive".into());
        }
        if max_steps == 0 {
            return Err("max steps must be positive".into());
        }
        if let EvaluationMode::DiscountedReward { gamma } = mode
            && (!gamma.is_finite() || !(0.0..=1.0).contains(&gamma))
        {
            return Err("gamma must be finite and in [0, 1]".into());
        }
        Ok(Self {
            episodes,
            max_steps,
            mode,
        })
    }

    pub fn mode(self) -> EvaluationMode {
        self.mode
    }
}

pub fn evaluate_policy<Observation, Action>(
    environment: &mut dyn Environment<Observation = Observation, Action = Action>,
    policy: &dyn Policy<Observation = Observation, Action = Action>,
    config: EvaluationConfig,
    rng: &mut dyn rand::Rng,
) -> Result<f64, String> {
    let mut res: f64 = 0.;

    for i in 0..config.episodes {
        let s0 = environment.reset(rng);

        let rewards = (0..config.max_steps).scan(Some(s0), |maybe_s, _| {
            let s = maybe_s.take()?;
            let a = policy.sample_action(&s, rng);
            let Step {
                observation,
                reward,
                terminated,
                truncated,
            } = environment.step(a, rng);
            if !terminated && !truncated {
                *maybe_s = Some(observation);
            }
            Some(reward)
        });

        let episode_reward = match config.mode {
            EvaluationMode::AverageReward => {
                let (total, steps) = rewards.fold((0., 0usize), |(total, steps), reward| {
                    (total + reward, steps + 1)
                });
                total / steps as f64
            }
            EvaluationMode::DiscountedReward { gamma } => {
                let (total, _) = rewards.fold((0., 1.), |(total, gamma_factor), reward| {
                    (total + gamma_factor * reward, gamma_factor * gamma)
                });
                total
            }
        };

        // streaming mean value
        res += (episode_reward - res) / (i + 1) as f64;
    }

    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::{BlackBox, Space, Step};
    use rand::{SeedableRng, rngs::StdRng};

    struct FixedPolicy;

    struct TerminatingEnvironment;

    impl Environment for TerminatingEnvironment {
        type Observation = usize;
        type Action = usize;

        fn state_space(&self) -> Space {
            Space::Discrete { size: 1 }
        }

        fn action_space(&self) -> Space {
            Space::Discrete { size: 1 }
        }

        fn reset(&mut self, _rng: &mut dyn rand::Rng) -> usize {
            0
        }

        fn step(&mut self, _action: usize, _rng: &mut dyn rand::Rng) -> Step<usize> {
            Step {
                observation: 0,
                reward: 2.0,
                terminated: true,
                truncated: false,
            }
        }
    }

    impl Policy for FixedPolicy {
        type Observation = usize;
        type Action = usize;

        fn sample_action(&self, _observation: &usize, _rng: &mut dyn rand::Rng) -> usize {
            0
        }
    }

    #[test]
    fn evaluates_average_and_discounted_reward() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut environment = BlackBox::new(1, 1, 0, |state, _| (state, 2.0));

        let average = evaluate_policy(
            &mut environment,
            &FixedPolicy,
            EvaluationConfig::new(2, 3, EvaluationMode::AverageReward).unwrap(),
            &mut rng,
        )
        .unwrap();
        let discounted = evaluate_policy(
            &mut environment,
            &FixedPolicy,
            EvaluationConfig::new(2, 3, EvaluationMode::DiscountedReward { gamma: 0.5 }).unwrap(),
            &mut rng,
        )
        .unwrap();

        assert_eq!(average, 2.0);
        assert_eq!(discounted, 3.5);
    }

    #[test]
    fn rejects_invalid_configuration() {
        assert!(EvaluationConfig::new(0, 1, EvaluationMode::AverageReward).is_err());
        assert!(EvaluationConfig::new(1, 0, EvaluationMode::AverageReward).is_err());
        assert!(
            EvaluationConfig::new(1, 1, EvaluationMode::DiscountedReward { gamma: -0.1 }).is_err()
        );
    }

    #[test]
    fn stops_evaluation_at_termination() {
        let mut environment = TerminatingEnvironment;
        let mut rng = StdRng::seed_from_u64(0);

        let value = evaluate_policy(
            &mut environment,
            &FixedPolicy,
            EvaluationConfig::new(1, 10, EvaluationMode::DiscountedReward { gamma: 0.9 }).unwrap(),
            &mut rng,
        )
        .unwrap();

        assert_eq!(value, 2.0);
    }
}
