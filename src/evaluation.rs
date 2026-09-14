use crate::environment::Environment;
use crate::policy::Policy;

pub enum EvaluationMode {
    DiscountedReward { gamma: f64 },
    AverageReward,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvaluationConfig {
    episodes: usize,
    max_steps: usize,
}

impl EvaluationConfig {
    pub fn new(episodes: usize, max_steps: usize) -> Result<Self, String> {
        if episodes == 0 {
            return Err("episodes must be positive".into());
        }
        if max_steps == 0 {
            return Err("max steps must be positive".into());
        }
        Ok(Self {
            episodes,
            max_steps,
        })
    }
}

pub fn evaluate_policy<Observation, Action>(
    environment: &mut dyn Environment<Observation = Observation, Action = Action>,
    policy: &dyn Policy<Observation = Observation, Action = Action>,
    config: EvaluationConfig,
    mode: EvaluationMode,
    rng: &mut dyn rand::Rng,
) -> Result<f64, String> {
    if let EvaluationMode::DiscountedReward { gamma } = mode
        && (!gamma.is_finite() || !(0.0..=1.0).contains(&gamma))
    {
        return Err("gamma must be finite and in [0, 1]".into());
    }

    let mut res: f64 = 0.;

    for i in 0..config.episodes {
        let s0 = environment.reset(rng);

        let rewards = (0..config.max_steps).scan(s0, |s, _| {
            let a = policy.sample_action(s, rng);
            let (s_next, r) = environment.step(a, rng);
            *s = s_next;
            Some(r)
        });

        let episode_reward = match mode {
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
    use crate::environment::BlackBox;
    use rand::{SeedableRng, rngs::StdRng};

    struct FixedPolicy;

    impl Policy for FixedPolicy {
        type Observation = usize;
        type Action = usize;

        fn sample_action(&self, _observation: &usize, _rng: &mut dyn rand::Rng) -> usize {
            0
        }
    }

    #[test]
    fn evaluates_average_and_discounted_reward() {
        let config = EvaluationConfig::new(2, 3).unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        let mut environment = BlackBox::new(1, 1, 0, |state, _| (state, 2.0));

        let average = evaluate_policy(
            &mut environment,
            &FixedPolicy,
            config,
            EvaluationMode::AverageReward,
            &mut rng,
        )
        .unwrap();
        let discounted = evaluate_policy(
            &mut environment,
            &FixedPolicy,
            config,
            EvaluationMode::DiscountedReward { gamma: 0.5 },
            &mut rng,
        )
        .unwrap();

        assert_eq!(average, 2.0);
        assert_eq!(discounted, 3.5);
    }

    #[test]
    fn rejects_invalid_configuration() {
        assert!(EvaluationConfig::new(0, 1).is_err());
        assert!(EvaluationConfig::new(1, 0).is_err());
    }
}
