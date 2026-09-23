use crate::environment::{Environment, Step};
use crate::model::TabularModel;
use crate::policy::Policy;
use crate::reporting::print_evaluation_progress;
use ndarray::Array2;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq)]
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

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvaluationStats {
    pub return_mean: f64,
    pub return_std: f64,
    pub episode_length_mean: f64,
    pub max_steps_rate: f64,
}

pub fn evaluate_rollouts<Observation, Action, P>(
    environment: &mut dyn Environment<Observation = Observation, Action = Action>,
    policy: &P,
    config: EvaluationConfig,
    rng: &mut dyn rand::Rng,
) -> EvaluationStats
where
    P: Policy<Observation = Observation, Action = Action> + ?Sized,
{
    let mut mean_return = 0.0;
    let mut squared_deviation = 0.0;
    let mut total_length = 0;
    let mut max_steps_episodes = 0;
    print_evaluation_progress("Evaluation", 0, config.episodes);

    for i in 0..config.episodes {
        let s0 = environment.reset(rng);
        let mut ended = false;

        let rewards = (0..config.max_steps).scan(Some(s0), |maybe_s, _| {
            let s = maybe_s.take()?;
            let a = policy.sample_action(&s, rng);
            let Step {
                observation,
                reward,
                terminated,
                truncated,
            } = environment.step(a, rng);
            ended = terminated || truncated;
            if !ended {
                *maybe_s = Some(observation);
            }
            Some(reward)
        });

        let (episode_return, steps) = rewards.fold((0.0, 0usize), |(total, steps), reward| {
            (total + reward, steps + 1)
        });
        total_length += steps;
        max_steps_episodes += usize::from(!ended);

        let difference = episode_return - mean_return;
        mean_return += difference / (i + 1) as f64;
        squared_deviation += difference * (episode_return - mean_return);
        print_evaluation_progress("Evaluation", i + 1, config.episodes);
    }

    let count = config.episodes as f64;
    EvaluationStats {
        return_mean: mean_return,
        return_std: (squared_deviation / count).sqrt(),
        episode_length_mean: total_length as f64 / count,
        max_steps_rate: max_steps_episodes as f64 / count,
    }
}

pub fn fixed_policy_values(
    model: &TabularModel,
    probabilities: &Array2<f64>,
    gamma: f64,
) -> Result<Vec<f64>, String> {
    if !gamma.is_finite() || !(0.0..1.0).contains(&gamma) {
        return Err("value gamma must be finite and in [0, 1)".into());
    }
    let n_s = model.states().len();
    let n_a = model.actions().len();
    if probabilities.dim() != (n_s, n_a) {
        return Err(format!(
            "expected policy probabilities with shape {n_s}x{n_a}"
        ));
    }

    let mut values = vec![0.0; n_s];
    for _ in 0..10_000 {
        let next = (0..n_s)
            .map(|state| {
                (0..n_a)
                    .map(|action| {
                        let action_value = model
                            .outcomes(state, action)
                            .iter()
                            .map(|(probability, (next_state, reward))| {
                                probability * (reward + gamma * values[*next_state])
                            })
                            .sum::<f64>();
                        probabilities[[state, action]] * action_value
                    })
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let max_change = values
            .iter()
            .zip(&next)
            .map(|(previous, current)| (previous - current).abs())
            .fold(0.0_f64, f64::max);
        if max_change < 1e-8 {
            return Ok(next);
        }
        values = next;
    }
    Err("fixed-policy Bellman evaluation did not converge".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Action, State};
    use crate::probability::Distribution;

    #[test]
    fn bellman_evaluates_stochastic_policy() {
        let model = TabularModel::new(
            Distribution::point_mass(0),
            vec![State::new("s")],
            vec![Action::new("a"), Action::new("b")],
            vec![vec![
                Distribution::point_mass((0, 1.0)),
                Distribution::point_mass((0, 3.0)),
            ]],
            None,
        )
        .unwrap();
        let probabilities = Array2::from_shape_vec((1, 2), vec![0.25, 0.75]).unwrap();
        let values = fixed_policy_values(&model, &probabilities, 0.5).unwrap();
        assert!((values[0] - 5.0).abs() < 1e-7);
    }
}
