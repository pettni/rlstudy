//! Common interface for environments used by reinforcement-learning algorithms.

use crate::model::TabularModel;
use ndarray::Array1;

#[derive(Debug, Clone, PartialEq)]
pub enum Space {
    Discrete { size: usize },
    Continuous { dim: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Step<Observation> {
    pub observation: Observation,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
}

/// A stateful environment that can be reset and stepped.
pub trait Environment {
    type Observation;
    type Action;

    fn state_space(&self) -> Space;

    fn action_space(&self) -> Space;

    fn reset(&mut self, rng: &mut dyn rand::Rng) -> Self::Observation;

    fn step(&mut self, action: Self::Action, rng: &mut dyn rand::Rng) -> Step<Self::Observation>;
}

pub struct Rollout<'a, E: Environment + ?Sized> {
    environment: &'a mut E,
    observation: E::Observation,
}

impl<'a, E: Environment + ?Sized> Rollout<'a, E> {
    pub fn new(environment: &'a mut E, rng: &mut dyn rand::Rng) -> Self {
        let observation = environment.reset(rng);
        Self {
            environment,
            observation,
        }
    }

    pub fn observation(&self) -> &E::Observation {
        &self.observation
    }

    pub fn step(&mut self, action: E::Action, rng: &mut dyn rand::Rng) -> Step<E::Observation>
    where
        E::Observation: Clone,
    {
        let step = self.environment.step(action, rng);
        self.observation = step.observation.clone();
        step
    }

    pub fn reset(&mut self, rng: &mut dyn rand::Rng) {
        self.observation = self.environment.reset(rng);
    }
}

pub struct TabularEnvironment<'a> {
    model: &'a TabularModel,
    state: Option<usize>,
}

impl<'a> TabularEnvironment<'a> {
    pub fn new(model: &'a TabularModel) -> Result<Self, String> {
        Ok(Self { model, state: None })
    }
}

impl Environment for TabularEnvironment<'_> {
    type Observation = usize;
    type Action = usize;

    fn state_space(&self) -> Space {
        Space::Discrete {
            size: self.model.states().len(),
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete {
            size: self.model.actions().len(),
        }
    }

    fn reset(&mut self, rng: &mut dyn rand::Rng) -> usize {
        self.state = Some(*self.model.initial_distribution().sample(rng));
        self.state.unwrap()
    }

    fn step(&mut self, action: usize, rng: &mut dyn rand::Rng) -> Step<usize> {
        let &(next_state, reward) = match self.state {
            Some(state) => self.model.outcomes(state, action).sample(rng),
            None => panic!("Call reset() before step()"),
        };
        self.state = Some(next_state);
        Step {
            observation: next_state,
            reward,
            terminated: false,
            truncated: false,
        }
    }
}

/// A tabular environment observed through its state-feature vectors.
pub struct FeatureEnvironment<'a> {
    environment: TabularEnvironment<'a>,
    dim: usize,
}

impl<'a> FeatureEnvironment<'a> {
    pub fn new(model: &'a TabularModel) -> Result<Self, String> {
        let first = model
            .state_features(0)
            .ok_or_else(|| "model has no state features".to_string())?;

        Ok(Self {
            environment: TabularEnvironment::new(model)?,
            dim: first.len(),
        })
    }

    fn observe(&self, state: usize) -> Array1<f64> {
        self.environment
            .model
            .state_features(state)
            .expect("feature environment validates state features")
            .to_owned()
    }
}

impl Environment for FeatureEnvironment<'_> {
    type Observation = Array1<f64>;
    type Action = usize;

    fn state_space(&self) -> Space {
        Space::Continuous { dim: self.dim }
    }

    fn action_space(&self) -> Space {
        self.environment.action_space()
    }

    fn reset(&mut self, rng: &mut dyn rand::Rng) -> Array1<f64> {
        let state = self.environment.reset(rng);
        self.observe(state)
    }

    fn step(&mut self, action: usize, rng: &mut dyn rand::Rng) -> Step<Array1<f64>> {
        let step = self.environment.step(action, rng);
        Step {
            observation: self.observe(step.observation),
            reward: step.reward,
            terminated: step.terminated,
            truncated: step.truncated,
        }
    }
}

/// A black-box environment defined by a transition function.
pub struct BlackBox<F> {
    num_states: usize,
    num_actions: usize,
    initial_state: usize,
    state: usize,
    step_fn: F,
}

impl<F> BlackBox<F>
where
    F: FnMut(usize, usize) -> (usize, f64),
{
    /// Create a black-box environment.
    pub fn new(num_states: usize, num_actions: usize, initial_state: usize, step_fn: F) -> Self {
        Self {
            num_states,
            num_actions,
            initial_state,
            state: initial_state,
            step_fn,
        }
    }
}

impl<F> Environment for BlackBox<F>
where
    F: FnMut(usize, usize) -> (usize, f64),
{
    type Observation = usize;
    type Action = usize;

    fn state_space(&self) -> Space {
        Space::Discrete {
            size: self.num_states,
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete {
            size: self.num_actions,
        }
    }

    fn reset(&mut self, _rng: &mut dyn rand::Rng) -> usize {
        self.state = self.initial_state;
        self.state
    }

    fn step(&mut self, action: usize, _rng: &mut dyn rand::Rng) -> Step<usize> {
        let (next_state, reward) = (self.step_fn)(self.state, action);
        self.state = next_state;
        Step {
            observation: next_state,
            reward,
            terminated: false,
            truncated: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environments::grid_world;
    use crate::model::{Action, State};
    use crate::probability::Distribution;
    use ndarray::Array2;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn tabular_model_acts_as_environment() {
        let model = grid_world(3).unwrap();
        let mut environment = TabularEnvironment::new(&model).unwrap();
        let mut rng = StdRng::seed_from_u64(0);

        assert_eq!(environment.state_space(), Space::Discrete { size: 9 });
        assert_eq!(environment.action_space(), Space::Discrete { size: 5 });
        assert_eq!(environment.reset(&mut rng), 0);
        assert_eq!(
            environment.step(1, &mut rng),
            Step {
                observation: 1,
                reward: 0.0,
                terminated: false,
                truncated: false,
            }
        );
    }

    #[test]
    fn feature_environment_exposes_state_features() {
        let model = TabularModel::new(
            Distribution::point_mass(0),
            vec![State::new("s0")],
            vec![Action::new("a0")],
            vec![vec![Distribution::point_mass((0, 0.0))]],
            Some(Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap()),
        )
        .unwrap();
        let mut environment = FeatureEnvironment::new(&model).unwrap();
        let mut rng = StdRng::seed_from_u64(0);

        assert_eq!(environment.reset(&mut rng).to_vec(), vec![1.0, 2.0]);
        assert_eq!(environment.state_space(), Space::Continuous { dim: 2 });
    }

    #[test]
    fn black_box_acts_as_environment() {
        let mut bb = BlackBox::new(3, 2, 0, |state: usize, action: usize| {
            let next = (state + action) % 3;
            (next, next as f64)
        });

        assert_eq!(bb.state_space(), Space::Discrete { size: 3 });
        assert_eq!(bb.action_space(), Space::Discrete { size: 2 });
        let mut rng = StdRng::seed_from_u64(0);

        assert_eq!(bb.reset(&mut rng), 0);

        let step = bb.step(1, &mut rng);
        assert_eq!(step.observation, 1);
        assert_eq!(step.reward, 1.0);

        let step = bb.step(1, &mut rng);
        assert_eq!(step.observation, 2);
        assert_eq!(step.reward, 2.0);

        assert_eq!(bb.reset(&mut rng), 0);
    }
}
