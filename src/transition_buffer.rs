//! Transition storage and tensor conversion.

use crate::environment::{Environment, Rollout};
use crate::policy::Policy;
use burn::{
    Tensor,
    backend::{Autodiff, Flex, flex::FlexDevice},
    tensor::{Bool, Int, TensorData},
};
use ndarray::Array1;
use std::collections::VecDeque;

type TrainingBackend = Autodiff<Flex>;

pub trait ActionType: Clone {
    type TensorType;

    fn tensor_data(actions: &[&Self]) -> Self::TensorType;
}

impl ActionType for usize {
    type TensorType = Tensor<TrainingBackend, 1, Int>;

    fn tensor_data(actions: &[&Self]) -> Self::TensorType {
        let values = actions.iter().map(|x| **x as i64).collect::<Vec<_>>();
        Tensor::from_data(TensorData::new(values, [actions.len()]), &FlexDevice)
    }
}

impl ActionType for Array1<f64> {
    type TensorType = Tensor<TrainingBackend, 2>;

    fn tensor_data(actions: &[&Self]) -> Self::TensorType {
        let action_dim = actions.first().expect("buffer is empty").len();
        let values = actions
            .iter()
            .flat_map(|a| a.iter().copied())
            .collect::<Vec<_>>();
        Tensor::from_data(
            TensorData::new(values, [actions.len(), action_dim]),
            &FlexDevice,
        )
    }
}

pub struct TransitionBufferTensorData<A: ActionType> {
    // [s0, ..., s_{T-1}]
    pub s: Tensor<TrainingBackend, 2>,
    // [a0, ..., a_{T-1}]
    pub a: A::TensorType,
    // [r1, ..., r_T]
    pub r: Tensor<TrainingBackend, 1>,
    // [s1, ..., s_T]
    pub sp: Tensor<TrainingBackend, 2>,
    // [l1, ..., l_{t}]  s.t. lt is 1. if episode terminated at t
    pub terminated: Tensor<TrainingBackend, 1, Bool>,
    // [l1, ..., l_{t}]  s.t. lt is 1. if episode truncated at t
    pub truncated: Tensor<TrainingBackend, 1, Bool>,
}

impl<A: ActionType> TransitionBufferTensorData<A> {
    fn boundaries(&self) -> Tensor<TrainingBackend, 1, Bool> {
        self.terminated.clone().bool_or(self.truncated.clone())
    }

    /// Generalized advantages GAE_t = delta_t + gamma * lambda * delta_{t+1} + ...
    pub fn gae<F>(&self, value: F, gamma: f64, lambda: f64) -> Tensor<TrainingBackend, 1>
    where
        F: Fn(Tensor<TrainingBackend, 2>) -> Tensor<TrainingBackend, 1>,
    {
        segmented_discounted_cumsum(
            self.td_residuals(value, gamma),
            self.boundaries(),
            gamma * lambda,
        )
    }

    /// Compute temporal difference (TD) residuals delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    /// w.r.t. a value function
    pub fn td_residuals<F>(&self, value: F, gamma: f64) -> Tensor<TrainingBackend, 1>
    where
        F: Fn(Tensor<TrainingBackend, 2>) -> Tensor<TrainingBackend, 1>,
    {
        let values = value(self.s.clone()).detach();
        let next_values = value(self.sp.clone())
            .detach()
            .mask_fill(self.terminated.clone(), 0.0);
        self.r.clone() + gamma * next_values - values
    }

    /// Compute G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...
    ///
    /// Bootstraps truncated episodes and the last transition if it doesn't terminate.
    ///
    /// Uses the identity G_t = V_t + GAE(lambda=1). When discounting from the
    /// episode start, the first transition in the buffer is treated as t = 0.
    pub fn discounted_returns<F>(&self, value: F, gamma: f64) -> Tensor<TrainingBackend, 1>
    where
        F: Fn(Tensor<TrainingBackend, 2>) -> Tensor<TrainingBackend, 1>,
    {
        value(self.s.clone()).detach() + self.gae(value, gamma, 1.0)
    }

    /// Compute discounted returns with V(s) = 0.
    pub fn discounted_returns_no_bootstrapping(&self, gamma: f64) -> Tensor<TrainingBackend, 1> {
        self.discounted_returns(
            |states| Tensor::zeros([states.shape()[0]], &FlexDevice),
            gamma,
        )
    }

    /// Discount 'values' from sequence start
    ///
    ///  v_t -> gamma^t v_t
    pub fn discount_from_episode_start(
        &self,
        values: Tensor<TrainingBackend, 1>,
        gamma: f64,
    ) -> Tensor<TrainingBackend, 1> {
        values * discounts_from_episode_start(self.boundaries(), gamma)
    }
}

pub struct Transition<A: ActionType> {
    pub s: ndarray::Array1<f64>,
    pub a: A,
    pub r: f64,
    pub sp: ndarray::Array1<f64>,
    pub terminated: bool,
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpisodeStats {
    pub count: usize,
    pub return_mean: f64,
    pub length_mean: f64,
}

// Store (s,a,r,s') data with convenient tensor data access
pub struct TransitionBuffer<A: ActionType> {
    max_size: Option<usize>,
    transitions: VecDeque<Transition<A>>,
}

impl<A: ActionType> TransitionBuffer<A> {
    pub fn new(max_size: Option<usize>) -> Self {
        TransitionBuffer {
            max_size,
            transitions: VecDeque::new(),
        }
    }

    pub fn with_capacity(capacity: usize, max_size: Option<usize>) -> Self {
        TransitionBuffer {
            max_size,
            transitions: VecDeque::with_capacity(capacity),
        }
    }

    /// Number of transitions in buffer
    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    /// Remove all data.
    pub fn clear(&mut self) {
        self.transitions.clear();
    }

    pub fn episode_stats(&self) -> Option<EpisodeStats> {
        let mut episode_return = 0.0;
        let mut episode_length = 0;
        let mut completed_episode_return = 0.0;
        let mut completed_episode_length = 0;
        let mut completed_episodes = 0;

        for transition in &self.transitions {
            episode_return += transition.r;
            episode_length += 1;

            if transition.terminated || transition.truncated {
                completed_episode_return += episode_return;
                completed_episode_length += episode_length;
                completed_episodes += 1;
                episode_return = 0.0;
                episode_length = 0;
            }
        }

        if completed_episodes == 0 {
            return None;
        }

        Some(EpisodeStats {
            count: completed_episodes,
            return_mean: completed_episode_return / completed_episodes as f64,
            length_mean: completed_episode_length as f64 / completed_episodes as f64,
        })
    }

    /// Add transition to buffer with eviction logic
    pub fn push(&mut self, transition: Transition<A>) {
        if self
            .max_size
            .is_some_and(|max_size| self.transitions.len() == max_size)
        {
            self.transitions.pop_front();
        }
        self.transitions.push_back(transition);
    }

    /// Take one environment step and add its transition to the buffer.
    pub fn fill_one<E, P>(
        &mut self,
        rollout: &mut Rollout<'_, E>,
        pi: &P,
        rng: &mut dyn rand::Rng,
    ) -> bool
    where
        E: Environment<Observation = Array1<f64>, Action = A> + ?Sized,
        P: Policy<Observation = Array1<f64>, Action = A> + ?Sized,
    {
        let s = rollout.observation().clone();
        let a = pi.sample_action(&s, rng);
        let step = rollout.step(a.clone(), rng);
        let done = step.terminated || step.truncated;

        self.push(Transition {
            s,
            a,
            r: step.reward,
            sp: step.observation,
            terminated: step.terminated,
            truncated: step.truncated,
        });

        done
    }

    /// Fill the buffer with one complete episode.
    pub fn fill_episode<P>(
        &mut self,
        env: &mut dyn Environment<Observation = Array1<f64>, Action = A>,
        pi: &P,
        rng: &mut dyn rand::Rng,
    ) where
        P: Policy<Observation = Array1<f64>, Action = A> + ?Sized,
    {
        let mut rollout = Rollout::new(env, rng);

        while !self.fill_one(&mut rollout, pi, rng) {}
    }

    /// Fill the buffer with data from stepping the environment n times.
    /// Resets the environment when episodes finish.
    pub fn fill<E, P>(
        &mut self,
        rollout: &mut Rollout<'_, E>,
        pi: &P,
        rng: &mut dyn rand::Rng,
        n: usize,
    ) where
        E: Environment<Observation = Array1<f64>, Action = A> + ?Sized,
        P: Policy<Observation = Array1<f64>, Action = A> + ?Sized,
    {
        for _ in 0..n {
            if self.fill_one(rollout, pi, rng) {
                rollout.reset(rng);
            }
        }
    }

    /// Return all data in order
    pub fn tensor_data(&self) -> TransitionBufferTensorData<A> {
        self.tensor_data_impl(self.transitions.iter())
    }

    /// Return a random data selection
    pub fn random_batch_tensor_data(
        &self,
        n: usize,
        rng: &mut dyn rand::Rng,
    ) -> TransitionBufferTensorData<A> {
        let transitions = rand::seq::index::sample(rng, self.transitions.len(), n)
            .into_iter()
            .map(|index| &self.transitions[index]);

        self.tensor_data_impl(transitions)
    }

    fn tensor_data_impl<'a>(
        &self,
        transitions: impl ExactSizeIterator<Item = &'a Transition<A>>,
    ) -> TransitionBufferTensorData<A>
    where
        A: 'a,
    {
        let state_dim = self.transitions.front().expect("buffer is empty").s.len();
        let size = transitions.len();

        let mut states = Vec::with_capacity(size * state_dim);
        let mut actions = Vec::with_capacity(size);
        let mut rewards = Vec::with_capacity(size);
        let mut next_states = Vec::with_capacity(size * state_dim);
        let mut terminated = Vec::with_capacity(size);
        let mut truncated = Vec::with_capacity(size);

        for transition in transitions {
            states.extend(transition.s.iter().copied());
            actions.push(&transition.a);
            rewards.push(transition.r);
            next_states.extend(transition.sp.iter().copied());
            terminated.push(transition.terminated);
            truncated.push(transition.truncated);
        }

        TransitionBufferTensorData::<A> {
            s: Tensor::from_data(TensorData::new(states, [size, state_dim]), &FlexDevice),
            a: A::tensor_data(&actions),
            r: Tensor::from_data(TensorData::new(rewards, [size]), &FlexDevice),
            sp: Tensor::from_data(TensorData::new(next_states, [size, state_dim]), &FlexDevice),
            truncated: Tensor::from_data(TensorData::new(truncated, [size]), &FlexDevice),
            terminated: Tensor::from_data(TensorData::new(terminated, [size]), &FlexDevice),
        }
    }
}

fn segment_offsets(
    starts: Tensor<TrainingBackend, 1, Bool>,
) -> (
    Tensor<TrainingBackend, 1, Int>,
    Tensor<TrainingBackend, 1, Int>,
) {
    let size = starts.shape()[0];
    let indices = Tensor::<TrainingBackend, 1, Int>::arange(0..size as i64, &FlexDevice);
    let start_indices = indices.clone().mask_fill(starts.bool_not(), 0).cummax(0);
    let offsets = indices - start_indices.clone();
    (offsets, start_indices)
}

/// Compute y_t = x_t + discount * y_{t+1}, resetting at boundaries.
fn segmented_discounted_cumsum(
    values: Tensor<TrainingBackend, 1>,
    boundaries: Tensor<TrainingBackend, 1, Bool>,
    discount: f64,
) -> Tensor<TrainingBackend, 1> {
    if discount == 0.0 {
        return values;
    }

    let values = values.flip([0]);
    let starts = boundaries.flip([0]).slice_assign(
        0..1,
        Tensor::<TrainingBackend, 1, Bool>::from_data([true], &FlexDevice),
    );
    let (offsets, start_indices) = segment_offsets(starts);
    let powers = Tensor::<TrainingBackend, 1>::full(values.shape(), discount, &FlexDevice)
        .powf(offsets.float());
    let cumulative = (values / powers.clone()).cumsum(0);
    let before_start = cumulative
        .clone()
        .select(0, (start_indices.clone() - 1).clamp_min(0))
        .mask_fill(start_indices.equal_elem(0), 0.0);

    (powers * (cumulative - before_start)).flip([0])
}

fn discounts_from_episode_start(
    boundaries: Tensor<TrainingBackend, 1, Bool>,
    discount: f64,
) -> Tensor<TrainingBackend, 1> {
    let size = boundaries.shape()[0];
    let starts = Tensor::cat(
        vec![
            Tensor::<TrainingBackend, 1, Bool>::from_data([true], &FlexDevice),
            boundaries,
        ],
        0,
    )
    .slice(0..size);
    let (offsets, _) = segment_offsets(starts);

    Tensor::<TrainingBackend, 1>::full([size], discount, &FlexDevice).powf(offsets.float())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::{Space, Step};
    use rand::{SeedableRng, rngs::StdRng};

    struct TwoStepEnvironment {
        state: Option<usize>,
    }

    impl Environment for TwoStepEnvironment {
        type Observation = Array1<f64>;
        type Action = usize;

        fn state_space(&self) -> Space {
            Space::Continuous { dim: 1 }
        }

        fn action_space(&self) -> Space {
            Space::Discrete { size: 1 }
        }

        fn reset(&mut self, _rng: &mut dyn rand::Rng) -> Array1<f64> {
            self.state = Some(0);
            Array1::from_vec(vec![0.0])
        }

        fn step(&mut self, _action: usize, _rng: &mut dyn rand::Rng) -> Step<Array1<f64>> {
            let state = self.state.expect("environment must be initialized") + 1;
            self.state = Some(state);
            Step {
                observation: Array1::from_vec(vec![state as f64]),
                reward: 1.0,
                terminated: state == 2,
                truncated: false,
            }
        }
    }

    struct FixedPolicy;

    impl Policy for FixedPolicy {
        type Observation = Array1<f64>;
        type Action = usize;

        fn sample_action(&self, _observation: &Array1<f64>, _rng: &mut dyn rand::Rng) -> usize {
            0
        }
    }

    fn transition(reward: f64) -> Transition<usize> {
        Transition {
            s: Array1::zeros(1),
            a: 0,
            r: reward,
            sp: Array1::zeros(1),
            terminated: false,
            truncated: false,
        }
    }

    fn valued_transition(
        state_value: f64,
        reward: f64,
        next_state_value: f64,
        terminated: bool,
        truncated: bool,
    ) -> Transition<usize> {
        Transition {
            s: Array1::from_vec(vec![state_value]),
            a: 0,
            r: reward,
            sp: Array1::from_vec(vec![next_state_value]),
            terminated,
            truncated,
        }
    }

    #[test]
    fn bounded_buffer_evicts_oldest_transitions() {
        let mut buffer = TransitionBuffer::new(Some(2));

        buffer.push(transition(1.0));
        buffer.push(transition(2.0));
        buffer.push(transition(3.0));

        assert_eq!(buffer.len(), 2);
        assert_eq!(
            buffer
                .transitions
                .iter()
                .map(|transition| transition.r)
                .collect::<Vec<_>>(),
            vec![2.0, 3.0]
        );
    }

    #[test]
    fn consecutive_fills_continue_the_rollout() {
        let mut buffer = TransitionBuffer::new(None);
        let mut environment = TwoStepEnvironment { state: None };
        let mut rng = StdRng::seed_from_u64(0);
        let mut rollout = Rollout::new(&mut environment, &mut rng);

        buffer.fill(&mut rollout, &FixedPolicy, &mut rng, 1);
        buffer.fill(&mut rollout, &FixedPolicy, &mut rng, 1);

        assert_eq!(buffer.transitions[0].s, Array1::from_vec(vec![0.0]));
        assert_eq!(buffer.transitions[1].s, Array1::from_vec(vec![1.0]));
        assert!(buffer.transitions[1].terminated);
    }

    #[test]
    fn gae_and_returns_respect_boundaries_and_bootstrap() {
        let mut buffer = TransitionBuffer::new(None);
        // first episode terminates (no bootstrap)
        buffer.push(valued_transition(0.5, 1.0, 0.6, false, false));
        buffer.push(valued_transition(0.6, 2.0, 0.7, false, false));
        buffer.push(valued_transition(0.7, 3.0, 99.0, true, false));
        // second episode truncates (bootstrap)
        buffer.push(valued_transition(0.5, 1.0, 0.6, false, false));
        buffer.push(valued_transition(0.6, 2.0, 0.7, false, false));
        buffer.push(valued_transition(0.7, 3.0, 99.0, false, true));
        // third episode neither truncates nor terminates (bootstrap)
        buffer.push(valued_transition(0.5, 1.0, 0.6, false, false));
        buffer.push(valued_transition(0.6, 2.0, 0.7, false, false));
        buffer.push(valued_transition(0.7, 3.0, 99.0, false, false));

        let batch = buffer.tensor_data();
        let value = |states: Tensor<TrainingBackend, 2>| states.squeeze_dim(1);

        let gamma = 0.9;
        let lambda = 0.8;

        let returns_t = batch.discounted_returns(value, gamma);
        let returns = returns_t
            .clone()
            .into_data()
            .iter::<f64>()
            .collect::<Vec<_>>();

        for (actual, expected) in returns.iter().zip([
            // first episode
            1.0 + gamma * 2.0 + gamma.powi(2) * 3.0,
            2.0 + gamma * 3.0,
            3.0,
            // second episode
            1.0 + gamma * 2.0 + gamma.powi(2) * 3.0 + gamma.powi(3) * 99.0,
            2.0 + gamma * 3.0 + gamma.powi(2) * 99.0,
            3.0 + gamma * 99.0,
            // third episode
            1.0 + gamma * 2.0 + gamma.powi(2) * 3.0 + gamma.powi(3) * 99.0,
            2.0 + gamma * 3.0 + gamma.powi(2) * 99.0,
            3.0 + gamma * 99.0,
        ]) {
            assert!((actual - expected).abs() < 1e-5);
        }

        let returns_from_episode_start = batch
            .discount_from_episode_start(returns_t, gamma)
            .into_data()
            .iter::<f64>()
            .collect::<Vec<_>>();

        for (index, (actual, expected)) in
            returns_from_episode_start.iter().zip(&returns).enumerate()
        {
            let expected = expected * gamma.powi((index % 3) as i32);
            assert!((actual - expected).abs() < 1e-5);
        }

        let deltas = batch
            .td_residuals(value, gamma)
            .into_data()
            .iter::<f64>()
            .collect::<Vec<_>>();

        for (actual, expected) in deltas.iter().zip([
            1.0 + gamma * 0.6 - 0.5,
            2.0 + gamma * 0.7 - 0.6,
            3.0 + gamma * 0. - 0.7,
            1.0 + gamma * 0.6 - 0.5,
            2.0 + gamma * 0.7 - 0.6,
            3.0 + gamma * 99.0 - 0.7, // bootstrap
            1.0 + gamma * 0.6 - 0.5,
            2.0 + gamma * 0.7 - 0.6,
            3.0 + gamma * 99.0 - 0.7, // bootstrap
        ]) {
            assert!((actual - expected).abs() < 1e-5);
        }

        let gaes = batch
            .gae(value, gamma, lambda)
            .into_data()
            .iter::<f64>()
            .collect::<Vec<_>>();

        let l = gamma * lambda;

        for (actual, expected) in gaes.iter().zip([
            deltas[0] + l * deltas[1] + l.powi(2) * deltas[2],
            deltas[1] + l * deltas[2],
            deltas[2],
            deltas[3] + l * deltas[4] + l.powi(2) * deltas[5],
            deltas[4] + l * deltas[5],
            deltas[5],
            deltas[6] + l * deltas[7] + l.powi(2) * deltas[8],
            deltas[7] + l * deltas[8],
            deltas[8],
        ]) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }
}
