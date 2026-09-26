use crate::algorithms::{ModelFreeSolver, SolveResult, SolveStatus};
use crate::backend::{TrainingBackend, device};
use crate::environment::{Environment, Rollout, Space};
use crate::network::action::ActionType;
use crate::network::actor::{Actor, CategoricalActor, SquashedGaussianActor};
use crate::network::{AdamOptimizer, MlpModel, MlpSettings, ema};
use crate::policy::{
    Policy,
    uniform::{UniformContinuousPolicy, UniformPolicy},
};
use crate::reporting::{MetricAverages, Reporter, Step};
use crate::transition_buffer::TransitionBuffer;
use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{ElementConversion, Tensor, activation::log_softmax, backend::Backend},
};
use ndarray::Array1;
use rand::RngExt;
use std::marker::PhantomData;

pub struct SoftActorCriticConfig {
    pub gamma: f64,
    pub alpha: f64,
    pub iterations: usize,
    pub batch_size: usize,
    pub replay_capacity: usize,
    pub warm_up_transitions: usize,
    pub steps_per_iteration: usize,
    pub q_updates_per_iteration: usize,
    pub ema_rate: f64,
    pub q_network: MlpSettings,
    pub pi_network: MlpSettings,
}

impl Default for SoftActorCriticConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            alpha: 0.1,
            iterations: 1_000,
            batch_size: 64,
            replay_capacity: 1_000,
            warm_up_transitions: 64,
            steps_per_iteration: 1,
            q_updates_per_iteration: 1,
            ema_rate: 0.01,
            q_network: MlpSettings::default(),
            pi_network: MlpSettings::default(),
        }
    }
}

pub struct SoftActorCritic<Action> {
    config: SoftActorCriticConfig,
    action: PhantomData<Action>,
}

impl<Action> SoftActorCritic<Action> {
    pub fn new(config: SoftActorCriticConfig) -> Result<Self, String> {
        Ok(Self {
            config,
            action: PhantomData,
        })
    }
}

impl<Action> ModelFreeSolver for SoftActorCritic<Action>
where
    Action: SacActorCritic,
    <Action::Actor as Actor<Action>>::InferencePolicy: 'static,
{
    type Observation = Array1<f64>;
    type Action = Action;

    fn solve_with_reporter(
        &self,
        env: &mut dyn Environment<Observation = Self::Observation, Action = Self::Action>,
        rng: &mut dyn rand::Rng,
        reporter: &mut Reporter<'_, Self::Observation, Self::Action>,
    ) -> Result<SolveResult<Self::Observation, Self::Action>, String> {
        let n_s = env
            .state_space()
            .continuous_dim()
            .ok_or("Sac requires a continuous state space")?;

        <TrainingBackend as Backend>::seed(device(), rng.random());

        let mut pi = Action::create_actor(&self.config.pi_network, n_s, env.action_space())?;
        let mut pi_optimizer: AdamOptimizer<Action::Actor> = AdamConfig::new().init();

        let mut q1 = Action::create_critic(&self.config.q_network, n_s, env.action_space())?;
        let mut q2 = Action::create_critic(&self.config.q_network, n_s, env.action_space())?;

        let mut buffer = TransitionBuffer::<Self::Action>::new(Some(self.config.replay_capacity));

        let warmup_policy = Action::create_uniform_policy(env.action_space())?;
        let mut rollout = Rollout::new(env, rng);
        buffer.fill_n(
            self.config.warm_up_transitions,
            &mut rollout,
            &warmup_policy,
            rng,
        );

        for iteration in 0..self.config.iterations {
            let step = Step::Iteration(iteration + 1);
            reporter.progress(step, self.config.iterations);

            buffer.fill_n(self.config.steps_per_iteration, &mut rollout, &pi, rng);

            let q_lr = self.config.q_network.learning_rate.value(iteration);
            let pi_lr = self.config.pi_network.learning_rate.value(iteration);
            let mut update_metrics = MetricAverages::new();

            for _ in 0..self.config.q_updates_per_iteration {
                let batch = buffer.random_batch_tensor_data(self.config.batch_size, rng);

                let (soft_next_values, _) = Action::soft_state_values(
                    &pi,
                    &q1.target,
                    &q2.target,
                    batch.sp.clone(),
                    self.config.alpha,
                    rng,
                );
                let td_target = batch.r
                    + self.config.gamma * batch.terminated.bool_not().float() * soft_next_values;
                let td_target = td_target.detach();

                // critic updates

                let q1val = Action::eval_q(&q1.pred, batch.s.clone(), batch.a.clone());
                let q1_loss = (td_target.clone() - q1val).square().mean();
                update_metrics.record("q1/loss", q1_loss.clone().into_scalar().elem::<f64>());
                let q1_grad = GradientsParams::from_grads(q1_loss.backward(), &q1.pred);
                q1.pred = q1.optimizer.step(q_lr, q1.pred, q1_grad);

                let q2val = Action::eval_q(&q2.pred, batch.s.clone(), batch.a.clone());
                let q2_loss = (td_target.clone() - q2val).square().mean();
                update_metrics.record("q2/loss", q2_loss.clone().into_scalar().elem::<f64>());
                let q2_grad = GradientsParams::from_grads(q2_loss.backward(), &q2.pred);
                q2.pred = q2.optimizer.step(q_lr, q2.pred, q2_grad);

                // actor update

                let (soft_values, entropy) = Action::soft_state_values(
                    &pi,
                    &q1.pred,
                    &q2.pred,
                    batch.s.clone(),
                    self.config.alpha,
                    rng,
                );
                update_metrics.record("pi/entropy", entropy.mean().into_scalar().elem::<f64>());

                let pi_loss = -soft_values.mean();
                update_metrics.record("pi/loss", pi_loss.clone().into_scalar().elem::<f64>());

                let pi_grad = GradientsParams::from_grads(pi_loss.backward(), &pi);
                pi = pi_optimizer.step(pi_lr, pi, pi_grad);

                // ema on target networks
                q1.target = ema::update(q1.target, &q1.pred, self.config.ema_rate);
                q2.target = ema::update(q2.target, &q2.pred, self.config.ema_rate);
            }

            let mut metrics = vec![
                ("q/lr", q_lr),
                ("pi/lr", pi_lr),
                ("buffer/size", buffer.len() as f64),
            ];
            metrics.extend(update_metrics.finish());
            reporter.record(step, &metrics);
            reporter.policy(step, || pi.valid());
        }

        Ok(SolveResult {
            policy: Box::new(pi.valid()),
            status: SolveStatus::IterationLimit {
                iterations: self.config.iterations,
            },
        })
    }
}

trait SacActorCritic: ActionType {
    type Actor: Actor<Self>;
    type Policy: Policy<Observation = Array1<f64>, Action = Self>;

    fn create_actor(
        settings: &MlpSettings,
        state_dim: usize,
        action_space: Space,
    ) -> Result<Self::Actor, String>;

    fn create_critic(
        settings: &MlpSettings,
        state_dim: usize,
        action_space: Space,
    ) -> Result<Qdata, String>;

    fn create_uniform_policy(action_space: Space) -> Result<Self::Policy, String>;

    fn eval_q(
        model: &MlpModel<TrainingBackend>,
        states: Tensor<TrainingBackend, 2>,
        actions: Self::TensorType,
    ) -> Tensor<TrainingBackend, 1>;

    fn soft_state_values(
        pi: &Self::Actor,
        q1: &MlpModel<TrainingBackend>,
        q2: &MlpModel<TrainingBackend>,
        states: Tensor<TrainingBackend, 2>,
        alpha: f64,
        rng: &mut dyn rand::Rng,
    ) -> (Tensor<TrainingBackend, 1>, Tensor<TrainingBackend, 1>);
}

impl SacActorCritic for usize {
    type Actor = CategoricalActor<TrainingBackend>;
    type Policy = UniformPolicy;

    fn create_actor(
        settings: &MlpSettings,
        state_dim: usize,
        action_space: Space,
    ) -> Result<Self::Actor, String> {
        let Space::Discrete { size: action_dim } = action_space else {
            return Err("Sac requires a discrete action space".into());
        };

        Ok(CategoricalActor::new(
            state_dim,
            &settings.hidden_dims,
            action_dim,
            settings.activation,
            device(),
        ))
    }

    fn create_critic(
        settings: &MlpSettings,
        state_dim: usize,
        action_space: Space,
    ) -> Result<Qdata, String> {
        let Space::Discrete { size: n_a } = action_space else {
            return Err("Sac requires a discrete action space".into());
        };
        Ok(Qdata::new(settings.init(state_dim, n_a)))
    }

    fn create_uniform_policy(action_space: Space) -> Result<Self::Policy, String> {
        let Space::Discrete { size } = action_space else {
            return Err("Sac requires a discrete action space".into());
        };
        Ok(UniformPolicy::new(size))
    }

    fn eval_q(
        model: &MlpModel<TrainingBackend>,
        states: Tensor<TrainingBackend, 2>,
        actions: Self::TensorType,
    ) -> Tensor<TrainingBackend, 1> {
        model
            .forward(states)
            .gather(1, actions.reshape([-1, 1]))
            .squeeze_dim(1)
    }

    fn soft_state_values(
        pi: &Self::Actor,
        q1: &MlpModel<TrainingBackend>,
        q2: &MlpModel<TrainingBackend>,
        states: Tensor<TrainingBackend, 2>,
        alpha: f64,
        _rng: &mut dyn rand::Rng,
    ) -> (Tensor<TrainingBackend, 1>, Tensor<TrainingBackend, 1>) {
        let log_probs = log_softmax(pi.forward(states.clone()), 1);
        let probabilities = log_probs.clone().exp();
        let q_values = q1
            .forward(states.clone())
            .min_pair(q2.forward(states))
            .detach();

        let expected_q = (probabilities.clone() * q_values).sum_dim(1).squeeze_dim(1);
        let entropy = -(probabilities * log_probs).sum_dim(1).squeeze_dim(1);
        let soft_values = expected_q + entropy.clone() * alpha;

        (soft_values, entropy)
    }
}

impl SacActorCritic for Array1<f64> {
    type Actor = SquashedGaussianActor<TrainingBackend>;
    type Policy = UniformContinuousPolicy;

    fn create_actor(
        settings: &MlpSettings,
        state_dim: usize,
        action_space: Space,
    ) -> Result<Self::Actor, String> {
        Ok(SquashedGaussianActor::new(
            state_dim,
            &settings.hidden_dims,
            action_space,
            settings.activation,
            device(),
        ))
    }

    fn create_critic(
        settings: &MlpSettings,
        state_dim: usize,
        action_space: Space,
    ) -> Result<Qdata, String> {
        let action_dim = action_space
            .continuous_dim()
            .ok_or("Sac requires a continuous action space")?;
        Ok(Qdata::new(settings.init(state_dim + action_dim, 1)))
    }

    fn create_uniform_policy(action_space: Space) -> Result<Self::Policy, String> {
        let (action_low, action_high) = action_space
            .into_continuous_bounds()
            .ok_or("DeterministicPolicyGradient requires a continuous action space")?;
        Ok(UniformContinuousPolicy::new(action_low, action_high))
    }

    fn eval_q(
        model: &MlpModel<TrainingBackend>,
        states: Tensor<TrainingBackend, 2>,
        actions: Self::TensorType,
    ) -> Tensor<TrainingBackend, 1> {
        model
            .forward(Tensor::cat(vec![states, actions], 1))
            .squeeze_dim(1)
    }

    fn soft_state_values(
        pi: &Self::Actor,
        q1: &MlpModel<TrainingBackend>,
        q2: &MlpModel<TrainingBackend>,
        states: Tensor<TrainingBackend, 2>,
        alpha: f64,
        rng: &mut dyn rand::Rng,
    ) -> (Tensor<TrainingBackend, 1>, Tensor<TrainingBackend, 1>) {
        let (actions, log_probs) = pi.sample_actions(states.clone(), rng);
        let q_values = Self::eval_q(q1, states.clone(), actions.clone()).min_pair(Self::eval_q(
            q2,
            states.clone(),
            actions.clone(),
        ));
        let entropy = -log_probs;
        let soft_values = q_values + entropy.clone() * alpha;

        (soft_values, entropy)
    }
}

struct Qdata {
    pred: MlpModel<TrainingBackend>,
    target: MlpModel<TrainingBackend>,

    optimizer: AdamOptimizer<MlpModel<TrainingBackend>>,
}

impl Qdata {
    fn new(network: MlpModel<TrainingBackend>) -> Self {
        Self {
            pred: network.clone(),
            target: network,
            optimizer: AdamConfig::new().init(),
        }
    }
}
