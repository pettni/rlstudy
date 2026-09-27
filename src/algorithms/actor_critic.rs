use crate::algorithms::{ModelFreeSolver, SolveResult, SolveStatus};
use crate::backend::{TrainingBackend, device};
use crate::environment::{Environment, Space};
use crate::network::{
    AdamOptimizer, MlpModel, MlpSettings, TensorValue,
    actor::{Actor, CategoricalActor},
    gradient_norm, state_tensor,
};
use crate::policy::Policy;
use crate::reporting::{Reporter, Step};
use crate::schedule::Schedule;
use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{ElementConversion, backend::Backend},
};
use ndarray::Array1;
use rand::RngExt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ActorCriticConfig {
    pub gamma: f64,
    pub pi_network: MlpSettings,
    pub v_network: MlpSettings,
    pub episodes: usize,
    pub episode_length: usize,
}

impl Default for ActorCriticConfig {
    fn default() -> Self {
        let learning_rate = Schedule::Linear {
            initial: 0.001,
            final_value: 0.0001,
            steps: 999,
        };
        Self {
            gamma: 0.9,
            pi_network: MlpSettings {
                learning_rate: learning_rate.clone(),
                ..MlpSettings::default()
            },
            v_network: MlpSettings {
                learning_rate,
                ..MlpSettings::default()
            },
            episodes: 1_000,
            episode_length: 100,
        }
    }
}

pub struct ActorCritic {
    config: ActorCriticConfig,
}

impl ActorCritic {
    pub fn new(config: ActorCriticConfig) -> Result<Self, String> {
        if !config.gamma.is_finite() || !(0.0..=1.0).contains(&config.gamma) {
            return Err("gamma must be finite and in [0, 1]".into());
        }
        let valid_rate = |rate: f64| rate.is_finite() && rate > 0.0 && rate <= 1.0;
        for schedule in [
            &config.pi_network.learning_rate,
            &config.v_network.learning_rate,
        ] {
            let valid = match schedule {
                Schedule::Constant { value } => valid_rate(*value),
                Schedule::Linear {
                    initial,
                    final_value,
                    steps,
                } => valid_rate(*initial) && valid_rate(*final_value) && *steps > 0,
                Schedule::Exponential {
                    initial,
                    final_value,
                    decay,
                } => valid_rate(*initial) && valid_rate(*final_value) && *decay > 0,
                Schedule::Stepwise { initial, changes } => {
                    valid_rate(*initial)
                        && changes.iter().all(|(_, rate)| valid_rate(*rate))
                        && changes.windows(2).all(|pair| pair[0].0 < pair[1].0)
                }
            };
            if !valid {
                return Err(
                    "learning rates must be finite and in (0, 1], with increasing steps".into(),
                );
            }
        }
        if [&config.pi_network, &config.v_network]
            .into_iter()
            .any(|network| network.hidden_dims.is_empty() || network.hidden_dims.contains(&0))
        {
            return Err("hidden dims must be positive".into());
        }
        if config.episodes == 0 || config.episode_length == 0 {
            return Err("episodes and episode length must be positive".into());
        }
        Ok(Self { config })
    }
}

impl ModelFreeSolver for ActorCritic {
    type Observation = Array1<f64>;
    type Action = usize;

    fn solve_with_reporter(
        &self,
        env: &mut dyn Environment<Observation = Array1<f64>, Action = usize>,
        rng: &mut dyn rand::Rng,
        reporter: &mut Reporter<'_, Array1<f64>>,
    ) -> Result<SolveResult<Array1<f64>>, String> {
        let n_s = env
            .state_space()
            .continuous_dim()
            .ok_or("Actor-critic requires a continuous state space")?;
        let Space::Discrete { size: n_a } = env.action_space() else {
            return Err("Actor-critic requires a discrete action space".into());
        };

        <TrainingBackend as Backend>::seed(device(), rng.random());

        let mut pi = CategoricalActor::new(
            n_s,
            &self.config.pi_network.hidden_dims,
            n_a,
            self.config.pi_network.activation,
            device(),
        );
        let mut pi_optimizer: AdamOptimizer<CategoricalActor<TrainingBackend>> =
            AdamConfig::new().init();
        let mut v = self.config.v_network.init(n_s, 1);
        let mut v_optimizer: AdamOptimizer<MlpModel<TrainingBackend>> = AdamConfig::new().init();

        for episode_index in 0..self.config.episodes {
            let mut state = env.reset(rng);
            let mut episode_return = 0.0;
            let mut episode_length = 0;
            let mut entropy_sum = 0.0;
            let mut actor_gradient_norm_sum = 0.0;
            let mut critic_gradient_norm_sum = 0.0;
            let actor_learning_rate = self.config.pi_network.learning_rate.value(episode_index);
            let critic_learning_rate = self.config.v_network.learning_rate.value(episode_index);
            for t in 0..self.config.episode_length {
                let action = pi.sample_action(&state, rng);
                let step = env.step(action, rng);

                let st = state_tensor::<TrainingBackend>(&state);
                let at = usize::singleton_tensor(&action);
                let stp = state_tensor::<TrainingBackend>(&step.observation);

                let live = if step.terminated { 0.0 } else { 1.0 };

                let td_target = step.reward + live * self.config.gamma * v.forward(stp).detach();
                let td_error = td_target - v.forward(st.clone());

                // critic update
                let critic_loss = td_error.clone().square();
                let critic_grad = GradientsParams::from_grads(critic_loss.backward(), &v);

                critic_gradient_norm_sum += gradient_norm(&v, &critic_grad);

                v = v_optimizer.step(critic_learning_rate, v, critic_grad);

                // actor update
                let ln_pi = pi.log_probs(st.clone(), &at);
                let entropy = pi.entropy(st).mean().into_scalar().elem::<f64>();
                let actor_loss = (td_error.clone().detach().squeeze_dim(1) * ln_pi).neg();
                let actor_grad = GradientsParams::from_grads(actor_loss.backward(), &pi);

                actor_gradient_norm_sum += gradient_norm(&pi, &actor_grad);

                pi = pi_optimizer.step(actor_learning_rate, pi, actor_grad);

                // metrics
                entropy_sum += entropy;
                episode_return += step.reward;
                episode_length = t + 1;

                // loop state
                state = step.observation;
                if step.terminated || step.truncated {
                    break;
                }
            }

            reporter.record(
                Step::Episode(episode_index + 1),
                &[
                    ("train/return", episode_return),
                    ("train/length", episode_length as f64),
                    ("policy/entropy", entropy_sum / episode_length as f64),
                    (
                        "actor/grad/norm",
                        actor_gradient_norm_sum / episode_length as f64,
                    ),
                    (
                        "critic/grad/norm",
                        critic_gradient_norm_sum / episode_length as f64,
                    ),
                    ("optimizer/actor_lr", actor_learning_rate),
                    ("optimizer/critic_lr", critic_learning_rate),
                ],
            );
            reporter.policy(Step::Episode(episode_index + 1), || pi.valid());
        }

        Ok(SolveResult {
            policy: Box::new(pi.valid()),
            status: SolveStatus::IterationLimit {
                iterations: self.config.episodes,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_is_validated() {
        assert!(
            ActorCritic::new(ActorCriticConfig {
                episodes: 0,
                ..ActorCriticConfig::default()
            })
            .is_err()
        );
        assert!(
            ActorCritic::new(ActorCriticConfig {
                pi_network: MlpSettings {
                    learning_rate: Schedule::Constant { value: 0.0 },
                    ..MlpSettings::default()
                },
                ..ActorCriticConfig::default()
            })
            .is_err()
        );
        assert!(ActorCritic::new(ActorCriticConfig::default()).is_ok());
        assert!(
            ActorCritic::new(ActorCriticConfig {
                gamma: 1.0,
                ..ActorCriticConfig::default()
            })
            .is_ok()
        );
    }
}
