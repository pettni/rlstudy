use super::DqnMode;
use crate::backend::{InferenceBackend, TrainingBackend, device};
use crate::network::loss::Loss;
use crate::network::mlp::MlpModel;
use crate::network::{ema, gradient_norm};
use crate::policy::dq::{DeepQPolicy, DeepQPolicyMode};
use crate::transition_buffer::TransitionBufferTensorData;
use burn::{
    module::AutodiffModule,
    optim::{
        Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor,
        grad_clipping::GradientClippingConfig,
    },
    tensor::backend::Backend,
};

pub(super) struct DqnTrainer {
    online: MlpModel<TrainingBackend>,
    target: MlpModel<TrainingBackend>,
    optimizer: OptimizerAdaptor<Adam, MlpModel<TrainingBackend>, TrainingBackend>,
    mode: DqnMode,
}

impl DqnTrainer {
    pub(super) fn new(
        feature_dim: usize,
        hidden_dim: usize,
        num_actions: usize,
        gradient_clip_value: Option<f32>,
        mode: DqnMode,
        seed: u64,
    ) -> Self {
        <TrainingBackend as Backend>::seed(device(), seed);
        let target = MlpModel::new(feature_dim, hidden_dim, num_actions, device());

        Self {
            online: target.clone(),
            target,
            optimizer: AdamConfig::new()
                .with_grad_clipping(gradient_clip_value.map(GradientClippingConfig::Value))
                .init(),
            mode,
        }
    }

    pub(super) fn update(
        mut self,
        batch: TransitionBufferTensorData<usize>,
        learning_rate: f64,
        gamma: f64,
        loss: Loss,
    ) -> (Self, f64, f64) {
        let batch_size = batch.s.shape()[0];

        let predicted = self
            .online
            .forward(batch.s)
            .gather(1, batch.a.reshape([batch_size, 1]))
            .squeeze_dim(1);
        let actions = match self.mode {
            DqnMode::Dqn => self.target.forward(batch.sp.clone()).argmax(1),
            DqnMode::Ddqn => self.online.forward(batch.sp.clone()).argmax(1),
        };
        let next_values = self
            .target
            .forward(batch.sp)
            .gather(1, actions)
            .squeeze_dim(1);
        let bootstrap = batch.terminated.bool_not().float();
        let target = batch.r + bootstrap * gamma * next_values.detach();
        let loss = loss.forward(predicted, target);
        let loss_value = loss
            .clone()
            .into_data()
            .iter::<f64>()
            .next()
            .expect("loss tensor has one element");
        let gradients = GradientsParams::from_grads(loss.backward(), &self.online);
        let gradient_norm = gradient_norm::<TrainingBackend, _>(&self.online, &gradients);

        self.online = self.optimizer.step(learning_rate, self.online, gradients);
        (self, loss_value, gradient_norm)
    }

    pub(super) fn sync_target(&mut self) {
        self.target = self.online.clone();
    }

    pub(super) fn soft_update_target(&mut self, rate: f64) {
        self.target = ema::update::<TrainingBackend, _>(self.target.clone(), &self.online, rate);
    }

    pub(super) fn policy_view(&self, mode: DeepQPolicyMode) -> DeepQPolicy<'_, TrainingBackend> {
        DeepQPolicy::borrowed(&self.online, mode)
    }

    pub(super) fn inference_model(&self) -> MlpModel<InferenceBackend> {
        self.online.valid()
    }

    pub(super) fn into_policy(self) -> DeepQPolicy<'static, InferenceBackend> {
        DeepQPolicy::new(self.online.valid(), DeepQPolicyMode::Greedy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::optim::grad_clipping::GradientClipping;

    #[test]
    fn configures_gradient_value_clipping() {
        let trainer = DqnTrainer::new(4, 64, 2, Some(100.0), DqnMode::Dqn, 42);

        assert!(matches!(
            trainer.optimizer.grad_clipping(),
            Some(GradientClipping::Value(100.0))
        ));
    }
}
