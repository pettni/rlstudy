use crate::{
    backend::{TrainingBackend, device},
    network::{Activation, MlpModel},
    schedule::Schedule,
};
use burn::optim::{Adam, adaptor::OptimizerAdaptor};
use serde::{Deserialize, Serialize};

pub type AdamOptimizer<M> = OptimizerAdaptor<Adam, M, TrainingBackend>;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MlpSettings {
    pub learning_rate: Schedule,
    pub hidden_dims: Vec<usize>,
    pub activation: Activation,
}

impl Default for MlpSettings {
    fn default() -> Self {
        Self {
            learning_rate: Schedule::Constant { value: 1e-4 },
            hidden_dims: vec![64],
            activation: Activation::Relu,
        }
    }
}

impl MlpSettings {
    pub fn init(&self, input_dim: usize, output_dim: usize) -> MlpModel<TrainingBackend> {
        MlpModel::new(
            input_dim,
            &self.hidden_dims,
            output_dim,
            self.activation,
            device(),
        )
    }
}
