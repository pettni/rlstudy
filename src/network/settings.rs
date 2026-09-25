use crate::{network::mlp::Activation, schedule::Schedule};
use serde::{Deserialize, Serialize};

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
