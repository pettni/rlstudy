use burn::{
    Tensor,
    nn::loss::{Reduction, SmoothL1LossConfig},
    tensor::backend::Backend,
};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, ValueEnum, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Loss {
    #[default]
    Huber,
    L2,
}

impl Loss {
    pub(crate) fn forward<B: Backend, const D: usize>(
        self,
        predicted: Tensor<B, D>,
        target: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        match self {
            Self::Huber => SmoothL1LossConfig::new().init().forward_with_reduction(
                predicted,
                target,
                Reduction::Mean,
            ),
            Self::L2 => (target - predicted).square().mean(),
        }
    }
}
