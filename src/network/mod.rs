pub mod action;
pub mod actor;
pub mod ema;
pub mod loss;
mod mlp;
mod settings;
use crate::backend::{InferenceBackend, TrainingBackend};
use burn::{
    module::{Module, ModuleVisitor, Param},
    optim::GradientsParams,
    tensor::Tensor,
};
pub use mlp::{Activation, MlpModel, OutputTransform};
pub use settings::{AdamOptimizer, MlpSettings};

pub fn gradient_norm<M>(module: &M, gradients: &GradientsParams) -> f64
where
    M: Module<TrainingBackend>,
{
    let mut visitor = GradientNormVisitor {
        gradients,
        norm_squared: 0.0,
    };
    module.visit(&mut visitor);
    visitor.norm_squared.sqrt()
}

struct GradientNormVisitor<'a> {
    gradients: &'a GradientsParams,
    norm_squared: f64,
}

impl ModuleVisitor<TrainingBackend> for GradientNormVisitor<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<TrainingBackend, D>>) {
        if let Some(gradient) = self.gradients.get::<InferenceBackend, D>(param.id) {
            self.norm_squared += gradient
                .square()
                .sum()
                .into_data()
                .iter::<f64>()
                .next()
                .expect("gradient norm tensor has one element");
        }
    }
}
