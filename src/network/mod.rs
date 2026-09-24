pub mod action;
pub mod actor;
pub mod ema;
pub mod loss;
pub mod mlp;
use burn::{
    module::{Module, ModuleVisitor, Param},
    optim::GradientsParams,
    tensor::{Tensor, backend::AutodiffBackend},
};

pub fn gradient_norm<B, M>(module: &M, gradients: &GradientsParams) -> f64
where
    B: AutodiffBackend,
    M: Module<B>,
{
    struct GradientNormVisitor<'a, B: AutodiffBackend> {
        gradients: &'a GradientsParams,
        norm_squared: f64,
        backend: std::marker::PhantomData<B>,
    }

    impl<B: AutodiffBackend> ModuleVisitor<B> for GradientNormVisitor<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(gradient) = self.gradients.get::<B::InnerBackend, D>(param.id) {
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

    let mut visitor = GradientNormVisitor::<B> {
        gradients,
        norm_squared: 0.0,
        backend: std::marker::PhantomData,
    };
    module.visit(&mut visitor);
    visitor.norm_squared.sqrt()
}
