/// Trivial MLP model
use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct MlpModel<B: Backend> {
    l1: Linear<B>,
    act: Relu,
    l2: Linear<B>,
}

impl<B: Backend> MlpModel<B> {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(input_dim, hidden_dim).init(device),
            act: Relu::new(),
            l2: LinearConfig::new(hidden_dim, output_dim).init(device),
        }
    }

    pub fn forward(&self, observations: Tensor<B, 2>) -> Tensor<B, 2> {
        let x1 = self.l1.forward(observations);
        let x2 = self.act.forward(x1);
        self.l2.forward(x2)
    }
}
