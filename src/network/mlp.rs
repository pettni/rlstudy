/// Trivial MLP model
use burn::{
    module::{Module, Param},
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

    pub fn soft_update(mut self, online: &Self, rate: f64) -> Self {
        self.l1.weight = average_param(self.l1.weight, &online.l1.weight, rate);
        self.l1.bias = self
            .l1
            .bias
            .map(|target| average_param(target, online.l1.bias.as_ref().unwrap(), rate));
        self.l2.weight = average_param(self.l2.weight, &online.l2.weight, rate);
        self.l2.bias = self
            .l2
            .bias
            .map(|target| average_param(target, online.l2.bias.as_ref().unwrap(), rate));
        self
    }
}

fn average_param<B: Backend, const D: usize>(
    target: Param<Tensor<B, D>>,
    online: &Param<Tensor<B, D>>,
    rate: f64,
) -> Param<Tensor<B, D>> {
    let (id, target, mapper) = target.consume();
    let value = (target * (1.0 - rate) + online.val().detach() * rate).detach();
    Param::from_mapped_value(id, value, mapper)
}
