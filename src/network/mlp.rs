/// Trivial MLP model
use burn::{
    module::ParamId,
    module::{Module, Param},
    nn::{
        Linear, LinearConfig,
        activation::{Activation as BurnActivation, ActivationConfig as BurnActivationConfig},
    },
    optim::GradientsParams,
    tensor::{
        Tensor,
        backend::{AutodiffBackend, Backend},
    },
};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Relu,
    Tanh,
}

#[derive(Module, Debug)]
pub struct MlpModel<B: Backend> {
    layers: Vec<Linear<B>>,
    activation: BurnActivation<B>,
}

impl<B: AutodiffBackend> MlpModel<B> {
    pub fn gradient_norm(&self, gradients: &GradientsParams) -> f64 {
        let norm_squared = self
            .layers
            .iter()
            .map(|layer| {
                gradient_squared::<B::InnerBackend, 2>(gradients, layer.weight.id)
                    + gradient_squared::<B::InnerBackend, 1>(
                        gradients,
                        layer.bias.as_ref().unwrap().id,
                    )
            })
            .sum::<f64>();
        norm_squared.sqrt()
    }
}

fn gradient_squared<B: Backend, const D: usize>(gradients: &GradientsParams, id: ParamId) -> f64 {
    gradients
        .get::<B, D>(id)
        .unwrap()
        .square()
        .sum()
        .into_data()
        .iter::<f64>()
        .next()
        .unwrap()
}

impl<B: Backend> MlpModel<B> {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        Self::new_with_hidden_layers(
            input_dim,
            &[hidden_dim],
            output_dim,
            Activation::Relu,
            device,
        )
    }

    pub fn new_with_hidden_layers(
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
        activation: Activation,
        device: &B::Device,
    ) -> Self {
        let layer_dims = std::iter::once(input_dim)
            .chain(hidden_dims.iter().copied())
            .chain(std::iter::once(output_dim))
            .collect::<Vec<_>>();
        let layers = layer_dims
            .windows(2)
            .map(|dims| LinearConfig::new(dims[0], dims[1]).init(device))
            .collect();
        let activation = match activation {
            Activation::Relu => BurnActivationConfig::Relu,
            Activation::Tanh => BurnActivationConfig::Tanh,
        }
        .init(device);

        Self { layers, activation }
    }

    pub fn forward(&self, mut observations: Tensor<B, 2>) -> Tensor<B, 2> {
        let (output_layer, hidden_layers) =
            self.layers.split_last().expect("MLP has an output layer");
        for layer in hidden_layers {
            observations = self.activation.forward(layer.forward(observations));
        }
        output_layer.forward(observations)
    }

    pub fn soft_update(mut self, online: &Self, rate: f64) -> Self {
        self.layers = self
            .layers
            .into_iter()
            .enumerate()
            .map(|(index, target)| average_linear(target, &online.layers[index], rate))
            .collect();
        self
    }
}

fn average_linear<B: Backend>(mut target: Linear<B>, online: &Linear<B>, rate: f64) -> Linear<B> {
    target.weight = average_param(target.weight, &online.weight, rate);
    target.bias = target
        .bias
        .map(|target| average_param(target, online.bias.as_ref().unwrap(), rate));
    target
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{backend::Flex, tensor::Tensor};

    #[test]
    fn supports_multiple_hidden_layers_and_activations() {
        let device = Default::default();

        for activation in [Activation::Relu, Activation::Tanh] {
            let model =
                MlpModel::<Flex>::new_with_hidden_layers(3, &[4, 5], 2, activation, &device);
            let output = model.forward(Tensor::zeros([7, 3], &device));

            assert_eq!(output.dims(), [7, 2]);
        }
    }
}
