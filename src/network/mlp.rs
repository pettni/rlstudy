/// Trivial MLP model
use burn::{
    module::Module,
    nn::{
        Linear, LinearConfig,
        activation::{Activation as BurnActivation, ActivationConfig as BurnActivationConfig},
    },
    tensor::{Tensor, backend::Backend},
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
