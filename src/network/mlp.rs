/// Trivial MLP model
use burn::{
    module::Module,
    nn::{
        Linear, LinearConfig,
        activation::{Activation as BurnActivation, ActivationConfig as BurnActivationConfig},
    },
    tensor::{Tensor, TensorData, backend::Backend},
};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::schedule::Schedule;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Relu,
    Tanh,
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Module, Debug)]
pub struct OutputTransform<B: Backend> {
    center: Tensor<B, 2>,
    scale: Tensor<B, 2>,
}

impl<B: Backend> OutputTransform<B> {
    pub fn bounded_tanh(low: &[f64], high: &[f64], device: &B::Device) -> Self {
        let center = low
            .iter()
            .zip(high)
            .map(|(&low, &high)| (low + high) / 2.0)
            .collect::<Vec<_>>();
        let scale = low
            .iter()
            .zip(high)
            .map(|(&low, &high)| (high - low) / 2.0)
            .collect::<Vec<_>>();
        let output_dim = center.len();

        Self {
            center: Tensor::from_data(TensorData::new(center, [1, output_dim]), device),
            scale: Tensor::from_data(TensorData::new(scale, [1, output_dim]), device),
        }
    }

    fn forward(&self, output: Tensor<B, 2>) -> Tensor<B, 2> {
        self.center.clone() + self.scale.clone() * output.tanh()
    }
}

#[derive(Module, Debug)]
pub struct MlpModel<B: Backend> {
    layers: Vec<Linear<B>>,
    activation: BurnActivation<B>,
    output_transform: Option<OutputTransform<B>>,
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

        Self {
            layers,
            activation,
            output_transform: None,
        }
    }

    pub fn with_output_transform(mut self, output_transform: OutputTransform<B>) -> Self {
        self.output_transform = Some(output_transform);
        self
    }

    pub fn forward(&self, mut observations: Tensor<B, 2>) -> Tensor<B, 2> {
        let (output_layer, hidden_layers) =
            self.layers.split_last().expect("MLP has an output layer");
        for layer in hidden_layers {
            observations = self.activation.forward(layer.forward(observations));
        }
        let output = output_layer.forward(observations);
        match &self.output_transform {
            Some(transform) => transform.forward(output),
            None => output,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{InferenceBackend, TrainingBackend, device};
    use burn::{
        backend::Flex,
        module::{AutodiffModule, Reinitializer},
        tensor::Tensor,
    };

    #[test]
    fn supports_multiple_hidden_layers_activations_and_output_transforms() {
        let flex_device = Default::default();

        for activation in [Activation::Relu, Activation::Tanh] {
            let model =
                MlpModel::<Flex>::new_with_hidden_layers(3, &[4, 5], 2, activation, &flex_device);
            let output = model.forward(Tensor::zeros([7, 3], &flex_device));

            assert_eq!(output.dims(), [7, 2]);
        }

        let model = Reinitializer::<TrainingBackend>::new()
            .constant_float(0.0)
            .apply(MlpModel::new(1, 2, 2, device()))
            .with_output_transform(OutputTransform::bounded_tanh(
                &[-2.0, 1.0],
                &[2.0, 3.0],
                device(),
            ))
            .valid();
        let output = model.forward(Tensor::<InferenceBackend, 2>::zeros([1, 1], device()));

        assert_eq!(
            output.into_data().iter::<f64>().collect::<Vec<_>>(),
            vec![0.0, 2.0]
        );
    }
}
