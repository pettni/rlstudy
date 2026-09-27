use crate::backend::{Device, TrainingBackend, device};
use burn::{
    Tensor,
    tensor::{ElementConversion, Int, TensorData, backend::Backend},
};
use ndarray::Array1;

pub fn state_tensor<B>(state: &Array1<f64>) -> Tensor<B, 2>
where
    B: Backend<Device = Device>,
{
    Tensor::from_data(TensorData::new(state.to_vec(), [1, state.len()]), device())
}

pub trait TensorValue: Clone {
    type BatchTensor: Clone;

    fn singleton_tensor(value: &Self) -> Self::BatchTensor;
    fn batch_tensor(values: &[&Self]) -> Self::BatchTensor;
    fn from_singleton_tensor(value: Self::BatchTensor) -> Self;
}

impl TensorValue for usize {
    type BatchTensor = Tensor<TrainingBackend, 1, Int>;

    fn singleton_tensor(value: &Self) -> Self::BatchTensor {
        let tensor_values = vec![*value as i64];
        Tensor::from_data(TensorData::new(tensor_values, [1]), device())
    }

    fn batch_tensor(values: &[&Self]) -> Self::BatchTensor {
        let tensor_values = values.iter().map(|x| **x as i64).collect::<Vec<_>>();
        Tensor::from_data(TensorData::new(tensor_values, [values.len()]), device())
    }

    fn from_singleton_tensor(value: Self::BatchTensor) -> Self {
        value.into_scalar().elem::<i64>() as usize
    }
}

impl TensorValue for Array1<f64> {
    type BatchTensor = Tensor<TrainingBackend, 2>;

    fn singleton_tensor(values: &Self) -> Self::BatchTensor {
        state_tensor::<TrainingBackend>(values)
    }

    fn batch_tensor(values: &[&Self]) -> Self::BatchTensor {
        let value_dim = values.first().expect("buffer is empty").len();
        let tensor_values = values
            .iter()
            .flat_map(|value| value.iter().copied())
            .collect::<Vec<_>>();
        Tensor::from_data(
            TensorData::new(tensor_values, [values.len(), value_dim]),
            device(),
        )
    }

    fn from_singleton_tensor(value: Self::BatchTensor) -> Self {
        Array1::from_iter(value.into_data().iter())
    }
}
