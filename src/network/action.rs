use crate::backend::{TrainingBackend, device};
use burn::{
    Tensor,
    tensor::{ElementConversion, Int, TensorData},
};
use ndarray::Array1;

pub trait ActionType: Clone {
    type TensorType: Clone;

    fn tensor_data(actions: &[&Self]) -> Self::TensorType;
    fn native_data(action: Self::TensorType) -> Self;
}

impl ActionType for usize {
    type TensorType = Tensor<TrainingBackend, 1, Int>;

    fn tensor_data(actions: &[&Self]) -> Self::TensorType {
        let values = actions.iter().map(|x| **x as i64).collect::<Vec<_>>();
        Tensor::from_data(TensorData::new(values, [actions.len()]), device())
    }

    fn native_data(action: Self::TensorType) -> Self {
        action.into_scalar().elem::<i64>() as usize
    }
}

impl ActionType for Array1<f64> {
    type TensorType = Tensor<TrainingBackend, 2>;

    fn tensor_data(actions: &[&Self]) -> Self::TensorType {
        let action_dim = actions.first().expect("buffer is empty").len();
        let values = actions
            .iter()
            .flat_map(|a| a.iter().copied())
            .collect::<Vec<_>>();
        Tensor::from_data(
            TensorData::new(values, [actions.len(), action_dim]),
            device(),
        )
    }

    fn native_data(action: Self::TensorType) -> Self {
        Array1::from_iter(action.into_data().iter())
    }
}
