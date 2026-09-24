use burn::backend::Autodiff;
use std::sync::LazyLock;

#[cfg(feature = "cuda")]
use burn::backend::{Cuda, cuda::CudaDevice};
#[cfg(not(feature = "cuda"))]
use burn::backend::{Flex, flex::FlexDevice};

#[cfg(feature = "cuda")]
pub type InferenceBackend = Cuda;
#[cfg(not(feature = "cuda"))]
pub type InferenceBackend = Flex;

pub type TrainingBackend = Autodiff<InferenceBackend>;

#[cfg(feature = "cuda")]
pub type Device = CudaDevice;
#[cfg(not(feature = "cuda"))]
pub type Device = FlexDevice;

static DEVICE: LazyLock<Device> = LazyLock::new(Device::default);

pub fn device() -> &'static Device {
    &DEVICE
}
