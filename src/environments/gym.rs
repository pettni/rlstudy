use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};

pub struct GymEnv {
    env: Py<PyAny>,
}

impl GymEnv {
    pub fn new() -> PyResult<Self> {
        Python::attach(|py| {
            let python_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
            let sys = PyModule::import(py, "sys")?;

            sys.getattr("path")?
                .cast_into::<PyList>()?
                .insert(0, python_dir.to_string_lossy().as_ref())?;

            let module = PyModule::import(py, "gym_bridge")?;

            let env = module.getattr("GymBridge")?.call0()?.unbind();

            Ok(Self { env })
        })
    }

    pub fn reset(&self, seed: u64) -> PyResult<[f32; 4]> {
        Python::attach(|py| {
            let observation: Vec<f32> = self
                .env
                .bind(py)
                .call_method1("reset", (seed,))?
                .extract()?;

            Ok(observation.try_into().expect("Observation must extract"))
        })
    }

    pub fn step(&self, action: usize) -> PyResult<([f32; 4], f32, bool, bool)> {
        Python::attach(|py| {
            let (observation, reward, terminated, truncated): (Vec<f32>, f32, bool, bool) = self
                .env
                .bind(py)
                .call_method1("step", (action,))?
                .extract()?;

            Ok((
                observation.try_into().expect("Observation must extract"),
                reward,
                terminated,
                truncated,
            ))
        })
    }

    pub fn close(&self) -> PyResult<()> {
        Python::attach(|py| {
            self.env.bind(py).call_method0("close")?;
            Ok(())
        })
    }
}
