use crate::environment::{Environment, Space, Step};
use ndarray::Array1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rand::RngExt;

pub struct GymEnvironment {
    env: Py<PyAny>,
    observation_space: Space,
    action_space: Space,
}

impl GymEnvironment {
    pub fn new(id: &str) -> PyResult<Self> {
        Python::attach(|py| {
            let env = PyModule::import(py, "gymnasium")?
                .call_method1("make", (id,))?
                .unbind();
            let bound = env.bind(py);
            let observation_space = bound.getattr("observation_space")?;
            let action_space = bound.getattr("action_space")?;

            let low = extract_vector(&observation_space.getattr("low")?)?;
            let _high = extract_vector(&observation_space.getattr("high")?)?;
            let num_actions = action_space.getattr("n")?.extract()?;

            Ok(Self {
                env,
                observation_space: Space::Continuous { dim: low.len() },
                action_space: Space::Discrete { size: num_actions },
            })
        })
    }

    fn reset_with_seed(&mut self, seed: u64) -> PyResult<Array1<f64>> {
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("seed", seed)?;
            let result = self.env.bind(py).call_method("reset", (), Some(&kwargs))?;
            let tuple = result.cast_into::<PyTuple>()?;
            extract_vector(&tuple.get_item(0)?)
        })
    }

    fn try_step(&mut self, action: usize) -> PyResult<Step<Array1<f64>>> {
        Python::attach(|py| {
            let result = self.env.bind(py).call_method1("step", (action,))?;
            let tuple = result.cast_into::<PyTuple>()?;
            Ok(Step {
                observation: extract_vector(&tuple.get_item(0)?)?,
                reward: tuple.get_item(1)?.extract()?,
                terminated: tuple.get_item(2)?.extract()?,
                truncated: tuple.get_item(3)?.extract()?,
            })
        })
    }
}

impl Environment for GymEnvironment {
    type Observation = Array1<f64>;
    type Action = usize;

    fn state_space(&self) -> Space {
        self.observation_space.clone()
    }

    fn action_space(&self) -> Space {
        self.action_space.clone()
    }

    fn reset(&mut self, rng: &mut dyn rand::Rng) -> Array1<f64> {
        self.reset_with_seed(rng.random())
            .expect("Gymnasium reset failed")
    }

    fn step(&mut self, action: usize, _rng: &mut dyn rand::Rng) -> Step<Array1<f64>> {
        self.try_step(action).expect("Gymnasium step failed")
    }
}

impl Drop for GymEnvironment {
    fn drop(&mut self) {
        Python::attach(|py| {
            let _ = self.env.bind(py).call_method0("close");
        });
    }
}

fn extract_vector(value: &Bound<'_, PyAny>) -> PyResult<Array1<f64>> {
    Ok(Array1::from_vec(value.call_method0("tolist")?.extract()?))
}
