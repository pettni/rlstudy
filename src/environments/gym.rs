use crate::environment::{Environment, Space, Step};
use ndarray::Array1;
use pyo3::{
    IntoPyObjectExt,
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyTuple},
};
use rand::RngExt;
use std::path::PathBuf;

pub struct GymOptions {
    pub max_episode_steps: Option<usize>,
    pub normalize_observations: bool,
    pub video_folder: Option<PathBuf>,
    pub video_every: usize,
}

impl Default for GymOptions {
    fn default() -> Self {
        Self {
            max_episode_steps: None,
            normalize_observations: false,
            video_folder: None,
            video_every: 1,
        }
    }
}

pub enum GymEnvironment {
    Discrete(DiscreteGymEnvironment),
    Continuous(ContinuousGymEnvironment),
}

pub struct DiscreteGymEnvironment {
    core: GymCore,
    action_space: Space,
}

pub struct ContinuousGymEnvironment {
    core: GymCore,
    action_space: Space,
}

struct GymCore {
    env: Py<PyAny>,
    observation_space: Space,
}

impl GymEnvironment {
    pub fn new(id: &str, options: &GymOptions) -> PyResult<Self> {
        Python::attach(|py| {
            if options.video_folder.is_some()
                && std::env::var_os("MUJOCO_GL").is_none()
                && std::env::var_os("DISPLAY").is_none()
            {
                PyModule::import(py, "os")?
                    .getattr("environ")?
                    .set_item("MUJOCO_GL", "egl")?;
            }
            let gym = PyModule::import(py, "gymnasium")?;
            let kwargs = PyDict::new(py);
            if let Some(limit) = options.max_episode_steps {
                kwargs.set_item("max_episode_steps", limit)?;
            }
            if options.video_folder.is_some() {
                kwargs.set_item("render_mode", "rgb_array")?;
            }
            let mut env = gym.call_method("make", (id,), Some(&kwargs))?;
            if options.normalize_observations {
                env = gym
                    .getattr("wrappers")?
                    .call_method1("RescaleObservation", (env, -1.0, 1.0))?;
            }
            if let Some(folder) = &options.video_folder {
                let folder = folder
                    .to_str()
                    .ok_or_else(|| PyValueError::new_err("video folder must be valid UTF-8"))?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("video_folder", folder)?;
                let globals = PyDict::new(py);
                globals.set_item("every", options.video_every)?;
                let episode_trigger = PyModule::import(py, "builtins")?.call_method1(
                    "eval",
                    ("lambda episode_id: episode_id % every == 0", globals),
                )?;
                kwargs.set_item("episode_trigger", episode_trigger)?;
                env = gym
                    .getattr("wrappers")?
                    .call_method("RecordVideo", (env,), Some(&kwargs))?;
            }
            let env = env.unbind();
            let bound = env.bind(py);
            let observation_space = bound.getattr("observation_space")?;
            let action_space = bound.getattr("action_space")?;

            let observation_low = extract_vector(&observation_space.getattr("low")?)?;
            let observation_high = extract_vector(&observation_space.getattr("high")?)?;
            if observation_low.len() != observation_high.len() {
                return Err(PyValueError::new_err(
                    "Gym observation-space bounds have different dimensions",
                ));
            }
            let core = GymCore {
                env,
                observation_space: Space::Continuous {
                    low: observation_low,
                    high: observation_high,
                },
            };

            if let Ok(num_actions) = action_space
                .getattr("n")
                .and_then(|value| value.extract::<usize>())
            {
                return Ok(Self::Discrete(DiscreteGymEnvironment {
                    core,
                    action_space: Space::Discrete { size: num_actions },
                }));
            }

            let action_low = extract_vector(&action_space.getattr("low")?)?;
            let action_high = extract_vector(&action_space.getattr("high")?)?;
            if action_low.len() != action_high.len() {
                return Err(PyValueError::new_err(
                    "Gym action-space bounds have different dimensions",
                ));
            }

            Ok(Self::Continuous(ContinuousGymEnvironment {
                core,
                action_space: Space::Continuous {
                    low: action_low,
                    high: action_high,
                },
            }))
        })
    }
}

impl GymCore {
    fn reset_with_seed(&mut self, seed: u64) -> PyResult<Array1<f64>> {
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("seed", seed)?;
            let result = self.env.bind(py).call_method("reset", (), Some(&kwargs))?;
            let tuple = result.cast_into::<PyTuple>()?;
            extract_vector(&tuple.get_item(0)?)
        })
    }

    fn step(&mut self, action: &Bound<'_, PyAny>) -> PyResult<Step<Array1<f64>>> {
        let result = self.env.bind(action.py()).call_method1("step", (action,))?;
        let tuple = result.cast_into::<PyTuple>()?;
        Ok(Step {
            observation: extract_vector(&tuple.get_item(0)?)?,
            reward: tuple.get_item(1)?.extract()?,
            terminated: tuple.get_item(2)?.extract()?,
            truncated: tuple.get_item(3)?.extract()?,
        })
    }
}

impl Environment for DiscreteGymEnvironment {
    type Observation = Array1<f64>;
    type Action = usize;

    fn state_space(&self) -> Space {
        self.core.observation_space.clone()
    }

    fn action_space(&self) -> Space {
        self.action_space.clone()
    }

    fn reset(&mut self, rng: &mut dyn rand::Rng) -> Array1<f64> {
        self.core
            .reset_with_seed(rng.random())
            .expect("Gymnasium reset failed")
    }

    fn step(&mut self, action: usize, _rng: &mut dyn rand::Rng) -> Step<Array1<f64>> {
        Python::attach(|py| {
            let action = action.into_bound_py_any(py)?;
            self.core.step(&action)
        })
        .expect("Gymnasium step failed")
    }
}

impl Environment for ContinuousGymEnvironment {
    type Observation = Array1<f64>;
    type Action = Array1<f64>;

    fn state_space(&self) -> Space {
        self.core.observation_space.clone()
    }

    fn action_space(&self) -> Space {
        self.action_space.clone()
    }

    fn reset(&mut self, rng: &mut dyn rand::Rng) -> Array1<f64> {
        self.core
            .reset_with_seed(rng.random())
            .expect("Gymnasium reset failed")
    }

    fn step(&mut self, action: Array1<f64>, _rng: &mut dyn rand::Rng) -> Step<Array1<f64>> {
        Python::attach(|py| {
            let numpy = PyModule::import(py, "numpy")?;
            let action = numpy.call_method1("asarray", (action.to_vec(),))?;
            let action = action.call_method1("astype", (numpy.getattr("float32")?,))?;
            self.core.step(&action)
        })
        .expect("Gymnasium step failed")
    }
}

impl Drop for GymCore {
    fn drop(&mut self) {
        Python::attach(|py| {
            let _ = self.env.bind(py).call_method0("close");
        });
    }
}

fn extract_vector(value: &Bound<'_, PyAny>) -> PyResult<Array1<f64>> {
    Ok(Array1::from_vec(value.call_method0("tolist")?.extract()?))
}
