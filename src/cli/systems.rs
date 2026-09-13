use rlstudy::environment::{BlackBox, Environment, Space};
use rlstudy::environments::{grid_world, load_grid_world};
use rlstudy::model::TabularModel;
use std::path::Path;

pub(super) enum SelectedSystem {
    Model(TabularModel),
    BlackBox(Box<dyn Environment<Observation = usize, Action = usize>>),
}

impl SelectedSystem {
    pub(super) fn as_model(&self) -> Option<&TabularModel> {
        match self {
            Self::Model(model) => Some(model),
            Self::BlackBox(_) => None,
        }
    }

    pub(super) fn num_actions(&self) -> usize {
        let space = match self {
            Self::Model(model) => return model.actions().len(),
            Self::BlackBox(environment) => environment.action_space(),
        };
        let Space::Discrete { size } = space else {
            unreachable!("CLI systems have discrete actions")
        };
        size
    }
}

pub(super) fn build_system(name: &str) -> Result<SelectedSystem, Box<dyn std::error::Error>> {
    let path = Path::new(name);
    if matches!(
        path.extension().and_then(|extension| extension.to_str()),
        Some("yaml") | Some("yml")
    ) {
        return Ok(SelectedSystem::Model(load_grid_world(path)?));
    }

    match name {
        "gridworld" => Ok(SelectedSystem::Model(grid_world(3)?)),
        "blackbox-line" => Ok(SelectedSystem::BlackBox(Box::new(BlackBox::new(
            3,
            2,
            0,
            |state, action| ((state + action).min(2), if state == 2 { 1.0 } else { 0.0 }),
        )))),
        other => Err(format!("unknown system: {other}").into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_grid_world_from_yaml_path() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("grid_worlds/zhao_3x3.yaml");
        let system = build_system(path.to_str().unwrap()).unwrap();
        let model = system.as_model().unwrap();

        assert_eq!(model.states().len(), 9);
        assert_eq!(model.initial_distribution().probability_of(&0), 1.0);
    }
}
