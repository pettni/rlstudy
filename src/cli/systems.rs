use rlstudy::environment::{BlackBox, Environment};
use rlstudy::environments::grid_world;
use rlstudy::model::TabularModel;

pub(super) enum SelectedSystem {
    Model(TabularModel),
    BlackBox(Box<dyn Environment<State = usize, Action = usize>>),
}

impl SelectedSystem {
    pub(super) fn as_model(&self) -> Option<&TabularModel> {
        match self {
            Self::Model(model) => Some(model),
            Self::BlackBox(_) => None,
        }
    }
}

pub(super) fn build_system(name: &str) -> Result<SelectedSystem, Box<dyn std::error::Error>> {
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
