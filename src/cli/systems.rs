use rlstudy::environments::{
    ContinuousGymEnvironment, DiscreteGymEnvironment, GymEnvironment, GymOptions, grid_world,
    load_grid_world,
};
use rlstudy::model::TabularModel;
use std::path::Path;

pub(super) enum SelectedSystem {
    Model(TabularModel),
    DiscreteGym(DiscreteGymEnvironment),
    ContinuousGym(ContinuousGymEnvironment),
}

impl SelectedSystem {
    pub(super) fn as_model(&self) -> Option<&TabularModel> {
        match self {
            Self::Model(model) => Some(model),
            Self::DiscreteGym(_) | Self::ContinuousGym(_) => None,
        }
    }
}

pub(super) fn build_system(
    grid_world_path: Option<&Path>,
    gym_id: Option<&str>,
    gym_options: &GymOptions,
) -> Result<SelectedSystem, Box<dyn std::error::Error>> {
    match (grid_world_path, gym_id) {
        (Some(path), None) => Ok(SelectedSystem::Model(load_grid_world(path)?)),
        (None, Some(id)) => Ok(match GymEnvironment::new(id, gym_options)? {
            GymEnvironment::Discrete(environment) => SelectedSystem::DiscreteGym(environment),
            GymEnvironment::Continuous(environment) => SelectedSystem::ContinuousGym(environment),
        }),
        (None, None) => Ok(SelectedSystem::Model(grid_world(3)?)),
        (Some(_), Some(_)) => unreachable!("Clap makes system arguments mutually exclusive"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_grid_world_from_yaml_path() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("grid_worlds/zhao_3x3.yaml");
        let system = build_system(Some(&path), None, &GymOptions::default()).unwrap();
        let model = system.as_model().unwrap();

        assert_eq!(model.states().len(), 9);
        assert_eq!(model.initial_distribution().probability_of(&0), 1.0);
    }
}
