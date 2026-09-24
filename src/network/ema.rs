use burn::{
    Tensor,
    module::{Module, ModuleMapper, ModuleVisitor, Param},
    tensor::{backend::Backend, container::TensorContainer},
};

pub fn update<B, M>(target: M, source: &M, rate: f64) -> M
where
    B: Backend,
    M: Module<B>,
{
    let mut collector = ParameterCollector::new();
    source.visit(&mut collector);

    let mut mapper = EmaMapper {
        source: collector.parameters,
        index: 0,
        rate,
    };
    let target = target.map(&mut mapper);
    assert!(
        mapper.source.is_empty(),
        "source and target modules have different parameters"
    );
    target
}

struct ParameterCollector {
    parameters: TensorContainer<usize>,
    index: usize,
}

impl ParameterCollector {
    fn new() -> Self {
        Self {
            parameters: TensorContainer::new(),
            index: 0,
        }
    }
}

impl<B: Backend> ModuleVisitor<B> for ParameterCollector {
    fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<B, D>>) {
        self.parameters
            .register(self.index, parameter.val().detach().into_primitive());
        self.index += 1;
    }
}

struct EmaMapper {
    source: TensorContainer<usize>,
    index: usize,
    rate: f64,
}

impl<B: Backend> ModuleMapper<B> for EmaMapper {
    fn map_float<const D: usize>(&mut self, parameter: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, target, parameter_mapper) = parameter.consume();
        let source = self
            .source
            .remove::<B>(&self.index)
            .map(Tensor::<B, D>::from_primitive)
            .expect("source and target modules have different parameters")
            .to_device(&target.device());
        self.index += 1;

        let value = (target * (1.0 - self.rate) + source * self.rate).detach();
        Param::from_mapped_value(id, value, parameter_mapper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::{Flex, flex::FlexDevice},
        module::Reinitializer,
        nn::{Linear, LinearConfig},
    };

    #[derive(Module, Debug)]
    struct TestModel<B: Backend> {
        first: Linear<B>,
        second: Linear<B>,
    }

    fn model(device: &FlexDevice) -> TestModel<Flex> {
        TestModel {
            first: LinearConfig::new(2, 3).init(device),
            second: LinearConfig::new(3, 1).init(device),
        }
    }

    #[test]
    fn updates_every_float_parameter() {
        let device = FlexDevice;
        let target = Reinitializer::<Flex>::new()
            .constant_float(2.0)
            .apply(model(&device));
        let source = Reinitializer::<Flex>::new()
            .constant_float(6.0)
            .apply(model(&device));

        let updated = update::<Flex, _>(target, &source, 0.25);

        assert!(
            updated
                .first
                .weight
                .val()
                .into_data()
                .iter::<f64>()
                .all(|value| value == 3.0)
        );
        assert!(
            updated
                .second
                .bias
                .as_ref()
                .unwrap()
                .val()
                .into_data()
                .iter::<f64>()
                .all(|value| value == 3.0)
        );
    }
}
