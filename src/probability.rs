/// Discrete probability distribution over types
#[derive(Debug, Clone)]
pub struct Distribution<T> {
    pt: Vec<(f64, T)>,
}

impl<T> Distribution<T> {
    /// Point mass distribution
    pub fn point_mass(t: T) -> Self {
        Self { pt: vec![(1., t)] }
    }

    /// Sample the distribution.
    pub fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> &T {
        use rand::RngExt;

        let u: f64 = rng.random();

        let mut acc = 0.0;
        for (p, t) in &self.pt {
            acc += *p;
            if u < acc {
                return t;
            }
        }
        &self.pt.last().expect("distribution is non-empty").1
    }

    /// Iterate over `(probability, value)` pairs.
    pub fn iter(&self) -> std::slice::Iter<'_, (f64, T)> {
        self.pt.iter()
    }

    /// Number of outcomes in the distribution.
    pub fn len(&self) -> usize {
        self.pt.len()
    }

    /// True if the distribution contains no outcomes.
    pub fn is_empty(&self) -> bool {
        self.pt.is_empty()
    }
}

impl<T: Ord> Distribution<T> {
    /// Constructor from paired data
    pub fn new(data: Vec<(f64, T)>) -> Result<Self, String> {
        let mut data: Vec<_> = data
            .into_iter()
            .filter(|(probability, _)| *probability != 0.0)
            .collect();

        if data.is_empty() || data.iter().any(|(p, _)| *p < 0. || !p.is_finite()) {
            return Err("at least one probability must be non-zero".into());
        }

        data.sort_by(|(_, left), (_, right)| left.cmp(right));

        if data.windows(2).any(|pair| pair[0].1 == pair[1].1) {
            return Err("distribution outcomes must be unique".into());
        }

        let ps_sum: f64 = data.iter().map(|(p, _)| p).sum();
        let pt = data.into_iter().map(|(p, t)| (p / ps_sum, t)).collect();

        Ok(Self { pt })
    }

    /// Uniform distribution over values
    pub fn uniform(ts: Vec<T>) -> Result<Self, String> {
        let p = 1.0 / ts.len() as f64;
        Self::new(ts.into_iter().map(|t| (p, t)).collect())
    }

    pub fn probability_of(&self, value: &T) -> f64 {
        self.pt
            .binary_search_by(|(_, outcome)| outcome.cmp(value))
            .map(|index| self.pt[index].0)
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn test_basic() {
        let data = vec![(0.25, 'a'), (0.25, 'b'), (0.5, 'c')];
        let dist = Distribution::<char>::new(data).unwrap();
        let mut rng = StdRng::seed_from_u64(0);

        for _ in 0..20 {
            let s = dist.sample(&mut rng);
            assert!(*s == 'a' || *s == 'b' || *s == 'c')
        }
    }

    #[test]
    fn probability_lookup_uses_values() {
        let dist = Distribution::new(vec![(1.0, 'b'), (3.0, 'a')]).unwrap();

        assert_eq!(dist.probability_of(&'a'), 0.75);
        assert_eq!(dist.probability_of(&'b'), 0.25);
        assert_eq!(dist.probability_of(&'c'), 0.0);
        assert_eq!(
            dist.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
            vec!['a', 'b']
        );
    }

    #[test]
    fn sampling_is_reproducible_from_a_seed() {
        let dist = Distribution::new(vec![(1.0, 'a'), (1.0, 'b'), (1.0, 'c')]).unwrap();
        let mut first_rng = StdRng::seed_from_u64(42);
        let mut second_rng = StdRng::seed_from_u64(42);

        let first: Vec<_> = (0..20).map(|_| *dist.sample(&mut first_rng)).collect();
        let second: Vec<_> = (0..20).map(|_| *dist.sample(&mut second_rng)).collect();

        assert_eq!(first, second);
    }

    #[test]
    fn invalid_distributions_return_errors() {
        assert!(Distribution::<usize>::new(vec![]).is_err());
        assert!(Distribution::new(vec![(-1.0, 0)]).is_err());
        assert!(Distribution::new(vec![(f64::NAN, 0)]).is_err());
        assert!(Distribution::new(vec![(f64::INFINITY, 0)]).is_err());
        assert!(Distribution::new(vec![(0.0, 0)]).is_err());
        assert!(Distribution::new(vec![(1.0, 0), (1.0, 0)]).is_err());
        assert!(Distribution::<usize>::uniform(vec![]).is_err());
    }

    #[test]
    fn zero_probability_outcomes_are_removed() {
        let dist = Distribution::new(vec![(0.0, 'a'), (1.0, 'a'), (0.0, 'b')]).unwrap();

        assert_eq!(dist.len(), 1);
        assert_eq!(dist.probability_of(&'a'), 1.0);
        assert_eq!(dist.probability_of(&'b'), 0.0);
    }
}
