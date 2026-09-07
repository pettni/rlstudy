use std::ops::Index;

/// Discrete probability distribution over types
#[derive(Debug, Clone)]
pub struct Distribution<T> {
    pt: Vec<(f32, T)>,
}

impl<T> Distribution<T> {
    /// Constructor from paired data
    pub fn new(data: Vec<(f32, T)>) -> Self {
        let (ps_unnormalized, ts): (Vec<_>, Vec<_>) = data.into_iter().unzip();

        assert!(
            ps_unnormalized.iter().all(|&p| p >= 0.0),
            "probabilities must be non-negative"
        );

        let ps_sum: f32 = ps_unnormalized.iter().sum();
        assert!(ps_sum > 0.0, "at least one probability must be non-zero");

        let ps = ps_unnormalized.into_iter().map(|p| p / ps_sum);
        let pt = ps.zip(ts).collect();

        Self { pt }
    }

    /// Point mass distribution
    pub fn point_mass(t: T) -> Self {
        Self { pt: vec![(1., t)] }
    }

    /// Uniform distribution over values
    pub fn uniform(ts: Vec<T>) -> Self {
        let p = 1. / (ts.len() as f32);
        Self::new(ts.into_iter().map(|t| (p, t)).collect::<Vec<_>>())
    }

    /// @brief Sample the distribution
    pub fn sample(&self) -> &T {
        let u: f32 = rand::random();

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
    pub fn iter(&self) -> std::slice::Iter<'_, (f32, T)> {
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

impl<T> Index<usize> for Distribution<T> {
    type Output = (f32, T);

    fn index(&self, index: usize) -> &Self::Output {
        &self.pt[index]
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_basic() {
        let data = vec![(0.25, 'a'), (0.25, 'b'), (0.5, 'c')];
        let dist = Distribution::<char>::new(data);

        for _ in 0..20 {
            let s = dist.sample();
            assert!(*s == 'a' || *s == 'b' || *s == 'c')
        }
    }

    #[test]
    fn index_returns_probability_and_value() {
        let data = vec![(0.25, 'a'), (0.25, 'b'), (0.5, 'c')];
        let dist = Distribution::<char>::new(data);

        assert_eq!(dist[0].0, 0.25);
        assert_eq!(dist[0].1, 'a');
        assert_eq!(dist[1].0, 0.25);
        assert_eq!(dist[1].1, 'b');
        assert_eq!(dist[2].0, 0.50);
        assert_eq!(dist[2].1, 'c');
    }
}
