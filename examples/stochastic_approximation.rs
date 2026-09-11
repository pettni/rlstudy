/// Stochastic approximation example
///
/// One-liner with plotting
///   cargo run --quiet --example stochastic_approximation | uv run scripts/plot_stochastic_approximation.py  | icat
///
/// Theory
/// ======
///
///  y_t = μ_t + ε_t   ε_t \sim N(0, σ^2)
///
/// Want to estimate μ_t with a streaming estimator.
///
/// Batch estimate is the mean:  \hat μ = (1/T) ∑_t y_t  (but does not capture variation)
///
/// Streaming mean estimate is  \hat μ_{t+1} = \hat μ_t + (1/t+1) * (y_{t+1} - \hat μ_t) but we
/// can show that
///   \hat μ_{t+1} = \hat μ_t + α_t * (y_{t+1} - \hat μ_t)
/// converges to stationary targets for any {α_t} with certain properties.
///
/// For α_t = (1 / t)^β, the valid range is β \in (0.5, 1] so that
///
///   ∑_t α_t    = ∞
///   ∑_t \α_t^2 < ∞
///
/// Relation to gradient descent
/// ============================
///
/// Consider min_theta L(theta)   where L(theta) = E_Y J(theta, Y)
///
/// Gradient descent is theta_{k+1} = theta_k - alpha_k \grad L(theta_k)
///
///   \grad L(theta_k) = \grad E_Y J(Y, theta_k) = E_Y \grad J(Y, theta_k)
///
/// We can estimate \grad J(Y, theta) from many samples of Y (ideally the full batch), but
/// the _stochastic_ approximation is to estimate it from one sample y_k. That gives
///
///   \grad L(\theta_k) \approx \grad_{\theta_k} J(y_k, \theta_k)
///
/// For this example J(theta, Y) = (1/2) (\theta - Y) ^ 2, meaning that \grad_{\theta_k} J(y_k, \theta_k) = \theta_k - y_k,
/// so we recover
///
///   theta_{k+1} = \theta_k - \alpha_k  (\theta_k - y_k)
///
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

fn estimator(ys: &[f64], alpha: impl Fn(usize) -> f64) -> Vec<f64> {
    let hat_mu = ys.iter().enumerate().scan(0., |mu, (t, y)| {
        *mu = *mu + alpha(t) * (y - *mu);
        Some(*mu)
    });

    hat_mu.collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let normal = Normal::new(0., 1.).unwrap();
    let mut rng = StdRng::seed_from_u64(0);

    let tt = 20000;
    let mus: Vec<f64> = (0..tt).map(|t| if t < tt / 4 { 5. } else { -2. }).collect();
    let ys = mus
        .iter()
        .map(|mu| {
            let eps = normal.sample(&mut rng);
            mu + eps
        })
        .collect::<Vec<_>>();

    // regular mean estimator
    let hat_mus_1 = estimator(&ys, |t| 1. / (1. + t as f64));
    // larger step size [infinite sum, finite squared sum]
    let hat_mus_2 = estimator(&ys, |t| 1. / (1. + (t as f64).powf(0.75)));
    // smaller step size [finite sum, may not converge]
    let hat_mus_3 = estimator(&ys, |t| 1. / (1. + (t as f64).powf(1.25)));
    // constant step size
    let hat_mus_4 = estimator(&ys, |_| 0.05);

    println!("t,mu,hat_mu_1,hat_mu_2,hat_mu_3,hat_mu_4");
    for t in 0..tt {
        println!(
            "{},{},{},{},{},{}",
            t, mus[t], hat_mus_1[t], hat_mus_2[t], hat_mus_3[t], hat_mus_4[t]
        );
    }

    Ok(())
}
