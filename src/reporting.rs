use crate::environment::Environment;
use crate::evaluation::{EvaluationConfig, EvaluationStats, evaluate_rollouts};
use crate::policy::DiscretePolicy;
use clap::Args;
use rand::{SeedableRng, rngs::StdRng};
use serde::Serialize;
use std::{collections::BTreeMap, io::IsTerminal, str::FromStr};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Episode(usize),
    Iteration(usize),
}

impl Step {
    fn parts(self) -> (&'static str, usize) {
        match self {
            Self::Episode(number) => ("episode", number),
            Self::Iteration(number) => ("iteration", number),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Frequency {
    PerEpisode(usize),
    PerIteration(usize),
}

impl Serialize for Frequency {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl Frequency {
    pub fn matches(self, step: Step) -> bool {
        match (self, step) {
            (Self::PerEpisode(interval), Step::Episode(number))
            | (Self::PerIteration(interval), Step::Iteration(number)) => number % interval == 0,
            _ => false,
        }
    }
}

impl FromStr for Frequency {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (unit, count) = value
            .split_once(':')
            .ok_or("expected episode:N or iteration:N")?;
        let count = count
            .parse::<usize>()
            .map_err(|_| "expected a positive interval")?;
        if count == 0 {
            return Err("interval must be positive".into());
        }
        match unit {
            "episode" => Ok(Self::PerEpisode(count)),
            "iteration" => Ok(Self::PerIteration(count)),
            _ => Err("expected episode:N or iteration:N".into()),
        }
    }
}

impl std::fmt::Display for Frequency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PerEpisode(count) => write!(f, "episode:{count}"),
            Self::PerIteration(count) => write!(f, "iteration:{count}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Args, Serialize)]
pub struct ReporterConfig {
    #[arg(long)]
    pub evaluation_frequency: Option<Frequency>,
    #[arg(long, default_value_t = 10)]
    pub evaluation_episodes: usize,
    #[arg(long, default_value_t = 1_000)]
    pub evaluation_max_steps: usize,
}

impl Default for ReporterConfig {
    fn default() -> Self {
        Self {
            evaluation_frequency: None,
            evaluation_episodes: 10,
            evaluation_max_steps: 1_000,
        }
    }
}

impl ReporterConfig {
    pub fn validate(&self) -> Result<(), String> {
        EvaluationConfig::new(self.evaluation_episodes, self.evaluation_max_steps)?;
        if matches!(
            self.evaluation_frequency,
            Some(Frequency::PerEpisode(0) | Frequency::PerIteration(0))
        ) {
            return Err("evaluation interval must be positive".into());
        }
        Ok(())
    }

    fn evaluation(&self) -> EvaluationConfig {
        EvaluationConfig::new(self.evaluation_episodes, self.evaluation_max_steps)
            .expect("reporter configuration was validated")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HistoryRow {
    pub unit: &'static str,
    pub step: usize,
    pub metrics: BTreeMap<&'static str, f64>,
}

pub trait AbstractReporter<Observation> {
    fn record(&mut self, step: Step, metrics: &[(&'static str, f64)]);

    fn policy(
        &mut self,
        step: Step,
        make_policy: &mut dyn FnMut() -> Box<dyn DiscretePolicy<Observation = Observation>>,
    );
}

pub struct Reporter<'a, Observation> {
    config: ReporterConfig,
    evaluation_environment:
        Option<&'a mut dyn Environment<Observation = Observation, Action = usize>>,
    evaluation_rng: StdRng,
    final_evaluation_rng: StdRng,
    history: Vec<HistoryRow>,
}

impl<'a, Observation> Reporter<'a, Observation> {
    pub fn new(
        config: ReporterConfig,
        seed: u64,
        evaluation_environment: Option<
            &'a mut dyn Environment<Observation = Observation, Action = usize>,
        >,
    ) -> Self {
        Self {
            config,
            evaluation_environment,
            evaluation_rng: StdRng::seed_from_u64(seed),
            final_evaluation_rng: StdRng::seed_from_u64(seed),
            history: Vec::new(),
        }
    }

    pub fn final_evaluation(
        &mut self,
        policy: &dyn DiscretePolicy<Observation = Observation>,
    ) -> EvaluationStats {
        let environment = self
            .evaluation_environment
            .as_deref_mut()
            .expect("evaluation environment is available");
        evaluate_rollouts(
            environment,
            policy,
            self.config.evaluation(),
            &mut self.final_evaluation_rng,
        )
    }

    pub fn into_history(self) -> Vec<HistoryRow> {
        self.history
    }
}

impl<Observation> AbstractReporter<Observation> for Reporter<'_, Observation> {
    fn record(&mut self, step: Step, metrics: &[(&'static str, f64)]) {
        let (unit, number) = step.parts();
        self.history.push(HistoryRow {
            unit,
            step: number,
            metrics: metrics.iter().copied().collect(),
        });
        if unit == "episode" && std::io::stderr().is_terminal() {
            let episode_return = metrics.iter().find(|(name, _)| *name == "train/return");
            if let Some((_, value)) = episode_return {
                eprint!("\rEpisode {number} return {value:.2}");
            } else {
                eprint!("\rEpisode {number}");
            }
        }
    }

    fn policy(
        &mut self,
        step: Step,
        make_policy: &mut dyn FnMut() -> Box<dyn DiscretePolicy<Observation = Observation>>,
    ) {
        if !self
            .config
            .evaluation_frequency
            .is_some_and(|frequency| frequency.matches(step))
        {
            return;
        }
        let Some(environment) = self.evaluation_environment.as_deref_mut() else {
            return;
        };
        let policy = make_policy();
        let stats = evaluate_rollouts(
            environment,
            policy.as_ref(),
            self.config.evaluation(),
            &mut self.evaluation_rng,
        );
        let (unit, number) = step.parts();
        let metrics = if let Some(row) = self
            .history
            .iter_mut()
            .rev()
            .find(|row| row.unit == unit && row.step == number)
        {
            &mut row.metrics
        } else {
            self.history.push(HistoryRow {
                unit,
                step: number,
                metrics: BTreeMap::new(),
            });
            &mut self.history.last_mut().expect("row was pushed").metrics
        };
        metrics.insert("eval/return_mean", stats.return_mean);
        metrics.insert("eval/return_std", stats.return_std);
        metrics.insert("eval/episode_length_mean", stats.episode_length_mean);
        metrics.insert("eval/max_steps_rate", stats.max_steps_rate);
    }
}

pub(crate) fn print_evaluation_progress(label: &str, episode: usize, episodes: usize) {
    if !std::io::stderr().is_terminal() {
        return;
    }
    if episode == 0 {
        eprintln!();
    }
    const WIDTH: usize = 30;
    let filled = WIDTH * episode / episodes;
    let bar = format!("{}{}", "=".repeat(filled), " ".repeat(WIDTH - filled));
    eprint!("\r{label} [{bar}] {episode}/{episodes}");
    if episode == episodes {
        eprintln!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::BlackBox;
    use crate::policy::TabularPolicy;

    #[test]
    fn frequency_matches_its_step_unit() {
        assert!(Frequency::PerEpisode(2).matches(Step::Episode(4)));
        assert!(!Frequency::PerEpisode(2).matches(Step::Iteration(4)));
        assert!(!Frequency::PerIteration(3).matches(Step::Iteration(4)));
        assert!("episode:0".parse::<Frequency>().is_err());
    }

    #[test]
    fn policy_factory_is_lazy_and_records_every_scalar_row() {
        let mut environment = BlackBox::new(1, 1, 0, |state, _| (state, 1.0));
        let mut reporter = Reporter::new(
            ReporterConfig {
                evaluation_frequency: Some(Frequency::PerEpisode(2)),
                evaluation_episodes: 1,
                evaluation_max_steps: 1,
            },
            0,
            Some(&mut environment),
        );
        let mut calls = 0;
        for episode in 1..=2 {
            reporter.record(Step::Episode(episode), &[("train/return", episode as f64)]);
            reporter.policy(Step::Episode(episode), &mut || {
                calls += 1;
                Box::new(TabularPolicy::uniform(1, 1).unwrap())
            });
        }
        let history = reporter.into_history();
        assert_eq!(calls, 1);
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].metrics["train/return"], 1.0);
        assert_eq!(history[1].metrics["eval/return_mean"], 1.0);
    }
}
