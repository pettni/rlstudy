use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Schedule {
    Constant {
        value: f64,
    },
    Linear {
        initial: f64,
        final_value: f64,
        steps: usize,
    },
    Exponential {
        initial: f64,
        final_value: f64,
        decay: usize,
    },
}

impl Default for Schedule {
    fn default() -> Self {
        Self::Exponential {
            initial: 1.0,
            final_value: 0.05,
            decay: 20_000,
        }
    }
}

impl Schedule {
    pub(crate) fn value(self, step: usize) -> f64 {
        match self {
            Self::Constant { value } => value,
            Self::Linear {
                initial,
                final_value,
                steps,
            } => {
                let progress = (step as f64 / steps as f64).min(1.0);
                initial + progress * (final_value - initial)
            }
            Self::Exponential {
                initial,
                final_value,
                decay,
            } => final_value + (initial - final_value) * (-(step as f64) / decay as f64).exp(),
        }
    }
}

// Clap's ValueEnum only supports fieldless variants, so parse this data-carrying enum directly.
impl FromStr for Schedule {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let fields = value.split(':').collect::<Vec<_>>();
        let number = |value: &str| {
            value
                .parse::<f64>()
                .map_err(|_| format!("invalid schedule value: {value}"))
        };
        let steps = |value: &str| {
            value
                .parse::<usize>()
                .map_err(|_| format!("invalid schedule step count: {value}"))
        };

        match fields.as_slice() {
            ["constant", value] => Ok(Self::Constant {
                value: number(value)?,
            }),
            ["linear", initial, final_value, count] => Ok(Self::Linear {
                initial: number(initial)?,
                final_value: number(final_value)?,
                steps: steps(count)?,
            }),
            ["exponential", initial, final_value, decay] => Ok(Self::Exponential {
                initial: number(initial)?,
                final_value: number(final_value)?,
                decay: steps(decay)?,
            }),
            _ => Err(
                "expected constant:<value>, linear:<initial>:<final>:<steps>, or \
                 exponential:<initial>:<final>:<decay>"
                    .into(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linearly_interpolates_and_stops_at_final_value() {
        let schedule = Schedule::Linear {
            initial: 1.0,
            final_value: 0.0,
            steps: 10,
        };

        assert_eq!(schedule.value(0), 1.0);
        assert_eq!(schedule.value(5), 0.5);
        assert_eq!(schedule.value(10), 0.0);
        assert_eq!(schedule.value(20), 0.0);
    }

    #[test]
    fn exponentially_decays_towards_final_value() {
        let schedule = Schedule::Exponential {
            initial: 1.0,
            final_value: 0.0,
            decay: 10,
        };

        assert_eq!(schedule.value(0), 1.0);
        assert!((schedule.value(10) - (-1.0_f64).exp()).abs() < 1e-12);
        assert!(schedule.value(100) < 0.0001);
    }
}
