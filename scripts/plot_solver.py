#!/usr/bin/env python3
# /// script
# dependencies = ["matplotlib"]
# ///
import json
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

history = json.load(sys.stdin)["history"]
series = defaultdict(list)
for row in history:
    for name, value in row["metrics"].items():
        series[(row["unit"], name)].append((row["step"], value))

if not series:
    raise ValueError("solver output contains no history data to plot")

figure, axes = plt.subplots(
    len(series), 1, squeeze=False, figsize=(9, 2.5 * len(series))
)
for axis, ((unit, name), points) in zip(axes[:, 0], sorted(series.items())):
    steps, values = zip(*points)
    if unit == "episode" and not name.startswith("eval/"):
        axis.plot(steps, values, alpha=0.25, label="Raw")
        averages = [
            sum(values[max(0, index - 99) : index + 1])
            / min(index + 1, 100)
            for index in range(len(values))
        ]
        axis.plot(steps, averages, label="Trailing 100 episodes")
        axis.legend()
    else:
        axis.plot(steps, values, marker="o" if name.startswith("eval/") else None)
    axis.set_ylabel(name)
    axis.set_xlabel(unit.capitalize())

figure.tight_layout()
figure.savefig(sys.stdout.buffer, format="png", dpi=150)
