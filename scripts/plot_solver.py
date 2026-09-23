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

columns = 2 if len(series) > 5 else 1
rows = (len(series) + columns - 1) // columns
figure, axes = plt.subplots(
    rows, columns, squeeze=False, figsize=(9 * columns, 2.5 * rows)
)
for axis, ((unit, name), points) in zip(axes.flat, sorted(series.items())):
    steps, values = zip(*points)
    if not name.startswith("eval/"):
        window = 100 if unit == "episode" else 10
        axis.plot(steps, values, alpha=0.25, label="Raw")
        averages = [
            sum(values[max(0, index - window + 1) : index + 1])
            / min(index + 1, window)
            for index in range(len(values))
        ]
        axis.plot(steps, averages, label=f"Trailing {window} {unit}s")
        axis.legend()
    else:
        axis.plot(steps, values, marker="o" if name.startswith("eval/") else None)
    axis.set_ylabel(name)
    axis.set_xlabel(unit.capitalize())

for axis in list(axes.flat)[len(series) :]:
    axis.set_visible(False)

figure.tight_layout()
figure.savefig(sys.stdout.buffer, format="png", dpi=150)
