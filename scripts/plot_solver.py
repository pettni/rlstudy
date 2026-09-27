#!/usr/bin/env python3
# /// script
# dependencies = ["matplotlib"]
# ///
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("experiments", nargs="*", type=Path)
args = parser.parse_args()

outputs = []
if args.experiments:
    for path in args.experiments:
        with path.open() as file:
            outputs.append((path.stem, json.load(file)))
else:
    outputs.append(("stdin", json.load(sys.stdin)))

experiments = []
panes = set()
for experiment_name, output in outputs:
    series = defaultdict(list)
    for row in output["history"]:
        for name, value in row["metrics"].items():
            if value is not None:
                series[(row["unit"], name)].append((row["step"], value))
    if not series:
        raise ValueError(f"{experiment_name} contains no history data to plot")
    experiments.append((experiment_name, series))
    panes.update(series)

columns = 2 if len(panes) > 5 else 1
rows = (len(panes) + columns - 1) // columns
figure, axes = plt.subplots(
    rows, columns, squeeze=False, figsize=(9 * columns, 2.5 * rows)
)

if len(experiments) <= 10:
    colors = plt.get_cmap("tab10").colors[: len(experiments)]
else:
    color_map = plt.get_cmap("hsv", len(experiments))
    colors = [color_map(index) for index in range(len(experiments))]

multiple_experiments = len(experiments) > 1
for axis, (unit, name) in zip(axes.flat, sorted(panes)):
    for (experiment_name, series), color in zip(experiments, colors):
        points = series.get((unit, name))
        if not points:
            continue

        steps, values = zip(*points)
        if not name.startswith("eval/"):
            window = 100 if unit == "episode" else 10
            axis.plot(
                steps,
                values,
                alpha=0.25,
                color=color,
                label=None if multiple_experiments else "Raw",
            )
            averages = [
                sum(values[max(0, index - window + 1) : index + 1])
                / min(index + 1, window)
                for index in range(len(values))
            ]
            axis.plot(
                steps,
                averages,
                color=color,
                label=experiment_name
                if multiple_experiments
                else f"Trailing {window} {unit}s",
            )
        else:
            axis.plot(
                steps,
                values,
                color=color,
                marker="o",
                label=experiment_name if multiple_experiments else None,
            )

    if multiple_experiments or not name.startswith("eval/"):
        axis.legend()
    axis.set_ylabel(name)
    axis.set_xlabel(unit.capitalize())

for axis in list(axes.flat)[len(panes) :]:
    axis.set_visible(False)

figure.tight_layout()
figure.savefig(sys.stdout.buffer, format="png", dpi=150)
