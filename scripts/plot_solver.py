#!/usr/bin/env python3
# /// script
# dependencies = ["matplotlib"]
# ///
import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--state", type=int, default=0)
args = parser.parse_args()

snapshots = json.load(sys.stdin)["snapshots"]
plots = []

value_snapshots = [
    snapshot
    for snapshot in snapshots
    if "state_values" in snapshot and args.state < len(snapshot["state_values"])
]
if value_snapshots:
    plots.append(
        (
            "State value",
            [snapshot["iteration"] for snapshot in value_snapshots],
            [snapshot["state_values"][args.state] for snapshot in value_snapshots],
            False,
        )
    )

for field, label, logarithmic in (
    ("state_value", "Initial state value", False),
    ("average_loss", "Average loss", True),
    ("evaluation_return", "Evaluation return", False),
    ("learning_rate", "Learning rate", False),
):
    available = [snapshot for snapshot in snapshots if field in snapshot]
    if available:
        plots.append(
            (
                label,
                [snapshot["iteration"] for snapshot in available],
                [snapshot[field] for snapshot in available],
                logarithmic,
            )
        )

if not plots:
    raise ValueError("solver output contains no snapshot data to plot")

figure, axes = plt.subplots(len(plots), 1, sharex=True, squeeze=False)
for axis, (label, iterations, values, logarithmic) in zip(axes[:, 0], plots):
    axis.plot(iterations, values)
    if logarithmic:
        axis.set_yscale("log")
    axis.set_ylabel(label)

axes[-1, 0].set_xlabel("Iteration")
figure.tight_layout()
figure.savefig(sys.stdout.buffer, format="png", dpi=150)
