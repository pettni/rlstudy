#!/usr/bin/env python3
# /// script
# dependencies = ["matplotlib", "pyyaml"]
# ///
import argparse
import re
import sys

import matplotlib
import yaml

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches


parser = argparse.ArgumentParser()
parser.add_argument("grid_world")
args = parser.parse_args()

with open(args.grid_world) as file:
    grid_world = yaml.safe_load(file)

side = grid_world["side"]
forbidden_states = {tuple(state) for state in grid_world["forbidden_states"]}
goal_states = {tuple(state) for state in grid_world["goal_states"]}

solution_pattern = re.compile(
    r"^\s*state (?P<state>\d+) \[value (?P<value>[-+.\deE]+)\] -> action (?P<action>\d+)$"
)
solution = {}
for line in sys.stdin:
    if match := solution_pattern.match(line):
        state = int(match["state"])
        solution[state] = (float(match["value"]), int(match["action"]))

num_states = side * side
if set(solution) != set(range(num_states)):
    missing = sorted(set(range(num_states)) - set(solution))
    raise ValueError(f"solver output is missing states: {missing}")

figure, axis = plt.subplots(figsize=(side, side))
axis.set_xlim(-0.5, side - 0.5)
axis.set_ylim(side - 0.5, -0.5)
axis.set_aspect("equal")
axis.set_xticks(range(side), labels=range(side))
axis.set_yticks(range(side), labels=range(side))
axis.xaxis.tick_top()
axis.tick_params(length=0)

for row in range(side):
    for column in range(side):
        coordinates = (row, column)
        if coordinates in forbidden_states:
            color = "moccasin"
        elif coordinates in goal_states:
            color = "lightskyblue"
        else:
            color = "white"
        axis.add_patch(
            patches.Rectangle(
                (column - 0.5, row - 0.5),
                1,
                1,
                facecolor=color,
                edgecolor="gray",
                linewidth=1,
            )
        )

actions = {
    0: (0, -1),
    1: (1, 0),
    2: (0, 1),
    3: (-1, 0),
    4: (0, 0),
}

for state, (value, action) in solution.items():
    row, column = divmod(state, side)
    axis.text(
        column - 0.42,
        row - 0.38,
        f"{value:.2f}",
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=9,
    )

    dx, dy = actions[action]
    if (dx, dy) == (0, 0):
        axis.add_patch(
            patches.Circle(
                (column, row),
                radius=0.09,
                fill=False,
                edgecolor="forestgreen",
                linewidth=2,
            )
        )
    else:
        axis.add_patch(
            patches.FancyArrowPatch(
                (column - 0.28 * dx, row - 0.28 * dy),
                (column + 0.28 * dx, row + 0.28 * dy),
                arrowstyle="-|>",
                color="forestgreen",
                linewidth=2,
                mutation_scale=12,
            )
        )

initial_distribution = grid_world["initial_distribution"]
if initial_distribution["type"] == "fixed":
    row, column = initial_distribution["state"]
    axis.plot(
        column - 0.32,
        row + 0.32,
        marker="*",
        color="royalblue",
        markersize=10,
    )

axis.set_xlabel("column")
axis.xaxis.set_label_position("top")
axis.set_ylabel("row")
figure.tight_layout()
figure.savefig(sys.stdout.buffer, format="png", dpi=150)
