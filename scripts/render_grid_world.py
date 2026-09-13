#!/usr/bin/env python3
# /// script
# dependencies = ["matplotlib", "pyyaml"]
# ///
import argparse
import json
import sys

import matplotlib
import yaml

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches


parser = argparse.ArgumentParser()
parser.parse_args()

solution = json.load(sys.stdin)
with open(solution["grid_world_definition"]) as file:
    grid_world = yaml.safe_load(file)

side = grid_world["side"]
forbidden_states = {tuple(state) for state in grid_world["forbidden_states"]}
goal_states = {tuple(state) for state in grid_world["goal_states"]}

num_states = side * side
policy_actions = solution["policy_actions"]
state_values = solution.get("state_values")
if len(policy_actions) != num_states:
    raise ValueError(f"expected {num_states} policy actions, got {len(policy_actions)}")
if state_values is not None and len(state_values) != num_states:
    raise ValueError(f"expected {num_states} state values, got {len(state_values)}")

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

for state, action in enumerate(policy_actions):
    row, column = divmod(state, side)
    if state_values is not None:
        axis.text(
            column - 0.42,
            row - 0.38,
            f"{state_values[state]:.2f}",
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
