#!/usr/bin/env python3
# /// script
# dependencies = ["matplotlib"]
# ///
import json
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


training = json.load(sys.stdin)["training"]
iterations = [snapshot["iteration"] for snapshot in training]

figure, (loss_axis, reward_axis, learning_rate_axis) = plt.subplots(3, 1, sharex=True)
loss_axis.plot(iterations, [snapshot["average_loss"] for snapshot in training])
loss_axis.set_yscale("log")
loss_axis.set_ylabel("Average loss")

reward_axis.plot(iterations, [snapshot["evaluation_reward"] for snapshot in training])
reward_axis.set_ylabel("Evaluation return")

learning_rate_axis.plot(iterations, [snapshot["learning_rate"] for snapshot in training])
learning_rate_axis.set_xlabel("Training iteration")
learning_rate_axis.set_ylabel("Learning rate")

figure.tight_layout()
figure.savefig(sys.stdout.buffer, format="png", dpi=150)
