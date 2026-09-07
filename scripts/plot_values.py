#!/usr/bin/env python3
# /// script
# dependencies = ["matplotlib"]
# ///
import argparse
import csv

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("csv", nargs="+")
parser.add_argument("--state", type=int, default=0)
parser.add_argument("--output")
args = parser.parse_args()

for path in args.csv:
    with open(path, newline="") as file:
        rows = [row for row in csv.DictReader(file) if int(row["state"]) == args.state]
    if rows:
        plt.plot(
            [int(row["evaluation_sweep"]) for row in rows],
            [float(row["value"]) for row in rows],
            label=rows[0]["solver"],
        )

plt.xlabel("Evaluation sweeps")
plt.ylabel(f"V(state {args.state})")
plt.legend()
plt.tight_layout()
if args.output:
    plt.savefig(args.output)
else:
    plt.show()
