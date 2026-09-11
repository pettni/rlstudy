#!/usr/bin/env python3
# /// script
# dependencies = ["matplotlib"]
# ///
import csv
import sys

import matplotlib.pyplot as plt


rows = list(csv.DictReader(sys.stdin))
ts = [int(row["t"]) for row in rows]

for column in ("hat_mu_1", "hat_mu_2", "hat_mu_3", "hat_mu_4", "mu"):
    plt.plot(ts, [float(row[column]) for row in rows], label=column)

plt.xlabel("t")
plt.ylabel("mu")
plt.legend()
plt.tight_layout()
plt.savefig(sys.stdout.buffer, format="png")
