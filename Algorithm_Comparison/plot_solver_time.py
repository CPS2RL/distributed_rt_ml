
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Files
SWEEP_FILE = "sweep_k_results.csv"
OUTPUT = "solver_time_runs"

# Models and colors
models = {
    "LeNet5": ("LeNet-5", "#0072B2"),
    "ResNet20": ("ResNet-20", "#E69F00"),
    "MobileNetV1_0.25": ("MobileNetV1-0.25", "#009E73")
}

styles = {"MNIST": "-", "CIPHAR-10": "--"}

# Read CSV
with open(SWEEP_FILE, newline="") as f:
    data = list(csv.DictReader(f))

# Group solve times by model, mode, and K
times = defaultdict(list)

for row in data:
    key = (row["model"], row["mode"], int(row["K"]))
    times[key].append(float(row["solve_time_ms"]))

# Plot
fig, ax = plt.subplots(figsize=(1.9, 2))

positive_values = []
plotted = 0

for model, (name, color) in models.items():
    for mode in styles:

        ks = sorted(
            k for m, md, k in times
            if m == model and md == mode
        )

        if not ks:
            continue

        means = np.array([
            np.mean(times[(model, mode, k)])
            for k in ks
        ])

        stds = np.array([
            np.std(times[(model, mode, k)], ddof=1)
            if len(times[(model, mode, k)]) > 1 else 0
            for k in ks
        ])

        # Mean solve time
        ax.plot(
            ks, means,
            color=color,
            linestyle=styles[mode],
            linewidth=1.1,
            label=f"{name} ({mode})"
        )

        # Standard deviation shading
        ax.fill_between(
            ks,
            np.maximum(means - stds, 1e-12),
            means + stds,
            color=color,
            alpha=0.1
        )

        positive_values.extend(means[means > 0])
        plotted += 1

if plotted == 0:
    raise ValueError("No Top-K solve time data found")

# Use log scale for large runtime differences
if positive_values and max(positive_values) / min(positive_values) >= 30:
    ax.set_yscale("log")

# Formatting
ax.set_xlabel("Value of K", fontsize=9)
ax.set_ylabel("Solve time (ms)", fontsize=9)
ax.tick_params(labelsize=8, direction="in")
ax.grid(False)

# Uncomment to show legend
# ax.legend(fontsize=5, loc="upper right")

fig.tight_layout()

# Save PDF and PNG
output = Path(OUTPUT)
output.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")

plt.close(fig)

