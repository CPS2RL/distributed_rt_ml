
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Files
DP_FILE = "dp_results.csv"
SWEEP_FILE = "sweep_k_results.csv"
OUTPUT = "optimality_gap_runs"

# Models and colors
models = {
    "LeNet5": ("LeNet", "#0072B2"),
    "ResNet20": ("ResNet", "#E69F00"),
    "MobileNetV1_0.25": ("MobileNet", "#009E73")
}

styles = {"MNIST": "-", "CIPHAR-10": "--"}

# Read CSV files
with open(DP_FILE) as f:
    dp_data = list(csv.DictReader(f))

with open(SWEEP_FILE) as f:
    sweep_data = list(csv.DictReader(f))

# Full DP baseline for each run
dp = {
    (r["model"], r["mode"], r["run"]): float(r["inference_ms"])
    for r in dp_data
}

# Calculate optimality gaps
gaps = defaultdict(list)

for r in sweep_data:
    key = (r["model"], r["mode"], r["run"])

    if key not in dp:
        raise ValueError(f"Missing DP result: {key}")

    optimum = dp[key]
    estimate = float(r["inference_ms"])

    if optimum <= 0:
        continue

    gap = 100 * (estimate - optimum) / optimum

    if gap < -1e-7:
        raise ValueError(f"Top-K outperformed Full DP: {key}")

    group = (r["model"], r["mode"], int(r["K"]))
    gaps[group].append(max(0, gap))

# Plot
fig, ax = plt.subplots(figsize=(1.9, 2))

for model, (name, color) in models.items():
    for mode in styles:

        ks = sorted(
            k for m, md, k in gaps
            if m == model and md == mode
        )

        if not ks:
            continue

        means = np.array([
            np.mean(gaps[(model, mode, k)]) for k in ks
        ])

        stds = np.array([
            np.std(gaps[(model, mode, k)], ddof=1)
            if len(gaps[(model, mode, k)]) > 1 else 0
            for k in ks
        ])

        ax.plot(
            ks, means,
            color=color,
            linestyle=styles[mode],
            linewidth=1.1,
            label=f"{name} ({mode})"
        )

        ax.fill_between(
            ks,
            np.maximum(0, means - stds),
            means + stds,
            color=color,
            alpha=0.10
        )

# Formatting
ax.axhline(0, color="black", linestyle=":", linewidth=1)
ax.set_xlabel("Value of K", fontsize=9)
ax.set_ylabel("Optimality gap (%)", fontsize=9)
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=8, direction="in")
ax.legend(fontsize=5, loc="upper right")
ax.grid(False)

fig.tight_layout()

# Save
output = Path(OUTPUT)
output.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")

plt.close(fig)
