
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

CSV_FILE = "workloads.csv"

strategies = ["DeepThings", "EdgeFlow", "Proposed"]
filenames = ["deepthings_workload", "edgeflow_workload", "ours_workload"]

with open(CSV_FILE, "r") as f:
    data = list(csv.DictReader(f))

layers = list(dict.fromkeys(row["Layer"] for row in data))
devices = sorted(set(row["Device"] for row in data),
                 key=lambda x: int(x[1:]))

workloads = {
    s: np.zeros((len(devices), len(layers)))
    for s in strategies
}

for row in data:
    s = row["Strategy"]
    if s in workloads:
        d = devices.index(row["Device"])
        l = layers.index(row["Layer"])
        workloads[s][d, l] = float(row["Operations"])

max_workload = np.maximum.reduce([
    m.max(axis=0) for m in workloads.values()
])
max_workload = np.maximum(max_workload, 1)

for i, strategy in enumerate(strategies):

    matrix = workloads[strategy] / max_workload

    fig = plt.figure(figsize=(1, 2.2))
    ax = fig.add_axes([0.22, 0.22, 0.58, 0.63])

    im = ax.imshow(matrix, cmap="coolwarm",
                   aspect="auto", vmin=0, vmax=1)

    ticks = [1, 15, 30]
    positions = [(t - 1) * (len(layers) - 1) / 29 for t in ticks]

    ax.set_xticks(positions, ticks, fontsize=6)
    ax.set_xlabel("Layer number", fontsize=6)

    ax.set_yticks(range(len(devices)), devices, fontsize=6)
    ax.set_ylabel("Devices", fontsize=6)

    ax.tick_params(length=0, pad=2)

    if strategy == "Proposed":
        cax = fig.add_axes([0.83, 0.22, 0.035, 0.63])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_ticks([0, 0.5, 1])
        cbar.ax.tick_params(labelsize=6, length=2)
        cbar.set_label("Normalized workload", fontsize=6)

    path = os.path.join(filenames[i] + ".pdf")
    fig.savefig(path)
    plt.close(fig)
