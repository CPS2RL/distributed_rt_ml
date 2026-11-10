import pandas as pd
import matplotlib.pyplot as plt

# CSV files
files = {
    "DeepThings": "layer_times-deepthings-lenet-ciphar.csv",
    "EdgeFlow": "layer_times-edgeflow-lenet-ciphar.csv",
    "Proposed": "layer_times-proposed-lenet-ciphar.csv"
}


colors = ["#0072B2", "#E69F00", "#009E73"]
layers = ["Conv1", "Pool1", "Conv2", "Pool2", "FC1", "FC2", "FC3"]
x = list(range(1, len(layers) + 1))

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7
})

fig, ax = plt.subplots(figsize=(1.9, 1.9))

for (label, file), color in zip(files.items(), colors):
    df = pd.read_csv(file)

    # cumulative latency for each row
    cumulative = df[layers].cumsum(axis=1)

    # P99 at each cumulative stage
    p99 = cumulative.quantile(0.99)

    ax.plot(
        x,
        p99.values,
        # marker='o',
        # markerfacecolor="none",
        # markersize=3,
        linewidth=1.2,
        color=color,
        label=label
    )

# horizontal line at y = 60
ax.axhline(y=65, linestyle='--', linewidth=1, color='black')

ax.set_xticks(x)
ax.set_xlabel("Layer Number")
ax.set_ylabel("Cumulative P99 (ms)")
ax.legend(frameon=True, loc="upper left")

plt.tight_layout(pad=0.3)

# plt.savefig("LeNet-Ciphar.png", dpi=300, bbox_inches="tight")
plt.savefig("LeNet-Ciphar.pdf", bbox_inches="tight")
plt.show()