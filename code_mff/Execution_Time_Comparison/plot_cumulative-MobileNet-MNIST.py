import pandas as pd
import matplotlib.pyplot as plt

# CSV files
files = {
    "DeepThings": "layer_times-deepthings-mobilenet-mnist.csv",
    "EdgeFlow": "layer_times-edgeflow-mobilenet-mnist.csv",
    "Proposed": "layer_times-proposed-mobilenet-mnist.csv"
}


colors = ["#0072B2", "#E69F00", "#009E73"]
layers = [
    "Stem",
    "D1", "P1", "D2", "P2", "D3", "P3", "D4", "P4",
    "D5", "P5", "D6", "P6", "D7", "P7", "D8", "P8",
    "D9", "P9", "D10", "P10", "D11", "P11", "D12", "P12",
    "D13", "P13", "GAP", "FC"
]
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
ax.axhline(y=250, linestyle='--', linewidth=1, color='black')

ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.set_xlabel("Layer Number")
ax.set_ylabel("Cumulative P99 (ms)")
ax.legend(frameon=True, loc="lower right")

plt.tight_layout(pad=0.3)

# plt.savefig("mobilenet-mnist.png", dpi=300, bbox_inches="tight")
plt.savefig("mobilenet-mnist.pdf", bbox_inches="tight")
plt.show()