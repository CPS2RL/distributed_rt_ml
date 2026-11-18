import pandas as pd
import matplotlib.pyplot as plt

# CSV files
files = {
    "DeepThings": "layer_times-deepthings-resnet-mnist.csv",
    "EdgeFlow": "layer_times-edgeflow-resnet-mnist.csv",
    "Proposed": "layer_times-proposed-resnet-mnist.csv"
}


colors = ["#0072B2", "#E69F00", "#009E73"]
layers = [
    "Stem", "B1C1", "B1C2", "B1Add", "B2C1", "B2C2", "B2Add",
    "B3C1", "B3C2", "B3Add", "B4C1", "B4C2", "B4Add", "B5C1", "B5C2", "B5Add",
    "B6C1", "B6C2", "B6Add","B7C1", "B7C2", "B7Add","B8C1", "B8C2", "B8Add",
    "B9C1", "B9C2", "B9Add","GAP", "FC"
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

ax.axhline(y=5000, linestyle='--', linewidth=1, color='black')

ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.set_xlim(1, 30)
ax.set_xlabel("Layer Number")
ax.set_ylabel("Cumulative P99 (ms)")
ax.legend(frameon=True, loc="upper left")

plt.tight_layout(pad=0.3)

plt.savefig("ResNet-mnist.png", dpi=300, bbox_inches="tight")
plt.savefig("ResNet-mnist.pdf", bbox_inches="tight")
plt.show()