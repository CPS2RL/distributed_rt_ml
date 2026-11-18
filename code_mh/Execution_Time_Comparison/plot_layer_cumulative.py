"""Fig. 2: per-layer P99 execution time (bars, left axis) and cumulative P99
execution time (lines, right axis) with the task deadline, for six workloads.

Reads the measured CSVs from code_mff/Execution_Time_Comparison (unchanged)
and writes one PDF per workload into ./figures (or --out).

Usage:
    python plot_layer_cumulative.py [--out DIR] [--only LeNet-MNIST ...]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "code_mff" / "Execution_Time_Comparison"

METHODS = ["DeepThings", "EdgeFlow", "Proposed"]
COLORS = {"DeepThings": "#0072B2", "EdgeFlow": "#E69F00", "Proposed": "#009E73"}

LENET = ["Conv1", "Pool1", "Conv2", "Pool2", "FC1", "FC2", "FC3"]
MOBILENET = (["Stem"]
             + [f"{k}{i}" for i in range(1, 14) for k in ("D", "P")]
             + ["GAP", "FC"])
RESNET = (["Stem"]
          + [f"B{i}{k}" for i in range(1, 10) for k in ("C1", "C2", "Add")]
          + ["GAP", "FC"])

# name -> (csv tag, layers, deadline [ms], output pdf name, display unit)
WORKLOADS = {
    "LeNet-MNIST": ("lenet-mnist", LENET, 40, "LeNet-MNIST.pdf", "ms"),
    "LeNet-Ciphar": ("lenet-ciphar", LENET, 65, "LeNet-Ciphar.pdf", "ms"),
    "MobileNet-MNIST": ("mobilenet-mnist", MOBILENET, 250, "mobilenet-mnist.pdf", "ms"),
    "MobileNet-Ciphar": ("mobilenet-ciphar", MOBILENET, 250, "mobilenet-ciphar.pdf", "ms"),
    "ResNet-MNIST": ("resnet-mnist", RESNET, 5000, "ResNet-mnist.pdf", "s"),
    "ResNet-Ciphar": ("resnet-ciphar", RESNET, 6000, "ResNet-Ciphar.pdf", "s"),
}
SCALE = {"ms": 1.0, "s": 1e-3}  # CSVs are in ms
BAR_ALPHA = 0.6

# Proposed ResNet-MNIST CSV has a typo'd name in code_mff; fall back to it.
CSV_FALLBACK = {"layer_times-proposed-resnet-mnist.csv":
                "layer_times-proposed-resnet-mnistt.csv"}

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load(method: str, tag: str) -> pd.DataFrame:
    name = f"layer_times-{method.lower()}-{tag}.csv"
    path = DATA / name
    if not path.exists() and name in CSV_FALLBACK:
        path = DATA / CSV_FALLBACK[name]
    return pd.read_csv(path)


def plot(name: str, out_dir: Path) -> Path:
    tag, layers, deadline, pdf_name, unit = WORKLOADS[name]
    k = SCALE[unit]
    deadline = deadline * k
    n = len(layers)
    x = np.arange(1, n + 1)
    width = 0.8 / len(METHODS)

    fig, ax_bar = plt.subplots(figsize=(1.95, 1.75))
    ax_cum = ax_bar.twinx()
    # Lines/deadline above bars.
    ax_cum.set_zorder(ax_bar.get_zorder() + 1)
    ax_cum.patch.set_visible(False)

    bar_max, cum_max = 0.0, 0.0
    for i, m in enumerate(METHODS):
        df = load(m, tag)
        per_layer = df[layers].quantile(0.99).to_numpy() * k
        cumulative = df[layers].cumsum(axis=1).quantile(0.99).to_numpy() * k
        bar_max = max(bar_max, per_layer.max())
        cum_max = max(cum_max, cumulative.max())

        ax_bar.bar(x + (i - 1) * width, per_layer, width=width,
                   color=COLORS[m], alpha=BAR_ALPHA, linewidth=0)
        ax_cum.plot(x, cumulative, color=COLORS[m], linewidth=1.1)

    ax_cum.axhline(deadline, linestyle="--", linewidth=0.8, color="black")

    # Bars live in the lower part of the panel; lines/deadline use full height.
    ax_bar.set_ylim(0, bar_max * 1.8)
    ax_cum.set_ylim(0, max(cum_max, deadline) * 1.08)

    ax_bar.set_xlim(0.4, n + 0.6)
    if n <= 10:
        ax_bar.set_xticks(x)
    else:
        ax_bar.set_xticks([1, 5, 10, 15, 20, 25, 30][: (n // 5) + 1])

    ax_bar.set_xlabel("Layer Number")
    ax_bar.set_ylabel(f"Per-layer P99 ({unit})")
    ax_cum.set_ylabel(f"Cumulative P99 ({unit})")

    handles = [(Patch(color=COLORS[m], alpha=BAR_ALPHA, linewidth=0),
                Line2D([], [], color=COLORS[m], linewidth=1.1))
               for m in METHODS]
    handles.append(Line2D([], [], color="black", linestyle="--", linewidth=0.8))
    ax_cum.legend(handles, METHODS + ["Deadline"],
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
                  loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                  frameon=False, handlelength=2.0, borderpad=0.1,
                  labelspacing=0.15, columnspacing=0.8, handletextpad=0.4)

    fig.tight_layout(pad=0.2)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / pdf_name
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    ap.add_argument("--only", nargs="*", choices=list(WORKLOADS))
    args = ap.parse_args()
    for name in args.only or WORKLOADS:
        print(plot(name, args.out))


if __name__ == "__main__":
    main()
