"""Fig. 2: per-layer P99 execution time (bars, left axis) and cumulative P99
execution time (lines, right axis) with the task deadline, for six workloads.

Reads the measured CSVs from code_mff/Execution_Time_Comparison (unchanged).
Default: one combined 1x6 PDF for a two-column figure* with shared axis
labels and one legend (--layout 2x3 for a grid: rows = dataset, cols = model).
--single: one PDF per workload instead (old one-column subfloat layout).

Usage:
    python plot_layer_cumulative.py [--out DIR] [--layout {1x6,2x3}]
    python plot_layer_cumulative.py --single [--only NAME ...]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import (FuncFormatter, LogLocator, MaxNLocator,
                               NullFormatter)

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "code_mff" / "Execution_Time_Comparison"

METHODS = ["DeepThings", "EdgeFlow", "Proposed"]
# Seaborn palettes: "pastel" for bar fills, matching "deep" hue for lines.
BAR_COLORS = {"DeepThings": "#A1C9F4", "EdgeFlow": "#FFB482", "Proposed": "#8DE5A1"}
LINE_COLORS = {"DeepThings": "#4C72B0", "EdgeFlow": "#DD8452", "Proposed": "#3A8A4C"}
DEADLINE_COLOR = "#800000"  # maroon
MARKERS = {"DeepThings": "o", "EdgeFlow": "s", "Proposed": "^"}
LINE_KW = dict(linewidth=1.0, markersize=3.0, markeredgewidth=0.8,
               markerfacecolor="white")  # hollow; edge = line color

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

# Proposed ResNet-MNIST CSV has a typo'd name in code_mff; fall back to it.
CSV_FALLBACK = {"layer_times-proposed-resnet-mnist.csv":
                "layer_times-proposed-resnet-mnistt.csv"}

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
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




def kfmt(v, _pos):
    """1500 -> 1.5k, 2000 -> 2k; small values unchanged."""
    if abs(v) >= 1000:
        s = f"{v / 1000:.1f}".rstrip("0").rstrip(".")
        return f"{s}k"
    return f"{v:g}"


def legend_handles():
    if PER_LAYER == "dots":
        first = [Line2D([], [], linestyle="none", marker=MARKERS[m],
                        markersize=2.6, markeredgewidth=0.4,
                        markeredgecolor=LINE_COLORS[m],
                        markerfacecolor=BAR_COLORS[m]) for m in METHODS]
    else:
        first = [Patch(facecolor=BAR_COLORS[m], edgecolor="black",
                       linewidth=0.4) for m in METHODS]
    return [(f, Line2D([], [], color=LINE_COLORS[m], marker=MARKERS[m],
                       **LINE_KW)) for f, m in zip(first, METHODS)]


LEGEND_KW = dict(handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
                 frameon=False, handlelength=1.8, borderpad=0.1,
                 labelspacing=0.15, columnspacing=0.6, handletextpad=0.3)


def blocks_of(layers):
    """Group a deep net's layers into blocks: Stem | each D/P pair or residual
    block | GAP+FC head. Returns a list of column lists."""
    groups, cur, key = [], [], None
    for l in layers:
        if l in ("Stem",):
            groups.append([l]); continue
        if l in ("GAP", "FC"):
            if cur:
                groups.append(cur); cur, key = [], None
            if groups and groups[-1][0] == "GAP":
                groups[-1].append(l)
            else:
                groups.append([l])
            continue
        # MobileNet: D3/P3 -> "3"; ResNet: B3C1/B3C2/B3Add -> "B3"
        k = l[1:] if l[0] in "DP" else l[:2] if l[2].isalpha() else l[:3]
        if k != key and cur:
            groups.append(cur); cur = []
        cur.append(l); key = k
    if cur:
        groups.append(cur)
    return groups


def draw_panel(ax_bar, name, unit=None, compact=False):
    """Draw one workload into ax_bar (+ a twin axis). Returns (ax_bar, ax_cum, unit)."""
    tag, layers, deadline, _pdf, default_unit = WORKLOADS[name]
    unit = unit or default_unit
    k = SCALE[unit]
    group = GROUP_BLOCKS and len(layers) > 10
    cols = blocks_of(layers) if group else [[l] for l in layers]
    n = len(cols)
    x = np.arange(1, n + 1)
    width = 0.8 / len(METHODS)

    ax_cum = ax_bar.twinx()
    ax_cum.set_zorder(ax_bar.get_zorder() + 1)  # lines/deadline above bars
    ax_cum.patch.set_visible(False)

    per_layer, cumulative = {}, {}
    for m in METHODS:
        df = load(m, tag)
        t = pd.concat([df[c].sum(axis=1) for c in cols], axis=1)  # runs x units
        per_layer[m] = t.quantile(0.99).to_numpy() * k
        cumulative[m] = t.cumsum(axis=1).quantile(0.99).to_numpy() * k
    bar_max = max(v.max() for v in per_layer.values())
    cum_max = max(v.max() for v in cumulative.values())

    for i, m in enumerate(METHODS):
        if PER_LAYER == "bars":
            ax_bar.bar(x + (i - 1) * width, per_layer[m], width=width,
                       color=BAR_COLORS[m],
                       edgecolor="black" if BAR_EDGES else "none",
                       linewidth=0.3 if compact else 0.4)
        elif PER_LAYER == "step":
            # Thin per-layer staircase with a light fill underneath.
            ax_bar.stairs(per_layer[m], np.arange(0.5, n + 1), baseline=0,
                          fill=True, color=BAR_COLORS[m], alpha=0.35)
            ax_bar.stairs(per_layer[m], np.arange(0.5, n + 1), baseline=None,
                          color=LINE_COLORS[m], linewidth=0.6)
        elif PER_LAYER == "dots":
            # Lollipop-free dot plot: one small marker per method per layer.
            ax_bar.plot(x + (i - 1) * width, per_layer[m], linestyle="none",
                        marker=MARKERS[m], markersize=2.2, markeredgewidth=0.4,
                        markeredgecolor=LINE_COLORS[m],
                        markerfacecolor=BAR_COLORS[m])
    for m in METHODS:
        # Mark every unit for short sequences; thin out markers for long ones.
        ax_cum.plot(x, cumulative[m], color=LINE_COLORS[m], marker=MARKERS[m],
                    markevery=1 if n <= 16 else 3, **LINE_KW)

    ax_cum.axhline(deadline * k, linestyle="--", linewidth=0.8,
                   color=DEADLINE_COLOR)

    # Bars live in the lower part of the panel; lines/deadline use full height.
    if PER_LAYER == "dots":
        # Per-layer times span ~3-4 decades (pool/add/GAP vs conv/FC), so a
        # linear axis flattens the cheap layers onto zero; use log scale.
        bar_min = min(v[v > 0].min() for v in per_layer.values())
        ax_bar.set_yscale("log")
        ax_bar.set_ylim(bar_min / 2, bar_max * 2)
    else:
        ax_bar.set_ylim(0, bar_max * 1.8)
    ax_cum.set_ylim(0, max(cum_max, deadline * k) * 1.08)

    ax_bar.set_xlim(0.4, n + 0.6)
    if n <= 10:
        ax_bar.set_xticks(x)
    elif n <= 16:
        ax_bar.set_xticks([1, 5, 10, 15][: n // 5 + 1])
    else:
        ax_bar.set_xticks([1, 10, 20, 30] if compact else
                          [1, 5, 10, 15, 20, 25, 30][: n // 5 + 1])
    for ax in (ax_bar, ax_cum):
        if ax.get_yscale() == "log":
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto",
                                                  numticks=10))
            ax.yaxis.set_minor_formatter(NullFormatter())
        elif compact:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
        if compact:
            ax.tick_params(axis="both", labelsize=6, pad=1.5)
    return ax_bar, ax_cum, unit


def plot_single(name, out_dir):
    fig, ax_bar = plt.subplots(figsize=(1.95, 1.75))
    ax_bar, ax_cum, unit = draw_panel(ax_bar, name)
    ax_bar.set_xlabel("Layer Index")
    ax_bar.set_ylabel(f"Per-Layer\nExecution Time ({unit})", linespacing=0.95, multialignment="center")
    ax_cum.set_ylabel(f"Cumulative\nExecution Time ({unit})", linespacing=0.95, multialignment="center")
    ax_cum.legend(legend_handles(), METHODS, loc="lower center",
                  bbox_to_anchor=(0.5, 1.01), ncol=3, **LEGEND_KW)
    fig.tight_layout(pad=0.2)
    out = out_dir / WORKLOADS[name][3]
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return out


# Panel order for each combined layout (row-major).
LAYOUTS = {
    "1x6": [["LeNet-MNIST", "LeNet-Ciphar", "MobileNet-MNIST",
             "MobileNet-Ciphar", "ResNet-MNIST", "ResNet-Ciphar"]],
    "2x3": [["LeNet-MNIST", "MobileNet-MNIST", "ResNet-MNIST"],
            ["LeNet-Ciphar", "MobileNet-Ciphar", "ResNet-Ciphar"]],
}
TITLES = {"LeNet-MNIST": "LeNet (MNIST)",
          "MobileNet-MNIST": "MobileNet (MNIST)",
          "ResNet-MNIST": "ResNet (MNIST)",
          "LeNet-Ciphar": "LeNet (CIFAR-10)",
          "MobileNet-Ciphar": "MobileNet (CIFAR-10)",
          "ResNet-Ciphar": "ResNet (CIFAR-10)"}
# figsize (in), subplots_adjust kwargs, y of the shared x label
GEOMETRY = {
    "1x6": ((7.16, 1.6), dict(left=0.06, right=0.94, bottom=0.19,
                              top=0.80, wspace=0.62), 0.0),
    "2x3": ((7.0, 3.1), dict(left=0.065, right=0.935, bottom=0.11, top=0.86,
                             wspace=0.42, hspace=0.42), 0.015),
}
GRID_PDF = "layer_cumulative_{}.pdf"
GROUP_BLOCKS = False  # --blocks: MobileNet/ResNet bars per block
BAR_EDGES = True      # --no-edges: flat pastel bars
PER_LAYER = "dots"    # --per-layer {dots,bars,step}


def plot_grid(out_dir, layout="1x6"):
    grid = LAYOUTS[layout]
    figsize, adjust, xlabel_y = GEOMETRY[layout]
    fig, axes = plt.subplots(len(grid), len(grid[0]), figsize=figsize,
                             squeeze=False)
    fig.subplots_adjust(**adjust)
    for r, row in enumerate(grid):
        for c, name in enumerate(row):
            ax_bar, ax_cum, _ = draw_panel(axes[r, c], name, unit="ms",
                                         compact=(layout == "1x6"))
            ax_bar.set_title(TITLES[name],
                             fontsize=6.5, pad=2.5)

    mid_y = (adjust["bottom"] + adjust["top"]) / 2
    fig.supxlabel("Layer Index", fontsize=7, y=xlabel_y, va="bottom")
    fig.text(0.005, mid_y, "Per-Layer\nExec. Time (ms)" if layout == "1x6"
             else "Per-Layer Execution Time (ms)", rotation=90,
             ha="left", va="center", fontsize=7, linespacing=0.95, multialignment="center")
    fig.text(0.995, mid_y, "Cumulative\nExec. Time (ms)" if layout == "1x6"
             else "Cumulative Execution Time (ms)", rotation=270,
             ha="right", va="center", fontsize=7, linespacing=0.95, multialignment="center")
    fig.legend(legend_handles(), METHODS, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=3,
               **{**LEGEND_KW, "columnspacing": 1.6})
    out = out_dir / GRID_PDF.format(layout)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return out


def plot_split(out_dir):
    """2x6 figure*: top row cumulative P99 (lines + deadline), bottom row
    per-layer P99 (thin lines), sharing x per column. No bars, no twin axes."""
    order = LAYOUTS["1x6"][0]
    fig, axes = plt.subplots(2, 6, figsize=(7.16, 2.35), sharex="col",
                             gridspec_kw=dict(height_ratios=[1.25, 1]))
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.13, top=0.84,
                        wspace=0.42, hspace=0.12)
    for c, name in enumerate(order):
        tag, layers, deadline, _pdf, _u = WORKLOADS[name]
        n = len(layers)
        x = np.arange(1, n + 1)
        top, bot = axes[0, c], axes[1, c]
        for m in METHODS:
            df = load(m, tag)
            cum = df[layers].cumsum(axis=1).quantile(0.99).to_numpy()
            per = df[layers].quantile(0.99).to_numpy()
            me = 1 if n <= 10 else 3
            top.plot(x, cum, color=LINE_COLORS[m], marker=MARKERS[m],
                     markevery=me, **LINE_KW)
            bot.plot(x, per, color=LINE_COLORS[m], marker=MARKERS[m],
                     markevery=me, **{**LINE_KW, "linewidth": 0.8,
                                      "markersize": 2.4})
        top.axhline(deadline, linestyle="--", linewidth=0.8,
                    color=DEADLINE_COLOR)
        top.set_ylim(0, None)
        bot.set_ylim(0, None)
        top.set_title(TITLES[name], fontsize=6.5, pad=2.5)
        bot.set_xlim(0.5, n + 0.5)
        bot.set_xticks(x if n <= 10 else [1, 10, 20, 30])
        for ax in (top, bot):
            ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis="both", labelsize=6, pad=1.5)
            ax.grid(axis="y", linewidth=0.3, color="0.85")
            ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("Cumulative\nTime (ms)", linespacing=0.95, multialignment="center")
    axes[1, 0].set_ylabel("Per-Layer\nTime (ms)", linespacing=0.95, multialignment="center")
    fig.align_ylabels(axes[:, 0])
    fig.supxlabel("Layer Index", fontsize=7, y=0.0, va="bottom")
    handles = [Line2D([], [], color=LINE_COLORS[m], marker=MARKERS[m],
                      **LINE_KW) for m in METHODS]
    handles.append(Line2D([], [], color=DEADLINE_COLOR, linestyle="--",
                          linewidth=0.8))
    fig.legend(handles, METHODS + ["Deadline"], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=4,
               **{k: v for k, v in LEGEND_KW.items() if k != "handler_map"},
               )
    out = out_dir / "layer_cumulative_split.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    ap.add_argument("--layout", choices=list(LAYOUTS) + ["split"], default="1x6")
    ap.add_argument("--blocks", action="store_true",
                    help="aggregate MobileNet/ResNet into blocks")
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--per-layer", choices=["dots", "bars", "step"], default="dots")
    ap.add_argument("--single", action="store_true",
                    help="one PDF per workload instead of a combined figure")
    ap.add_argument("--only", nargs="*", choices=list(WORKLOADS))
    args = ap.parse_args()
    global GROUP_BLOCKS, BAR_EDGES, PER_LAYER
    GROUP_BLOCKS, BAR_EDGES = args.blocks, not args.no_edges
    PER_LAYER = args.per_layer
    args.out.mkdir(parents=True, exist_ok=True)
    if args.single:
        for name in args.only or WORKLOADS:
            print(plot_single(name, args.out))
    else:
        print(plot_split(args.out) if args.layout == "split"
              else plot_grid(args.out, args.layout))


if __name__ == "__main__":
    main()
