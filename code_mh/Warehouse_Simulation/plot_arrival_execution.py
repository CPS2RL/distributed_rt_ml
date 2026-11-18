"""Fig. 4: periodic job releases, execution (average compute time) and
deadline misses for DeepThings, EdgeFlow and the proposed method.

One periodic task with period T = deadline D = 10 ms; each image is a job.
A job starts at max(release, finish of previous job). The part of a job that
executes after its deadline is hatched.

Reads code_mff/Warehouse_Simulation/average_compute_time.csv (unchanged).

Usage:
    python plot_arrival_execution.py [--out DIR] [--waiting]
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
CSV_FILE = HERE.parents[1] / "code_mff" / "Warehouse_Simulation" / "average_compute_time.csv"

METHODS = ["DeepThings", "EdgeFlow", "Proposed"]
PRODUCTS = ["product_0001", "product_0002", "product_0003"]
PERIOD = 10.0  # ms; implicit deadline D = T

JOB_COLORS = ["#D6DCE5", "#E9DCC3", "#D3E2CF"]  # cool grey, sand, sage
DEADLINE_COLOR = "#800000"  # maroon (same as Fig. 2)
INK = "black"

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2,
    "ytick.major.size": 0,
    "hatch.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load():
    with open(CSV_FILE, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {(r["configuration"], r["product"]): float(r["average_total_compute_ms"])
            for r in rows if r["configuration"] in METHODS and r["product"] in PRODUCTS}


def plot(out_dir: Path, show_waiting: bool) -> Path:
    compute = load()
    releases = [i * PERIOD for i in range(len(PRODUCTS))]
    deadlines = [r + PERIOD for r in releases]

    fig, ax = plt.subplots(figsize=(3.5, 1.75))
    bar_h, gap = 0.42, 0.8
    arrow_len = 0.2

    for m, method in enumerate(METHODS):
        y = (len(METHODS) - 1 - m) * gap
        finish = 0.0
        for i, prod in enumerate(PRODUCTS):
            dur = compute[(method, prod)]
            start = max(releases[i], finish)
            finish = start + dur

            if show_waiting and start > releases[i]:
                # Waiting (queued behind the previous job): release -> start.
                ax.plot([releases[i], start], [y, y], color="0.45", lw=0.8,
                        solid_capstyle="butt", zorder=1)
                ax.plot([releases[i]] * 2, [y - 0.07, y + 0.07], color="0.45",
                        lw=0.8, zorder=1)

            ax.add_patch(Rectangle((start, y - bar_h / 2), dur, bar_h,
                                   facecolor=JOB_COLORS[i], edgecolor="none",
                                   zorder=2))
            late_from = max(start, deadlines[i])
            if finish > late_from:
                ax.add_patch(Rectangle((late_from, y - bar_h / 2),
                                       finish - late_from, bar_h,
                                       facecolor="none", hatch="//////",
                                       edgecolor=DEADLINE_COLOR, linewidth=0,
                                       zorder=3))
            ax.add_patch(Rectangle((start, y - bar_h / 2), dur, bar_h,
                                   facecolor="none", edgecolor=INK,
                                   linewidth=0.5, zorder=4))

        # Release (up, below the bar) and deadline (down, above the bar).
        for r in releases:
            ax.annotate("", xy=(r, y - bar_h / 2), xytext=(r, y - bar_h / 2 - arrow_len),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7,
                                        mutation_scale=6, shrinkA=0, shrinkB=0),
                        annotation_clip=False, zorder=5)
        for d in deadlines:
            ax.annotate("", xy=(d, y + bar_h / 2), xytext=(d, y + bar_h / 2 + arrow_len),
                        arrowprops=dict(arrowstyle="-|>", color=DEADLINE_COLOR, lw=0.7,
                                        mutation_scale=6, shrinkA=0, shrinkB=0),
                        annotation_clip=False, zorder=5)

    ax.set_yticks([(len(METHODS) - 1 - m) * gap for m in range(len(METHODS))])
    ax.set_yticklabels(METHODS)
    ax.set_ylim(-bar_h / 2 - arrow_len - 0.08,
                (len(METHODS) - 1) * gap + bar_h / 2 + arrow_len + 0.08)
    ax.set_xlim(-1, 43)
    ax.set_xticks(range(0, 50, 10))
    ax.set_xlabel("Time (ms)")
    ax.grid(axis="x", color="0.88", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    handles = [Patch(facecolor=c, edgecolor=INK, linewidth=0.5, label=f"Job {k+1}")
               for k, c in enumerate(JOB_COLORS)]
    handles += [
        Line2D([], [], linestyle="none", marker="^", color=INK, markersize=4,
               label="Release"),
        Line2D([], [], linestyle="none", marker="v", color=DEADLINE_COLOR,
               markersize=4, label="Deadline"),
        Patch(facecolor="white", edgecolor=DEADLINE_COLOR, hatch="//////",
              linewidth=0, label="After Deadline"),
    ]
    if show_waiting:
        handles.append(Line2D([], [], color="0.45", lw=0.8, label="Waiting"))
    # matplotlib fills legend columns first; interleave so rows read
    # "Job 1, Job 2, Job 3" / "Release, Deadline, After deadline".
    if not show_waiting:
        handles = [handles[k] for k in (0, 3, 1, 4, 2, 5)]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=4 if show_waiting else 3, frameon=False, handlelength=1.4,
              columnspacing=1.0, handletextpad=0.4, labelspacing=0.25,
              borderpad=0.1)

    fig.tight_layout(pad=0.2)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / ("arrival_execution_waiting.pdf" if show_waiting
                     else "arrival_execution.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    ap.add_argument("--waiting", action="store_true",
                    help="also draw the waiting time (release -> start)")
    args = ap.parse_args()
    print(plot(args.out, args.waiting))


if __name__ == "__main__":
    main()
