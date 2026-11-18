import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "average_compute_time.csv"

methods = ["DeepThings", "EdgeFlow", "Proposed"]

products = [
    "product_0001",
    "product_0002",
    "product_0003",
]

colors = ["#0072B2", "#009E73", "#E69F00"]

PERIOD = 10  # ms

arrivals = {
    product: i * PERIOD
    for i, product in enumerate(products)
}

deadlines = {
    product: arrivals[product] + PERIOD
    for product in products
}

# ============================================================
# READ CSV
# ============================================================

with open(CSV_FILE, newline="", encoding="utf-8-sig") as f:
    data = list(csv.DictReader(f))

compute = {}

for row in data:
    method = row["configuration"]
    product = row["product"]

    if method in methods and product in products:
        compute[(method, product)] = float(
            row["average_total_compute_ms"]
        )

# ============================================================
# PLOT
# ============================================================

plt.rcParams["font.size"] = 9

fig, ax = plt.subplots(figsize=(7.5, 3.0))

finish = {method: 0 for method in methods}

bar_height = 0.40
row_gap = 0.65

for m, method in enumerate(methods):

    y = (2 - m) * row_gap

    # EXECUTION BARS
    for i, product in enumerate(products):

        duration = compute[(method, product)]
        arrival = arrivals[product]

        start = max(arrival, finish[method])
        finish[method] = start + duration

        ax.barh(
            y,
            duration,
            left=start,
            height=bar_height,
            color=colors[i],
            edgecolor="black",
            linewidth=1,
            zorder=2
        )

    # ARRIVAL ARROWS (UPWARD)
    for product in products:

        arrival = arrivals[product]

        ax.annotate(
            "",
            xy=(arrival, y + 0.27),
            xytext=(arrival, y),
            arrowprops=dict(
                arrowstyle="-|>",
                color="black",
                lw=1.2,
                mutation_scale=12,
                shrinkA=0,
                shrinkB=0
            ),
            annotation_clip=False,
            zorder=4
        )

    # DEADLINE ARROWS (DOWNWARD)
    for product in products:

        deadline = deadlines[product]

        ax.annotate(
            "",
            xy=(deadline, y - 0.28),
            xytext=(deadline, y+ 0.25),
            arrowprops=dict(
                arrowstyle="-|>",
                color="black",
                lw=1.2,
                mutation_scale=13,
                shrinkA=0,
                shrinkB=0
            ),
            annotation_clip=False,
            zorder=5
        )

# ============================================================
# LEGEND
# ============================================================

legend_elements = [
    Patch(
        facecolor=colors[0],
        edgecolor="black",
        label="Task 1"
    ),
    Patch(
        facecolor=colors[1],
        edgecolor="black",
        label="Task 2"
    ),
    Patch(
        facecolor=colors[2],
        edgecolor="black",
        label="Task 3"
    )
]

ax.legend(
    handles=legend_elements,
    loc="lower right",
    frameon=True,
    edgecolor="black",
    fontsize=8
)

# ============================================================
# FORMATTING
# ============================================================

ax.set_yticks([2 * row_gap, row_gap, 0])
ax.set_yticklabels(methods)

ax.set_ylim(-0.40, 2 * row_gap + 0.40)

max_time = max(
    max(finish.values()),
    max(deadlines.values())
)

ax.set_xlim(-1, max_time * 1.05)

ax.set_xlabel("Time (ms)")

ax.set_xticks(
    range(0, int(max_time) + 10, 10)
)

# COMPLETE RECTANGULAR BORDER
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(1.0)

ax.tick_params(
    axis="both",
    direction="out",
    width=1
)

plt.tight_layout()

plt.savefig(
    "arrival_execution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "arrival_execution.pdf",
    bbox_inches="tight"
)

plt.show()