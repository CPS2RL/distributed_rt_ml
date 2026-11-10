import csv
import matplotlib.pyplot as plt

with open("arrival_execution.csv") as f:
    data = list(csv.DictReader(f))

methods = ["DeepThings", "EdgeFlow", "Proposed"]
colors = ["#0072B2", "#56B4E9", "#009E73", "#E69F00"]

plt.rcParams["font.size"] = 6
fig, ax = plt.subplots(figsize=(3.5, 1.9))

finish = {method: 0 for method in methods}

for row in data:
    method = row["method"]
    task = int(row["Task"])
    arrival = float(row["arrival_ms"])
    duration = float(row["processing_ms"])

    start = max(arrival, finish[method])
    finish[method] = start + duration

    y = (2 - methods.index(method)) * 0.6

    ax.barh(y, duration, left=start, height=0.4,
            color=colors[(task-1) % 4], edgecolor="black")

for x in sorted(set(float(r["arrival_ms"]) for r in data)):
    ax.axvline(x, color="gray", linestyle="--",
               linewidth=0.7, alpha=0.5)

ax.set_yticks([1.2, 0.6, 0], methods)
ax.set_ylim(-0.4, 1.6)
ax.set_xlim(0, max(finish.values()) * 1.05)

plt.tight_layout()
plt.savefig("arrival_execution.png", dpi=220)
plt.savefig("arrival_execution.pdf")
plt.close()