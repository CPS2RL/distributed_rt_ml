import csv
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path(".\Data")

OUTPUT_FILE = DATA_DIR / "average_compute_time.csv"

# ============================================================
# READ ALL CSV FILES
# ============================================================

data = defaultdict(list)

files = sorted(DATA_DIR.rglob("all_measurements.csv"))

print(f"Found {len(files)} CSV files")

if not files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

for file in files:
    print(f"Reading: {file.parent.name}")

    with open(file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            configuration = row["configuration"]
            product = row["product"]
            compute_ms = float(row["total_compute_ms"])

            data[(configuration, product)].append(compute_ms)

# ============================================================
# CALCULATE AVERAGE AND SAVE
# ============================================================

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "configuration",
        "product",
        "average_total_compute_ms"
    ])

    for (configuration, product), values in sorted(data.items()):
        average = sum(values) / len(values)

        writer.writerow([
            configuration,
            product,
            f"{average:.6f}"
        ])


print(f"Output saved: {OUTPUT_FILE}")
