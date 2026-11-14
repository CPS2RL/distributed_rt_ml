#!/usr/bin/env python3
"""Run three distributed LeNet configurations repeatedly in an active Gazebo world.

Examples:
    python3 run_script.py
    python3 run_script.py --runs 10 --count 10 --workers 1
"""

import argparse
import csv
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIGS = (
    ("DeepThings", "independent", "DeepThings.json"),
    ("Proposed", "dependent", "Proposed.json"),
    ("EdgeFlow", "dependent", "EdgeFlow.json"),
)
LOG_FILE = "distributed_object_times.csv"


def save_csv(path, fields, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=50,
                        help="repetitions per configuration (default: 10)")
    parser.add_argument("--count", type=int, default=5,
                        help="objects per run (default: 10)")
    parser.add_argument("--workers", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--save-captures", action="store_true",
                        help="deprecated; ignored because the sorter no longer saves captures")
    args = parser.parse_args()

    if args.runs < 1 or not 1 <= args.count <= 30:
        parser.error("--runs must be positive and --count must be between 1 and 30")
    if args.save_captures:
        print("[NOTE] --save-captures is ignored by the updated five_camera_sorter.py.")

    needed = ("five_camera_sorter.py", "distributed_lenet.py") + tuple(
        config for _, _, config in CONFIGS
    )
    for filename in needed:
        if not (ROOT / filename).is_file():
            parser.error(f"Missing required file: {ROOT / filename}")

    output = ROOT / ("three_method_comparison_" +
                     datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True)

    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")

    all_rows = []
    samples = defaultdict(list)
    completed = defaultdict(int)

    for name, method, config in CONFIGS:
        shutil.copy2(ROOT / config, output / config)

        for trial in range(1, args.runs + 1):
            folder = output / name / f"run_{trial:02d}"
            folder.mkdir(parents=True)

            # Prevent stale measurements from a previous execution being reused.
            source = ROOT / LOG_FILE
            source.unlink(missing_ok=True)

            cmd = [
                sys.executable, str(ROOT / "five_camera_sorter.py"), "run",
                "--count", str(args.count),
                "--workers", str(args.workers),
                "--method", method,
                "--config", str(ROOT / config),
            ]

            print(f"[{name}] Run {trial}/{args.runs}", flush=True)
            with (folder / "console.log").open("w") as logfile:
                proc = subprocess.run(
                    cmd, cwd=ROOT, env=env, stdout=logfile,
                    stderr=subprocess.STDOUT, check=False
                )

            if source.is_file():
                csv_path = folder / LOG_FILE
                shutil.copy2(source, csv_path)
            else:
                print(f"  [WARN] Missing {LOG_FILE}; see {folder / 'console.log'}")
                continue

            if proc.returncode != 0:
                print(f"  [WARN] Controller exited with code {proc.returncode}; "
                      f"see {folder / 'console.log'}")
                continue

            valid = []
            with csv_path.open(newline="") as f:
                for row in csv.DictReader(f):
                    product = (row.get("product") or "").strip()
                    try:
                        ms = float(row["total_compute_ms"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not product or not math.isfinite(ms) or ms <= 0:
                        continue
                    valid.append((product, ms))

            # Incomplete runs stay archived, but cannot skew comparisons.
            if len(valid) != args.count or len({p for p, _ in valid}) != args.count:
                print(f"  [WARN] Only {len(valid)}/{args.count} valid object "
                      "measurements; excluding this run from aggregation")
                continue

            for product, ms in valid:
                all_rows.append({
                    "configuration": name,
                    "run": trial,
                    "product": product,
                    "total_compute_ms": f"{ms:.6f}",
                })
                samples[name].append(ms)
            completed[name] += 1
            print(f"  [OK] {len(valid)} objects; "
                  f"average compute time = {statistics.fmean(ms for _, ms in valid):.3f} ms")

    combined = output / "all_measurements.csv"
    save_csv(combined,
             ["configuration", "run", "product", "total_compute_ms"],
             all_rows)

    print("\n=== Overall results: compute time per object ===")
    for name, _, _ in CONFIGS:
        vals = samples[name]
        if vals:
            print(f"{name:12s}: {completed[name]}/{args.runs} complete runs; "
                  f"{len(vals)} objects; mean = {statistics.fmean(vals):.3f} ms")
        else:
            print(f"{name:12s}: no complete runs")
    print(f"\nCombined measurements: {combined}")
    if not all_rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
