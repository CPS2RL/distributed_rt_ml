import argparse
import csv
from pathlib import Path
from time import perf_counter

from partition_core import (
    RESULT_FIELDS, Simulator, cases, load_config, make_row,
    make_writer, show_solution, timed_run
)


def main():
    p = argparse.ArgumentParser(description="Run exact full DP and export partitions")
    p.add_argument("--config", default="experiment_config.json")
    p.add_argument("--model", default=None)
    p.add_argument("--mode", choices=["MNIST", "CIPHAR-10"], default=None)
    args = p.parse_args()

    cfg, directory = load_config(args.config)
    output = directory / cfg["outputs"]["dp"]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = make_writer(f)
        for case in cases(cfg, args.model, args.mode):
            start = perf_counter()
            sim = Simulator(case, cfg)
            terminal = sim.terminal_cost(case["classifier_spec"])
            precompute_ms = 1000 * (perf_counter() - start)
            solution, solve_ms = timed_run(sim, terminal, k=None)
            row = make_row(case, "FullDP", None, solution, solve_ms, precompute_ms)
            writer.writerow(row)
            f.flush()
            show_solution(row, verbose=True)
    print("Saved:", output)


if __name__ == "__main__":
    main()
