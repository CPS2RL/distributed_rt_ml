import argparse
from time import perf_counter

from partition_core import (
    Simulator, cases, load_config, make_row,
    make_writer, show_solution, timed_run
)


def main():
    p = argparse.ArgumentParser(description="Run Top-K for one shortlist size")
    p.add_argument("--config", default="experiment_config.json")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--mode", choices=["MNIST", "CIPHAR-10"], default=None)
    args = p.parse_args()

    cfg, directory = load_config(args.config)
    k = int(cfg["single_k"] if args.k is None else args.k)
    if k < 1:
        raise ValueError("K must be >= 1")
    output = directory / cfg["outputs"]["single_k"]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = make_writer(f)
        for case in cases(cfg, args.model, args.mode):
            start = perf_counter()
            sim = Simulator(case, cfg)
            terminal = sim.terminal_cost(case["classifier_spec"])
            precompute_ms = 1000 * (perf_counter() - start)
            solution, solve_ms = timed_run(sim, terminal, k=k)
            row = make_row(case, "TopK", k, solution, solve_ms, precompute_ms)
            writer.writerow(row)
            f.flush()
            show_solution(row, verbose=True)
    print("Saved:", output)


if __name__ == "__main__":
    main()
