import argparse
import csv
from time import perf_counter

from partition_core import (
    RESULT_FIELDS,
    Simulator,
    cases,
    load_config,
    make_row,
    show_solution,
    timed_run,
)


def run(args):
    cfg, folder = load_config(args.config)
    k_start = int(cfg["k_start"])
    k_end = int(cfg["k_end"])
    k_step = int(cfg["k_step"])
    if args.runs < 1 or k_start < 1 or k_end < k_start or k_step < 1:
        raise ValueError("Runs and K range/step must be positive and valid")

    k_values = list(range(k_start, k_end + 1, k_step))
    dp_file = folder / "dp_results.csv"
    sweep_file = folder / "sweep_k_results.csv"
    fields = ["run", *RESULT_FIELDS]

    with dp_file.open("w", newline="", encoding="utf-8") as dp_out, \
         sweep_file.open("w", newline="", encoding="utf-8") as sweep_out:

        dp_writer = csv.DictWriter(dp_out, fieldnames=fields)
        sweep_writer = csv.DictWriter(sweep_out, fieldnames=fields)
        dp_writer.writeheader()
        sweep_writer.writeheader()

        for case in cases(cfg, args.model, args.mode):
            begin = perf_counter()
            sim = Simulator(case, cfg)
            terminal = sim.terminal_cost(case["classifier_spec"])
            precompute_ms = (perf_counter() - begin) * 1000.0

            print(
                f"\n{case['model']} | {case['mode']} | "
                f"{case['classifier']} | "
                f"input={case['input_shape']} | "
                f"devices={case['devices']}",
                flush=True,
            )

            for run_number in range(1, args.runs + 1):
                # One exact Full DP measurement for this run.
                dp_solution, dp_time = timed_run(sim, terminal, k=None)
                dp_row = make_row(
                    case, "FullDP", None,
                    dp_solution, dp_time, precompute_ms,
                )
                dp_writer.writerow({"run": run_number, **dp_row})
                dp_out.flush()
                print(f"Run {run_number}/{args.runs}: DP", end="  ")
                show_solution(dp_row, verbose=args.show_paths)

                # Measure every requested K (including infeasible outcomes).
                for k in k_values:
                    solution, solve_ms = timed_run(sim, terminal, k=k)
                    row = make_row(
                        case, "TopK", k,
                        solution, solve_ms, precompute_ms,
                    )
                    sweep_writer.writerow({"run": run_number, **row})

                    if args.show_paths:
                        print(f"Run {run_number}/{args.runs}: ", end="")
                        show_solution(row, verbose=True)

                sweep_out.flush()
                print(f"  Completed K sweep: {len(k_values)} values", flush=True)

    print("\nSaved:", dp_file, "\n      ", sweep_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment_config.json")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--model", default=None,
                        help="Optional: LeNet5, ResNet20, MobileNetV1_0.25")
    parser.add_argument("--mode", choices=["MNIST", "CIPHAR-10"], default=None)
    parser.add_argument("--show-paths", action="store_true",
                        help="Also print every saved layer/device partition")
    run(parser.parse_args())
