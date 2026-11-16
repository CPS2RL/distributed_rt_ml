import csv
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from math import comb
from pathlib import Path
from time import perf_counter

import numpy as np


RESULT_FIELDS = [
    "case_id", "model", "mode", "classifier", "algorithm", "K",
    "input_h", "input_w", "devices", "deadline_us", "feasible",
    "inference_ms", "solve_time_ms", "precompute_ms", "partitions_json"
]


@dataclass(frozen=True)
class Layer:
    name: str
    op: str
    in_h: int
    in_w: int
    in_c: int
    out_h: int
    out_w: int
    out_c: int
    kernel: int
    stride: int
    padding: int


def load_config(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    h, w = cfg["input_size"]
    n = int(cfg["devices"])
    if min(h, w, n) <= 0:
        raise ValueError("input_size and devices must be positive")
    return cfg, path.parent


def cases(cfg, selected_model=None, selected_mode=None):
    h, w = cfg["input_size"]
    n = int(cfg["devices"])
    chosen = set(cfg.get("classifiers_to_run", ["original_10class"]))
    for model, spec in cfg["models"].items():
        if selected_model is not None and model != selected_model:
            continue
        for mode, channels in cfg["input_modes"].items():
            if selected_mode is not None and mode != selected_mode:
                continue
            for classifier_name, classifier in spec["classifiers"].items():
                if classifier_name not in chosen:
                    continue
                signature = {
                    "input_size": [h, w], "devices": n,
                    "channels": channels, "spec": spec,
                    "classifier_name": classifier_name
                }
                case_id = hashlib.sha256(
                    json.dumps(signature, sort_keys=True).encode("utf-8")
                ).hexdigest()[:16]
                yield {
                    "case_id": case_id,
                    "model": model, "mode": mode,
                    "classifier": classifier_name,
                    "classifier_spec": classifier,
                    "spec": spec, "input_shape": (h, w, channels),
                    "devices": n
                }


def conv_size(size, kernel, stride, padding):
    answer = (size + 2 * padding - kernel) // stride + 1
    if answer <= 0:
        raise ValueError("Input is too small for the configured CNN")
    return answer


def build_layers(spec, input_shape):
    h, w, c = input_shape
    layers = []

    def append(name, op, out_c, kernel, stride, padding):
        nonlocal h, w, c
        oh = conv_size(h, kernel, stride, padding)
        ow = conv_size(w, kernel, stride, padding)
        layers.append(Layer(
            name, op, h, w, c, oh, ow, out_c, kernel, stride, padding
        ))
        h, w, c = oh, ow, out_c

    arch = spec["architecture"]
    if arch == "sequential":
        for item in spec["layers"]:
            append(
                item["name"], item["op"], item.get("out_channels", c),
                item["kernel"], item["stride"], item["padding"]
            )
    elif arch == "resnet20":
        append("stem", "standard", spec["stem_channels"], 3, 1, 1)
        for stage_id, stage in enumerate(spec["stages"], 1):
            for block in range(stage["blocks"]):
                stride = 2 if stage_id > 1 and block == 0 else 1
                append(
                    f"stage{stage_id}_block{block + 1}", "residual",
                    stage["channels"], 3, stride, 1
                )
    elif arch == "mobilenet_v1":
        append("stem", "standard", spec["stem_channels"], 3, 2, 1)
        for i, (out_channels, stride) in enumerate(spec["blocks"], 1):
            append(f"DW{i}", "depthwise", c, 3, stride, 1)
            append(f"PW{i}", "pointwise", out_channels, 1, 1, 0)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return layers


def row_partitions(height, devices, allow_zero, max_states):
    if allow_zero:
        count = comb(height + devices - 1, devices - 1)
    else:
        count = comb(height - 1, devices - 1) if height >= devices else 0
    if count == 0:
        raise ValueError(
            f"Cannot give {devices} devices positive rows at height {height}; "
            "set allow_zero_rows=true"
        )
    if count > max_states:
        raise ValueError(
            f"Height={height}, devices={devices} gives {count:,} partitions "
            f"(limit {max_states:,}). Exact DP is too large for this "
            "enumeration; reduce input size/devices or raise the limit "
            "only if compute resources permit."
        )

    if devices == 1:
        return np.array([[height]], dtype=np.int32)
    if allow_zero:
        # Stars-and-bars; zero-row devices are allowed.
        cuts = combinations(range(height + devices - 1), devices - 1)
        parts = (
            [edge[i + 1] - edge[i] - 1 for i in range(devices)]
            for bars in cuts
            for edge in [(-1, *bars, height + devices - 1)]
        )
    else:
        cuts = combinations(range(1, height), devices - 1)
        parts = (
            [edge[i + 1] - edge[i] for i in range(devices)]
            for bars in cuts
            for edge in [(0, *bars, height)]
        )
    return np.asarray(list(parts), dtype=np.int32)


class Simulator:
    def __init__(self, case, cfg):
        self.case = case
        self.n = case["devices"]
        self.layers = build_layers(case["spec"], case["input_shape"])
        self.timing = case["spec"]["timing"]
        self.batch_size = int(cfg.get("batch_size", 64))
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.activation_bytes = int(self.timing.get("activation_bytes", 1))
        if self.activation_bytes <= 0:
            raise ValueError("activation_bytes must be positive")
        rate_spec = self.timing["macs_per_us"]
        if isinstance(rate_spec, (int, float)):
            self.rates = np.full(self.n, float(rate_spec))
        else:
            self.rates = np.asarray(rate_spec, dtype=float)
            if self.rates.size != self.n:
                raise ValueError("macs_per_us list must have one value per device")
        if np.any(self.rates <= 0):
            raise ValueError("Device MAC rates must be positive")
        if (
            self.timing["packet_bytes"] <= 0
            or self.timing["link_bytes_per_us"] <= 0
        ):
            raise ValueError("Communication rates/packet sizes must be positive")

        allow_zero = bool(cfg.get("allow_zero_rows", True))
        max_states = int(cfg.get("max_states_per_layer", 50000))
        self.states = {
            height: row_partitions(height, self.n, allow_zero, max_states)
            for height in sorted({layer.out_h for layer in self.layers})
        }
        sync = float(self.timing["sync_us"]) if self.n > 1 else 0.0
        self.local_cost = []
        for layer in self.layers:
            states = self.states[layer.out_h]
            cost = self.compute_us(layer, states)
            if layer.op == "residual":
                cost = cost + self.residual_halo_us(layer, states) + 2 * sync
            else:
                cost = cost + sync
            self.local_cost.append(cost)

    def transfer_us(self, nbytes):
        p = self.timing
        return np.where(
            nbytes > 0,
            np.ceil(nbytes / p["packet_bytes"]) * p["packet_overhead_us"]
            + nbytes / p["link_bytes_per_us"],
            0.0
        )

    @staticmethod
    def spans(states):
        end = np.cumsum(states, axis=1)
        return end - states, end

    def comm_matrix(self, layer, previous, current):
        """Costs for source partitions x destination partitions (a small batch)."""
        ps, pe = self.spans(previous)
        qs, qe = self.spans(current)
        req_s = np.clip(
            qs * layer.stride - layer.padding, 0, layer.in_h
        )
        req_e = np.clip(
            (qe - 1) * layer.stride - layer.padding + layer.kernel,
            0, layer.in_h
        )
        # Empty destination partitions must request no input halo rows.
        req_s = np.where(current > 0, req_s, 0)
        req_e = np.where(current > 0, req_e, 0)
        cost = np.zeros((len(previous), len(current)), dtype=float)
        for src in range(self.n):
            for dst in range(self.n):
                if src == dst:
                    continue
                overlap = np.maximum(
                    0,
                    np.minimum(pe[:, src, None], req_e[None, :, dst])
                    - np.maximum(ps[:, src, None], req_s[None, :, dst])
                )
                cost += self.transfer_us(
                    overlap * layer.in_w * layer.in_c * self.activation_bytes
                )
        return cost

    def residual_halo_us(self, layer, states):
        """Internal exchange for the second 3x3 convolution of a block."""
        starts, ends = self.spans(states)
        req_s = np.where(states > 0, np.maximum(0, starts - 1), 0)
        req_e = np.where(
            states > 0, np.minimum(layer.out_h, ends + 1), 0
        )
        cost = np.zeros(len(states), dtype=float)
        for src in range(self.n):
            for dst in range(self.n):
                if src == dst:
                    continue
                overlap = np.maximum(
                    0,
                    np.minimum(ends[:, src], req_e[:, dst])
                    - np.maximum(starts[:, src], req_s[:, dst])
                )
                cost += self.transfer_us(
                    overlap * layer.out_w * layer.out_c * self.activation_bytes
                )
        return cost

    def compute_us(self, layer, states):
        l = layer
        setup = float(self.timing["setup_us"])
        rows = states.astype(float)

        def stage(macs_per_row, include_setup=True):
            overhead = setup if include_setup else 0.0
            # Idle devices have no per-layer setup charge.
            per_device = np.where(
                rows > 0, overhead + rows * macs_per_row / self.rates, 0.0
            )
            return np.max(per_device, axis=1)

        if l.op == "residual":
            first = l.out_w * l.in_c * l.out_c * 9
            second = l.out_w * l.out_c * l.out_c * 9
            main_path = stage(first) + stage(second)
            if l.stride != 1 or l.in_c != l.out_c:
                projection = l.out_w * l.in_c * l.out_c
                main_path = np.maximum(main_path, stage(projection))
            return main_path + stage(l.out_w * l.out_c, include_setup=False)
        if l.op == "standard":
            macs_per_row = l.out_w * l.in_c * l.out_c * l.kernel**2
        elif l.op == "depthwise":
            macs_per_row = l.out_w * l.out_c * l.kernel**2
        elif l.op == "pointwise":
            macs_per_row = l.out_w * l.in_c * l.out_c
        elif l.op == "pool":
            macs_per_row = l.out_w * l.out_c * l.kernel**2
        else:
            raise ValueError(f"Unknown operation: {l.op}")
        return stage(macs_per_row)

    def terminal_cost(self, classifier):
        last = self.layers[-1]
        states = self.states[last.out_h]
        row_bytes = last.out_w * last.out_c * self.activation_bytes
        gather = np.zeros(len(states), dtype=float)
        # Entire classifier runs on device 0.
        for dev in range(1, self.n):
            gather += self.transfer_us(states[:, dev] * row_bytes)

        if classifier["type"] == "fc":
            flattened = last.out_h * last.out_w * last.out_c
            dims = [flattened] + classifier["hidden_and_output"]
            macs = sum(a * b for a, b in zip(dims[:-1], dims[1:]))
        elif classifier["type"] == "gap_fc":
            macs = (
                last.out_h * last.out_w * last.out_c
                + last.out_c * classifier["classes"]
            )
        else:
            raise ValueError("Unknown classifier type")
        return gather + macs / self.rates[0] + classifier["fixed_us"]

    def solve(self, terminal, k=None):
        """Exact DP if k=None, otherwise a K-state global beam at each layer."""
        if k is not None and k < 1:
            raise ValueError("K must be positive")
        deadline = float(self.timing["deadline_us"])
        previous = np.zeros((1, self.n), dtype=np.int32)
        previous[0, 0] = self.layers[0].in_h
        previous_cost = np.array([0.0], dtype=float)
        retained_states = []
        retained_parents = []
        final_keep = None

        for layer_id, layer in enumerate(self.layers):
            current = self.states[layer.out_h]
            minimum = np.full(len(current), np.inf)
            parents = np.full(len(current), -1, dtype=np.int32)

            for start in range(0, len(current), self.batch_size):
                end = min(len(current), start + self.batch_size)
                transfer = self.comm_matrix(layer, previous, current[start:end])
                candidates = previous_cost[:, None] + transfer
                parent = np.argmin(candidates, axis=0)
                minimum[start:end] = (
                    candidates[parent, np.arange(end - start)]
                    + self.local_cost[layer_id][start:end]
                )
                parents[start:end] = parent

            keep = np.flatnonzero(minimum <= deadline)
            if keep.size == 0:
                return None
            if k is not None and keep.size > k:
                rank = np.argsort(minimum[keep], kind="stable")
                keep = keep[rank[:k]]

            retained_states.append(current[keep])
            retained_parents.append(parents[keep])
            previous = current[keep]
            previous_cost = minimum[keep]
            final_keep = keep

        complete_cost = previous_cost + terminal[final_keep]
        feasible = np.flatnonzero(complete_cost <= deadline)
        if feasible.size == 0:
            return None
        position = int(feasible[np.argmin(complete_cost[feasible])])
        cost_us = float(complete_cost[position])

        path = [None] * len(self.layers)
        for layer_id in range(len(self.layers) - 1, -1, -1):
            path[layer_id] = retained_states[layer_id][position].tolist()
            position = int(retained_parents[layer_id][position])
        return cost_us / 1000.0, self.describe_path(path)

    def describe_path(self, path):
        result = []
        for layer, rows in zip(self.layers, path):
            cut_points = [0]
            for count in rows:
                cut_points.append(cut_points[-1] + count)
            assignments = [
                {
                    "device": i,
                    "start": cut_points[i],
                    "end": cut_points[i + 1],
                    "rows": rows[i]
                }
                for i in range(self.n)
            ]
            result.append({
                "layer": layer.name,
                "output_shape": [layer.out_h, layer.out_w, layer.out_c],
                "partition_points": cut_points,
                "device_assignment": assignments
            })
        return result


def timed_run(sim, terminal, k=None):
    start = perf_counter()
    solution = sim.solve(terminal, k=k)
    solve_ms = 1000.0 * (perf_counter() - start)
    return solution, solve_ms


def make_row(case, algorithm, k, solution, solve_ms, precompute_ms):
    return {
        "case_id": case["case_id"],
        "model": case["model"], "mode": case["mode"],
        "classifier": case["classifier"], "algorithm": algorithm,
        "K": "ALL" if k is None else k,
        "input_h": case["input_shape"][0],
        "input_w": case["input_shape"][1],
        "devices": case["devices"],
        "deadline_us": case["spec"]["timing"]["deadline_us"],
        "feasible": int(solution is not None),
        "inference_ms": "" if solution is None else solution[0],
        "solve_time_ms": solve_ms,
        "precompute_ms": precompute_ms,
        "partitions_json": "" if solution is None else json.dumps(
            solution[1], separators=(",", ":")
        )
    }


def show_solution(row, verbose=False):
    name = f"{row['model']} / {row['mode']} / {row['classifier']}"
    if not row["feasible"]:
        print(f"{name}: {row['algorithm']} K={row['K']} INFEASIBLE; "
              f"solve={row['solve_time_ms']:.3f} ms", flush=True)
        return
    print(
        f"{name}: {row['algorithm']} K={row['K']} "
        f"inference={row['inference_ms']:.3f} ms, "
        f"solve={row['solve_time_ms']:.3f} ms", flush=True
    )
    if verbose:
        for layer in json.loads(row["partitions_json"]):
            allocation = " ".join(
                f"D{x['device']}[{x['start']}:{x['end']})"
                for x in layer["device_assignment"]
            )
            print(
                f"  {layer['layer']:22s} "
                f"cuts={layer['partition_points']} {allocation}"
            )


def make_writer(file):
    writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
    writer.writeheader()
    return writer
