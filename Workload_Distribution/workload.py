import csv
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


N_DEVICES = 10
INPUT_HEIGHT = 224


DeepThings = (4, 5, 5, 6, 7, 8, 6, 5, 5, 5)

EdgeFlow = {
    "Stem": (25, 23, 23, 23, 22, 22, 22, 22, 21, 21),
    "S1B1": (25, 23, 23, 23, 22, 22, 22, 22, 21, 21),
    "S1B2": (25, 23, 23, 23, 22, 22, 22, 22, 21, 21),
    "S1B3": (25, 23, 23, 23, 22, 22, 22, 22, 21, 21),

    "S2B1": (13, 12, 12, 11, 11, 11, 11, 11, 10, 10),
    "S2B2": (13, 12, 12, 11, 11, 11, 11, 11, 10, 10),
    "S2B3": (13, 12, 12, 11, 11, 11, 11, 11, 10, 10),

    "S3B1": (7, 7, 6, 6, 6, 6, 5, 5, 4, 4),
    "S3B2": (7, 7, 6, 6, 6, 6, 5, 5, 4, 4),
    "S3B3": (7, 7, 6, 6, 6, 6, 5, 5, 4, 4),
}


Proposed = {
    "Stem": (25, 24, 24, 23, 23, 22, 22, 21, 20, 20),
    "S1B1": (25, 24, 24, 23, 23, 22, 22, 21, 20, 20),
    "S1B2": (25, 24, 24, 23, 23, 22, 22, 21, 20, 20),
    "S1B3": (25, 24, 24, 23, 23, 22, 22, 21, 20, 20),
    "S2B1": (12, 12, 12, 12, 11, 11, 11, 11, 10, 10),
    "S2B2": (12, 12, 12, 12, 11, 11, 11, 11, 10, 10),
    "S2B3": (12, 12, 12, 12, 11, 11, 11, 11, 10, 10),
    "S3B1": (6, 6, 6, 6, 6, 6, 5, 5, 5, 5),
    "S3B2": (6, 6, 6, 6, 6, 6, 5, 5, 5, 5),
    "S3B3": (6, 6, 6, 6, 6, 6, 5, 5, 5, 5),
}

# 2. RESNET-20
@dataclass(frozen=True)
class Block:
    name: str
    in_h: int
    in_c: int
    out_h: int
    out_c: int
    stride: int
    stem: bool = False


def make_blocks():
    blocks = [Block("Stem", 224, 3, 224, 16, 1, True)]
    in_h, in_c = 224, 16
    for stage, (height, channels) in enumerate(
        ((224, 16), (112, 32), (56, 64)), start=1
    ):
        for b in range(1, 4):
            stride = 2 if in_h != height else 1
            blocks.append(Block(
                f"S{stage}B{b}", in_h, in_c, height, channels, stride
            ))
            in_h, in_c = height, channels
    return blocks


BLOCKS = make_blocks()
LAYER_NAMES = ["Stem"]
for block in BLOCKS[1:]:
    LAYER_NAMES.extend([f"{block.name}-C1", f"{block.name}-C2"])


def validate_partition(partition, height, label):
    if len(partition) != N_DEVICES or any(int(v) != v or v <= 0 for v in partition):
        raise ValueError(f"{label}: expected {N_DEVICES} positive integer row counts")
    if sum(partition) != height:
        raise ValueError(f"{label}: row counts sum to {sum(partition)}, not {height}")


def validate_all():
    validate_partition(DeepThings, BLOCKS[-1].out_h, "DeepThings")
    for strategy, partitions in (
        ("EdgeFlow", EdgeFlow),
        ("Proposed", Proposed),
    ):
        if set(partitions) != {b.name for b in BLOCKS}:
            raise ValueError(f"{strategy}: specify one partition for each block")
        for block in BLOCKS:
            validate_partition(partitions[block.name], block.out_h,
                               f"{strategy}[{block.name}]")


# 3. WORKLOAD PER OUTPUT ROW 
def ops_per_row(block):
    conv1 = block.out_h * block.out_c * block.in_c * 9
    if block.stem:
        return conv1, 0, 0
    conv2 = block.out_h * block.out_c * block.out_c * 9
    addition = block.out_h * block.out_c
    return conv1, conv2, addition


# 4. NO REDUNDANT CONV ROWS
def overlap_workload(partitions):
    work = np.zeros((N_DEVICES, len(LAYER_NAMES)), dtype=np.float64)
    for i, block in enumerate(BLOCKS):
        rows = np.asarray(partitions[block.name], dtype=np.float64)
        conv1, conv2, addition = ops_per_row(block)
        if block.stem:
            work[:, 0] = rows * conv1
        else:
            j = 1 + 2 * (i - 1)
            work[:, j] = rows * conv1
            work[:, j + 1] = rows * (conv2 + addition)
    return work


# 5. EXPAND RECEPTIVE FIELDS BACK THROUGH THE NETWORK
def intervals(partition):
    edges = np.cumsum((0, *partition))
    return [(int(edges[d]), int(edges[d+1])) for d in range(N_DEVICES)]


def required_input_rows(output_interval, in_height, stride):
    a, b = output_interval
    if a >= b:
        return (0, 0)
    # 3x3 kernel, padding 1
    return (max(0, a * stride - 1),
            min(in_height, (b - 1) * stride + 2))


def no_overlap_workload(final_partition):
    required = intervals(final_partition)
    block_outputs = [None] * len(BLOCKS)
    intermediate_outputs = [None] * len(BLOCKS)

    for i in reversed(range(len(BLOCKS))):
        block = BLOCKS[i]
        block_outputs[i] = required
        if block.stem:
            required = [required_input_rows(r, block.in_h, 1) for r in required]
        else:
            # Conv2 requires a 1-row overlap on its Conv1 output.
            intermediate = [
                required_input_rows(r, block.out_h, 1) for r in required
            ]
            intermediate_outputs[i] = intermediate
            # Conv1 in turn requires a overlap from the previous block.
            required = [
                required_input_rows(r, block.in_h, block.stride)
                for r in intermediate
            ]

    work = np.zeros((N_DEVICES, len(LAYER_NAMES)), dtype=np.float64)
    for i, block in enumerate(BLOCKS):
        conv1, conv2, addition = ops_per_row(block)
        for d in range(N_DEVICES):
            output_rows = block_outputs[i][d][1] - block_outputs[i][d][0]
            if block.stem:
                work[d, 0] = output_rows * conv1
            else:
                conv1_rows = (
                    intermediate_outputs[i][d][1] - intermediate_outputs[i][d][0]
                )
                j = 1 + 2 * (i - 1)
                work[d, j] = conv1_rows * conv1
                work[d, j + 1] = output_rows * (conv2 + addition)
    return work




def save_csv(datasets):
    with open("workloads.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Strategy", "Layer", "Device", "Operations"])
        for title, matrix, _ in datasets:
            for j, layer in enumerate(LAYER_NAMES):
                for d in range(N_DEVICES):
                    writer.writerow([title, layer, f"D{d+1}", int(matrix[d, j])])


def main():
    validate_all()
    no_overlap = no_overlap_workload(DeepThings)
    nonoptimal = overlap_workload(EdgeFlow)
    optimal = overlap_workload(Proposed)
    datasets = [
        ("DeepThings", no_overlap, "workloadsno_overlap"),
        ("EdgeFlow", nonoptimal,
         "workloadsEdgeFlow"),
        ("Proposed", optimal, "workloadsProposed"),
    ]
    save_csv(datasets)
    print("Saved workloads.csv")


if __name__ == "__main__":
    main()
