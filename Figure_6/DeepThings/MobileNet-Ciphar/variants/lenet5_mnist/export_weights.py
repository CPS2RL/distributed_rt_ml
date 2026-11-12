"""Export lenet5_mnist.pth from train_six_cnns.py directly into ESP-IDF.

PowerShell example (run from ESP-IDF project root):
    python tools/export_weights.py --pth C:\\path\\to\\models\\lenet5_mnist.pth

Produces main/model_weights_generated.h; no retraining or external upload.
"""
import argparse
from pathlib import Path
import os
import tempfile

import torch

EXPECTED = (
    ("c1_w", "conv1.weight", (6, 1, 5, 5)),
    ("c1_b", "conv1.bias", (6,)),
    ("c2_w", "conv2.weight", (16, 6, 5, 5)),
    ("c2_b", "conv2.bias", (16,)),
    ("f1_w", "fc1.weight", (120, 256)),
    ("f1_b", "fc1.bias", (120,)),
    ("f2_w", "fc2.weight", (84, 120)),
    ("f2_b", "fc2.bias", (84,)),
    ("f3_w", "fc3.weight", (10, 84)),
    ("f3_b", "fc3.bias", (10,)),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pth", type=Path, required=True,
                    help="Path to trained lenet5_mnist.pth state_dict")
    ap.add_argument("--out", type=Path, default=Path("main/model_weights_generated.h"))
    args = ap.parse_args()

    if not args.pth.is_file():
        ap.error(f"Missing model file: {args.pth}")
    state = torch.load(args.pth, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        ap.error("Expected a PyTorch state_dict")
    unexpected = set(state) - {k for _, k, _ in EXPECTED}
    if unexpected:
        raise ValueError(f"Unexpected keys: {sorted(unexpected)}. Use the training architecture from train_six_cnns.py")

    values = []
    total = 0
    for cname, key, shape in EXPECTED:
        if key not in state:
            raise ValueError(f"Checkpoint is missing {key}")
        v = state[key]
        if tuple(v.shape) != shape:
            raise ValueError(f"{key}: expected shape {shape}, got {tuple(v.shape)}")
        v = v.detach().to(dtype=torch.float32, device="cpu").contiguous().flatten()
        if not torch.isfinite(v).all().item():
            raise ValueError(f"{key} has nonfinite values")
        values.append((cname, key, shape, v.tolist()))
        total += v.numel()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Use a temporary file to avoid leaving an incomplete header on failure.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".h",
                                     prefix="weights_", dir=args.out.parent,
                                     delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write("/* Generated from lenet5_mnist.pth; do not edit. */\n")
            f.write("#pragma once\n#include \"distributed_lenet.h\"\n")
            f.write('#define MODEL_SOURCE_DESC "LeNet-5 MNIST trained FP32"\n')
            for cname, key, shape, items in values:
                f.write(f"\n/* {key} shape {shape} / count {len(items)} */\n")
                f.write(f"static const float model_{cname}[{len(items)}] = {{\n")
                for i in range(0, len(items), 8):
                    f.write("    " + ", ".join(f"{float(x):.9e}f" for x in items[i:i+8]) + ",\n")
                f.write("};\n")
            f.write("\nstatic const LenetWeights g_lenet_weights = {\n")
            for stem in ("c1", "c2", "f1", "f2", "f3"):
                f.write(f"    .{stem}_w = model_{stem}_w, .{stem}_b = model_{stem}_b,\n")
            f.write("};\n")
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, args.out)
    print(f"Success: {args.out} ({total:,} FP32 values, {total * 4:,} bytes)")
    print("Architecture: Conv1 1->6, Conv2 6->16, FC1 256->120, FC2 120->84, FC3 84->10")
    print("Tensor layouts: Conv OIHW; FC output-major; final feature flatten HWC")


if __name__ == "__main__":
    main()
