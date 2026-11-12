# Three-ESP32 CNN benchmark: NO intermediate halo exchange, with redundant computation

This project is **not the zero-halo/zero-padding approximation**. Its goal is to
obtain final output features without intermediate wireless transfers by receiving
an overlapping/enlarged input region **once** and redundantly calculating the
neighbouring intermediate rows locally. Three ESP32s use the **same firmware**.

## Configuration

Edit `experiment_config.cmake`:

```cmake
set(EXPERIMENT_MODEL "lenet5")        # lenet5 | resnet20 | mobilenetv1_025
set(EXPERIMENT_DATASET "mnist")       # mnist | cifar10
set(EXPERIMENT_RUNS 10)               # 1..250
```

Edit ONLY `main/partition_config.h` for horizontal stage ownership.
The input ranges required by this scheme are **derived automatically by
backward receptive-field expansion of each device's final row ownership**.
The header has `PART_LENET[5][4]`, `PART_RESNET[3][4]`, `PART_MOBILE[6][4]`.
Use the same final stage cuts in the halo project and this project to compare
the final features owned by each device. Earlier-stage computations in this
project intentionally overlap and **do not** follow disjoint halo-stage cuts.

### Default per-device image rows actually sent (half-open)

| Model | Dataset | COM3 | COM7 | COM8 |
|---|---|---:|---:|---:|
| LeNet-5 | MNIST | 0:16 | 4:20 | 8:28 |
| LeNet-5 | CIFAR-10 | 0:20 | 8:28 | 16:32 |
| ResNet-20 | MNIST | 0:28 | 0:28 | 0:28 |
| ResNet-20 | CIFAR-10 | 0:32 | 0:32 | 0:32 |
| MobileNetV1-0.25 | MNIST | 0:28 | 0:28 | 0:28 |
| MobileNetV1-0.25 | CIFAR-10 | 0:32 | 0:32 | 0:32 |

**Why full-image ResNet and MobileNet?** With 28/32-pixel images the deep
network receptive field reaches the input boundaries. In ResNet, early stages
must be computed on the full image to produce an exact partial final feature
stripe, although later stages compute narrower, potentially different ranges.
For MobileNet the final spatial height is 1, and three disjoint nonempty
spatial owners do not exist. Instead this experiment uses complete local
replication of MobileNet, with three identical full-image predictions; it is
**not** a like-for-like spatial-workload comparison with the halo MobileNet.

The overlapped LeNet calculations exactly reconstruct each device's owned
final Pool2 rows, subject to the normal FP32 arithmetic and trained weights.
As with the halo version, each LeNet device still runs its own FC on partial
Pool2 features (unowned rows are zero). ResNet performs GAP on its owned final
Stage-3 feature rows, not on the extra redundant intermediate rows.

## Export trained weights

From the extracted project folder:

```powershell
python tools/export_selected.py --pth "..\models\lenet5_mnist.pth"
```

For different model/dataset selections, pass its `.pth` file, for example
`..\models\resnet20_cifar10.pth`.
The exporter creates the matching weight header inside `variants/`.
Checkpoint files are **not included**; a trained checkpoint must be exported
before the project will run inference.

Optional real test input (otherwise synthetic deterministic input is used):

```powershell
python tools/export_mnist.py --images "C:\path\t10k-images-idx3-ubyte" --index 0
python tools/export_cifar10.py --input "C:\path\cifar-10-batches-bin\test_batch.bin" --index 0
```

## Build and flash: same firmware three times

```powershell
idf.py build
idf.py -p COM7 flash
idf.py -p COM8 flash
idf.py -p COM3 flash
idf.py -p COM3 monitor
```

The MACs are preconfigured from the previously identified three boards:
COM3 `28:05:A5:73:60:40`, COM7 `28:05:A5:73:93:B8`, and
COM8 `28:05:A5:75:35:BC`.

### Terminal output (on COM3)

```text
run 1
execution time of layers (ms): {Conv1: ..., Pool1: ..., Conv2: ..., Pool2: ..., FC1: ..., FC2: ..., FC3: ...}
run 2
execution time of layers (ms): {...}
...
run 10
execution time of layers (ms): {...}
```

Each number is **max(device 0 compute time, device 1 compute time, device 2
compute time)**. Sending overlapping input happens **once before** the first
run and is excluded from per-layer compute timers. No ESP-NOW calls occur
inside any inference kernel; only the final timing arrays are returned.
Any failure prints ERROR instead of fake timing values. Boot ROM messages
may appear independently of the benchmark output.

Save COM3 output to `redundant.log`, then generate plots:

```powershell
python -m pip install matplotlib
python tools/plot_cumulative.py redundant.log
python tools/compare_cumulative.py redundant.log halo.log
```

### Interpreting comparisons

This redundancy-based approach removes **intermediate radio** but adds
**recomputation**. Hence a higher compute-only time than halo exchange is
quite possible, and the tradeoff must consider communication/synchronization
as a separate metric. `sum_l(max_i T[i,l])` is a layer-synchronized proxy,
not necessarily the physical end-to-end latency of unsynchronized pipelines.
The current terminal format does not measure the initial image-transfer time.
To compare total latency, measure initial transfer and end-to-end completion
separately. For the MobileNet full-replication choice, do not claim a strictly
matched per-layer partition comparison against halo exchange.

### Memory limitations

The full-field redundant ResNet requires large FP32 activation buffers:
about 175 KiB of static scratch for MNIST and 224 KiB for CIFAR-10.
ESP32-D0WD SRAM can be tight after Wi-Fi allocation. If ESP-IDF reports
DRAM overflow or insufficient memory, this exact redundant schedule does not
fit in your current memory layout and should be further optimized (e.g.
streaming, in-place buffers, or PSRAM when supported), rather than reducing
the input overlap and silently producing incorrect boundary features.

### Validation

All six C kernels were syntax-checked with GCC and run on a Linux host with
synthetic, deterministic FP32 weights, checking all three device paths.
ESP-IDF linking and real ESP-NOW/hardware latency are **not validated** here.
