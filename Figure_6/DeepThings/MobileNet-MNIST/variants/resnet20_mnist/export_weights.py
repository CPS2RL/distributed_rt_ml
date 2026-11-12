"""Export the exact ResNet20 from train_six_cnns.py into one ESP-IDF header.

Run from the ESP-IDF project root:
    python tools/export_weights.py --pth "../models/resnet20_mnist.pth"

This script accepts a PyTorch .pth checkpoint, NOT a previously generated .h file.
It verifies tensor dimensions and folds each BatchNorm into its preceding conv:
    weight_fused = weight * gamma / sqrt(running_var + eps)
    bias_fused   = beta - running_mean * gamma / sqrt(running_var + eps)

PyTorch conv layout [output, input, ky, kx] is used unchanged in C.
"""
import argparse
import math
from pathlib import Path


def model_from_checkpoint(path):
    if path.suffix.lower() not in ('.pt', '.pth'):
        raise SystemExit('Expected a .pth/.pt checkpoint, not a C .h header. ' 
                         'Use --pth ../models/resnet20_mnist.pth')
    try:
        import torch
    except ImportError as exc:
        raise SystemExit('Install PyTorch: python -m pip install torch') from exc
    if not path.is_file():
        raise SystemExit(f'Checkpoint not found: {path}')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    if not isinstance(state, dict):
        raise SystemExit('Expected a PyTorch state_dict; check the training script')
    if all(key.startswith('module.') for key in state):
        state = {key[7:]: value for key, value in state.items()}
    return state


def tensor(state, name, shape):
    if name not in state:
        raise SystemExit(f'Missing checkpoint tensor: {name}')
    t = state[name].detach().cpu().float().contiguous()
    if tuple(t.shape) != tuple(shape):
        raise SystemExit(f'{name}: expected {shape}, got {tuple(t.shape)}. ' 
                         'You may have trained a different ResNet20 architecture.')
    if not t.isfinite().all().item():
        raise SystemExit(f'{name} contains NaN or infinity')
    return t


def fuse(state, conv_name, bn_name, in_channels, out_channels, kernel):
    """Read conv (bias=False) and associated PyTorch BatchNorm2d state."""
    import torch
    w = tensor(state, conv_name + '.weight', (out_channels, in_channels, kernel, kernel))
    gamma = tensor(state, bn_name + '.weight', (out_channels,))
    beta = tensor(state, bn_name + '.bias', (out_channels,))
    mean = tensor(state, bn_name + '.running_mean', (out_channels,))
    var = tensor(state, bn_name + '.running_var', (out_channels,))
    if (var < 0).any().item():
        raise SystemExit(f'{bn_name} has negative running variance')
    scale = gamma / torch.sqrt(var + 1e-5)  # nn.BatchNorm2d default eps
    fused_w = w * scale[:, None, None, None]
    fused_b = beta - mean * scale
    return fused_w, fused_b


def write_array(fp, name, values):
    vals = values.detach().cpu().reshape(-1).tolist()
    fp.write(f'\nstatic const float {name}[{len(vals)}] = {{\n')
    for i in range(0, len(vals), 8):
        fp.write('    ' + ', '.join(f'{v:.9e}f' for v in vals[i:i+8]) + ',\n')
    fp.write('};\n')
    return name


def export(path, out):
    state = model_from_checkpoint(path)
    specs = []
    # Explicit shape+name enumeration, matching train_six_cnns.py ResNet20.
    for stage, out_ch in ((1, 16), (2, 32), (3, 64)):
        for block in range(3):
            in_ch = 16 if stage == 1 else (16 if stage == 2 and block == 0 else 32 if stage == 2 else 32 if block == 0 else 64)
            # stage1 in=16; stage2 first in=16, others 32; stage3 first in=32, others 64
            prefix = f'layer{stage}.{block}'
            stride = 2 if (stage > 1 and block == 0) else 1
            specs.append((prefix, in_ch, out_ch, stride, stage > 1 and block == 0))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf8') as fp:
        fp.write('/* Auto-generated from PyTorch ResNet20 MNIST, do not edit. */\n')
        fp.write('#pragma once\n#include "distributed_resnet.h"\n')
        sw, sb = fuse(state, 'conv1', 'bn1', 1, 16, 3)
        write_array(fp, 'rw_stem_w', sw)
        write_array(fp, 'rw_stem_b', sb)
        for i, (prefix, ci, co, stride, projection) in enumerate(specs):
            for j in (1, 2):
                w, b = fuse(state, f'{prefix}.conv{j}', f'{prefix}.bn{j}',
                            ci if j == 1 else co, co, 3)
                write_array(fp, f'rw_b{i}_c{j}_w', w)
                write_array(fp, f'rw_b{i}_c{j}_b', b)
            if projection:
                w, b = fuse(state, f'{prefix}.shortcut.0',
                            f'{prefix}.shortcut.1', ci, co, 1)
                write_array(fp, f'rw_b{i}_p_w', w)
                write_array(fp, f'rw_b{i}_p_b', b)

        fc_w = tensor(state, 'fc.weight', (10, 64))
        fc_b = tensor(state, 'fc.bias', (10,))
        write_array(fp, 'rw_fc_w', fc_w)
        write_array(fp, 'rw_fc_b', fc_b)

        fp.write('\nstatic const ResnetWeights g_resnet_weights = {\n')
        fp.write('  .stem = {rw_stem_w, rw_stem_b, 1, 16, 3, 1},\n')
        fp.write('  .blocks = {\n')
        for i, (prefix, ci, co, stride, projection) in enumerate(specs):
            fp.write('    {\n')
            fp.write(f'      .conv1 = {{rw_b{i}_c1_w, rw_b{i}_c1_b, {ci}, {co}, 3, {stride}}},\n')
            fp.write(f'      .conv2 = {{rw_b{i}_c2_w, rw_b{i}_c2_b, {co}, {co}, 3, 1}},\n')
            if projection:
                fp.write(f'      .projection = {{rw_b{i}_p_w, rw_b{i}_p_b, {ci}, {co}, 1, 2}},\n')
            else:
                fp.write('      .projection = {0},\n')
            fp.write('    },\n')
        fp.write('  },\n')
        fp.write('  .fc_weights = rw_fc_w,\n  .fc_bias = rw_fc_b,\n};\n')
    print(f'Created: {out} ({out.stat().st_size / (1024*1024):.2f} MiB of C source)')
    print('BatchNorm was folded into every convolution (including projections).')
    print('No changes to main.c or model_weights.c are required.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pth', type=Path, required=True, help='resnet20_mnist.pth')
    p.add_argument('--out', type=Path, default=Path('main/resnet_weights_generated.h'))
    args = p.parse_args()
    export(args.pth, args.out)


if __name__ == '__main__':
    main()
