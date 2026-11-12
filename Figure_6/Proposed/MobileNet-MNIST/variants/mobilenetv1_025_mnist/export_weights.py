"""Export MobileNetV1-0.25 trained using train_six_cnns.py.

Usage: python tools/export_weights.py --pth ../models/mobilenetv1_025_mnist.pth
       python tools/export_weights.py --pth ../models/mobilenetv1_025_cifar10.pth

Only .pth/.pt torch state_dict checkpoints are accepted. BatchNorm is folded
into every preceding convolution including depthwise and pointwise convolutions.
"""
import argparse
from pathlib import Path

DATASET = "MNIST"
INPUT_CHANNELS = 1
CONFIG = [
    (16, 1), (32, 2), (32, 1), (64, 2), (64, 1),
    (128, 2), (128, 1), (128, 1), (128, 1), (128, 1),
    (128, 1), (256, 2), (256, 1),
]


def get_checkpoint(path):
    if path.suffix.lower() not in {'.pth', '.pt'}:
        raise SystemExit('Expected a PyTorch .pth/.pt file, not a .h file')
    if not path.is_file():
        raise SystemExit(f'Checkpoint not found: {path}')
    try:
        import torch
    except ImportError:
        raise SystemExit('Install torch: python -m pip install torch')
    s = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(s, dict):
        for field in ('state_dict', 'model_state_dict'):
            if field in s and isinstance(s[field], dict):
                s = s[field]
    if not isinstance(s, dict):
        raise SystemExit('Expected a state_dict; verify your checkpoint')
    if s and all(k.startswith('module.') for k in s):
        s = {k[7:]: v for k, v in s.items()}
    return s


def take(s, key, shape):
    if key not in s:
        raise SystemExit(f'Missing checkpoint tensor: {key}')
    t = s[key].detach().cpu().float().contiguous()
    if tuple(t.shape) != tuple(shape):
        raise SystemExit(f'{key}: expected {shape}, found {tuple(t.shape)}')
    if not t.isfinite().all().item():
        raise SystemExit(f'{key}: NaN or infinity found')
    return t


def fuse(s, conv_key, bn_key, in_ch, out_ch, kernel, depthwise=False):
    import torch
    shape = (out_ch, 1 if depthwise else in_ch, kernel, kernel)
    w = take(s, conv_key + '.weight', shape)
    gamma = take(s, bn_key + '.weight', (out_ch,))
    beta = take(s, bn_key + '.bias', (out_ch,))
    mean = take(s, bn_key + '.running_mean', (out_ch,))
    var = take(s, bn_key + '.running_var', (out_ch,))
    if (var < 0).any().item():
        raise SystemExit(f'{bn_key}: invalid running variance')
    k = gamma / torch.sqrt(var + 1e-5)
    return w * k[:, None, None, None], beta - mean * k


def array(f, name, t):
    vals = t.reshape(-1).tolist()
    f.write(f'\nstatic const float {name}[{len(vals)}] = {{\n')
    for i in range(0, len(vals), 8):
        f.write('    ' + ', '.join(f'{v:.9e}f' for v in vals[i:i+8]) + ',\n')
    f.write('};\n')


def export(pth, out):
    s = get_checkpoint(pth)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf8') as f:
        f.write(f'/* Auto-generated {DATASET} MobileNetV1-0.25 BN-fused weights. */\n')
        f.write('#pragma once\n#include "distributed_mobilenet.h"\n')
        stem_w, stem_b = fuse(s, 'stem.0', 'stem.1', INPUT_CHANNELS, 8, 3)
        array(f, 'mw_stem_w', stem_w)
        array(f, 'mw_stem_b', stem_b)
        ci = 8
        for i, (co, stride) in enumerate(CONFIG):
            dw_w, dw_b = fuse(s, f'features.{i}.0', f'features.{i}.1',
                              ci, ci, 3, depthwise=True)
            pw_w, pw_b = fuse(s, f'features.{i}.3', f'features.{i}.4',
                              ci, co, 1)
            array(f, f'mw_{i}_dw_w', dw_w)
            array(f, f'mw_{i}_dw_b', dw_b)
            array(f, f'mw_{i}_pw_w', pw_w)
            array(f, f'mw_{i}_pw_b', pw_b)
            ci = co
        array(f, 'mw_fc_w', take(s, 'fc.weight', (10, 256)))
        array(f, 'mw_fc_b', take(s, 'fc.bias', (10,)))
        f.write('\nstatic const MobileWeights g_mobilenet_weights = {\n')
        f.write(f'    .stem = {{mw_stem_w, mw_stem_b, {INPUT_CHANNELS}, 8, 3, 2, 0}},\n')
        f.write('    .blocks = {\n')
        ci = 8
        for i, (co, stride) in enumerate(CONFIG):
            f.write('        {\n')
            f.write(f'            .depthwise = {{mw_{i}_dw_w, mw_{i}_dw_b, {ci}, {ci}, 3, {stride}, 1}},\n')
            f.write(f'            .pointwise = {{mw_{i}_pw_w, mw_{i}_pw_b, {ci}, {co}, 1, 1, 0}},\n')
            f.write('        },\n')
            ci = co
        f.write('    },\n')
        f.write('    .fc_weights = mw_fc_w,\n')
        f.write('    .fc_bias = mw_fc_b,\n};\n')
    print(f'Exported {DATASET} trained weights: {out} ({out.stat().st_size / 1048576:.2f} MiB of C text)')
    print('BatchNorm folded into conv/depthwise/pointwise; original layer strides preserved.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pth', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('main/mobilenet_weights_generated.h'))
    args = p.parse_args()
    export(args.pth, args.out)


if __name__ == '__main__':
    main()
