#!/usr/bin/env python3
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

HEIGHTS = {'input': 32, 'conv1': 28, 'pool1': 14, 'conv2': 10, 'pool2': 5}
ORDER = ('conv1', 'pool1', 'conv2', 'pool2')
CHANNELS = {'input': 3, 'conv1': 6, 'pool1': 6, 'conv2': 16, 'pool2': 16}
WIDTHS = {'input': 32, 'conv1': 28, 'pool1': 14, 'conv2': 10, 'pool2': 5}


def check_partition(label, rows, height, n=5):
    if not isinstance(rows, list) or len(rows) != n:
        raise ValueError(f'{label} must contain exactly {n} [start,end] pairs')
    previous = 0
    for i, item in enumerate(rows):
        if not isinstance(item, list) or len(item) != 2 or any(type(x) is not int for x in item):
            raise ValueError(f'{label}[{i}] must be [integer_start, integer_end]')
        a, b = item
        if a != previous or b <= a or b > height:
            raise ValueError(f'{label}: output rows must be contiguous, nonoverlapping, nonempty, cover 0:{height}')
        previous = b
    if previous != height:
        raise ValueError(f'{label} must cover all {height} rows')
    return [tuple(x) for x in rows]


def load_config(filename):
    data = json.loads(Path(filename).read_text())
    if data.get('devices') != 5:
        raise ValueError('Only five logical devices are supported')
    partitions = {'input': check_partition('input_rows', data['input_rows'], 32),
                  'independent': check_partition('independent.pool2_rows',
                                                 data['independent']['pool2_rows'], 5)}
    for layer in ORDER:
        partitions[layer] = check_partition(f'halo.{layer}_rows',
                                             data['halo'][f'{layer}_rows'], HEIGHTS[layer])
    extra = data['independent'].get('extra_context_rows', 0)
    if type(extra) is not int or extra < 0 or extra % 4 != 0:
        raise ValueError('independent.extra_context_rows must be a nonnegative multiple of 4')
    if data.get('link_mbps', 0) <= 0 or data.get('message_latency_ms', -1) < 0:
        raise ValueError('link_mbps must be positive and message_latency_ms nonnegative')
    return data, partitions


class DistributedLeNet:
    def __init__(self, net, config):
        self.net = net
        self.cfg, self.rows = load_config(config)
        # Preserve original pretrained model's CONV, pool and FC weights.
        seq = net.net
        self.conv1, self.pool1 = seq[0], seq[2]
        self.conv2, self.pool2 = seq[3], seq[5]
        self.fc = seq[6:]

    def _net_ms(self, nbytes):
        return nbytes * 8 / (float(self.cfg['link_mbps']) * 1e6) * 1000 + float(self.cfg['message_latency_ms'])

    def _transfer(self, target_device, source_device, piece, layer, events, cost):
        """Copy actual tensor data; add one simulated inter-device transfer."""
        if source_device == target_device:
            return piece
        out = piece.clone()
        nbytes = out.numel() * out.element_size()
        cost['bytes'] += nbytes
        cost['messages'] += 1
        cost['comm_sum_ms'] += self._net_ms(nbytes)
        cost['receiver_ms'][target_device] += self._net_ms(nbytes)
        events.append({'layer': layer, 'source': source_device + 1,
                       'destination': target_device + 1, 'bytes': nbytes,
                       'estimated_ms': self._net_ms(nbytes)})
        return out

    def _new_cost(self):
        return {'bytes': 0, 'messages': 0, 'comm_sum_ms': 0.0,
                'receiver_ms': [0.] * 5}

    def _sleep_network(self, cost):
        """Simulate overlappable transfers once per synchronized layer."""
        delay_ms = max(cost['receiver_ms'])
        if self.cfg.get('simulate_transfer_delay', True) and delay_ms > 0:
            time.sleep(delay_ms / 1000.)
        return delay_ms

    def _layer(self, name, tensor):
        if name == 'conv1':
            return torch.tanh(self.conv1(tensor))
        if name == 'pool1':
            return self.pool1(tensor)
        if name == 'conv2':
            return torch.tanh(self.conv2(tensor))
        if name == 'pool2':
            return self.pool2(tensor)
        raise ValueError(name)

    @staticmethod
    def _needed_input(layer, start, stop):
        if layer in ('conv1', 'conv2'):
            return start, stop + 4  # valid 5x5 convolution
        return 2 * start, 2 * stop  # 2x2 pool, stride 2

    def _gather(self, shards, prev_ranges, start, stop, dev, layer, messages, cost):
        """Assemble exactly the needed input range from PREVIOUS layer shards."""
        parts = []
        for src, ((a, b), tensor) in enumerate(zip(prev_ranges, shards)):
            lo, hi = max(start, a), min(stop, b)
            if lo >= hi:
                continue
            view = tensor[:, :, lo-a:hi-a, :]
            parts.append(self._transfer(dev, src, view, layer, messages, cost))
        if sum(v.shape[2] for v in parts) != stop-start:
            raise AssertionError(f'Incomplete halo input for {layer}, device {dev+1}')
        return torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]

    def _common_fc(self, pieces, device_compute, layer_events):
        t = time.perf_counter()
        final = torch.cat(pieces, dim=2)
        scores = self.fc(final)
        fc_ms = (time.perf_counter()-t) * 1000
        device_compute[0] += fc_ms  # device 1 assembles final feature map and FCs
        layer_events.append(dict(device=1, layer='merge_fc', start_row=0, end_row=5,
                                 compute_ms=fc_ms, remote_bytes=0, remote_messages=0,
                                 communication_ms=0.0, input_start=0, input_end=5,
                                 output_elements=scores.numel()))
        return scores, fc_ms

    def execute(self, image, method):
        """image: [1,3,32,32]. Returns exact LeNet logits and measurements."""
        if method not in ('independent', 'halo'):
            raise ValueError('method must be independent or halo')
        if tuple(image.shape) != (1, 3, 32, 32):
            raise ValueError('Expected a single RGB input [1,3,32,32]')
        beginning = time.perf_counter()
        device_compute = [0.] * 5
        messages = []
        layer_events = []
        comm_sum = 0.
        critical_comm = 0.
        phase_cost = self._new_cost()
        phases = []
        independent_spans = []
        # The fused image is constructed at device 1. Unique input strips or
        # independent receptive fields are supplied to other logical devices.
        initial_inputs = []
        if method == 'independent':
            final_ranges = self.rows['independent']
            for dev, (a, b) in enumerate(final_ranges):
                extra = self.cfg['independent'].get('extra_context_rows', 0)
                lo, hi = max(0, 4*a-extra), min(32, 4*b+12+extra)
                independent_spans.append((lo, hi))
                piece = image[:, :, lo:hi, :].contiguous()
                initial_inputs.append(self._transfer(dev, 0, piece, 'input_scatter',
                                                     messages, phase_cost))
        else:
            for dev, (a, b) in enumerate(self.rows['input']):
                piece = image[:, :, a:b, :].contiguous()
                initial_inputs.append(self._transfer(dev, 0, piece, 'input_scatter',
                                                     messages, phase_cost))
        critical_comm += self._sleep_network(phase_cost)
        comm_sum += phase_cost['comm_sum_ms']
        phases.append(('input_scatter', phase_cost['receiver_ms'][:]))

        pieces = []
        if method == 'independent':
            for dev, (a, b) in enumerate(self.rows['independent']):
                lo, hi = independent_spans[dev]
                x = initial_inputs[dev]
                global_start = lo
                for name in ORDER:
                    in_start, in_end = global_start, global_start+x.shape[2]
                    t = time.perf_counter()
                    x = self._layer(name, x)
                    elapsed = (time.perf_counter()-t)*1000
                    # Conv5 keeps coordinate origin, Pool2 halves it. The
                    # expanded input starts on multiples of four, ensuring
                    # exact alignment with the original unpartitioned CNN.
                    if name.startswith('pool'):
                        global_start //= 2
                    out_start, out_end = global_start, global_start+x.shape[2]
                    device_compute[dev] += elapsed
                    layer_events.append(dict(device=dev+1, layer=name,
                                             start_row=out_start, end_row=out_end,
                                             input_start=in_start, input_end=in_end,
                                             compute_ms=elapsed,
                                             remote_bytes=0, remote_messages=0,
                                             communication_ms=0.0,
                                             output_elements=x.numel()))
                # Discard extra overlapping output rows, keeping ONLY the
                # configured final nonoverlapping feature-map segment.
                trimmed = x[:, :, a-global_start:b-global_start, :].contiguous()
                if trimmed.shape[2] != b-a:
                    raise AssertionError('Final independent portion has wrong output height')
                pieces.append(trimmed)
        else:
            previous = initial_inputs
            prev_ranges = self.rows['input']
            for name in ORDER:
                current = []
                cost = self._new_cost()
                assembled = []
                # Gather required input halos from the preceding disjoint
                # output shards, then pay the link delay BEFORE computing.
                for dev, (a, b) in enumerate(self.rows[name]):
                    lo, hi = self._needed_input(name, a, b)
                    x = self._gather(previous, prev_ranges, lo, hi, dev,
                                     name, messages, cost)
                    assembled.append((a, b, lo, hi, x))
                critical_comm += self._sleep_network(cost)
                comm_sum += cost['comm_sum_ms']
                phases.append((name, cost['receiver_ms'][:]))
                for dev, (a, b, lo, hi, x) in enumerate(assembled):
                    t = time.perf_counter()
                    out = self._layer(name, x)
                    elapsed = (time.perf_counter()-t)*1000
                    if out.shape[2] != b-a:
                        raise AssertionError(f'{name} output height wrong on device {dev+1}')
                    device_compute[dev] += elapsed
                    current.append(out)
                    links = [m for m in messages if m['layer'] == name and m['destination'] == dev+1]
                    layer_events.append(dict(device=dev+1, layer=name,
                                             start_row=a, end_row=b,
                                             input_start=lo, input_end=hi,
                                             compute_ms=elapsed,
                                             remote_bytes=sum(m['bytes'] for m in links),
                                             remote_messages=len(links),
                                             communication_ms=sum(m['estimated_ms'] for m in links),
                                             output_elements=out.numel()))
                previous, prev_ranges = current, self.rows[name]
            pieces = previous

        # Merge all five final output pieces at device 1. The portion of
        # every other device is transmitted to the FC device.
        gather_cost = self._new_cost()
        for dev, piece in enumerate(pieces):
            if dev != 0:
                pieces[dev] = self._transfer(0, dev, piece, 'feature_merge',
                                              messages, gather_cost)
        critical_comm += self._sleep_network(gather_cost)
        comm_sum += gather_cost['comm_sum_ms']
        phases.append(('feature_merge', gather_cost['receiver_ms'][:]))

        logits, fc_ms = self._common_fc(pieces, device_compute, layer_events)
        measured_ms = (time.perf_counter()-beginning)*1000
        sum_compute = sum(device_compute)
        # One-model emulation on a single host serially performs computations.
        # An idealized parallel estimate is the max device compute plus phases.
        if method == 'independent':
            # Independent branches run without layer barriers.
            parallel_compute_ms = max(
                sum(e['compute_ms'] for e in layer_events
                    if e['device'] == d and e['layer'] != 'merge_fc')
                for d in range(1, 6)) + fc_ms
        else:
            # Halo creates one synchronization barrier at each stage.
            parallel_compute_ms = sum(
                max(e['compute_ms'] for e in layer_events if e['layer'] == stage)
                for stage in ORDER) + fc_ms
        parallel_ms = parallel_compute_ms + critical_comm
        return {'logits': logits, 'layer_events': layer_events,
                'messages': messages, 'device_compute_ms': device_compute,
                'sum_compute_ms': sum_compute, 'fc_ms': fc_ms,
                'network_bytes': sum(m['bytes'] for m in messages),
                'network_messages': len(messages),
                'network_sum_ms': comm_sum,
                'network_critical_ms': critical_comm,
                'parallel_estimate_ms': parallel_ms,
                'measured_inference_ms': measured_ms}


def verify_equivalence(net, config):
    """Compare BOTH partitioned methods against ordinary LeNet inference."""
    net.eval()
    engine = DistributedLeNet(net, config)
    for seed in range(3):
        torch.manual_seed(seed)
        x = torch.rand(1, 3, 32, 32)
        with torch.inference_mode():
            reference = net(x)
            for method in ('independent', 'halo'):
                out = engine.execute(x, method)
                err = (reference-out['logits']).abs().max().item()
                if not torch.allclose(reference, out['logits'], atol=2e-5, rtol=2e-5):
                    raise AssertionError(f'{method} differs from full LeNet (max abs error={err:.5g})')
    return True
