#!/usr/bin/env python3
"""
Run Koltsov3 BFS for n=5..12 with VRAM monitoring.
Measures: torch.cuda.max_memory_allocated, peak layer size, runtime.
"""

import argparse, json, os, time
import numpy as np
import torch
from cayleypy import CayleyGraphDef, CayleyGraph
from cayleypy.cayley_graph_def import GeneratorType

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, nargs='*', default=list(range(5, 13)),
                    help='n values (default: 5..12)')
parser.add_argument('--output-dir', default='.',
                    help='Output directory')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

results = []

for n in args.n:
    print(f'\n{"="*60}')
    print(f'n = {n}')
    print(f'{"="*60}')

    # Reset CUDA memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()

    # Build Koltsov3 generators
    I = list(range(n))
    for i in range(0, n - 1, 2):
        I[i], I[i + 1] = I[i + 1], I[i]

    K = list(range(n))
    for i in range(1, n - 1, 2):
        K[i], K[i + 1] = K[i + 1], K[i]

    S = list(range(n))
    k = 0
    if k + 2 < n:
        S[k], S[k + 2] = S[k + 2], S[k]

    defn = CayleyGraphDef(
        generators_type=GeneratorType.PERMUTATION,
        generators_permutations=[I, K, S],
        generators_matrices=[],
        generator_names=['I', 'K', 'S'],
        central_state=list(range(n)),
        name=f'koltsov3_n{n}',
    )

    graph = CayleyGraph(defn)
    t0 = time.time()
    result = graph.bfs()
    elapsed = time.time() - t0

    # VRAM measurements
    if torch.cuda.is_available():
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    else:
        peak_allocated = 0
        peak_reserved = 0

    diameter = result.diameter()
    n_states = sum(result.layer_sizes)
    last_layer = result.get_layer(diameter)
    longest_elements = [[int(x) for x in s] for s in last_layer]

    max_layer = max(result.layer_sizes)
    peak_expand_bytes = max_layer * 3 * n * 8  # theoretical peak: all neighbors

    print(f'  States:      {n_states:,}')
    print(f'  Diameter:    {diameter}')
    print(f'  Layers:      {len(result.layer_sizes)}')
    print(f'  Max layer:   {max_layer:,}')
    print(f'  Last layer:  {len(longest_elements)}')
    print(f'  Time:        {elapsed:.1f}s')
    print(f'  Peak VRAM:   {peak_allocated/1e9:.3f} GB allocated, {peak_reserved/1e9:.3f} GB reserved')
    print(f'  Peak expand: {peak_expand_bytes/1e9:.3f} GB (theoretical, pre-batching)')

    # Save JSON
    output = {
        'n': n,
        'generators': {
            'I': [int(x) for x in I],
            'K': [int(x) for x in K],
            'S': [int(x) for x in S],
        },
        'n_states': n_states,
        'diameter': diameter,
        'n_layers': len(result.layer_sizes),
        'layer_sizes': result.layer_sizes,
        'max_layer_size': max_layer,
        'n_longest_elements': len(longest_elements),
        'longest_elements': longest_elements,
        'runtime_sec': elapsed,
        'peak_vram_allocated_gb': peak_allocated / 1e9,
        'peak_vram_reserved_gb': peak_reserved / 1e9,
        'peak_expand_theoretical_gb': peak_expand_bytes / 1e9,
    }
    out_path = os.path.join(args.output_dir, f'koltsov3_bfs_n{n:02d}.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'  Saved: {out_path}')

    results.append(output)

# Summary
print(f'\n{"="*80}')
print('SUMMARY: VRAM & RUNTIME')
print(f'{"="*80}')
print(f'{"n":>4s}  {"States":>14s}  {"Diameter":>9s}  {"MaxLayer":>12s}  {"VRAM(GB)":>9s}  {"PeakTh(GB)":>10s}  {"Time(s)":>7s}')
for r in results:
    print(f'{r["n"]:4d}  {r["n_states"]:>14,}  {r["diameter"]:>9d}  {r["max_layer_size"]:>12,}  '
          f'{r["peak_vram_allocated_gb"]:>8.3f}  {r["peak_expand_theoretical_gb"]:>9.3f}  {r["runtime_sec"]:>6.1f}')
