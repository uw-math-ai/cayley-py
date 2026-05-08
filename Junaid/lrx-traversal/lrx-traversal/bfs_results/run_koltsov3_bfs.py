#!/usr/bin/env python3
"""
Compute Koltsov3 diameters and longest elements via BFS (cayleypy).

Requires: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (for n>=12)

Usage:
    python run_koltsov3_bfs.py              # run all n=5..12
    python run_koltsov3_bfs.py --n 7 8 9   # specific n values
"""

import argparse, json, os, time
import numpy as np
from cayleypy import CayleyGraphDef, CayleyGraph
from cayleypy.cayley_graph_def import GeneratorType

# =============================================================================
# CLI
# =============================================================================
parser = argparse.ArgumentParser(description='BFS for Koltsov3 diameters and longest elements')
parser.add_argument('--n', type=int, nargs='*', default=None,
                    help='Specific n values (default: 5..12)')
parser.add_argument('--output-dir', default='.',
                    help='Directory to save results')
args = parser.parse_args()

N_VALUES = args.n if args.n else list(range(5, 13))
os.makedirs(args.output_dir, exist_ok=True)

# =============================================================================
# Run BFS for each n
# =============================================================================
for n in N_VALUES:
    print(f'\n{"="*60}')
    print(f'n = {n}')
    print(f'{"="*60}')

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

    print(f'  I = {I}')
    print(f'  K = {K}')
    print(f'  S = {S}')

    # Create Cayley graph and run BFS
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

    diameter = result.diameter()
    n_states = sum(result.layer_sizes)
    last_layer = result.get_layer(diameter)
    longest_elements = [[int(x) for x in s] for s in last_layer]

    print(f'  States:    {n_states:,}')
    print(f'  Diameter:  {diameter}')
    print(f'  Layers:    {len(result.layer_sizes)}')
    print(f'  Last layer size: {len(longest_elements)}')
    print(f'  Time:      {elapsed:.1f}s')

    # Show first few longest elements
    print(f'  Longest elements (first 5):')
    for i, elem in enumerate(longest_elements[:5]):
        print(f'    [{i}] {elem}')

    # Save results to JSON
    output = {
        'n': n,
        'generators': {
            'I': [int(x) for x in I],
            'K': [int(x) for x in K],
            'S': [int(x) for x in S],
            'k': k,
        },
        'n_states': n_states,
        'diameter': diameter,
        'n_layers': len(result.layer_sizes),
        'layer_sizes': result.layer_sizes,
        'n_longest_elements': len(longest_elements),
        'longest_elements': longest_elements,
        'runtime_sec': elapsed,
    }

    out_path = os.path.join(args.output_dir, f'koltsov3_bfs_n{n:02d}.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'  Saved: {out_path}')

# =============================================================================
# Summary table
# =============================================================================
print(f'\n{"="*60}')
print('SUMMARY')
print(f'{"="*60}')
print(f'{"n":>4s}  {"States":>12s}  {"Diameter":>10s}  {"Layers":>8s}  {"Last layer":>12s}  {"Time":>8s}')
for n in N_VALUES:
    path = os.path.join(args.output_dir, f'koltsov3_bfs_n{n:02d}.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f'{n:4d}  {data["n_states"]:>12,}  {data["diameter"]:>10d}  '
              f'{data["n_layers"]:>8d}  {data["n_longest_elements"]:>12,}  '
              f'{data["runtime_sec"]:>7.1f}s')
