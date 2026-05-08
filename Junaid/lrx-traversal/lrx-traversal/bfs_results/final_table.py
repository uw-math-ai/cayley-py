#!/usr/bin/env python3
"""Clean table: VRAM & time estimates for Koltsov3 BFS, n=5..16."""

import json, math
import numpy as np

def format_time(sec):
    if sec < 60: return f'{sec:.1f}s'
    elif sec < 3600: return f'{sec/60:.1f}m'
    elif sec < 86400: return f'{sec/3600:.1f}h'
    else: return f'{sec/86400:.1f}d'

def gpu_class(vram_gb):
    if vram_gb < 0.001: return 'Any'
    if vram_gb <= 4: return '4 GB (consumer)'
    if vram_gb <= 8: return '8 GB'
    if vram_gb <= 12: return '12 GB'
    if vram_gb <= 16: return '16 GB'
    if vram_gb <= 24: return '24 GB (RTX 4090)'
    if vram_gb <= 32: return '32 GB'
    if vram_gb <= 48: return '48 GB (A6000)'
    if vram_gb <= 80: return '80 GB (A100/H100)'
    if vram_gb <= 160: return '2× A100'
    if vram_gb <= 320: return '4× A100'
    if vram_gb <= 640: return '8× A100'
    return f'{math.ceil(vram_gb/80)}× A100 cluster'

# Load measured data
data = {}
for n in range(5, 13):
    with open(f'koltsov3_bfs_n{n:02d}.json') as f:
        d = json.load(f)
    data[n] = d

ns = np.array(list(data.keys()))
vrams = np.array([data[n]['peak_vram_allocated_gb'] for n in ns])
times = np.array([data[n]['runtime_sec'] for n in ns])
states = np.array([math.factorial(n) for n in ns])
max_layers = np.array([data[n]['max_layer_size'] for n in ns])

# Fit window: n >= 9 (small n VRAM is noise)
mask = ns >= 9

# === VRAM fits ===
# B: VRAM = a * n! + b
coeff_b = np.polyfit(states[mask], vrams[mask], 1)
# C: VRAM = a * (max_layer * 3 * n * 8/1e9) + b  (peak expand model)
peak_expand = max_layers[mask] * 3 * ns[mask] * 8 / 1e9
coeff_c = np.polyfit(peak_expand, vrams[mask], 1)

# === Time fits ===
# Use n >= 10 for time (n=12 shows non-linear jump)
mask_t = ns >= 10
coeff_t = np.polyfit(ns[mask_t], np.log10(times[mask_t]), 1)

# === Max layer fit ===
coeff_ml = np.polyfit(ns[mask], np.log10(max_layers[mask]), 1)

print("KOLTSOV3 BFS — VRAM & TIME ESTIMATES")
print("=" * 95)
print(f"{'n':>3s}  {'n!':>17s}  {'Measured':>10s}  {'Model B':>10s}  {'Model C':>10s}  {'Estimate':>10s}  {'Time':>10s}  {'GPU Required':>20s}")
print(f"{'':>3s}  {'':>17s}  {'VRAM(GB)':>10s}  {'(n! fit)':>10s}  {'(peak)':>10s}  {'VRAM(GB)':>10s}  {'':>10s}  {'':>20s}")
print("-" * 95)

for n in range(5, 17):
    nfact = math.factorial(n)
    
    if n in data:
        vram_meas = data[n]['peak_vram_allocated_gb']
        time_meas = data[n]['runtime_sec']
        meas_str = f'{vram_meas:>9.4f}'
        time_str = format_time(time_meas)
    else:
        meas_str = '      n/a'
        time_str = ''
    
    # Model predictions
    vram_b = coeff_b[0] * nfact + coeff_b[1]
    
    pred_ml = 10 ** (coeff_ml[0] * n + coeff_ml[1])
    pred_pe = pred_ml * 3 * n * 8 / 1e9
    vram_c = coeff_c[0] * pred_pe + coeff_c[1]
    
    # Average of B and C for final estimate
    vram_est = (vram_b + vram_c) / 2
    
    # Time prediction
    if n <= 12:
        time_est = ''
    else:
        time_pred = 10 ** (coeff_t[0] * n + coeff_t[1])
        time_est = format_time(time_pred)
    
    gpu = gpu_class(vram_est)
    
    print(f'{n:3d}  {nfact:>17,}  {meas_str}  {vram_b:>9.3f}  {vram_c:>9.3f}  {vram_est:>9.3f}  {time_str:>10s}  {gpu:>20s}')

print()
print(f"Model B: VRAM = {coeff_b[0]:.3e} × n! + {coeff_b[1]:.3f}   ({coeff_b[0]*1e9:.1f} bytes/state)")
print(f"Model C: VRAM = {coeff_c[0]:.3f} × peak_theoretical + {coeff_c[1]:.3f}   (batching efficiency: {coeff_c[0]:.0%})")
print(f"Time:    log10(s) = {coeff_t[0]:.3f}·n + {coeff_t[1]:.3f}   (fit on n>=10)")
print(f"Extrapolated values use average of Models B and C.")
print(f"GPU: NVIDIA GTX 1650 SUPER (4 GB). PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for n>=12.")
