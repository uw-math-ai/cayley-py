#!/usr/bin/env python3
"""Fit VRAM and runtime models to Koltsov3 BFS data, extrapolate to higher n."""

import json, math, os
import numpy as np

def format_time(sec):
    if sec < 60:
        return f'{sec:.1f}s'
    elif sec < 3600:
        return f'{sec/60:.1f}m'
    elif sec < 86400:
        return f'{sec/3600:.1f}h'
    else:
        return f'{sec/86400:.1f}d'

def gpu_class(vram_gb):
    if vram_gb < 4:
        return 'Any (<=4GB)'
    elif vram_gb < 8:
        return '8GB (RTX 3070)'
    elif vram_gb < 12:
        return '12GB (RTX 3080 Ti)'
    elif vram_gb < 16:
        return '16GB (RTX 4080)'
    elif vram_gb < 24:
        return '24GB (RTX 4090)'
    elif vram_gb < 32:
        return '32GB (A100 40GB)'
    elif vram_gb < 48:
        return '48GB (A6000)'
    elif vram_gb < 80:
        return '80GB (A100/H100)'
    else:
        return f'>{vram_gb:.0f}GB (multi-GPU)'

# Load data
data = []
for n in range(5, 13):
    path = f'koltsov3_bfs_n{n:02d}.json'
    with open(path) as f:
        d = json.load(f)
    data.append(d)

print("=" * 100)
print("RAW DATA")
print("=" * 100)
print(f"{'n':>4s}  {'n!':>16s}  {'States':>14s}  {'MaxLayer':>12s}  {'PeakExpandMB':>14s}  {'VRAM_GB':>9s}  {'Time_s':>7s}  {'log10(States)':>13s}  {'log10(VRAM)':>11s}")
for d in data:
    n = d['n']
    nfact = math.factorial(n)
    ml = d['max_layer_size']
    pe_mb = ml * 3 * n * 8 / 1e6
    vram = d['peak_vram_allocated_gb']
    t = d['runtime_sec']
    print(f'{n:4d}  {nfact:>16,}  {d["n_states"]:>14,}  {ml:>12,}  {pe_mb:>14.1f}  {vram:>8.4f}  {t:>6.1f}  {math.log10(d["n_states"]):>12.4f}  {math.log10(max(vram,1e-6)):>10.4f}')

# =============================================================================
# Fit models
# =============================================================================
ns = np.array([d['n'] for d in data], dtype=float)
states = np.array([d['n_states'] for d in data], dtype=float)
max_layers = np.array([d['max_layer_size'] for d in data], dtype=float)
vrams = np.array([d['peak_vram_allocated_gb'] for d in data], dtype=float)
times = np.array([d['runtime_sec'] for d in data], dtype=float)
peak_expand_gb = max_layers * 3 * ns * 8 / 1e9

# For small n, VRAM is too small - use n >= 9 for fits
mask_fit = ns >= 9

print("\n" + "=" * 100)
print("MODEL FITS (using n >= 9 for VRAM, all n for time)")
print("=" * 100)

# --- Model A: log(VRAM) vs n ---
coeffs_a = np.polyfit(ns[mask_fit], np.log10(vrams[mask_fit]), 1)
print(f"\n[A] log10(VRAM) = {coeffs_a[0]:.4f} * n + {coeffs_a[1]:.4f}")
print(f"    → VRAM ≈ 10^({coeffs_a[1]:.4f}) × 10^({coeffs_a[0]:.4f}×n)")

# --- Model B: VRAM vs n! (linear) ---
coeffs_b = np.polyfit(states[mask_fit], vrams[mask_fit], 1)
bytes_per_state = coeffs_b[0] * 1e9  # convert GB/state to bytes/state
print(f"\n[B] VRAM(GB) = {coeffs_b[0]:.6e} × n! + {coeffs_b[1]:.4f}")
print(f"    → {bytes_per_state:.2f} bytes per state + {coeffs_b[1]*1e3:.1f} MB overhead")

# --- Model C: VRAM vs peak expand theoretical ---
coeffs_c = np.polyfit(peak_expand_gb[mask_fit], vrams[mask_fit], 1)
print(f"\n[C] VRAM(GB) = {coeffs_c[0]:.4f} × peak_expand_theoretical + {coeffs_c[1]:.4f}")
print(f"    → Batching efficiency: {coeffs_c[0]:.2%} of theoretical peak memory used")

# --- Runtime model: log(time) vs n ---
coeffs_t = np.polyfit(ns, np.log10(times.clip(0.01)), 1)
print(f"\n[TIME] log10(Time) = {coeffs_t[0]:.4f} * n + {coeffs_t[1]:.4f}")
print(f"    → Time ≈ 10^({coeffs_t[1]:.4f}) × 10^({coeffs_t[0]:.4f}×n)")

# --- Runtime model: time vs total operations (n! * n * generators) ---
ops = states * ns * 3  # total state-generator applications (proxy for work)
coeffs_t2 = np.polyfit(ops, times, 1)
print(f"\n[TIME2] Time(s) = {coeffs_t2[0]:.3e} × n!×n×3 + {coeffs_t2[1]:.4f}")
print(f"    → {coeffs_t2[0]*1e9:.4f} ns per state-generator application")

# =============================================================================
# Extrapolation to n=13..16
# =============================================================================
print("\n" + "=" * 100)
print("EXTRAPOLATION TO HIGHER n")
print("=" * 100)

future_ns = [13, 14, 15, 16]

# Estimate max_layer for higher n using scaling of max_layer_size vs n
# max_layer_size grows roughly like n! / (something)
# Fit log10(max_layer) vs n
coeffs_ml = np.polyfit(ns[mask_fit], np.log10(max_layers[mask_fit]), 1)
print(f"\nMax layer fit: log10(max_layer) = {coeffs_ml[0]:.4f} * n + {coeffs_ml[1]:.4f}")

results = []
for n in future_ns:
    nfact = math.factorial(n)
    # Model A prediction
    vram_a = 10 ** (coeffs_a[0] * n + coeffs_a[1])
    # Model B prediction
    vram_b = coeffs_b[0] * nfact + coeffs_b[1]
    # Model C: first predict max_layer, then peak_expand, then VRAM
    pred_max_layer = 10 ** (coeffs_ml[0] * n + coeffs_ml[1])
    pred_peak_expand = pred_max_layer * 3 * n * 8 / 1e9
    vram_c = coeffs_c[0] * pred_peak_expand + coeffs_c[1]
    # Time prediction
    time_pred = 10 ** (coeffs_t[0] * n + coeffs_t[1])
    # Time via operations
    time_pred2 = coeffs_t2[0] * nfact * n * 3 + coeffs_t2[1]
    
    results.append({
        'n': n, 'nfact': nfact,
        'vram_a': vram_a, 'vram_b': vram_b, 'vram_c': vram_c,
        'pred_max_layer': pred_max_layer,
        'time_log': time_pred, 'time_ops': time_pred2,
        'pred_peak_expand': pred_peak_expand,
    })

# Print extrapolation table
print(f"\n{'n':>4s}  {'n!':>18s}  {'PredMaxLayer':>16s}  {'VRAM_A(GB)':>12s}  {'VRAM_B(GB)':>12s}  {'VRAM_C(GB)':>12s}  {'Time_A':>10s}  {'Time_ops':>10s}")
print("-" * 120)
for r in results:
    time_str = format_time(r['time_log'])
    time_ops_str = format_time(r['time_ops'])
    print(f'{r["n"]:4d}  {r["nfact"]:>18,}  {r["pred_max_layer"]:>16,.0f}  '
          f'{r["vram_a"]:>11.3f}  {r["vram_b"]:>11.3f}  {r["vram_c"]:>11.3f}  '
          f'{time_str:>10s}  {time_ops_str:>10s}')

# =============================================================================
# Final consolidated table
# =============================================================================
print("\n" + "=" * 100)
print("FINAL TABLE: VRAM & TIME ESTIMATES FOR KOLTSOV3 BFS")
print("=" * 100)
print(f"\n{'n':>4s}  {'n!':>16s}  {'VRAM (GB)':>11s}  {'Time':>12s}  {'GPU needed':>20s}")
print("-" * 80)

# Use model B (states-based) as primary since it's most interpretable
for d in data:
    n = d['n']
    nfact = math.factorial(n)
    vram = d['peak_vram_allocated_gb']
    t = d['runtime_sec']
    gpu = gpu_class(vram)
    print(f'{n:4d}  {nfact:>16,}  {vram:>10.4f}  {format_time(t):>12s}  {gpu:>20s}')

for r in results:
    # Average of models B and C (both are empirical)
    vram_avg = (r['vram_b'] + r['vram_c']) / 2
    time_avg = (r['time_log'] + r['time_ops']) / 2
    gpu = gpu_class(vram_avg)
    print(f'{r["n"]:4d}  {r["nfact"]:>16,}  {vram_avg:>10.3f}  {format_time(time_avg):>12s}  {gpu:>20s}')

print(f"\nNote: Estimates for n>=13 use average of models B and C.")
print(f"VRAM is measured (n=5..12) or estimated (n=13..16) peak allocated GPU memory.")
print(f"GPU: GTX 1650 SUPER (4GB). Time measured on this GPU.")

# =============================================================================
# FLOPS proxy: count integer operations
# =============================================================================
print("\n" + "=" * 100)
print("MEMORY BANDWIDTH / OPERATION COUNTS (Proxy for compute cost)")
print("=" * 100)
print(f"\n{'n':>4s}  {'States':>14s}  {'TotalOps (n!×n×3)':>20s}  {'BytesRead (est)':>16s}  {'BW(GB/s)':>10s}")
for d in data:
    n = d['n']
    nfact = math.factorial(n)
    # Each state-generator application:
    # - gather: reads n int64s, writes n int64s → 16n bytes
    # - hash: reads n int64s, writes 1 int64 → 8(n+1) bytes
    # - per layer, do this for all states × 3 generators
    # Total bytes moved ≈ layers × avg_layer × 3 × (16n + 8(n+1))
    # Approximate: total_states × 3 × 24n bytes
    total_bytes = nfact * 3 * 24 * n  # rough
    bw = total_bytes / (d['runtime_sec'] + 0.001) / 1e9
    ops_count = nfact * n * 3
    print(f'{n:4d}  {nfact:>14,}  {ops_count:>20,.0f}  {total_bytes/1e9:>15.3f}  {bw:>9.1f}')

