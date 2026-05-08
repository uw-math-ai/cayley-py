"""Batch evaluation for beam search — Issue #005.

Runs beam search on multiple target states and beam widths, producing
a CSV of per-run metrics.
"""

import time
from typing import List, Dict, Optional

import pandas as pd
import torch

from beam_search import beam_search, BeamResult
from mlp_model import load_model
from state_expand import build_koltsov3_generator_tensors


def evaluate_beam_search(
    model_path: str,
    target_states: List[List[int]],
    beam_widths: List[int],
    step_limit: int,
    *,
    optimal_lengths: Optional[Dict[tuple, int]] = None,
    device: str = "auto",
) -> pd.DataFrame:
    """Run beam search for each (state, beam_width) combination.

    Args:
        model_path: Path to .pth model file.
        target_states: List of (n,) permutation lists.
        beam_widths: Beam widths to sweep.
        step_limit: Max search iterations per run.
        optimal_lengths: Dict mapping tuple(state) -> optimal path length.
        device: "auto", "cuda", or "cpu".

    Returns:
        DataFrame with columns: n, state_idx, start_state, path_found,
        path_length, steps_taken, beam_width, runtime_sec, path,
        states_visited, [path_vs_optimal_ratio].
    """
    t0 = time.perf_counter()

    # Load model
    model = load_model(model_path, device=device)
    model_device = next(model.parameters()).device

    n = len(target_states[0])
    gens = build_koltsov3_generator_tensors(n)

    rows = []
    for state_idx, state in enumerate(target_states):
        state_t = torch.tensor(state, dtype=torch.int64, device=model_device)
        for bw in beam_widths:
            result = beam_search(
                state_t, model, gens,
                beam_width=bw, step_limit=step_limit,
                deduplicate=True, history_size=32,
            )
            rows.append({
                "n": n,
                "state_idx": state_idx,
                "start_state": str(state),
                "path_found": result.path_found,
                "path_length": result.path_length,
                "steps_taken": result.steps_taken,
                "beam_width": bw,
                "runtime_sec": round(result.runtime_sec, 4),
                "path": str(result.path) if result.path else None,
                "states_visited": result.states_visited,
            })

    df = pd.DataFrame(rows)

    if optimal_lengths is not None:
        df["optimal_length"] = df["start_state"].apply(
            lambda s: optimal_lengths.get(tuple(eval(s)), None)
        )
        mask = df["path_found"] & df["optimal_length"].notna()
        df["path_vs_optimal_ratio"] = None
        df.loc[mask, "path_vs_optimal_ratio"] = (
            df.loc[mask, "path_length"] / df.loc[mask, "optimal_length"]
        )

    return df


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Batch beam search evaluation"
    )
    ap.add_argument("--model", required=True,
                    help="Path to .pth model file")
    ap.add_argument("--states", required=True,
                    help="JSON file with target states (list of lists or "
                         "{\"longest_elements\": [...]})")
    ap.add_argument("--beam-widths", type=int, nargs="+", required=True,
                    help="Beam widths to sweep")
    ap.add_argument("--step-limit", type=int, required=True,
                    help="Max steps per beam search run")
    ap.add_argument("--output", required=True,
                    help="Output CSV path")
    ap.add_argument("--device", default="auto",
                    help="Device: auto, cuda, cpu")
    args = ap.parse_args()

    # Load states
    with open(args.states) as f:
        data = json.load(f)
    if isinstance(data, dict) and "longest_elements" in data:
        target_states = data["longest_elements"]
    elif isinstance(data, list):
        target_states = data
    else:
        raise ValueError(
            f"Unrecognized states file format. "
            f"Expected list or dict with 'longest_elements' key."
        )

    df = evaluate_beam_search(
        model_path=args.model,
        target_states=target_states,
        beam_widths=args.beam_widths,
        step_limit=args.step_limit,
        device=args.device,
    )

    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")
