#!/usr/bin/env python3
"""Evaluate a saved Koltsov3 PPO checkpoint on benchmark and holdout hard states."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Michael.lrx_koltsov3_ppo import (
    ActorCritic,
    PPOConfig,
    beam_search_with_policy_prior,
    build_hard_state_benchmark,
    evaluate_hard_state_benchmark,
    generate_fixed_depth_walk_states,
    get_koltsov3_generators,
)


def evaluate_state_set(
    model: ActorCritic,
    generators: np.ndarray,
    states: np.ndarray,
    reference_lengths: np.ndarray,
    *,
    beam_width: int,
    step_limit: int,
    policy_alpha: float,
    history_size: int,
    apply_x_trick: bool,
    device: str,
) -> dict:
    rows = []
    solved_ratios = []
    solved_lengths = []
    solved_runtimes = []

    for index, state in enumerate(states):
        result = beam_search_with_policy_prior(
            state,
            model,
            generators,
            beam_width=beam_width,
            step_limit=step_limit,
            policy_alpha=policy_alpha,
            history_size=history_size,
            apply_x_trick=apply_x_trick,
            device=device,
        )
        reference_length = float(reference_lengths[index])
        ratio = None
        if result.path_found and reference_length > 0:
            ratio = result.path_length / reference_length
            solved_ratios.append(ratio)
            solved_lengths.append(float(result.path_length))
            solved_runtimes.append(result.runtime_sec)
        rows.append(
            {
                "index": index,
                "state": state.tolist(),
                "path_found": result.path_found,
                "path_length": result.path_length,
                "reference_length": reference_length,
                "length_ratio": ratio,
                "runtime_sec": result.runtime_sec,
            }
        )

    success_rate = sum(row["path_found"] for row in rows) / len(rows)
    summary = {
        "success_rate": success_rate,
        "solved_count": int(sum(row["path_found"] for row in rows)),
        "total_count": len(rows),
        "median_length_ratio": float(np.median(solved_ratios)) if solved_ratios else None,
        "median_path_length": float(np.median(solved_lengths)) if solved_lengths else None,
        "median_runtime_sec": float(np.median(solved_runtimes)) if solved_runtimes else None,
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Koltsov3 PPO checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--step-limit", type=int, default=None)
    parser.add_argument("--policy-alpha", type=float, default=None)
    parser.add_argument("--history-size", type=int, default=None)
    parser.add_argument("--apply-x-trick", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--extra-num-states", type=int, default=8)
    parser.add_argument("--extra-walk-length", type=int, default=None)
    parser.add_argument("--extra-seed", type=int, default=999)
    parser.add_argument("--extra-history-size", type=int, default=32)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = PPOConfig(**checkpoint["config"])
    device = args.device if args.device != "auto" else config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ActorCritic(config.n, config.hidden_dim, k=config.k)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    generators = get_koltsov3_generators(config.n, config.k)
    beam_width = args.beam_width or config.eval_beam_width
    step_limit = args.step_limit or config.eval_step_limit
    policy_alpha = args.policy_alpha if args.policy_alpha is not None else config.eval_policy_alpha
    history_size = args.history_size if args.history_size is not None else config.eval_history_size
    apply_x_trick = (
        args.apply_x_trick if args.apply_x_trick is not None else config.eval_apply_x_trick
    )

    benchmark_summary = evaluate_hard_state_benchmark(
        model,
        config,
        generators,
        device=device,
    )

    holdout = None
    if args.extra_num_states > 0:
        extra_walk_length = args.extra_walk_length or config.warmup_walk_length
        extra_states, extra_reference_lengths = generate_fixed_depth_walk_states(
            generators,
            num_states=args.extra_num_states,
            walk_length=extra_walk_length,
            seed=args.extra_seed,
            history_size=args.extra_history_size,
        )
        holdout = evaluate_state_set(
            model,
            generators,
            extra_states,
            extra_reference_lengths,
            beam_width=beam_width,
            step_limit=step_limit,
            policy_alpha=policy_alpha,
            history_size=history_size,
            apply_x_trick=apply_x_trick,
            device=device,
        )

    report = {
        "checkpoint": args.checkpoint,
        "config": {
            "n": config.n,
            "beam_width": beam_width,
            "step_limit": step_limit,
            "policy_alpha": policy_alpha,
            "history_size": history_size,
            "apply_x_trick": apply_x_trick,
        },
        "saved_hard_eval": checkpoint.get("hard_eval"),
        "benchmark_eval": asdict(benchmark_summary),
        "holdout_eval": holdout,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
