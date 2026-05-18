#!/usr/bin/env python3
"""Compare PPO and DQN Koltsov3 checkpoints on shared benchmark and holdout states."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Michael.lrx_koltsov3_ppo import (
    ActorCritic,
    beam_search_with_policy_prior,
    build_hard_state_benchmark,
    generate_fixed_depth_walk_states,
    get_koltsov3_generators,
)


@dataclass(frozen=True)
class LoadedSearchModel:
    label: str
    checkpoint_path: str
    algorithm: str
    config: dict
    model: ActorCritic
    saved_hard_eval: dict | None


def _select_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _parse_labeled_model(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must use LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("--model must use LABEL=PATH")
    return label, path


def load_search_model(label: str, checkpoint_path: str, *, device: str) -> LoadedSearchModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(checkpoint["config"])
    model = ActorCritic(
        config["n"],
        config.get("hidden_dim", 512),
        k=config.get("k", 0),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return LoadedSearchModel(
        label=label,
        checkpoint_path=checkpoint_path,
        algorithm=checkpoint.get("algorithm", "ppo"),
        config=config,
        model=model,
        saved_hard_eval=checkpoint.get("hard_eval"),
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
                "states_scored": result.states_scored,
                "steps_taken": result.steps_taken,
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
        "median_states_scored": (
            float(np.median([row["states_scored"] for row in rows if row["path_found"]]))
            if solved_lengths
            else None
        ),
        "median_steps_taken": (
            float(np.median([row["steps_taken"] for row in rows if row["path_found"]]))
            if solved_lengths
            else None
        ),
    }
    return {"summary": summary, "rows": rows}


def _profile_policy_alpha(profile_name: str, loaded: LoadedSearchModel) -> float:
    if profile_name == "heuristic_only":
        return 0.0
    if profile_name == "checkpoint_default":
        return float(loaded.config.get("eval_policy_alpha", 0.0))
    raise ValueError(f"unknown profile {profile_name}")


def _build_summary_rows(report: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_report in report["models"]:
        for profile_name, profile_report in model_report["profiles"].items():
            for state_set_name in ("benchmark_eval", "holdout_eval"):
                state_set_report = profile_report.get(state_set_name)
                if state_set_report is None:
                    continue
                summary = state_set_report["summary"]
                rows.append(
                    {
                        "label": model_report["label"],
                        "algorithm": model_report["algorithm"],
                        "profile": profile_name,
                        "state_set": state_set_name,
                        "beam_width": profile_report["search"]["beam_width"],
                        "step_limit": profile_report["search"]["step_limit"],
                        "history_size": profile_report["search"]["history_size"],
                        "policy_alpha": profile_report["search"]["policy_alpha"],
                        "apply_x_trick": profile_report["search"]["apply_x_trick"],
                        "success_rate": summary["success_rate"],
                        "solved_count": summary["solved_count"],
                        "total_count": summary["total_count"],
                        "median_length_ratio": summary["median_length_ratio"],
                        "median_path_length": summary["median_path_length"],
                        "median_runtime_sec": summary["median_runtime_sec"],
                        "median_states_scored": summary.get("median_states_scored"),
                        "median_steps_taken": summary.get("median_steps_taken"),
                    }
                )
    return rows


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "algorithm",
        "profile",
        "state_set",
        "beam_width",
        "step_limit",
        "history_size",
        "policy_alpha",
        "apply_x_trick",
        "success_rate",
        "solved_count",
        "total_count",
        "median_length_ratio",
        "median_path_length",
        "median_runtime_sec",
        "median_states_scored",
        "median_steps_taken",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Koltsov3 PPO and DQN checkpoints")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec as LABEL=PATH. Repeat for each checkpoint.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=["heuristic_only", "checkpoint_default"],
        default=["heuristic_only", "checkpoint_default"],
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--beam-width", type=int, default=256)
    parser.add_argument("--step-limit", type=int, default=256)
    parser.add_argument("--history-size", type=int, default=32)
    parser.add_argument(
        "--apply-x-trick",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--holdout-num-states", type=int, default=8)
    parser.add_argument("--holdout-walk-length", type=int, default=128)
    parser.add_argument("--holdout-seed", type=int, default=999)
    parser.add_argument("--holdout-history-size", type=int, default=32)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    device = _select_device(args.device)
    labeled_models = [_parse_labeled_model(value) for value in args.model]
    loaded_models = [load_search_model(label, path, device=device) for label, path in labeled_models]

    n_values = sorted({loaded.config["n"] for loaded in loaded_models})
    k_values = sorted({loaded.config.get("k", 0) for loaded in loaded_models})
    hidden_dims = sorted({loaded.config.get("hidden_dim", 512) for loaded in loaded_models})
    if len(n_values) != 1:
        raise ValueError(f"all models must use the same n, got {n_values}")
    if len(k_values) != 1:
        raise ValueError(f"all models must use the same k, got {k_values}")

    n = n_values[0]
    k = k_values[0]
    generators = get_koltsov3_generators(n, k)

    benchmark_states_list, benchmark_reference_map, benchmark_source, benchmark_reference_kind = (
        build_hard_state_benchmark(n)
    )
    benchmark_states = np.stack(benchmark_states_list, axis=0)
    benchmark_reference_lengths = np.asarray(
        [benchmark_reference_map[tuple(state.tolist())] for state in benchmark_states],
        dtype=np.float32,
    )

    holdout = None
    if args.holdout_num_states > 0:
        holdout = generate_fixed_depth_walk_states(
            generators,
            num_states=args.holdout_num_states,
            walk_length=args.holdout_walk_length,
            seed=args.holdout_seed,
            history_size=args.holdout_history_size,
        )

    report = {
        "comparison_spec": {
            "n": n,
            "k": k,
            "hidden_dims": hidden_dims,
            "profiles": args.profiles,
            "search": {
                "beam_width": args.beam_width,
                "step_limit": args.step_limit,
                "history_size": args.history_size,
                "apply_x_trick": args.apply_x_trick,
            },
            "benchmark": {
                "source": benchmark_source,
                "reference_kind": benchmark_reference_kind,
                "num_states": int(benchmark_states.shape[0]),
            },
            "holdout": {
                "num_states": args.holdout_num_states,
                "walk_length": args.holdout_walk_length,
                "seed": args.holdout_seed,
                "history_size": args.holdout_history_size,
            },
        },
        "models": [],
    }

    for loaded in loaded_models:
        model_report = {
            "label": loaded.label,
            "algorithm": loaded.algorithm,
            "checkpoint": loaded.checkpoint_path,
            "saved_hard_eval": loaded.saved_hard_eval,
            "training_config": loaded.config,
            "profiles": {},
        }
        for profile_name in args.profiles:
            policy_alpha = _profile_policy_alpha(profile_name, loaded)
            profile_report = {
                "search": {
                    "beam_width": args.beam_width,
                    "step_limit": args.step_limit,
                    "history_size": args.history_size,
                    "apply_x_trick": args.apply_x_trick,
                    "policy_alpha": policy_alpha,
                },
                "benchmark_eval": {
                    "source": benchmark_source,
                    "reference_kind": benchmark_reference_kind,
                    **evaluate_state_set(
                        loaded.model,
                        generators,
                        benchmark_states,
                        benchmark_reference_lengths,
                        beam_width=args.beam_width,
                        step_limit=args.step_limit,
                        policy_alpha=policy_alpha,
                        history_size=args.history_size,
                        apply_x_trick=args.apply_x_trick,
                        device=device,
                    ),
                },
                "holdout_eval": None,
            }
            if holdout is not None:
                holdout_states, holdout_reference_lengths = holdout
                profile_report["holdout_eval"] = {
                    "source": "fixed_depth_holdout",
                    "reference_kind": "witness_length",
                    **evaluate_state_set(
                        loaded.model,
                        generators,
                        holdout_states,
                        holdout_reference_lengths,
                        beam_width=args.beam_width,
                        step_limit=args.step_limit,
                        policy_alpha=policy_alpha,
                        history_size=args.history_size,
                        apply_x_trick=args.apply_x_trick,
                        device=device,
                    ),
                }
            model_report["profiles"][profile_name] = profile_report
        report["models"].append(model_report)

    summary_rows = _build_summary_rows(report)
    report["summary_rows"] = summary_rows

    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(report, indent=2))

    if args.output_csv is not None:
        _write_csv(args.output_csv, summary_rows)

    print(json.dumps({"output_json": args.output_json, "summary_rows": summary_rows}, indent=2))


if __name__ == "__main__":
    main()