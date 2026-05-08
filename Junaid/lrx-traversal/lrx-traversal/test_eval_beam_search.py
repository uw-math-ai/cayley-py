"""Tests for eval_beam_search.py — Issue #005."""

import os
import tempfile
import json
import torch
import pytest
import pandas as pd

from mlp_model import train_koltsov3_mlp, save_model, load_model
from state_expand import build_koltsov3_generator_tensors


# =============================================================================
# Helpers
# =============================================================================

@pytest.fixture(scope="module")
def tiny_model_path():
    """Train a tiny n=5 model, save to temp file, return path."""
    model = train_koltsov3_mlp(n=5, hidden_dim=64, epochs=10, device="cpu")
    tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
    tmp.close()
    save_model(model, tmp.name)
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture(scope="module")
def n5_target_states():
    """A few simple n=5 target states at varying distances."""
    gens = build_koltsov3_generator_tensors(5)
    identity = torch.arange(5, dtype=torch.int64, device=gens.device)
    return [
        identity[gens[0]].cpu().tolist(),        # 1 step (I)
        identity[gens[0]][gens[1]].cpu().tolist(),  # 2 steps (I then K)
        [4, 3, 2, 1, 0],                          # reversed (~7 steps)
    ]


# =============================================================================
# T1: Tracer Bullet — Returns DataFrame with correct columns
# =============================================================================

class TestDataFrameOutput:
    """evaluate_beam_search returns a well-formed DataFrame."""

    def test_returns_dataframe_with_correct_columns(
        self, tiny_model_path, n5_target_states
    ):
        """Call evaluate_beam_search, check result shape and columns."""
        from eval_beam_search import evaluate_beam_search

        df = evaluate_beam_search(
            model_path=tiny_model_path,
            target_states=n5_target_states,
            beam_widths=[8, 16],
            step_limit=10,
        )

        assert isinstance(df, pd.DataFrame)
        required_cols = {
            "n", "state_idx", "start_state", "path_found",
            "path_length", "steps_taken", "beam_width",
            "runtime_sec", "path", "states_visited",
        }
        missing = required_cols - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

        # 3 states × 2 beam widths = 6 rows
        assert len(df) == 6, f"Expected 6 rows, got {len(df)}"


# =============================================================================
# T2: Failed searches produce rows without crashing
# =============================================================================

class TestFailedSearches:
    """Failed beam searches are reported as path_found=False rows."""

    def test_failed_searches_produce_rows(
        self, tiny_model_path, n5_target_states
    ):
        """With step_limit=0, all searches fail but produce valid rows."""
        from eval_beam_search import evaluate_beam_search

        df = evaluate_beam_search(
            model_path=tiny_model_path,
            target_states=n5_target_states[:1],  # 1 state
            beam_widths=[8],
            step_limit=0,
        )

        assert len(df) == 1
        row = df.iloc[0]
        assert row["path_found"] == False
        assert row["path"] is None
        assert row["steps_taken"] == 0
        assert row["states_visited"] == 0


# =============================================================================
# T3: path_vs_optimal_ratio column when optimal_lengths provided
# =============================================================================

class TestOptimalRatio:
    """When optimal_lengths is given, path_vs_optimal_ratio is computed."""

    def test_optimal_ratio_computed(
        self, tiny_model_path, n5_target_states
    ):
        """Provide optimal_lengths, verify ratio column and values."""
        from eval_beam_search import evaluate_beam_search

        optimal = {
            tuple(n5_target_states[0]): 1,   # 1 step from identity
            tuple(n5_target_states[1]): 2,   # 2 steps
            tuple(n5_target_states[2]): 7,   # diameter
        }

        df = evaluate_beam_search(
            model_path=tiny_model_path,
            target_states=n5_target_states,
            beam_widths=[16],
            step_limit=14,
            optimal_lengths=optimal,
        )

        assert "optimal_length" in df.columns
        assert "path_vs_optimal_ratio" in df.columns

        # For the 1-step state, if path found, ratio should be >= 1.0
        row0 = df[df["state_idx"] == 0].iloc[0]
        if row0["path_found"]:
            assert row0["path_vs_optimal_ratio"] >= 1.0

        # For failed searches, ratio should be NaN/None
        failed = df[~df["path_found"]]
        if len(failed) > 0:
            assert failed["path_vs_optimal_ratio"].isna().all()


# =============================================================================
# T4: CLI parses args and produces CSV output
# =============================================================================

class TestCLI:
    """Command-line interface works end-to-end."""

    def test_cli_writes_csv(self, tiny_model_path):
        """Invoke CLI with states file, verify CSV output."""
        import subprocess

        # Create a temporary states file
        states = [[1, 0, 3, 2, 4], [4, 3, 2, 1, 0]]  # n=5
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(states, f)
            states_path = f.name

        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False
        ) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [
                    "python", "eval_beam_search.py",
                    "--model", tiny_model_path,
                    "--states", states_path,
                    "--beam-widths", "8", "16",
                    "--step-limit", "10",
                    "--output", output_path,
                    "--device", "cpu",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert result.returncode == 0, (
                f"CLI failed: {result.stderr}"
            )

            df = pd.read_csv(output_path)
            assert len(df) == 4  # 2 states × 2 beam widths
            assert "path_found" in df.columns
            assert "runtime_sec" in df.columns
        finally:
            os.unlink(states_path)
            os.unlink(output_path)
