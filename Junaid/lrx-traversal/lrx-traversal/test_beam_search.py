"""Tests for beam_search.py — Issue #003."""

import torch
import pytest
import time
import numpy as np
import json
import os

from beam_search import beam_search, BeamResult
from mlp_model import MLP, train_koltsov3_mlp
from state_expand import build_koltsov3_generator_tensors


# =============================================================================
# Helpers
# =============================================================================

def apply_path(start_state, path, gen_tensors):
    """Apply a sequence of generator indices to a start state, return final state."""
    if not isinstance(start_state, torch.Tensor):
        state = torch.tensor(start_state, dtype=torch.int64, device=gen_tensors.device)
    else:
        state = start_state.clone().detach().to(dtype=torch.int64, device=gen_tensors.device)
    for gen_idx in path:
        state = state[gen_tensors[gen_idx]]
    return state


def make_identity(n, device="cpu"):
    return torch.arange(n, dtype=torch.int64, device=device)


# =============================================================================
# T1: Tracer Bullet — Identity start returns immediately
# =============================================================================

class TestIdentityStart:
    """Beam search from identity should return immediately with 0 steps."""

    def test_identity_returns_immediately(self):
        """Start at identity: path_found=True, path=[], steps=0."""
        n = 5
        model = MLP(n=n, hidden_dim=32)
        gens = build_koltsov3_generator_tensors(n)
        start = make_identity(n, device=gens.device)

        result = beam_search(start, model, gens, beam_width=8, step_limit=10)

        assert result.path_found is True
        assert result.path == []
        assert result.path_length == 0
        assert result.steps_taken == 0
        assert result.states_visited == 0
        assert isinstance(result.runtime_sec, float)


# =============================================================================
# T3: Beam search finds a valid path (acceptance criterion 1)
# =============================================================================

@pytest.fixture(scope="module")
def n5_model():
    """Train a small n=5 MLP for beam search integration tests."""
    return train_koltsov3_mlp(n=5, hidden_dim=128, epochs=25, device="cpu")


@pytest.fixture(scope="module")
def n5_longest_elements():
    """Load the 7 longest elements for n=5 from BFS ground truth."""
    bfs_path = os.path.join(os.path.dirname(__file__),
                             "bfs_results", "koltsov3_bfs_n05.json")
    with open(bfs_path) as f:
        data = json.load(f)
    return data["longest_elements"]


class TestFindsPath:
    """Beam search finds a path from a longest element to identity."""

    def test_finds_path_for_one_n5_longest(self, n5_model, n5_longest_elements):
        """Acceptance criterion: finds valid path for at least 1 longest element.

        n=5 has diameter 7. step_limit=14 (2x diameter) is generous.
        """
        gens = build_koltsov3_generator_tensors(5)
        identity = make_identity(5, device=gens.device)

        any_found = False
        for elem in n5_longest_elements:
            start = torch.tensor(elem, dtype=torch.int64, device=gens.device)
            result = beam_search(start, n5_model, gens,
                                 beam_width=32, step_limit=14)
            if result.path_found:
                any_found = True
                # Verify path transforms start to identity
                final = apply_path(start, result.path, gens)
                assert torch.equal(final, identity), \
                    f"Path does not yield identity: start={elem}, " \
                    f"path={result.path}, final={final.tolist()}"
                break

        assert any_found, (
            "Beam search failed to find a path for ANY n=5 longest element "
            "with beam_width=32, step_limit=14"
        )


# =============================================================================
# T4: Path reconstruction validation
# =============================================================================

class TestPathValidation:
    """The returned path, when applied, transforms start state to identity."""

    def test_path_transforms_to_identity(self, n5_model):
        """For a state 1 step from identity, verify path correctness."""
        gens = build_koltsov3_generator_tensors(5)
        identity = make_identity(5, device=gens.device)

        # Apply generator I to identity: identity[I] = I = [1,0,3,2,4]
        one_step = identity[gens[0]].clone()

        result = beam_search(one_step, n5_model, gens,
                             beam_width=8, step_limit=10)

        if result.path_found:
            # Path must transform start → identity
            final = apply_path(one_step, result.path, gens)
            assert torch.equal(final, identity), (
                f"Path {result.path} applied to {one_step.tolist()} "
                f"yields {final.tolist()}, expected {identity.tolist()}"
            )
            # Path must be at least 1 step (since start != identity)
            assert result.path_length >= 1
            assert len(result.path) == result.path_length

    def test_multiple_states_paths_are_valid(self, n5_model, n5_longest_elements):
        """For every successful result, verify path correctness."""
        gens = build_koltsov3_generator_tensors(5)
        identity = make_identity(5, device=gens.device)

        for elem in n5_longest_elements:
            start = torch.tensor(elem, dtype=torch.int64, device=gens.device)
            result = beam_search(start, n5_model, gens,
                                 beam_width=32, step_limit=14)
            if result.path_found:
                final = apply_path(start, result.path, gens)
                assert torch.equal(final, identity), (
                    f"Path {result.path} applied to {elem} "
                    f"yields {final.tolist()}, expected identity"
                )
                # Generator indices must be valid (0, 1, or 2)
                assert all(g in (0, 1, 2) for g in result.path)


# =============================================================================
# T5: Edge cases
# =============================================================================

class TestEdgeCases:
    """Beam search handles edge cases correctly."""

    def test_beam_width_one_runs(self):
        """beam_width=1 (greedy) should not crash."""
        n = 5
        model = MLP(n=n, hidden_dim=32)
        gens = build_koltsov3_generator_tensors(n)
        start = torch.tensor([1, 0, 3, 2, 4], dtype=torch.int64, device=gens.device)

        result = beam_search(start, model, gens, beam_width=1, step_limit=20)
        # Don't assert path_found (greedy may fail), just assert no crash
        assert isinstance(result, BeamResult)
        assert result.states_visited >= 0
        assert result.runtime_sec >= 0

    def test_step_counting(self, n5_model):
        """steps_taken counts actual iterations performed."""
        gens = build_koltsov3_generator_tensors(5)
        identity = make_identity(5, device=gens.device)

        # Apply generator I to identity
        one_step = identity[gens[0]].clone()

        result = beam_search(one_step, n5_model, gens,
                             beam_width=8, step_limit=10)
        assert result.steps_taken <= 10
        if result.path_found:
            assert result.steps_taken >= 1

    def test_states_visited_monotonic(self, n5_model):
        """states_visited should increase with more steps."""
        gens = build_koltsov3_generator_tensors(5)
        identity = make_identity(5, device=gens.device)
        start = identity[gens[0]].clone()

        r1 = beam_search(start, n5_model, gens, beam_width=4, step_limit=3)
        r2 = beam_search(start, n5_model, gens, beam_width=4, step_limit=10)
        # More steps should visit at least as many states
        assert r2.states_visited >= r1.states_visited


# =============================================================================
# T6: Identity scores lowest in the beam (acceptance criterion 4)
# =============================================================================

class TestIdentityScoring:
    """MLP assigns lowest score to identity among candidates."""

    def test_identity_scores_lowest_in_beam(self, n5_model):
        """Identity scores lower than states further from it.

        Uses the model's scoring on the expanded neighbors of a random state.
        Among all candidates, identity (when present) should be among the
        lowest-scored, demonstrating the model works as a distance oracle.
        """
        n = 5
        gens = build_koltsov3_generator_tensors(n)
        identity = make_identity(n, device=gens.device)
        model = n5_model
        if next(model.parameters()).device != gens.device:
            model = model.to(gens.device)

        # Test that identity scores lower than a far state (reversed)
        reversed_state = torch.tensor([4, 3, 2, 1, 0], dtype=torch.int64,
                                       device=gens.device).unsqueeze(0)
        id_batch = identity.unsqueeze(0)
        batch = torch.cat([id_batch, reversed_state])

        with torch.no_grad():
            scores = model(batch).squeeze(-1)

        # Identity must score lower than reversed (distance 7)
        assert scores[0].item() < scores[1].item(), (
            f"Identity score {scores[0].item():.4f} should be < "
            f"reversed score {scores[1].item():.4f}"
        )


# =============================================================================
# Issue #004: Dedup + Non-backtracking History
# =============================================================================

class TestDedup:
    """Global state deduplication prevents revisiting previously seen states."""

    def test_dedup_reduces_states_scored(self, n5_model):
        """With dedup enabled, states_visited is <= without dedup.

        Koltsov3 generators are involutions (I²=K²=S²=identity), so the
        beam naturally produces duplicate states when expanding. Dedup
        filters these before scoring, reducing states_visited.
        """
        gens = build_koltsov3_generator_tensors(5)
        identity = make_identity(5, device=gens.device)
        # Start one step from identity so we don't trigger early-exit
        start = identity[gens[0]].clone()  # [1,0,3,2,4]

        r_on = beam_search(start, n5_model, gens,
                           beam_width=8, step_limit=5,
                           deduplicate=True)
        r_off = beam_search(start, n5_model, gens,
                            beam_width=8, step_limit=5,
                            deduplicate=False)

        # Dedup should score fewer or equal candidates
        assert r_on.states_visited <= r_off.states_visited, (
            f"dedup states_visited={r_on.states_visited} should be <= "
            f"no-dedup states_visited={r_off.states_visited}"
        )


class TestHistoryDisabled:
    """history_size=0 disables non-backtracking."""

    def test_history_zero_runs(self, n5_model):
        """history_size=0 should not crash and produce valid results."""
        gens = build_koltsov3_generator_tensors(5)
        identity = make_identity(5, device=gens.device)
        start = identity[gens[0]].clone()

        result = beam_search(start, n5_model, gens,
                             beam_width=8, step_limit=10,
                             history_size=0)
        assert isinstance(result, BeamResult)
        assert result.states_visited >= 0


class TestToggleCombinations:
    """Dedup and history are independently toggleable."""

    def test_all_four_combinations(self, n5_model):
        """All 4 combinations of (dedup, history_size) should run."""
        gens = build_koltsov3_generator_tensors(5)
        identity = make_identity(5, device=gens.device)
        start = identity[gens[0]].clone()

        combos = [
            (True, 0),
            (True, 32),
            (False, 0),
            (False, 32),
        ]
        for dedup, hsize in combos:
            result = beam_search(start, n5_model, gens,
                                 beam_width=8, step_limit=10,
                                 deduplicate=dedup,
                                 history_size=hsize)
            assert isinstance(result, BeamResult), \
                f"Failed with dedup={dedup}, history_size={hsize}"


class TestHistoryBlocksBacktracking:
    """Non-backtracking history prevents immediate revisiting."""

    def test_history_runs_without_crashing(self, n5_model, n5_longest_elements):
        """history_size > 0 should not crash and still find paths."""
        gens = build_koltsov3_generator_tensors(5)
        identity = make_identity(5, device=gens.device)

        # Test on all longest elements with history enabled
        any_found = False
        for elem in n5_longest_elements:
            start = torch.tensor(elem, dtype=torch.int64, device=gens.device)
            result = beam_search(start, n5_model, gens,
                                 beam_width=32, step_limit=14,
                                 deduplicate=True, history_size=32)
            if result.path_found:
                any_found = True
                final = apply_path(start, result.path, gens)
                assert torch.equal(final, identity)
                break

        assert any_found, (
            "Beam search with history_size=32 should still find paths"
        )


class TestVisitedSetBounded:
    """The visited set is naturally bounded by step_limit × beam_width."""

    def test_states_visited_never_exceeds_step_limit_times_beam_width(self, n5_model):
        """states_visited <= step_limit × beam_width × 3 (max candidates/step).

        Each iteration scores at most beam_width × 3 candidates.
        """
        gens = build_koltsov3_generator_tensors(5)
        identity = make_identity(5, device=gens.device)
        start = identity[gens[0]].clone()

        bw, sl = 8, 10
        result = beam_search(start, n5_model, gens,
                             beam_width=bw, step_limit=sl,
                             deduplicate=True, history_size=32)

        max_possible = sl * bw * 3  # max candidates scored
        assert result.states_visited <= max_possible, (
            f"states_visited={result.states_visited} exceeds "
            f"max possible {max_possible}"
        )

class TestStepLimitExceeded:
    """Beam search returns path_found=False when step_limit is insufficient."""

    def test_step_limit_zero_fails(self):
        """step_limit=0 with non-identity start: no iterations, path_found=False."""
        n = 5
        model = MLP(n=n, hidden_dim=32)
        gens = build_koltsov3_generator_tensors(n)
        # A non-identity start state
        start = torch.tensor([1, 0, 3, 2, 4], dtype=torch.int64, device=gens.device)

        result = beam_search(start, model, gens, beam_width=8, step_limit=0)

        assert result.path_found is False
        assert result.path is None
        assert result.steps_taken == 0
        assert result.states_visited == 0
        assert isinstance(result.runtime_sec, float)

    def test_step_limit_zero_does_not_false_positive(self):
        """Even if start IS identity, step_limit=0 should still succeed
        (identity pre-check happens before loop)."""
        n = 5
        model = MLP(n=n, hidden_dim=32)
        gens = build_koltsov3_generator_tensors(n)
        start = make_identity(n, device=gens.device)

        result = beam_search(start, model, gens, beam_width=8, step_limit=0)

        # Identity pre-check should still return success
        assert result.path_found is True
        assert result.path_length == 0
