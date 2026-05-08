"""Tests for state_expand.py — Issue #002."""

import torch
import pytest
from state_expand import build_koltsov3_generator_tensors, expand_neighbors


class TestExpandIdentity:
    """Tracer bullet: expanding identity yields the generator permutations."""

    def test_expand_identity_returns_generators(self):
        """expand_neighbors(identity) should return I, K, S in order."""
        n = 8
        gens = build_koltsov3_generator_tensors(n)

        identity = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]],
                                dtype=torch.int64, device=gens.device)
        neighbors = expand_neighbors(identity, gens)

        # neighbors shape: (3, n) = [I, K, S]
        assert neighbors.shape == (3, n), f"Expected (3, {n}), got {neighbors.shape}"

        # Row 0 = I generator
        assert torch.equal(neighbors[0], gens[0]), f"I mismatch: {neighbors[0].tolist()} vs {gens[0].tolist()}"
        # Row 1 = K generator
        assert torch.equal(neighbors[1], gens[1]), f"K mismatch"
        # Row 2 = S generator
        assert torch.equal(neighbors[2], gens[2]), f"S mismatch"


class TestGeneratorConstruction:
    """build_koltsov3_generator_tensors produces valid permutations."""

    def test_generators_are_valid_permutations(self):
        """Each generator row should be a permutation of 0..n-1."""
        for n in [5, 8, 13, 32]:
            gens = build_koltsov3_generator_tensors(n)
            assert gens.shape == (3, n)
            for i in range(3):
                row = gens[i]
                assert set(row.tolist()) == set(range(n)), \
                    f"Generator {i} for n={n} is not a valid permutation: {row.tolist()}"

    def test_generators_are_on_gpu_when_available(self):
        """build_koltsov3_generator_tensors should place tensors on GPU if available."""
        gens = build_koltsov3_generator_tensors(8)
        if torch.cuda.is_available():
            assert gens.is_cuda, "Expected GPU tensor when CUDA is available"
        # If no CUDA, CPU is fine


class TestExpandShape:
    """expand_neighbors produces correct output shapes."""

    def test_output_shape(self):
        """For W states of size n, output should be (W*3, n)."""
        for n, W in [(5, 1), (8, 3), (13, 10), (32, 1)]:
            gens = build_koltsov3_generator_tensors(n)
            states = torch.randint(0, n, (W, n), dtype=torch.int64, device=gens.device)
            neighbors = expand_neighbors(states, gens)
            assert neighbors.shape == (W * 3, n), \
                f"n={n}, W={W}: expected ({W*3},{n}), got {neighbors.shape}"


class TestExpandValidPermutations:
    """All expanded neighbors are valid permutations."""

    def test_neighbors_are_valid_permutations(self):
        """For random input states, each neighbor is a valid permutation."""
        n = 8
        W = 100
        gens = build_koltsov3_generator_tensors(n)
        torch.manual_seed(42)
        # Generate valid random permutations by starting from identity and shuffling
        # Actually, random ints in [0, n-1] aren't necessarily valid permutations.
        # Use torch.randperm to generate valid random states.
        states = torch.stack([torch.randperm(n, device=gens.device) for _ in range(W)])
        neighbors = expand_neighbors(states, gens)

        # Each row should contain exactly 0..n-1 once
        for i in range(W * 3):
            row = neighbors[i]
            assert set(row.tolist()) == set(range(n)), \
                f"Neighbor {i} from state {i//3} is not a permutation: {row.tolist()}"


class TestExpandPerformance:
    """expand_neighbors meets GPU performance targets."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs CUDA")
    def test_expand_10k_states_under_10ms(self):
        """Expanding 10K states should complete in < 10ms on GPU."""
        import time
        n = 16
        W = 10_000
        gens = build_koltsov3_generator_tensors(n)
        states = torch.stack([torch.randperm(n, device=gens.device) for _ in range(W)])

        # Warmup
        _ = expand_neighbors(states, gens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed
        t0 = time.perf_counter()
        _ = expand_neighbors(states, gens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 10, f"Expanding {W} states took {elapsed_ms:.1f}ms, expected < 10ms"
