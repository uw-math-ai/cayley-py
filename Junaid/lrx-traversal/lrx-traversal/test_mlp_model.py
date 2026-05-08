"""Tests for mlp_model.py — Issue #001."""

import os, tempfile, torch
import pytest

# Import will fail until we create the module — RED phase
from mlp_model import MLP, save_model, load_model, train_koltsov3_mlp


class TestModelRoundTrip:
    """Tracer bullet: save and reload preserves model outputs."""

    def test_save_load_round_trip(self):
        """Save an MLP, reload it, and verify predictions match."""
        # Train a tiny model so we don't spend minutes in tests
        n = 5
        hidden_dim = 32
        model = train_koltsov3_mlp(n=n, hidden_dim=hidden_dim, epochs=3, device="cpu")

        # Predict on a single state
        test_state = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64)
        with torch.no_grad():
            output_before = model(test_state).item()

        # Save and reload
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            tmp_path = f.name
        try:
            save_model(model, tmp_path)
            reloaded = load_model(tmp_path, device="cpu")

            with torch.no_grad():
                output_after = reloaded(test_state).item()

            assert output_before == pytest.approx(output_after, rel=1e-5), \
                f"Round-trip mismatch: {output_before} vs {output_after}"
        finally:
            os.unlink(tmp_path)


class TestModelLearns:
    """Model trained on random walks learns distance ordering."""

    @pytest.fixture(scope="class")
    def trained_model(self):
        """Train once, reuse across tests in this class."""
        return train_koltsov3_mlp(n=8, hidden_dim=64, epochs=5, device="cpu")

    def test_identity_scores_lower_than_reversed(self, trained_model):
        """Identity should score lower than its reverse — model learned something."""
        identity = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.int64)
        reversed_state = torch.tensor([[7, 6, 5, 4, 3, 2, 1, 0]], dtype=torch.int64)
        with torch.no_grad():
            score_id = trained_model(identity).item()
            score_rev = trained_model(reversed_state).item()
        assert score_id < score_rev, \
            f"Identity score {score_id:.2f} should be < reversed score {score_rev:.2f}"


class TestBatchInference:
    """Batch predictions produce correct output shapes."""

    def test_batch_shape(self):
        model = MLP(n=5, hidden_dim=16)
        batch = torch.randint(0, 5, (100, 5), dtype=torch.int64)
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (100, 1), f"Expected (100, 1), got {out.shape}"

    def test_single_state_shape(self):
        model = MLP(n=5, hidden_dim=16)
        single = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64)
        with torch.no_grad():
            out = model(single)
        assert out.shape == (1, 1), f"Expected (1, 1), got {out.shape}"


class TestMetadata:
    """Save/load preserves model metadata."""

    def test_metadata_round_trip(self):
        n, hdim = 7, 128
        model = MLP(n=n, hidden_dim=hdim)
        tmp_path = "/tmp/test_metadata.pth"
        try:
            save_model(model, tmp_path)
            reloaded = load_model(tmp_path, device="cpu")
            assert reloaded.n == n, f"n: {reloaded.n} != {n}"
            assert reloaded.hidden_dim == hdim, f"hidden_dim: {reloaded.hidden_dim} != {hdim}"
        finally:
            os.unlink(tmp_path)
