"""MLP model for Koltsov3 value function — save/load/inference.

Issue #001: Extracted from run_mlp_extended.py, adds local .pth save/load.
"""

import random
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from state_expand import build_koltsov3_generator_tensors


# =============================================================================
# Model
# =============================================================================

class MLP(nn.Module):
    """One-hot → Linear → ReLU → Linear → scalar. Input: (batch, n) int64."""

    def __init__(self, n: int, hidden_dim: int):
        super().__init__()
        self.n = n
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(n * n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n) int64 → one-hot → (batch, n*n) float
        x = torch.nn.functional.one_hot(x.long(), num_classes=self.n).float()
        x = x.flatten(start_dim=-2)
        return self.layers(x)


# =============================================================================
# Save / Load
# =============================================================================

def save_model(model: MLP, path: str) -> None:
    """Save model state dict and metadata to a .pth file."""
    torch.save({
        'state_dict': model.state_dict(),
        'n': model.n,
        'hidden_dim': model.hidden_dim,
    }, path)


def load_model(path: str, device: str = "auto") -> MLP:
    """Load model from .pth file. device='auto' picks cuda if available."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = MLP(n=checkpoint['n'], hidden_dim=checkpoint['hidden_dim'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model


# =============================================================================
# Training
# =============================================================================

def train_koltsov3_mlp(
    n: int,
    hidden_dim: int = 512,
    epochs: int = 25,
    lr: float = 0.001,
    device: str = "auto",
    seed: int = 42,
) -> MLP:
    """Train an MLP value function on Koltsov3 random walks.

    Uses the same data generation as run_mlp_extended.py:
    random walks from identity, labels = walk step number.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    moves_t = build_koltsov3_generator_tensors(n).to(device)
    state_dest = torch.arange(n, dtype=torch.int64, device=device)
    n_gen = 3
    walk_length = 8 * n

    model = MLP(n=n, hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Generate random walks
        n_walks = 20_000 if n <= 40 else 10_000
        states_t = state_dest.unsqueeze(0).repeat(n_walks, 1).to(torch.uint8)
        all_states, all_labels = [], []
        for step in range(1, walk_length + 1):
            move_ids = torch.randint(0, n_gen, (n_walks,), device=device)
            states_t = torch.gather(states_t, 1, moves_t[move_ids])
            all_states.append(states_t.clone())
            all_labels.append(torch.full((n_walks,), step,
                                         dtype=torch.float32, device=device))
        X_tr = torch.cat(all_states, dim=0).long()
        y_tr = torch.cat(all_labels, dim=0).unsqueeze(-1)

        # Shuffle
        perm = torch.randperm(len(X_tr), device=device)
        X_tr, y_tr = X_tr[perm], y_tr[perm]

        # Simple train loop (no validation for minimal implementation)
        model.train()
        batch_size = 1024
        for i in range(0, len(X_tr), batch_size):
            xb = X_tr[i:i + batch_size]
            yb = y_tr[i:i + batch_size]
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model
