#!/bin/bash
#SBATCH --job-name=ppo_n_probe_cpu
#SBATCH --account=stf
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "/mmfs1/gscratch/stf/zengrf/cayley-py"
mkdir -p logs

if [[ ! -f "Michael/.venv/bin/activate" ]]; then
  echo "Missing Michael/.venv. Recreate the uv environment before running this probe." >&2
  exit 1
fi

source "Michael/.venv/bin/activate"

python - <<'PY'
import time

import numpy as np
import torch

from Michael.lrx_koltsov3_ppo import (
    ActorCritic,
    PPOConfig,
    apply_generator,
    distance_to_identity,
    get_koltsov3_generators,
    train_ppo,
)


def evaluate_model(model, config: PPOConfig, n_episodes: int = 200):
    device = next(model.parameters()).device
    generators = get_koltsov3_generators(config.n, config.k)
    identity = np.arange(config.n, dtype=np.int64)
    success_count = 0
    final_distances = []
    solved_steps = []

    for episode_idx in range(n_episodes):
        rng = np.random.default_rng(config.seed + 10_000 + episode_idx)
        scramble_steps = int(rng.integers(1, config.max_scramble_steps + 1))
        state = identity.copy()
        for _ in range(scramble_steps):
            action = int(rng.integers(0, generators.shape[0]))
            state = apply_generator(state, generators[action])

        solved = False
        for step_idx in range(config.max_episode_steps):
            obs_t = torch.tensor(state, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())
            state = apply_generator(state, generators[action])
            if distance_to_identity(state) == 0:
                success_count += 1
                solved_steps.append(step_idx + 1)
                solved = True
                break

        if not solved:
            final_distances.append(distance_to_identity(state))
        else:
            final_distances.append(0)

    mean_solved_steps = float(np.mean(solved_steps)) if solved_steps else float("nan")
    return {
        "success_rate": success_count / n_episodes,
        "mean_final_distance": float(np.mean(final_distances)),
        "mean_solved_steps": mean_solved_steps,
    }


print(f"torch_version={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")

for n in (16, 32, 64):
    config = PPOConfig(
        n=n,
        total_steps=8192,
        rollout_steps=1024,
        update_epochs=2,
        minibatch_size=256,
        device="cpu",
        log_every_updates=2,
    )
    param_count = sum(p.numel() for p in ActorCritic(n, config.hidden_dim).parameters())
    print(f"=== probe_start n={n} params={param_count} ===")
    started_at = time.perf_counter()
    model = train_ppo(config)
    elapsed = time.perf_counter() - started_at
    metrics = evaluate_model(model, config)
    print(
        "probe_summary "
        f"n={n} "
        f"train_seconds={elapsed:.2f} "
        f"success_rate={metrics['success_rate']:.3f} "
        f"mean_final_distance={metrics['mean_final_distance']:.3f} "
        f"mean_solved_steps={metrics['mean_solved_steps']:.3f}"
    )
PY