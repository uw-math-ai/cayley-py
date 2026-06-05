from __future__ import annotations

import random
from uuid import uuid4

from .bfs import ensure_exact_bfs
from .generators import build_generators, generator_map
from .models import ChallengeState, GraphSpec
from .permutations import apply_generator, initial_state


SESSIONS: dict[str, ChallengeState] = {}


def _distance_for(difficulty: str, diameter: int, custom_distance: int | None) -> int:
    if custom_distance is not None:
        return max(0, min(diameter, int(custom_distance)))
    if diameter == 0:
        return 0
    if difficulty == "easy":
        return max(1, round(diameter * 0.33))
    if difficulty == "hard":
        return diameter
    return max(1, round(diameter * 0.66))


def start_challenge(
    spec: GraphSpec,
    difficulty: str = "medium",
    custom_distance: int | None = None,
    seed: int | None = None,
    exact_cap: int = 500_000,
) -> ChallengeState:
    bfs = ensure_exact_bfs(spec, cap=exact_cap)
    target_distance = _distance_for(difficulty, bfs.diameter, custom_distance)
    layer = bfs.layers[target_distance]
    rng = random.Random(seed)
    target = rng.choice(layer)
    certified = bfs.path_to(target)
    start = initial_state(spec)
    session = ChallengeState(
        session_id=str(uuid4()),
        spec=spec,
        target=target,
        current=start,
        target_distance=target_distance,
        user_states=[start],
        certified_path=certified,
        generators=tuple(build_generators(spec)),
    )
    SESSIONS[session.session_id] = session
    return session


def move_challenge(session_id: str, generator_id: str) -> ChallengeState:
    if session_id not in SESSIONS:
        raise ValueError("unknown challenge session")
    session = SESSIONS[session_id]
    if session.status != "active":
        return session
    generators = generator_map(session.spec)
    if generator_id not in generators:
        raise ValueError(f"unknown generator for session: {generator_id}")
    session.current = apply_generator(session.current, generators[generator_id].permutation)
    session.user_path.append(generator_id)
    session.user_states.append(session.current)
    if session.current == session.target:
        session.status = "completed"
    return session


def forfeit_challenge(session_id: str) -> ChallengeState:
    if session_id not in SESSIONS:
        raise ValueError("unknown challenge session")
    session = SESSIONS[session_id]
    if session.status == "active":
        session.status = "forfeited"
    return session
