from __future__ import annotations

from collections import deque
from math import factorial
from typing import Iterable

from .models import GraphSpec, State


def identity(n: int) -> State:
    return tuple(range(n))


def state_space_kind(spec: GraphSpec) -> str:
    raw = str(
        spec.params.get(
            "stateSpace",
            spec.params.get("cosetKind", spec.params.get("space", "cayley")),
        )
    )
    normalized = raw.strip().lower().replace("-", "_")
    if normalized in ("cayley", "full", "group", "s_n", "sn"):
        return "cayley"
    if normalized in ("k_different", "kdifferent", "k_different_coset", "schreier", "coset"):
        return "k_different"
    raise ValueError(f"unknown state space: {raw}")


def different_k(spec: GraphSpec) -> int:
    value = spec.params.get("differentK", spec.params.get("different_k", spec.params.get("cosetK", 2)))
    k = int(value)
    if not (2 <= k <= spec.n):
        raise ValueError(f"differentK must be in [2, {spec.n}]")
    return k


def initial_state(spec: GraphSpec) -> State:
    if state_space_kind(spec) == "cayley":
        return identity(spec.n)
    k = different_k(spec)
    return tuple(list(range(k - 1)) + [k - 1] * (spec.n - k + 1))


def state_space_upper_bound(spec: GraphSpec, cap: int = 10**18) -> int:
    if state_space_kind(spec) == "cayley":
        return factorial_or_cap(spec.n, cap=cap)
    k = different_k(spec)
    try:
        value = factorial(spec.n) // factorial(spec.n - k + 1)
    except ValueError:
        return 0
    return min(value, cap)


def bruhat_rank(state: State) -> int:
    return sum(1 for i in range(len(state)) for j in range(i + 1, len(state)) if state[i] > state[j])


def state_key(state: State) -> str:
    return ",".join(str(x) for x in state)


def parse_state_key(key: str) -> State:
    if not key:
        return ()
    return tuple(int(x) for x in key.split(","))


def display_state(state: State) -> str:
    shown = [x + 1 for x in state]
    if len(shown) <= 9:
        return "".join(str(x) for x in shown)
    return "[" + " ".join(str(x) for x in shown) + "]"


def validate_permutation(perm: Iterable[int], n: int) -> State:
    tup = tuple(int(x) for x in perm)
    if len(tup) != n or sorted(tup) != list(range(n)):
        raise ValueError(f"not a permutation of 0..{n - 1}: {tup}")
    return tup


def apply_generator(state: State, generator: State) -> State:
    return tuple(state[i] for i in generator)


def inverse_permutation(perm: State) -> State:
    inv = [0] * len(perm)
    for i, value in enumerate(perm):
        inv[value] = i
    return tuple(inv)


def is_involutive(perm: State) -> bool:
    return apply_generator(perm, perm) == identity(len(perm))


def swap_permutation(n: int, a: int, b: int) -> State:
    perm = list(range(n))
    perm[a], perm[b] = perm[b], perm[a]
    return tuple(perm)


def swaps_permutation(n: int, swaps: Iterable[tuple[int, int]]) -> State:
    perm = list(range(n))
    for a, b in swaps:
        perm[a], perm[b] = perm[b], perm[a]
    return tuple(perm)


def cycle_permutation(n: int, positions: Iterable[int]) -> State:
    pos = [int(x) for x in positions]
    if len(pos) < 2:
        raise ValueError("cycle needs at least two positions")
    if len(set(pos)) != len(pos) or any(p < 0 or p >= n for p in pos):
        raise ValueError(f"invalid cycle positions for n={n}: {pos}")
    perm = list(range(n))
    for idx, current in enumerate(pos):
        perm[current] = pos[(idx + 1) % len(pos)]
    return tuple(perm)


def prefix_reversal(n: int, k: int) -> State:
    if not (2 <= k <= n):
        raise ValueError(f"prefix reversal k must be in [2, {n}]")
    return tuple(reversed(range(k))) + tuple(range(k, n))


def reachable_shell(
    generators: Iterable[State],
    focus: State,
    radius: int,
    cap: int,
) -> tuple[dict[State, int], list[tuple[State, State, int]]]:
    gens = list(generators)
    seen = {focus: 0}
    edges: list[tuple[State, State, int]] = []
    queue: deque[State] = deque([focus])

    while queue and len(seen) < cap:
        state = queue.popleft()
        depth = seen[state]
        if depth >= radius:
            continue
        for idx, gen in enumerate(gens):
            nxt = apply_generator(state, gen)
            edges.append((state, nxt, idx))
            if nxt not in seen:
                seen[nxt] = depth + 1
                if len(seen) >= cap:
                    break
                queue.append(nxt)

    return seen, edges


def factorial_or_cap(n: int, cap: int = 10**18) -> int:
    try:
        value = factorial(n)
    except ValueError:
        return 0
    return min(value, cap)
