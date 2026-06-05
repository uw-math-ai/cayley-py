from __future__ import annotations

from math import cos, pi, sin, sqrt
from typing import Any

from .bfs import BfsLimitExceeded, ensure_exact_bfs, exact_cache_exists, read_cache_metadata
from .generators import FAMILIES, build_generators
from .models import GraphSpec, GraphViewRequest, State
from .permutations import (
    apply_generator,
    bruhat_rank,
    display_state,
    identity,
    initial_state,
    inverse_permutation,
    reachable_shell,
    state_key,
    state_space_kind,
    state_space_upper_bound,
)


def graph_summary(spec: GraphSpec) -> dict[str, Any]:
    generators = build_generators(spec)
    spec_hash = spec.hash()
    metadata = read_cache_metadata(spec_hash) if exact_cache_exists(spec) else None
    family = FAMILIES[spec.family]
    return {
        "specHash": spec_hash,
        "family": family.to_dict(),
        "n": spec.n,
        "stateSpace": state_space_kind(spec),
        "startState": list(initial_state(spec)),
        "estimatedGroupUpperBound": state_space_upper_bound(spec),
        "generators": [gen.to_dict() for gen in generators],
        "generatorCount": len(generators),
        "exactCached": metadata is not None,
        "exact": metadata,
    }


def _layer_positions(layers: list[list[State]], distances: dict[State, int], max_nodes: int | None = None) -> dict[State, tuple[float, float]]:
    positions: dict[State, tuple[float, float]] = {}
    diameter = max(1, len(layers) - 1)
    for depth, layer in enumerate(layers):
        radius = 0.05 + 0.9 * (depth / diameter)
        count = max(1, len(layer))
        for idx, state in enumerate(layer):
            if depth == 0:
                positions[state] = (0.0, 0.0)
            else:
                angle = (2 * pi * idx / count) + (depth * 0.37)
                positions[state] = (radius * cos(angle), radius * sin(angle))
    return positions


def _bruhat_positions(states: list[State], distances: dict[State, int]) -> dict[State, tuple[float, float]]:
    ranks: dict[int, list[State]] = {}
    for state in states:
        ranks.setdefault(bruhat_rank(state), []).append(state)
    rank_values = sorted(ranks)
    max_rank = max(rank_values, default=1)
    positions: dict[State, tuple[float, float]] = {}
    for rank in rank_values:
        rank_states = sorted(ranks[rank], key=lambda state: (distances.get(state, 0), state_key(state)))
        count = max(1, len(rank_states))
        y = -0.9 + 1.8 * (rank / max(1, max_rank))
        for idx, state in enumerate(rank_states):
            x = 0.0 if count == 1 else -0.92 + 1.84 * (idx / (count - 1))
            jitter = ((distances.get(state, 0) % 5) - 2) * 0.012
            positions[state] = (x, y + jitter)
    return positions


def _normalize_positions(raw: dict[State, tuple[float, float]], pad: float = 0.92) -> dict[State, tuple[float, float]]:
    if not raw:
        return {}
    xs = [point[0] for point in raw.values()]
    ys = [point[1] for point in raw.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    return {
        state: (
            0.0 if span_x <= 1e-9 else -pad + 2 * pad * ((x - min_x) / span_x),
            0.0 if span_y <= 1e-9 else -pad + 2 * pad * ((y - min_y) / span_y),
        )
        for state, (x, y) in raw.items()
    }


def _lehmer_vector(state: State) -> list[int]:
    return [
        sum(1 for later in state[idx + 1 :] if later < value)
        for idx, value in enumerate(state)
    ]


def _lehmer_positions(states: list[State], distances: dict[State, int]) -> dict[State, tuple[float, float]]:
    raw: dict[State, tuple[float, float]] = {}
    for state in states:
        vector = _lehmer_vector(state)
        x = sum(value * cos((idx + 1) * 1.61803398875) for idx, value in enumerate(vector))
        y = sum(value * sin((idx + 1) * 2.41421356237) for idx, value in enumerate(vector))
        raw[state] = (x, y + distances.get(state, 0) * 0.01)
    return _normalize_positions(raw)


def _coset_positions(states: list[State], distances: dict[State, int]) -> dict[State, tuple[float, float]]:
    raw: dict[State, tuple[float, float]] = {}
    if not states:
        return raw
    n = max(1, len(states[0]) - 1)
    for state in states:
        labels = sorted(set(state))
        repeated_label = labels[-1]
        distinguished = [label for label in labels if label != repeated_label]
        positions = {label: state.index(label) for label in distinguished if label in state}
        if not positions:
            raw[state] = (distances.get(state, 0), bruhat_rank(state))
            continue
        x = positions.get(distinguished[0], 0) / n
        if len(distinguished) >= 2:
            y = positions.get(distinguished[1], 0) / n
        else:
            y = distances.get(state, 0) / max(1, max(distances.values(), default=1))
        if len(distinguished) > 2:
            tail = sum((idx + 1) * positions[label] for idx, label in enumerate(distinguished[2:]))
            y += 0.07 * (tail / max(1, n * len(distinguished[2:])))
        raw[state] = (x, y)
    return _normalize_positions(raw)


def _visible_edges_for_positions(
    states: list[State],
    spec: GraphSpec,
) -> list[tuple[State, State]]:
    state_set = set(states)
    edges: list[tuple[State, State]] = []
    for source in states:
        for gen in build_generators(spec):
            target = apply_generator(source, gen.permutation)
            if target in state_set and target != source:
                edges.append((source, target))
    return edges


def _spectral_positions(states: list[State], spec: GraphSpec, distances: dict[State, int]) -> dict[State, tuple[float, float]]:
    if len(states) <= 2:
        return _layer_positions([states], distances)

    index = {state: idx for idx, state in enumerate(states)}
    adjacency: list[set[int]] = [set() for _ in states]
    for source, target in _visible_edges_for_positions(states, spec):
        a = index[source]
        b = index[target]
        adjacency[a].add(b)
        adjacency[b].add(a)

    if not any(adjacency):
        return _lehmer_positions(states, distances)

    def matvec(vector: list[float]) -> list[float]:
        out = [0.0] * len(vector)
        for idx, neighbors in enumerate(adjacency):
            if not neighbors:
                continue
            scale = 1 / sqrt(len(neighbors))
            total = 0.0
            for nbr in neighbors:
                total += vector[nbr] / sqrt(max(1, len(adjacency[nbr])))
            out[idx] = scale * total
        return out

    def orthogonalize(vector: list[float], bases: list[list[float]]) -> list[float]:
        mean = sum(vector) / max(1, len(vector))
        out = [value - mean for value in vector]
        for basis in bases:
            denom = sum(value * value for value in basis) or 1.0
            coeff = sum(value * basis[idx] for idx, value in enumerate(out)) / denom
            out = [value - coeff * basis[idx] for idx, value in enumerate(out)]
        norm = sqrt(sum(value * value for value in out))
        if norm < 1e-9:
            return [sin((idx + 1) * 1.37) for idx in range(len(vector))]
        return [value / norm for value in out]

    vectors: list[list[float]] = []
    seeds = (
        [sin((idx + 1) * 1.61803398875) for idx in range(len(states))],
        [cos((idx + 1) * 2.41421356237) for idx in range(len(states))],
    )
    for seed in seeds:
        vector = orthogonalize(seed, vectors)
        for _ in range(64):
            vector = orthogonalize(matvec(vector), vectors)
        vectors.append(vector)

    raw = {
        state: (vectors[0][idx], vectors[1][idx])
        for state, idx in index.items()
    }
    return _normalize_positions(raw)


def _reverse_distances_to_target(
    states: list[State],
    spec: GraphSpec,
    target: State,
) -> dict[State, int]:
    state_set = set(states)
    if target not in state_set:
        return {}
    inverse_generators = [inverse_permutation(gen.permutation) for gen in build_generators(spec)]
    seen: dict[State, int] = {target: 0}
    frontier = [target]
    while frontier:
        state = frontier.pop(0)
        next_depth = seen[state] + 1
        for gen in inverse_generators:
            prev = apply_generator(state, gen)
            if prev in state_set and prev not in seen:
                seen[prev] = next_depth
                frontier.append(prev)
    return seen


def _target_distance_positions(
    states: list[State],
    distances: dict[State, int],
    spec: GraphSpec,
    target: State | None,
) -> dict[State, tuple[float, float]]:
    if target is None or target not in set(states):
        target = max(states, key=lambda state: (distances.get(state, 0), state_key(state)), default=None)
    target_distances = _reverse_distances_to_target(states, spec, target) if target is not None else {}
    max_start = max(distances.values(), default=1)
    max_target = max(target_distances.values(), default=1)
    raw: dict[State, tuple[float, float]] = {}
    for state in states:
        start_distance = distances.get(state, max_start + 1)
        target_distance = target_distances.get(state, max_target + 1)
        raw[state] = (
            start_distance / max(1, max_start),
            -target_distance / max(1, max_target),
        )
    return _normalize_positions(raw)


def _positions_for(
    layout: str,
    states: list[State],
    layers: list[list[State]],
    distances: dict[State, int],
    spec: GraphSpec,
    target_state: State | None = None,
) -> dict[State, tuple[float, float]]:
    normalized = layout.strip().lower().replace("-", "_")
    if normalized == "bruhat":
        return _bruhat_positions(states, distances)
    if normalized == "spectral":
        return _spectral_positions(states, spec, distances)
    if normalized in ("lehmer", "lehmer_projection"):
        return _lehmer_positions(states, distances)
    if normalized in ("coset", "coset_coordinates"):
        return _coset_positions(states, distances)
    if normalized in ("target", "target_distance"):
        return _target_distance_positions(states, distances, spec, target_state)
    if normalized not in ("layers", "radial", "distance"):
        raise ValueError("layout must be layers, bruhat, spectral, lehmer, coset, or target-distance")
    return _layer_positions(layers, distances)


def _nodes_payload(states: list[State], distances: dict[State, int], positions: dict[State, tuple[float, float]]) -> list[dict[str, Any]]:
    return [
        {
            "id": state_key(state),
            "state": list(state),
            "label": display_state(state),
            "distance": distances.get(state),
            "x": positions[state][0],
            "y": positions[state][1],
        }
        for state in states
        if state in positions
    ]


def _edges_payload(
    states: list[State],
    distances: dict[State, int],
    spec: GraphSpec,
    edge_cap: int,
) -> list[dict[str, Any]]:
    state_set = set(states)
    edges: list[dict[str, Any]] = []
    for source in states:
        for gen in build_generators(spec):
            target = apply_generator(source, gen.permutation)
            if target not in state_set:
                continue
            edges.append(
                {
                    "source": state_key(source),
                    "target": state_key(target),
                    "generatorId": gen.id,
                    "generatorLabel": gen.label,
                    "forwardLayer": distances.get(target, 0) == distances.get(source, 0) + 1,
                }
            )
            if len(edges) >= edge_cap:
                return edges
    return edges


def _sample_layers(layers: list[list[State]], cap: int) -> list[State]:
    if sum(len(layer) for layer in layers) <= cap:
        return [state for layer in layers for state in layer]
    per_layer = max(2, cap // max(1, len(layers)))
    sampled: list[State] = []
    for layer in layers:
        if len(layer) <= per_layer:
            sampled.extend(layer)
        else:
            step = max(1, len(layer) // per_layer)
            sampled.extend(layer[::step][:per_layer])
    return sampled[:cap]


def _with_pinned_states(states: list[State], pinned_states: tuple[State, ...], distances: dict[State, int]) -> list[State]:
    state_set = set(states)
    out = list(states)
    for state in pinned_states:
        if state in distances and state not in state_set:
            out.append(state)
            state_set.add(state)
    return sorted(out, key=lambda state: (distances.get(state, 0), state_key(state)))


def _exact_view(request: GraphViewRequest) -> dict[str, Any]:
    result = ensure_exact_bfs(request.spec, cap=request.exact_cap)
    all_states = [state for layer in result.layers for state in layer]
    mode = request.mode
    if mode == "auto":
        mode = "full" if len(all_states) <= request.full_node_cap else "layers"

    if mode == "full" and len(all_states) * len(result.generator_ids) <= request.full_edge_cap and len(all_states) <= request.full_node_cap:
        states = all_states
    else:
        mode = "layers"
        states = _sample_layers(result.layers, request.full_node_cap)
    states = _with_pinned_states(states, request.pinned_states, result.distances)

    layer_subset: list[list[State]] = [[] for _ in result.layers]
    subset = set(states)
    for depth, layer in enumerate(result.layers):
        layer_subset[depth] = [state for state in layer if state in subset]
    positions = _positions_for(request.layout, states, layer_subset, result.distances, request.spec, request.target_state)

    return {
        "kind": mode,
        "layout": request.layout,
        "certified": True,
        "specHash": request.spec.hash(),
        "metadata": result.metadata(),
        "nodes": _nodes_payload(states, result.distances, positions),
        "edges": _edges_payload(states, result.distances, request.spec, request.full_edge_cap),
        "truncated": len(states) < len(all_states),
    }


def _local_view(request: GraphViewRequest) -> dict[str, Any]:
    focus = request.focus_state or initial_state(request.spec)
    generators = build_generators(request.spec)
    distances, _raw_edges = reachable_shell(
        [gen.permutation for gen in generators],
        focus=focus,
        radius=max(1, min(4, request.radius)),
        cap=request.full_node_cap,
    )
    if request.target_state is not None and request.target_state not in distances and len(distances) < request.full_node_cap:
        distances[request.target_state] = max(distances.values(), default=0) + 1
    for pinned in request.pinned_states:
        if pinned not in distances and len(distances) < request.full_node_cap:
            distances[pinned] = max(distances.values(), default=0) + 1
    layers: list[list[State]] = []
    for state, depth in distances.items():
        while len(layers) <= depth:
            layers.append([])
        layers[depth].append(state)
    states = [state for layer in layers for state in layer]
    positions = _positions_for(request.layout, states, layers, distances, request.spec, request.target_state)
    edges = _edges_payload(states, distances, request.spec, request.full_edge_cap)
    return {
        "kind": "local",
        "layout": request.layout,
        "certified": False,
        "specHash": request.spec.hash(),
        "metadata": {
            "nStates": len(states),
            "diameter": max(distances.values(), default=0),
            "layerSizes": [len(layer) for layer in layers],
        },
        "nodes": _nodes_payload(states, distances, positions),
        "edges": edges,
        "truncated": len(states) >= request.full_node_cap,
    }


def graph_view(request: GraphViewRequest) -> dict[str, Any]:
    if request.mode == "local":
        return _local_view(request)
    try:
        return _exact_view(request)
    except BfsLimitExceeded as exc:
        view = _local_view(request)
        view["limitExceeded"] = {"cap": exc.cap, "visited": exc.visited}
        return view


def shortest_paths_to_state(
    spec: GraphSpec,
    target: State,
    layout: str = "layers",
    exact_cap: int = 500_000,
    edge_cap: int = 80_000,
) -> dict[str, Any]:
    result = ensure_exact_bfs(spec, cap=exact_cap)
    if target not in result.distances:
        raise ValueError("target is not reachable in this exact BFS result")

    target_distance = result.distances[target]
    generators = build_generators(spec)
    parents: dict[State, list[tuple[State, str, str]]] = {}
    for source, distance in result.distances.items():
        if distance >= target_distance:
            continue
        for gen in generators:
            nxt = apply_generator(source, gen.permutation)
            if result.distances.get(nxt) == distance + 1 and result.distances[nxt] <= target_distance:
                parents.setdefault(nxt, []).append((source, gen.id, gen.label))

    path_nodes: set[State] = {target}
    path_edges: list[dict[str, Any]] = []
    stack = [target]
    truncated = False
    while stack:
        child = stack.pop()
        for parent, gen_id, gen_label in parents.get(child, []):
            if len(path_edges) >= edge_cap:
                truncated = True
                continue
            path_edges.append(
                {
                    "source": state_key(parent),
                    "target": state_key(child),
                    "generatorId": gen_id,
                    "generatorLabel": gen_label,
                    "forwardLayer": True,
                }
            )
            if parent not in path_nodes:
                path_nodes.add(parent)
                stack.append(parent)

    states = sorted(path_nodes, key=lambda state: (result.distances[state], state_key(state)))
    layers: list[list[State]] = [[] for _ in range(target_distance + 1)]
    for state in states:
        layers[result.distances[state]].append(state)
    positions = _positions_for(layout, states, layers, result.distances, spec, target)

    edge_targets_by_source: dict[State, list[State]] = {}
    for edge in path_edges:
        source = tuple(int(x) for x in edge["source"].split(",")) if edge["source"] else ()
        child = tuple(int(x) for x in edge["target"].split(",")) if edge["target"] else ()
        edge_targets_by_source.setdefault(source, []).append(child)

    count_cap = 10**18
    path_counts: dict[State, int] = {result.start_state: 1}
    count_capped = False
    for state in states:
        current_count = path_counts.get(state, 0)
        if current_count == 0:
            continue
        for child in edge_targets_by_source.get(state, []):
            next_count = path_counts.get(child, 0) + current_count
            if next_count > count_cap:
                next_count = count_cap
                count_capped = True
            path_counts[child] = next_count

    canonical = result.path_to(target)
    edge_keys = [
        f"{edge['source']}|{edge['target']}|{edge['generatorId']}"
        for edge in path_edges
    ]
    return {
        "certified": True,
        "specHash": spec.hash(),
        "layout": layout,
        "start": list(result.start_state),
        "target": list(target),
        "targetId": state_key(target),
        "length": target_distance,
        "pathCount": path_counts.get(target, 1 if target == result.start_state else 0),
        "pathCountCapped": count_capped,
        "canonicalPath": canonical.to_dict(),
        "nodeIds": [state_key(state) for state in states],
        "edgeKeys": edge_keys,
        "nodes": _nodes_payload(states, result.distances, positions),
        "edges": path_edges,
        "truncated": truncated,
    }
