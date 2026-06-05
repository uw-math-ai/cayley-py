from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

from .generators import build_generators
from .models import CertifiedPath, GraphSpec, State
from .permutations import apply_generator, identity, initial_state, parse_state_key, state_key


DEFAULT_EXACT_CAP = 500_000
CACHE_ROOT = Path(__file__).resolve().parents[2] / ".cache" / "bfs"


class BfsLimitExceeded(RuntimeError):
    def __init__(self, cap: int, visited: int):
        super().__init__(f"exact BFS exceeded cap={cap} after visiting {visited} states")
        self.cap = cap
        self.visited = visited


@dataclass
class BfsResult:
    spec_hash: str
    spec_normalized: dict
    generator_ids: tuple[str, ...]
    generator_labels: dict[str, str]
    start_state: State
    distances: dict[State, int]
    predecessors: dict[State, tuple[State, str]]
    layers: list[list[State]]
    generated_at: str

    @property
    def n_states(self) -> int:
        return len(self.distances)

    @property
    def diameter(self) -> int:
        return len(self.layers) - 1

    def metadata(self) -> dict:
        return {
            "specHash": self.spec_hash,
            "spec": self.spec_normalized,
            "generatorIds": list(self.generator_ids),
            "generatorLabels": dict(self.generator_labels),
            "startState": list(self.start_state),
            "nStates": self.n_states,
            "diameter": self.diameter,
            "layerSizes": [len(layer) for layer in self.layers],
            "generatedAt": self.generated_at,
        }

    def path_to(self, target: State) -> CertifiedPath:
        if target not in self.distances:
            raise ValueError("target is not reachable in this BFS result")
        states: list[State] = [target]
        moves: list[str] = []
        cursor = target
        start = self.start_state
        while cursor != start:
            parent, gen_id = self.predecessors[cursor]
            moves.append(gen_id)
            states.append(parent)
            cursor = parent
        states.reverse()
        moves.reverse()
        return CertifiedPath(
            target=target,
            generator_ids=tuple(moves),
            states=tuple(states),
            length=len(moves),
            certified=True,
        )


def _cache_path(spec_hash: str) -> Path:
    return CACHE_ROOT / f"{spec_hash}.npz"


def exact_cache_exists(spec: GraphSpec) -> bool:
    return _cache_path(spec.hash()).exists()


def read_cache_metadata(spec_hash: str) -> dict | None:
    path = _cache_path(spec_hash)
    if not path.exists():
        return None
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return json.loads(zf.read("metadata.json").decode("utf-8"))
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError):
        return None


def load_exact_bfs(spec: GraphSpec) -> BfsResult:
    path = _cache_path(spec.hash())
    with zipfile.ZipFile(path, "r") as zf:
        metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
        distances_raw = json.loads(zf.read("distances.json").decode("utf-8"))
        predecessors_raw = json.loads(zf.read("predecessors.json").decode("utf-8"))
        layers_raw = json.loads(zf.read("layers.json").decode("utf-8"))

    distances = {parse_state_key(key): int(value) for key, value in distances_raw.items()}
    start_state = tuple(int(x) for x in metadata.get("startState", identity(len(next(iter(distances), ())))))
    predecessors = {
        parse_state_key(key): (parse_state_key(value["parent"]), str(value["generator"]))
        for key, value in predecessors_raw.items()
    }
    layers = [[parse_state_key(key) for key in layer] for layer in layers_raw]
    return BfsResult(
        spec_hash=metadata["specHash"],
        spec_normalized=metadata["spec"],
        generator_ids=tuple(metadata["generatorIds"]),
        generator_labels={str(k): str(v) for k, v in metadata["generatorLabels"].items()},
        start_state=start_state,
        distances=distances,
        predecessors=predecessors,
        layers=layers,
        generated_at=str(metadata["generatedAt"]),
    )


def save_exact_bfs(result: BfsResult) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    path = _cache_path(result.spec_hash)
    distances = {state_key(state): distance for state, distance in result.distances.items()}
    predecessors = {
        state_key(state): {"parent": state_key(parent), "generator": gen_id}
        for state, (parent, gen_id) in result.predecessors.items()
    }
    layers = [[state_key(state) for state in layer] for layer in result.layers]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(result.metadata(), sort_keys=True))
        zf.writestr("distances.json", json.dumps(distances, sort_keys=True))
        zf.writestr("predecessors.json", json.dumps(predecessors, sort_keys=True))
        zf.writestr("layers.json", json.dumps(layers))
    return path


def compute_exact_bfs(spec: GraphSpec, cap: int = DEFAULT_EXACT_CAP) -> BfsResult:
    generators = build_generators(spec)
    start = initial_state(spec)
    distances: dict[State, int] = {start: 0}
    predecessors: dict[State, tuple[State, str]] = {}
    layers: list[list[State]] = [[start]]
    queue: deque[State] = deque([start])

    while queue:
        state = queue.popleft()
        next_depth = distances[state] + 1
        for gen in generators:
            nxt = apply_generator(state, gen.permutation)
            if nxt in distances:
                continue
            if len(distances) >= cap:
                raise BfsLimitExceeded(cap=cap, visited=len(distances))
            distances[nxt] = next_depth
            predecessors[nxt] = (state, gen.id)
            while len(layers) <= next_depth:
                layers.append([])
            layers[next_depth].append(nxt)
            queue.append(nxt)

    return BfsResult(
        spec_hash=spec.hash(),
        spec_normalized=spec.normalized(),
        generator_ids=tuple(gen.id for gen in generators),
        generator_labels={gen.id: gen.label for gen in generators},
        start_state=start,
        distances=distances,
        predecessors=predecessors,
        layers=layers,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def ensure_exact_bfs(
    spec: GraphSpec,
    cap: int = DEFAULT_EXACT_CAP,
    use_cache: bool = True,
) -> BfsResult:
    if use_cache and exact_cache_exists(spec):
        try:
            return load_exact_bfs(spec)
        except (zipfile.BadZipFile, KeyError, json.JSONDecodeError):
            pass
    result = compute_exact_bfs(spec, cap=cap)
    save_exact_bfs(result)
    return result
