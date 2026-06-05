from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping


State = tuple[int, ...]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class GraphSpec:
    family: str
    n: int
    params: Mapping[str, Any] = field(default_factory=dict)
    inverse_policy: str = "default"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraphSpec":
        return cls(
            family=str(data.get("family", "")),
            n=int(data.get("n", 0)),
            params=dict(data.get("params") or {}),
            inverse_policy=str(data.get("inversePolicy", data.get("inverse_policy", "default"))),
        )

    def normalized(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "n": self.n,
            "params": _jsonable(self.params),
            "inversePolicy": self.inverse_policy,
        }

    def hash(self) -> str:
        payload = json.dumps(self.normalized(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def to_dict(self) -> dict[str, Any]:
        return self.normalized()


@dataclass(frozen=True)
class GeneratorSpec:
    id: str
    label: str
    permutation: State
    involutive: bool = False
    inverse_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "permutation": list(self.permutation),
            "involutive": self.involutive,
            "inverseOf": self.inverse_of,
        }


@dataclass(frozen=True)
class GraphViewRequest:
    spec: GraphSpec
    mode: str = "auto"
    layout: str = "layers"
    focus_state: State | None = None
    target_state: State | None = None
    pinned_states: tuple[State, ...] = ()
    path: tuple[str, ...] = ()
    radius: int = 2
    exact_cap: int = 500_000
    full_node_cap: int = 15_000
    full_edge_cap: int = 80_000

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraphViewRequest":
        focus = data.get("focusState", data.get("focus_state"))
        target = data.get("targetState", data.get("target_state"))
        pinned = data.get("pinnedStates", data.get("pinned_states", ()))
        return cls(
            spec=GraphSpec.from_dict(data.get("spec") or {}),
            mode=str(data.get("mode", "auto")),
            layout=str(data.get("layout", "layers")),
            focus_state=tuple(int(x) for x in focus) if focus is not None else None,
            target_state=tuple(int(x) for x in target) if target is not None else None,
            pinned_states=tuple(tuple(int(x) for x in state) for state in pinned),
            path=tuple(str(x) for x in data.get("path", ())),
            radius=int(data.get("radius", 2)),
            exact_cap=int(data.get("exactCap", data.get("exact_cap", 500_000))),
            full_node_cap=int(data.get("fullNodeCap", data.get("full_node_cap", 15_000))),
            full_edge_cap=int(data.get("fullEdgeCap", data.get("full_edge_cap", 80_000))),
        )


@dataclass(frozen=True)
class CertifiedPath:
    target: State
    generator_ids: tuple[str, ...]
    states: tuple[State, ...]
    length: int
    certified: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": list(self.target),
            "generatorIds": list(self.generator_ids),
            "states": [list(s) for s in self.states],
            "length": self.length,
            "certified": self.certified,
        }


@dataclass
class ChallengeState:
    session_id: str
    spec: GraphSpec
    target: State
    current: State
    target_distance: int
    user_path: list[str] = field(default_factory=list)
    user_states: list[State] = field(default_factory=list)
    status: str = "active"
    certified_path: CertifiedPath | None = None
    generators: tuple[GeneratorSpec, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "spec": self.spec.to_dict(),
            "specHash": self.spec.hash(),
            "target": list(self.target),
            "current": list(self.current),
            "start": list(self.user_states[0]) if self.user_states else list(self.current),
            "userStates": [list(state) for state in self.user_states],
            "targetDistance": self.target_distance,
            "userPath": list(self.user_path),
            "userLength": len(self.user_path),
            "status": self.status,
            "certifiedPath": self.certified_path.to_dict() if self.certified_path else None,
            "optimalLength": self.certified_path.length if self.certified_path else None,
            "excess": len(self.user_path) - self.certified_path.length if self.certified_path else None,
            "generators": [gen.to_dict() for gen in self.generators],
        }
