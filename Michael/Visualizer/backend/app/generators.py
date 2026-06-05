from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .models import GeneratorSpec, GraphSpec, State
from .permutations import (
    cycle_permutation,
    inverse_permutation,
    is_involutive,
    prefix_reversal,
    swap_permutation,
    swaps_permutation,
    validate_permutation,
)


Builder = Callable[[int, Mapping[str, Any]], list[GeneratorSpec]]


@dataclass(frozen=True)
class FamilyDef:
    id: str
    label: str
    min_n: int
    default_inverse_policy: str
    description: str
    parameters: tuple[dict[str, Any], ...]
    builder: Builder

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "minN": self.min_n,
            "defaultInversePolicy": self.default_inverse_policy,
            "description": self.description,
            "parameters": list(self.parameters),
        }


def _int_param(params: Mapping[str, Any], name: str, default: int) -> int:
    value = params.get(name, default)
    return int(value)


def _gen(id_: str, label: str, perm: State) -> GeneratorSpec:
    return GeneratorSpec(
        id=id_,
        label=label,
        permutation=validate_permutation(perm, len(perm)),
        involutive=is_involutive(perm),
    )


def _adjacent_transpositions(n: int, params: Mapping[str, Any]) -> list[GeneratorSpec]:
    return [_gen(f"s{i + 1}", f"s{i + 1}", swap_permutation(n, i, i + 1)) for i in range(n - 1)]


def _all_transpositions(n: int, params: Mapping[str, Any]) -> list[GeneratorSpec]:
    return [
        _gen(f"t{i + 1}_{j + 1}", f"({i + 1} {j + 1})", swap_permutation(n, i, j))
        for i in range(n)
        for j in range(i + 1, n)
    ]


def _star_transpositions(n: int, params: Mapping[str, Any]) -> list[GeneratorSpec]:
    return [_gen(f"x{j + 1}", f"(1 {j + 1})", swap_permutation(n, 0, j)) for j in range(1, n)]


def _adjacent_cycles(n: int, params: Mapping[str, Any]) -> list[GeneratorSpec]:
    if n < 3:
        raise ValueError("adjacent cycles require n >= 3")
    return [
        _gen(f"c{i + 1}", f"({i + 1} {i + 2} {i + 3})", cycle_permutation(n, [i, i + 1, i + 2]))
        for i in range(n - 2)
    ]


def _consecutive_k_cycles(n: int, params: Mapping[str, Any]) -> list[GeneratorSpec]:
    k = _int_param(params, "k", 3)
    if not (2 <= k <= n):
        raise ValueError(f"k must be in [2, {n}]")
    return [
        _gen(
            f"c{i + 1}_{k}",
            "(" + " ".join(str(j + 1) for j in range(i, i + k)) + ")",
            cycle_permutation(n, range(i, i + k)),
        )
        for i in range(n - k + 1)
    ]


def _wrapped_k_cycles(n: int, params: Mapping[str, Any]) -> list[GeneratorSpec]:
    k = _int_param(params, "k", 3)
    if not (2 <= k <= n):
        raise ValueError(f"k must be in [2, {n}]")
    gens: list[GeneratorSpec] = []
    for start in range(n):
        positions = [(start + offset) % n for offset in range(k)]
        label = "(" + " ".join(str(j + 1) for j in positions) + ")"
        gens.append(_gen(f"w{start + 1}_{k}", label, cycle_permutation(n, positions)))
    return gens


def _pancake(n: int, params: Mapping[str, Any]) -> list[GeneratorSpec]:
    return [_gen(f"R{k}", f"R{k}", prefix_reversal(n, k)) for k in range(2, n + 1)]


def _koltsov3(n: int, params: Mapping[str, Any]) -> list[GeneratorSpec]:
    if n < 3:
        raise ValueError("Koltsov3 requires n >= 3")
    perm_type = _int_param(params, "permType", _int_param(params, "perm_type", 1))
    k = _int_param(params, "k", 0)

    i_gen = swaps_permutation(n, ((i, i + 1) for i in range(0, n - 1, 2)))
    k_gen = swaps_permutation(n, ((i, i + 1) for i in range(1, n - 1, 2)))

    if perm_type == 1:
        d = _int_param(params, "d", 2)
        if not (0 <= k < n and 1 <= d and k + d < n):
            raise ValueError(f"Koltsov3 type 1 needs 0 <= k and k + d < n; got k={k}, d={d}, n={n}")
        s_gen = swap_permutation(n, k, k + d)
        s_label = f"S({k + 1},{k + d + 1})"
    elif perm_type == 2:
        if not (0 <= k and k + 3 < n):
            raise ValueError(f"Koltsov3 type 2 needs k + 3 < n; got k={k}, n={n}")
        s_gen = swaps_permutation(n, [(k, k + 3), (k + 1, k + 2)])
        s_label = f"S({k + 1},{k + 4})({k + 2},{k + 3})"
    else:
        raise ValueError("Koltsov3 permType must be 1 or 2")

    return [_gen("I", "I", i_gen), _gen("K", "K", k_gen), _gen("S", s_label, s_gen)]


def _cubic_pancake(n: int, params: Mapping[str, Any]) -> list[GeneratorSpec]:
    if n < 4:
        raise ValueError("cubic pancake requires n >= 4")
    fixed = _int_param(params, "fixed", 2)
    default_k = 3 if fixed != 3 else 2
    k = _int_param(params, "k", default_k)
    if fixed not in (2, 3):
        raise ValueError("fixed must be 2 or 3")
    if not (2 <= k <= n):
        raise ValueError(f"k must be in [2, {n}]")
    if len({n, fixed, k}) != 3:
        raise ValueError("R_n, R_fixed, and R_k must be distinct")
    return [
        _gen(f"R{n}", f"R{n}", prefix_reversal(n, n)),
        _gen(f"R{fixed}", f"R{fixed}", prefix_reversal(n, fixed)),
        _gen(f"R{k}", f"R{k}", prefix_reversal(n, k)),
    ]


def _lrx(n: int, params: Mapping[str, Any]) -> list[GeneratorSpec]:
    if n < 2:
        raise ValueError("LRX requires n >= 2")
    left = tuple(list(range(1, n)) + [0])
    right = inverse_permutation(left)
    exchange = swap_permutation(n, 0, 1)
    return [_gen("L", "L", left), _gen("R", "R", right), _gen("X", "X", exchange)]


FAMILIES: dict[str, FamilyDef] = {
    "adjacent_transpositions": FamilyDef(
        id="adjacent_transpositions",
        label="Adjacent transpositions",
        min_n=2,
        default_inverse_policy="listed",
        description="s_i swaps adjacent positions i and i+1.",
        parameters=(),
        builder=_adjacent_transpositions,
    ),
    "all_transpositions": FamilyDef(
        id="all_transpositions",
        label="All transpositions",
        min_n=2,
        default_inverse_policy="listed",
        description="All swaps (i j).",
        parameters=(),
        builder=_all_transpositions,
    ),
    "star_transpositions": FamilyDef(
        id="star_transpositions",
        label="Star transpositions",
        min_n=2,
        default_inverse_policy="listed",
        description="Swaps (1 i).",
        parameters=(),
        builder=_star_transpositions,
    ),
    "adjacent_cycles": FamilyDef(
        id="adjacent_cycles",
        label="Adjacent 3-cycles",
        min_n=3,
        default_inverse_policy="closed",
        description="Consecutive 3-cycles with inverse closure by default.",
        parameters=(),
        builder=_adjacent_cycles,
    ),
    "consecutive_k_cycles": FamilyDef(
        id="consecutive_k_cycles",
        label="Consecutive k-cycles",
        min_n=2,
        default_inverse_policy="closed",
        description="Cycles on consecutive windows.",
        parameters=({"id": "k", "label": "k", "type": "integer", "default": 3, "min": 2},),
        builder=_consecutive_k_cycles,
    ),
    "wrapped_k_cycles": FamilyDef(
        id="wrapped_k_cycles",
        label="Wrapped k-cycles",
        min_n=2,
        default_inverse_policy="closed",
        description="Cycles on wrapped windows.",
        parameters=({"id": "k", "label": "k", "type": "integer", "default": 3, "min": 2},),
        builder=_wrapped_k_cycles,
    ),
    "pancake": FamilyDef(
        id="pancake",
        label="Pancake",
        min_n=2,
        default_inverse_policy="listed",
        description="All prefix reversals R2..Rn.",
        parameters=(),
        builder=_pancake,
    ),
    "koltsov3": FamilyDef(
        id="koltsov3",
        label="Koltsov3",
        min_n=3,
        default_inverse_policy="listed",
        description="I, K, S with Koltsov type 1 or 2.",
        parameters=(
            {"id": "permType", "label": "type", "type": "select", "default": 1, "options": [1, 2]},
            {"id": "k", "label": "k", "type": "integer", "default": 0, "min": 0},
            {"id": "d", "label": "d", "type": "integer", "default": 2, "min": 1},
        ),
        builder=_koltsov3,
    ),
    "cubic_pancake": FamilyDef(
        id="cubic_pancake",
        label="Cubic pancake",
        min_n=4,
        default_inverse_policy="listed",
        description="Three prefix reversals Rn, Rf, Rk.",
        parameters=(
            {"id": "fixed", "label": "f", "type": "select", "default": 2, "options": [2, 3]},
            {"id": "k", "label": "k", "type": "integer", "default": 3, "min": 2},
        ),
        builder=_cubic_pancake,
    ),
    "lrx": FamilyDef(
        id="lrx",
        label="LRX",
        min_n=2,
        default_inverse_policy="listed",
        description="Left shift, right shift, and first-position exchange.",
        parameters=(),
        builder=_lrx,
    ),
}


def list_families() -> list[dict[str, Any]]:
    return [family.to_dict() for family in FAMILIES.values()]


def build_generators(spec: GraphSpec) -> list[GeneratorSpec]:
    if spec.family not in FAMILIES:
        raise ValueError(f"unknown family: {spec.family}")
    family = FAMILIES[spec.family]
    if spec.n < family.min_n:
        raise ValueError(f"{family.label} requires n >= {family.min_n}")

    base = family.builder(spec.n, spec.params)
    inverse_policy = spec.inverse_policy
    if inverse_policy == "default":
        inverse_policy = family.default_inverse_policy
    if inverse_policy not in ("listed", "closed"):
        raise ValueError("inverse policy must be default, listed, or closed")
    if inverse_policy == "listed":
        return base

    existing = {g.permutation for g in base}
    out = list(base)
    for gen in base:
        if gen.involutive:
            continue
        inv = inverse_permutation(gen.permutation)
        if inv in existing:
            continue
        existing.add(inv)
        out.append(
            GeneratorSpec(
                id=f"{gen.id}_inv",
                label=f"{gen.label}^-1",
                permutation=inv,
                involutive=False,
                inverse_of=gen.id,
            )
        )
    return out


def generator_map(spec: GraphSpec) -> dict[str, GeneratorSpec]:
    return {gen.id: gen for gen in build_generators(spec)}

