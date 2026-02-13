"""Generator configuration for the experiment framework.

Defines ParamSpec (describes a single parameter) and GeneratorConfig
(describes a generator type). Built-in configs for Koltsov3, consecutive
k-cycles, and wrapped k-cycles are provided at module level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from cayleypy import PermutationGroups

# Type alias: factory function takes (n, **params) -> cayleypy group definition
GroupFactory = Callable[..., Any]


@dataclass(frozen=True)
class ParamSpec:
    """Specification for a single parameter in the generator's parameter space.

    Attributes:
        name: Parameter name (e.g., "k", "d"). Must match the kwarg
              name passed to the factory function.
        default_min: Fallback minimum when not overridden at run time.
        default_max: Fallback maximum. None means "use max_n - 1".
    """
    name: str
    default_min: int = 0
    default_max: Optional[int] = None


@dataclass(frozen=True)
class GeneratorConfig:
    """Complete configuration for a generator type.

    This is the single object provided to define a new
    generator type. It describes how to create the group definition,
    what parameters exist, and how to validate parameter combinations.

    Attributes:
        name: Human-readable name used in plot titles and directory names.
        factory: Callable that creates a cayleypy group definition.
                 Signature: factory(n, **params) -> group_definition.
        params: Ordered list of ParamSpec objects. Iteration order
                matches list order. "n" is implicit and not included.
        is_valid: Optional callable (n, params_dict) -> bool.
                  Return False to skip invalid combinations.
        make_inverse_closed: If True, calls .make_inverse_closed() on
                             the group definition after factory creates it.
        description: Optional longer description for documentation.
    """
    name: str
    factory: GroupFactory
    params: List[ParamSpec] = field(default_factory=list)
    is_valid: Optional[Callable[[int, Dict[str, int]], bool]] = None
    make_inverse_closed: bool = False
    description: str = ""

    @property
    def param_names(self) -> List[str]:
        return [p.name for p in self.params]

    @property
    def cache_key_columns(self) -> List[str]:
        return ["coset"] + self.param_names + ["n"]

    @property
    def sort_columns(self) -> List[str]:
        return self.cache_key_columns


# =========================================================================
# Factory functions
# =========================================================================

def _koltsov3_perm1_factory(n: int, k: int = 0, d: int = 1) -> Any:
    return PermutationGroups.koltsov3(n, perm_type=1, k=k, d=d)


def _koltsov3_perm2_factory(n: int, k: int = 0) -> Any:
    return PermutationGroups.koltsov3(n, perm_type=2, k=k)


def _consecutive_k_cycles_factory(n: int, k: int = 2) -> Any:
    return PermutationGroups.consecutive_k_cycles(n, k)


def _wrapped_k_cycles_factory(n: int, k: int = 2) -> Any:
    return PermutationGroups.wrapped_k_cycles(n, k)


# =========================================================================
# Built-in generator configs
# =========================================================================

KOLTSOV3_PERM1 = GeneratorConfig(
    name="koltsov3_perm1",
    factory=_koltsov3_perm1_factory,
    params=[
        ParamSpec(name="k", default_min=0),
        ParamSpec(name="d", default_min=1),
    ],
    is_valid=lambda n, p: p["k"] >= 0 and p["d"] >= 1 and p["k"] + p["d"] < n,
    description="Koltsov3 perm_type=1: S generator is transposition (k, k+d).",
)

KOLTSOV3_PERM2 = GeneratorConfig(
    name="koltsov3_perm2",
    factory=_koltsov3_perm2_factory,
    params=[
        ParamSpec(name="k", default_min=0, default_max=None),
    ],
    is_valid=lambda n, p: p["k"] >= 0 and p["k"] + 3 < n,
    description="Koltsov3 perm_type=2: S generator is (k,k+3)(k+1,k+2).",
)

CONSECUTIVE_K_CYCLES = GeneratorConfig(
    name="consecutive_k_cycles",
    factory=_consecutive_k_cycles_factory,
    params=[
        ParamSpec(name="k", default_min=2, default_max=None),
    ],
    is_valid=lambda n, p: p["k"] >= 2 and p["k"] < n,
    description="Consecutive k-cycles generator.",
)

WRAPPED_K_CYCLES = GeneratorConfig(
    name="wrapped_k_cycles",
    factory=_wrapped_k_cycles_factory,
    params=[
        ParamSpec(name="k", default_min=2, default_max=None),
    ],
    is_valid=lambda n, p: p["k"] >= 2 and p["k"] < n,
    description="Wrapped consecutive k-cycles generator.",
)

CONSECUTIVE_K_CYCLES_INV = GeneratorConfig(
    name="consecutive_k_cycles_inv",
    factory=_consecutive_k_cycles_factory,
    params=[
        ParamSpec(name="k", default_min=2, default_max=None),
    ],
    is_valid=lambda n, p: p["k"] >= 2 and p["k"] < n,
    make_inverse_closed=True,
    description="Consecutive k-cycles generator (inverse closed).",
)

WRAPPED_K_CYCLES_INV = GeneratorConfig(
    name="wrapped_k_cycles_inv",
    factory=_wrapped_k_cycles_factory,
    params=[
        ParamSpec(name="k", default_min=2, default_max=None),
    ],
    is_valid=lambda n, p: p["k"] >= 2 and p["k"] < n,
    make_inverse_closed=True,
    description="Wrapped consecutive k-cycles generator (inverse closed).",
)

# Registry for lookup by name
GENERATOR_REGISTRY: Dict[str, GeneratorConfig] = {
    "koltsov3_perm1": KOLTSOV3_PERM1,
    "koltsov3_perm2": KOLTSOV3_PERM2,
    "consecutive_k_cycles": CONSECUTIVE_K_CYCLES,
    "wrapped_k_cycles": WRAPPED_K_CYCLES,
    "consecutive_k_cycles_inv": CONSECUTIVE_K_CYCLES_INV,
    "wrapped_k_cycles_inv": WRAPPED_K_CYCLES_INV,
}
