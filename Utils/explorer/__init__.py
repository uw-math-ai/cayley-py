"""Generalized experiment framework for Cayley graph research.

Usage:
    from explorer import Explorer
    from explorer.config import KOLTSOV3_PERM2

    exp = Explorer(KOLTSOV3_PERM2, output_dir="results", max_n=25)
    df = exp.run_and_save("different")
"""

__version__ = "0.2.0"

from .explorer import Explorer, ExperimentResult
from .config import (
    GeneratorConfig,
    ParamSpec,
    KOLTSOV3_PERM1,
    KOLTSOV3_PERM2,
    CONSECUTIVE_K_CYCLES,
    WRAPPED_K_CYCLES,
    CONSECUTIVE_K_CYCLES_INV,
    WRAPPED_K_CYCLES_INV,
    CUBIC_PANCAKE,
    GENERATOR_REGISTRY,
)
from .coset_groups import (
    COSET_GROUPS,
    CosetFunc,
    list_groups,
    list_cosets,
    get_coset_func,
    print_summary,
)

__all__ = [
    "Explorer",
    "ExperimentResult",
    "GeneratorConfig",
    "ParamSpec",
    "KOLTSOV3_PERM1",
    "KOLTSOV3_PERM2",
    "CONSECUTIVE_K_CYCLES",
    "WRAPPED_K_CYCLES",
    "CONSECUTIVE_K_CYCLES_INV",
    "WRAPPED_K_CYCLES_INV",
    "CUBIC_PANCAKE",
    "GENERATOR_REGISTRY",
    "COSET_GROUPS",
    "CosetFunc",
    "list_groups",
    "list_cosets",
    "get_coset_func",
    "print_summary",
]
