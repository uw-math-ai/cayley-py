"""
Koltsov Explorer - A library for Koltsov3 parameter space exploration.

This library provides tools for systematically exploring Koltsov3 generator
parameter spaces on Cayley graphs and Schreier coset graphs.

Example:
    >>> from koltsov_explorer import KoltsovExplorer, COSET_GROUPS
    >>>
    >>> # Create explorer for perm_type=2
    >>> explorer = KoltsovExplorer(perm_type=2, output_dir="results")
    >>>
    >>> # Run experiments and save results
    >>> df = explorer.run_and_save("different", min_n=4, max_n=20)
    >>>
    >>> # Or step by step:
    >>> results = explorer.run_group("different", min_n=4, max_n=20)
    >>> df = explorer.save_results("different", results)
    >>> explorer.plot_results("different", df)
"""

from .explorer import KoltsovExplorer, ExperimentResult
from .coset_groups import (
    COSET_GROUPS,
    list_groups,
    list_cosets,
    get_coset_func,
    print_summary,
)

__version__ = "0.1.0"
__all__ = [
    "KoltsovExplorer",
    "ExperimentResult",
    "COSET_GROUPS",
    "list_groups",
    "list_cosets",
    "get_coset_func",
    "print_summary",
]
