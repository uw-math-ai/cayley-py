"""Main KoltsovExplorer class for running experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from cayleypy import CayleyGraph, PermutationGroups

from .coset_groups import COSET_GROUPS, CosetFunc
from .storage import get_computed_combinations, load_results, save_results
from .plotting import plot_group_results


@dataclass
class ExperimentResult:
    """Result of a single BFS experiment."""
    n: int
    k: int
    d: Optional[int]
    coset: str
    diameter: int
    growth: List[int]
    last_layer_size: int
    total_states: int
    central: Optional[List[int]]


class KoltsovExplorer:
    """
    Explorer for Koltsov3 parameter space experiments.

    Supports both perm_type=1 (with d parameter) and perm_type=2 (no d parameter).
    Results are stored in CSV format with incremental caching.

    Example:
        >>> explorer = KoltsovExplorer(perm_type=2, output_dir="my_results")
        >>> results = explorer.run_group("different", min_n=4, max_n=20)
        >>> df = explorer.save_results("different", results)
        >>> explorer.plot_results("different", df)

        # Or all-in-one:
        >>> df = explorer.run_and_save("different", min_n=4, max_n=20)
    """

    def __init__(
        self,
        perm_type: int = 2,
        output_dir: str = "results",
        min_n: int = 4,
        max_n: int = 30,
        max_d: int = 10,
    ):
        """
        Initialize KoltsovExplorer.

        Args:
            perm_type: Type of Koltsov3 generator (1 or 2).
                - perm_type=1: S generator is transposition (k, k+d), requires d param
                - perm_type=2: S generator is (k,k+3)(k+1,k+2), no d param
            output_dir: Base directory for storing results.
            min_n: Default minimum n value for experiments.
            max_n: Default maximum n value for experiments.
            max_d: Maximum d value for perm_type=1 experiments.
        """
        if perm_type not in (1, 2):
            raise ValueError(f"perm_type must be 1 or 2, got {perm_type}")

        self.perm_type = perm_type
        self.output_dir = output_dir
        self.min_n = min_n
        self.max_n = max_n
        self.max_d = max_d

        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_group_dir(self, group_name: str) -> str:
        """Get output directory for a specific group."""
        group_dir = os.path.join(self.output_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)
        return group_dir

    def is_valid_params(self, n: int, k: int, d: Optional[int] = None) -> bool:
        """
        Check if parameters are valid for current perm_type.

        Args:
            n: Permutation size.
            k: k parameter for S generator.
            d: d parameter (only for perm_type=1).

        Returns:
            True if parameters are valid, False otherwise.
        """
        if self.perm_type == 1:
            if d is None:
                return False
            return k >= 0 and d >= 1 and k + d < n
        else:  # perm_type == 2
            return k >= 0 and k + 3 < n

    def _is_valid_central(self, central: Optional[List[int]]) -> bool:
        """Check if central state has >1 unique value."""
        return central is not None and len(np.unique(central)) > 1

    def run_single_experiment(
        self,
        n: int,
        k: int,
        coset_name: str,
        coset_func: CosetFunc,
        d: Optional[int] = None,
    ) -> Optional[ExperimentResult]:
        """
        Run a single BFS experiment.

        Args:
            n: Permutation size.
            k: k parameter for S generator.
            coset_name: Name of the coset type.
            coset_func: Function that generates central state for given n.
            d: d parameter (required for perm_type=1, ignored for perm_type=2).

        Returns:
            ExperimentResult if successful, None if invalid or error.
        """
        if not self.is_valid_params(n, k, d):
            return None

        try:
            if self.perm_type == 1:
                defn = PermutationGroups.koltsov3(n, perm_type=1, k=k, d=d)
            else:
                defn = PermutationGroups.koltsov3(n, perm_type=2, k=k)

            central = coset_func(n)
            if coset_name != "FullGraph":
                if not self._is_valid_central(central):
                    return None
                defn = defn.with_central_state(central)

            graph = CayleyGraph(defn)
            result = graph.bfs(return_all_edges=False, return_all_hashes=False)

            return ExperimentResult(
                n=n,
                k=k,
                d=d,
                coset=coset_name,
                diameter=result.diameter(),
                growth=result.layer_sizes,
                last_layer_size=len(result.last_layer()),
                total_states=sum(result.layer_sizes),
                central=central,
            )
        except Exception as e:
            print(f"Error: {coset_name}, k={k}, d={d}, n={n}: {e}")
            return None

    def _filter_cosets(
        self,
        cosets: Dict[str, CosetFunc],
        coset_filter: Optional[Union[str, List[str]]],
    ) -> Dict[str, CosetFunc]:
        """Filter cosets based on coset_filter parameter."""
        if coset_filter is None:
            return cosets
        if isinstance(coset_filter, str):
            if coset_filter not in cosets:
                raise ValueError(f"Coset '{coset_filter}' not found. Available: {list(cosets.keys())}")
            return {coset_filter: cosets[coset_filter]}
        # List case
        filtered = {k: v for k, v in cosets.items() if k in coset_filter}
        missing = set(coset_filter) - set(filtered.keys())
        if missing:
            raise ValueError(f"Cosets not found: {missing}. Available: {list(cosets.keys())}")
        return filtered

    def run_group(
        self,
        group_name: str,
        min_n: Optional[int] = None,
        max_n: Optional[int] = None,
        k_range: Optional[tuple] = None,
        d_range: Optional[tuple] = None,
        coset_filter: Optional[Union[str, List[str]]] = None,
        skip_computed: bool = True,
    ) -> Dict:
        """
        Run experiments for a group of cosets.

        Args:
            group_name: Name of coset group (e.g., "different", "then").
            min_n: Minimum n value (defaults to instance config).
            max_n: Maximum n value (defaults to instance config).
            k_range: Tuple (k_min, k_max) or None for full range.
            d_range: Tuple (d_min, d_max) for perm_type=1, or None for default.
            coset_filter: Filter cosets - None (all), str (single), or list.
            skip_computed: If True, skip already computed combinations.

        Returns:
            Dictionary with metadata and results.
        """
        # Use instance defaults if not specified
        min_n = min_n if min_n is not None else self.min_n
        max_n = max_n if max_n is not None else self.max_n

        if group_name not in COSET_GROUPS:
            raise ValueError(f"Unknown group: {group_name}. Available: {list(COSET_GROUPS.keys())}")

        cosets = self._filter_cosets(COSET_GROUPS[group_name], coset_filter)
        group_dir = self._get_group_dir(group_name)
        computed = get_computed_combinations(group_dir, self.perm_type) if skip_computed else set()

        # Build parameter ranges
        k_values = list(range(k_range[0], k_range[1] + 1)) if k_range else list(range(max_n))

        if self.perm_type == 1:
            d_values = list(range(d_range[0], d_range[1] + 1)) if d_range else list(range(1, self.max_d + 1))
        else:
            d_values = [None]  # perm_type=2 doesn't use d

        results = {
            "metadata": {
                "perm_type": self.perm_type,
                "group": group_name,
                "timestamp": datetime.now().isoformat(),
                "n_range": [min_n, max_n],
                "k_range": list(k_range) if k_range else None,
                "d_range": list(d_range) if d_range else None,
                "coset_filter": coset_filter,
            },
            "results": {},
        }

        total_new = 0

        for coset_name, coset_func in cosets.items():
            skipped = 0
            computed_count = 0
            results["results"][coset_name] = {}

            # Create progress bar
            total_iterations = len(d_values) * len(k_values) * (max_n - min_n + 1)
            pbar = tqdm(total=total_iterations, desc=f"{coset_name}", leave=True)

            for d in d_values:
                if self.perm_type == 1:
                    d_key = f"d={d}"
                    results["results"][coset_name][d_key] = {}

                for k in k_values:
                    k_key = f"k={k}"
                    if self.perm_type == 1:
                        results["results"][coset_name][d_key][k_key] = {}
                    else:
                        results["results"][coset_name][k_key] = {}

                    for n in range(min_n, max_n + 1):
                        if self.perm_type == 1:
                            pbar.set_postfix({"d": d, "k": k, "n": n})
                        else:
                            pbar.set_postfix({"k": k, "n": n})
                        pbar.update(1)

                        if not self.is_valid_params(n, k, d):
                            continue

                        # Check cache
                        if self.perm_type == 1:
                            cache_key = (coset_name, d, k, n)
                        else:
                            cache_key = (coset_name, k, n)

                        if cache_key in computed:
                            skipped += 1
                            continue

                        result = self.run_single_experiment(n, k, coset_name, coset_func, d)
                        if result is not None:
                            result_dict = {
                                "diameter": result.diameter,
                                "growth": result.growth,
                                "last_layer_size": result.last_layer_size,
                            }
                            if self.perm_type == 1:
                                results["results"][coset_name][d_key][k_key][f"n={n}"] = result_dict
                            else:
                                results["results"][coset_name][k_key][f"n={n}"] = result_dict
                            computed_count += 1

            pbar.close()
            total_new += computed_count
            print(f"  {coset_name}: Skipped {skipped} cached, computed {computed_count} new")

        print(f"Completed {group_name} ({total_new} new results)")
        return results

    def save_results(
        self,
        group_name: str,
        results: Dict,
        append: bool = True,
    ) -> pd.DataFrame:
        """
        Save results to CSV.

        Args:
            group_name: Name of coset group.
            results: Results dictionary from run_group().
            append: If True, append to existing CSV; if False, overwrite.

        Returns:
            DataFrame with all results.
        """
        group_dir = self._get_group_dir(group_name)
        cosets = COSET_GROUPS.get(group_name, {})
        return save_results(group_dir, results, cosets, self.perm_type, append)

    def load_results(self, group_name: str) -> pd.DataFrame:
        """
        Load results from CSV.

        Args:
            group_name: Name of coset group.

        Returns:
            DataFrame with parsed results.
        """
        group_dir = self._get_group_dir(group_name)
        return load_results(group_dir)

    def plot_results(
        self,
        group_name: str,
        df: Optional[pd.DataFrame] = None,
        show: bool = True,
    ) -> None:
        """
        Create interactive Plotly plots for results.

        Args:
            group_name: Name of coset group.
            df: DataFrame to plot (if None, loads from CSV).
            show: If True, display plots in notebook.
        """
        if df is None:
            df = self.load_results(group_name)

        group_dir = self._get_group_dir(group_name)
        plot_group_results(group_dir, df, self.perm_type, group_name, show)

    def run_and_save(
        self,
        group_name: str,
        plot: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Convenience method: run experiments, save, and optionally plot.

        Args:
            group_name: Name of coset group.
            plot: If True, generate plots after saving.
            **kwargs: Additional arguments passed to run_group().

        Returns:
            DataFrame with all results.
        """
        results = self.run_group(group_name, **kwargs)
        df = self.save_results(group_name, results)
        if plot:
            self.plot_results(group_name, df)
        return df

    def __repr__(self) -> str:
        return (
            f"KoltsovExplorer(perm_type={self.perm_type}, "
            f"output_dir='{self.output_dir}', "
            f"min_n={self.min_n}, max_n={self.max_n}"
            + (f", max_d={self.max_d})" if self.perm_type == 1 else ")")
        )
