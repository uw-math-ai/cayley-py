"""Generalized experiment explorer for Cayley graph research."""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from cayleypy import CayleyGraph

from .config import GeneratorConfig
from .coset_groups import COSET_GROUPS, CosetFunc
from .storage import get_computed_combinations, load_results, save_results
from .plotting import plot_group_results


def _init_gpu_worker(gpu_queue: "multiprocessing.Queue") -> None:
    """Assign each spawned worker process a dedicated GPU.

    Called once per worker at startup. Sets CUDA_VISIBLE_DEVICES before any
    CUDA initialization so that device="auto" in CayleyGraph sees only the
    assigned GPU. Workers with index >= num_gpus share GPUs round-robin.
    """
    gpu_id = gpu_queue.get()
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


@dataclass
class ExperimentResult:
    """Result of a single BFS experiment.

    The params dict holds all generator-specific parameters
    (e.g. {"k": 3} or {"k": 1, "d": 5}).
    """
    n: int
    params: Dict[str, int]
    coset: str
    diameter: int
    growth: List[int]
    last_layer_size: int
    total_states: int
    central: Optional[List[int]]


class Explorer:
    """
    Generalized explorer for Cayley graph parameter space experiments.

    Works with any generator type defined by a GeneratorConfig.

    Example:
        >>> from explorer import Explorer
        >>> from explorer.config import KOLTSOV3_PERM2
        >>> exp = Explorer(KOLTSOV3_PERM2, output_dir="results_perm2")
        >>> df = exp.run_and_save("different", min_n=4, max_n=20)
    """

    def __init__(
        self,
        config: GeneratorConfig,
        output_dir: str = "results",
        min_n: int = 4,
        max_n: int = 30,
        param_overrides: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        """
        Args:
            config: GeneratorConfig defining the generator type.
            output_dir: Base directory for storing results.
            min_n: Default minimum n value.
            max_n: Default maximum n value.
            param_overrides: Dict mapping param names to (min, max) tuples,
                overriding the defaults in config. E.g. {"d": (1, 5)}.
        """
        self.config = config
        self.output_dir = output_dir
        self.min_n = min_n
        self.max_n = max_n
        self.param_overrides = param_overrides or {}
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_group_dir(self, group_name: str) -> str:
        group_dir = os.path.join(self.output_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)
        return group_dir

    def _get_param_range(
        self,
        spec,
        max_n: int,
        runtime_overrides: Dict[str, Tuple[int, int]],
    ) -> range:
        """Compute the iteration range for a parameter.

        Priority: runtime_overrides > self.param_overrides > defaults.
        """
        name = spec.name

        if name in runtime_overrides:
            lo, hi = runtime_overrides[name]
            return range(lo, hi + 1)

        if name in self.param_overrides:
            lo, hi = self.param_overrides[name]
            return range(lo, hi + 1)

        lo = spec.default_min
        hi = spec.default_max if spec.default_max is not None else max_n - 1
        return range(lo, hi + 1)

    def run_single_experiment(
        self,
        n: int,
        params: Dict[str, int],
        coset_name: str,
        group_name: str,
    ) -> Optional[ExperimentResult]:
        """Run a single BFS experiment.

        Accepts group_name (str) instead of coset_func so that this method
        is picklable for use with ProcessPoolExecutor.
        """
        try:
            defn = self.config.factory(n, **params)

            if self.config.make_inverse_closed:
                defn = defn.make_inverse_closed()

            coset_func = COSET_GROUPS[group_name][coset_name]
            central = coset_func(n)
            if coset_name != "FullGraph":
                if central is None or len(np.unique(central)) <= 1:
                    return None
                defn = defn.with_central_state(central)

            graph = CayleyGraph(defn)
            result = graph.bfs(return_all_edges=False, return_all_hashes=False)
            graph.free_memory()

            return ExperimentResult(
                n=n,
                params=dict(params),
                coset=coset_name,
                diameter=result.diameter(),
                growth=result.layer_sizes,
                last_layer_size=len(result.last_layer()),
                total_states=sum(result.layer_sizes),
                central=central,
            )
        except Exception as e:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"Error: {coset_name}, {param_str}, n={n}: {e}")
            return None

    def _iterate_params(
        self,
        min_n: int,
        max_n: int,
        runtime_overrides: Dict[str, Tuple[int, int]],
    ):
        """Yield (n, params_dict) for every valid combination.

        Iterates all params in declared order, then n innermost.
        Invalid combos (per config.is_valid) are filtered out.
        """
        def _recurse(specs, accumulated):
            if not specs:
                # All params chosen — iterate n
                for n in range(min_n, max_n + 1):
                    if self.config.is_valid is None or self.config.is_valid(n, accumulated):
                        yield n, dict(accumulated)
                return

            spec = specs[0]
            for val in self._get_param_range(spec, max_n, runtime_overrides):
                accumulated[spec.name] = val
                yield from _recurse(specs[1:], accumulated)
            accumulated.pop(spec.name, None)

        yield from _recurse(self.config.params, {})

    def _make_cache_key(
        self, coset_name: str, params: Dict[str, int], n: int,
    ) -> tuple:
        return (coset_name,) + tuple(
            params[p] for p in self.config.param_names
        ) + (n,)

    def _filter_cosets(
        self,
        cosets: Dict[str, CosetFunc],
        coset_filter: Optional[Union[str, List[str]]],
    ) -> Dict[str, CosetFunc]:
        if coset_filter is None:
            return cosets
        if isinstance(coset_filter, str):
            if coset_filter not in cosets:
                raise ValueError(
                    f"Coset '{coset_filter}' not found. "
                    f"Available: {list(cosets.keys())}"
                )
            return {coset_filter: cosets[coset_filter]}
        filtered = {k: v for k, v in cosets.items() if k in coset_filter}
        missing = set(coset_filter) - set(filtered.keys())
        if missing:
            raise ValueError(
                f"Cosets not found: {missing}. "
                f"Available: {list(cosets.keys())}"
            )
        return filtered

    def run_group(
        self,
        group_name: str,
        save_every: int = 10,
        min_n: Optional[int] = None,
        max_n: Optional[int] = None,
        coset_filter: Optional[Union[str, List[str]]] = None,
        skip_computed: bool = True,
        **param_ranges,
    ) -> List[ExperimentResult]:
        """Run experiments for a group of cosets.

        Results are saved to CSV every save_every experiments, so
        progress is preserved if the run is interrupted.

        Args:
            group_name: Name of coset group (e.g. "different", "then").
            save_every: Save to CSV after this many results complete.
            min_n: Override minimum n.
            max_n: Override maximum n.
            coset_filter: None (all), str (single), or list.
            skip_computed: Skip already cached combinations.
            **param_ranges: Per-param range overrides as tuples.
                E.g. k_range=(0, 5), d_range=(1, 3).

        Returns:
            List of ExperimentResult objects.
        """
        min_n = min_n if min_n is not None else self.min_n
        max_n = max_n if max_n is not None else self.max_n

        if group_name not in COSET_GROUPS:
            raise ValueError(
                f"Unknown group: {group_name}. "
                f"Available: {list(COSET_GROUPS.keys())}"
            )

        cosets = self._filter_cosets(COSET_GROUPS[group_name], coset_filter)
        group_dir = self._get_group_dir(group_name)
        computed = (
            get_computed_combinations(group_dir, self.config)
            if skip_computed
            else set()
        )

        # Parse param_ranges: k_range=(0,5) -> {"k": (0,5)}
        runtime_overrides: Dict[str, Tuple[int, int]] = {}
        for key, val in param_ranges.items():
            if key.endswith("_range"):
                runtime_overrides[key[:-6]] = val

        all_results: List[ExperimentResult] = []
        unsaved: List[ExperimentResult] = []

        for coset_name in cosets:
            skipped = 0

            combos = list(self._iterate_params(min_n, max_n, runtime_overrides))
            pbar = tqdm(total=len(combos), desc=f"{coset_name}", leave=True)

            for n, params in combos:
                pbar.set_postfix({"n": n, **params})
                pbar.update(1)

                cache_key = self._make_cache_key(coset_name, params, n)
                if cache_key in computed:
                    skipped += 1
                    continue

                result = self.run_single_experiment(n, params, coset_name, group_name)
                if result is not None:
                    all_results.append(result)
                    unsaved.append(result)

                    if len(unsaved) >= save_every:
                        self.save_results(group_name, unsaved)
                        unsaved = []

            pbar.close()
            print(
                f"  {coset_name}: Skipped {skipped} cached"
            )

        # Save any remaining results
        if unsaved:
            self.save_results(group_name, unsaved)

        print(f"Completed {group_name} ({len(all_results)} new results)")
        return all_results

    def run_group_parallel(
        self,
        group_name: str,
        max_workers: int = 4,
        save_every: int = 10,
        min_n: Optional[int] = None,
        max_n: Optional[int] = None,
        coset_filter: Optional[Union[str, List[str]]] = None,
        skip_computed: bool = True,
        **param_ranges,
    ) -> List[ExperimentResult]:
        """Run experiments in parallel using a process pool.

        Results are saved to CSV every save_every completed experiments,
        so progress is preserved if the run is interrupted. Work items
        are sorted by n descending so expensive experiments start first.

        Args:
            group_name: Name of coset group.
            max_workers: Number of parallel worker processes.
            save_every: Save to CSV after this many results complete.
            min_n: Override minimum n.
            max_n: Override maximum n.
            coset_filter: None (all), str (single), or list.
            skip_computed: Skip already cached combinations.
            **param_ranges: Per-param range overrides as tuples.

        Returns:
            List of ExperimentResult objects.
        """
        min_n = min_n if min_n is not None else self.min_n
        max_n = max_n if max_n is not None else self.max_n

        if group_name not in COSET_GROUPS:
            raise ValueError(
                f"Unknown group: {group_name}. "
                f"Available: {list(COSET_GROUPS.keys())}"
            )

        cosets = self._filter_cosets(COSET_GROUPS[group_name], coset_filter)
        group_dir = self._get_group_dir(group_name)
        computed = (
            get_computed_combinations(group_dir, self.config)
            if skip_computed
            else set()
        )

        runtime_overrides: Dict[str, Tuple[int, int]] = {}
        for key, val in param_ranges.items():
            if key.endswith("_range"):
                runtime_overrides[key[:-6]] = val

        # Build flat list of all work items: (n, params, coset_name, group_name)
        # group_name is a plain string (picklable); the worker looks up coset_func itself.
        work_items = []
        for coset_name in cosets:
            for n, params in self._iterate_params(min_n, max_n, runtime_overrides):
                cache_key = self._make_cache_key(coset_name, params, n)
                if cache_key not in computed:
                    work_items.append((n, params, coset_name, group_name))

        # Sort by n descending so expensive items start first
        work_items.sort(key=lambda x: x[0], reverse=True)

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus > 0:
            print(f"Dispatching {len(work_items)} experiments across {max_workers} workers "
                  f"({num_gpus} GPU{'s' if num_gpus > 1 else ''}, round-robin assigned)")
            _spawn_ctx = multiprocessing.get_context("spawn")
            gpu_queue: multiprocessing.Queue = _spawn_ctx.Queue()
            for i in range(max_workers):
                gpu_queue.put(i % num_gpus)
            pool_kwargs = dict(
                max_workers=max_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_gpu_worker,
                initargs=(gpu_queue,),
            )
        else:
            print(f"Dispatching {len(work_items)} experiments across {max_workers} workers (CPU)")
            pool_kwargs = dict(
                max_workers=max_workers,
                mp_context=multiprocessing.get_context("spawn"),
            )

        all_results: List[ExperimentResult] = []
        unsaved: List[ExperimentResult] = []

        with ProcessPoolExecutor(**pool_kwargs) as pool:
            futures = {
                pool.submit(self.run_single_experiment, n, params, coset_name, grp_name): i
                for i, (n, params, coset_name, grp_name) in enumerate(work_items)
            }
            pbar = tqdm(total=len(futures), desc="Parallel BFS", leave=True)

            for future in as_completed(futures):
                pbar.update(1)
                result = future.result()
                if result is not None:
                    all_results.append(result)
                    unsaved.append(result)

                if len(unsaved) >= save_every:
                    self.save_results(group_name, unsaved)
                    unsaved = []

            pbar.close()

        # Save any remaining results
        if unsaved:
            self.save_results(group_name, unsaved)

        print(f"Completed {group_name} ({len(all_results)} new results)")
        return all_results

    def save_results(
        self,
        group_name: str,
        results: List[ExperimentResult],
        append: bool = True,
    ) -> pd.DataFrame:
        """Save results to CSV."""
        group_dir = self._get_group_dir(group_name)
        cosets = COSET_GROUPS.get(group_name, {})
        return save_results(group_dir, results, cosets, self.config, append)

    def load_results(self, group_name: str) -> pd.DataFrame:
        """Load results from CSV."""
        group_dir = self._get_group_dir(group_name)
        return load_results(group_dir)

    def plot_results(
        self,
        group_name: str,
        df: Optional[pd.DataFrame] = None,
        show: bool = True,
    ) -> None:
        """Create interactive Plotly plots for results."""
        if df is None:
            df = self.load_results(group_name)
        group_dir = self._get_group_dir(group_name)
        plot_group_results(group_dir, df, self.config, group_name, show)

    def run_and_save(
        self,
        group_name: str,
        plot: bool = True,
        parallel: bool = False,
        max_workers: int = 4,
        **kwargs,
    ) -> pd.DataFrame:
        """Run experiments, save incrementally, and optionally plot.

        Args:
            group_name: Name of coset group.
            plot: If True, generate plots after completing.
            parallel: If True, use run_group_parallel instead of run_group.
            max_workers: Number of workers (only used if parallel=True).
            **kwargs: Additional arguments passed to run_group/run_group_parallel.
        """
        if parallel:
            self.run_group_parallel(group_name, max_workers=max_workers, **kwargs)
        else:
            self.run_group(group_name, **kwargs)

        df = self.load_results(group_name)
        if plot:
            self.plot_results(group_name, df)
        return df

    def __repr__(self) -> str:
        overrides = ""
        if self.param_overrides:
            overrides = ", " + ", ".join(
                f"{k}={v}" for k, v in self.param_overrides.items()
            )
        return (
            f"Explorer(config='{self.config.name}', "
            f"output_dir='{self.output_dir}', "
            f"n=[{self.min_n}, {self.max_n}]{overrides})"
        )
