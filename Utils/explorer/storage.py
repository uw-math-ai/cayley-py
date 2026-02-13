"""Config-driven CSV storage for experiment results."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import pandas as pd

if TYPE_CHECKING:
    from .config import GeneratorConfig
    from .coset_groups import CosetFunc


def get_computed_combinations(
    group_dir: str, config: GeneratorConfig,
) -> Set[Tuple]:
    """Return set of already computed parameter combinations.

    Keys are tuples matching config.cache_key_columns order,
    e.g. (coset, k, n) or (coset, k, d, n).
    """
    csv_path = os.path.join(group_dir, "data.csv")
    if not os.path.exists(csv_path):
        return set()

    df = pd.read_csv(csv_path)
    key_cols = config.cache_key_columns
    existing_cols = [c for c in key_cols if c in df.columns]
    if not existing_cols:
        return set()
    return set(df[existing_cols].itertuples(index=False, name=None))


def save_results(
    group_dir: str,
    results: List,
    cosets: Dict[str, CosetFunc],
    config: GeneratorConfig,
    append: bool = True,
) -> pd.DataFrame:
    """Save ExperimentResult list to CSV.

    Columns are built dynamically from config.param_names:
      coset, <param1>, <param2>, ..., n, diameter, last_layer_size,
      total_states, growth, central
    """
    csv_path = os.path.join(group_dir, "data.csv")

    rows = []
    for r in results:
        row = {"coset": r.coset}
        for pname in config.param_names:
            row[pname] = r.params.get(pname)
        row["n"] = r.n
        row["diameter"] = r.diameter
        row["last_layer_size"] = r.last_layer_size
        row["total_states"] = r.total_states
        row["growth"] = json.dumps(r.growth)
        rows.append(row)

    df_new = pd.DataFrame(rows)

    if append and os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
        key_cols = config.cache_key_columns
        df = df.drop_duplicates(subset=key_cols, keep="last")
    else:
        df = df_new

    # Compute central states for all rows
    def compute_central(row):
        coset_func = cosets.get(row["coset"])
        if coset_func is None:
            return None
        central = coset_func(int(row["n"]))
        return json.dumps(central) if central is not None else None

    if len(df) > 0:
        df["central"] = df.apply(compute_central, axis=1)

    sort_cols = [c for c in config.sort_columns if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df.to_csv(csv_path, index=False)

    print(f"Saved: {csv_path} ({len(df)} total rows, {len(df_new)} new)")
    return df


def load_results(group_dir: str) -> pd.DataFrame:
    """Load results from CSV, parsing JSON columns."""
    csv_path = os.path.join(group_dir, "data.csv")
    df = pd.read_csv(csv_path)
    df["growth"] = df["growth"].apply(json.loads)
    if "central" in df.columns:
        df["central"] = df["central"].apply(
            lambda x: json.loads(x) if pd.notna(x) else None
        )
    return df
