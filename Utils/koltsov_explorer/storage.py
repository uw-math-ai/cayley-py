"""CSV storage operations for Koltsov experiments."""

import json
import os
from typing import Dict, Optional, Set, Tuple, Union

import pandas as pd

from .coset_groups import CosetFunc


def get_computed_combinations(group_dir: str, perm_type: int) -> Set[Tuple]:
    """
    Return set of already computed parameter combinations.

    For perm_type=1: returns set of (coset, d, k, n) tuples.
    For perm_type=2: returns set of (coset, k, n) tuples.

    Args:
        group_dir: Directory containing the CSV file.
        perm_type: 1 or 2.

    Returns:
        Set of tuples representing computed combinations.
    """
    csv_path = os.path.join(group_dir, "data.csv")
    if not os.path.exists(csv_path):
        return set()

    df = pd.read_csv(csv_path)
    if perm_type == 1:
        return set(zip(df['coset'], df['d'], df['k'], df['n']))
    else:
        return set(zip(df['coset'], df['k'], df['n']))


def save_results(
    group_dir: str,
    results: Dict,
    cosets: Dict[str, CosetFunc],
    perm_type: int,
    append: bool = True,
) -> pd.DataFrame:
    """
    Save results to CSV with growth and central stored as JSON strings.

    Args:
        group_dir: Directory for this group's results.
        results: Results dictionary from run_group().
        cosets: Dictionary of coset functions.
        perm_type: 1 or 2.
        append: If True, append to existing CSV.

    Returns:
        DataFrame with all results.
    """
    csv_path = os.path.join(group_dir, "data.csv")

    # Build DataFrame from new results
    rows = []

    if perm_type == 1:
        for coset, d_data in results["results"].items():
            for d_key, k_data in d_data.items():
                d_val = int(d_key.split("=")[1])
                for k_key, n_data in k_data.items():
                    k_val = int(k_key.split("=")[1])
                    for n_key, metrics in n_data.items():
                        n_val = int(n_key.split("=")[1])
                        rows.append({
                            "coset": coset,
                            "d": d_val,
                            "k": k_val,
                            "n": n_val,
                            "diameter": metrics["diameter"],
                            "last_layer_size": metrics["last_layer_size"],
                            "total_states": sum(metrics["growth"]),
                            "growth": json.dumps(metrics["growth"]),
                        })
    else:  # perm_type == 2
        for coset, k_data in results["results"].items():
            for k_key, n_data in k_data.items():
                k_val = int(k_key.split("=")[1])
                for n_key, metrics in n_data.items():
                    n_val = int(n_key.split("=")[1])
                    rows.append({
                        "coset": coset,
                        "k": k_val,
                        "n": n_val,
                        "diameter": metrics["diameter"],
                        "last_layer_size": metrics["last_layer_size"],
                        "total_states": sum(metrics["growth"]),
                        "growth": json.dumps(metrics["growth"]),
                    })

    df_new = pd.DataFrame(rows)

    # Handle append mode
    if append and os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
        # Remove duplicates, keeping last (new) values
        key_cols = ['coset', 'd', 'k', 'n'] if perm_type == 1 else ['coset', 'k', 'n']
        df = df.drop_duplicates(subset=key_cols, keep='last')
    else:
        df = df_new

    # Compute central states for all rows (backfills existing rows too)
    def compute_central(row):
        coset_func = cosets.get(row['coset'])
        if coset_func is None:
            return None
        central = coset_func(int(row['n']))
        return json.dumps(central) if central is not None else None

    df['central'] = df.apply(compute_central, axis=1)

    # Sort and save
    sort_cols = ['coset', 'd', 'k', 'n'] if perm_type == 1 else ['coset', 'k', 'n']
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df.to_csv(csv_path, index=False)

    print(f"Saved: {csv_path} ({len(df)} total rows, {len(df_new)} new)")
    return df


def load_results(group_dir: str) -> pd.DataFrame:
    """
    Load results from CSV, parsing growth and central back to lists.

    Args:
        group_dir: Directory containing the CSV file.

    Returns:
        DataFrame with parsed growth and central columns.
    """
    csv_path = os.path.join(group_dir, "data.csv")
    df = pd.read_csv(csv_path)
    df['growth'] = df['growth'].apply(json.loads)
    if 'central' in df.columns:
        df['central'] = df['central'].apply(
            lambda x: json.loads(x) if pd.notna(x) else None
        )
    return df
