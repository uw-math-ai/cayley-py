"""Shared coset group definitions for Koltsov exploration experiments."""

from typing import Callable, Dict, List, Optional

# Type alias for coset functions
CosetFunc = Callable[[int], Optional[List[int]]]

COSET_GROUPS: Dict[str, Dict[str, CosetFunc]] = {
    # =========================================================================
    # FULL GRAPH - No coset, explores entire permutation space
    # =========================================================================
    "full_graph": {
        # n=6: None (uses identity, explores all 6! = 720 permutations)
        "FullGraph": lambda n: None,
    },

    # =========================================================================
    # DIFFERENT - First D-1 elements are distinct, rest are all the same
    # Pattern: [0, 1, ..., D-2, D-1, D-1, D-1, ...]
    # =========================================================================
    "different": {
        # n=6: [0, 1, 1, 1, 1, 1]
        "2Different": lambda n: list(range(1)) + [1]*(n-1) if n >= 2 else None,
        # n=6: [0, 1, 2, 2, 2, 2]
        "3Different": lambda n: list(range(2)) + [2]*(n-2) if n >= 3 else None,
        # n=6: [0, 1, 2, 3, 3, 3]
        "4Different": lambda n: list(range(3)) + [3]*(n-3) if n >= 4 else None,
    },

    # =========================================================================
    # THEN - Blocks of consecutive same values
    # Pattern: [0,0,..., 1,1,..., 2,2,...] (divided into equal-ish parts)
    # =========================================================================
    "then": {
        # n=6: [0, 0, 0, 1, 1, 1]
        "Binary0then1": lambda n: [0]*(n//2) + [1]*(n - n//2),
        # n=6: [0, 0, 1, 1, 2, 2]
        "0then1then2": lambda n: [0]*(n//3) + [1]*(n//3) + [2]*(n - 2*(n//3)),
        # n=8: [0, 0, 1, 1, 2, 2, 3, 3]
        "0then1then2then3": lambda n: [0]*(n//4) + [1]*(n//4) + [2]*(n//4) + [3]*(n - 3*(n//4)),
        # n=10: [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        "0then1then2then3then4": lambda n: [0]*(n//5) + [1]*(n//5) + [2]*(n//5) + [3]*(n//5) + [4]*(n - 4*(n//5)),
    },

    # =========================================================================
    # COINCIDE - Sequential values, but last C elements are all the same
    # Pattern: [0, 1, 2, ..., n-C-1, n-C, n-C, ..., n-C]
    # =========================================================================
    "coincide": {
        # n=6: [0, 1, 2, 3, 4, 4]
        "2Coincide": lambda n: list(range(n-2)) + [n-2]*2 if n > 2 else None,
        # n=6: [0, 1, 2, 3, 3, 3]
        "3Coincide": lambda n: list(range(n-3)) + [n-3]*3 if n > 3 else None,
        # n=6: [0, 1, 2, 2, 2, 2]
        "4Coincide": lambda n: list(range(n-4)) + [n-4]*4 if n > 4 else None,
        # n=6: [0, 1, 1, 1, 1, 1]
        "5Coincide": lambda n: list(range(n-5)) + [n-5]*5 if n > 5 else None,
        # n=7: [0, 1, 1, 1, 1, 1, 1]
        "6Coincide": lambda n: list(range(n-6)) + [n-6]*6 if n > 6 else None,
    },

    # =========================================================================
    # REPEATS - Repeating pattern of a short block
    # Pattern: [block] * (n // len(block)) + partial
    # =========================================================================
    "repeats": {
        # n=6: [0, 1, 0, 1, 0, 1]
        # n=7: [0, 1, 0, 1, 0, 1, 0]
        "Binary01Repeats": lambda n: [0,1]*(n//2) + [0]*(n - 2*(n//2)),
        # n=6: [0, 1, 0, 1, 0, 1]
        # n=7: [0, 1, 0, 1, 0, 1, 1]
        "Binary01Repeats_1": lambda n: [0,1]*(n//2) + [1]*(n - 2*(n//2)),
        # n=6: [0, 1, 2, 0, 1, 2]
        # n=7: [0, 1, 2, 0, 1, 2, 0]
        "012Repeats": lambda n: [0,1,2]*(n//3) + [0,1,2][:(n%3)],
        # n=6: [0, 1, 1, 0, 1, 1]
        # n=7: [0, 1, 1, 0, 1, 1, 0]
        "011Repeats": lambda n: [0,1,1]*(n//3) + [0,1,1][:(n%3)],
    },
}


def list_groups() -> List[str]:
    """Return list of available group names."""
    return list(COSET_GROUPS.keys())


def list_cosets(group_name: str) -> List[str]:
    """Return list of coset names in a group."""
    if group_name not in COSET_GROUPS:
        raise ValueError(f"Unknown group: {group_name}. Available: {list_groups()}")
    return list(COSET_GROUPS[group_name].keys())


def get_coset_func(group_name: str, coset_name: str) -> CosetFunc:
    """Get a coset function by group and coset name."""
    if group_name not in COSET_GROUPS:
        raise ValueError(f"Unknown group: {group_name}. Available: {list_groups()}")
    if coset_name not in COSET_GROUPS[group_name]:
        raise ValueError(f"Unknown coset: {coset_name}. Available: {list_cosets(group_name)}")
    return COSET_GROUPS[group_name][coset_name]


def print_summary() -> None:
    """Print a summary of all coset groups with examples."""
    print("Coset Groups Summary:")
    print("=" * 60)
    for group_name, cosets in COSET_GROUPS.items():
        print(f"\n{group_name.upper()}:")
        for coset_name, func in cosets.items():
            example = func(6) if func(6) is not None else "None (full graph)"
            print(f"  {coset_name}: n=6 -> {example}")
