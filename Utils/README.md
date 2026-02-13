# Explorer

A generalized Python framework for systematic exploration of Cayley graph parameter spaces. Works with any generator type from cayleypy — Koltsov3, consecutive k-cycles, wrapped k-cycles, and custom generators.

## Installation

Add the Utils directory to your Python path:

```python
import sys
sys.path.insert(0, '/path/to/Utils')

from explorer import Explorer, KOLTSOV3_PERM2
```

## Quick Start

```python
from explorer import Explorer
from explorer.config import KOLTSOV3_PERM2, CONSECUTIVE_K_CYCLES_INV

# Koltsov3 perm_type=2
exp = Explorer(KOLTSOV3_PERM2, output_dir="results_perm2", max_n=25)
df = exp.run_and_save("different")

# Consecutive k-cycles (inverse closed)
exp2 = Explorer(CONSECUTIVE_K_CYCLES_INV, output_dir="results_consec", max_n=30)
df2 = exp2.run_and_save("then")

# Koltsov3 perm_type=1 (with d parameter)
from explorer.config import KOLTSOV3_PERM1
exp1 = Explorer(KOLTSOV3_PERM1, output_dir="results_perm1", max_n=20)
df1 = exp1.run_and_save("different", d_range=(1, 5))
```

## Built-in Generator Configs

| Config | Generator | Parameters | Notes |
|--------|-----------|------------|-------|
| `KOLTSOV3_PERM1` | `koltsov3(perm_type=1)` | k, d | S = transposition (k, k+d) |
| `KOLTSOV3_PERM2` | `koltsov3(perm_type=2)` | k | S = (k,k+3)(k+1,k+2) |
| `CONSECUTIVE_K_CYCLES` | `consecutive_k_cycles` | k | |
| `WRAPPED_K_CYCLES` | `wrapped_k_cycles` | k | |
| `CONSECUTIVE_K_CYCLES_INV` | `consecutive_k_cycles` | k | + inverse closure |
| `WRAPPED_K_CYCLES_INV` | `wrapped_k_cycles` | k | + inverse closure |

## Adding a Custom Generator

Define a new generator in ~10 lines — no subclassing needed:

```python
from explorer.config import GeneratorConfig, ParamSpec
from cayleypy import PermutationGroups

MY_GEN = GeneratorConfig(
    name="my_generator",
    factory=lambda n, k=2, m=1: PermutationGroups.my_gen(n, k=k, m=m),
    params=[
        ParamSpec(name="k", default_min=2),
        ParamSpec(name="m", default_min=1, default_max=5),
    ],
    is_valid=lambda n, p: p["k"] + p["m"] < n,
)

exp = Explorer(MY_GEN, output_dir="results_my_gen")
df = exp.run_and_save("different", max_n=20)
```

## Parallel Experiments

Each `Explorer` instance is self-contained. Run different generators in parallel:

```python
from concurrent.futures import ProcessPoolExecutor
from explorer import Explorer
from explorer.config import KOLTSOV3_PERM2, CONSECUTIVE_K_CYCLES_INV

def run(config, out_dir, group, max_n):
    return Explorer(config, output_dir=out_dir, max_n=max_n).run_and_save(group, plot=False)

with ProcessPoolExecutor(max_workers=3) as pool:
    f1 = pool.submit(run, KOLTSOV3_PERM2, "res_k3", "different", 30)
    f2 = pool.submit(run, CONSECUTIVE_K_CYCLES_INV, "res_cc", "different", 30)
    f1.result()
    f2.result()
```

## Features

### Incremental Computation

Results are cached in CSV files. Only new parameter combinations are computed on subsequent runs.

### 5 Coset Groups (17 coset types)

| Group | Cosets | Pattern |
|-------|--------|---------|
| `full_graph` | FullGraph | No coset (full permutation space) |
| `different` | 2Different, 3Different, 4Different | First D-1 elements distinct, rest same |
| `then` | Binary0then1, 0then1then2, ... | Blocks of consecutive same values |
| `coincide` | 2Coincide, 3Coincide, ... | Last C elements coincide |
| `repeats` | Binary01Repeats, 012Repeats, ... | Repeating pattern blocks |

### Interactive Visualizations

Generates Plotly HTML plots with dropdown selectors:
- **diameter.html** — Diameter vs n
- **growth.html** — Growth curves (layer sizes)
- **lastlayer.html** — Last layer size vs n

## API Reference

### Explorer

```python
Explorer(
    config: GeneratorConfig,         # Which generator type
    output_dir: str = "results",
    min_n: int = 4,
    max_n: int = 30,
    param_overrides: dict = None,    # E.g. {"d": (1, 5)}
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `run_group(group_name, ...)` | Run experiments for a coset group |
| `save_results(group_name, results)` | Save results to CSV |
| `load_results(group_name)` | Load results from CSV |
| `plot_results(group_name, df)` | Generate interactive plots |
| `run_and_save(group_name, ...)` | Convenience: run + save + plot |

#### run_group Parameters

```python
exp.run_group(
    group_name,              # "different", "then", "coincide", "repeats", "full_graph"
    min_n=None,              # Override instance min_n
    max_n=None,              # Override instance max_n
    coset_filter=None,       # None (all), str (single), or list
    skip_computed=True,      # Skip already computed combinations
    k_range=(0, 5),          # Per-param range overrides as <name>_range kwargs
    d_range=(1, 3),          # (only relevant if config has a "d" param)
)
```

### GeneratorConfig

```python
GeneratorConfig(
    name: str,                       # Human-readable name
    factory: Callable,               # (n, **params) -> cayleypy group definition
    params: List[ParamSpec],         # Parameter specifications
    is_valid: Callable = None,       # (n, params_dict) -> bool
    make_inverse_closed: bool = False,
    description: str = "",
)
```

### ParamSpec

```python
ParamSpec(
    name: str,                       # Must match factory kwarg name
    default_min: int = 0,
    default_max: int = None,         # None = max_n - 1
    depends_on_n: bool = False,      # If True, range recomputed per n
    dynamic_range: Callable = None,  # (n, other_params) -> (min, max)
)
```

### Coset Groups

```python
from explorer import COSET_GROUPS, list_groups, list_cosets

print(list_groups())  # ['full_graph', 'different', 'then', 'coincide', 'repeats']
print(list_cosets("different"))  # ['2Different', '3Different', '4Different']

central = COSET_GROUPS["different"]["4Different"](10)
# [0, 1, 2, 3, 3, 3, 3, 3, 3, 3]
```

## Output Format

CSV columns are built dynamically from the config's parameter list:

```
coset, <param1>, <param2>, ..., n, diameter, last_layer_size, total_states, growth, central
```

The `growth` and `central` columns contain JSON-encoded lists.

## Directory Structure

```
output_dir/
├── different/
│   ├── data.csv
│   ├── diameter.html
│   ├── growth.html
│   └── lastlayer.html
├── then/
│   └── ...
└── ...
```

## Dependencies

- cayleypy
- pandas
- numpy
- plotly
- tqdm
