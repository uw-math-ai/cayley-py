# Koltsov Explorer

A Python library for systematic exploration of Koltsov3 generator parameter spaces on Cayley graphs and Schreier coset graphs.

## Installation

Add the Utils directory to your Python path:

```python
import sys
sys.path.insert(0, '/home/ec2-user/desktop/Utils')

from koltsov_explorer import KoltsovExplorer, COSET_GROUPS
```

## Quick Start

```python
from koltsov_explorer import KoltsovExplorer

# Create explorer for perm_type=2 experiments
explorer = KoltsovExplorer(perm_type=2, output_dir="my_results")

# Run experiments, save to CSV, and generate plots
df = explorer.run_and_save("different", min_n=4, max_n=20)

# Or for perm_type=1 (with d parameter)
explorer1 = KoltsovExplorer(perm_type=1, output_dir="results_perm1")
df = explorer1.run_and_save("different", min_n=4, max_n=20, d_range=(1, 5))
```

## Features

### Two perm_types Supported

- **perm_type=1**: S generator is transposition (k, k+d)
  - Parameters: n, k, d
  - Validity constraint: k + d < n

- **perm_type=2**: S generator is (k,k+3)(k+1,k+2)
  - Parameters: n, k
  - Validity constraint: k + 3 < n

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
- **diameter.html** - Diameter vs n
- **growth.html** - Growth curves (layer sizes)
- **lastlayer.html** - Last layer size vs n

## API Reference

### KoltsovExplorer

```python
KoltsovExplorer(
    perm_type: int = 2,      # 1 or 2
    output_dir: str = "results",
    min_n: int = 4,
    max_n: int = 30,
    max_d: int = 10,         # Only for perm_type=1
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
explorer.run_group(
    group_name,              # "different", "then", "coincide", "repeats", "full_graph"
    min_n=None,              # Override instance min_n
    max_n=None,              # Override instance max_n
    k_range=None,            # Tuple (k_min, k_max) or None for full range
    d_range=None,            # Tuple (d_min, d_max) for perm_type=1
    coset_filter=None,       # None (all), str (single), or list
    skip_computed=True,      # Skip already computed combinations
)
```

### COSET_GROUPS

Dictionary of coset group definitions.

```python
from koltsov_explorer import COSET_GROUPS, list_groups, list_cosets

# List all groups
print(list_groups())  # ['full_graph', 'different', 'then', 'coincide', 'repeats']

# List cosets in a group
print(list_cosets("different"))  # ['2Different', '3Different', '4Different']

# Get central state for a coset at n=10
central = COSET_GROUPS["different"]["4Different"](10)
print(central)  # [0, 1, 2, 3, 3, 3, 3, 3, 3, 3]
```

## Output Format

Results are stored in CSV files with the following columns:

**perm_type=1:**
```
coset, d, k, n, diameter, last_layer_size, total_states, growth, central
```

**perm_type=2:**
```
coset, k, n, diameter, last_layer_size, total_states, growth, central
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

## Examples

### Filter Specific Cosets

```python
# Run only 3Different and 4Different
df = explorer.run_and_save(
    "different",
    coset_filter=["3Different", "4Different"],
    min_n=4,
    max_n=25
)
```

### Custom Parameter Ranges

```python
# perm_type=1 with specific k and d ranges
df = explorer1.run_and_save(
    "different",
    min_n=10,
    max_n=20,
    k_range=(0, 5),
    d_range=(1, 3)
)
```

### Load and Re-plot Existing Results

```python
df = explorer.load_results("different")
explorer.plot_results("different", df)
```

### Step-by-Step Workflow

```python
# 1. Run experiments
results = explorer.run_group("different", min_n=4, max_n=20)

# 2. Save to CSV
df = explorer.save_results("different", results)

# 3. Generate plots
explorer.plot_results("different", df)
```
