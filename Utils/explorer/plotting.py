"""Generic Plotly visualization for experiment results.

Plots are driven by the GeneratorConfig's param_names list:
- The last param becomes the "line variable" (separate colored traces).
- All other params + coset become dropdown selectors.
- If there are no params, coset alone is the dropdown.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from .config import GeneratorConfig


def plot_group_results(
    group_dir: str,
    df: pd.DataFrame,
    config: GeneratorConfig,
    group_name: str,
    show: bool = True,
) -> None:
    """Create interactive Plotly plots for diameter, growth, and last layer size."""
    df = df.copy()
    df["growth_parsed"] = df["growth"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    param_names = config.param_names

    # Convention: last param is line variable, rest are dropdown dimensions
    if len(param_names) == 0:
        line_param = None
        dropdown_params = []
    elif len(param_names) == 1:
        line_param = param_names[0]
        dropdown_params = []
    else:
        line_param = param_names[-1]
        dropdown_params = param_names[:-1]

    title_suffix = config.name

    _plot_diameter(group_dir, df, group_name, title_suffix,
                   dropdown_params, line_param, show)
    _plot_growth(group_dir, df, group_name, title_suffix,
                 dropdown_params, line_param, show)
    _plot_last_layer(group_dir, df, group_name, title_suffix,
                     dropdown_params, line_param, show)
    print(f"Interactive plots saved to {group_dir}/")


def _get_dropdown_keys(df: pd.DataFrame, dropdown_params: List[str]) -> List[Tuple]:
    """Get unique combinations of (coset, *dropdown_params) for dropdown menu."""
    cols = ["coset"] + dropdown_params
    unique = df[cols].drop_duplicates().sort_values(cols)
    return list(unique.itertuples(index=False, name=None))


def _dropdown_label(key: Tuple, dropdown_params: List[str]) -> str:
    """Build a human-readable label for a dropdown entry."""
    coset = key[0]
    if not dropdown_params:
        return coset
    parts = [coset] + [f"{p}={v}" for p, v in zip(dropdown_params, key[1:])]
    return ", ".join(parts)


def _filter_df(df: pd.DataFrame, key: Tuple, dropdown_params: List[str]) -> pd.DataFrame:
    """Filter dataframe to rows matching a dropdown key."""
    mask = df["coset"] == key[0]
    for i, p in enumerate(dropdown_params):
        mask = mask & (df[p] == key[1 + i])
    return df[mask]


def _plot_diameter(
    group_dir: str,
    df: pd.DataFrame,
    group_name: str,
    title_suffix: str,
    dropdown_params: List[str],
    line_param: Optional[str],
    show: bool,
) -> None:
    """Diameter vs n plot with dropdown selectors."""
    fig = go.Figure()
    dropdown_keys = _get_dropdown_keys(df, dropdown_params)
    trace_map: Dict[Tuple, List[int]] = {}
    trace_idx = 0

    line_values = sorted(df[line_param].unique()) if line_param else [None]

    for key in dropdown_keys:
        trace_map[key] = []
        subset = _filter_df(df, key, dropdown_params)
        for lv in line_values:
            if lv is not None:
                lv_df = subset[subset[line_param] == lv].sort_values("n")
            else:
                lv_df = subset.sort_values("n")
            if len(lv_df) == 0:
                continue

            name = f"{line_param}={lv}" if line_param else _dropdown_label(key, dropdown_params)
            hover_parts = ["n=%{x}", "diameter=%{y}"]
            if line_param:
                hover_parts.append(f"{line_param}={lv}")
            for i, p in enumerate(dropdown_params):
                hover_parts.append(f"{p}={key[1 + i]}")

            fig.add_trace(go.Scatter(
                x=lv_df["n"], y=lv_df["diameter"],
                mode="lines+markers", name=name,
                visible=(key == dropdown_keys[0]),
                hovertemplate="<br>".join(hover_parts),
            ))
            trace_map[key].append(trace_idx)
            trace_idx += 1

    total_traces = trace_idx
    buttons = []
    first_button = True
    init_n_range = [0, 10]
    init_diam_range = [0, 10]

    for key in dropdown_keys:
        if not trace_map.get(key, []):
            continue
        visibility = [False] * total_traces
        for idx in trace_map[key]:
            visibility[idx] = True

        subset = _filter_df(df, key, dropdown_params)
        n_range = [subset["n"].min() - 0.5, subset["n"].max() + 0.5]
        diam_range = [0, subset["diameter"].max() * 1.05]

        buttons.append(dict(
            label=_dropdown_label(key, dropdown_params),
            method="update",
            args=[{"visible": visibility},
                  {"xaxis.range": n_range, "yaxis.range": diam_range}],
        ))

        if first_button:
            init_n_range = n_range
            init_diam_range = diam_range
            first_button = False

    fig.update_layout(
        title=dict(text=f"Diameter vs n - {group_name} ({title_suffix})", y=0.95),
        xaxis_title="n", yaxis_title="Diameter",
        xaxis=dict(range=init_n_range),
        yaxis=dict(range=init_diam_range),
        updatemenus=[dict(
            buttons=buttons, direction="down",
            x=0.0, xanchor="left", y=1.02, yanchor="bottom", showactive=True,
        )],
        legend=dict(x=1.02, y=1), height=650, margin=dict(t=100),
    )
    fig.write_html(f"{group_dir}/diameter.html")
    if show:
        fig.show()


def _plot_growth(
    group_dir: str,
    df: pd.DataFrame,
    group_name: str,
    title_suffix: str,
    dropdown_params: List[str],
    line_param: Optional[str],
    show: bool,
) -> None:
    """Growth curves plot. Dropdown: coset + dropdown_params + line_param. Lines: n values."""
    fig = go.Figure()

    # For growth, each (dropdown_key, line_value) combo is a dropdown entry,
    # and each n value within that combo is a separate trace.
    growth_dropdown_params = dropdown_params + ([line_param] if line_param else [])
    growth_keys = _get_dropdown_keys(df, growth_dropdown_params)

    trace_map: Dict[Tuple, List[int]] = {}
    trace_idx = 0

    for key in growth_keys:
        trace_map[key] = []
        subset = _filter_df(df, key, growth_dropdown_params).sort_values("n")
        for _, row in subset.iterrows():
            growth = row["growth_parsed"]
            fig.add_trace(go.Scatter(
                x=list(range(len(growth))), y=growth,
                mode="lines+markers", name=f"n={row['n']}",
                visible=(key == growth_keys[0] if growth_keys else False),
                hovertemplate="distance=%{x}<br>layer_size=%{y}<br>n=" + str(row["n"]),
            ))
            trace_map[key].append(trace_idx)
            trace_idx += 1

    total_traces = trace_idx
    buttons = []
    first_button = True
    init_dist_range = [0, 10]
    init_layer_range = [0, 6]

    for key in growth_keys:
        if not trace_map.get(key, []):
            continue
        visibility = [False] * total_traces
        for idx in trace_map[key]:
            visibility[idx] = True

        subset = _filter_df(df, key, growth_dropdown_params)
        growths = subset["growth_parsed"].tolist()
        if growths:
            max_distance = max(len(g) for g in growths)
            max_layer = max(max(g) for g in growths)
            min_layer = min(min(g) for g in growths if min(g) > 0)
            dist_range = [-0.5, max_distance + 0.5]
            layer_range = [np.log10(min_layer * 0.5), np.log10(max_layer * 2)]
        else:
            dist_range = [0, 10]
            layer_range = [0, 6]

        buttons.append(dict(
            label=_dropdown_label(key, growth_dropdown_params),
            method="update",
            args=[{"visible": visibility},
                  {"xaxis.range": dist_range, "yaxis.range": layer_range}],
        ))

        if first_button:
            init_dist_range = dist_range
            init_layer_range = layer_range
            first_button = False

    fig.update_layout(
        title=dict(text=f"Growth Curves - {group_name} ({title_suffix})", y=0.95),
        xaxis_title="Distance", yaxis_title="Layer Size", yaxis_type="log",
        xaxis=dict(range=init_dist_range),
        yaxis=dict(range=init_layer_range),
        updatemenus=[dict(
            buttons=buttons, direction="down",
            x=0.0, xanchor="left", y=1.02, yanchor="bottom", showactive=True,
        )],
        legend=dict(x=1.02, y=1), height=650, margin=dict(t=100),
    )
    fig.write_html(f"{group_dir}/growth.html")
    if show:
        fig.show()


def _plot_last_layer(
    group_dir: str,
    df: pd.DataFrame,
    group_name: str,
    title_suffix: str,
    dropdown_params: List[str],
    line_param: Optional[str],
    show: bool,
) -> None:
    """Last layer size vs n plot with dropdown selectors."""
    fig = go.Figure()
    dropdown_keys = _get_dropdown_keys(df, dropdown_params)
    trace_map: Dict[Tuple, List[int]] = {}
    trace_idx = 0

    line_values = sorted(df[line_param].unique()) if line_param else [None]

    for key in dropdown_keys:
        trace_map[key] = []
        subset = _filter_df(df, key, dropdown_params)
        for lv in line_values:
            if lv is not None:
                lv_df = subset[subset[line_param] == lv].sort_values("n")
            else:
                lv_df = subset.sort_values("n")
            if len(lv_df) == 0:
                continue

            name = f"{line_param}={lv}" if line_param else _dropdown_label(key, dropdown_params)
            hover_parts = ["n=%{x}", "last_layer=%{y}"]
            if line_param:
                hover_parts.append(f"{line_param}={lv}")
            for i, p in enumerate(dropdown_params):
                hover_parts.append(f"{p}={key[1 + i]}")

            fig.add_trace(go.Scatter(
                x=lv_df["n"], y=lv_df["last_layer_size"],
                mode="lines+markers", name=name,
                visible=(key == dropdown_keys[0]),
                hovertemplate="<br>".join(hover_parts),
            ))
            trace_map[key].append(trace_idx)
            trace_idx += 1

    total_traces = trace_idx
    buttons = []
    first_button = True
    init_n_range = [0, 10]
    init_ll_range = [0, 6]

    for key in dropdown_keys:
        if not trace_map.get(key, []):
            continue
        visibility = [False] * total_traces
        for idx in trace_map[key]:
            visibility[idx] = True

        subset = _filter_df(df, key, dropdown_params)
        n_range = [subset["n"].min() - 0.5, subset["n"].max() + 0.5]
        ll_min = subset["last_layer_size"].min()
        ll_max = subset["last_layer_size"].max()
        ll_range = [np.log10(max(ll_min * 0.5, 0.1)), np.log10(ll_max * 2)]

        buttons.append(dict(
            label=_dropdown_label(key, dropdown_params),
            method="update",
            args=[{"visible": visibility},
                  {"xaxis.range": n_range, "yaxis.range": ll_range}],
        ))

        if first_button:
            init_n_range = n_range
            init_ll_range = ll_range
            first_button = False

    fig.update_layout(
        title=dict(text=f"Last Layer Size vs n - {group_name} ({title_suffix})", y=0.95),
        xaxis_title="n", yaxis_title="Last Layer Size", yaxis_type="log",
        xaxis=dict(range=init_n_range),
        yaxis=dict(range=init_ll_range),
        updatemenus=[dict(
            buttons=buttons, direction="down",
            x=0.0, xanchor="left", y=1.02, yanchor="bottom", showactive=True,
        )],
        legend=dict(x=1.02, y=1), height=650, margin=dict(t=100),
    )
    fig.write_html(f"{group_dir}/lastlayer.html")
    if show:
        fig.show()
