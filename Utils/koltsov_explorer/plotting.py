"""Plotly visualization functions for Koltsov experiments."""

import json
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_group_results(
    group_dir: str,
    df: pd.DataFrame,
    perm_type: int,
    group_name: str,
    show: bool = True,
) -> None:
    """
    Create interactive Plotly plots for diameter, growth, and last layer size.

    Args:
        group_dir: Directory to save HTML files.
        df: DataFrame with results.
        perm_type: 1 or 2.
        group_name: Name for plot titles.
        show: If True, display plots.
    """
    # Pre-parse growth column if needed
    df = df.copy()
    df['growth_parsed'] = df['growth'].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    if perm_type == 1:
        _plot_perm1_results(group_dir, df, group_name, show)
    else:
        _plot_perm2_results(group_dir, df, group_name, show)


def _plot_perm1_results(group_dir: str, df: pd.DataFrame, group_name: str, show: bool) -> None:
    """Plot results for perm_type=1 (with d parameter)."""
    coset_names = list(df['coset'].unique())
    d_values = sorted(df['d'].unique())
    k_values = sorted(df['k'].unique())

    # ===== PLOT 1: Diameter vs n (dropdown for coset+d, lines for k) =====
    fig1 = go.Figure()
    trace_idx = 0
    trace_map1 = {}

    for coset_name in coset_names:
        for d_val in d_values:
            trace_map1[(coset_name, d_val)] = []
            coset_d_df = df[(df['coset'] == coset_name) & (df['d'] == d_val)]
            for k_val in k_values:
                k_df = coset_d_df[coset_d_df['k'] == k_val].sort_values('n')
                if len(k_df) > 0:
                    fig1.add_trace(go.Scatter(
                        x=k_df['n'], y=k_df['diameter'],
                        mode='lines+markers', name=f'k={k_val}',
                        visible=(coset_name == coset_names[0] and d_val == d_values[0]),
                        hovertemplate='n=%{x}<br>diameter=%{y}<br>k=' + str(k_val) + f'<br>d={d_val}'
                    ))
                    trace_map1[(coset_name, d_val)].append(trace_idx)
                    trace_idx += 1

    total_traces1 = trace_idx
    buttons1 = []
    first_button = True
    init_n_range = [0, 10]
    init_diameter_range = [0, 10]

    for coset_name in coset_names:
        for d_val in d_values:
            if not trace_map1.get((coset_name, d_val), []):
                continue
            visibility = [False] * total_traces1
            for idx in trace_map1[(coset_name, d_val)]:
                visibility[idx] = True

            # Calculate ranges for this (coset, d) combination
            coset_d_df = df[(df['coset'] == coset_name) & (df['d'] == d_val)]
            n_range = [coset_d_df['n'].min() - 0.5, coset_d_df['n'].max() + 0.5]
            diameter_range = [0, coset_d_df['diameter'].max() * 1.05]

            buttons1.append(dict(label=f'{coset_name}, d={d_val}', method='update',
                                args=[{'visible': visibility},
                                      {'xaxis.range': n_range, 'yaxis.range': diameter_range}]))

            if first_button:
                init_n_range = n_range
                init_diameter_range = diameter_range
                first_button = False

    fig1.update_layout(
        title=dict(text=f'Diameter vs n - {group_name} (perm_type=1)', y=0.95),
        xaxis_title='n', yaxis_title='Diameter',
        xaxis=dict(range=init_n_range),
        yaxis=dict(range=init_diameter_range),
        updatemenus=[dict(buttons=buttons1, direction='down', x=0.0, xanchor='left', y=1.02, yanchor='bottom', showactive=True)],
        legend=dict(x=1.02, y=1), height=650, margin=dict(t=100)
    )
    fig1.write_html(f'{group_dir}/diameter.html')
    if show:
        fig1.show()

    # ===== PLOT 2: Growth curves (dropdown for coset+d+k, lines for n) =====
    fig2 = go.Figure()
    trace_map2 = {}
    trace_idx = 0

    for coset_name in coset_names:
        for d_val in d_values:
            for k_val in k_values:
                trace_map2[(coset_name, d_val, k_val)] = []
                coset_d_k_df = df[(df['coset'] == coset_name) & (df['d'] == d_val) & (df['k'] == k_val)].sort_values('n')
                for _, row in coset_d_k_df.iterrows():
                    growth = row['growth_parsed']
                    fig2.add_trace(go.Scatter(
                        x=list(range(len(growth))), y=growth,
                        mode='lines+markers', name=f"n={row['n']}",
                        visible=(coset_name == coset_names[0] and d_val == d_values[0] and k_val == k_values[0]),
                        hovertemplate='distance=%{x}<br>layer_size=%{y}<br>n=' + str(row['n'])
                    ))
                    trace_map2[(coset_name, d_val, k_val)].append(trace_idx)
                    trace_idx += 1

    total_traces2 = trace_idx
    buttons2 = []
    first_button = True
    init_distance_range = [0, 10]
    init_layer_range = [0, 6]

    for coset_name in coset_names:
        for d_val in d_values:
            for k_val in k_values:
                if not trace_map2.get((coset_name, d_val, k_val), []):
                    continue
                visibility = [False] * total_traces2
                for idx in trace_map2[(coset_name, d_val, k_val)]:
                    visibility[idx] = True

                # Calculate ranges for this (coset, d, k) combination
                coset_d_k_df = df[(df['coset'] == coset_name) & (df['d'] == d_val) & (df['k'] == k_val)]
                growths = coset_d_k_df['growth_parsed'].tolist()
                if growths:
                    max_distance = max(len(g) for g in growths)
                    max_layer = max(max(g) for g in growths)
                    min_layer = min(min(g) for g in growths if min(g) > 0)
                    distance_range = [-0.5, max_distance + 0.5]
                    layer_range = [np.log10(min_layer * 0.5), np.log10(max_layer * 2)]
                else:
                    distance_range = [0, 10]
                    layer_range = [0, 6]

                buttons2.append(dict(label=f'{coset_name}, d={d_val}, k={k_val}', method='update',
                                    args=[{'visible': visibility},
                                          {'xaxis.range': distance_range, 'yaxis.range': layer_range}]))

                if first_button:
                    init_distance_range = distance_range
                    init_layer_range = layer_range
                    first_button = False

    fig2.update_layout(
        title=dict(text=f'Growth Curves - {group_name} (perm_type=1)', y=0.95),
        xaxis_title='Distance', yaxis_title='Layer Size', yaxis_type='log',
        xaxis=dict(range=init_distance_range),
        yaxis=dict(range=init_layer_range),
        updatemenus=[dict(buttons=buttons2, direction='down', x=0.0, xanchor='left', y=1.02, yanchor='bottom', showactive=True)],
        legend=dict(x=1.02, y=1), height=650, margin=dict(t=100)
    )
    fig2.write_html(f'{group_dir}/growth.html')
    if show:
        fig2.show()

    # ===== PLOT 3: Last layer size vs n (dropdown for coset+d, lines for k) =====
    fig3 = go.Figure()
    trace_idx = 0
    trace_map3 = {}

    for coset_name in coset_names:
        for d_val in d_values:
            trace_map3[(coset_name, d_val)] = []
            coset_d_df = df[(df['coset'] == coset_name) & (df['d'] == d_val)]
            for k_val in k_values:
                k_df = coset_d_df[coset_d_df['k'] == k_val].sort_values('n')
                if len(k_df) > 0:
                    fig3.add_trace(go.Scatter(
                        x=k_df['n'], y=k_df['last_layer_size'],
                        mode='lines+markers', name=f'k={k_val}',
                        visible=(coset_name == coset_names[0] and d_val == d_values[0]),
                        hovertemplate='n=%{x}<br>last_layer=%{y}<br>k=' + str(k_val) + f'<br>d={d_val}'
                    ))
                    trace_map3[(coset_name, d_val)].append(trace_idx)
                    trace_idx += 1

    total_traces3 = trace_idx
    buttons3 = []
    first_button = True
    init_n_range3 = [0, 10]
    init_ll_range = [0, 6]

    for coset_name in coset_names:
        for d_val in d_values:
            if not trace_map3.get((coset_name, d_val), []):
                continue
            visibility = [False] * total_traces3
            for idx in trace_map3[(coset_name, d_val)]:
                visibility[idx] = True

            # Calculate ranges for this (coset, d) combination
            coset_d_df = df[(df['coset'] == coset_name) & (df['d'] == d_val)]
            n_range = [coset_d_df['n'].min() - 0.5, coset_d_df['n'].max() + 0.5]
            ll_min = coset_d_df['last_layer_size'].min()
            ll_max = coset_d_df['last_layer_size'].max()
            last_layer_range = [np.log10(max(ll_min * 0.5, 0.1)), np.log10(ll_max * 2)]

            buttons3.append(dict(label=f'{coset_name}, d={d_val}', method='update',
                                args=[{'visible': visibility},
                                      {'xaxis.range': n_range, 'yaxis.range': last_layer_range}]))

            if first_button:
                init_n_range3 = n_range
                init_ll_range = last_layer_range
                first_button = False

    fig3.update_layout(
        title=dict(text=f'Last Layer Size vs n - {group_name} (perm_type=1)', y=0.95),
        xaxis_title='n', yaxis_title='Last Layer Size', yaxis_type='log',
        xaxis=dict(range=init_n_range3),
        yaxis=dict(range=init_ll_range),
        updatemenus=[dict(buttons=buttons3, direction='down', x=0.0, xanchor='left', y=1.02, yanchor='bottom', showactive=True)],
        legend=dict(x=1.02, y=1), height=650, margin=dict(t=100)
    )
    fig3.write_html(f'{group_dir}/lastlayer.html')
    if show:
        fig3.show()

    print(f"Interactive plots saved to {group_dir}/")


def _plot_perm2_results(group_dir: str, df: pd.DataFrame, group_name: str, show: bool) -> None:
    """Plot results for perm_type=2 (no d parameter)."""
    coset_names = list(df['coset'].unique())
    k_values = sorted(df['k'].unique())

    # ===== PLOT 1: Diameter vs n (dropdown for coset, lines for k) =====
    fig1 = go.Figure()
    trace_idx = 0
    trace_map1 = {}

    for coset_name in coset_names:
        trace_map1[coset_name] = []
        coset_df = df[df['coset'] == coset_name]
        for k_val in k_values:
            k_df = coset_df[coset_df['k'] == k_val].sort_values('n')
            if len(k_df) > 0:
                fig1.add_trace(go.Scatter(
                    x=k_df['n'], y=k_df['diameter'],
                    mode='lines+markers', name=f'k={k_val}',
                    visible=(coset_name == coset_names[0]),
                    hovertemplate='n=%{x}<br>diameter=%{y}<br>k=' + str(k_val)
                ))
                trace_map1[coset_name].append(trace_idx)
                trace_idx += 1

    total_traces1 = trace_idx
    buttons1 = []
    first_button = True
    init_n_range = [0, 10]
    init_diameter_range = [0, 10]

    for coset_name in coset_names:
        if not trace_map1.get(coset_name, []):
            continue
        visibility = [False] * total_traces1
        for idx in trace_map1[coset_name]:
            visibility[idx] = True

        # Calculate ranges for this coset
        coset_df = df[df['coset'] == coset_name]
        n_range = [coset_df['n'].min() - 0.5, coset_df['n'].max() + 0.5]
        diameter_range = [0, coset_df['diameter'].max() * 1.05]

        buttons1.append(dict(label=coset_name, method='update',
                            args=[{'visible': visibility},
                                  {'xaxis.range': n_range, 'yaxis.range': diameter_range}]))

        if first_button:
            init_n_range = n_range
            init_diameter_range = diameter_range
            first_button = False

    fig1.update_layout(
        title=dict(text=f'Diameter vs n - {group_name} (perm_type=2)', y=0.95),
        xaxis_title='n', yaxis_title='Diameter',
        xaxis=dict(range=init_n_range),
        yaxis=dict(range=init_diameter_range),
        updatemenus=[dict(buttons=buttons1, direction='down', x=0.0, xanchor='left', y=1.02, yanchor='bottom', showactive=True)],
        legend=dict(x=1.02, y=1), height=650, margin=dict(t=100)
    )
    fig1.write_html(f'{group_dir}/diameter.html')
    if show:
        fig1.show()

    # ===== PLOT 2: Growth curves (dropdown for coset+k, lines for n) =====
    fig2 = go.Figure()
    trace_map2 = {}
    trace_idx = 0

    for coset_name in coset_names:
        for k_val in k_values:
            trace_map2[(coset_name, k_val)] = []
            coset_k_df = df[(df['coset'] == coset_name) & (df['k'] == k_val)].sort_values('n')
            for _, row in coset_k_df.iterrows():
                growth = row['growth_parsed']
                fig2.add_trace(go.Scatter(
                    x=list(range(len(growth))), y=growth,
                    mode='lines+markers', name=f"n={row['n']}",
                    visible=(coset_name == coset_names[0] and k_val == k_values[0]),
                    hovertemplate='distance=%{x}<br>layer_size=%{y}<br>n=' + str(row['n'])
                ))
                trace_map2[(coset_name, k_val)].append(trace_idx)
                trace_idx += 1

    total_traces2 = trace_idx
    buttons2 = []
    first_button = True
    init_distance_range = [0, 10]
    init_layer_range = [0, 6]

    for coset_name in coset_names:
        for k_val in k_values:
            if not trace_map2.get((coset_name, k_val), []):
                continue
            visibility = [False] * total_traces2
            for idx in trace_map2[(coset_name, k_val)]:
                visibility[idx] = True

            # Calculate ranges for this (coset, k) combination
            coset_k_df = df[(df['coset'] == coset_name) & (df['k'] == k_val)]
            growths = coset_k_df['growth_parsed'].tolist()
            if growths:
                max_distance = max(len(g) for g in growths)
                max_layer = max(max(g) for g in growths)
                min_layer = min(min(g) for g in growths if min(g) > 0)
                distance_range = [-0.5, max_distance + 0.5]
                layer_range = [np.log10(min_layer * 0.5), np.log10(max_layer * 2)]
            else:
                distance_range = [0, 10]
                layer_range = [0, 6]

            buttons2.append(dict(label=f'{coset_name}, k={k_val}', method='update',
                                args=[{'visible': visibility},
                                      {'xaxis.range': distance_range, 'yaxis.range': layer_range}]))

            if first_button:
                init_distance_range = distance_range
                init_layer_range = layer_range
                first_button = False

    fig2.update_layout(
        title=dict(text=f'Growth Curves - {group_name} (perm_type=2)', y=0.95),
        xaxis_title='Distance', yaxis_title='Layer Size', yaxis_type='log',
        xaxis=dict(range=init_distance_range),
        yaxis=dict(range=init_layer_range),
        updatemenus=[dict(buttons=buttons2, direction='down', x=0.0, xanchor='left', y=1.02, yanchor='bottom', showactive=True)],
        legend=dict(x=1.02, y=1), height=650, margin=dict(t=100)
    )
    fig2.write_html(f'{group_dir}/growth.html')
    if show:
        fig2.show()

    # ===== PLOT 3: Last layer size vs n (dropdown for coset, lines for k) =====
    fig3 = go.Figure()
    trace_idx = 0
    trace_map3 = {}

    for coset_name in coset_names:
        trace_map3[coset_name] = []
        coset_df = df[df['coset'] == coset_name]
        for k_val in k_values:
            k_df = coset_df[coset_df['k'] == k_val].sort_values('n')
            if len(k_df) > 0:
                fig3.add_trace(go.Scatter(
                    x=k_df['n'], y=k_df['last_layer_size'],
                    mode='lines+markers', name=f'k={k_val}',
                    visible=(coset_name == coset_names[0]),
                    hovertemplate='n=%{x}<br>last_layer=%{y}<br>k=' + str(k_val)
                ))
                trace_map3[coset_name].append(trace_idx)
                trace_idx += 1

    total_traces3 = trace_idx
    buttons3 = []
    first_button = True
    init_n_range3 = [0, 10]
    init_ll_range = [0, 6]

    for coset_name in coset_names:
        if not trace_map3.get(coset_name, []):
            continue
        visibility = [False] * total_traces3
        for idx in trace_map3[coset_name]:
            visibility[idx] = True

        # Calculate ranges for this coset
        coset_df = df[df['coset'] == coset_name]
        n_range = [coset_df['n'].min() - 0.5, coset_df['n'].max() + 0.5]
        ll_min = coset_df['last_layer_size'].min()
        ll_max = coset_df['last_layer_size'].max()
        last_layer_range = [np.log10(max(ll_min * 0.5, 0.1)), np.log10(ll_max * 2)]

        buttons3.append(dict(label=coset_name, method='update',
                            args=[{'visible': visibility},
                                  {'xaxis.range': n_range, 'yaxis.range': last_layer_range}]))

        if first_button:
            init_n_range3 = n_range
            init_ll_range = last_layer_range
            first_button = False

    fig3.update_layout(
        title=dict(text=f'Last Layer Size vs n - {group_name} (perm_type=2)', y=0.95),
        xaxis_title='n', yaxis_title='Last Layer Size', yaxis_type='log',
        xaxis=dict(range=init_n_range3),
        yaxis=dict(range=init_ll_range),
        updatemenus=[dict(buttons=buttons3, direction='down', x=0.0, xanchor='left', y=1.02, yanchor='bottom', showactive=True)],
        legend=dict(x=1.02, y=1), height=650, margin=dict(t=100)
    )
    fig3.write_html(f'{group_dir}/lastlayer.html')
    if show:
        fig3.show()

    print(f"Interactive plots saved to {group_dir}/")
