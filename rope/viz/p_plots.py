#!/usr/bin/env python3
"""P-Axis 시각화 — Histogram + Heatmap + Scatter"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional

from rope.viz import colors as C

C.apply_style()
BBOX = C.LABEL_BBOX

# (col, label, lower_better)
_HIST_LAYOUT = [
    # Safety (row 0)
    ('ttc_violation_2s',  'TTC <2s (%)',    True),
    ('ttc_violation_2s', 'TTC Viol(%)', True),
    ('drac_critical_ratio',         'DRAC Crit.Ratio',       True),
    ('d_min',             'Min Dist (m)',    False),
    # Stability (row 1)
    ('ego_jerk_rms',    'Ego Jerk RMS',    True),
    ('ego_jerk_peak',   'Ego Jerk Peak',   True),
    ('ego_yaw_rms',     'Ego Yaw RMS',     True),
    ('target_jerk_rms', 'Tgt Jerk RMS',    True),
    # Efficiency (row 2)
    ('tct_norm',    'TCT_norm',       True),
    ('v_mean_norm', 'v_mean_norm',    False),
    ('t_merge',     'Merge Time (s)', True),
    ('avoid_dist_tgt', 'Avoid Dist Tgt', False),
]
_NROW, _NCOL = 3, 4


def plot_performance_histogram(scenario_df: pd.DataFrame) -> Optional[plt.Figure]:
    if scenario_df.empty: return None
    wc  = scenario_df[scenario_df['mode'] == 'WC']
    woc = scenario_df[scenario_df['mode'] == 'WOC']
    fig, axes = plt.subplots(_NROW, _NCOL, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (col, label, lower_better) in zip(axes, _HIST_LAYOUT):
        wc_v  = wc[col].dropna().values  if col in scenario_df.columns else np.array([])
        woc_v = woc[col].dropna().values if col in scenario_df.columns else np.array([])
        if not (len(wc_v) and len(woc_v)):
            ax.set_visible(False); continue
        all_v = np.concatenate([wc_v, woc_v])
        bins = np.linspace(np.percentile(all_v, 2), np.percentile(all_v, 98), 20)
        ax.hist(woc_v, bins=bins, color=C.WOC_TINT60, edgecolor=C.WOC, alpha=0.75, label='SAD', density=True)
        ax.hist(wc_v,  bins=bins, color=C.WC_LIGHT_TINT60, edgecolor=C.WC, alpha=0.75, label='CAD', density=True)
        ax.axvline(woc_v.mean(), color=C.WOC, linewidth=1.2, linestyle='--')
        ax.axvline(wc_v.mean(),  color=C.WC,  linewidth=1.2, linestyle='--')
        delta = wc_v.mean() - woc_v.mean()
        pct = 100 * delta / woc_v.mean() if woc_v.mean() != 0 else 0
        improved = (delta < 0) == lower_better
        ax.text(0.97, 0.96, f'Δ={pct:+.1f}%', transform=ax.transAxes,
                fontsize=7, ha='right', va='top', color=C.WC if improved else C.SEC6, bbox=BBOX)
        ax.set_title(label, fontsize=8, color=C.TITLE, pad=3)
        ax.tick_params(labelsize=6); ax.set_yticks([])

    for i, rl in enumerate(['Safety', 'Stability', 'Efficiency']):
        axes[i * _NCOL].set_ylabel(rl, fontsize=9, fontweight='bold')

    fig.legend(handles=[
        mpatches.Patch(facecolor=C.WOC_TINT60, edgecolor=C.WOC, alpha=0.75, label='SAD'),
        mpatches.Patch(facecolor=C.WC_LIGHT_TINT60, edgecolor=C.WC, alpha=0.75, label='CAD'),
    ], loc='lower right', ncol=2, fontsize=8, frameon=True)
    fig.suptitle('[P] Performance Distribution: CAD vs SAD', fontsize=11, color=C.TITLE, y=1.01)
    plt.tight_layout()
    return fig


def plot_improvement_heatmap(stat_table: pd.DataFrame) -> Optional[plt.Figure]:
    if stat_table.empty: return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    for ax, cat in zip(axes, ['safety', 'stability', 'efficiency']):
        sub = stat_table[stat_table['category'] == cat]
        if sub.empty: ax.set_visible(False); continue
        metrics = sub['label'].tolist() if 'label' in sub.columns else sub['metric'].tolist()
        pcts    = sub['pct_change'].values
        sigs    = sub['significant'].values
        betters = sub['better_if'].values
        bar_colors = [C.WC if (p < 0) == (b == 'lower') else C.SEC6
                      for p, b in zip(pcts, betters)]
        bars = ax.barh(range(len(metrics)), pcts, color=bar_colors, alpha=0.8, edgecolor='white')
        for j, (bar, sig, pct) in enumerate(zip(bars, sigs, pcts)):
            if sig: ax.text(bar.get_width() + 0.3, j, '★', va='center', fontsize=8, color=C.SEC6)
            if np.isfinite(pct):
                ax.text(0, j, f'{pct:+.1f}%', va='center', ha='center', fontsize=7, color='white')
        ax.set_yticks(range(len(metrics))); ax.set_yticklabels(metrics, fontsize=7)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('% Change (CAD vs SAD)', fontsize=8)
        ax.set_title(cat.capitalize(), fontsize=9, color=C.TITLE)
        ax.tick_params(labelsize=7)

    fig.suptitle('[P] Improvement Rate: CAD vs SAD (★ p<0.05)', fontsize=10, color=C.TITLE)
    plt.tight_layout()
    return fig


def plot_scatter_comparison(scenario_df: pd.DataFrame) -> Optional[plt.Figure]:
    x_col, y_col = 'd_min', 'v_mean_norm'
    if not {x_col, y_col}.issubset(scenario_df.columns) or scenario_df.empty: return None
    fig, ax = plt.subplots(figsize=(4, 3.5))
    for mode, label, color, marker in [('WC','CAD',C.WC,'o'),('WOC','SAD',C.WOC,'s')]:
        sub = scenario_df[scenario_df['mode'] == mode]
        for stype, alpha in [('CLM', 0.9), ('ETrA', 0.6)]:
            s2 = sub[sub['type'] == stype]
            if not s2.empty:
                ax.scatter(s2[x_col], s2[y_col], c=color, marker=marker, s=40, alpha=alpha,
                           label=f'{label}-{stype}', edgecolors='white', linewidths=0.5)
    ax.set_xlabel('Min Distance (m)', fontsize=8)
    ax.set_ylabel('Mean Speed (m/s)', fontsize=8)
    ax.set_title('[P] Safety vs Efficiency', fontsize=9, color=C.TITLE)
    ax.legend(fontsize=6, ncol=2); ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig
