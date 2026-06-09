#!/usr/bin/env python3
"""Cross-Axis 시각화 — R↔O, O↔P, R↔P"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional
from scipy.stats import spearmanr

from rope.viz import colors as C

C.apply_style()
BBOX = C.LABEL_BBOX


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> Optional[plt.Figure]:
    """3개 경로(R→O, O→P, R→P)의 Spearman ρ 히트맵"""
    if corr_df.empty:
        return None

    paths = ['R→O', 'O→P', 'R→P']
    path_dfs = {p: corr_df[corr_df['path'] == p] for p in paths}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, path in zip(axes, paths):
        sub = path_dfs[path]
        if sub.empty:
            ax.set_visible(False)
            continue

        x_labels = sub['x'].unique().tolist()
        y_labels = sub['y'].unique().tolist()
        mat = np.full((len(y_labels), len(x_labels)), np.nan)
        sig = np.zeros_like(mat, dtype=bool)

        for _, row in sub.iterrows():
            xi = x_labels.index(row['x'])
            yi = y_labels.index(row['y'])
            mat[yi, xi] = row['rho']
            sig[yi, xi] = row['p_value'] < 0.05 if np.isfinite(row['p_value']) else False

        im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(x_labels))); ax.set_xticklabels(x_labels, fontsize=8, rotation=20, ha='right')
        ax.set_yticks(range(len(y_labels))); ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_title(path, fontsize=10, color=C.TITLE, fontweight='bold')

        for yi in range(len(y_labels)):
            for xi in range(len(x_labels)):
                v = mat[yi, xi]
                if np.isfinite(v):
                    star = '★' if sig[yi, xi] else ''
                    ax.text(xi, yi, f'{v:.2f}{star}', ha='center', va='center',
                            fontsize=7, color='white' if abs(v) > 0.5 else 'black')

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Cross-Axis Spearman Correlation (★ p<0.05)', fontsize=11, color=C.TITLE)
    plt.tight_layout()
    return fig


def plot_chain_scatter(r_table: pd.DataFrame, o_table: pd.DataFrame,
                       p_table: pd.DataFrame) -> Optional[plt.Figure]:
    """R→O→P 인과 체인 핵심 산점도 (3-panel)"""
    # 모든 축 WC만, 시나리오 기준 병합
    r_wc = r_table[r_table['mode']=='WC'] if not r_table.empty else r_table
    o_wc = o_table[o_table['mode']=='WC'] if not o_table.empty else o_table

    ro = pd.merge(r_wc, o_wc, on='scenario', how='inner', suffixes=('_r','_o'))
    rop = pd.merge(ro, p_table, on='scenario', how='inner') if not p_table.empty else pd.DataFrame()

    if ro.empty:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    def _scatter(ax, xdata, ydata, xlabel, ylabel, title, color=C.WC):
        x = np.asarray(xdata, dtype=float)
        y = np.asarray(ydata, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            ax.set_visible(False)
            return
        ax.scatter(x[mask], y[mask], c=color, s=35, alpha=0.7, edgecolors='white', linewidths=0.5)
        rho, p = spearmanr(x[mask], y[mask])
        sig = '★' if p < 0.05 else ''
        ax.text(0.02, 0.97, f'ρ={rho:.3f}{sig}\np={p:.4f}', transform=ax.transAxes,
                fontsize=8, va='top', bbox=BBOX)
        ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, color=C.TITLE); ax.tick_params(labelsize=7)

    # Panel 1: R→O  Delay vs ΔOST
    if 'latency_ms_r' in ro.columns and 'delta_ost_o' in ro.columns:
        _scatter(axes[0], ro['latency_ms_r'], ro['delta_ost_o'],
                 'Latency (ms)', 'ΔOST', '(a) R→O: Latency vs ΔOST')
    elif 'latency_ms' in ro.columns and 'delta_ost' in ro.columns:
        _scatter(axes[0], ro['latency_ms'], ro['delta_ost'],
                 'Latency (ms)', 'ΔOST', '(a) R→O: Latency vs ΔOST')
    else:
        axes[0].set_visible(False)

    # Panel 2: O→P  ΔOST vs Headway viol
    if not rop.empty and 'delta_ost' in rop.columns and 'd_mean_m' in rop.columns:
        _scatter(axes[1], rop['delta_ost'], rop['d_mean_m'],
                 'ΔOST', 'Mean Dist (m)', '(b) O→P: ΔOST vs Safety', color=C.SEC5)
    else:
        axes[1].set_visible(False)

    # Panel 3: R→P  PRR vs Jerk RMS
    if not rop.empty and 'prr_mean_r' in rop.columns and 'Jerk_RMS' in rop.columns:
        _scatter(axes[2], rop['prr_mean_r'], rop['Jerk_RMS'],
                 'PRR (%)', 'Jerk RMS', '(c) R→P: PRR vs Stability', color=C.SEC4)
    elif not rop.empty and 'prr_mean' in rop.columns and 'Jerk_RMS' in rop.columns:
        _scatter(axes[2], rop['prr_mean'], rop['Jerk_RMS'],
                 'PRR (%)', 'Jerk RMS', '(c) R→P: PRR vs Stability', color=C.SEC4)
    else:
        axes[2].set_visible(False)

    fig.suptitle('Cross-Axis Chain: R → O → P  (CAD only)', fontsize=10, color=C.TITLE)
    plt.tight_layout()
    return fig


def plot_cross_axis_summary(corr_df: pd.DataFrame) -> Optional[plt.Figure]:
    """전체 상관 계수 크기 순 막대 그래프 (3 경로별 색상)"""
    if corr_df.empty:
        return None

    sub = corr_df.dropna(subset=['rho'])
    sub = sub.copy()
    sub['abs_rho'] = sub['rho'].abs()
    sub['label'] = sub['x'] + ' → ' + sub['y']
    sub = sub.sort_values('abs_rho', ascending=True)

    path_colors = {'R→O': C.WC, 'O→P': C.SEC5, 'R→P': C.SEC4}

    fig, ax = plt.subplots(figsize=(7, max(3, len(sub)*0.25)))
    colors_bar = [path_colors.get(p, C.WOC) for p in sub['path']]
    bars = ax.barh(range(len(sub)), sub['rho'], color=colors_bar, alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub['label'], fontsize=7)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(0.3,  color=C.GRID, linestyle='--', linewidth=0.8)
    ax.axvline(-0.3, color=C.GRID, linestyle='--', linewidth=0.8)
    ax.set_xlabel('Spearman ρ', fontsize=8)
    ax.set_title('Cross-Axis Correlation Summary', fontsize=10, color=C.TITLE)

    # sig markers
    for i, (_, row) in enumerate(sub.iterrows()):
        if row['p_value'] < 0.05:
            ax.text(row['rho'] + 0.02, i, '★', va='center', fontsize=8, color='black')

    # legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, alpha=0.8, label=p) for p, c in path_colors.items()]
    ax.legend(handles=handles, fontsize=8, loc='lower right')
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig
