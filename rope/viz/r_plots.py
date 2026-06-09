#!/usr/bin/env python3
"""R-Axis 시각화 (cbf/axA/graph.py 기반, Figure 반환형)"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from rope.viz import colors as C

C.apply_style()

BBOX = C.LABEL_BBOX


def plot_icq_violin(df: pd.DataFrame, location: str = None) -> Optional[plt.Figure]:
    """Delay / PRR / Packet Size 바이올린 플롯 → Figure 반환"""
    df_wc = df[df['mode'] == 'WC']
    df_plot = df_wc[df_wc['location'] == location] if location else df_wc
    groups = sorted(df_plot['type'].unique())
    if not len(groups):
        return None

    fig, axes = plt.subplots(1, 3, figsize=(C.FIG_WIDTH_FULL, 2.5))
    mean_colors = {'CLM': C.WC, 'ETrA': C.WC_LIGHT}
    panels = [
        ('latency_ms',    'Latency (ms)',    '(a) Latency'),
        ('prr_mean',      'PRR (%)',         '(b) PRR'),
        ('pkt_size_mean', 'Packet Size (B)', '(c) Packet Size'),
    ]

    for ax, (metric, ylabel, title) in zip(axes, panels):
        for i, g in enumerate(groups, 1):
            data = df_plot[df_plot['type'] == g][metric].dropna().values
            if not len(data):
                continue
            vp = ax.violinplot([data], positions=[i], showmeans=True, showmedians=False, widths=0.7)
            vp['bodies'][0].set_facecolor(C.WOC_TINT60)
            vp['bodies'][0].set_edgecolor(C.WOC)
            vp['bodies'][0].set_alpha(0.8)
            mc = mean_colors.get(g, C.WC)
            vp['cmeans'].set_color(mc); vp['cmeans'].set_linewidth(1.5)
            for part in ['cbars', 'cmins', 'cmaxes']:
                vp[part].set_color(C.WOC); vp[part].set_linewidth(0.8)
            ax.annotate(f'μ={data.mean():.1f}\nn={len(data)}',
                        xy=(i, data.mean()), xytext=(3, 3),
                        textcoords='offset points', fontsize=7, color=mc, bbox=BBOX)
        ax.set_xticks(range(1, len(groups)+1)); ax.set_xticklabels(groups)
        ax.set_ylabel(ylabel, fontsize=8); ax.set_title(title, fontsize=9, color=C.TITLE)
        ax.tick_params(labelsize=7)

    loc_label = location or 'All'
    fig.suptitle(f'[R-1] ICQ Distribution — {loc_label}', fontsize=10, color=C.TITLE)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_data_quality(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Location × Type 별 데이터 품질 바 차트 → Figure 반환"""
    df_wc = df[df['mode'] == 'WC'].copy()
    locations = [l for l in ['KIAPI', 'Yeongjong', 'CARLA'] if l in df_wc['location'].values]
    if not locations:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(C.FIG_WIDTH_FULL, 2.8))
    panels = [
        ('n_samples',        'Samples',    '(a) Sample Count'),
        ('sampling_rate_hz', 'Rate (Hz)',  '(b) Sampling Rate'),
        ('valid_rate',        'Valid (%)',  '(c) Valid Rate'),
    ]
    bw = 0.35
    types = sorted(df_wc['type'].unique())
    x = np.arange(len(types))

    for ax, (metric, ylabel, title) in zip(axes, panels):
        for i, loc in enumerate(locations):
            loc_df = df_wc[df_wc['location'] == loc]
            means = [loc_df[loc_df['type'] == t][metric].mean() for t in types]
            stds  = [loc_df[loc_df['type'] == t][metric].std()  for t in types]
            offset = (i - 0.5) * bw
            kw = dict(capsize=3, error_kw={'linewidth': 0.8, 'color': C.WOC})
            if i == 0:
                ax.bar(x+offset, means, bw, yerr=stds, label=loc,
                       color=C.WOC_TINT60, edgecolor=C.WOC, **kw)
            else:
                ax.bar(x+offset, means, bw, yerr=stds, label=loc,
                       facecolor='none', edgecolor=C.WOC, hatch='//', **kw)
        ax.set_xticks(x); ax.set_xticklabels(types)
        ax.set_ylabel(ylabel, fontsize=8); ax.set_title(title, fontsize=9, color=C.TITLE)
        ax.tick_params(labelsize=7)

    axes[0].legend(fontsize=7)
    fig.suptitle('[R-1] Data Quality by Location', fontsize=10, color=C.TITLE)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_icq_timeline(ts_dir: Path, scenario: str) -> Optional[plt.Figure]:
    """대표 시나리오 Delay/PRR 시계열 → Figure 반환"""
    pq = Path(ts_dir) / f'{scenario}.parquet'
    if not pq.exists():
        return None

    df = pd.read_parquet(pq)
    dist = df['ego_cumulative_dist'].values if 'ego_cumulative_dist' in df.columns else np.arange(len(df))
    dist_rel = dist - dist[0]

    fig, axes = plt.subplots(2, 1, figsize=(C.FIG_WIDTH_HALF, 3.5))

    for ax, col, ylabel in zip(axes, ['delay', 'prr'], ['Delay (ms)', 'PRR (%)']):
        if col in df.columns:
            ax.plot(dist_rel, df[col], color=C.WC, linewidth=0.8)
            ax.axvline(x=50, color=C.SEC6, linestyle='--', linewidth=1)
            mu = df[col].dropna().mean()
            ax.text(0.02, 0.97, f'μ={mu:.1f}', transform=ax.transAxes, fontsize=7, va='top', bbox=BBOX)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0].set_title(f'[R-2] {scenario}', fontsize=8, color=C.TITLE)
    axes[1].set_xlabel('Distance (m)', fontsize=8)
    plt.tight_layout()
    return fig
