#!/usr/bin/env python3
"""O-Axis 시각화 — OST 기반"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from scipy.stats import spearmanr

from rope.viz import colors as C

C.apply_style()
BBOX = C.LABEL_BBOX


def plot_ost_bar(pairs: pd.DataFrame) -> Optional[plt.Figure]:
    """CAD vs SAD OST 막대 비교."""
    if pairs.empty or 'ost_wc' not in pairs.columns:
        return None
    fig, ax = plt.subplots(figsize=(max(6, len(pairs)*0.5), 4))
    x = np.arange(len(pairs))
    w = 0.35
    ax.bar(x - w/2, pairs['ost_wc'],  w, label='CAD (V2X)',    color=C.WC,  alpha=0.85)
    ax.bar(x + w/2, pairs['ost_woc'], w, label='SAD (LiDAR)', color=C.WOC, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs['scenario'].str.split('_wc_').str[-1], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('OST Score')
    ax.set_ylim(0, 1.05)
    ax.set_title('Observable State Tube — CAD vs SAD')
    ax.legend()
    ax.axhline(0, color='gray', lw=0.5)
    fig.tight_layout()
    return fig


def plot_ost_scatter(pairs: pd.DataFrame) -> Optional[plt.Figure]:
    """ΔOST scatter — V2X vs LiDAR OST 상관."""
    if pairs.empty or 'ost_wc' not in pairs.columns:
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    for loc, grp in pairs.groupby('location'):
        ax.scatter(grp['ost_woc'], grp['ost_wc'], label=loc, alpha=0.75, s=40)
    lim = [0, 1]
    ax.plot(lim, lim, 'k--', lw=0.8, alpha=0.4)
    ax.set_xlabel('OST — SAD (LiDAR)')
    ax.set_ylabel('OST — CAD (V2X)')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('OST: CAD vs SAD')
    ax.legend(fontsize=8)

    if len(pairs) >= 3:
        rho, p = spearmanr(pairs['ost_woc'].dropna(), pairs['ost_wc'].dropna())
        ax.text(0.05, 0.92, f'ρ={rho:.2f}  p={p:.3f}', transform=ax.transAxes,
                fontsize=9, bbox=BBOX)
    fig.tight_layout()
    return fig


def plot_ost_timeline(ts: dict, title: str = '') -> Optional[plt.Figure]:
    """Per-timestep |eₜ|, |Δeₜ|, valid flag — CAD vs SAD."""
    e_wc  = np.linalg.norm(np.where(np.isfinite(ts['e_wc']),  ts['e_wc'],  np.nan), axis=1)
    e_woc = np.linalg.norm(np.where(np.isfinite(ts['e_woc']), ts['e_woc'], np.nan), axis=1)
    de_wc  = np.linalg.norm(np.where(np.isfinite(ts['de_wc']),  ts['de_wc'],  np.nan), axis=1)
    de_woc = np.linalg.norm(np.where(np.isfinite(ts['de_woc']), ts['de_woc'], np.nan), axis=1)
    t = np.arange(len(e_wc))

    fig, axes = plt.subplots(3, 1, figsize=(9, 5), sharex=True)
    kw_wc  = dict(color=C.WC,  lw=0.8, label='CAD (V2X)')
    kw_woc = dict(color=C.WOC, lw=0.8, label='SAD (LiDAR)', alpha=0.85)

    axes[0].plot(t, e_wc,  **kw_wc);  axes[0].plot(t, e_woc, **kw_woc)
    axes[0].set_ylabel('|eₜ| (m)', fontsize=8); axes[0].legend(fontsize=7, loc='upper right')

    axes[1].plot(t, de_wc,  **kw_wc);  axes[1].plot(t, de_woc, **kw_woc)
    axes[1].set_ylabel('|Δeₜ| (m)', fontsize=8)

    for i, (flag, color) in enumerate([(ts['valid_wc'], C.WC), (ts['valid_woc'], C.WOC)]):
        axes[2].fill_between(t, i * 1.1, i * 1.1 + flag.astype(float),
                             step='post', color=color, alpha=0.7)
    axes[2].set_yticks([0.5, 1.6]); axes[2].set_yticklabels(['SAD', 'CAD'], fontsize=7)
    axes[2].set_ylabel('valid', fontsize=8); axes[2].set_xlabel('timestep', fontsize=8)

    for ax in axes:
        ax.tick_params(labelsize=7); ax.grid(False)
    if title:
        fig.suptitle(title, fontsize=9, color=C.TITLE, y=1.01)
    plt.tight_layout()
    return fig


def plot_delta_ost_histogram(pairs: pd.DataFrame) -> Optional[plt.Figure]:
    """ΔOST 분포 히스토그램."""
    if pairs.empty or 'delta_ost' not in pairs.columns:
        return None
    d = pairs['delta_ost'].dropna()
    if len(d) == 0:
        return None
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(d, bins=15, color=C.WC, alpha=0.8, edgecolor='white')
    ax.axvline(0, color='red', lw=1.2, ls='--', label='ΔOST=0')
    ax.axvline(d.mean(), color='navy', lw=1.2, ls='-.', label=f'mean={d.mean():.2f}')
    ax.set_xlabel('ΔOST  (CAD−SAD)')
    ax.set_ylabel('Count')
    ax.set_title('ΔOST Distribution')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
