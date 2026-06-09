#!/usr/bin/env python3
"""축/교차축별 대표 요약 패널 — 각 1개 figure."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from rope.viz import colors as C


def plot_r_summary(r_table: pd.DataFrame):
    """R: delay·PRR·pkt_size violin by scenario type (CAD only)."""
    if r_table.empty:
        return None
    wc = r_table[r_table['mode'] == 'WC'] if 'mode' in r_table.columns else r_table
    if wc.empty:
        return None
    C.apply_style()
    types = sorted(wc['type'].unique()) if 'type' in wc.columns else ['All']
    metrics = [('latency_ms', 'Latency (ms)'), ('prr_mean', 'PRR (%)'), ('pkt_size_mean', 'PktSize (B)')]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    for ax, (col, ylabel) in zip(axes, metrics):
        if col not in wc.columns:
            ax.set_visible(False); continue
        data = [wc[wc['type'] == t][col].dropna().values for t in types]
        data = [d for d in data if len(d) > 0]
        if not data:
            ax.set_visible(False); continue
        vp = ax.violinplot(data, positions=range(1, len(data)+1), showmeans=True, widths=0.6)
        for body in vp['bodies']:
            body.set_facecolor(C.WC); body.set_alpha(0.7)
        vp['cmeans'].set_color(C.WOC); vp['cmeans'].set_linewidth(1.5)
        ax.set_xticks(range(1, len(data)+1)); ax.set_xticklabels(types, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8); ax.tick_params(labelsize=7)
        for i, d in enumerate(data, 1):
            ax.text(i, np.nanmean(d), f'μ={np.nanmean(d):.1f}', ha='center', va='bottom',
                    fontsize=6, color=C.WOC, bbox=C.LABEL_BBOX)
    fig.suptitle('R-Axis: ICQ Distribution (CAD)', fontsize=10, fontweight='bold', color=C.TITLE)
    plt.tight_layout(); return fig


def plot_o_summary(o_pairs: pd.DataFrame):
    """O: OST-4D/2D CAD vs SAD scatter + ΔOST histogram."""
    if o_pairs is None or o_pairs.empty or 'ost_wc' not in o_pairs.columns:
        return None
    C.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))

    def _scatter(ax, woc, wc, title):
        m = np.isfinite(woc) & np.isfinite(wc)
        ax.scatter(woc[m], wc[m], c=C.WC, s=50, alpha=0.75, edgecolors='white', lw=0.5)
        hi = max(woc[m].max() if m.any() else 0, wc[m].max() if m.any() else 0, 0.05) * 1.05
        ax.plot([0, hi], [0, hi], 'k--', lw=0.8, alpha=0.4)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_xlabel('SAD (LiDAR)', fontsize=8); ax.set_ylabel('CAD (V2X)', fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold', color=C.TITLE)
        ax.tick_params(labelsize=7)
        if m.sum() >= 3:
            rho, _ = spearmanr(woc[m], wc[m])
            ax.text(0.05, 0.95, f'ρ={rho:.2f}', transform=ax.transAxes, fontsize=8, va='top', bbox=C.LABEL_BBOX)

    def _hist(ax, d, xlabel, title):
        d = d.dropna()
        if len(d) > 0:
            ax.hist(d, bins=min(15, len(d)), color=C.WC, alpha=0.8, edgecolor='white')
            ax.axvline(0, color='red', lw=1.2, ls='--', label='Δ=0')
            ax.axvline(d.mean(), color='navy', lw=1.2, ls='-.', label=f'μ={d.mean():.3f}')
            ax.legend(fontsize=7)
        ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel('Count', fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold', color=C.TITLE)
        ax.tick_params(labelsize=7)

    _scatter(axes[0][0], o_pairs['ost_woc'].values,   o_pairs['ost_wc'].values,   '(a) OST: CAD vs SAD')
    _hist(axes[0][1], o_pairs['delta_ost'],   'ΔOST (CAD − SAD)',   '(b) ΔOST')
    _scatter(axes[1][0], o_pairs['ost_c_woc'].values, o_pairs['ost_c_wc'].values, '(c) OST_c: CAD vs SAD')
    _hist(axes[1][1], o_pairs['delta_ost_c'], 'ΔOST_c (CAD − SAD)', '(d) ΔOST_c')

    fig.suptitle('O-Axis: Observable State Tube', fontsize=10, fontweight='bold', color=C.TITLE)
    plt.tight_layout(); return fig


def plot_p_summary(stat_table: pd.DataFrame):
    """P: CAD vs SAD Cohen's d 개선 요약."""
    from rope.viz.overall_plots import plot_improvement_summary
    return plot_improvement_summary(stat_table)


def plot_cross_summary(r_table: pd.DataFrame, o_pairs: pd.DataFrame, p_scenarios: pd.DataFrame):
    """Cross: R→O(PRR→ΔOST) · O→P(ΔOST→Safety) · R→P(PRR→Jerk) 체인 산점도."""
    if r_table.empty or o_pairs is None or o_pairs.empty:
        return None
    r_wc = r_table[r_table['mode'] == 'WC'] if 'mode' in r_table.columns else r_table
    p_wc = (p_scenarios[p_scenarios['mode'] == 'WC']
            if (not p_scenarios.empty and 'mode' in p_scenarios.columns)
            else p_scenarios)

    ro  = pd.merge(r_wc, o_pairs, on='scenario', how='inner', suffixes=('_r', '_o'))
    rop = pd.merge(ro,  p_wc,    on='scenario', how='inner') if not p_wc.empty else pd.DataFrame()

    C.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))

    def _sp(ax, x, y, xl, yl, title, color=C.WC):
        x, y = np.asarray(x, float), np.asarray(y, float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            ax.set_visible(False); return
        ax.scatter(x[mask], y[mask], c=color, s=35, alpha=0.7, edgecolors='white', lw=0.5)
        rho, p = spearmanr(x[mask], y[mask])
        ax.text(0.05, 0.95, f'ρ={rho:.2f}{"★" if p<0.05 else ""}',
                transform=ax.transAxes, fontsize=8, va='top', bbox=C.LABEL_BBOX)
        ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight='bold', color=C.TITLE); ax.tick_params(labelsize=7)

    prr = 'prr_mean_r' if 'prr_mean_r' in ro.columns else 'prr_mean'
    _sp(axes[0], ro.get(prr), ro.get('delta_ost'), 'PRR (%)', 'ΔOST', '(a) R→O: PRR → ΔOST')

    hw = next((c for c in ['ttc_violation_2s', 'ttc_violation_2s', 'd_min'] if c in rop.columns), None)
    if not rop.empty and hw:
        _sp(axes[1], rop.get('delta_ost'), rop[hw], 'ΔOST',
            hw.replace('_', ' ').title(), '(b) O→P: ΔOST → Safety', C.SEC5)
    else:
        axes[1].set_visible(False)

    jerk = 'ego_jerk_rms'
    prr_rop = 'prr_mean_r' if 'prr_mean_r' in rop.columns else prr
    if not rop.empty and jerk in rop.columns:
        _sp(axes[2], rop.get(prr_rop), rop[jerk], 'PRR (%)', 'Jerk RMS', '(c) R→P: PRR → Stability', C.SEC4)
    else:
        axes[2].set_visible(False)

    fig.suptitle('R → O → P Chain', fontsize=10, fontweight='bold', color=C.TITLE)
    plt.tight_layout(); return fig
