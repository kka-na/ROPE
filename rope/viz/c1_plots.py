#!/usr/bin/env python3
"""C-1 Driving Dynamics graphs: per-scenario Safety/Stability timeseries"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

from rope.viz import colors as C


def _resolve_path(ts_dir, name):
    """Exact match first, then glob for timestamp suffix."""
    p = ts_dir / f'{name}.parquet'
    if p.exists():
        return p
    hits = sorted(ts_dir.glob(f'{name}_*.parquet'))
    return hits[0] if hits else None


def _load_scenario_data(ts_dir, scenario_base):
    ts_dir = Path(ts_dir)
    for speed_tag in ('_same', '_faster', '_slower'):
        if speed_tag in scenario_base:
            wc_name  = scenario_base.replace(speed_tag, f'_WC{speed_tag}')
            woc_name = scenario_base.replace(speed_tag, f'_WOC{speed_tag}')
            break
    else:
        wc_name  = scenario_base + '_WC'
        woc_name = scenario_base + '_WOC'
    wc_path  = _resolve_path(ts_dir, wc_name)
    woc_path = _resolve_path(ts_dir, woc_name)
    if wc_path is None or woc_path is None:
        return None
    wc_name, woc_name = wc_path.stem, woc_path.stem
    df_wc, df_woc = pd.read_parquet(wc_path), pd.read_parquet(woc_path)
    dist_wc  = df_wc['ego_cumulative_dist'].values  - df_wc['ego_cumulative_dist'].values[0]
    dist_woc = df_woc['ego_cumulative_dist'].values - df_woc['ego_cumulative_dist'].values[0]
    adist_wc  = df_wc['action_start_dist'].iloc[0]  if 'action_start_dist' in df_wc.columns  else np.nan
    adist_woc = df_woc['action_start_dist'].iloc[0] if 'action_start_dist' in df_woc.columns else np.nan
    return dict(wc_name=wc_name, woc_name=woc_name, df_wc=df_wc, df_woc=df_woc,
                dist_wc=dist_wc, dist_woc=dist_woc,
                action_dist_wc=adist_wc, action_dist_woc=adist_woc,
                action_label='LC Start' if 'clm' in scenario_base.lower() else 'Avoid Start')


def _load_entropy_aligned(ts_dir, scenario_base):
    data = _load_scenario_data(ts_dir, scenario_base)
    if data is None:
        return None
    df_wc, df_woc = data['df_wc'], data['df_woc']
    dist_wc, dist_woc = data['dist_wc'], data['dist_woc']
    dist_min = max(dist_wc.min(), dist_woc.min())
    dist_max = min(dist_wc.max(), dist_woc.max())
    common_dist = np.linspace(dist_min, dist_max, min(len(dist_wc), len(dist_woc)))
    # ΔOST: h_v2v/h_lidar 있으면 사용, 없으면 속도 차이(SAD-CAD) 대체
    if 'h_v2v' in df_wc.columns and 'h_lidar' in df_woc.columns:
        h_v2v_wc    = np.interp(common_dist, dist_wc,  df_wc['h_v2v'].values)
        h_lidar_woc = np.interp(common_dist, dist_woc, df_woc['h_lidar'].values)
        delta_h = h_lidar_woc - h_v2v_wc
    else:
        v_wc  = np.interp(common_dist, dist_wc,  df_wc['ego_velocity'].values)
        v_woc = np.interp(common_dist, dist_woc, df_woc['ego_velocity'].values)
        delta_h = v_woc - v_wc

    def interp_col(df, dist, col):
        if col not in df.columns:
            return np.full_like(common_dist, np.nan)
        vals = df[col].replace([np.inf, -np.inf], np.nan).values
        valid = ~np.isnan(vals)
        return np.interp(common_dist, dist[valid], vals[valid]) if valid.sum() > 1 else np.full_like(common_dist, np.nan)

    cols = ['ttc', 'drac', 'd_rel', 'ego_jerk_total', 'ego_yaw_rate', 'target_jerk_total', 'target_yaw_rate']
    return dict(delta_h=delta_h, common_dist=common_dist,
                metrics_wc={c: interp_col(df_wc, dist_wc, c) for c in cols},
                metrics_woc={c: interp_col(df_woc, dist_woc, c) for c in cols},
                action_dist_wc=data['action_dist_wc'], action_dist_woc=data['action_dist_woc'],
                action_label=data['action_label'])


def _vlines(ax, action_dist, action_label):
    ax.axvline(x=20, color=C.SEC6, linestyle='--', linewidth=1, alpha=0.6)
    if pd.notna(action_dist):
        ax.axvline(x=action_dist, color=C.SEC4, linestyle='--', linewidth=1.5, alpha=0.7)


def _dh_ylim(delta_h):
    dh_min, dh_max = np.nanmin(delta_h), np.nanmax(delta_h)
    r = (dh_max - dh_min) or 1
    return (dh_min - r * 0.1, dh_max + r * 4.0)


def _metric_ylim(a, b, bm=1.5, tm=0.15):
    v = np.concatenate([np.asarray(a).flatten(), np.asarray(b).flatten()])
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    mn, mx = v.min(), v.max()
    r = (mx - mn) or abs(mx) * 0.5 or 1
    return (mn - r * bm, mx + r * tm)


def plot_safety_dynamics(ts_dir, scenario_base, out_path):
    """C1-1: Safety timeseries WC|WOC (4×2 grid)"""
    from rope.viz.trajectory import plot_trajectory
    data = _load_scenario_data(ts_dir, scenario_base)
    if data is None:
        return
    df_wc, df_woc = data['df_wc'], data['df_woc']
    dist_wc, dist_woc = data['dist_wc'], data['dist_woc']
    xlim = (0, max(dist_wc.max(), dist_woc.max()))

    fig, axes = plt.subplots(4, 2, figsize=(C.FIG_WIDTH_FULL, 7),
                             gridspec_kw={'height_ratios': [1.2, 1, 1, 1]})
    plot_trajectory(data['wc_name'],  ts_dir=ts_dir, ax=axes[0, 0])
    axes[0, 0].set_title('(a) Trajectory - CAD', fontsize=9, color=C.TITLE)
    plot_trajectory(data['woc_name'], ts_dir=ts_dir, ax=axes[0, 1])
    axes[0, 1].set_title('(b) Trajectory - SAD', fontsize=9, color=C.TITLE)

    ttc_wc  = df_wc['ttc'].replace([np.inf, -np.inf], np.nan).clip(upper=100)
    ttc_woc = df_woc['ttc'].replace([np.inf, -np.inf], np.nan).clip(upper=100)
    drac_wc  = df_wc['drac'].replace([np.inf, -np.inf], np.nan)
    drac_woc = df_woc['drac'].replace([np.inf, -np.inf], np.nan)

    all_ttc = pd.concat([ttc_wc.dropna(), ttc_woc.dropna()])
    ttc_ylim = (0, min(20, all_ttc.max() * 1.1)) if len(all_ttc) > 0 else (0, 20)
    all_drac = pd.concat([drac_wc.dropna(), drac_woc.dropna()])
    drac_ylim = (0, all_drac.max() * 1.1) if len(all_drac) > 0 else (0, 0.1)
    all_hw = pd.concat([df_wc['d_rel'].dropna(), df_woc['d_rel'].dropna()])
    hw_ylim = (0, all_hw.max() * 1.1) if len(all_hw) > 0 else None

    for col, (dist, ttc, mode, color, adist) in enumerate([
        (dist_wc, ttc_wc, 'CAD', C.WC, data['action_dist_wc']),
        (dist_woc, ttc_woc, 'SAD', C.WOC, data['action_dist_woc'])]):
        ax = axes[1, col]
        ax.plot(dist, ttc, color=color, linewidth=0.8, alpha=0.8)
        ax.axhline(y=2, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
        _vlines(ax, adist, data['action_label'])
        ax.set_ylabel('TTC (s)', fontsize=8, color=C.AXES)
        ax.set_title(f'({"c" if col==0 else "d"}) TTC - {mode}', fontsize=9, color=C.TITLE)
        ax.set_xlim(xlim); ax.set_ylim(ttc_ylim); ax.tick_params(labelsize=7)

    for col, (dist, drac, mode, color, adist) in enumerate([
        (dist_wc, drac_wc, 'CAD', C.WC, data['action_dist_wc']),
        (dist_woc, drac_woc, 'SAD', C.WOC, data['action_dist_woc'])]):
        ax = axes[2, col]
        ax.plot(dist, drac, color=color, linewidth=0.8, alpha=0.8)
        ax.axhline(y=3.4, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
        _vlines(ax, adist, data['action_label'])
        ax.set_ylabel('DRAC (m/s²)', fontsize=8, color=C.AXES)
        ax.set_title(f'({"e" if col==0 else "f"}) DRAC - {mode}', fontsize=9, color=C.TITLE)
        ax.set_xlim(xlim)
        if drac_ylim: ax.set_ylim(drac_ylim)
        ax.tick_params(labelsize=7)

    for col, (dist, df, mode, color, adist) in enumerate([
        (dist_wc, df_wc, 'CAD', C.WC, data['action_dist_wc']),
        (dist_woc, df_woc, 'SAD', C.WOC, data['action_dist_woc'])]):
        ax = axes[3, col]
        ax.plot(dist, df['d_rel'], color=color, linewidth=0.8, alpha=0.8)
        _vlines(ax, adist, data['action_label'])
        ax.set_ylabel('Headway (m)', fontsize=8, color=C.AXES)
        ax.set_xlabel('Distance (m)', fontsize=8, color=C.AXES)
        ax.set_title(f'({"g" if col==0 else "h"}) Headway - {mode}', fontsize=9, color=C.TITLE)
        ax.set_xlim(xlim)
        if hw_ylim: ax.set_ylim(hw_ylim)
        ax.tick_params(labelsize=7)

    fig.suptitle(f'{scenario_base} - Safety', fontsize=11, color=C.TITLE)
    fig.legend(handles=[Line2D([0],[0],color=C.SEC6,ls='--',lw=1,label='Signal (20m)'),
                        Line2D([0],[0],color=C.SEC4,ls='--',lw=1.5,label=data['action_label'])],
               loc='upper right', fontsize=7, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_stability_dynamics(ts_dir, scenario_base, out_path):
    """C1-2: Stability timeseries WC|WOC (3×2 grid)"""
    from rope.viz.trajectory import plot_trajectory
    data = _load_scenario_data(ts_dir, scenario_base)
    if data is None:
        return
    df_wc, df_woc = data['df_wc'], data['df_woc']
    dist_wc, dist_woc = data['dist_wc'], data['dist_woc']
    xlim = (0, max(dist_wc.max(), dist_woc.max()))
    is_etra = 'etra' in scenario_base.lower()
    prefix, subject = ('target', 'Target') if is_etra else ('ego', 'Ego')
    jerk_col, yaw_col = f'{prefix}_jerk_total', f'{prefix}_yaw_rate'

    fig, axes = plt.subplots(3, 2, figsize=(C.FIG_WIDTH_FULL, 5),
                             gridspec_kw={'height_ratios': [1.2, 1, 1]})
    plot_trajectory(data['wc_name'],  ts_dir=ts_dir, ax=axes[0, 0])
    axes[0, 0].set_title('(a) Trajectory - CAD', fontsize=9, color=C.TITLE)
    plot_trajectory(data['woc_name'], ts_dir=ts_dir, ax=axes[0, 1])
    axes[0, 1].set_title('(b) Trajectory - SAD', fontsize=9, color=C.TITLE)

    all_jerk = pd.concat([df_wc[jerk_col].dropna(), df_woc[jerk_col].dropna()]) if jerk_col in df_wc.columns else pd.Series([], dtype=float)
    jerk_ylim = (all_jerk.min()*1.1, all_jerk.max()*1.1) if len(all_jerk) > 0 else None
    all_yaw  = pd.concat([df_wc[yaw_col].dropna(),  df_woc[yaw_col].dropna()])  if yaw_col  in df_wc.columns else pd.Series([], dtype=float)
    yaw_ylim  = (all_yaw.min()*1.1,  all_yaw.max()*1.1)  if len(all_yaw)  > 0 else None

    for col, (dist, df, mode, color, adist) in enumerate([
        (dist_wc, df_wc, 'CAD', C.WC, data['action_dist_wc']),
        (dist_woc, df_woc, 'SAD', C.WOC, data['action_dist_woc'])]):
        ax = axes[1, col]
        if jerk_col in df.columns:
            ax.plot(dist, df[jerk_col], color=color, linewidth=0.8, alpha=0.8)
        _vlines(ax, adist, data['action_label'])
        ax.set_ylabel(f'{subject} Jerk (m/s³)', fontsize=8, color=C.AXES)
        ax.set_title(f'({"c" if col==0 else "d"}) {subject} Jerk - {mode}', fontsize=9, color=C.TITLE)
        ax.set_xlim(xlim)
        if jerk_ylim: ax.set_ylim(jerk_ylim)
        ax.tick_params(labelsize=7)

    for col, (dist, df, mode, color, adist) in enumerate([
        (dist_wc, df_wc, 'CAD', C.WC, data['action_dist_wc']),
        (dist_woc, df_woc, 'SAD', C.WOC, data['action_dist_woc'])]):
        ax = axes[2, col]
        if yaw_col in df.columns:
            ax.plot(dist, df[yaw_col], color=color, linewidth=0.8, alpha=0.8)
        ax.axhline(y=0, color='black', ls='-', lw=0.5, alpha=0.3)
        _vlines(ax, adist, data['action_label'])
        ax.set_ylabel(f'{subject} Yaw Rate (rad/s)', fontsize=8, color=C.AXES)
        ax.set_xlabel('Distance (m)', fontsize=8, color=C.AXES)
        ax.set_title(f'({"e" if col==0 else "f"}) {subject} Yaw - {mode}', fontsize=9, color=C.TITLE)
        ax.set_xlim(xlim)
        if yaw_ylim: ax.set_ylim(yaw_ylim)
        ax.tick_params(labelsize=7)

    fig.suptitle(f'{scenario_base} - {subject} Stability', fontsize=11, color=C.TITLE)
    fig.legend(handles=[Line2D([0],[0],color=C.SEC6,ls='--',lw=1,label='Signal (20m)'),
                        Line2D([0],[0],color=C.SEC4,ls='--',lw=1.5,label=data['action_label'])],
               loc='upper right', fontsize=7, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close()


def _dual_axis_panel(ax, dist, main_wc, main_woc, delta_h, dh_ylim, main_ylim,
                     adist_wc, adist_woc, ylabel, title, hline=None, wc_worse_fn=None):
    ax.grid(False)
    if wc_worse_fn is not None:
        wc_worse = wc_worse_fn(main_wc, main_woc)
        ax.fill_between(dist, 0, 1, where=wc_worse,
                        transform=ax.get_xaxis_transform(), alpha=0.2, color=C.WC_LIGHT)
    ax.plot(dist, main_wc,  color=C.WC,  linewidth=0.8, alpha=0.8, label='CAD')
    ax.plot(dist, main_woc, color=C.WOC, linewidth=0.8, alpha=0.8, label='SAD')
    if hline is not None:
        ax.axhline(y=hline, color='red', ls=':', lw=0.8, alpha=0.5)
    ax.axvline(x=20, color=C.SEC6, ls='--', lw=1, alpha=0.6)
    if pd.notna(adist_wc):
        ax.axvline(x=adist_wc,  color=C.SEC4, ls='--', lw=1.5, alpha=0.7)
    if pd.notna(adist_woc) and adist_woc != adist_wc:
        ax.axvline(x=adist_woc, color=C.SEC4, ls=':',  lw=1.5, alpha=0.5)
    if main_ylim:
        ax.set_ylim(main_ylim)
    ax.set_ylabel(ylabel, fontsize=8, color=C.AXES)
    ax.set_title(title, fontsize=9, color=C.TITLE)
    ax.legend(fontsize=6, loc='upper left')
    ax.tick_params(labelsize=7)
    ax2 = ax.twinx()
    ax2.plot(dist, delta_h, color=C.SEC5, linewidth=0.8, alpha=0.7, label='Δ')
    ax2.axhline(y=0, color=C.SEC5, ls='-', lw=0.5, alpha=0.3)
    ax2.set_ylabel('Δ', fontsize=8, color=C.SEC5)
    ax2.tick_params(axis='y', labelcolor=C.SEC5, labelsize=7)
    ax2.set_ylim(dh_ylim)


def plot_safety_vs_entropy(ts_dir, scenario_base, out_path):
    """C1-4: Safety vs ΔH dual-axis (3 panels)"""
    data = _load_entropy_aligned(ts_dir, scenario_base)
    if data is None:
        return
    dh, cd = data['delta_h'], data['common_dist']
    mwc, mwoc = data['metrics_wc'], data['metrics_woc']
    dh_yl = _dh_ylim(dh)
    al = data['action_label']

    ttc_wc  = np.clip(mwc.get('ttc', []), 0, 20)
    ttc_woc = np.clip(mwoc.get('ttc', []), 0, 20)

    fig, axes = plt.subplots(3, 1, figsize=(C.FIG_WIDTH_HALF, 6.5))
    _dual_axis_panel(axes[0], cd, ttc_wc, ttc_woc, dh, dh_yl,
                     _metric_ylim(ttc_wc, ttc_woc),
                     data['action_dist_wc'], data['action_dist_woc'],
                     'TTC (s)', '(a) TTC', hline=2,
                     wc_worse_fn=lambda a,b: a < b)
    _dual_axis_panel(axes[1], cd, mwc.get('drac',[]), mwoc.get('drac',[]), dh, dh_yl,
                     _metric_ylim(mwc.get('drac',[]), mwoc.get('drac',[])),
                     data['action_dist_wc'], data['action_dist_woc'],
                     'DRAC (m/s²)', '(b) DRAC', hline=3.4,
                     wc_worse_fn=lambda a,b: a > b)
    _dual_axis_panel(axes[2], cd, mwc.get('d_rel',[]), mwoc.get('d_rel',[]), dh, dh_yl,
                     _metric_ylim(mwc.get('d_rel',[]), mwoc.get('d_rel',[])),
                     data['action_dist_wc'], data['action_dist_woc'],
                     'Headway (m)', '(c) Headway',
                     wc_worse_fn=lambda a,b: a < b)
    axes[2].set_xlabel('Distance (m)', fontsize=8, color=C.AXES)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle(f'{scenario_base} - Safety with ΔOST', fontsize=10, color=C.TITLE, y=0.98)
    fig.legend(handles=[Line2D([0],[0],color=C.SEC6,ls='--',lw=1,label='Signal (20m)'),
                        Line2D([0],[0],color=C.SEC4,ls='--',lw=1.5,label=al)],
               loc='upper center', bbox_to_anchor=(0.5, 0.96), fontsize=7, frameon=False, ncol=2)
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_stability_vs_entropy(ts_dir, scenario_base, out_path):
    """C1-5: Stability vs ΔH dual-axis (2 panels)"""
    data = _load_entropy_aligned(ts_dir, scenario_base)
    if data is None:
        return
    dh, cd = data['delta_h'], data['common_dist']
    mwc, mwoc = data['metrics_wc'], data['metrics_woc']
    dh_yl = _dh_ylim(dh)
    is_etra = 'etra' in scenario_base.lower()
    prefix, subject = ('target', 'Target') if is_etra else ('ego', 'Ego')
    al = data['action_label']

    jk = f'{prefix}_jerk_total'; yw = f'{prefix}_yaw_rate'
    fig, axes = plt.subplots(2, 1, figsize=(C.FIG_WIDTH_HALF, 4.5))
    _dual_axis_panel(axes[0], cd, mwc.get(jk,[]), mwoc.get(jk,[]), dh, dh_yl,
                     _metric_ylim(mwc.get(jk,[]), mwoc.get(jk,[])),
                     data['action_dist_wc'], data['action_dist_woc'],
                     f'{subject} Jerk (m/s³)', f'(a) {subject} Jerk')
    _dual_axis_panel(axes[1], cd, mwc.get(yw,[]), mwoc.get(yw,[]), dh, dh_yl,
                     _metric_ylim(mwc.get(yw,[]), mwoc.get(yw,[])),
                     data['action_dist_wc'], data['action_dist_woc'],
                     f'{subject} Yaw Rate (rad/s)', f'(b) {subject} Yaw Rate')
    axes[1].set_xlabel('Distance (m)', fontsize=8, color=C.AXES)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.suptitle(f'{scenario_base} - {subject} Stability with ΔOST', fontsize=10, color=C.TITLE, y=0.98)
    fig.legend(handles=[Line2D([0],[0],color=C.SEC6,ls='--',lw=1,label='Signal (20m)'),
                        Line2D([0],[0],color=C.SEC4,ls='--',lw=1.5,label=al)],
               loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize=7, frameon=False, ncol=2)
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close()


def _icq_ylim(vals, bm=0.1, tm=4.0):
    v = vals[np.isfinite(vals)]
    if len(v) == 0: return None
    mn, mx = v.min(), v.max()
    r = (mx-mn) or abs(mx)*0.5 or 1
    return (mn - r*bm, mx + r*tm)

def _safety_ylim(vals, bm=1.5, tm=0.15):
    v = vals[np.isfinite(vals)]
    if len(v) == 0: return None
    mn, mx = v.min(), v.max()
    r = (mx-mn) or abs(mx)*0.5 or 1
    return (mn - r*bm, mx + r*tm)


def plot_safety_vs_icq(ts_dir, scenario_base, out_path):
    """C1-6: Safety vs ICQ (3×3 grid, CAD only)"""
    data = _load_scenario_data(ts_dir, scenario_base)
    if data is None: return
    df_wc, dist_wc = data['df_wc'], data['dist_wc']
    adist_wc, al = data['action_dist_wc'], data['action_label']

    icq_cfgs = [(c, l, b) for c, l, b in [
        ('delay', 'Delay (ms)', 'lower'),
        ('prr', 'PRR (%)', 'higher'),
        ('packet_size', 'Pkt Size (B)', 'lower')] if c in df_wc.columns]
    safety_cfgs = [('ttc', 'TTC (s)', 'higher', 20), ('drac', 'DRAC (m/s²)', 'lower', None), ('d_rel', 'Headway (m)', 'higher', None)]
    if not icq_cfgs: return

    fig, axes = plt.subplots(3, len(icq_cfgs), figsize=(C.FIG_WIDTH_FULL, 7))
    if len(icq_cfgs) == 1: axes = axes.reshape(-1, 1)

    for ri, (sc, sl, sb, clip_max) in enumerate(safety_cfgs):
        sv = df_wc[sc].replace([np.inf,-np.inf], np.nan).values
        if clip_max: sv = np.clip(sv, None, clip_max)
        sm = np.nanmean(sv)
        sy = _safety_ylim(sv)
        sg = sv > sm if sb == 'higher' else sv < sm
        for ci, (ic, il, ib) in enumerate(icq_cfgs):
            ax = axes[ri, ci]; ax.grid(False)
            iv = df_wc[ic].values
            im = np.nanmean(iv)
            iy = _icq_ylim(iv)
            ig = iv > im if ib == 'higher' else iv < im
            bg = sg & ig
            for i in range(len(bg)-1):
                if bg[i]: ax.axvspan(dist_wc[i], dist_wc[i+1], color=C.WC_LIGHT, alpha=0.3, lw=0)
            ax.plot(dist_wc, sv, color=C.WC, lw=0.8, alpha=0.8)
            ax.axhline(y=sm, color=C.WC, ls='--', lw=0.8, alpha=0.5)
            if sy: ax.set_ylim(sy)
            ax.axvline(x=20, color=C.SEC6, ls='--', lw=1, alpha=0.6)
            if pd.notna(adist_wc): ax.axvline(x=adist_wc, color=C.SEC4, ls='--', lw=1.5, alpha=0.7)
            ax2 = ax.twinx()
            ax2.plot(dist_wc, iv, color=C.SEC5, lw=0.8, alpha=0.7)
            ax2.axhline(y=im, color=C.SEC5, ls='--', lw=0.8, alpha=0.5)
            ax2.set_ylabel(il, fontsize=7, color=C.SEC5)
            ax2.tick_params(axis='y', labelcolor=C.SEC5, labelsize=6)
            if iy: ax2.set_ylim(iy)
            pi = ri * len(icq_cfgs) + ci
            ax.set_title(f'({chr(97+pi)}) {sl.split()[0]} vs {ic}', fontsize=8, color=C.TITLE)
            if ci == 0: ax.set_ylabel(sl, fontsize=7, color=C.WC)
            ax.tick_params(axis='y', labelcolor=C.WC, labelsize=6)
            ax.tick_params(axis='x', labelsize=6)
            if ri == 2: ax.set_xlabel('Distance (m)', fontsize=7)

    plt.tight_layout(rect=[0,0,1,0.94])
    fig.suptitle(f'{scenario_base} - Safety vs ICQ (CAD)', fontsize=10, color=C.TITLE, y=0.98)
    fig.legend(handles=[Line2D([0],[0],color=C.WC,lw=1,label='Safety'),
                        Line2D([0],[0],color=C.SEC5,lw=1,label='ICQ'),
                        Patch(facecolor=C.WC_LIGHT,alpha=0.3,label='Both>mean'),
                        Line2D([0],[0],color=C.SEC6,ls='--',lw=1,label='Signal'),
                        Line2D([0],[0],color=C.SEC4,ls='--',lw=1,label=al)],
               loc='upper center', fontsize=6, frameon=False, ncol=5, bbox_to_anchor=(0.5,0.97))
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_stability_vs_icq(ts_dir, scenario_base, out_path):
    """C1-7: Stability vs ICQ (2×3 grid, CAD only)"""
    data = _load_scenario_data(ts_dir, scenario_base)
    if data is None: return
    df_wc, dist_wc = data['df_wc'], data['dist_wc']
    adist_wc, al = data['action_dist_wc'], data['action_label']
    is_etra = 'etra' in scenario_base.lower()
    prefix, subject = ('target','Target') if is_etra else ('ego','Ego')

    icq_cfgs = [(c,l,b) for c,l,b in [
        ('delay','Delay (ms)','lower'),('prr','PRR (%)','higher'),('packet_size','Pkt Size (B)','lower')]
        if c in df_wc.columns]
    stab_cfgs = [(f'{prefix}_jerk_total', f'{subject} Jerk (m/s³)'),
                 (f'{prefix}_yaw_rate',   f'{subject} Yaw (rad/s)')]
    if not icq_cfgs: return

    fig, axes = plt.subplots(2, len(icq_cfgs), figsize=(C.FIG_WIDTH_FULL, 4.5))
    if len(icq_cfgs) == 1: axes = axes.reshape(-1, 1)

    for ri, (sc, sl) in enumerate(stab_cfgs):
        sv = df_wc[sc].values if sc in df_wc.columns else np.array([])
        if len(sv) == 0: continue
        sy = _safety_ylim(sv)
        for ci, (ic, il, ib) in enumerate(icq_cfgs):
            ax = axes[ri, ci]; ax.grid(False)
            iv = df_wc[ic].values
            im = np.nanmean(iv)
            iy = _icq_ylim(iv)
            ig = iv > im if ib == 'higher' else iv < im
            for i in range(len(ig)-1):
                if ig[i]: ax.axvspan(dist_wc[i], dist_wc[i+1], color=C.WC_LIGHT, alpha=0.3, lw=0)
            ax.plot(dist_wc, sv, color=C.WC, lw=0.8, alpha=0.8)
            ax.axhline(y=0, color='black', ls='-', lw=0.5, alpha=0.3)
            if sy: ax.set_ylim(sy)
            ax.axvline(x=20, color=C.SEC6, ls='--', lw=1, alpha=0.6)
            if pd.notna(adist_wc): ax.axvline(x=adist_wc, color=C.SEC4, ls='--', lw=1.5, alpha=0.7)
            ax2 = ax.twinx()
            ax2.plot(dist_wc, iv, color=C.SEC5, lw=0.8, alpha=0.7)
            ax2.axhline(y=im, color=C.SEC5, ls='--', lw=0.8, alpha=0.5)
            ax2.set_ylabel(il, fontsize=7, color=C.SEC5)
            ax2.tick_params(axis='y', labelcolor=C.SEC5, labelsize=6)
            if iy: ax2.set_ylim(iy)
            pi = ri * len(icq_cfgs) + ci
            metric_short = 'Jerk' if 'jerk' in sc else 'Yaw'
            ax.set_title(f'({chr(97+pi)}) {metric_short} vs {ic}', fontsize=8, color=C.TITLE)
            if ci == 0: ax.set_ylabel(sl, fontsize=7, color=C.WC)
            ax.tick_params(axis='y', labelcolor=C.WC, labelsize=6)
            ax.tick_params(axis='x', labelsize=6)
            if ri == 1: ax.set_xlabel('Distance (m)', fontsize=7)

    plt.tight_layout(rect=[0,0,1,0.92])
    fig.suptitle(f'{scenario_base} - {subject} Stability vs ICQ (CAD)', fontsize=10, color=C.TITLE, y=0.98)
    fig.legend(handles=[Line2D([0],[0],color=C.WC,lw=1,label='Stability'),
                        Line2D([0],[0],color=C.SEC5,lw=1,label='ICQ'),
                        Patch(facecolor=C.WC_LIGHT,alpha=0.3,label='ICQ>mean'),
                        Line2D([0],[0],color=C.SEC6,ls='--',lw=1,label='Signal'),
                        Line2D([0],[0],color=C.SEC4,ls='--',lw=1,label=al)],
               loc='upper center', fontsize=6, frameon=False, ncol=5, bbox_to_anchor=(0.5,0.96))
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.close()


def run_all(ts_dir, scenario_bases, out_dir):
    """Generate all C1 graphs for given scenario base names."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for base in scenario_bases:
        plot_safety_dynamics(ts_dir, base, out_dir / f'axC1_1_safety_{base}.svg')
        plot_stability_dynamics(ts_dir, base, out_dir / f'axC1_2_stability_{base}.svg')
        plot_safety_vs_entropy(ts_dir, base, out_dir / f'axC1_4_safety_entropy_{base}.svg')
        plot_stability_vs_entropy(ts_dir, base, out_dir / f'axC1_5_stability_entropy_{base}.svg')
        plot_safety_vs_icq(ts_dir, base, out_dir / f'axC1_6_safety_icq_{base}.svg')
        plot_stability_vs_icq(ts_dir, base, out_dir / f'axC1_7_stability_icq_{base}.svg')
