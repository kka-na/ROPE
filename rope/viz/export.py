#!/usr/bin/env python3
"""Pipeline 결과물 → PDF 저장"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

from rope.viz import colors


def _savefig(path):
    plt.savefig(path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"Saved: {path}")


def save_icq_violin(df, out_path, location=None):
    df_wc = df[df['mode'] == 'WC'].copy()
    df_plot = df_wc[df_wc['location'] == location] if location else df_wc
    groups = sorted(df_plot['type'].unique())
    if not groups: return
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.2))
    for ax, (metric, ylabel, panel) in zip(axes, [
        ('latency_ms',    'Latency (ms)',    '(a) Latency'),
        ('prr_mean',      'PRR (%)',         '(b) PRR'),
        ('pkt_size_mean', 'Packet Size (B)', '(c) Packet Size'),
    ]):
        for i, g in enumerate(groups, 1):
            d = df_plot[df_plot['type'] == g][metric].dropna().values
            if not len(d): continue
            vp = ax.violinplot([d], positions=[i], showmeans=True, showmedians=False, widths=0.7)
            vp['bodies'][0].set_facecolor(colors.WOC_TINT60)
            vp['bodies'][0].set_edgecolor(colors.WOC)
            vp['bodies'][0].set_alpha(0.8)
            mc = colors.WC if g == 'CLM' else colors.WC_LIGHT
            vp['cmeans'].set_color(mc); vp['cmeans'].set_linewidth(1.5)
            for part in ['cbars', 'cmins', 'cmaxes']:
                vp[part].set_color(colors.WOC); vp[part].set_linewidth(0.8)
            ax.annotate(f'μ={d.mean():.1f}\nn={len(d)}', xy=(i, d.mean()),
                        xytext=(3, 3), textcoords='offset points', fontsize=7, color=mc, bbox=colors.LABEL_BBOX)
        ax.set_xticks(range(1, len(groups)+1)); ax.set_xticklabels(groups)
        ax.set_ylabel(ylabel, fontsize=8); ax.set_title(panel, fontsize=9, color=colors.TITLE)
        ax.tick_params(labelsize=7)
    loc_label = location or 'All'
    fig.suptitle(f'[R-1] ICQ Distribution - {loc_label}', fontsize=10, color=colors.TITLE)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(out_path)


def save_data_quality(df, out_path):
    df = df[df['mode'] == 'WC'].copy()
    locations = [l for l in ['KIAPI', 'Yeongjong', 'CARLA'] if l in df['location'].values]
    if not locations: return
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    bw = 0.35; types = sorted(df['type'].unique()); x = np.arange(len(types))
    for ax, (metric, ylabel, panel) in zip(axes, [
        ('n_samples',        'Samples',   '(a) Sample Count'),
        ('sampling_rate_hz', 'Rate (Hz)', '(b) Sampling Rate'),
        ('valid_rate',      'Valid (%)', '(c) Valid Ratio'),
    ]):
        for i, loc in enumerate(locations):
            loc_df = df[df['location'] == loc]
            means = [loc_df[loc_df['type'] == t][metric].mean() for t in types]
            stds  = [loc_df[loc_df['type'] == t][metric].std()  for t in types]
            offset = (i - len(locations)/2 + 0.5) * bw
            kw = dict(capsize=3, error_kw={'linewidth': 0.8, 'color': colors.WOC})
            if i == 0:
                ax.bar(x+offset, means, bw, yerr=stds, label=loc, color=colors.WOC_TINT60, edgecolor=colors.WOC, **kw)
            else:
                ax.bar(x+offset, means, bw, yerr=stds, label=loc, facecolor='none', edgecolor=colors.WOC, hatch='//', **kw)
        ax.set_xticks(x); ax.set_xticklabels(types)
        ax.set_ylabel(ylabel, fontsize=8); ax.set_title(panel, fontsize=9, color=colors.TITLE)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    fig.suptitle('[R-1] Data Quality by Location', fontsize=10, color=colors.TITLE)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(out_path)


def save_icq_timeline(ts_dir, scenario, out_path):
    pq = Path(ts_dir) / f'{scenario}.parquet'
    if not pq.exists(): return
    df = pd.read_parquet(pq)
    if 'ego_cumulative_dist' not in df.columns: return
    dist_rel = df['ego_cumulative_dist'].values - df['ego_cumulative_dist'].values[0]
    action_dist = df['action_start_dist'].iloc[0] if 'action_start_dist' in df.columns else np.nan
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.5))
    for ax, col, ylabel in zip(axes, ['delay', 'prr'], ['Delay (ms)', 'PRR (%)']):
        if col not in df.columns: continue
        ax.plot(dist_rel, df[col], color=colors.WC, linewidth=0.8)
        ax.axvline(x=20, color=colors.SEC6, linestyle='--', linewidth=1)
        if pd.notna(action_dist):
            ax.axvline(x=action_dist, color=colors.SEC4, linestyle=':', linewidth=1.5)
        ax.text(0.02, 0.97, f'μ={df[col].dropna().mean():.1f}', transform=ax.transAxes,
                fontsize=7, va='top', bbox=colors.LABEL_BBOX)
        ax.set_ylabel(ylabel, fontsize=8); ax.tick_params(labelsize=7)
    axes[0].set_title(f'(a) {scenario}', fontsize=8, color=colors.TITLE)
    axes[1].set_xlabel('Distance (m)', fontsize=8)
    plt.tight_layout()
    _savefig(out_path)


def save_entropy_timeline(ts_dir, scenario, out_path):
    pq = Path(ts_dir) / f'{scenario}.parquet'
    if not pq.exists(): return
    df = pd.read_parquet(pq)
    if 'ego_cumulative_dist' not in df.columns: return
    dist_rel = df['ego_cumulative_dist'].values - df['ego_cumulative_dist'].values[0]
    action_dist = df['action_start_dist'].iloc[0] if 'action_start_dist' in df.columns else np.nan
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.5))

    ax = axes[0]
    if 'h_lidar' in df.columns and 'h_v2v' in df.columns:
        h_lidar, h_v2v = df['h_lidar'].values, df['h_v2v'].values
        ax.plot(dist_rel, h_lidar, color=colors.WOC, linewidth=0.8, label='H(LiDAR)')
        ax.plot(dist_rel, h_v2v,   color=colors.WC,  linewidth=0.8, alpha=0.3, label='H(V2V)')
        ax.plot(dist_rel, np.where(h_v2v < h_lidar, h_v2v, np.nan), color=colors.WC, linewidth=1.2)
    ax.axvline(x=20, color=colors.SEC6, linestyle='--', linewidth=1, alpha=0.7)
    if pd.notna(action_dist):
        ax.axvline(x=action_dist, color=colors.SEC4, linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_ylabel('H (Entropy)', fontsize=8)
    ax.set_title(f'(a) {scenario}', fontsize=8, color=colors.TITLE)
    ax.legend(fontsize=6); ax.tick_params(labelsize=7)

    ax = axes[1]
    if 'delta_h' in df.columns:
        dh = df['delta_h'].values
        ax.plot(dist_rel, dh, color=colors.SEC5, linewidth=0.8)
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        ax.fill_between(dist_rel, 0, dh, where=(dh > 0), color=colors.SEC5, alpha=0.3)
        ax.text(0.02, 0.97, f'μ={np.nanmean(dh):.3f}', transform=ax.transAxes,
                fontsize=7, va='top', bbox=colors.LABEL_BBOX)
    ax.axvline(x=20, color=colors.SEC6, linestyle='--', linewidth=1, alpha=0.7)
    ax.set_ylabel('ΔOST', fontsize=8); ax.set_xlabel('Distance (m)', fontsize=8)
    ax.set_title('(b) ΔOST = OST(CAD) − OST(SAD)', fontsize=8, color=colors.TITLE)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    _savefig(out_path)


def save_entropy_boxplot(table_path, out_path):
    df = pd.read_csv(table_path)
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    def _vp(ax, data, pos, fc, ec):
        if not len(data): return
        vp = ax.violinplot([data], positions=[pos], showmeans=True, showextrema=False)
        vp['bodies'][0].set_facecolor(fc); vp['bodies'][0].set_edgecolor(ec); vp['bodies'][0].set_alpha(0.7)
        vp['cmeans'].set_color(ec); vp['cmeans'].set_linewidth(1.5)
        ax.annotate(f'μ={data.mean():.2f}\nn={len(data)}', xy=(pos, data.mean()),
                    xytext=(5, 0), textcoords='offset points', fontsize=7, va='center', bbox=colors.LABEL_BBOX)

    for ax, stype, title in zip(axes[:2], ['CLM', 'ETrA'], ['(a) CLM', '(b) ETrA']):
        sub = df[df['type'] == stype]
        _vp(ax, sub['h_v2v_mean'].dropna().values,   1, colors.WC_LIGHT_TINT60, colors.WC)
        _vp(ax, sub['h_lidar_mean'].dropna().values, 2, colors.WOC_TINT60,      colors.WOC)
        ax.set_xticks([1, 2]); ax.set_xticklabels(['CAD', 'SAD'])
        ax.set_ylabel('H (Entropy)', fontsize=8); ax.set_title(title, fontsize=9, color=colors.TITLE)
        ax.tick_params(labelsize=7)

    ax = axes[2]
    _vp(ax, df[df['type'] == 'CLM']['delta_h_mean'].dropna().values,  1, '#b8a6d9', colors.SEC5)
    _vp(ax, df[df['type'] == 'ETrA']['delta_h_mean'].dropna().values, 2, '#b8a6d9', colors.SEC5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks([1, 2]); ax.set_xticklabels(['CLM', 'ETrA'])
    ax.set_ylabel('ΔOST', fontsize=8); ax.set_title('(c) ΔOST', fontsize=9, color=colors.TITLE)
    ax.tick_params(labelsize=7)

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor=colors.WC_LIGHT_TINT60, edgecolor=colors.WC, alpha=0.7, label='CAD (V2X)'),
        Patch(facecolor=colors.WOC_TINT60, edgecolor=colors.WOC, alpha=0.7, label='SAD (LiDAR)'),
    ], loc='upper right', ncol=2, fontsize=7, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _savefig(out_path)


def save_paired_scatter(pairs_path, out_path):
    df = pd.read_csv(pairs_path)
    if df.empty: return
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    df['color'] = np.where(df['delta_H'] > 0, colors.WC, colors.WOC)
    for stype, marker in [('CLM', 'o'), ('ETrA', 's')]:
        sub = df[df['type'] == stype] if 'type' in df.columns else df
        if not sub.empty:
            ax.scatter(sub['delta_H_WOC'], sub['delta_H_WC'], c=sub['color'],
                       marker=marker, s=40, alpha=0.7, label=stype, edgecolors='white', linewidths=0.5)
    all_vals = pd.concat([df['delta_H_WOC'], df['delta_H_WC']]).dropna()
    if all_vals.empty: plt.close(); return
    lims = [all_vals.min() - 0.1, all_vals.max() + 0.1]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('H_SAD (LiDAR)', fontsize=8); ax.set_ylabel('H_CAD (V2X)', fontsize=8)
    ax.set_title('[B-2] Observation Quality Difference: CAD vs SAD', fontsize=9, color=colors.TITLE)
    ax.legend(fontsize=7)
    ax.text(0.02, 0.97, f'CAD better: {(df["delta_H"]>0).sum()}/{len(df)}',
            transform=ax.transAxes, fontsize=7, va='top', bbox=colors.LABEL_BBOX)
    plt.tight_layout()
    _savefig(out_path)


def save_delta_h_cdf(ts_dir, out_path):
    all_dh = []
    for pq in Path(ts_dir).glob('*.parquet'):
        try:
            df = pd.read_parquet(pq)
            if 'delta_h' in df.columns:
                all_dh.extend(df['delta_h'].dropna().tolist())
        except Exception:
            continue
    if not all_dh: return
    data = np.array(all_dh)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))
    axes[0].hist(data, bins=50, color=colors.WC_LIGHT_TINT60, edgecolor=colors.WC, alpha=0.7)
    axes[0].axvline(data.mean(), color=colors.SEC6, linestyle='--', linewidth=1.5, label=f'Mean={data.mean():.3f}')
    axes[0].axvline(0, color='black', linewidth=0.8, alpha=0.5)
    axes[0].set_xlabel('ΔOST', fontsize=8); axes[0].set_ylabel('Count', fontsize=8)
    axes[0].set_title('(a) Histogram', fontsize=9, color=colors.TITLE)
    axes[0].legend(fontsize=7); axes[0].tick_params(labelsize=7)
    sorted_d = np.sort(data)
    axes[1].plot(sorted_d, np.arange(1, len(sorted_d)+1)/len(sorted_d), color=colors.WC, linewidth=1)
    axes[1].axvline(0, color='black', linewidth=0.8, alpha=0.5)
    axes[1].text(0.02, 0.97, f'ΔOST>0: {100*(data>0).sum()/len(data):.1f}%',
                 transform=axes[1].transAxes, fontsize=7, va='top', bbox=colors.LABEL_BBOX)
    axes[1].set_xlabel('ΔOST', fontsize=8); axes[1].set_ylabel('CDF', fontsize=8)
    axes[1].set_title('(b) CDF', fontsize=9, color=colors.TITLE); axes[1].tick_params(labelsize=7)
    fig.suptitle('[B-A1] ΔOST Distribution', fontsize=10, color=colors.TITLE)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(out_path)


def save_qos_vs_delta_ost(o_pairs_path, r_table_path, out_path):
    df_o = pd.read_csv(o_pairs_path)
    df_r = pd.read_csv(r_table_path)
    # o_pairs has no 'mode' column; strip _wc_ suffix to get base scenario key
    df_o = df_o.copy()
    df_o['_key'] = df_o['scenario'].str.replace('_wc_', '_', regex=False)
    df_r = df_r[df_r['mode'] == 'WC'].copy()
    df_r['_key'] = df_r['scenario']
    merged = pd.merge(df_o[['_key', 'delta_ost']], df_r[['_key', 'latency_ms', 'prr_mean', 'pkt_size_mean']], on='_key', how='inner')
    if merged.empty: return
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    for ax, xcol, xlabel in [
        (axes[0], 'latency_ms',    'Latency (ms)'),
        (axes[1], 'prr_mean',      'PRR (%)'),
        (axes[2], 'pkt_size_mean', 'PktSize (B)'),
    ]:
        sub = merged[[xcol, 'delta_ost']].dropna()
        ax.scatter(sub[xcol], sub['delta_ost'], c=colors.WC, s=30, alpha=0.6, edgecolors='white')
        if len(sub) > 2:
            rho, p = spearmanr(sub[xcol], sub['delta_ost'])
            ax.text(0.02, 0.97, f'ρ={rho:.3f}\np={p:.4f}', transform=ax.transAxes,
                    fontsize=7, va='top', bbox=colors.LABEL_BBOX)
        ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel('ΔOST', fontsize=8)
        ax.set_title(f'{xlabel} vs ΔOST', fontsize=9, color=colors.TITLE); ax.tick_params(labelsize=7)
    fig.suptitle('[O-A2] ICQ vs ΔOST', fontsize=10, color=colors.TITLE)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig(out_path)


def run_export(output_dir: str):
    colors.apply_style()
    out     = Path(output_dir)
    fig_dir = out / 'figures'
    fig_dir.mkdir(exist_ok=True)
    ts_dir  = out / 'timeseries'
    r_table = out / 'r_axis_table.csv'
    o_table = out / 'o_axis_table.csv'
    o_pairs = out / 'o_axis_pairs.csv'

    if r_table.exists():
        df_r = pd.read_csv(r_table)
        for loc in ['KIAPI', 'Yeongjong', 'CARLA']:
            if loc in df_r['location'].values:
                save_icq_violin(df_r, fig_dir / f'axR_1_icq_{loc.lower()}.pdf', location=loc)
        save_data_quality(df_r, fig_dir / 'axR_1_data_quality.pdf')
        for s in df_r[df_r['mode'] == 'WC']['scenario'].tolist()[:4]:
            save_icq_timeline(ts_dir, s, fig_dir / f'axR_2_timeline_{s}.pdf')

    if o_pairs.exists():
        from rope.viz.o_plots import plot_ost_bar, plot_ost_scatter, plot_delta_ost_histogram
        df_pairs = pd.read_csv(o_pairs)
        if not df_pairs.empty:
            for fn, fig_fn in [(plot_ost_bar, 'axO_1_ost_bar.pdf'),
                               (plot_ost_scatter, 'axO_2_ost_scatter.pdf'),
                               (plot_delta_ost_histogram, 'axO_3_delta_ost_hist.pdf')]:
                fig = fn(df_pairs)
                if fig:
                    fig.savefig(fig_dir / fig_fn, format='pdf', bbox_inches='tight', pad_inches=0.02)
                    plt.close(fig)
                    print(f"Saved: {fig_dir / fig_fn}")
            if r_table.exists():
                save_qos_vs_delta_ost(o_pairs, r_table, fig_dir / 'axO_A2_qos_delta_ost.pdf')

    print(f"\n✅ Figures saved → {fig_dir}")
