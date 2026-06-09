#!/usr/bin/env python3
"""C-2 Scenario Performance graphs: paired scatter, boxplot, heatmap, radar"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy import stats as scipy_stats

from rope.viz import colors as C

TYPE_MARKERS = {'CLM': 'o', 'ETrA': 's'}
ICQ_COLS   = ['delay_mean', 'prr_mean', 'pkt_size_mean']
ICQ_LABELS = {'delay_mean': 'Delay (ms)', 'prr_mean': 'PRR (%)', 'pkt_size_mean': 'Packet Size (B)'}

HEATMAP_METRICS = [
    ('ttc_violation_2s',  'TTC Viol. (2s)', 'lower'),
    ('ttc_violation_3s',  'TTC Viol. (3s)', 'lower'),
    ('ttc_violation_2s', 'TTC Viol.',   'lower'),
    ('drac_critical_ratio',         'DRAC Crit.Ratio',       'lower'),
    ('d_min',             'Min Distance',    'higher'),
    ('ego_jerk_rms',      'Jerk RMS',        'lower'),
    ('ego_jerk_peak',     'Jerk Peak',       'lower'),
    ('ego_yaw_rms',       'Yaw RMS',         'lower'),
    ('ego_yaw_zcr',       'Yaw ZCR',         'lower'),
    ('tct',               'TCT',             'lower'),
    ('v_mean',            'Mean Speed',      'higher'),
]


def _imp_pct(wc, woc, better='lower'):
    if pd.isna(wc) or pd.isna(woc): return np.nan
    denom = max(abs(wc), abs(woc))
    if denom < 1e-9: return 0
    return ((woc - wc) / denom * 100) if better == 'lower' else ((wc - woc) / denom * 100)


def _shared_legend(fig, type_handles=True, color_handles=True, best_handle=True):
    handles = []
    if type_handles:
        handles += [plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',ms=8,label='CLM'),
                    plt.Line2D([0],[0],marker='s',color='w',markerfacecolor='gray',ms=8,label='ETrA')]
    if color_handles:
        handles += [Patch(facecolor=C.WC,label='CAD better'),
                    Patch(facecolor='gray',label='Tie'),
                    Patch(facecolor=C.WOC,label='SAD better')]
    if best_handle:
        handles += [plt.Line2D([0],[0],marker='*',color='w',markerfacecolor=C.SEC2,ms=10,label='Best')]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), fontsize=7, bbox_to_anchor=(0.5,1.0))


def _short_name(s):
    s2 = s.replace('kiapi','K').replace('yeongjong','Y')
    parts = s2.split('_')
    return '_'.join(parts[:2]) if len(parts) >= 2 else s2


def plot_paired_scatter(ax, df, x_col, y_col, title, xlabel, ylabel, better='lower'):
    valid = df[x_col].notna() & df[y_col].notna()
    dv = df[valid].copy()
    if len(dv) == 0:
        ax.text(0.5,0.5,'No data',ha='center',va='center',transform=ax.transAxes)
        ax.set_title(title, fontsize=9, fontweight='bold', color=C.TITLE); return
    dv['_wc_better'] = dv[y_col] < dv[x_col] if better=='lower' else dv[y_col] > dv[x_col]
    dv['_tie'] = dv[y_col] == dv[x_col]
    dv['_imp'] = (dv[x_col]-dv[y_col]) if better=='lower' else (dv[y_col]-dv[x_col])
    dv['_c'] = np.where(dv['_wc_better'], C.WC, np.where(dv['_tie'], 'gray', C.WOC))
    for t, mk in TYPE_MARKERS.items():
        s = dv[dv['type']==t] if 'type' in dv.columns else dv
        if len(s): ax.scatter(s[x_col], s[y_col], c=s['_c'], marker=mk, s=50, alpha=0.7,
                              edgecolors='white', linewidths=0.5, label=t)
    best = dv.loc[dv['_imp'].idxmax()]
    ax.scatter([best[x_col]], [best[y_col]], c=C.SEC2, marker='*', s=150, edgecolors='white', lw=0.8, zorder=10)
    ax.annotate(_short_name(best['scenario']) if 'scenario' in best else '',
                (best[x_col], best[y_col]), xytext=(5,5), textcoords='offset points',
                fontsize=6, color=C.SEC2, fontweight='bold')
    all_v = pd.concat([dv[x_col], dv[y_col]])
    mg = (all_v.max()-all_v.min())*0.1 or 0.1
    lims = [all_v.min()-mg, all_v.max()+mg]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold', color=C.TITLE)
    ax.tick_params(labelsize=7)
    nw = dv['_wc_better'].sum(); nt = dv['_tie'].sum(); nb = len(dv)-nw-nt
    ax.text(0.02,0.98,f'WC/Tie/WOC: {nw}/{nt}/{nb}',transform=ax.transAxes,
            fontsize=7,va='top',bbox=C.LABEL_BBOX)


def plot_safety_scatter(pairs_df, out_path):
    """C2-1: Safety paired scatter (1×3)"""
    C.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))
    plot_paired_scatter(axes[0], pairs_df, 'ttc_violation_2s_WOC','ttc_violation_2s_WC','(a) TTC Violation Rate','SAD (%)','CAD (%)')
    plot_paired_scatter(axes[1], pairs_df, 'ttc_violation_2s_WOC','ttc_violation_2s_WC','(b) TTC Violation Rate','SAD (%)','CAD (%)')
    plot_paired_scatter(axes[2], pairs_df, 'drac_critical_ratio_WOC','drac_critical_ratio_WC','(c) DRAC Crit.Ratio','SAD (m/s²)','CAD (m/s²)')
    _shared_legend(fig)
    fig.suptitle('Safety Metrics: SAD vs CAD', fontsize=10, fontweight='bold', color=C.TITLE, y=1.08)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02)
    plt.close()
    return fig


def plot_stability_scatter(pairs_df, out_path):
    """C2-2: Stability paired scatter (2×2)"""
    C.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 5))
    plot_paired_scatter(axes[0,0], pairs_df, 'ego_jerk_rms_WOC','ego_jerk_rms_WC','(a) Jerk RMS','SAD (m/s³)','CAD (m/s³)')
    plot_paired_scatter(axes[0,1], pairs_df, 'ego_jerk_peak_WOC','ego_jerk_peak_WC','(b) Jerk Peak','SAD (m/s³)','CAD (m/s³)')
    plot_paired_scatter(axes[1,0], pairs_df, 'ego_yaw_rms_WOC','ego_yaw_rms_WC','(c) Yaw RMS','SAD (rad/s)','CAD (rad/s)')
    plot_paired_scatter(axes[1,1], pairs_df, 'ego_yaw_zcr_WOC','ego_yaw_zcr_WC','(d) Yaw ZCR','SAD','CAD')
    _shared_legend(fig)
    fig.suptitle('Stability Metrics: SAD vs CAD', fontsize=10, fontweight='bold', color=C.TITLE, y=1.05)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02)
    plt.close()
    return fig


def plot_efficiency_scatter(pairs_df, out_path):
    """C2-3: Efficiency paired scatter (2×3: CLM/ETrA rows)"""
    C.apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    clm = pairs_df[pairs_df['type']=='CLM'] if 'type' in pairs_df.columns else pairs_df
    etr = pairs_df[pairs_df['type']=='ETrA'] if 'type' in pairs_df.columns else pairs_df
    plot_paired_scatter(axes[0,0], clm, 'tct_norm_WOC','tct_norm_WC','(a) TCT_norm','SAD','CAD')
    plot_paired_scatter(axes[0,1], clm, 'v_mean_norm_WOC','v_mean_norm_WC','(b) v_mean_norm','SAD','CAD', better='higher')
    plot_paired_scatter(axes[0,2], clm, 't_merge_WOC','t_merge_WC','(c) Merge Time','SAD (s)','CAD (s)')
    axes[0,0].set_ylabel('CLM\n\nCAD', fontsize=8)
    plot_paired_scatter(axes[1,0], etr, 'tct_norm_WOC','tct_norm_WC','(d) TCT_norm','SAD','CAD')
    plot_paired_scatter(axes[1,1], etr, 'v_mean_norm_WOC','v_mean_norm_WC','(e) v_mean_norm','SAD','CAD', better='higher')
    plot_paired_scatter(axes[1,2], etr, 'avoid_dist_tgt_WOC','avoid_dist_tgt_WC','(f) Avoid Dist Tgt','SAD (m)','CAD (m)', better='higher')
    axes[1,0].set_ylabel('ETrA\n\nCAD (s)', fontsize=8)
    _shared_legend(fig, type_handles=False)
    fig.suptitle('Efficiency Metrics: SAD vs CAD', fontsize=10, fontweight='bold', color=C.TITLE, y=1.05)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02)
    plt.close()
    return fig


def _dh_scatter(ax, df, metric_col, title, ylabel, better='lower'):
    wc_col, woc_col = f'{metric_col}_WC', f'{metric_col}_WOC'
    if 'delta_H' not in df.columns or wc_col not in df.columns or woc_col not in df.columns:
        ax.text(0.5,0.5,'No data',ha='center',va='center',transform=ax.transAxes)
        ax.set_title(title,fontsize=9,fontweight='bold',color=C.TITLE); return
    df = df.copy()
    df['_imp'] = df.apply(lambda r: _imp_pct(r[wc_col], r[woc_col], better), axis=1)
    ok = df['delta_H'].notna() & df['_imp'].notna()
    dv = df[ok]
    if len(dv) == 0:
        ax.text(0.5,0.5,'No data',ha='center',va='center',transform=ax.transAxes)
        ax.set_title(title,fontsize=9,fontweight='bold',color=C.TITLE); return
    x, y = dv['delta_H'], dv['_imp']
    pt_c = [C.WC if v>0 else C.WOC for v in y]
    for t, mk in [('CLM','o'),('ETrA','s')]:
        mask = dv['type']==t if 'type' in dv.columns else pd.Series([True]*len(dv))
        if mask.sum(): ax.scatter(x[mask], y[mask], c=[pt_c[i] for i,m in enumerate(mask) if m],
                                  marker=mk, s=50, alpha=0.7, edgecolors='white', lw=0.5)
    slope, intercept, *_ = scipy_stats.linregress(x, y)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, slope*xl+intercept, color=C.SEC1, ls='--', lw=1.5, alpha=0.8)
    rho, pv = scipy_stats.spearmanr(x, y)
    sig = '**' if pv<0.01 else ('*' if pv<0.05 else ('†' if pv<0.1 else ''))
    ax.text(0.05,0.1,f'ρ={rho:.2f}{sig}',transform=ax.transAxes,fontsize=8,fontweight='bold',va='top',bbox=C.LABEL_BBOX)
    ax.axhline(y=0,color='gray',ls='-',lw=0.5,alpha=0.5); ax.axvline(x=0,color='gray',ls='-',lw=0.5,alpha=0.5)
    ax.set_xlabel('ΔOST',fontsize=8); ax.set_ylabel(ylabel,fontsize=8)
    ax.set_title(title,fontsize=9,fontweight='bold',color=C.TITLE); ax.tick_params(labelsize=7)


def _merge_dh(pairs_df, axb_df):
    if axb_df is None or 'delta_H' not in axb_df.columns: return pairs_df
    return pd.merge(pairs_df, axb_df[['scenario','delta_H']], on='scenario', how='inner')


def plot_safety_deltaH_scatter(pairs_df, axb_df, out_path):
    df = _merge_dh(pairs_df, axb_df)
    C.apply_style(); fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))
    _dh_scatter(axes[0], df, 'ttc_violation_2s', '(a) TTC Violation Rate', 'Improvement (%)')
    _dh_scatter(axes[1], df, 'ttc_violation_2s',  '(b) TTC Violation Rate', 'Improvement (%)')
    _dh_scatter(axes[2], df, 'drac_critical_ratio',           '(c) DRAC Crit.Ratio', 'Improvement (%)')
    _shared_legend(fig, color_handles=False)
    fig.suptitle('Safety: ΔOST vs Improvement', fontsize=10, fontweight='bold', color=C.TITLE, y=1.08)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02); plt.close()


def plot_stability_deltaH_scatter(pairs_df, axb_df, out_path):
    df = _merge_dh(pairs_df, axb_df)
    C.apply_style(); fig, axes = plt.subplots(2, 2, figsize=(5.5, 5))
    _dh_scatter(axes[0,0], df, 'ego_jerk_rms',  '(a) Jerk RMS',  'Improvement (%)')
    _dh_scatter(axes[0,1], df, 'ego_jerk_peak', '(b) Jerk Peak', 'Improvement (%)')
    _dh_scatter(axes[1,0], df, 'ego_yaw_rms',   '(c) Yaw RMS',   'Improvement (%)')
    _dh_scatter(axes[1,1], df, 'ego_yaw_zcr',   '(d) Yaw ZCR',   'Improvement (%)')
    _shared_legend(fig, color_handles=False)
    fig.suptitle('Stability: ΔOST vs Improvement', fontsize=10, fontweight='bold', color=C.TITLE, y=1.05)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02); plt.close()


def plot_efficiency_deltaH_scatter(pairs_df, axb_df, out_path):
    df = _merge_dh(pairs_df, axb_df)
    C.apply_style(); fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    clm = df[df['type']=='CLM'] if 'type' in df.columns else df
    etr = df[df['type']=='ETrA'] if 'type' in df.columns else df
    _dh_scatter(axes[0,0], clm, 'tct_norm',        '(a) TCT_norm',    'Improvement (%)')
    _dh_scatter(axes[0,1], clm, 'v_mean_norm',     '(b) v_mean_norm', 'Improvement (%)', better='higher')
    _dh_scatter(axes[0,2], clm, 't_merge',         '(c) Merge Time',  'Improvement (%)')
    axes[0,0].set_ylabel('CLM\n\nImprovement (%)', fontsize=8)
    _dh_scatter(axes[1,0], etr, 'tct_norm',        '(d) TCT_norm',    'Improvement (%)')
    _dh_scatter(axes[1,1], etr, 'v_mean_norm',     '(e) v_mean_norm', 'Improvement (%)', better='higher')
    _dh_scatter(axes[1,2], etr, 'avoid_dist_tgt',  '(f) Avoid Dist Tgt', 'Improvement (%)', better='higher')
    axes[1,0].set_ylabel('ETrA\n\nImprovement (%)', fontsize=8)
    _shared_legend(fig, type_handles=False, color_handles=False)
    fig.suptitle('Efficiency: ΔOST vs Improvement', fontsize=10, fontweight='bold', color=C.TITLE, y=1.05)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02); plt.close()


def _icq_scatter(ax, df_metrics, metric_col, icq_col, title, ylabel, better='lower'):
    dv = df_metrics[df_metrics['mode']=='WC'] if 'mode' in df_metrics.columns else df_metrics.copy()
    if metric_col not in dv.columns or icq_col not in dv.columns:
        ax.text(0.5,0.5,'No data',ha='center',va='center',transform=ax.transAxes)
        ax.set_title(title,fontsize=9,fontweight='bold',color=C.TITLE); return
    ok = dv[icq_col].notna() & dv[metric_col].notna()
    dv = dv[ok]
    if len(dv) < 5:
        ax.text(0.5,0.5,f'N={len(dv)} (insufficient)',ha='center',va='center',transform=ax.transAxes)
        ax.set_title(title,fontsize=9,fontweight='bold',color=C.TITLE); return
    x, y = dv[icq_col], dv[metric_col]
    rho, pv = scipy_stats.spearmanr(x, y)
    pc = C.WC if abs(rho) >= 0.3 else C.WOC
    for t, mk in [('CLM','o'),('ETrA','s')]:
        mask = dv['type']==t if 'type' in dv.columns else pd.Series([True]*len(dv))
        if mask.sum(): ax.scatter(x[mask],y[mask],c=pc,marker=mk,s=50,alpha=0.7,edgecolors='white',lw=0.5,label=t)
    slope, intercept, *_ = scipy_stats.linregress(x, y)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, slope*xl+intercept, color=pc, ls='--', lw=1.5, alpha=0.8)
    sig = '**' if pv<0.01 else ('*' if pv<0.05 else ('†' if pv<0.1 else ''))
    ax.text(0.05,0.95,f'ρ={rho:.2f}{sig}',transform=ax.transAxes,fontsize=8,fontweight='bold',va='top',bbox=C.LABEL_BBOX)
    ax.set_xlabel(ICQ_LABELS.get(icq_col,icq_col),fontsize=8); ax.set_ylabel(ylabel,fontsize=8)
    ax.set_title(title,fontsize=9,fontweight='bold',color=C.TITLE); ax.tick_params(labelsize=7)

def _icq_legend(fig):
    fig.legend(handles=[
        plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=C.WC,ms=8,label='|ρ|≥0.3'),
        plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=C.WOC,ms=8,label='|ρ|<0.3'),
        plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',ms=6,label='CLM'),
        plt.Line2D([0],[0],marker='s',color='w',markerfacecolor='gray',ms=6,label='ETrA'),
    ], loc='upper center', ncol=4, fontsize=7, bbox_to_anchor=(0.5,1.0))


def plot_safety_icq_scatter(scenarios_df, out_path):
    """C2-1-3: Safety vs ICQ (3×3 grid)"""
    C.apply_style(); fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    sm = [('ttc_violation_2s','TTC Viol. (%)','lower'),('ttc_violation_2s','TTC Viol. (%)','lower'),('drac_critical_ratio','DRAC Crit.Ratio (m/s²)','lower')]
    for ri,(m,yl,b) in enumerate(sm):
        for ci,ic in enumerate(ICQ_COLS):
            il = ICQ_LABELS[ic].split()[0]
            _icq_scatter(axes[ri,ci], scenarios_df, m, ic, f'({chr(97+ri*3+ci)}) {yl.split()[0]} vs {il}', yl, b)
    _icq_legend(fig)
    fig.suptitle('Safety vs ICQ (CAD mode)', fontsize=10, fontweight='bold', color=C.TITLE, y=1.05)
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02); plt.close()


def plot_stability_icq_scatter(scenarios_df, out_path):
    """C2-2-3: Stability vs ICQ (2×3 grid)"""
    C.apply_style(); fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    sm = [('ego_jerk_rms','Jerk RMS (m/s³)','lower'),('ego_yaw_rms','Yaw RMS (rad/s)','lower')]
    for ri,(m,yl,b) in enumerate(sm):
        for ci,ic in enumerate(ICQ_COLS):
            il = ICQ_LABELS[ic].split()[0]
            _icq_scatter(axes[ri,ci], scenarios_df, m, ic, f'({chr(97+ri*3+ci)}) {yl.split()[0]} vs {il}', yl, b)
    _icq_legend(fig)
    fig.suptitle('Stability vs ICQ (CAD mode)', fontsize=10, fontweight='bold', color=C.TITLE, y=1.05)
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02); plt.close()


def plot_efficiency_icq_scatter(scenarios_df, out_path):
    """C2-3-3: Efficiency vs ICQ (2×3 grid)"""
    C.apply_style(); fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    sm = [('tct_norm','TCT_norm','lower'),('v_mean_norm','v_mean_norm','higher')]
    for ri,(m,yl,b) in enumerate(sm):
        for ci,ic in enumerate(ICQ_COLS):
            il = ICQ_LABELS[ic].split()[0]
            _icq_scatter(axes[ri,ci], scenarios_df, m, ic, f'({chr(97+ri*3+ci)}) {yl.split()[0]} vs {il}', yl, b)
    _icq_legend(fig)
    fig.suptitle('Efficiency vs ICQ (CAD mode)', fontsize=10, fontweight='bold', color=C.TITLE, y=1.05)
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02); plt.close()


def plot_radar(radar_df, out_path):
    """C2-4: Radar plots (Safety, Stability, Efficiency)"""
    C.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), subplot_kw=dict(polar=True))
    for idx, cat in enumerate(['Safety', 'Stability', 'Efficiency']):
        ax = axes[idx]
        sub = radar_df[radar_df['category']==cat]
        if len(sub) == 0: ax.set_visible(False); continue
        labels = sub['label'].tolist()
        wc_v = sub['WC_norm'].tolist(); woc_v = sub['WOC_norm'].tolist()
        wc_r = sub['WC_raw'].tolist();  woc_r = sub['WOC_raw'].tolist()
        n = len(labels)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist() + [0]
        wcp = wc_v + wc_v[:1]; wocp = woc_v + woc_v[:1]
        ax.plot(angles, wocp, 'o-', color=C.WOC, lw=1.5, label='SAD', ms=5)
        ax.fill(angles, wocp, color=C.WOC, alpha=0.2)
        ax.plot(angles, wcp,  'o-', color=C.WC,  lw=1.5, label='CAD',  ms=5)
        ax.fill(angles, wcp,  color=C.WC,  alpha=0.2)
        for i in range(n):
            ang = angles[i]
            wl = f'{wc_r[i]:.2f}' if abs(wc_r[i]) < 100 else f'{wc_r[i]:.1f}'
            ol = f'{woc_r[i]:.2f}' if abs(woc_r[i]) < 100 else f'{woc_r[i]:.1f}'
            ax.annotate(wl, xy=(ang, wc_v[i]+0.1), fontsize=8, color=C.WC, ha='center', va='bottom', fontweight='bold')
            ax.annotate(ol, xy=(ang, woc_v[i]-0.1), fontsize=8, color=C.WOC, ha='center', va='top')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([l.replace(' ','\n') if len(l)>10 else l for l in labels], fontsize=7)
        ax.set_ylim(0, 1.15); ax.set_yticks([0.25,0.5,0.75,1.0]); ax.set_yticklabels(['','0.5','','1.0'],fontsize=6)
        ax.set_title(cat, fontsize=10, fontweight='bold', color=C.TITLE, pad=15)
    fig.legend(handles=[plt.Line2D([0],[0],marker='o',color=C.WC, lw=1.5,ms=5,label='CAD'),
                        plt.Line2D([0],[0],marker='o',color=C.WOC,lw=1.5,ms=5,label='SAD')],
               loc='upper center', fontsize=8, bbox_to_anchor=(0.5,0.93), ncol=2)
    fig.suptitle('Performance Profile (higher = better)', fontsize=11, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02); plt.close()


def plot_boxplot(pairs_df, out_path):
    """C2-A1: Boxplot (3×3 grid)"""
    C.apply_style()
    metrics = [
        ('ttc_violation_2s','TTC Viol. (%)','Safety',None),
        ('ttc_violation_2s','TTC Viol. (%)','Safety',None),
        ('drac_critical_ratio','DRAC Crit.Ratio (m/s²)','Safety',None),
        ('ego_jerk_rms','Jerk RMS (m/s³)','Stability',None),
        ('ego_yaw_rms','Yaw RMS (rad/s)','Stability',None),
        ('tct_norm','TCT_norm','Efficiency',None),
        ('v_mean_norm','v_mean_norm','Efficiency',None),
        ('t_merge','Merge Time (s) [CLM]','Efficiency','CLM'),
        ('avoid_dist_tgt','Avoid Dist Tgt (m)','Efficiency','ETrA'),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(7, 6)); axes = axes.flatten()
    cat_col = {'Safety':C.SEC2,'Stability':C.SEC5,'Efficiency':C.SEC4}
    for i, (m, lbl, cat, tf) in enumerate(metrics):
        ax = axes[i]
        wc_c, woc_c = f'{m}_WC', f'{m}_WOC'
        if wc_c not in pairs_df.columns or woc_c not in pairs_df.columns:
            ax.set_visible(False); continue
        sub = pairs_df[pairs_df['type']==tf] if tf and 'type' in pairs_df.columns else pairs_df
        wc_d = sub[wc_c].dropna(); woc_d = sub[woc_c].dropna()
        if len(wc_d)==0 or len(woc_d)==0:
            ax.text(0.5,0.5,'No data',ha='center',va='center',transform=ax.transAxes)
            ax.set_title(lbl,fontsize=8,fontweight='bold'); continue
        bp = ax.boxplot([woc_d,wc_d], positions=[1,2], widths=0.6, patch_artist=True,
                        showmeans=True, meanline=True, showfliers=False,
                        medianprops={'color':'none'}, meanprops={'color':'black','linewidth':1.5})
        bp['boxes'][0].set_facecolor(C.WOC_TINT60); bp['boxes'][0].set_edgecolor(C.WOC)
        bp['boxes'][1].set_facecolor(C.WC_LIGHT_TINT60); bp['boxes'][1].set_edgecolor(C.WC)
        def fv(v): return f'{v:.4f}' if abs(v)<0.01 else (f'{v:.3f}' if abs(v)<1 else (f'{v:.2f}' if abs(v)<100 else f'{v:.1f}'))
        yl = ax.get_ylim(); yr = yl[1]-yl[0]; lo = yr*0.05
        ax.text(1,woc_d.mean()+lo,fv(woc_d.mean()),ha='center',va='bottom',fontsize=6,fontweight='bold',color=C.WOC,bbox=C.LABEL_BBOX)
        ax.text(2,wc_d.mean()+lo, fv(wc_d.mean()), ha='center',va='bottom',fontsize=6,fontweight='bold',color=C.WC, bbox=C.LABEL_BBOX)
        ax.set_xticks([1,2]); ax.set_xticklabels(['SAD','CAD'],fontsize=8)
        ax.set_title(lbl,fontsize=8,fontweight='bold'); ax.tick_params(labelsize=7)
        ax.axhline(y=ax.get_ylim()[1],color=cat_col[cat],lw=3,alpha=0.5)
    fig.legend(handles=[Patch(facecolor=C.WOC_TINT60,edgecolor=C.WOC,label='SAD'),
                        Patch(facecolor=C.WC_LIGHT_TINT60,edgecolor=C.WC,label='CAD'),
                        plt.Line2D([0],[0],color='black',lw=1.5,label='Mean')],
               loc='upper right', fontsize=8, bbox_to_anchor=(0.98,0.98), framealpha=0.9)
    fig.suptitle('Metrics Distribution: SAD vs CAD', fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02); plt.close()


def _heatmap_cmap():
    return LinearSegmentedColormap.from_list('wc_woc', C.HEATMAP_COLORS)


def _build_imp_matrix(df, metrics_config, extra_task_col=True):
    data_matrix, labels = [], []
    for m, lbl, b in metrics_config:
        wc_c, woc_c = f'{m}_WC', f'{m}_WOC'
        if wc_c not in df.columns or woc_c not in df.columns: continue
        row = [_imp_pct(r[wc_c], r[woc_c], b) for _, r in df.iterrows()]
        data_matrix.append(row); labels.append(lbl)
    if extra_task_col:
        task_row = []
        for _, r in df.iterrows():
            if r.get('type') == 'CLM':
                task_row.append(_imp_pct(r.get('t_merge_WC'), r.get('t_merge_WOC'), 'lower'))
            else:
                task_row.append(_imp_pct(r.get('avoid_dist_tgt_WC'), r.get('avoid_dist_tgt_WOC'), 'higher'))
        data_matrix.append(task_row); labels.append('Task Efficiency')
    return np.array(data_matrix, dtype=float), labels


def plot_heatmap_metric_scenario(pairs_df, out_path):
    cmap = _heatmap_cmap()
    mat, ylbls = _build_imp_matrix(pairs_df, HEATMAP_METRICS)
    xlbls = pairs_df['scenario'].tolist() if 'scenario' in pairs_df.columns else [str(i) for i in range(mat.shape[1])]
    C.apply_style(); fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=-100, vmax=100); ax.grid(False)
    ax.set_xticks(range(len(xlbls))); ax.set_yticks(range(len(ylbls)))
    ax.set_xticklabels(xlbls, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(ylbls, fontsize=8)
    for i in range(len(ylbls)):
        for j in range(mat.shape[1]):
            v = mat[i,j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=5,
                        color='white' if abs(v)>50 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8).set_label('Improvement % (CAD vs SAD)', fontsize=9)
    ax.set_title('Metric × Scenario: CAD Improvement (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Scenario', fontsize=9); ax.set_ylabel('Metric', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02); plt.close()


def plot_heatmap_location_metric(pairs_df, out_path):
    cmap = _heatmap_cmap(); tasks = ['CLM','ETrA']
    data_matrix, ylbls = [], []
    task_eff = {'CLM':('t_merge','lower'), 'ETrA':('avoid_dist_tgt','higher')}
    for m, lbl, b in HEATMAP_METRICS:
        wc_c, woc_c = f'{m}_WC', f'{m}_WOC'
        if wc_c not in pairs_df.columns: continue
        row = []
        for t in tasks:
            sub = pairs_df[pairs_df['type']==t] if 'type' in pairs_df.columns else pairs_df
            wc_v = sub[wc_c].dropna(); woc_v = sub[woc_c].dropna()
            row.append(_imp_pct(wc_v.mean(), woc_v.mean(), b) if len(wc_v)>0 and len(woc_v)>0 else np.nan)
        data_matrix.append(row); ylbls.append(lbl)
    te_row = []
    for t in tasks:
        met, b = task_eff[t]; sub = pairs_df[pairs_df['type']==t] if 'type' in pairs_df.columns else pairs_df
        wc_c, woc_c = f'{met}_WC', f'{met}_WOC'
        if wc_c in sub.columns:
            wc_v = sub[wc_c].dropna(); woc_v = sub[woc_c].dropna()
            te_row.append(_imp_pct(wc_v.mean(),woc_v.mean(),b) if len(wc_v)>0 and len(woc_v)>0 else np.nan)
        else: te_row.append(np.nan)
    data_matrix.append(te_row); ylbls.append('Task Efficiency')
    mat = np.array(data_matrix, dtype=float)
    C.apply_style(); fig, ax = plt.subplots(figsize=(4, 6))
    im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=-100, vmax=100); ax.grid(False)
    ax.set_xticks(range(2)); ax.set_yticks(range(len(ylbls)))
    ax.set_xticklabels(tasks, fontsize=10, fontweight='bold'); ax.set_yticklabels(ylbls, fontsize=9)
    for i in range(len(ylbls)):
        for j in range(2):
            v = mat[i,j]
            if not np.isnan(v):
                ax.text(j,i,f'{("+" if v>0 else "")}{v:.1f}%',ha='center',va='center',
                        fontsize=9,color='white' if abs(v)>50 else 'black',fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8).set_label('Improvement % (CAD vs SAD)', fontsize=9)
    ax.set_title('Task × Metric: CAD Improvement (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Scenario Type', fontsize=10); ax.set_ylabel('Metric', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02); plt.close()


def plot_heatmap_category_summary(pairs_df, out_path):
    cats = {
        'Safety':    [('ttc_violation_2s','lower'),('ttc_violation_3s','lower'),('ttc_violation_2s','lower'),('drac_critical_ratio','lower'),('d_min','higher')],
        'Stability': [('ego_jerk_rms','lower'),('ego_jerk_peak','lower'),('ego_yaw_rms','lower'),('ego_yaw_zcr','lower')],
        'Efficiency':[('tct_norm','lower'),('v_mean_norm','higher')],
    }
    task_eff = {'CLM':('t_merge','lower'), 'ETrA':('avoid_dist_tgt','higher')}
    groups = ['Overall','CLM','ETrA']
    cmap = _heatmap_cmap(); data_matrix = []
    for cat_name, metrics in cats.items():
        row = []
        for grp in groups:
            sub = pairs_df if grp=='Overall' else (pairs_df[pairs_df['type']==grp] if 'type' in pairs_df.columns else pairs_df)
            imps = []
            for m, b in metrics:
                wc_c, woc_c = f'{m}_WC', f'{m}_WOC'
                if wc_c not in sub.columns: continue
                wc_v = sub[wc_c].dropna(); woc_v = sub[woc_c].dropna()
                if len(wc_v)>0 and len(woc_v)>0: imps.append(_imp_pct(wc_v.mean(),woc_v.mean(),b))
            if cat_name == 'Efficiency':
                if grp == 'Overall':
                    for tt, (met, b) in task_eff.items():
                        ts = pairs_df[pairs_df['type']==tt] if 'type' in pairs_df.columns else pairs_df
                        if f'{met}_WC' in ts.columns:
                            wc_v = ts[f'{met}_WC'].dropna(); woc_v = ts[f'{met}_WOC'].dropna()
                            if len(wc_v)>0 and len(woc_v)>0: imps.append(_imp_pct(wc_v.mean(),woc_v.mean(),b))
                elif grp in task_eff:
                    met, b = task_eff[grp]
                    if f'{met}_WC' in sub.columns:
                        wc_v = sub[f'{met}_WC'].dropna(); woc_v = sub[f'{met}_WOC'].dropna()
                        if len(wc_v)>0 and len(woc_v)>0: imps.append(_imp_pct(wc_v.mean(),woc_v.mean(),b))
            row.append(np.mean(imps) if imps else np.nan)
        data_matrix.append(row)
    mat = np.array(data_matrix, dtype=float)
    C.apply_style(); fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=-50, vmax=50); ax.grid(False)
    cat_names = list(cats.keys())
    ax.set_xticks(range(3)); ax.set_yticks(range(len(cat_names)))
    ax.set_xticklabels(groups,fontsize=10); ax.set_yticklabels(cat_names,fontsize=10)
    for i in range(len(cat_names)):
        for j in range(3):
            v = mat[i,j]
            if not np.isnan(v):
                ax.text(j,i,f'{("+" if v>0 else "")}{v:.1f}%',ha='center',va='center',
                        fontsize=11,color='white' if abs(v)>25 else 'black',fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8).set_label('Avg Improvement % (CAD vs SAD)', fontsize=9)
    ax.set_title('Category Summary: CAD Improvement (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Scenario Type',fontsize=10); ax.set_ylabel('Category',fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02); plt.close()
