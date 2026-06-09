#!/usr/bin/env python3
"""Overall / Cross-Axis summary graphs"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy import stats

from rope.viz import colors as C

ICQ_METRICS  = ['latency_ms', 'prr_mean', 'pkt_size_mean']
PERF_METRICS = {
    'ttc_violation_2s': 'TTC Viol.',
    'ttc_violation_2s': 'TTC Viol.',
    'drac_critical_ratio':        'DRAC Crit.Ratio',
    'd_min':            'Min Dist.',
    'ego_jerk_rms':     'Jerk RMS',
    'ego_yaw_rms':      'Yaw RMS',
    'tct_norm':         'TCT_norm',
    'v_mean_norm':      'v_mean_norm',
    't_merge':          'Merge Time',
    'avoid_dist_tgt':   'Avoid Dist Tgt',
}


def plot_chain_scatter(scenarios_df, out_path=None):
    """O-1: R→O→P chain scatter (1×3): PRR→ΔOST, ΔOST→Headway, ΔOST→Jerk"""
    df_wc = scenarios_df[scenarios_df['mode']=='WC'].copy() if 'mode' in scenarios_df.columns else scenarios_df.copy()
    C.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

    def scatter_panel(ax, x_col, y_col, xlabel, ylabel, title, color):
        if x_col not in df_wc.columns or y_col not in df_wc.columns:
            ax.set_visible(False); return
        valid = df_wc[[x_col, y_col]].dropna()
        if len(valid) < 5:
            ax.text(0.5,0.5,'Insufficient data',ha='center',va='center',transform=ax.transAxes)
            ax.set_title(title,fontsize=9,fontweight='bold'); return
        x, y = valid[x_col], valid[y_col]
        for t, mk in [('CLM','o'),('ETrA','s')]:
            if 'type' in df_wc.columns:
                mask = valid.index.isin(df_wc[df_wc['type']==t].index)
            else:
                mask = [True]*len(valid)
            if sum(mask): ax.scatter(x[mask],y[mask],c=color,marker=mk,s=50,alpha=0.7,edgecolors='white',lw=0.5)
        sl, ic, _, _, _ = stats.linregress(x, y)
        xl = np.linspace(x.min(), x.max(), 100)
        ax.plot(xl, sl*xl+ic, color=color, ls='--', lw=1.5, alpha=0.8)
        rho, pv = stats.spearmanr(x, y)
        sig = '**' if pv<0.01 else ('*' if pv<0.05 else ('†' if pv<0.1 else ''))
        ax.text(0.05,0.95,f'ρ={rho:.2f}{sig}',transform=ax.transAxes,fontsize=8,fontweight='bold',va='top',bbox=C.LABEL_BBOX)
        ax.set_xlabel(xlabel,fontsize=8); ax.set_ylabel(ylabel,fontsize=8)
        ax.set_title(title,fontsize=9,fontweight='bold',color=C.TITLE); ax.tick_params(labelsize=7); ax.grid(False)

    scatter_panel(axes[0],'prr_mean','delta_ost','PRR (%)','ΔOST','(a) R→O: PRR → ΔOST',C.WC)
    scatter_panel(axes[1],'delta_ost','ttc_violation_2s','ΔOST','TTC Viol. (%)','(b) O→P: ΔOST → Safety',C.SEC2)
    scatter_panel(axes[2],'delta_ost','ego_jerk_rms','ΔOST','Jerk RMS (m/s³)','(c) O→P: ΔOST → Stability',C.SEC4)

    fig.legend(handles=[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',ms=7,label='CLM'),
                        plt.Line2D([0],[0],marker='s',color='w',markerfacecolor='gray',ms=7,label='ETrA')],
               loc='upper center', ncol=2, fontsize=7, bbox_to_anchor=(0.5,1.02))
    fig.suptitle('R → O → P Chain: ICQ → ΔOST → Performance', fontsize=10, fontweight='bold', color=C.TITLE, y=1.08)
    plt.tight_layout(rect=[0,0,1,0.95])
    if out_path:
        plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02); plt.close()
    return fig


def plot_correlation_heatmap(corr_df, out_path=None):
    """O-2: Full pairwise correlation heatmap"""
    icq_v  = [v for v in ICQ_METRICS if v in corr_df.get('var1',corr_df.columns).tolist() or v in corr_df.get('var2',[]).tolist()]
    dh_v   = ['delta_ost']
    perf_v = [v for v in PERF_METRICS if v in corr_df.get('var1',pd.Index([])).tolist() or v in corr_df.get('var2',pd.Index([])).tolist()]
    all_v  = icq_v + dh_v + perf_v
    if 'var1' not in corr_df.columns:
        return None
    avail = [v for v in all_v if v in corr_df['var1'].values or v in corr_df['var2'].values]
    n = len(avail)
    if n == 0: return None
    mat, pmat = np.zeros((n,n)), np.zeros((n,n)); np.fill_diagonal(mat,1.0)
    for _, row in corr_df.iterrows():
        if row['var1'] in avail and row['var2'] in avail:
            i,j = avail.index(row['var1']), avail.index(row['var2'])
            mat[i,j] = mat[j,i] = row['rho']
            pmat[i,j] = pmat[j,i] = row['p_value']
    lbls = [PERF_METRICS.get(v, v.replace('_mean','').replace('pkt_size','PktSize').replace('delta_ost','ΔOST')) for v in avail]

    C.apply_style()
    cmap = LinearSegmentedColormap.from_list('corr',[C.SEC4,'white',C.SEC5])
    fig, ax = plt.subplots(figsize=(8,7))
    im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    from matplotlib.patches import Rectangle
    for i in range(n):
        ax.add_patch(Rectangle((i-.5,i-.5),1,1,facecolor='white',edgecolor=C.WOC,hatch='//',lw=0.5))
    for i in range(n):
        for j in range(n):
            if i!=j:
                v,p = mat[i,j], pmat[i,j]
                sig = '**' if p<0.01 else ('*' if p<0.05 else '')
                ax.text(j,i,f'{v:.2f}{sig}',ha='center',va='center',fontsize=6,
                        color='white' if abs(v)>0.5 else 'black')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(lbls,rotation=45,ha='right',fontsize=7)
    ax.set_yticklabels(lbls,fontsize=7)
    icq_end = len(icq_v); dh_end = icq_end+1
    for pos in [icq_end-.5, dh_end-.5]:
        ax.axhline(y=pos,color='black',lw=1.5); ax.axvline(x=pos,color='black',lw=1.5)
    ax.grid(False)
    plt.colorbar(im,ax=ax,shrink=0.8).set_label('Spearman ρ',fontsize=8)
    ax.set_title('Cross-Axis Correlation Matrix (CAD mode)',fontsize=10,fontweight='bold',color=C.TITLE,pad=10)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02); plt.close()
    return fig


def plot_improvement_summary(improvement_df, out_path=None):
    """O-3: CAD vs SAD effect size bar chart (Cohen's d)"""
    if improvement_df.empty: return None
    cat_ord = {'Uncertainty':0,'Safety':1,'Stability':2,'Efficiency':3,'Task Efficiency':4}
    df = improvement_df.copy()
    df['_cs'] = df['category'].map(cat_ord).fillna(99)
    df = df.sort_values(['_cs','cohens_d'], ascending=[True,False])
    lbls = df['label'].tolist(); ds = df['cohens_d'].tolist()
    pvs  = df['p_value'].tolist(); cats = df['category'].tolist(); betters = df['better_if'].tolist()
    adj  = [-d if b=='lower' else d for d,b in zip(ds,betters)]
    bar_colors = [C.WC if d>0 else C.WOC for d in adj]
    C.apply_style()
    fig, ax = plt.subplots(figsize=(7,5))
    ax.barh(range(len(lbls)), adj, color=bar_colors, alpha=0.8, edgecolor='white')
    prev = None
    for i,(d,p,c) in enumerate(zip(adj,pvs,cats)):
        sig = '**' if p<0.01 else ('*' if p<0.05 else ('†' if p<0.1 else ''))
        off = 0.02 if d>=0 else -0.02
        ax.text(d+off,i,sig,ha='left' if d>=0 else 'right',va='center',fontsize=8,fontweight='bold')
        if prev and c!=prev: ax.axhline(y=i-.5,color='gray',ls='--',lw=0.5)
        prev = c
    ax.axvline(x=0,color='black',lw=1)
    for t in [0.2,0.5,0.8]:
        ax.axvline(x=t,color='gray',ls=':',lw=0.5,alpha=0.5); ax.axvline(x=-t,color='gray',ls=':',lw=0.5,alpha=0.5)
    ax.set_yticks(range(len(lbls))); ax.set_yticklabels(lbls,fontsize=8)
    ax.set_xlabel("Cohen's d (positive = CAD better)",fontsize=9); ax.set_xlim(-2,2); ax.grid(False)
    ax.legend(handles=[Patch(facecolor=C.WC,label='CAD better'),Patch(facecolor=C.WOC,label='SAD better')],
              loc='lower right',fontsize=7)
    ax.set_title('CAD vs SAD: Effect Size Summary',fontsize=10,fontweight='bold',color=C.TITLE)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, format='svg', bbox_inches='tight', pad_inches=0.02); plt.close()
    return fig
