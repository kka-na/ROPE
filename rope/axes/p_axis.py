#!/usr/bin/env python3
"""P-Axis: Performance — Safety / Stability / Efficiency (WC vs WOC)"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon


def _cohens_d(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2: return np.nan
    pooled = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2) / (len(a)+len(b)-2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


def _imp_pct(wc, woc, better='lower'):
    if pd.isna(wc) or pd.isna(woc): return np.nan
    denom = max(abs(wc), abs(woc))
    if denom < 1e-9: return 0
    return ((woc - wc) / denom * 100) if better == 'lower' else ((wc - woc) / denom * 100)




METRICS = {
    'safety': [
        ('ttc_violation_2s',  'TTC <2s (%)',       'lower'),
        ('drac_critical_ratio',     'DRAC Critical Ratio (%)',    'lower'),
        ('d_min',             'Min Long Dist (m)', 'higher'),
    ],
    'stability': [
        ('ego_yaw_rms',    'Ego Yaw RMS',  'lower'),
        ('target_yaw_rms', 'Tgt Yaw RMS',  'lower'),
        ('ego_jerk_rms',   'Ego Jerk RMS', 'lower'),
        ('target_jerk_rms','Tgt Jerk RMS', 'lower'),
    ],
    'efficiency': [
        ('tct_norm',      'TCT_norm',          'lower'),
        ('v_mean_norm',   'v_mean_norm',       'higher'),
        ('v_std_norm',    'v_std_norm',        'lower'),
        ('merge_success', 'Merge Success Rate','higher'),
        ('t_merge',       'Merge Time* (s)',   'lower'),
        ('avoid_success', 'Avoid Success Rate','higher'),
        ('avoid_dist_tgt','Avoid Dist Tgt (m)','higher'),
    ],
}

ALL_METRICS = [(col, lbl, bet) for metrics in METRICS.values() for col, lbl, bet in metrics]


import re as _re
def _base_name(scenario):
    s = _re.sub(r'_\d{6,}$', '', scenario)   # strip trailing timestamp
    for tag in ('_WC_', '_WOC_'):
        if tag in s: return s.replace(tag, '_', 1)
    for tag in ('_WC', '_WOC'):
        if s.endswith(tag): return s[:-len(tag)]
    return s


class PAxis:
    def __init__(self, output_dir: str, paired: bool = False):
        self.out     = Path(output_dir)
        self.paired  = paired
        self.metrics = self.out / 'sequences' / 'scenario_metrics.csv'

    def run(self) -> Dict:
        if not self.metrics.exists():
            return {'table': pd.DataFrame(), 'scenarios': pd.DataFrame(), 'figures': []}
        scenarios  = pd.read_csv(self.metrics)
        stat_table = self._compute_statistics(scenarios)
        pairs_df   = self._build_pairs_csv(scenarios)
        scenarios.to_csv(self.out / 'p_axis_scenarios.csv', index=False)
        stat_table.to_csv(self.out / 'p_axis_table.csv', index=False)
        if not pairs_df.empty:
            pairs_df.to_csv(self.out / 'p_axis_pairs.csv', index=False)
        radar_df = self._build_radar_data(stat_table)
        if not radar_df.empty:
            radar_df.to_csv(self.out / 'p_axis_radar.csv', index=False)
        figs = self._build_figures(scenarios, stat_table)
        return {'table': stat_table, 'scenarios': scenarios, 'figures': figs}

    def _compute_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'mode' not in df.columns:
            return pd.DataFrame()
        wc  = df[df['mode'] == 'WC']
        woc = df[df['mode'] == 'WOC']

        # Build WC/WOC pair index for paired Wilcoxon (FIELD only)
        pair_df = None
        if self.paired:
            wc_p  = wc.copy();  wc_p['_b']  = wc_p['scenario'].apply(_base_name)
            woc_p = woc.copy(); woc_p['_b'] = woc_p['scenario'].apply(_base_name)
            wc_p  = wc_p.groupby('_b').first()
            woc_p = woc_p.groupby('_b').first()
            pair_df = wc_p.join(woc_p, lsuffix='_wc', rsuffix='_woc', how='inner')

        rows = []
        for cat, metrics in METRICS.items():
            for col, label, better in metrics:
                wc_v  = wc[col].dropna().values  if col in wc.columns  else np.array([])
                woc_v = woc[col].dropna().values if col in woc.columns else np.array([])
                if len(wc_v) < 2 or len(woc_v) < 2: continue

                p, test = np.nan, 'Mann-Whitney'
                if self.paired and pair_df is not None:
                    wc_col, woc_col = f'{col}_wc', f'{col}_woc'
                    if wc_col in pair_df.columns and woc_col in pair_df.columns:
                        sub = pair_df[[wc_col, woc_col]].dropna()
                        if len(sub) >= 2:
                            try:
                                _, p = wilcoxon(sub[wc_col].values, sub[woc_col].values)
                                test = f'Wilcoxon(n={len(sub)})'
                            except Exception: p = np.nan
                if np.isnan(p):
                    try: _, p = mannwhitneyu(wc_v, woc_v, alternative='two-sided')
                    except Exception: p = np.nan
                    test = 'Mann-Whitney'

                delta = wc_v.mean() - woc_v.mean()
                pct   = 100 * delta / woc_v.mean() if woc_v.mean() != 0 else np.nan
                rows.append({
                    'category': cat, 'metric': col, 'label': label,
                    'WOC_mean': round(woc_v.mean(), 4), 'WOC_std': round(woc_v.std(), 4),
                    'WC_mean':  round(wc_v.mean(),  4), 'WC_std':  round(wc_v.std(),  4),
                    'delta':    round(delta, 4),
                    'pct_change': round(pct, 2) if np.isfinite(pct) else np.nan,
                    'p_value':    round(p, 4) if np.isfinite(p) else np.nan,
                    'cohens_d':   round(_cohens_d(wc_v, woc_v), 3),
                    'better_if': better,
                    'significant': bool(p < 0.05) if np.isfinite(p) else False,
                    'n_WC': len(wc_v), 'n_WOC': len(woc_v),
                    'test': test,
                })
        return pd.DataFrame(rows)

    def _build_pairs_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build WC/WOC paired CSV: one row per base scenario."""
        if df.empty or 'mode' not in df.columns:
            return pd.DataFrame()
        wc  = df[df['mode'] == 'WC'].copy()
        woc = df[df['mode'] == 'WOC'].copy()
        wc['_base']  = wc['scenario'].apply(_base_name)
        woc['_base'] = woc['scenario'].apply(_base_name)
        wc  = wc.set_index('_base')
        woc = woc.set_index('_base')
        common = wc.index.intersection(woc.index)
        rows = []
        for base in common:
            wc_r  = wc.loc[base]  if isinstance(wc.loc[base],  pd.Series) else wc.loc[base].iloc[0]
            woc_r = woc.loc[base] if isinstance(woc.loc[base], pd.Series) else woc.loc[base].iloc[0]
            row = {'scenario': base,
                   'type': wc_r.get('type', ''), 'location': wc_r.get('location', '')}
            for col, _, _ in ALL_METRICS:
                row[f'{col}_WC']  = wc_r.get(col, np.nan)
                row[f'{col}_WOC'] = woc_r.get(col, np.nan)
            rows.append(row)
        return pd.DataFrame(rows)

    def _build_radar_data(self, stat_table: pd.DataFrame) -> pd.DataFrame:
        """Build normalized radar chart data from stat_table."""
        if stat_table.empty: return pd.DataFrame()
        rows = []
        for cat in ['safety', 'stability', 'efficiency']:
            sub = stat_table[stat_table['category'] == cat]
            for _, r in sub.iterrows():
                wc_m, woc_m, better = r['WC_mean'], r['WOC_mean'], r['better_if']
                mx = max(abs(wc_m), abs(woc_m)) or 1
                if better == 'lower':
                    wc_n  = 1 - abs(wc_m)  / mx
                    woc_n = 1 - abs(woc_m) / mx
                else:
                    wc_n  = abs(wc_m)  / mx
                    woc_n = abs(woc_m) / mx
                rows.append({'category': cat.capitalize(), 'label': r['label'],
                             'WC_norm': round(max(0, min(1, wc_n)), 4),
                             'WOC_norm': round(max(0, min(1, woc_n)), 4),
                             'WC_raw': round(wc_m, 4), 'WOC_raw': round(woc_m, 4)})
        return pd.DataFrame(rows)

    def _build_figures(self, scenarios: pd.DataFrame, stat_table: pd.DataFrame, **_) -> List:
        from rope.viz.summary_panels import plot_p_summary
        f = plot_p_summary(stat_table)
        return [f] if f else []
