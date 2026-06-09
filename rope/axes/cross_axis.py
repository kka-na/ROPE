#!/usr/bin/env python3
"""Cross-Axis: source-wise (R_WC→OST_WC, R_WOC→OST_WOC) + difference chain (ΔR→ΔOST→ΔP)."""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rope.utils import match_key


def _iqr_mask(v, k=1.5):
    q1, q3 = np.nanpercentile(v, 25), np.nanpercentile(v, 75)
    iqr = q3 - q1
    return (v >= q1 - k * iqr) & (v <= q3 + k * iqr) if iqr > 0 else np.ones(len(v), bool)


def _spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4: return np.nan, np.nan, 0
    x, y = x[mask], y[mask]
    mask2 = _iqr_mask(x) & _iqr_mask(y)
    n = int(mask2.sum())
    if n < 4: return np.nan, np.nan, 0
    rho, p = spearmanr(x[mask2], y[mask2])
    return float(rho), float(p), n


def _corr_rows(df, xs, ys, path_label):
    rows = []
    for x_col, x_lbl in xs:
        for y_col, y_lbl in ys:
            if x_col not in df or y_col not in df: continue
            rho, p, n = _spearman(df[x_col].values.astype(float), df[y_col].values.astype(float))
            rows.append({'path': path_label, 'x': x_lbl, 'y': y_lbl, 'rho': rho, 'p_value': p, 'n': n})
    return rows


def _pair_by_key(wc_df, woc_df, cols):
    """Merge WC/WOC on match_key, take first match per key (handles duplicates)."""
    wc = wc_df.copy(); wc['_key'] = wc['scenario'].apply(match_key)
    woc = woc_df.copy(); woc['_key'] = woc['scenario'].apply(match_key)
    wc = wc.groupby('_key', as_index=False).first()
    woc = woc.groupby('_key', as_index=False).first()
    return pd.merge(wc[['_key', 'scenario'] + [c for c in cols if c in wc.columns]],
                    woc[['_key'] + [c for c in cols if c in woc.columns]],
                    on='_key', how='inner', suffixes=('_wc', '_woc'))


def _build_delta_r(r_table):
    if r_table.empty: return pd.DataFrame()
    cols = ['latency_ms', 'valid_rate']
    wc  = r_table[r_table['mode'] == 'WC']
    woc = r_table[r_table['mode'] == 'WOC']
    m = _pair_by_key(wc, woc, cols)
    if m.empty: return pd.DataFrame()
    delta = pd.DataFrame({'scenario': m['scenario'].values})
    for col in cols:
        wc_col = f'{col}_wc' if f'{col}_wc' in m else col
        woc_col = f'{col}_woc' if f'{col}_woc' in m else col
        if wc_col in m and woc_col in m:
            delta[f'd_{col}'] = m[wc_col].values - m[woc_col].values
    return delta


def _build_delta_p(p_table):
    if p_table.empty or 'mode' not in p_table.columns: return pd.DataFrame()
    cols = ['drac_critical_ratio', 'd_min', 'ttc_violation_2s',
            'tct_norm', 't_merge', 'v_mean_norm',
            'avoid_success', 'avoid_dist_tgt']
    wc  = p_table[p_table['mode'] == 'WC']
    woc = p_table[p_table['mode'] == 'WOC']
    m = _pair_by_key(wc, woc, cols)
    if m.empty: return pd.DataFrame()
    delta = pd.DataFrame({'scenario': m['scenario'].values})
    for col in cols:
        wc_col = f'{col}_wc' if f'{col}_wc' in m else col
        woc_col = f'{col}_woc' if f'{col}_woc' in m else col
        if wc_col in m and woc_col in m:
            delta[f'd_{col}'] = m[wc_col].values - m[woc_col].values
    return delta


class CrossAxis:
    def __init__(self, output_dir: str):
        self.out = Path(output_dir)

    def run(self, r_table=None, p_table=None, **_) -> Dict:
        def _load(path): return pd.read_csv(path) if path.exists() else pd.DataFrame()
        if r_table is None: r_table = _load(self.out / 'r_axis_table.csv')
        o_pairs = _load(self.out / 'o_axis_pairs.csv')
        if p_table is None: p_table = _load(self.out / 'p_axis_scenarios.csv')

        corr_df = pd.concat([
            self._source_wise(r_table, o_pairs),       # R_WC→OST_WC, R_WOC→OST_WOC
            self._r_to_o_component(r_table, o_pairs),  # R→O component-wise
            self._delta_r_to_o(r_table, o_pairs),      # ΔR→ΔOST
            self._o_to_p(o_pairs, p_table),            # ΔOST→P (WC scenarios)
            self._delta_o_to_p(o_pairs, p_table),      # ΔOST→ΔP (paired)
            self._delta_r_to_p(r_table, p_table),      # ΔR→ΔP (paired)
            self._r_to_p(r_table, p_table),            # R_WC→P_WC (legacy)
        ], ignore_index=True)
        corr_df.to_csv(self.out / 'cross_axis_correlation.csv', index=False)
        return {'table': corr_df, 'figures': self._build_figures(r_table, o_pairs, p_table)}

    _R_CHAIN  = [('latency_ms', 'latency_ms'), ('valid_rate', 'valid_rate')]
    _R_WC_ONLY = [('latency_ms', 'Latency (ms)'), ('valid_rate', 'ValidRate (%)')]
    _DOST    = [('delta_ost',   'ΔOST'), ('delta_ost_c', 'ΔOST_c')]
    _OST_WC  = [('ost_wc',     'OST_WC'), ('ost_c_wc',  'OST_c_WC')]
    _OST_WOC = [('ost_woc',    'OST_WOC'), ('ost_c_woc', 'OST_c_WOC')]
    _PERF = [
        ('ttc_violation_2s', 'TTC Viol 2s'), ('drac_critical_ratio', 'DRAC Crit.Ratio'), ('d_min', 'Min Long Dist'),
        ('ego_yaw_rms', 'Ego Yaw RMS'), ('target_yaw_rms', 'Tgt Yaw RMS'),
        ('ego_jerk_rms', 'Ego Jerk RMS'), ('target_jerk_rms', 'Tgt Jerk RMS'),
        ('tct_norm', 'TCT_norm'), ('merge_success', 'Merge Success'), ('t_merge', 'T_merge'),
        ('avoid_success', 'Avoid Success'), ('avoid_dist_tgt', 'Avoid Dist Tgt'),
        ('v_mean_norm', 'v_mean_norm'), ('v_std_norm', 'v_std_norm'),
    ]
    _DPERF = [
        ('d_drac_critical_ratio', 'ΔDRAC Critical Ratio (%)'), ('d_d_min', 'ΔMin Long Dist'),
        ('d_ttc_violation_2s', 'ΔTTC Viol 2s'),
        ('d_tct_norm', 'ΔTCT_norm'), ('d_t_merge', 'ΔT_merge'),
        ('d_avoid_success', 'ΔAvoid Success'), ('d_avoid_dist_tgt', 'ΔAvoid Dist Tgt'),
        ('d_v_mean_norm', 'Δv_mean_norm'), ('d_drac_norm', 'ΔDRAC_norm'),
    ]

    def _source_wise(self, r, o_pairs):
        """R_WC→OST_WC and R_WOC→OST_WOC source-internal chains."""
        if r.empty or o_pairs.empty: return pd.DataFrame()
        rows = []
        # WC chain
        wc_r = r[r['mode'] == 'WC'][['scenario', 'latency_ms', 'valid_rate']]
        m_wc = pd.merge(wc_r, o_pairs[['scenario'] + [c for c, _ in self._OST_WC]], on='scenario', how='inner')
        xs = [(c, f'{c}(WC)') for c in ['latency_ms', 'valid_rate'] if c in m_wc]
        rows += _corr_rows(m_wc, xs, self._OST_WC, 'R_WC→OST_WC')
        # WOC chain: join WOC r_axis with o_pairs via match_key
        woc_r = r[r['mode'] == 'WOC'].copy()
        woc_r['_key'] = woc_r['scenario'].apply(match_key)
        op = o_pairs.copy()
        op['_key'] = op['scenario'].apply(match_key)
        m_woc = pd.merge(woc_r[['_key', 'latency_ms', 'valid_rate']],
                         op[['_key'] + [c for c, _ in self._OST_WOC]], on='_key', how='inner')
        xs_woc = [(c, f'{c}(WOC)') for c in ['latency_ms', 'valid_rate'] if c in m_woc]
        rows += _corr_rows(m_woc, xs_woc, self._OST_WOC, 'R_WOC→OST_WOC')
        return pd.DataFrame(rows)

    def _r_to_o_component(self, r, o_pairs):
        """[Ablation] R(WC) → individual OST components: avail, Ce, Cde, OST_c."""
        if r.empty or o_pairs.empty: return pd.DataFrame()
        r_wc = r[r['mode'] == 'WC']
        comp_wc  = [('avail_wc',  'a_WC'),  ('Ce_wc',  'Ce_WC'),  ('Cde_wc',  'Cde_WC'),  ('ost_c_wc',  'OST_c_WC')]
        comp_woc = [('avail_woc', 'a_WOC'), ('Ce_woc', 'Ce_WOC'), ('Cde_woc', 'Cde_WOC'), ('ost_c_woc', 'OST_c_WOC')]
        r_xs = [('latency_ms', 'latency_ms'), ('valid_rate', 'valid_rate'),
                ('sampling_jitter_ms', 'jitter_ms')]
        rows = []
        # R_WC → WC components
        op_cols = ['scenario'] + [c for c, _ in comp_wc if c in o_pairs.columns]
        m_wc = pd.merge(r_wc[['scenario'] + [c for c, _ in r_xs if c in r_wc.columns]],
                        o_pairs[op_cols], on='scenario', how='inner')
        for x_col, x_lbl in r_xs:
            if x_col not in m_wc: continue
            for y_col, y_lbl in comp_wc:
                if y_col not in m_wc: continue
                rho, p, n = _spearman(m_wc[x_col].values.astype(float),
                                      m_wc[y_col].values.astype(float))
                rows.append({'path': 'R_WC→O_component', 'x': x_lbl, 'y': y_lbl,
                             'rho': rho, 'p_value': p, 'n': n})
        # R_WOC → WOC components (join via match_key)
        r_woc = r[r['mode'] == 'WOC'].copy()
        r_woc['_key'] = r_woc['scenario'].apply(match_key)
        op_woc = o_pairs.copy(); op_woc['_key'] = op_woc['scenario'].apply(match_key)
        woc_sel = [c for c, _ in comp_woc if c in op_woc.columns]
        m_woc = pd.merge(r_woc[['_key'] + [c for c, _ in r_xs if c in r_woc.columns]],
                         op_woc[['_key'] + woc_sel], on='_key', how='inner')
        for x_col, x_lbl in r_xs:
            if x_col not in m_woc: continue
            for y_col, y_lbl in comp_woc:
                if y_col not in m_woc: continue
                rho, p, n = _spearman(m_woc[x_col].values.astype(float),
                                      m_woc[y_col].values.astype(float))
                rows.append({'path': 'R_WOC→O_component', 'x': x_lbl + '(WOC)', 'y': y_lbl,
                             'rho': rho, 'p_value': p, 'n': n})
        return pd.DataFrame(rows)

    def _delta_r_to_o(self, r, o_pairs):
        """ΔR → ΔOST."""
        dr = _build_delta_r(r)
        if dr.empty or o_pairs.empty: return pd.DataFrame()
        merged = pd.merge(dr, o_pairs[['scenario'] + [c for c, _ in self._DOST]], on='scenario', how='inner')
        xs = [(c, 'Δ' + c[2:]) for c in dr.columns if c.startswith('d_')]
        return pd.DataFrame(_corr_rows(merged, xs, self._DOST, 'ΔR→ΔOST'))

    def _o_to_p(self, o_pairs, p):
        """ΔOST → P_WC (legacy WC-scenario level)."""
        if o_pairs.empty or p.empty: return pd.DataFrame()
        dost_cols = [c for c, _ in self._DOST if c in o_pairs.columns]
        merged = pd.merge(o_pairs[['scenario'] + dost_cols],
                          p[p['mode'] == 'WC'] if 'mode' in p.columns else p,
                          on='scenario', how='inner')
        rows = []
        for ost_col, ost_lbl in self._DOST:
            if ost_col not in merged: continue
            for perf_col, perf_lbl in self._PERF:
                if perf_col not in merged: continue
                rho, pv, n = _spearman(merged[ost_col].values.astype(float),
                                       merged[perf_col].values.astype(float))
                rows.append({'path': 'O→P', 'x': ost_lbl, 'y': perf_lbl, 'rho': rho, 'p_value': pv, 'n': n})
        return pd.DataFrame(rows)

    def _delta_o_to_p(self, o_pairs, p):
        """ΔOST → ΔP (paired)."""
        dp = _build_delta_p(p)
        if o_pairs.empty or dp.empty: return pd.DataFrame()
        dost = [c for c, _ in self._DOST if c in o_pairs.columns]
        merged = pd.merge(o_pairs[['scenario'] + dost], dp, on='scenario', how='inner')
        rows = []
        for ost_col, ost_lbl in self._DOST:
            if ost_col not in merged: continue
            for perf_col, perf_lbl in self._DPERF:
                if perf_col not in merged: continue
                rho, pv, n = _spearman(merged[ost_col].values.astype(float),
                                       merged[perf_col].values.astype(float))
                rows.append({'path': 'ΔO→ΔP', 'x': ost_lbl, 'y': perf_lbl, 'rho': rho, 'p_value': pv, 'n': n})
        return pd.DataFrame(rows)

    def _delta_r_to_p(self, r, p):
        """ΔR → ΔP (paired)."""
        dr = _build_delta_r(r)
        dp = _build_delta_p(p)
        if dr.empty or dp.empty: return pd.DataFrame()
        merged = pd.merge(dr, dp, on='scenario', how='inner')
        xs = [(c, 'Δ' + c[2:]) for c in dr.columns if c.startswith('d_')]
        return pd.DataFrame(_corr_rows(merged, xs, self._DPERF, 'ΔR→ΔP'))

    def _r_to_p(self, r, p):
        """R_WC → P_WC (legacy, WC-only)."""
        if r.empty or p.empty: return pd.DataFrame()
        icq = [c for c, _ in self._R_WC_ONLY if c in r.columns]
        r_wc = r[r['mode'] == 'WC'][['scenario'] + icq].rename(columns={c: f'_r_{c}' for c in icq})
        p_wc = p[p['mode'] == 'WC'] if 'mode' in p.columns else p
        merged = pd.merge(r_wc, p_wc, on='scenario', how='inner')
        rows = []
        for x_col, x_lbl in self._R_WC_ONLY:
            rc = f'_r_{x_col}'
            for perf_col, perf_lbl in self._PERF:
                if rc not in merged or perf_col not in merged: continue
                rho, pv, n = _spearman(merged[rc].values.astype(float),
                                       merged[perf_col].values.astype(float))
                rows.append({'path': 'R→P', 'x': x_lbl, 'y': perf_lbl, 'rho': rho, 'p_value': pv, 'n': n})
        return pd.DataFrame(rows)

    def _build_figures(self, r, o_pairs, p) -> List:
        from rope.viz.summary_panels import plot_cross_summary
        f = plot_cross_summary(r, o_pairs, p)
        return [f] if f else []
