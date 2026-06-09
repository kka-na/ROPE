#!/usr/bin/env python3
"""Time-wise O→P: per-timestep V2X position error vs ADS behavior (Spearman, Fisher z-mean)."""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rope.utils import match_key
from rope.modules.uncertainty import compute_residual_timeseries

_P_COLS = [
    ('d_rel',          'D_rel (m)'),
    ('ego_jerk_total', 'Jerk (m/s³)'),
    ('ttc',            'TTC (s)'),
    ('drac',           'DRAC (m/s²)'),
]


def _fisher_mean(rhos: np.ndarray) -> float:
    z = np.arctanh(np.clip(rhos, -0.9999, 0.9999))
    return float(np.tanh(np.nanmean(z)))


class TimewiseAxis:
    """Per-timestep |e_wc_t| vs P-axis timeseries: scenario-level Spearman + Fisher z aggregation."""

    def __init__(self, output_dir: str):
        self.out    = Path(output_dir)
        self.ts_dir = self.out / 'timeseries'

    def run(self) -> Dict:
        pqs      = {p.stem: p for p in self.ts_dir.glob('*.parquet')}
        wc_map   = {match_key(s): p for s, p in pqs.items() if '_wc_'  in s.lower()}
        woc_map  = {match_key(s): p for s, p in pqs.items() if '_woc_' in s.lower()}

        per_rows = []
        pooled   = {col: ([], []) for col, _ in _P_COLS}   # (e_list, p_list)

        for k, wc_p in wc_map.items():
            if k not in woc_map:
                continue
            df_wc  = pd.read_parquet(wc_p)
            df_woc = pd.read_parquet(woc_map[k])
            try:
                ts = compute_residual_timeseries(df_wc, df_woc)
            except Exception:
                continue

            e_wc, v_wc = ts['e_wc'], ts['valid_wc']
            n = len(v_wc)
            mag = np.where(v_wc, np.hypot(e_wc[:, 0], e_wc[:, 1]), np.nan)

            row = {'scenario': wc_p.stem}
            for col, _ in _P_COLS:
                if col not in df_wc.columns:
                    continue
                vals = df_wc[col].values[:n].astype(float)
                mask = np.isfinite(mag) & np.isfinite(vals) & (vals < 1e9)
                pooled[col][0].extend(mag[mask].tolist())
                pooled[col][1].extend(vals[mask].tolist())
                if mask.sum() >= 10:
                    rho, _ = spearmanr(mag[mask], vals[mask])
                    row[col + '_rho'] = float(rho)
            per_rows.append(row)

        per_df = pd.DataFrame(per_rows)

        summary = []
        for col, lbl in _P_COLS:
            rho_col = col + '_rho'
            rhos = per_df[rho_col].dropna().values if rho_col in per_df.columns else np.array([])
            pe, pp = pooled[col]
            p_rho, p_pv = spearmanr(pe, pp) if len(pe) >= 10 else (np.nan, np.nan)
            summary.append({
                'metric':           lbl,
                'n_scenarios':      len(rhos),
                'fisher_z_mean_rho': _fisher_mean(rhos) if len(rhos) else np.nan,
                'pooled_n':         len(pe),
                'pooled_rho':       float(p_rho),
                'pooled_p':         float(p_pv),
            })

        summary_df = pd.DataFrame(summary)
        if not per_df.empty:
            per_df.to_csv(self.out / 'timewise_op_per_scenario.csv', index=False)
        summary_df.to_csv(self.out / 'timewise_op_summary.csv', index=False)
        return {'per_scenario': per_df, 'summary': summary_df}
