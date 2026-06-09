#!/usr/bin/env python3
"""R-Axis: Target State Source Reliability — WC(V2X) vs WOC(LiDAR)"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from rope.utils import drop_outliers, parse_scenario_name

# Cross-Axis 호환: WC ICQ 컬럼 목록
_ICQ_COLS = ['latency_ms', 'valid_rate', 'sampling_rate_hz', 'sampling_jitter_ms']


class RAxis:
    def __init__(self, output_dir: str):
        self.out    = Path(output_dir)
        self.ts_dir = self.out / 'timeseries'

    def run(self) -> Dict:
        table = self._build_table()
        if table.empty:
            return {'table': table, 'figures': []}
        table.to_csv(self.out / 'r_axis_table.csv', index=False)
        table_clean = drop_outliers(table, _ICQ_COLS)
        return {'table': table_clean, 'figures': self._build_figures(table_clean)}

    def _build_table(self) -> pd.DataFrame:
        rows = []
        for pq in sorted(self.ts_dir.glob('*.parquet')):
            name = pq.stem
            location, stype, num, mode = parse_scenario_name(name)
            try:
                df = pd.read_parquet(pq)
            except Exception:
                continue

            total = len(df)
            duration_s = (df['ts_ns'].max() - df['ts_ns'].min()) / 1e9 \
                if 'ts_ns' in df.columns and total > 1 else np.nan

            # sampling rate / jitter
            if 'ts_ns' in df.columns:
                dt_ms  = df['ts_ns'].diff().dropna() / 1e6
                rate   = 1000 / dt_ms.mean() if dt_ms.mean() > 0 else np.nan
                jitter = dt_ms.std()
            else:
                rate = jitter = np.nan

            row = {
                'scenario': name, 'location': location, 'type': stype,
                'num': num, 'mode': mode, 'n_samples': total,
                'sampling_rate_hz':   round(rate,   2) if np.isfinite(rate)   else np.nan,
                'sampling_jitter_ms': round(jitter, 2) if np.isfinite(jitter) else np.nan,
            }

            if mode == 'WC':
                row.update(self._wc_metrics(df, total, duration_s))
            else:
                row.update(self._woc_metrics(df, total, duration_s))

            rows.append(row)
        return pd.DataFrame(rows)

    def _wc_metrics(self, df, total, duration_s):
        delay = df['delay'].dropna() if 'delay' in df.columns else pd.Series(dtype=float)
        n_valid = len(delay)
        valid_rate = 100 * n_valid / total if total > 0 else np.nan
        return {
            'latency_ms': round(delay.mean(), 2) if len(delay) else np.nan,
            'valid_rate': round(valid_rate, 1),
        }

    def _woc_metrics(self, df, total, duration_s):
        det_valid  = df['best_lidar_enu_x'].notna() if 'best_lidar_enu_x' in df.columns \
                     else pd.Series([False] * total)
        n_det = int(det_valid.sum())
        valid_rate = 100 * n_det / total if total > 0 else np.nan
        update_hz  = n_det / duration_s  if (np.isfinite(duration_s) and duration_s > 0) else np.nan

        ld = df['lidar_delay'].dropna() if 'lidar_delay' in df.columns else pd.Series(dtype=float)
        # filter out 0 (KIAPI placeholder) — already done in pipeline, but defensive
        ld = ld[ld > 0]

        return {
            'latency_ms':  round(ld.mean(), 2) if len(ld) else np.nan,
            'valid_rate':  round(valid_rate, 1),
        }

    def _build_figures(self, table: pd.DataFrame) -> List:
        from rope.viz.summary_panels import plot_r_summary
        f = plot_r_summary(table)
        return [f] if f else []
