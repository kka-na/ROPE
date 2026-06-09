#!/usr/bin/env python3
"""O-Axis: Observability — Observable State Tube (OST)"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from rope.utils import parse_scenario_name, match_key
from rope.modules.uncertainty import compute_ost_single, compute_ost_pair

_O_COLS = ['ost', 'ost_c']


class OAxis:
    def __init__(self, output_dir: str):
        self.out    = Path(output_dir)
        self.ts_dir = self.out / 'timeseries'

    def run(self) -> Dict:
        table = self._build_table()
        pairs = self._build_pairs(table)
        if not pairs.empty:
            pairs.to_csv(self.out / 'o_axis_pairs.csv', index=False)
        return {'table': table, 'pairs': pairs, 'figures': self._build_figures(table, pairs)}

    def _build_table(self) -> pd.DataFrame:
        """OST per parquet file: WC→V2X, WOC→LiDAR."""
        rows = []
        for pq in sorted(self.ts_dir.glob('*.parquet')):
            loc, stype, num, mode = parse_scenario_name(pq.stem)
            if mode not in ('WC', 'WOC'):
                continue
            src_x = 'v2v_enu_x'    if mode == 'WC' else 'best_lidar_enu_x'
            src_y = 'v2v_enu_y'    if mode == 'WC' else 'best_lidar_enu_y'
            src_v = 'v2v_velocity' if mode == 'WC' else 'lidar_velocity'
            src_h = 'v2v_h'        if mode == 'WC' else 'lidar_h'
            try:
                r = compute_ost_single(pd.read_parquet(pq), src_x, src_y, src_v, src_h)
            except Exception:
                continue
            rows.append({
                'scenario':    pq.stem,
                'location':    loc,
                'type':        stype,
                'num':         num,
                'mode':        mode,
                'source':      'V2X' if mode == 'WC' else 'LiDAR',
                'ost':   r['OST'],
                'ost_c': float(np.sqrt(r['C_e'] * r['C_de'])),
            })
        return pd.DataFrame(rows)

    def _build_pairs(self, table: pd.DataFrame) -> pd.DataFrame:
        """Match WC-WOC parquets by key and compute OST / OST_c via compute_ost_pair."""
        if table.empty:
            return pd.DataFrame()
        pqs     = {p.stem: p for p in sorted(self.ts_dir.glob('*.parquet'))}
        wc_pqs  = {match_key(s): (s, p) for s, p in pqs.items() if '_wc_'  in s.lower()}
        woc_pqs = {match_key(s): p       for s, p in pqs.items() if '_woc_' in s.lower()}

        meta = table[table['mode'] == 'WC'].set_index('scenario')[['location', 'type', 'num']]

        rows = []
        for k, (s_wc, wc_p) in wc_pqs.items():
            woc_p = woc_pqs.get(k)
            if woc_p is None:
                continue
            try:
                r = compute_ost_pair(pd.read_parquet(wc_p), pd.read_parquet(woc_p))
            except Exception:
                continue
            wc_r, woc_r = r['WC_V2X'], r['WOC_LiDAR']
            loc   = meta.loc[s_wc, 'location'] if s_wc in meta.index else 'Unknown'
            stype = meta.loc[s_wc, 'type']     if s_wc in meta.index else 'Unknown'
            num   = meta.loc[s_wc, 'num']      if s_wc in meta.index else 0
            rows.append({
                'scenario':   s_wc,
                'location':   loc,
                'type':       stype,
                'num':        num,
                'ost_wc':      wc_r['OST'],
                'ost_woc':     woc_r['OST'],
                'delta_ost':   r['delta_OST'],
                'ost_c_wc':    float(np.sqrt(wc_r['C_e']  * wc_r['C_de'])),
                'ost_c_woc':   float(np.sqrt(woc_r['C_e'] * woc_r['C_de'])),
                'delta_ost_c': float(np.sqrt(wc_r['C_e']  * wc_r['C_de'])
                                    - np.sqrt(woc_r['C_e'] * woc_r['C_de'])),
                # ablation
                'avail_wc':    wc_r['availability'],
                'avail_woc':   woc_r['availability'],
                'Ce_wc':       wc_r['C_e'],
                'Ce_woc':      woc_r['C_e'],
                'Cde_wc':      wc_r['C_de'],
                'Cde_woc':     woc_r['C_de'],
            })
        return pd.DataFrame(rows)

    def _build_figures(self, table: pd.DataFrame, pairs: pd.DataFrame) -> List:
        from rope.viz.summary_panels import plot_o_summary
        f = plot_o_summary(pairs)
        return [f] if f else []
