#!/usr/bin/env python3
"""Shared utilities for ROPE axes."""
import re
import numpy as np
import pandas as pd


def parse_scenario_name(name: str):
    """Parse scenario filename → (location, stype, num, mode)."""
    n = name.lower()
    location = ('KIAPI'      if 'kiapi'     in n else
                'Yeongjong'  if 'yeongjong' in n else
                'CARLA'      if 'carla'     in n else 'Unknown')
    stype = 'CLM' if 'clm' in n else 'ETrA' if 'etra' in n else 'Unknown'
    mode  = 'WC' if '_wc_' in n else 'WOC' if '_woc_' in n else 'Unknown'
    m = re.search(r'(clm|etra)(\d+)', n)
    return location, stype, int(m.group(2)) if m else 0, mode


def match_key(name: str) -> str:
    """Strip WC/WOC tag and trailing timestamp for pair matching."""
    k = re.sub(r'[_](wc|woc)[_]', '_', name, flags=re.IGNORECASE)
    return re.sub(r'_\d{8,}$', '', k).lower()


def drop_outliers(df: pd.DataFrame, cols, k: float = 1.5,
                  group_cols=('mode', 'type')) -> pd.DataFrame:
    """Replace per-column outliers with NaN using IQR method, grouped by mode+type."""
    df = df.copy()
    gcols = [c for c in (group_cols if hasattr(group_cols, '__iter__') and not isinstance(group_cols, str) else [group_cols]) if c in df.columns]
    groups = df.groupby(gcols).groups if gcols else {'_all': df.index}
    for col in cols:
        if col not in df.columns:
            continue
        for _, idx in groups.items():
            valid = df.loc[idx, col].dropna()
            if len(valid) < 4:
                continue
            q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - k * iqr, q3 + k * iqr
            outliers = valid[(valid < lo) | (valid > hi)].index
            df.loc[outliers, col] = np.nan
    return df
