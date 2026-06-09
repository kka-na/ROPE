#!/usr/bin/env python3
"""Post-process Carla CSVs: LC response time (CLM) + EmV distance at emergency (ETrA)."""
import os, math, glob
import numpy as np
import pandas as pd

DATA_DIR = os.path.expanduser('~/Documents/Dataset/CCAVT/ROPE/data/SIMULATION')
HZ = 10

_Y1, _Y2, _Y3, _Y4 = -237.65, -241.15, -244.65, -248.15
EMV_POS = {
    'ETrA1': (305.0, _Y1),
    'ETrA2': (305.0, _Y2),
    'ETrA3': (305.0, _Y1),
    'ETrA4': (305.0, (_Y3 + _Y4) / 2),
}
NEW_COLS = ['ego_lc_response_ms', 'target_lc_response_ms',
            'ego_emv_dist_at_emg', 'target_emv_dist_at_emg']


def _dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _first_heading_change(h, ref_pos, threshold=3.0):
    """Return frame index where |heading - start_h| > threshold within [+1s, +10s]."""
    start_h = float(h[ref_pos])
    for i in range(ref_pos + HZ, min(ref_pos + 10 * HZ, len(h))):
        val = h[i]
        if np.isnan(val):
            continue
        dh = abs(((float(val) - start_h + 180) % 360) - 180)
        if dh > threshold:
            return i
    return None


def _lc_response(df, sig_col, h_col):
    if sig_col not in df.columns or h_col not in df.columns:
        return float('nan')
    sig = df[sig_col].values
    h   = df[h_col].values
    ts  = df['ts_ns'].values
    refs = np.where(np.isin(sig, [1, 2]))[0]
    if not len(refs):
        return float('nan')
    ref   = int(refs[0])
    frame = _first_heading_change(h, ref)
    if frame is None:
        return float('nan')
    return float(ts[frame] - ts[ref]) / 1e6


def compute_clm(df):
    return {
        'ego_lc_response_ms':     _lc_response(df, 'ego_signal',    'ego_h'),
        'target_lc_response_ms':  _lc_response(df, 'target_signal', 'target_h'),
        'ego_emv_dist_at_emg':    float('nan'),
        'target_emv_dist_at_emg': float('nan'),
    }


def compute_etra(df, scenario):
    nan = float('nan')
    emv_x, emv_y = EMV_POS.get(scenario, (305.0, _Y1))

    # ego: first lidar_0_danger == 1
    ego_dist = nan
    ref_pos  = None
    if 'lidar_0_danger' in df.columns:
        hits = np.where(df['lidar_0_danger'].values == 1)[0]
        if len(hits):
            ref_pos = int(hits[0])
            ex = float(df['ego_enu_x'].values[ref_pos])
            ey = float(df['ego_enu_y'].values[ref_pos])
            ego_dist = _dist(ex, ey, emv_x, emv_y)

    # target: first target_signal == 7 (EMERGENCY_CHANGE)
    tgt_dist = nan
    if 'target_signal' in df.columns and 'target_enu_x' in df.columns:
        hits = np.where(df['target_signal'].values == 7)[0]
        if len(hits):
            f = int(hits[0])
            tgt_dist = _dist(float(df['target_enu_x'].values[f]),
                             float(df['target_enu_y'].values[f]), emv_x, emv_y)

    return {
        'ego_lc_response_ms':     nan,
        'target_lc_response_ms':  nan,
        'ego_emv_dist_at_emg':    ego_dist,
        'target_emv_dist_at_emg': tgt_dist,
    }


def main():
    csvs = sorted(glob.glob(os.path.join(DATA_DIR, 'carla_*.csv')))
    total = len(csvs)
    print(f'Processing {total} CSV files...')
    ok = err = skip = 0
    for idx, path in enumerate(csvs, 1):
        fname  = os.path.basename(path)
        parts  = fname.replace('.csv', '').split('_')
        scenario = parts[1] if len(parts) > 1 else ''
        print(f'\r  [{idx:4}/{total}] {fname[:70]:<70}', end='', flush=True)
        try:
            df = pd.read_csv(path)
            if scenario.startswith('CLM'):
                metrics = compute_clm(df)
            elif scenario.startswith('ETrA'):
                metrics = compute_etra(df, scenario)
            else:
                skip += 1
                continue
            for col, val in metrics.items():
                df[col] = val
            df.to_csv(path, index=False)
            ok += 1
        except Exception as e:
            print(f'\n  ERR {fname}: {e}')
            err += 1

    print(f'\nDone.  OK={ok}  ERR={err}  SKIP={skip}')


if __name__ == '__main__':
    main()
