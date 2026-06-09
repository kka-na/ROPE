#!/usr/bin/env python3
"""ROPE Pipeline — cbf-new/process_data.py 기반"""
import re
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from rope.config import ROPEConfig

LANE_WIDTH = 3.1

# Field-calibrated channel model (bag files: 1014/1015/1031/1102, n=196,266)
_CH_DELAY_MEAN = 48.0   # ms, distance-independent
_CH_DELAY_STD  = 21.8   # ms
_CH_PRR_FAR    = 0.551  # PRR at long range (3GPP LoS extrapolation)
_CH_PRR_NEAR   = 0.848  # PRR at short range (field-measured)
_CH_PRR_D50    = 97.8   # sigmoid inflection distance [m]
_CH_PRR_K      = 5.0    # sigmoid slope [m]


def _prr(d: float) -> float:
    return _CH_PRR_FAR + (_CH_PRR_NEAR - _CH_PRR_FAR) / (1 + np.exp((d - _CH_PRR_D50) / _CH_PRR_K))


def _inject_channel_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Replace simulation's fixed comm params with field-calibrated model (WC sim files only)."""
    df = df.copy()
    n = len(df)
    dist = df['v2v_distance'].values if 'v2v_distance' in df.columns else np.zeros(n)

    # Sample realistic delay (distance-independent)
    delay = rng.normal(_CH_DELAY_MEAN, _CH_DELAY_STD, n).clip(10, 200)

    # Distance-dependent packet drop
    prr_vals = np.array([_prr(float(d)) for d in dist])
    dropped  = rng.random(n) > prr_vals  # True = packet lost

    delay[dropped] = np.nan
    df['v2v_delay'] = delay
    if 'v2v_packet_rate' in df.columns:
        df['v2v_packet_rate'] = prr_vals * 100.0  # calibrated PRR [%] by distance

    # NaN-out all v2v measurement columns for dropped packets
    skip = {'v2v_distance', 'v2v_packet_rate', 'v2v_packet_size',
            'v2v_mbps', 'v2v_rtt', 'v2v_cnt', 'v2v_state',
            'v2v_signal', 'v2v_perf_state', 'v2v_v2x'}
    for col in [c for c in df.columns if c.startswith('v2v_') and c not in skip]:
        df.loc[dropped, col] = np.nan
    return df


def calc_cumulative_dist(x, y):
    dx, dy = np.diff(x), np.diff(y)
    return np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])


def find_signal_idx(df, scenario_type, mode='WC'):
    if scenario_type == 'CLM':
        sig = df['ego_signal'].values
        idx = np.where((sig == 1) | (sig == 2))[0]
    else:
        col = 'target_v2v_signal' if mode == 'WC' else 'target_signal'
        idx = np.where(df[col].values == 7)[0] if col in df.columns else np.array([])
    return idx[0] if len(idx) > 0 else 0


def apply_window(df, scenario_type, mode='WC', wb=20, wa=130):
    cum = calc_cumulative_dist(df['ego_enu_x'].values, df['ego_enu_y'].values)
    df = df.copy()
    df['ego_cumulative_dist'] = cum
    ref = cum[find_signal_idx(df, scenario_type, mode)]
    mask = (cum >= ref - wb) & (cum <= ref + wa)
    return df[mask].reset_index(drop=True), find_signal_idx(df, scenario_type, mode)


def calc_safety_metrics(df):
    dx = df['target_enu_x'].values - df['ego_enu_x'].values
    dy = df['target_enu_y'].values - df['ego_enu_y'].values
    theta = np.radians(df['ego_h'].values)
    x_rel = dx * np.cos(theta) + dy * np.sin(theta)
    y_rel = -dx * np.sin(theta) + dy * np.cos(theta)
    d_rel = np.sqrt(dx**2 + dy**2)
    rel_v = np.abs(df['ego_velocity'].values - df['target_velocity'].values)
    valid = (np.abs(y_rel) < LANE_WIDTH) & (rel_v > 0.01)
    ttc  = np.where(valid, np.abs(x_rel) / np.maximum(rel_v, 0.01), np.inf)
    drac = np.where((ttc > 0.1) & (ttc < np.inf), rel_v**2 / (2 * np.maximum(d_rel, 0.01)), 0)
    return ttc, d_rel, drac, x_rel, y_rel


def calc_jerk(acc, ts_ns):
    dt = np.diff(ts_ns) / 1e9
    dt = np.clip(np.where(dt > 0.001, dt, 1.0 / 14.0), 0.01, 1.0)
    return np.concatenate([[0], np.diff(acc) / dt])


_LIDAR_GATE_M2 = 3.1 ** 2  # lane-width gate (squared)


def _best_lidar(df) -> tuple:
    """Return (x, y, velocity, heading, delay_ms) of the LiDAR detection closest to target per row.
    NaN if no detection within 5 m gate. delay=0 treated as not recorded → NaN."""
    n = len(df)
    bx, by = np.full(n, np.nan), np.full(n, np.nan)
    bv, bh = np.full(n, np.nan), np.full(n, np.nan)
    bd      = np.full(n, np.nan)
    tx = df.get('target_enu_x', pd.Series([np.nan]*n)).values
    ty = df.get('target_enu_y', pd.Series([np.nan]*n)).values
    for i in range(n):
        if not (np.isfinite(tx[i]) and np.isfinite(ty[i])):
            continue
        best_d, best_x, best_y = _LIDAR_GATE_M2, np.nan, np.nan
        best_v, best_h, best_delay = np.nan, np.nan, np.nan
        for k in range(10):
            xc, yc = f'lidar_{k}_enu_x', f'lidar_{k}_enu_y'
            if xc not in df.columns:
                break
            lx, ly = df[xc].iloc[i], df[yc].iloc[i]
            if not (np.isfinite(lx) and np.isfinite(ly)):
                continue
            d = (lx - tx[i])**2 + (ly - ty[i])**2
            if d < best_d:
                best_d, best_x, best_y = d, lx, ly
                best_v = df[f'lidar_{k}_velocity'].iloc[i] if f'lidar_{k}_velocity' in df.columns else np.nan
                best_h = df[f'lidar_{k}_h'].iloc[i]        if f'lidar_{k}_h'        in df.columns else np.nan
                dc = f'lidar_{k}_lidar_delay'
                raw = df[dc].iloc[i] if dc in df.columns else np.nan
                if pd.notna(raw) and float(raw) > 0:
                    v = float(raw)
                    if v < 1.0: v *= 1000.0   # seconds → ms
                    best_delay = v if v >= 1.0 else np.nan  # <1ms = placeholder
                else:
                    best_delay = np.nan
        bx[i], by[i] = best_x, best_y
        bv[i], bh[i] = best_v, best_h
        bd[i]        = best_delay
    return bx, by, bv, bh, bd


def calc_timeseries(df, scenario_name):
    ts, n = df['ts_ns'].values, len(df)
    low = scenario_name.lower()
    location = 'KIAPI' if ('kiapi' in low or low.startswith('1102')) else 'Yeongjong'
    bx, by, lidar_v, lidar_h, lidar_delay = _best_lidar(df)

    result = pd.DataFrame({
        'ts_ns': ts, 'location': [location] * n,
        'ego_state':          df.get('ego_state', pd.Series([np.nan]*n)).values,
        'ego_cumulative_dist': df['ego_cumulative_dist'].values,
        'ego_enu_x': df['ego_enu_x'].values, 'ego_enu_y': df['ego_enu_y'].values,
        'ego_h': df['ego_h'].values,
        'target_enu_x': df['target_enu_x'].values, 'target_enu_y': df['target_enu_y'].values,
        'v2v_enu_x':         df.get('v2v_enu_x',      pd.Series([np.nan]*n)).values,
        'v2v_enu_y':         df.get('v2v_enu_y',      pd.Series([np.nan]*n)).values,
        'v2v_velocity':      df.get('v2v_velocity',   pd.Series([np.nan]*n)).values,
        'v2v_h':             df.get('v2v_h',           pd.Series([np.nan]*n)).values,
        'best_lidar_enu_x':  bx,
        'best_lidar_enu_y':  by,
        'lidar_velocity':    lidar_v,
        'lidar_h':           lidar_h,
        'lidar_delay':       lidar_delay,
        'target_velocity':   df.get('target_velocity', pd.Series([np.nan]*n)).values,
        'target_h':          df.get('target_h',        pd.Series([np.nan]*n)).values,
        'delay':       df.get('v2v_delay',       pd.Series([np.nan] * n)).values,
        'prr':         df.get('v2v_packet_rate', pd.Series([np.nan] * n)).values,
        'packet_size': df.get('v2v_packet_size', pd.Series([np.nan] * n)).values,
    })
    ttc, d_rel, drac, x_rel, y_rel = calc_safety_metrics(df)
    result['ttc'], result['d_rel'], result['drac'] = ttc, d_rel, drac
    result['x_rel'], result['y_rel'] = x_rel, y_rel
    for prefix, a_lon_col, a_lat_col, yaw_col in [
        ('ego',    'ego_longitudinal_acc',    'ego_lateral_acc',    'ego_yaw_rate'),
        ('target', 'target_longitudinal_acc', 'target_lateral_acc', 'target_yaw_rate'),
    ]:
        a_lon = df.get(a_lon_col, pd.Series([0] * n)).values
        a_lat = df.get(a_lat_col, pd.Series([0] * n)).values
        result[f'{prefix}_a_lon'], result[f'{prefix}_a_lat'] = a_lon, a_lat
        j_lon, j_lat = calc_jerk(a_lon, ts), calc_jerk(a_lat, ts)
        result[f'{prefix}_jerk_lon'], result[f'{prefix}_jerk_lat'] = j_lon, j_lat
        result[f'{prefix}_jerk_total'] = np.sqrt(j_lon**2 + j_lat**2)
        result[f'{prefix}_yaw_rate'] = df.get(yaw_col, pd.Series([0] * n)).values
    result['ego_velocity'] = df['ego_velocity'].values
    # h_lidar/h_v2v/delta_h 제거 — OST는 o_axis에서 paired 계산
    return result


def _rms(arr):
    arr = arr[np.isfinite(arr)]
    return np.sqrt(np.mean(arr**2)) if len(arr) > 0 else np.nan


def _zcr(arr):
    arr = arr[np.isfinite(arr)]
    return int(np.sum(np.diff(np.sign(arr)) != 0)) if len(arr) > 1 else 0


def calc_merge_time(df, ref_idx):
    if ref_idx >= len(df): return np.nan, -1
    ts, heading = df['ts_ns'].values, df['ego_h'].values
    hz = int(round(1e9 / np.median(np.diff(ts[:100])))) if len(ts) > 1 else 14
    start_h = heading[ref_idx]
    for i in range(ref_idx + hz, min(ref_idx + hz * 10, len(heading))):
        if abs(heading[i] - start_h) >= 3:
            return (ts[i] - ts[ref_idx]) / 1e9, i
    return np.nan, -1


def calc_avoid_distance(df):
    ego_arr = np.where(df['ego_signal'].values == 7)[0] if 'ego_signal' in df.columns else np.array([])
    ego_idx = ego_arr[0] if len(ego_arr) > 0 else None
    target_idx = None
    for col in ['target_signal', 'target_v2v_signal']:
        if col in df.columns:
            found = np.where(df[col].values == 7)[0]
            if len(found) > 0:
                target_idx = found[0]; break
    if ego_idx is None and target_idx is None:
        return np.nan, np.nan, -1
    ref = ego_idx if ego_idx is not None else target_idx
    tx, ty = df.iloc[ref]['target_enu_x'], df.iloc[ref]['target_enu_y']
    emv_x = emv_y = None
    if 'dangerous_enu_x' in df.columns and pd.notna(df.iloc[ref]['dangerous_enu_x']):
        emv_x, emv_y = df.iloc[ref]['dangerous_enu_x'], df.iloc[ref]['dangerous_enu_y']
    if emv_x is None:
        cands = []
        for col in df.columns:
            if 'danger' in col and col.endswith('_danger') and df.iloc[ref].get(col, 0) == 1:
                px = col.rsplit('_', 1)[0]
                ox, oy = df.iloc[ref].get(f'{px}_enu_x'), df.iloc[ref].get(f'{px}_enu_y')
                if pd.notna(ox) and pd.notna(oy) and np.sqrt((ox - tx)**2 + (oy - ty)**2) > 2.0:
                    cands.append((ox, oy))
        if cands:
            emv_x, emv_y = cands[-1] if len(cands) >= 2 else cands[0]
    if emv_x is None:
        return np.nan, np.nan, -1
    def _d(r, xk, yk): return np.sqrt((df.iloc[r][xk] - emv_x)**2 + (df.iloc[r][yk] - emv_y)**2)
    ego_d = _d(ego_idx, 'ego_enu_x', 'ego_enu_y') if ego_idx is not None else np.nan
    tgt_d = _d(target_idx, 'target_enu_x', 'target_enu_y') if target_idx is not None else np.nan
    td = np.sqrt((df['target_enu_x'].values - emv_x)**2 + (df['target_enu_y'].values - emv_y)**2)
    valid = ~np.isnan(td)
    action_idx = np.where(valid)[0][np.argmin(td[valid])] if valid.any() else -1
    return ego_d, tgt_d, action_idx


def calc_data_quality(df):
    ts = df['ts_ns'].values
    dt_ms = np.diff(ts) / 1e6
    med = np.median(dt_ms) if len(dt_ms) > 0 else 0
    sr = 1000 / med if med > 0 else np.nan
    jitter = np.std(dt_ms) if len(dt_ms) > 0 else np.nan
    sync_error = np.nan
    if 'target_ts' in df.columns:
        tts = df['target_ts'].values
        valid = ~np.isnan(tts) & (tts > 0)
        if valid.sum() > 0:
            sd = np.abs(ts[valid] - tts[valid]) / 1e6
            sync_error = np.mean(np.where(sd > 20, 0, sd))
    return {
        'sampling_rate_hz': sr, 'sampling_jitter_ms': jitter, 'sync_error_ms': sync_error,
        'n_samples': len(ts), 'duration_s': (ts[-1] - ts[0]) / 1e9 if len(ts) > 1 else 0,
    }


def calc_scenario_metrics(df_full, df_w, ts_df, name, stype, snum, mode, ref_idx):
    import warnings as _w
    m = {'scenario': name, 'mode': mode, 'type': stype, 'num': snum}
    m.update(calc_data_quality(df_w))
    for col, out in [('delay', 'delay_mean'), ('prr', 'prr_mean'), ('packet_size', 'pkt_size_mean')]:
        with _w.catch_warnings():
            _w.simplefilter('ignore', RuntimeWarning)
            m[out] = np.nanmean(ts_df[col].values) if col in ts_df else np.nan
    ttc = ts_df['ttc'].values
    ttc_v = ttc[np.isfinite(ttc)]
    m['ttc_violation_2s'] = 100 * np.sum(ttc_v < 2) / len(ttc_v) if len(ttc_v) > 0 else np.nan
    m['ttc_violation_3s'] = 100 * np.sum(ttc_v < 3) / len(ttc_v) if len(ttc_v) > 0 else np.nan
    drac_v = ts_df['drac'].values if 'drac' in ts_df.columns else np.array([])
    m['drac_critical_ratio'] = 100 * np.sum(drac_v > 3.4) / len(drac_v) if len(drac_v) > 0 else np.nan
    x_rel, y_rel = ts_df['x_rel'].values, ts_df['y_rel'].values
    overlap = np.abs(y_rel) < LANE_WIDTH
    m['d_min'] = float(np.nanmin(np.abs(x_rel[overlap]))) if overlap.any() else np.nan
    for prefix in ['ego', 'target']:
        m[f'{prefix}_a_lon_rms'] = _rms(ts_df[f'{prefix}_a_lon'].values)
        m[f'{prefix}_a_lat_rms'] = _rms(ts_df[f'{prefix}_a_lat'].values)
        m[f'{prefix}_jerk_rms']  = _rms(ts_df[f'{prefix}_jerk_total'].values)
        m[f'{prefix}_jerk_peak'] = np.nanmax(np.abs(ts_df[f'{prefix}_jerk_total'].values))
        m[f'{prefix}_yaw_rms']   = _rms(ts_df[f'{prefix}_yaw_rate'].values)
        m[f'{prefix}_yaw_zcr']   = _zcr(ts_df[f'{prefix}_yaw_rate'].values)
    m['tct']    = (df_w['ts_ns'].iloc[-1] - df_w['ts_ns'].iloc[0]) / 1e9
    m['v_mean'] = np.mean(df_w['ego_velocity'].values)
    m['v_std']  = np.std(df_w['ego_velocity'].values)
    action_idx = -1
    if stype == 'CLM':
        m['t_merge'], action_idx = calc_merge_time(df_full, ref_idx)
        m['merge_success'] = 0.0 if np.isnan(m['t_merge']) else 1.0
        m['avoid_success'] = np.nan
    else:
        m['t_merge'] = np.nan
        m['merge_success'] = np.nan
        ref_ts = df_full['ts_ns'].iloc[ref_idx] if ref_idx < len(df_full) else None
        post = (ts_df['ts_ns'].values >= ref_ts) if ref_ts is not None else np.ones(len(ts_df), bool)
        y_post = ts_df['y_rel'].values[post]
        m['avoid_success'] = float(np.mean(np.abs(y_post) > LANE_WIDTH)) if len(y_post) > 0 else np.nan
    for col, src in [('ego_lc_response_ms',   'ego_lc_response_ms'),
                     ('target_lc_response_ms', 'target_lc_response_ms')]:
        v = df_full[src].dropna() if src in df_full.columns else pd.Series(dtype=float)
        m[col] = float(v.iloc[0]) if len(v) else np.nan
    if 'target_emv_dist_at_emg' in df_full.columns:
        v = df_full['target_emv_dist_at_emg'].dropna()
        m['avoid_dist_tgt'] = float(v.iloc[0]) if len(v) else np.nan
    elif stype != 'CLM':
        _, tgt_d, _ = calc_avoid_distance(df_full)
        m['avoid_dist_tgt'] = tgt_d
    else:
        m['avoid_dist_tgt'] = np.nan
    return m, action_idx


def _parse_filename(fname):
    name = fname.stem
    stype = 'CLM' if 'CLM' in name.upper() else 'ETrA' if 'ETRA' in name.upper() else 'Unknown'
    mode  = ('WC'  if ('_WC_'  in name or name.endswith('_WC'))  else
             'WOC' if ('_WOC_' in name or name.endswith('_WOC')) else 'Unknown')
    match = re.search(r'(CLM|ETrA)(\d+)', name, re.IGNORECASE)
    v_match = re.search(r'(\d+)kmh', name, re.IGNORECASE)
    v_nominal = float(v_match.group(1)) / 3.6 if v_match else np.nan
    return name, stype, int(match.group(2)) if match else 0, mode, v_nominal


_RNG = np.random.default_rng(42)


def _process_file(csv_path, output_dir, wb=20, wa=130):
    df = pd.read_csv(csv_path)
    drop_cols = [c for c in ['ts_ns', 'ego_enu_x', 'ego_enu_y', 'ego_h'] if c in df.columns]
    df = df.dropna(subset=drop_cols).reset_index(drop=True)
    if df.empty:
        return None
    for c in ['ego_enu_x', 'ego_enu_y', 'ego_h', 'ego_velocity']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    name, stype, snum, mode, v_nominal = _parse_filename(csv_path)
    v_nom_kmh = v_nominal * 3.6 if np.isfinite(v_nominal) else 30.0  # FIELD = 30km/h
    vn = v_nominal if (np.isfinite(v_nominal) and v_nominal > 0) else 30.0 / 3.6
    scale = max(v_nom_kmh / 30.0, 1.0)
    wb_s, wa_s = wb * scale, wa * scale
    df_w, ref_idx = apply_window(df, stype, mode, wb_s, wa_s)
    if len(df_w) == 0:
        return None
    ts_df = calc_timeseries(df_w, name)
    metrics, action_idx = calc_scenario_metrics(df, df_w, ts_df, name, stype, snum, mode, ref_idx)
    metrics['v_nominal'] = vn
    metrics['tct_norm']         = metrics['tct']         * vn / (wb_s + wa_s)
    metrics['v_mean_norm']      = metrics['v_mean']      / vn
    metrics['v_std_norm']       = metrics['v_std']       / vn
    metrics['drac_norm']        = metrics['drac_critical_ratio'] / vn
    metrics['ego_yaw_rms_norm'] = metrics['ego_yaw_rms'] / vn
    if action_idx >= 0:
        cum_full = calc_cumulative_dist(df['ego_enu_x'].values, df['ego_enu_y'].values)
        ts_df['action_start_dist'] = cum_full[action_idx] - ts_df['ego_cumulative_dist'].iloc[0]
    else:
        ts_df['action_start_dist'] = np.nan
    ts_out = output_dir / 'timeseries' / f'{name}.parquet'
    ts_out.parent.mkdir(parents=True, exist_ok=True)
    ts_df.to_parquet(ts_out, index=False)
    return metrics


class ROPEPipeline:
    def __init__(self, config: ROPEConfig = None, progress_cb: Optional[Callable] = None):
        self.config = config or ROPEConfig()
        self.progress_cb = progress_cb

    def _emit(self, msg: str, pct: int = -1):
        if self.progress_cb:
            self.progress_cb(msg, pct)
        else:
            print(f"[{pct:3d}%] {msg}" if pct >= 0 else msg)

    def run_postprocessing(self, csv_dir: str, output_dir: str) -> str:
        csv_dir, output_dir = Path(csv_dir), Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_res = str(output_dir.resolve())
        csv_files = sorted(
            f for f in csv_dir.rglob('*.csv')
            if not str(f.resolve()).startswith(out_res) and 'rope_output' not in f.parts
        )
        if not csv_files:
            raise FileNotFoundError(f"No CSV files in {csv_dir}")

        all_metrics = []
        for i, f in enumerate(csv_files):
            self._emit(f"[{i+1}/{len(csv_files)}] {f.name}", int(5 + 45 * i / len(csv_files)))
            try:
                m = _process_file(f, output_dir, self.config.window_before_m, self.config.window_after_m)
                if m:
                    all_metrics.append(m)
            except Exception as e:
                self._emit(f"  ❌ {f.name}: {e}")

        summary_out = output_dir / 'sequences' / 'scenario_metrics.csv'
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        if all_metrics:
            pd.DataFrame(all_metrics).to_csv(summary_out, index=False)
        self._emit(f"Saved {len(all_metrics)} scenarios → {summary_out}", 50)
        return str(summary_out)

    def run(self, csv_dir: str, output_dir: str, axes: Optional[List[str]] = None) -> Dict:
        axes = axes or self.config.axes
        output_dir = Path(output_dir)

        self._emit("Step 1/4  Processing CSVs...", 2)
        metrics_path = self.run_postprocessing(csv_dir, str(output_dir))
        results: Dict = {'metrics_path': metrics_path}

        if 'R' in axes:
            self._emit("Step 2/4  R-Axis (Reliability)...", 52)
            from rope.axes.r_axis import RAxis
            results['R'] = RAxis(str(output_dir)).run()

        if 'O' in axes:
            self._emit("Step 3/4  O-Axis (Observability)...", 68)
            from rope.axes.o_axis import OAxis
            results['O'] = OAxis(str(output_dir)).run()

        if 'P' in axes:
            self._emit("Step 4/4  P-Axis (Performance)...", 82)
            from rope.axes.p_axis import PAxis
            results['P'] = PAxis(str(output_dir), paired=self.config.paired).run()

        if len([a for a in axes if a in ('R', 'O', 'P')]) >= 2:
            self._emit("Step 5/5  Cross-Axis...", 93)
            from rope.axes.cross_axis import CrossAxis
            results['Cross'] = CrossAxis(str(output_dir)).run(
                r_table=results.get('R', {}).get('table'),
                p_table=results.get('P', {}).get('scenarios'),
            )

        if 'T' in axes:
            self._emit("Time-wise O→P...", 96)
            from rope.axes.timewise_axis import TimewiseAxis
            results['T'] = TimewiseAxis(str(output_dir)).run()

        try:
            self._emit("Saving figures...", 97)
            from rope.viz.export import run_export
            run_export(str(output_dir))
        except Exception as e:
            self._emit(f"  ⚠️ Export failed: {e}")

        self._emit("✅ Done", 100)
        return results

    def run_from_existing(self, output_dir: str, axes: Optional[List[str]] = None) -> Dict:
        """Rerun axis analysis from existing parquets / scenario_metrics.csv (no raw CSV needed)."""
        axes = axes or self.config.axes
        output_dir = Path(output_dir)
        results: Dict = {}

        if 'R' in axes:
            self._emit("R-Axis (Reliability)...", 20)
            from rope.axes.r_axis import RAxis
            results['R'] = RAxis(str(output_dir)).run()

        if 'O' in axes:
            self._emit("O-Axis (Observability)...", 40)
            from rope.axes.o_axis import OAxis
            results['O'] = OAxis(str(output_dir)).run()

        if 'P' in axes:
            self._emit("P-Axis (Performance)...", 60)
            from rope.axes.p_axis import PAxis
            results['P'] = PAxis(str(output_dir), paired=self.config.paired).run()

        if len([a for a in axes if a in ('R', 'O', 'P')]) >= 2:
            self._emit("Cross-Axis...", 80)
            from rope.axes.cross_axis import CrossAxis
            results['Cross'] = CrossAxis(str(output_dir)).run(
                r_table=results.get('R', {}).get('table'),
                p_table=results.get('P', {}).get('scenarios'),
            )

        if 'T' in axes:
            self._emit("Time-wise O→P...", 92)
            from rope.axes.timewise_axis import TimewiseAxis
            results['T'] = TimewiseAxis(str(output_dir)).run()

        try:
            self._emit("Saving figures...", 95)
            from rope.viz.export import run_export
            run_export(str(output_dir))
        except Exception as e:
            self._emit(f"⚠️ Export failed: {e}")

        self._emit("✅ Done", 100)
        return results

    def load(self, output_dir: str, axes: Optional[List[str]] = None) -> Dict:
        axes = axes or self.config.axes
        out = Path(output_dir)
        results: Dict = {}

        def _read(path):
            try: return pd.read_csv(path)
            except Exception: return pd.DataFrame()

        if 'R' in axes:
            from rope.axes.r_axis import RAxis
            table = _read(out / 'r_axis_table.csv')
            figs = RAxis(str(out))._build_figures(table) if not table.empty else []
            results['R'] = {'table': table, 'figures': figs}

        if 'O' in axes:
            from rope.axes.o_axis import OAxis
            table = _read(out / 'o_axis_table.csv')
            pairs = _read(out / 'o_axis_pairs.csv')
            figs  = OAxis(str(out))._build_figures(table, pairs)
            results['O'] = {'table': table, 'pairs': pairs, 'figures': figs}

        if 'P' in axes:
            from rope.axes.p_axis import PAxis
            scenarios  = _read(out / 'p_axis_scenarios.csv')
            stat_table = _read(out / 'p_axis_table.csv')
            figs = PAxis(str(out), paired=self.config.paired)._build_figures(scenarios, stat_table)
            results['P'] = {'table': stat_table, 'scenarios': scenarios, 'figures': figs}

        if len([a for a in axes if a in ('R', 'O', 'P')]) >= 2:
            from rope.axes.cross_axis import CrossAxis
            corr_df  = _read(out / 'cross_axis_correlation.csv')
            o_pairs  = _read(out / 'o_axis_pairs.csv')
            figs = CrossAxis(str(out))._build_figures(
                results.get('R', {}).get('table', pd.DataFrame()),
                o_pairs,
                results.get('P', {}).get('scenarios', pd.DataFrame()),
            )
            results['Cross'] = {'table': corr_df, 'figures': figs}

        return results
