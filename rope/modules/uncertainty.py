#!/usr/bin/env python3
"""OST (Observable State Tube) — O-axis uncertainty metric."""
import numpy as np
import pandas as pd
from typing import Dict, Optional


def _ego_frame(obj_x, obj_y, ego_x, ego_y, ego_h_deg):
    psi = np.deg2rad(ego_h_deg)
    dx, dy = obj_x - ego_x, obj_y - ego_y
    return (np.cos(psi)*dx + np.sin(psi)*dy,
           -np.sin(psi)*dx + np.cos(psi)*dy)



def _vel_ego_frame(speed, heading_deg, ego_h_deg):
    """Scalar speed + heading → ego-frame (vx, vy)."""
    psi   = np.deg2rad(ego_h_deg)
    theta = np.deg2rad(heading_deg)
    vx_g  = speed * np.cos(theta)
    vy_g  = speed * np.sin(theta)
    return (np.cos(psi)*vx_g + np.sin(psi)*vy_g,
           -np.sin(psi)*vx_g + np.cos(psi)*vy_g)


def _compactness(R: np.ndarray) -> float:
    d   = R.shape[0]
    det = np.linalg.det(np.eye(d) + R)
    return float(det ** (-1/d)) if det > 0 else 0.0


_LIDAR_GATE_M2 = 3.1 ** 2  # lane-width gate (squared)


def _get_best_lidar(df) -> tuple:
    """Fallback: compute best_lidar from lidar_i columns. NaN if no detection within 5 m gate.
    Returns (x, y, velocity, heading)."""
    n = len(df)
    bx, by = np.full(n, np.nan), np.full(n, np.nan)
    bv, bh = np.full(n, np.nan), np.full(n, np.nan)
    tx = df['target_enu_x'].values if 'target_enu_x' in df.columns else np.full(n, np.nan)
    ty = df['target_enu_y'].values if 'target_enu_y' in df.columns else np.full(n, np.nan)
    for i in range(n):
        if not (np.isfinite(tx[i]) and np.isfinite(ty[i])):
            continue
        best_d = _LIDAR_GATE_M2
        for k in range(10):
            xc, yc = f'lidar_{k}_enu_x', f'lidar_{k}_enu_y'
            if xc not in df.columns:
                break
            lx, ly = df[xc].iloc[i], df[yc].iloc[i]
            if np.isfinite(lx) and np.isfinite(ly):
                d = (lx - tx[i])**2 + (ly - ty[i])**2
                if d < best_d:
                    best_d, bx[i], by[i] = d, lx, ly
                    vc, hc = f'lidar_{k}_velocity', f'lidar_{k}_h'
                    bv[i] = df[vc].iloc[i] if vc in df.columns else np.nan
                    bh[i] = df[hc].iloc[i] if hc in df.columns else np.nan
    return bx, by, bv, bh


def _residuals(df, src_x, src_y, src_vel=None, src_h_arr=None) -> np.ndarray:
    """Compute ego-frame residuals (GT - source).
    If src_vel/src_h_arr given (V2X): 4D [Δx, Δy, Δvx, Δvy].
    If None (LiDAR): velocity estimated from consecutive Δpos/Δt → 4D when target_velocity/h in df.
    Falls back to 2D when kinematic GT columns absent.
    """
    n = len(df)
    has_vel = ('target_velocity' in df.columns and 'target_h' in df.columns)
    ndim  = 4 if has_vel else 2
    res   = np.full((n, ndim), np.nan)
    gt_x  = df['target_enu_x'].values
    gt_y  = df['target_enu_y'].values
    ego_x = df['ego_enu_x'].values
    ego_y = df['ego_enu_y'].values
    ego_h = df['ego_h'].values
    gt_v  = df['target_velocity'].values if has_vel else None
    gt_ha = df['target_h'].values        if has_vel else None
    ts    = df['ts_ns'].values            if 'ts_ns' in df.columns else None

    eff_x, eff_y = src_x.copy(), src_y.copy()

    for i in range(n):
        if not np.isfinite([eff_x[i], eff_y[i], gt_x[i], gt_y[i],
                            ego_x[i], ego_y[i], ego_h[i]]).all():
            continue
        gt_rx,  gt_ry  = _ego_frame(gt_x[i],  gt_y[i],  ego_x[i], ego_y[i], ego_h[i])
        src_rx, src_ry = _ego_frame(eff_x[i], eff_y[i], ego_x[i], ego_y[i], ego_h[i])
        res[i, :2] = [gt_rx - src_rx, gt_ry - src_ry]

        if ndim < 4 or not np.isfinite([gt_v[i], gt_ha[i]]).all():
            continue
        gt_vx_e, gt_vy_e = _vel_ego_frame(gt_v[i], gt_ha[i], ego_h[i])

        if src_vel is not None:
            # V2X: direct velocity measurement
            if not np.isfinite([src_vel[i], src_h_arr[i]]).all():
                continue
            sv_e, sh_e = _vel_ego_frame(src_vel[i], src_h_arr[i], ego_h[i])
        else:
            # LiDAR: estimate velocity from consecutive positions
            if i == 0 or not np.isfinite([src_x[i-1], src_y[i-1]]).all():
                continue
            dt = float((ts[i] - ts[i-1]) / 1e9) if ts is not None else 1/14
            if dt < 0.001:
                continue
            psi   = np.deg2rad(ego_h[i])
            vx_g  = (src_x[i] - src_x[i-1]) / dt
            vy_g  = (src_y[i] - src_y[i-1]) / dt
            sv_e  = np.cos(psi)*vx_g + np.sin(psi)*vy_g
            sh_e  = -np.sin(psi)*vx_g + np.cos(psi)*vy_g

        res[i, 2:] = [gt_vx_e - sv_e, gt_vy_e - sh_e]
    return res


def _ost_from_res(res, window, Se_inv, Sde_inv) -> Dict:
    """Compute OST components given residuals and scale matrices."""
    W     = int(window.sum())
    valid = window & np.all(np.isfinite(res), axis=1)
    Vm    = int(valid.sum())

    if Vm == 0:
        return {'OST': 0.0, 'availability': 0.0, 'C_e': 0.0, 'C_de': 0.0,
                'valid': 0, 'window': W}

    a = Vm / W

    # Accuracy compactness
    r = (Se_inv @ res[valid].T).T
    R_e = (r.T @ r) / Vm
    C_e = _compactness(R_e)

    # Temporal compactness (adjacent valid rows only)
    idx = np.where(valid)[0]
    de_list = [res[idx[k]] - res[idx[k-1]]
               for k in range(1, len(idx)) if idx[k] == idx[k-1] + 1]

    if len(de_list) >= 1 and Sde_inv is not None:
        de_arr = np.array(de_list)
        dr = (Sde_inv @ de_arr.T).T
        R_de = (dr.T @ dr) / len(de_list)
        C_de = _compactness(R_de)
    else:
        C_de = 0.0

    OST = float((a * C_e * C_de) ** (1/3)) if (C_e > 0 and C_de > 0) else 0.0
    return {'OST': OST, 'availability': a, 'C_e': C_e, 'C_de': C_de,
            'valid': Vm, 'window': W}


_PHYS_SCALE_INV = np.diag([1.0, 1.0, 1.0, 1.0])   # σ = [1 m, 1 m, 1 m/s, 1 m/s]


def compute_ost_single(df: pd.DataFrame, src_x_col: str, src_y_col: str,
                       src_v_col: str = None, src_h_col: str = None) -> Dict:
    """OST for one file using fixed physical scale σ = [1m, 1m, 1m/s, 1m/s]."""
    win = (df['ego_state'].values == 1) if 'ego_state' in df.columns \
          else np.ones(len(df), bool)
    n   = len(df)

    if src_x_col == 'best_lidar_enu_x' and src_x_col not in df.columns:
        src_x, src_y, src_v, src_h = _get_best_lidar(df)
    else:
        src_x = df[src_x_col].values if src_x_col in df.columns else np.full(n, np.nan)
        src_y = df[src_y_col].values if src_y_col in df.columns else np.full(n, np.nan)
        src_v = df[src_v_col].values if src_v_col and src_v_col in df.columns else None
        src_h = df[src_h_col].values if src_h_col and src_h_col in df.columns else None

    res = _residuals(df, src_x, src_y, src_v, src_h)
    d   = res.shape[1]
    Se_inv = _PHYS_SCALE_INV[:d, :d]
    return _ost_from_res(res, win, Se_inv, Se_inv)


def compute_ost_pair(df_wc: pd.DataFrame, df_woc: pd.DataFrame) -> Dict:
    """OST for WC (V2X) vs WOC (LiDAR). Returns both 2D (pos-only) and 4D (pos+vel) variants.

    Fixed scale: σ_pos = 1.0 m, σ_vel = 1.0 m/s (avoids cross-mode MAD contamination).
    """
    Se2 = np.eye(2)
    Se4 = np.eye(4)

    def _window(df):
        return df['ego_state'].values == 1 if 'ego_state' in df.columns \
               else np.ones(len(df), dtype=bool)

    def _get_src(df, xc, yc, vc, hc):
        if xc == 'best_lidar_enu_x' and xc not in df.columns:
            return _get_best_lidar(df)
        src_x = df[xc].values if xc in df.columns else np.full(len(df), np.nan)
        src_y = df[yc].values if yc in df.columns else np.full(len(df), np.nan)
        src_v = df[vc].values if vc and vc in df.columns else None
        src_h = df[hc].values if hc and hc in df.columns else None
        return src_x, src_y, src_v, src_h

    win_wc  = _window(df_wc)
    win_woc = _window(df_woc)
    res_wc  = _residuals(df_wc,  *_get_src(df_wc,  'v2v_enu_x',        'v2v_enu_y',        'v2v_velocity',   'v2v_h'))
    res_woc = _residuals(df_woc, *_get_src(df_woc, 'best_lidar_enu_x', 'best_lidar_enu_y', 'lidar_velocity', 'lidar_h'))

    # 2D pos-only: slice first 2 dims
    r2_wc  = _ost_from_res(res_wc[:, :2],  win_wc,  Se2, Se2)
    r2_woc = _ost_from_res(res_woc[:, :2], win_woc, Se2, Se2)
    # 4D pos+vel
    r4_wc  = _ost_from_res(res_wc,  win_wc,  Se4, Se4)
    r4_woc = _ost_from_res(res_woc, win_woc, Se4, Se4)

    return {
        'WC_V2X':   r4_wc,  'WOC_LiDAR': r4_woc,
        'delta_OST': r4_wc['OST'] - r4_woc['OST'],
        'WC_V2X_pos':   r2_wc,  'WOC_LiDAR_pos': r2_woc,
        'delta_OST_pos': r2_wc['OST'] - r2_woc['OST'],
    }


def compute_residual_timeseries(df_wc: pd.DataFrame, df_woc: pd.DataFrame) -> dict:
    """Per-timestep e_t, Δe_t, valid_flag for WC (V2X) and WOC (LiDAR)."""
    def _window(df):
        return df['ego_state'].values == 1 if 'ego_state' in df.columns else np.ones(len(df), bool)

    def _src(df, xc, yc):
        if xc in df.columns:
            return df[xc].values, df[yc].values
        return (_get_best_lidar(df) if xc == 'best_lidar_enu_x'
                else (np.full(len(df), np.nan), np.full(len(df), np.nan)))

    def _delta_e(res, valid):
        de = np.full_like(res, np.nan)
        idx = np.where(valid)[0]
        for k in range(1, len(idx)):
            if idx[k] == idx[k-1] + 1:
                de[idx[k]] = res[idx[k]] - res[idx[k-1]]
        return de

    res_wc  = _residuals(df_wc,  *_src(df_wc,  'v2v_enu_x',       'v2v_enu_y'))
    res_woc = _residuals(df_woc, *_src(df_woc, 'best_lidar_enu_x', 'best_lidar_enu_y'))
    win_wc, win_woc = _window(df_wc), _window(df_woc)
    vf_wc  = win_wc  & np.isfinite(res_wc[:,  0]) & np.isfinite(res_wc[:,  1])
    vf_woc = win_woc & np.isfinite(res_woc[:, 0]) & np.isfinite(res_woc[:, 1])

    return {
        'e_wc': res_wc,   'e_woc': res_woc,
        'de_wc': _delta_e(res_wc, vf_wc), 'de_woc': _delta_e(res_woc, vf_woc),
        'valid_wc': vf_wc, 'valid_woc': vf_woc,
    }


# Optional future extension:
# z_t = [Δx_t, Δy_t, Δvx_t, Δvy_t]^T
#
# Compute global velocity:
# vx = velocity * cos(heading_rad)
# vy = velocity * sin(heading_rad)
#
# Relative velocity:
# Δv_global = v_obj_global - v_ego_global
#
# Transform to ego frame:
# [Δvx, Δvy]^T = R(-ego_heading) * Δv_global
