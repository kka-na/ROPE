#!/usr/bin/env python3
"""Trajectory Visualization - Parquet-only (ego_enu_x/y, target_enu_x/y, ego_h)"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from rope.viz import colors

BASE_DIR = Path('/home/kana/Documents/Dataset/CCAVT')
MAP_DIR  = BASE_DIR / 'cbf' / 'map'

LOCATIONS = {
    'KIAPI': {
        'base_lla': (35.6480998, 128.4018747, 7),
        'map_file': MAP_DIR / 'KIAPI.shp',
        'use_utm': True,
        'fixed_rotation_deg': -227.64,
    },
    'Yeongjong': {
        'base_lla': (37.527247, 126.5068108, 7),
        'map_dir': MAP_DIR,
        'use_utm': False,
        'fixed_rotation_deg': None,
    },
}


def _get_map_file(name, location):
    if location == 'KIAPI':
        return LOCATIONS['KIAPI']['map_file']
    m = re.search(r'(clm|etra)(\d+).*?(same|faster|slower)', name.lower())
    if m:
        shp = LOCATIONS['Yeongjong']['map_dir'] / f"{m[1].upper()}{m[2]}_{m[3]}.shp"
        if shp.exists():
            return shp
    return LOCATIONS['Yeongjong']['map_dir'] / 'Midan.shp'


def _rotate(x, y, angle):
    c, s = np.cos(angle), np.sin(angle)
    return c * x - s * y, s * x + c * y


def _map_lines(map_file, origin, angle, base_lla, use_utm):
    if not Path(map_file).exists():
        return []
    try:
        import geopandas as gpd
        import pymap3d as pm
        gdf = gpd.read_file(map_file)
    except Exception:
        return []
    lines = []
    for _, row in gdf.iterrows():
        geom = row['geometry']
        if geom is None or geom.geom_type != 'LineString':
            continue
        lx, ly = [], []
        for lon, lat in geom.coords:
            if use_utm:
                import utm
                lat, lon = utm.to_latlon(lon, lat, 52, 'U')
            x, y, _ = pm.geodetic2enu(lat, lon, 7, base_lla[0], base_lla[1], base_lla[2])
            lx.append(x); ly.append(y)
        lx, ly = np.array(lx) - origin[0], np.array(ly) - origin[1]
        rx, ry = _rotate(lx, ly, angle)
        kind = int(row.get('Kind', row.get('kind', 0)) or 0)
        if kind in [531, 515]:   ls, col, lw = '-',  'grey', 4
        elif kind in [525, 503, 506]: ls, col, lw = '--', 'grey', 1.5
        elif kind == 530:         ls, col, lw = '-',  'grey', 6
        else:                     ls, col, lw = '-',  'grey', 1
        lines.append({'x': rx, 'y': ry, 'ls': ls, 'color': col, 'lw': lw})
    return lines


def plot_trajectory(scenario_name, ts_dir=None, out_path=None, figsize=(5, 2), ax=None):
    ts_dir = Path(ts_dir) if ts_dir else BASE_DIR / 'cbf' / 'timeseries'
    pq_file = next((f for f in ts_dir.glob('*.parquet')
                    if scenario_name.lower() in f.stem.lower()), None)
    if pq_file is None:
        if ax:
            ax.text(0.5, 0.5, f'Not found: {scenario_name}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7)
        return None, None

    df = pd.read_parquet(pq_file)
    loc = df['location'].iloc[0] if 'location' in df.columns else 'Yeongjong'
    cfg = LOCATIONS.get(loc, LOCATIONS['Yeongjong'])

    ego_x, ego_y   = df['ego_enu_x'].values, df['ego_enu_y'].values
    tgt_x,  tgt_y  = df['target_enu_x'].values, df['target_enu_y'].values
    origin = np.array([ego_x[0], ego_y[0]])

    half = len(ego_x) // 4
    rot = -np.deg2rad(df['ego_h'].iloc[half]) if 'ego_h' in df.columns else 0.0

    ex, ey = _rotate(ego_x - origin[0], ego_y - origin[1], rot)
    tx, ty = _rotate(tgt_x  - origin[0], tgt_y  - origin[1], rot)

    map_file = _get_map_file(scenario_name, loc)
    map_lns  = _map_lines(map_file, origin, rot, cfg['base_lla'], cfg['use_utm'])

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for ln in map_lns:
        ax.plot(ln['x'], ln['y'], color=ln['color'], ls=ln['ls'], lw=ln['lw'], alpha=0.6, zorder=0)

    ax.plot(ex, ey, color='#505050', ls='solid', lw=3, label='Ego', zorder=2)

    dist = df['ego_cumulative_dist'].values
    dist_rel = dist - dist[0]
    sig_idx = np.argmin(np.abs(dist_rel - 20))
    ax.axvline(x=ex[sig_idx], color=colors.SEC6, linestyle='--', linewidth=1, label='Signal', zorder=4)

    action_dist = df['action_start_dist'].iloc[0] if 'action_start_dist' in df.columns else np.nan
    if pd.notna(action_dist):
        ai = np.argmin(np.abs(dist_rel - action_dist))
        lbl = 'LC Start' if 'clm' in scenario_name.lower() else 'Avoid Start'
        ax.axvline(x=ex[ai], color=colors.SEC4, linestyle='--', linewidth=1.5, label=lbl, zorder=4)

    valid = np.isfinite(tx) & np.isfinite(ty)
    if valid.any():
        ax.plot(tx[valid], ty[valid], color='#505050', ls='dotted', lw=3, label='Target', zorder=2)

    all_y = np.concatenate([ey, ty[valid]]) if valid.any() else ey
    ax.set_xlim(np.nanmin(ex) - 5, np.nanmax(ex) + 5)
    ax.set_ylim(np.nanmin(all_y) - 2, np.nanmax(all_y) + 2)
    ax.set_xlabel('Forward (m)', fontsize=8)
    ax.set_ylabel('Lateral (m)', fontsize=8)
    ax.set_title(f'(c) {scenario_name}', fontsize=8, color=colors.TITLE)
    ax.legend(fontsize=6, loc='best')
    ax.tick_params(labelsize=7)
    ax.grid(False)

    if fig:
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    return fig, ax
