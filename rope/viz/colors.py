#!/usr/bin/env python3
"""ROPE 통합 색상 팔레트 (cbf/colors.py 기반)"""
import matplotlib.pyplot as plt

# === WC / WOC Primary Colors ===
WC       = '#15478A'
WC_LIGHT = '#5bb5d5'
WOC      = '#8c8c91'

WC_TINT80  = '#446CA1'; WC_TINT60  = '#7391B9'; WC_TINT40  = '#8AA3C5'
WC_LIGHT_TINT80 = '#7CC4DD'; WC_LIGHT_TINT60 = '#9DD3E6'; WC_LIGHT_TINT40 = '#BDE1EE'
WOC_TINT80 = '#A3A3A7'; WOC_TINT60 = '#BABABD'; WOC_TINT40 = '#D1D1D3'

# === Secondary Colors ===
SEC1 = '#3b5bbe'   # Target
SEC2 = '#e14ba1'   # Ego / ETrA type
SEC3 = '#f9c555'   # CLM type
SEC4 = '#419164'   # Action point
SEC5 = '#876ec4'   # ΔH / uncertainty
SEC6 = '#f36428'   # Signal point (V2V)

# === Text / Grid ===
TITLE      = '#202020'
CAPTION    = '#505050'
AXES       = '#505050'
ANNOTATION = '#505050'
GRID       = '#D3D3D3'

# === Layout ===
FIG_WIDTH_FULL = 7.0
FIG_WIDTH_HALF = 3.5
FIG_DPI        = 150
LABEL_BBOX     = dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none')

# === Heatmap Colors (WOC better → Neutral → WC better) ===
HEATMAP_WOC_STRONG = '#f8a273'
HEATMAP_WOC_WEAK   = '#fbdc99'
HEATMAP_NEUTRAL    = '#f0eee9'
HEATMAP_WC_WEAK    = '#5bb5d5'
HEATMAP_WC_STRONG  = '#697ebc'
HEATMAP_COLORS = [HEATMAP_WOC_STRONG, HEATMAP_WOC_WEAK, HEATMAP_NEUTRAL, HEATMAP_WC_WEAK, HEATMAP_WC_STRONG]

# === Semantic Mappings ===
MODE_COLORS     = {'WC': WC,   'WOC': WOC}
LOCATION_COLORS = {'KIAPI': SEC1, 'Yeongjong': SEC4, 'CARLA': SEC6}
TYPE_COLORS     = {'CLM': SEC3, 'ETrA': SEC2}


def apply_style():
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'axes.labelcolor': AXES,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.figsize': (FIG_WIDTH_FULL, 4),
        'figure.dpi': FIG_DPI,
        'axes.grid': True,
        'grid.color': GRID,
        'grid.linewidth': 0.5,
    })
