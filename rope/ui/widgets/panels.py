#!/usr/bin/env python3
"""R / O / P 축 패널 위젯"""
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTableWidget, QTableWidgetItem, QLabel, QHeaderView
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont

from rope.ui.widgets.mpl_canvas import ScrollMplWidget


# ─── 공통 헬퍼 ──────────────────────────────────────────
def _make_table(df: pd.DataFrame) -> QTableWidget:
    if df is None or df.empty:
        t = QTableWidget(0, 1)
        t.setHorizontalHeaderLabels(['(no data)'])
        return t
    t = QTableWidget(len(df), len(df.columns))
    t.setHorizontalHeaderLabels(df.columns.tolist())
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.setAlternatingRowColors(True)
    t.setEditTriggers(QTableWidget.NoEditTriggers)

    font_bold = QFont(); font_bold.setBold(True)

    for r, row in enumerate(df.itertuples(index=False)):
        for c, val in enumerate(row):
            text = f'{val:.4g}' if isinstance(val, float) else str(val)
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)

            # P-axis: significant 행 강조
            col_name = df.columns[c]
            if col_name == 'significant' and val:
                item.setFont(font_bold)
                item.setForeground(QColor('#15478A'))
            if col_name == 'p_value' and isinstance(val, float) and val < 0.05:
                item.setBackground(QColor('#e8f0fa'))

            t.setItem(r, c, item)
    return t


def _header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet('font-size: 13px; font-weight: bold; color: #202020; padding: 4px;')
    return lbl


# ─── 축 패널 기반 클래스 ────────────────────────────────
class AxisPanel(QWidget):
    def __init__(self, axis_name: str, description: str, parent=None):
        super().__init__(parent)
        self.axis_name = axis_name

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(_header(f'{axis_name}  —  {description}'))

        splitter = QSplitter(Qt.Vertical)
        self._table_widget = QTableWidget()
        self._scroll_mpl   = ScrollMplWidget()

        splitter.addWidget(self._table_widget)
        splitter.addWidget(self._scroll_mpl)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)

        self._placeholder = QLabel('Run the pipeline to see results.')
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet('color: #808080; font-size: 12px;')
        layout.addWidget(self._placeholder)

    def update_results(self, result: dict):
        if not result:
            return
        self._placeholder.setVisible(False)

        # Table
        table_df = result.get('table')
        if table_df is None or table_df.empty:
            table_df = result.get('pairs')
        if table_df is not None and not table_df.empty:
            new_table = _make_table(table_df)
            self._table_widget.setParent(None)
            self._table_widget = new_table
            splitter = self._scroll_mpl.parent()
            if isinstance(splitter, QSplitter):
                splitter.insertWidget(0, self._table_widget)

        # Figures
        figs = result.get('figures', [])
        self._scroll_mpl.set_figures(figs)


class RPanel(AxisPanel):
    def __init__(self, parent=None):
        super().__init__('R-Axis: Reliability', 'Channel & Data Quality', parent)


class OPanel(AxisPanel):
    def __init__(self, parent=None):
        super().__init__('O-Axis: Observability', 'Observable State Tube (OST-4D / OST-2D)', parent)

    def update_results(self, result: dict):
        if not result:
            return
        self._placeholder.setVisible(False)
        pairs = result.get('pairs')
        if pairs is not None and not pairs.empty:
            _rename = {
                'scenario': 'Scenario', 'location': 'Location', 'type': 'Type',
                'ost_wc': 'OST CAD', 'ost_woc': 'OST SAD', 'delta_ost': 'ΔOST',
                'ost_c_wc': 'OST_c CAD', 'ost_c_woc': 'OST_c SAD', 'delta_ost_c': 'ΔOST_c',
            }
            sel = [c for c in _rename if c in pairs.columns]
            display_df = pairs[sel].rename(columns=_rename)
            new_table = _make_table(display_df)
            self._table_widget.setParent(None)
            self._table_widget = new_table
            splitter = self._scroll_mpl.parent()
            if isinstance(splitter, QSplitter):
                splitter.insertWidget(0, self._table_widget)
        figs = result.get('figures', [])
        self._scroll_mpl.set_figures(figs)


class PPanel(AxisPanel):
    def __init__(self, parent=None):
        super().__init__('P-Axis: Performance', 'Safety · Stability · Efficiency  (CAD vs SAD)', parent)


class CrossPanel(AxisPanel):
    def __init__(self, parent=None):
        super().__init__('Cross-Axis', 'R↔O (ICQ→ΔOST)  ·  O↔P (ΔOST→Performance)  ·  R↔P (ICQ→Performance)', parent)
