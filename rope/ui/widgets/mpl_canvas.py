#!/usr/bin/env python3
"""Matplotlib ↔ Qt 브릿지 위젯"""
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QSizePolicy


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, fig: Figure = None):
        if fig is None:
            fig = Figure(figsize=(8, 5))
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def set_figure(self, fig: Figure):
        self.figure = fig
        self.draw()


class MplWidget(QWidget):
    """Canvas + NavigationToolbar 조합 위젯"""
    def __init__(self, fig: Figure = None, parent=None):
        super().__init__(parent)
        self.canvas  = MplCanvas(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

    def set_figure(self, fig: Figure):
        self.canvas.figure = fig
        self.canvas.draw_idle()


class ScrollMplWidget(QScrollArea):
    """스크롤 가능한 Matplotlib 뷰어 (여러 Figure 수직 배치)"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._container = QWidget()
        self._layout    = QVBoxLayout(self._container)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(8)
        self.setWidget(self._container)
        self._widgets: list = []

    def set_figures(self, figs: list):
        for w in self._widgets:
            self._layout.removeWidget(w)
            w.deleteLater()
        self._widgets.clear()

        for fig in figs:
            w = MplWidget(fig)
            self._layout.addWidget(w)
            self._widgets.append(w)
