#!/usr/bin/env python3
"""ROPE Qt 메인 윈도우"""
import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QFileDialog, QTabWidget, QProgressBar, QStatusBar,
    QMessageBox, QFrame, QSizePolicy, QSplitter, QPlainTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette

from rope.ui.widgets.panels import RPanel, OPanel, PPanel, CrossPanel
from rope.config import ROPEConfig


# ─── Worker Threads ────────────────────────────────────────────────────────
class LoadWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, output_dir: str, axes: list):
        super().__init__()
        self.output_dir = output_dir
        self.axes = axes

    def run(self):
        try:
            from rope.pipeline import ROPEPipeline
            self.finished.emit(ROPEPipeline().load(self.output_dir, self.axes))
        except Exception as e:
            import traceback
            self.error.emit(f'{e}\n\n{traceback.format_exc()}')


class ReanalyzeWorker(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, output_dir: str, axes: list):
        super().__init__()
        self.output_dir = output_dir
        self.axes = axes

    def run(self):
        try:
            from rope.pipeline import ROPEPipeline
            pipeline = ROPEPipeline(progress_cb=self._cb)
            self.finished.emit(pipeline.run_from_existing(self.output_dir, self.axes))
        except Exception as e:
            import traceback
            self.error.emit(f'{e}\n\n{traceback.format_exc()}')

    def _cb(self, msg: str, pct: int):
        self.progress.emit(msg, pct)


class PipelineWorker(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, csv_dir: str, output_dir: str, axes: list):
        super().__init__()
        self.csv_dir    = csv_dir
        self.output_dir = output_dir
        self.axes       = axes

    def run(self):
        try:
            from rope.pipeline import ROPEPipeline
            pipeline = ROPEPipeline(progress_cb=self._cb)
            results  = pipeline.run(self.csv_dir, self.output_dir, self.axes)
            self.finished.emit(results)
        except Exception as e:
            import traceback
            self.error.emit(f'{e}\n\n{traceback.format_exc()}')

    def _cb(self, msg: str, pct: int):
        self.progress.emit(msg, pct)


# ─── Main Window ───────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ROPE — Reliability · Observability · Performance · Evaluation')
        self.resize(1280, 800)
        self._worker: QThread = None
        self._build_ui()

    # ── UI Construction ────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Left control panel
        left = self._build_left_panel()
        left.setFixedWidth(270)
        left.setFrameShape(QFrame.StyledPanel)

        # Right tab panel
        self._tabs = self._build_tabs()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

        # Status bar
        self._status_bar = QStatusBar()
        self._progress   = QProgressBar()
        self._progress.setFixedWidth(200)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._status_bar.addPermanentWidget(self._progress)
        self.setStatusBar(self._status_bar)
        self._status('Ready.')

    def _build_left_panel(self) -> QFrame:
        frame = QFrame()
        lay   = QVBoxLayout(frame)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        # Logo / title
        title = QLabel('ROPE\nFramework')
        title.setAlignment(Qt.AlignCenter)
        f = QFont('Arial', 16, QFont.Bold)
        title.setFont(f)
        title.setStyleSheet('color: #15478A; padding: 10px 0;')
        lay.addWidget(title)

        subtitle = QLabel('Reliability · Observability\nPerformance · Evaluation')
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet('color: #505050; font-size: 10px;')
        lay.addWidget(subtitle)

        # Input group
        grp_input = QGroupBox('Input / Output')
        g_lay = QVBoxLayout(grp_input)

        _default_data = Path(__file__).parents[2] / 'data'
        self._csv_edit = QLineEdit(str(_default_data))
        self._out_edit = QLineEdit(); self._out_edit.setPlaceholderText('Output folder...')

        for lbl_text, edit, slot in [
            ('CSV Folder',    self._csv_edit, self._browse_csv),
            ('Output Folder', self._out_edit, self._browse_out),
        ]:
            g_lay.addWidget(QLabel(lbl_text))
            row = QHBoxLayout()
            row.addWidget(edit)
            btn = QPushButton('…'); btn.setFixedWidth(30); btn.clicked.connect(slot)
            row.addWidget(btn)
            g_lay.addLayout(row)

        lay.addWidget(grp_input)

        # Axes group
        grp_axes = QGroupBox('Axes')
        a_lay = QVBoxLayout(grp_axes)
        self._chk_r = QCheckBox('R — Reliability');     self._chk_r.setChecked(True)
        self._chk_o = QCheckBox('O — Observability');   self._chk_o.setChecked(True)
        self._chk_p = QCheckBox('P — Performance');     self._chk_p.setChecked(True)
        for chk in [self._chk_r, self._chk_o, self._chk_p]:
            a_lay.addWidget(chk)
        lay.addWidget(grp_axes)

        # Run button
        self._run_btn = QPushButton('▶  Run ROPE')
        self._run_btn.setStyleSheet(
            'QPushButton { background:#15478A; color:white; font-size:13px; '
            'font-weight:bold; border-radius:6px; padding:8px; }'
            'QPushButton:hover { background:#446CA1; }'
            'QPushButton:disabled { background:#BABABD; }'
        )
        self._run_btn.clicked.connect(self._run)
        lay.addWidget(self._run_btn)

        # Reanalyze button (from existing parquets)
        self._reanalyze_btn = QPushButton('🔄  Reanalyze (from Parquets)')
        self._reanalyze_btn.setStyleSheet(
            'QPushButton { background:#7B5EA7; color:white; font-size:12px; '
            'font-weight:bold; border-radius:6px; padding:6px; }'
            'QPushButton:hover { background:#9B7EC7; }'
            'QPushButton:disabled { background:#BABABD; }'
        )
        self._reanalyze_btn.clicked.connect(self._reanalyze)
        lay.addWidget(self._reanalyze_btn)

        # Load button
        self._load_btn = QPushButton('📂  Load Results')
        self._load_btn.setStyleSheet(
            'QPushButton { background:#419164; color:white; font-size:12px; '
            'font-weight:bold; border-radius:6px; padding:6px; }'
            'QPushButton:hover { background:#5aab82; }'
            'QPushButton:disabled { background:#BABABD; }'
        )
        self._load_btn.clicked.connect(self._load)
        lay.addWidget(self._load_btn)

        # Log box (scrollable)
        self._log_box = QPlainTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMaximumHeight(160)
        self._log_box.setStyleSheet('font-size:9px; color:#303030; background:#f5f5f5;')
        lay.addWidget(self._log_box)

        lay.addStretch()
        return frame

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setStyleSheet('QTabBar::tab { min-width: 160px; padding: 6px 12px; }')
        self._r_panel     = RPanel()
        self._o_panel     = OPanel()
        self._p_panel     = PPanel()
        self._cross_panel = CrossPanel()
        tabs.addTab(self._r_panel,     'R: Reliability')
        tabs.addTab(self._o_panel,     'O: Observability')
        tabs.addTab(self._p_panel,     'P: Performance')
        tabs.addTab(self._cross_panel, '⇄ Cross-Axis')
        return tabs

    # ── Slots ──────────────────────────────────────────────────────────────
    def _browse_csv(self):
        start = self._csv_edit.text() or str(Path.home())
        d = QFileDialog.getExistingDirectory(self, 'Select CSV Folder', start)
        if d:
            self._csv_edit.setText(d)
            # auto-set output inside CSV dir
            if not self._out_edit.text():
                self._out_edit.setText(str(Path(d) / 'rope_output'))

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, 'Select Output Folder', str(Path.home()))
        if d:
            self._out_edit.setText(d)

    def _run(self):
        csv_dir = self._csv_edit.text().strip()
        out_dir = self._out_edit.text().strip()

        if not csv_dir or not Path(csv_dir).is_dir():
            QMessageBox.warning(self, 'Error', 'Please select a valid CSV folder.')
            return
        if not out_dir:
            QMessageBox.warning(self, 'Error', 'Please select an output folder.')
            return

        axes = []
        if self._chk_r.isChecked(): axes.append('R')
        if self._chk_o.isChecked(): axes.append('O')
        if self._chk_p.isChecked(): axes.append('P')
        if not axes:
            QMessageBox.warning(self, 'Error', 'Select at least one axis.')
            return

        self._set_buttons(False)
        self._progress.setValue(0)
        self._worker = PipelineWorker(csv_dir, out_dir, axes)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _reanalyze(self):
        out_dir = self._out_edit.text().strip()
        if not out_dir or not Path(out_dir).is_dir():
            QMessageBox.warning(self, 'Error', 'Please select a valid Output folder.')
            return
        ts_dir = Path(out_dir) / 'timeseries'
        if not ts_dir.exists() or not list(ts_dir.glob('*.parquet')):
            QMessageBox.warning(self, 'Error', f'No parquet files found in:\n{ts_dir}')
            return
        axes = []
        if self._chk_r.isChecked(): axes.append('R')
        if self._chk_o.isChecked(): axes.append('O')
        if self._chk_p.isChecked(): axes.append('P')
        if not axes:
            QMessageBox.warning(self, 'Error', 'Select at least one axis.')
            return
        self._set_buttons(False)
        self._progress.setValue(0)
        self._status('Reanalyzing from existing parquets...')
        self._worker = ReanalyzeWorker(out_dir, axes)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _load(self):
        out_dir = self._out_edit.text().strip()
        if not out_dir or not Path(out_dir).is_dir():
            QMessageBox.warning(self, 'Error', 'Please select a valid Output folder.')
            return
        axes = []
        if self._chk_r.isChecked(): axes.append('R')
        if self._chk_o.isChecked(): axes.append('O')
        if self._chk_p.isChecked(): axes.append('P')
        if not axes:
            QMessageBox.warning(self, 'Error', 'Select at least one axis.')
            return
        self._set_buttons(False)
        self._status('Loading existing results...')
        self._worker = LoadWorker(out_dir, axes)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _set_buttons(self, enabled: bool):
        self._run_btn.setEnabled(enabled)
        self._reanalyze_btn.setEnabled(enabled)
        self._load_btn.setEnabled(enabled)

    def _log(self, msg: str):
        self._log_box.appendPlainText(msg)
        self._log_box.verticalScrollBar().setValue(self._log_box.verticalScrollBar().maximum())

    def _on_progress(self, msg: str, pct: int):
        self._status(msg)
        self._log(msg)
        if pct >= 0:
            self._progress.setValue(pct)

    def _on_finished(self, results: dict):
        self._set_buttons(True)
        self._progress.setValue(100)
        self._status('✅ Pipeline complete.')
        self._log('--- Rendering results ---')

        if 'R' in results:
            self._r_panel.update_results(results['R'])
            self._tabs.setCurrentWidget(self._r_panel)
        if 'O' in results:
            self._o_panel.update_results(results['O'])
        if 'P' in results:
            self._p_panel.update_results(results['P'])
        if 'Cross' in results:
            self._cross_panel.update_results(results['Cross'])
            self._tabs.setCurrentWidget(self._cross_panel)
        elif 'P' in results:
            self._tabs.setCurrentWidget(self._p_panel)

        self._log('✅ Done.')

    def _on_error(self, msg: str):
        self._set_buttons(True)
        self._progress.setValue(0)
        self._status('❌ Error.')
        self._log(f'❌ ERROR:\n{msg}')
        QMessageBox.critical(self, 'Pipeline Error', msg[:1000])

    def _status(self, msg: str):
        self._status_bar.showMessage(msg)
