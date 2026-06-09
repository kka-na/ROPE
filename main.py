#!/usr/bin/env python3
"""ROPE Framework 진입점"""
import sys
import argparse
from pathlib import Path


def run_ui():
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QFont
    from rope.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setFont(QFont('Arial', 10))
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


def run_cli(csv_dir: str, output_dir: str, axes: list, paired: bool = False):
    from rope.pipeline import ROPEPipeline
    from rope.config import ROPEConfig
    pipeline = ROPEPipeline(config=ROPEConfig(axes=axes, paired=paired))
    results  = pipeline.run(csv_dir, output_dir, axes)
    print(f"\nMetrics: {results.get('metrics_path')}")
    for ax in ['R', 'O', 'P']:
        if ax in results:
            tbl = results[ax].get('table')
            n   = len(tbl) if tbl is not None else 0
            print(f"  {ax}-Axis: {n} rows, {len(results[ax].get('figures',[]))} figures")


def main():
    parser = argparse.ArgumentParser(description='ROPE Framework')
    parser.add_argument('--csv',    help='CSV input folder (CLI mode)')
    parser.add_argument('--out',    help='Output folder (CLI mode)', default='rope_output')
    parser.add_argument('--axes',   help='Axes to run (R,O,P)', default='R,O,P')
    parser.add_argument('--paired', action='store_true', help='Use paired Wilcoxon for P-Axis (FIELD)')
    args = parser.parse_args()

    if args.csv:
        run_cli(args.csv, args.out, args.axes.split(','), paired=args.paired)
    else:
        run_ui()


if __name__ == '__main__':
    main()
