#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import List


@dataclass
class ROPEConfig:
    axes: List[str] = field(default_factory=lambda: ['R', 'O', 'P', 'T'])
    timeseries_subdir: str = 'timeseries'
    sequences_subdir: str = 'sequences'
    scenario_metrics_filename: str = 'scenario_metrics.csv'
    # Analysis window
    window_before_m: float = 20.0
    window_after_m: float = 130.0
    # Statistics
    alpha: float = 0.05
    paired: bool = True   # Wilcoxon signed-rank (falls back to Mann-Whitney if no pair match)

    def enabled(self, axis: str) -> bool:
        return axis in self.axes
