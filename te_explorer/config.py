"""Central configuration for the te-explorer application.

Publication defaults mirror the frozen manuscript configuration
(``manuscript/pub_config.py``) and the original analysis data-preparation
pipeline (``generate_extended_datasets.py``, not shipped in this repo; its
alignment and normalization steps are reimplemented under
``te_explorer/prep/``). Values here are transcribed from those sources, not
invented: 30-day rolling windows, history length h=1, KSG k=3
(base 2, bits), 2000 IAAFT surrogates per window, 95th-percentile
significance band.

This module also performs the NPEET path injection so every other module
can ``from npeet import entropy_estimators as ee`` after importing it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = REPO_ROOT / 'core'
NPEET_PATH = CORE_PATH / 'NPEET'
DATA_DIR = REPO_ROOT / 'data'
OUTPUT_DIR = REPO_ROOT / 'output_plots'

for _path in (CORE_PATH, NPEET_PATH):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@dataclass
class AppConfig:
    """Application configuration with publication defaults.

    Every analysis parameter the GUI exposes defaults to the exact value
    used for the manuscript. The KSG neighbor count k=3 is hardcoded inside
    the engine of record (``core/TE_Calculator.py``) and is therefore
    displayed in the GUI as a fixed constant, not an editable control.

    Attributes
    ----------
    time_col : str
        Name of the datetime column expected in loaded CSV files.
    window_days : int
        Rolling window length in days (publication: 30).
    history_length : int
        Target history length h conditioning the TE estimate (publication: 1).
    tau_options : tuple of int
        Time-lag choices offered by the GUI, in units of the data interval.
    tau_default : int
        Default time lag (publication baseline: 1).
    n_surrogates : int
        IAAFT surrogates per window (publication: 2000).
    surrogate_type : str
        Fixed to 'iaaft'; the GUI never exposes a choice.
    significance_percentile : float
        Surrogate percentile drawn as the significance band (publication: 95).
    ksg_k : int
        KSG neighbor count. Informational: the engine hardcodes k=3.
    entropy_base : float
        Logarithm base for all information quantities (2.0 = bits).
    interval_hours : int
        Default resampling interval for data preparation (publication: 6).
    sigma_fraction : float
        Gaussian resampling kernel sigma as a fraction of the interval
        (pipeline of record: 0.25).
    max_gap_hours : float
        Longest gap bridged by conservative linear interpolation
        (pipeline of record: 3).
    mad_scale_factor : float
        MAD-to-sigma factor in the robust z-score (pipeline of record: 0.6745).
    calibration_k : int
        KSG neighbor count for the beta-calibration entropy and joint MI
        estimates (author ruling 2026-07-23: k=3, unified with the engine;
        the original pipeline script used k=10).
    calibration_target : str
        Default calibration target variable for beta normalization.
    calibration_inputs : tuple of str
        Default calibration input variables for beta normalization.
    """

    time_col: str = 'time'
    window_days: int = 30
    history_length: int = 1
    tau_options: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 12)
    tau_default: int = 1
    n_surrogates: int = 2000
    surrogate_type: str = 'iaaft'
    significance_percentile: float = 95.0
    ksg_k: int = 3
    entropy_base: float = 2.0

    interval_hours: int = 6
    sigma_fraction: float = 0.25
    max_gap_hours: float = 3.0
    mad_scale_factor: float = 0.6745
    calibration_k: int = 3
    calibration_target: str = 'met_rh'
    calibration_inputs: Tuple[str, ...] = ('met_temp', 'met_vapor_pressure')

    def __post_init__(self) -> None:
        if self.window_days <= 0:
            raise ValueError("window_days must be positive")
        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")
        if self.n_surrogates < 1:
            raise ValueError("n_surrogates must be >= 1")
        if self.surrogate_type != 'iaaft':
            raise ValueError("surrogate_type is fixed to 'iaaft' in this release")
        if not 0.0 < self.significance_percentile < 100.0:
            raise ValueError("significance_percentile must be in (0, 100)")
        if self.interval_hours <= 0:
            raise ValueError("interval_hours must be positive")
        if not 0.0 < self.sigma_fraction <= 1.0:
            raise ValueError("sigma_fraction must be in (0, 1]")
        if self.max_gap_hours < 0:
            raise ValueError("max_gap_hours must be non-negative")
        if self.calibration_k < 1:
            raise ValueError("calibration_k must be >= 1")
