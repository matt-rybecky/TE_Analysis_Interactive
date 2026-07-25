#!/usr/bin/env python3
"""
pub_fig_overview.py — Figure 1: the measured dD and d-excess records.

Data-only overview (author rulings 2026-07-08): the calibrated 6-hour dD
and d-excess series in real units, truncated at the analysis end
(``pub_config.analysis_end``, 2023-05-01, the continuous-snowpack scope),
as two stacked panels with a shared time axis — (a) dD, (b) d-excess —
each a single solid black line. No analysis annotations: no period boundaries, no episodic spans, no
site map (the period structure appears in Figure 3). No markers, no
legend; panel labels and axis labels identify the series. Instrument gaps
are bridged by linear interpolation drawn as a visibly distinct thin
dotted line, so real data and filled gaps remain distinguishable. This
figure opts into the per-figure grid (author ruling 2026-07-08).
(A twin-y single-axes form was tried first and failed the style guide's
identifiability check: the two high-frequency series fully interleave.)

Input: ``data/final_6hr.csv`` — the real-unit raw file of the frozen
publication chain (the declared input of ``build_circular_data.py``,
row-aligned with the beta file the sweep used; confirmed 2026-07-08).

Output: ``publication_output/fig_overview.{png,pdf}`` at 300 DPI.

Usage:
    .venv/bin/python3 pub_fig_overview.py

Author: Matthew Rybecky
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pub_config import PublicationConfig
from pub_labels import target_label
from pub_style import (date_axis, light_grid, panel_label, save_figure,
                       setup_style)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RAW_FILE = Path('data/final_6hr.csv')
OUT_STEM = Path('publication_output/fig_overview')
FIG_HEIGHT = 4.5        # inches; two stacked panels at full width


def load_series(path: Path, analysis_end: pd.Timestamp) -> pd.DataFrame:
    """Load the real-unit 6-h record, truncated to the analysis scope.

    Parameters
    ----------
    path : Path
        The raw real-unit CSV (``time``, ``dD``, ``d_excess`` among others).
    analysis_end : pd.Timestamp
        Exclusive cutoff (the frozen spec's ``analysis_end``, 2023-05-01).

    Returns
    -------
    pd.DataFrame
        Columns ``time``, ``dD``, ``d_excess``; rows before the cutoff;
        gaps as NaN.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required input missing: {path}")
    df = pd.read_csv(path, parse_dates=['time'],
                     usecols=['time', 'dD', 'd_excess'])
    df = df[df['time'] < analysis_end].reset_index(drop=True)
    logger.info(f"{path}: {len(df)} rows after truncation, "
                f"{df['time'].iloc[0]} to {df['time'].iloc[-1]}, "
                f"non-null dD {df['dD'].notna().sum()}, "
                f"d-excess {df['d_excess'].notna().sum()}")
    return df


def draw_panel(ax, time: pd.Series, values: pd.Series, ylabel: str,
               label: str) -> None:
    """One panel: solid data line over a dotted interpolation bridge.

    The interpolated series (gaps filled linearly, interior only) is drawn
    first as a thin dotted line; the measured series is drawn solid on top,
    so the dotted bridge shows only inside instrument gaps.
    """
    bridged = values.interpolate(method='linear', limit_area='inside')
    ax.plot(time, bridged, color='black', linestyle=':', linewidth=0.8,
            zorder=2)
    ax.plot(time, values, color='black', linestyle='-', zorder=3)
    ax.set_ylabel(ylabel)
    light_grid(ax)
    panel_label(ax, label)


def draw_overview(df: pd.DataFrame) -> plt.Figure:
    """Two stacked panels, shared time axis: (a) dD, (b) d-excess."""
    figsize = setup_style('full', height=FIG_HEIGHT)
    fig, (ax_dd, ax_dx) = plt.subplots(2, 1, sharex=True, figsize=figsize)

    draw_panel(ax_dd, df['time'], df['dD'],
               f"{target_label('dD')} (permil)", '(a)')
    draw_panel(ax_dx, df['time'], df['d_excess'],
               f"{target_label('d_excess')} (permil)", '(b)')

    ax_dx.set_xlabel('Date (2022-2023)')
    date_axis(ax_dx)
    return fig


def main() -> None:
    cfg = PublicationConfig()
    df = load_series(RAW_FILE, pd.Timestamp(cfg.analysis_end))
    fig = draw_overview(df)
    for p in save_figure(fig, OUT_STEM):
        logger.info(f"wrote {p}")


if __name__ == '__main__':
    main()
