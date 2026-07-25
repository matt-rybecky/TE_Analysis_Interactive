#!/usr/bin/env python3
"""
pub_validation.py — Figure 2: the Stefan-Boltzmann validation control.

Fresh publication-settings validation of the transfer entropy machinery
against a known deterministic relationship (Methods 4; Results 4.1):
upwelling longwave and near-surface air temperature are coupled through
the Stefan-Boltzmann law (LW = eps*sigma*T^4), so the single-variable
transfer entropy between them must be large and significant all winter.
Direction of record (author 2026-07-11, thesis Ch5 convention):
LW-up(t0) -> temperature.

Ruling record (author, 2026-07-11): the paper's validation figure is the
Stefan-Boltzmann control ALONE — one single, well-understood transfer
metric in the main text. The Clausius-Clapeyron control was run at
publication settings (joint + both members; CSVs in
``publication_output/stage2/met_rh/``) but is NOT shown: the C-C
relationship underpins the entropy calibration via the
mutual-information construction (SI S4) and is not restated as a TE
control. The reversed S-B direction (temp -> LW-up; CSV in
``stage2/rad_lw_up/``) was rendered, inspected, and not selected.

Curve and band come from the Stage 2 CSV (``pub_stage2.py``, 2000 IAAFT
per window at publication settings); smoothing and band form are
identical to the finalized results figures (12-day centered median;
band shaded below the per-window 95th percentile).

Data prerequisite (run 2026-07-11; sig95 in 649/649 windows):

    .venv/bin/python3 pub_stage2.py --target met_temp \
        --entities rad_lw_up__t0 --n-cores 8

Then:

    .venv/bin/python3 pub_validation.py

Output: ``publication_output/fig_validation.{png,pdf}`` plus the
companion ``fig_validation_legend`` artifacts.

Author: Matthew Rybecky
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from pub_config import PublicationConfig
from pub_driver_series import entity_label
from pub_selected_figures import (SIG_BAND_STYLE, cross_style,
                                  smooth_median, stage2_series)
from pub_style import BAND_GRAY, date_axis, save_figure, save_legend, \
    setup_style

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# The validation entity of record (author-confirmed 2026-07-11).
SB_TARGET = 'met_temp'
SB_ENTITY = 'rad_lw_up__t0'
SB_TARGET_NAME = 'Temperature'

FIG_HEIGHT = 3.25       # inches; matches the finalized results figures


def val_series(base: Path, end: pd.Timestamp, target: str, entity: str,
               column: str) -> Optional[pd.Series]:
    """Smoothed, winter-truncated Stage 2 column (12-day centered median).

    Parameters
    ----------
    base : Path
        ``publication_output`` root.
    end : pd.Timestamp
        Winter truncation (``pub_config.analysis_end``).
    target, entity : str
        Stage 2 CSV coordinates.
    column : str
        ``pct_h`` (data curve) or ``surr_p95_pct`` (IAAFT 95% level).

    Returns
    -------
    pd.Series or None
        None if the CSV is not on disk yet.
    """
    s = stage2_series(base, target, entity, column)
    if s is None:
        logger.warning(f"missing Stage 2 CSV: {target}/{entity} "
                       "(run pub_stage2.py first)")
        return None
    return smooth_median(s[s.index < end])


def draw_validation(base: Path, end: pd.Timestamp, out: Path) -> None:
    """Figure 2: the Stefan-Boltzmann TE control, one line + IAAFT band."""
    curve = val_series(base, end, SB_TARGET, SB_ENTITY, 'pct_h')
    band = val_series(base, end, SB_TARGET, SB_ENTITY, 'surr_p95_pct')
    if curve is None:
        logger.warning(f"{out.name}: data incomplete; figure skipped")
        return

    figsize = setup_style('full', height=FIG_HEIGHT)
    fig, ax = plt.subplots(figsize=figsize)
    style = cross_style(0)
    ax.plot(curve.index, curve, **style)
    if band is not None:
        ax.fill_between(band.index, 0.0, band, color=BAND_GRAY,
                        linewidth=0, zorder=0)
    ax.set_xlim(curve.index[0], curve.index[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel(f'TE (% of H({SB_TARGET_NAME}))')
    ax.set_xlabel('Date (2022-2023)')
    date_axis(ax)

    label = f"{entity_label(SB_ENTITY)} → {SB_TARGET_NAME}"
    entries: List[tuple] = [
        (label, style),
        ('Below IAAFT 95% level', SIG_BAND_STYLE),
    ]
    for p in [*save_figure(fig, out),
              *save_legend(entries, Path(f'{out}_legend'), ncol=2)]:
        logger.info(f"wrote {p}")


def main() -> None:
    cfg = PublicationConfig()
    draw_validation(Path(cfg.output_base), pd.Timestamp(cfg.analysis_end),
                    Path(cfg.output_base) / 'fig_validation')


if __name__ == '__main__':
    main()
