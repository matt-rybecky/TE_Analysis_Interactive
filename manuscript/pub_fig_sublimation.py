#!/usr/bin/env python3
"""
pub_fig_sublimation.py — Measured surface exchange beside the TE record.

Discussion figure (author-approved 2026-07-19): does independently measured
surface exchange line up with the 3 m flux -> d-excess transfer entropy
episodes? Three vertically stacked panels on a shared date axis:

  (a) Total snow depth (cm) from the SPLASH ASFS-30 SR-50A record
      (author swap 2026-07-19; was the daily depth-increase bars).
  (b) Daily sublimation/deposition rate (mm SWE/day) from the SPLASH
      Kettle Ponds 10 m EC latent heat flux (Meyers) — positive up =
      sublimation, negative = deposition. Independent tower and team from
      the SOS 3 m flux used as the TE driver.
  (c) Downwelling longwave radiation (W/m2, 6-h analyzed stream), the
      driver signal of the lagged channel (author add 2026-07-20).
  (d) The d-excess series itself (permil, 6-h analyzed stream from
      ``data/final_6hr.csv``, the frozen chain's real-unit record).
  (e) Transfer entropy to d-excess for BOTH the downwelling longwave
      (tau=1, the six-hour lag) and 3 m flux (tau=0) channels (author
      2026-07-19), % of per-window H(d-excess), UNSMOOTHED (author ruling
      2026-07-19: the 12-day median hides structure), with the per-window
      IAAFT band (max of the shown inputs' 95th percentiles) and an
      in-panel legend (author 2026-07-19; overrides the separate-legend
      convention for this panel).

Consistency check (logged + JSON artifact, not plotted): the 10 m EC
latent flux against the ASFS-30 bulk-aerodynamic ``bulk_Hl`` on the 6-h
grid (author ruling 2026-07-19: EC primary, bulk as check).

Inputs (run ``build_external_winter.py`` first):
  data/external/processed/kp10m_flux_winter_6h.csv
  data/external/processed/splash_winter_6h.csv
  data/final_6hr.csv
  publication_output/stage2/d_excess/stage2_rad_lw_down__t1.csv
  publication_output/stage2/d_excess/stage2_flux_3m_h2o__t0.csv

Outputs:
  publication_output/fig_sublimation.{png,pdf} (+ _legend companions)
  publication_output/fig_sublimation_check.json

Usage:
    python3 pub_fig_sublimation.py

Author: Matthew Rybecky
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pub_config import PublicationConfig
from pub_driver_series import entity_label
from pub_labels import target_label
from pub_selected_figures import SIG_BAND_STYLE, cross_style, stage2_series
from pub_style import (BAND_GRAY, date_axis, save_figure, save_legend,
                       setup_style)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET = 'd_excess'
# Style-priority order (headline first: solid, then dashed).
ENTITIES = ['rad_lw_down__t1',   # the six-hour-lag longwave channel
            'flux_3m_h2o__t0']   # the sublimation flux channel
PROCESSED = Path('data/external/processed')
ISOTOPE_CSV = Path('data/final_6hr.csv')   # frozen-chain real-unit stream
FIG_HEIGHT = 8.8        # inches; five stacked panels at full width
BAR_STYLE = dict(color='black', linewidth=0)
BAR_LEGEND = dict(patch=True, facecolor='black', edgecolor='none')
# Dense raw 6-h series: slightly lighter weight than the TE curve so the
# fine structure stays resolvable at print size.
SERIES_STYLE = dict(color='black', linestyle='-', linewidth=1.0)


# ═════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════
def load_daily_exchange(processed: Path) -> Tuple[pd.Series, pd.Series]:
    """Total snow depth (cm, 6-h) and daily sublimation rate (mm/day).

    Snow depth: the 6-h median SR-50 series as built. Sublimation rate:
    daily mean of the 6-h EC rate; NaN days draw no bar.
    """
    splash = pd.read_csv(processed / 'splash_winter_6h.csv',
                         parse_dates=['datetime']).set_index('datetime')
    kp10m = pd.read_csv(processed / 'kp10m_flux_winter_6h.csv',
                        parse_dates=['datetime']).set_index('datetime')
    subl_daily = kp10m['subl_mm_day'].resample('1D').mean()
    return splash['snow_depth_cm'], subl_daily


def load_analyzed(path: Path,
                  analysis_end: pd.Timestamp) -> Tuple[pd.Series, pd.Series]:
    """The analyzed 6-h d-excess and LW-down series, winter-truncated."""
    df = pd.read_csv(path, parse_dates=['time']).set_index('time')
    df = df[df.index < analysis_end]
    return df['d_excess'], df['rad_lw_down']


def te_curves(base: Path, analysis_end: pd.Timestamp
              ) -> Tuple[Dict[str, pd.Series], pd.Series]:
    """Raw per-window TE curves per entity and the combined IAAFT level.

    All from the Stage 2 fixed-entity CSVs, unsmoothed (author ruling
    2026-07-19: the 12-day median hides structure). The level is the
    per-window MAXIMUM of the shown entities' 95th percentiles (the
    standing family-maximum ruling).
    """
    curves: Dict[str, pd.Series] = {}
    levels = []
    for entity in ENTITIES:
        curve = stage2_series(base, TARGET, entity, 'pct_h')
        level = stage2_series(base, TARGET, entity, 'surr_p95_pct')
        if curve is None or level is None:
            raise FileNotFoundError(f"stage2/{TARGET}/stage2_{entity}.csv "
                                    "missing; run pub_stage2.py")
        curves[entity] = curve[curve.index < analysis_end]
        levels.append(level[level.index < analysis_end])
    combined = pd.concat(levels, axis=1).max(axis=1)
    return curves, combined


# ═════════════════════════════════════════════════════════════════════════
# EC vs bulk consistency check
# ═════════════════════════════════════════════════════════════════════════
def ec_bulk_check(processed: Path, out: Path) -> Dict:
    """Agreement of the 10 m EC latent flux with the ASFS-30 bulk flux.

    Overlapping 6-h bins only. Written to JSON so any caption statement
    traces to an artifact rather than a hand-typed number.
    """
    splash = pd.read_csv(processed / 'splash_winter_6h.csv',
                         parse_dates=['datetime']).set_index('datetime')
    kp10m = pd.read_csv(processed / 'kp10m_flux_winter_6h.csv',
                        parse_dates=['datetime']).set_index('datetime')
    both = pd.concat([kp10m['le_wm2'], splash['bulk_hl_wm2']],
                     axis=1).dropna()
    diff = both['le_wm2'] - both['bulk_hl_wm2']
    check = {
        'n_bins_6h': int(len(both)),
        'pearson_r': round(float(both.corr().iloc[0, 1]), 3),
        'mean_bias_ec_minus_bulk_wm2': round(float(diff.mean()), 2),
        'rmse_wm2': round(float(np.sqrt((diff ** 2).mean())), 2),
    }
    out.write_text(json.dumps(check, indent=2) + '\n')
    logger.info(f"EC vs bulk_Hl: {check}")
    return check


# ═════════════════════════════════════════════════════════════════════════
# Drawing
# ═════════════════════════════════════════════════════════════════════════
def draw(depth: pd.Series, subl: pd.Series, lwdn: pd.Series,
         dexcess: pd.Series, curves: Dict[str, pd.Series],
         level: pd.Series, out: Path) -> None:
    """Five stacked panels, shared date axis limited to the TE record."""
    figsize = setup_style('full', height=FIG_HEIGHT)
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=figsize)

    axs[0].plot(depth.index, depth, **SERIES_STYLE)
    axs[0].set_ylabel('Snow depth (cm)')
    axs[0].set_ylim(bottom=0.0)

    axs[1].bar(subl.index, subl, width=0.8, **BAR_STYLE)
    axs[1].axhline(0.0, color='black', linewidth=0.8)
    axs[1].set_ylabel('Sublimation rate\n(mm day$^{-1}$)')

    axs[2].plot(lwdn.index, lwdn, **SERIES_STYLE)
    axs[2].set_ylabel('LW$\\downarrow$ (W m$^{-2}$)')

    axs[3].plot(dexcess.index, dexcess, **SERIES_STYLE)
    axs[3].set_ylabel('d-excess (‰)')

    styles = {}
    for i, (entity, curve) in enumerate(curves.items()):
        styles[entity] = cross_style(i)
        axs[4].plot(curve.index, curve,
                    label=f'{entity_label(entity)} → {target_label(TARGET)}',
                    **styles[entity])
    axs[4].fill_between(level.index, 0.0, level, color=BAND_GRAY,
                        linewidth=0, zorder=0,
                        label='Below IAAFT 95% level')
    axs[4].legend(loc='upper right', fontsize=8, frameon=False)
    axs[4].set_ylabel(f'TE (% of H({target_label(TARGET)}))')
    axs[4].set_ylim(bottom=0.0)
    axs[4].set_xlabel('Date (2022-2023)')

    ref = next(iter(curves.values())).index
    tags = ['(a) Snow depth (SPLASH ASFS-30)',
            '(b) Sublimation / deposition (SPLASH 10 m EC)',
            '(c) Downwelling longwave radiation',
            '(d) Water vapor d-excess',
            '(e) Transfer entropy to d-excess']
    for ax, tag in zip(axs, tags):
        ax.text(0.02, 0.92, tag, transform=ax.transAxes, va='top',
                ha='left', fontsize=8)
        ax.set_xlim(ref[0], ref[-1])
    date_axis(axs[4])

    entries = [
        ('Snow depth (SR-50, 6-h median)', SERIES_STYLE),
        ('Sublimation (+) / deposition (-), 10 m EC latent heat flux',
         BAR_LEGEND),
        ('Downwelling longwave radiation (6-h analyzed series)',
         SERIES_STYLE),
        ('d-excess (6-h analyzed series)', SERIES_STYLE),
        *[(f'{entity_label(e)} → {target_label(TARGET)}', styles[e])
          for e in curves],
        ('Below IAAFT 95% level', SIG_BAND_STYLE),
    ]
    for p in [*save_figure(fig, out),
              *save_legend(entries, Path(f'{out}_legend'))]:
        logger.info(f"wrote {p}")


# ═════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measured surface exchange beside the TE record.")
    parser.add_argument('--processed-dir', default=str(PROCESSED))
    parser.add_argument('--output-base', default=None)
    parser.add_argument('--analysis-end', default=None)
    args = parser.parse_args()

    cfg = PublicationConfig()
    base = Path(args.output_base) if args.output_base else Path(cfg.output_base)
    analysis_end = pd.Timestamp(args.analysis_end if args.analysis_end
                                else cfg.analysis_end)
    processed = Path(args.processed_dir)

    depth, subl = load_daily_exchange(processed)
    dexcess, lwdn = load_analyzed(ISOTOPE_CSV, analysis_end)
    curves, level = te_curves(base, analysis_end)
    ec_bulk_check(processed, base / 'fig_sublimation_check.json')
    draw(depth, subl, lwdn, dexcess, curves, level,
         base / 'fig_sublimation')


if __name__ == '__main__':
    main()
