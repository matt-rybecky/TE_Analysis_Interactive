#!/usr/bin/env python3
"""
pub_driver_series.py — Lag-level driver series, continuous + episodic split.

First stage of the period-attribution layer (replaces peak-first discovery).
Every variable-at-lag is its own unique input: each single candidate
(``base__tK``) and every 2-/3-way combination from the run artifacts enters the
search on equal footing, at full lag resolution (~4,800 entities per target).
Deduplication of the same variables at different taus happens ONLY at
presentation, in pub_period_attribution's winner selection — never here.

Each entity's pure-joint-TE series (percent of the target's per-window entropy)
is decomposed into a CONTINUOUS baseline (centered rolling median: slowly
varying explanatory power) and an EPISODIC excess above that baseline.
Downstream, pub_periods segments the winter on the single-candidate baseline
matrix and pub_period_attribution selects 1-3 justified, base-disjoint winners
plus an episodic slot in every period.

Winter-truncated (< analysis_end). Pure post-processing; no TE is recomputed.

Outputs per target (``publication_output/<target>/``):
  driver_series_raw.parquet       — pct_jte [timestamp x entity]
  driver_series_baseline.parquet  — continuous component (rolling median)
  driver_series_excess.parquet    — episodic component, max(0, raw - baseline)
  driver_entities.csv             — entity key, kind, bases, taus, label
  driver_singles_baseline.csv     — readable single-candidate baseline matrix

Usage:
    python3 pub_driver_series.py
    python3 pub_driver_series.py --baseline-days 12

Author: Matthew Rybecky
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from pub_combo_stats import load_target_frames
from pub_config import PublicationConfig, base_of
from pub_labels import combo_label, tau_of

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

KIND_NAMES = {1: 'single', 2: 'pair', 3: 'triple'}


# ═════════════════════════════════════════════════════════════════════════
# Entity helpers (entity key = the run-artifact combo key, lag-level)
# ═════════════════════════════════════════════════════════════════════════
def entity_bases(entity: str) -> List[str]:
    """Distinct base variables of a lag-level entity key, sorted."""
    return sorted({base_of(p) for p in entity.split('+')})


def entity_kind(entity: str) -> str:
    """'single', 'pair', or 'triple' from the number of members."""
    return KIND_NAMES[entity.count('+') + 1]


def entity_label(entity: str) -> str:
    """Readable short label with taus attached, e.g. 'Temp (τ=1) + LW↑ (τ=0)'."""
    return combo_label(entity, short=True)


# ═════════════════════════════════════════════════════════════════════════
# Assembly and decomposition
# ═════════════════════════════════════════════════════════════════════════
def build_activity(long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pivot the unified per-window frame to the lag-level activity matrix.

    Parameters
    ----------
    long : pd.DataFrame
        Unified frame from ``load_target_frames`` (columns ``timestamp``,
        ``combo``, ``pct_jte``; singles included as size-one combos).

    Returns
    -------
    (wide, entities) : tuple of pd.DataFrame
        ``wide`` is [timestamp x entity] pure joint TE in percent of target
        entropy, one column per lag-level entity. ``entities`` is the catalog
        (key, kind, bases, taus, label).
    """
    wide = long.pivot_table(index='timestamp', columns='combo',
                            values='pct_jte')
    entities = pd.DataFrame({
        'entity': wide.columns,
        'kind': [entity_kind(e) for e in wide.columns],
        'n_bases': [len(entity_bases(e)) for e in wide.columns],
        'bases': [';'.join(entity_bases(e)) for e in wide.columns],
        'taus': [';'.join(str(tau_of(p)) for p in e.split('+'))
                 for e in wide.columns],
        'label': [entity_label(e) for e in wide.columns],
    }).sort_values(['n_bases', 'entity'], ignore_index=True)
    return wide, entities


def fit_baseline(wide: pd.DataFrame, n_windows: int) -> pd.DataFrame:
    """
    Continuous explanatory power: centered rolling median per entity.

    Parameters
    ----------
    wide : pd.DataFrame
        Activity matrix [timestamp x entity].
    n_windows : int
        Median window length in 6-hour windows (e.g. 48 = 12 days). The median
        (not the mean) keeps multi-day bursts from lifting the baseline, so the
        episodic excess is measured against the persistent level.

    Returns
    -------
    pd.DataFrame
        Baseline matrix, same shape/index as ``wide``. Edges use a shrinking
        window (``min_periods = n_windows // 3``).
    """
    min_periods = max(3, n_windows // 3)
    return wide.rolling(n_windows, center=True, min_periods=min_periods).median()


# ═════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════
def process_target(target: str, base: Path, baseline_windows: int,
                   analysis_end: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Assemble, decompose, and persist the driver series for one target."""
    target_dir = base / target
    loaded = load_target_frames(target_dir, analysis_end)
    if loaded is None:
        return None
    long = loaded[0]

    raw, entities = build_activity(long)
    baseline = fit_baseline(raw, baseline_windows)
    excess = (raw - baseline).clip(lower=0.0)

    raw.to_parquet(target_dir / 'driver_series_raw.parquet')
    baseline.to_parquet(target_dir / 'driver_series_baseline.parquet')
    excess.to_parquet(target_dir / 'driver_series_excess.parquet')
    entities.to_csv(target_dir / 'driver_entities.csv', index=False)

    singles = [e for e in raw.columns if '+' not in e]
    baseline[singles].round(3).to_csv(target_dir / 'driver_singles_baseline.csv')

    logger.info(f"{target}: {len(raw.columns)} lag-level entities "
                f"({len(singles)} singles) x {len(raw)} winter windows, "
                f"baseline={baseline_windows} windows")
    return entities


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lag-level driver series with continuous/episodic split.")
    parser.add_argument('--output-base', default=None)
    parser.add_argument('--baseline-days', type=float, default=12.0,
                        help='Rolling-median window for the continuous '
                             'baseline, in days (default 12 = 48 windows)')
    parser.add_argument('--analysis-end', default=None,
                        help='Winter cutoff (YYYY-MM-DD); default from pub_config')
    args = parser.parse_args()

    cfg = PublicationConfig()
    base = Path(args.output_base) if args.output_base else Path(cfg.output_base)
    analysis_end = pd.Timestamp(args.analysis_end if args.analysis_end
                                else cfg.analysis_end)
    baseline_windows = max(8, int(round(args.baseline_days * 4)))

    for target in cfg.targets:
        process_target(target, base, baseline_windows, analysis_end)


if __name__ == '__main__':
    main()
