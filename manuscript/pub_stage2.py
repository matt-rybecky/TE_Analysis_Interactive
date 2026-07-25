#!/usr/bin/env python3
"""
pub_stage2.py — Targeted fixed-entity runner: rolling TE/JTE + per-window
IAAFT surrogates for pre-specified entities (Stage 2 of the two-stage
design; handoff items 1-2).

Purpose (D10 ruled 2026-07-09: manuscript entities only): the frozen v3
sweep holds per-window JTE for every combination but NO per-combination
surrogate distributions (its 2000 IAAFT went to the selection-inflated
composite). Anything presented inferentially in the paper gets a fresh,
pre-specified run here: original rolling TE (one source) or pure joint
TE (2+ sources), plus ``n_surrogates`` IAAFT surrogates of the source(s)
per window, at publication settings (30-day windows, h=1, KSG k=3,
base 2; circle-preserving multivariate surrogate for vector sources).

This runner also serves entities OUTSIDE the frozen tau sets (e.g.
``rad_lw_down__t2`` for the LW-down key figure: the local-state group
swept tau {0,1} only): it runs on the BASE beta file with the lag applied
internally via the per-source tau list, exactly like the publication
engine, so any base-disjoint entity at any tau is reachable without
touching the frozen artifacts.

Output, per entity: ``publication_output/stage2/<target>/
stage2_<entity>.csv`` with window center, TE bits, % of per-window
H(target), per-window surrogate mean/std/95th/99th (bits and % for the
95th), p-value, and significance flags.

The author runs this (analysis run — never run in-session). Runtime is
dominated by n_surrogates x n_windows x KSG cost; a single-source entity
at 2000 surrogates over ~649 windows takes very roughly 10-20 min on 8
cores; combinations longer with dimension.

Usage (example — the LW key-figure set):
    .venv/bin/python3 pub_stage2.py --target d_excess \
        --entities rad_lw_down__t0 rad_lw_down__t1 rad_lw_down__t2 \
                   rad_lw_up__t0 --n-cores 8
    .venv/bin/python3 pub_stage2.py --target dD \
        --entities rad_lw_down__t0 rad_lw_down__t1 rad_lw_down__t2 \
                   rad_lw_up__t0 --n-cores 8

Added 2026-07-11 (the SI S9 lag-refinement sweep): ``--start``/``--end``
restrict the run to windows whose CENTERS fall in [start, end) —
artifact-derived spans, never hand-picked — and ``--out-subdir`` writes
under ``publication_output/<subdir>/<target>/`` so runs at other data
resolutions (``--data-file data/final_4hr_beta.csv`` etc.) never collide
with the frozen 6-h ``stage2/`` CSVs. Tau steps are in units of the data
file's resolution (tau=1 at 4-h data = 4 hours).

``--window-days`` (added 2026-07-12) sets the rolling-window length in
DAYS explicitly. Without it a 30-day window is held fixed across
resolutions (the point count grows with frequency), so the finer sweeps
were 30-day-smoothed and the S9 zoom revealed no new structure. Scaling
the window down in step with resolution (30/20/10/5 d at 6/4/2/1 h) holds
a constant 120-point window, giving comparable estimator variance while
the physical window contracts so each finer panel resolves finer
temporal structure. The flag sets both the window and its H(target)
denominator together.

Author: Matthew Rybecky
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Modified from TE_V1.0.0 for the te-explorer repo layout; math unchanged.
# The engines of record live in ../core relative to this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))

from pub_config import PublicationConfig, TAU_DELIM, base_of
from TE_Calculator import TECalculator, build_col_map, extract_source
from TE_Surrogate import (SurrogateAnalyzer,
                          calculate_jte_single_window_standalone,
                          calculate_te_single_window_standalone,
                          generate_surrogate)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_entity(entity: str) -> Tuple[List[str], List[int]]:
    """Split ``base__tK[+base__tK...]`` into (bases, taus); base-disjoint."""
    bases, taus = [], []
    for part in entity.split('+'):
        base, tau = part.rsplit(TAU_DELIM, 1)
        bases.append(base)
        taus.append(int(tau))
    if len(set(bases)) != len(bases):
        raise ValueError(f"{entity}: members must be distinct bases "
                         "(the engine applies one lag per source column)")
    return bases, taus


def stage2_single_window(args: Tuple) -> Dict:
    """One window: original TE/JTE + surrogate distribution statistics.

    Modeled on the frozen engine's Phase-2 surrogate branch
    (``TE_Composite.composite_uncertainty_single_window``) but returning
    the full distribution statistics that branch discards. Vector
    (multi-column) sources route through ``generate_surrogate``'s
    multivariate path automatically.
    """
    (window_data, bases, target, tau_list, n_surr, surrogate_type,
     history_length, col_map, idx, timestamp) = args
    n_win = len(window_data)
    try:
        target_window = window_data[target].values
        sources = [extract_source(window_data, b, col_map, 0, n_win)
                   for b in bases]
        if len(sources) == 1:
            def value(src_list):
                return calculate_te_single_window_standalone(
                    src_list[0], target_window, tau_list[0],
                    history_length=history_length)
        else:
            def value(src_list):
                return calculate_jte_single_window_standalone(
                    src_list, target_window, 1, tau_list=tau_list,
                    history_length=history_length)
        original = value(sources)
        if n_surr == 0:                # TE-only sweep (no significance)
            return {'window_idx': idx, 'timestamp': timestamp,
                    'te_bits': float(original),
                    'surr_mean': np.nan, 'surr_std': np.nan,
                    'surr_p95': np.nan, 'surr_p99': np.nan,
                    'p_value': np.nan, 'sig95': False, 'sig99': False,
                    'n_valid_surrogates': 0}
        surr_vals = []
        for _ in range(n_surr):
            surr_vals.append(value([generate_surrogate(s, surrogate_type)
                                    for s in sources]))
        arr = np.asarray(surr_vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ValueError('no valid surrogates')
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        return {'window_idx': idx, 'timestamp': timestamp,
                'te_bits': float(original),
                'surr_mean': float(arr.mean()), 'surr_std': float(arr.std()),
                'surr_p95': p95, 'surr_p99': p99,
                'p_value': float((arr >= original).sum() / arr.size),
                'sig95': bool(original > p95),
                'sig99': bool(original > p99),
                'n_valid_surrogates': int(arr.size)}
    except Exception as exc:                     # window fails -> NaN row
        return {'window_idx': idx, 'timestamp': timestamp,
                'te_bits': np.nan, 'surr_mean': np.nan, 'surr_std': np.nan,
                'surr_p95': np.nan, 'surr_p99': np.nan, 'p_value': np.nan,
                'sig95': False, 'sig99': False, 'n_valid_surrogates': 0,
                'error': str(exc)}


def window_entropy(df: pd.DataFrame, target: str, time_col: str,
                   window_days: int) -> pd.Series:
    """Per-window H(target) in bits, indexed by window-center timestamp."""
    calc = TECalculator(n_cores=1, window_days=window_days)
    values, centers = calc.calculate_entropy_rolling(df, target,
                                                     time_col=time_col)
    return pd.Series(values,
                     index=pd.to_datetime(df[time_col].iloc[centers].values))


def run_entity(entity: str, target: str, df: pd.DataFrame, windows: List,
               entropy: pd.Series, cfg: PublicationConfig, args) -> Path:
    """Run one entity over all windows and write its Stage 2 CSV."""
    bases, taus = parse_entity(entity)
    col_map = build_col_map(bases, cfg.vector_bases())
    missing = [c for cols in col_map.values() for c in cols
               if c not in df.columns]
    if missing:
        raise ValueError(f"{entity}: data columns missing: {missing}")

    tasks = [(wd, bases, target, taus, args.n_surrogates, 'iaaft',
              cfg.history_length, col_map, i, ts)
             for i, (wd, _, ts) in enumerate(windows)]
    t0 = time.time()
    if args.n_cores > 1:
        with Pool(processes=args.n_cores) as pool:
            rows = pool.map(stage2_single_window, tasks)
    else:
        rows = [stage2_single_window(t) for t in tasks]
    out = pd.DataFrame(sorted(rows, key=lambda r: r['window_idx']))
    out['timestamp'] = pd.to_datetime(out['timestamp'])
    h = out['timestamp'].map(entropy).where(lambda s: s > 0)
    out['h_target_bits'] = h
    out['pct_h'] = out['te_bits'] / h * 100.0
    out['surr_p95_pct'] = out['surr_p95'] / h * 100.0

    out_dir = Path(cfg.output_base) / args.out_subdir / target
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'stage2_{entity}.csv'
    out.to_csv(path, index=False)
    n_sig = int(out['sig95'].sum())
    logger.info(f"{target} <- {entity}: {len(out)} windows in "
                f"{(time.time() - t0) / 60:.1f} min; sig95 in "
                f"{n_sig}/{len(out)} windows; wrote {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: fixed-entity rolling TE/JTE + per-window "
                    "IAAFT surrogates at publication settings.")
    parser.add_argument('--target', required=True)
    parser.add_argument('--entities', nargs='+', required=True,
                        help="Entities as base__tK[+base__tK...] "
                             "(distinct bases; any taus)")
    parser.add_argument('--data-file', default=None,
                        help='Default: pub_config data_file_base '
                             '(base beta file; lags applied internally)')
    parser.add_argument('--n-surrogates', type=int, default=None,
                        help='Default from pub_config (2000); 0 = TE-only '
                             'sweep, no surrogates (SI descriptive use, '
                             'e.g. the S9 lag refinement — much faster)')
    parser.add_argument('--n-cores', type=int, default=8)
    parser.add_argument('--time-col', default=None)
    parser.add_argument('--start', default=None,
                        help='Keep windows with center >= this timestamp '
                             '(artifact-derived span only)')
    parser.add_argument('--end', default=None,
                        help='Keep windows with center < this timestamp')
    parser.add_argument('--out-subdir', default='stage2',
                        help="Output namespace under publication_output/ "
                             "(default 'stage2'; use e.g. 'stage2_4hr' for "
                             "other data resolutions)")
    parser.add_argument('--window-days', type=int, default=None,
                        help='Rolling window length in DAYS (default from '
                             'pub_config, 30). Scale it DOWN with data '
                             'resolution for the SI S9 zoom cascade so the '
                             'finer panels resolve finer structure at a '
                             'constant sample count: 30/20/10/5 d at '
                             '6/4/2/1 h all hold 120 points per window. '
                             'Sets both the window and its H(target) '
                             'denominator, so pct_h stays consistent.')
    args = parser.parse_args()

    cfg = PublicationConfig()
    if args.n_surrogates is None:
        args.n_surrogates = cfg.n_surrogates
    if args.window_days is None:
        args.window_days = cfg.window_days
    data_file = args.data_file or cfg.data_file_base
    time_col = args.time_col or cfg.time_col

    df = pd.read_csv(data_file)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    logger.info(f"{data_file}: {len(df)} rows; target {args.target}; "
                f"{args.n_surrogates} IAAFT/window; h={cfg.history_length}")

    helper = SurrogateAnalyzer(n_cores=args.n_cores)
    _, window_points = helper.determine_data_frequency(df, time_col,
                                                       args.window_days)
    windows = helper.create_rolling_windows(df, window_points, time_col)
    if args.start or args.end:
        lo = pd.Timestamp(args.start) if args.start else pd.Timestamp.min
        hi = pd.Timestamp(args.end) if args.end else pd.Timestamp.max
        windows = [w for w in windows if lo <= pd.Timestamp(w[2]) < hi]
        logger.info(f"span filter [{args.start}, {args.end}): "
                    f"{len(windows)} windows retained")
    entropy = window_entropy(df, args.target, time_col, args.window_days)
    logger.info(f"{len(windows)} windows of {window_points} points "
                f"({args.window_days}-day window)")

    for entity in args.entities:
        run_entity(entity, args.target, df, windows, entropy, cfg, args)


if __name__ == '__main__':
    main()
