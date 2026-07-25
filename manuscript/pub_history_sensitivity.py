#!/usr/bin/env python3
"""
pub_history_sensitivity.py — Transfer entropy history-length (h) sensitivity sweep.

Tests whether the publication results depend on the choice of conditioning
history length h = 1 (traditional TE, Schreiber 2000) by recomputing rolling
TE for representative source-target pairs at h = 1, 2, 3, 4 and comparing
magnitude and temporal structure across h. Supports the manuscript supplement:
the paper states h = 1; this sweep documents the sensitivity of that choice.

Pairs span the analysis's physical range: two deterministic controls
(Stefan-Boltzmann, Clausius-Clapeyron), the persistent equilibrium pathway
(temperature -> dD), a far-field pathway (500-hPa GPH -> d-excess), and three
local/episodic pathways (wind speed, H2O flux, circular wind direction ->
d-excess). Wind direction runs through the 2D vector-source path.

Interpretation aids emitted per pair and h: mean/median/std/max TE, Spearman
rank correlation of the TE time series against the h = 1 series (does the
temporal structure persist?), the top-decile episode-overlap (Jaccard index of
the top-10% TE windows at h vs at h = 1; robust to the zero-floor ties that
degrade Spearman for episodic pairs), and the KSG rule-of-thumb sample check
N_min ~ k * 2^d against the window size (larger h raises the joint dimension;
undersampled rows are flagged, not hidden).

Outputs (publication_output/history_sensitivity/):
  - history_sensitivity_timeseries.csv  (long format, one row per pair/h/window)
  - history_sensitivity_summary.csv     (one row per pair/h)
  - history_sensitivity.png / .pdf      (small-multiple time series by pair)

Run:      .venv/bin/python3 pub_history_sensitivity.py
Runtime:  ~2-5 minutes, single core (~19,000 KSG calls on ~120-point windows).
Success:  both CSVs written, figure written, and a printed summary table with
          finite TE values for every pair at every h.

Author: Matthew Rybecky
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Modified from TE_V1.0.0 for the te-explorer repo layout; math unchanged.
# The engines of record live in ../core relative to this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))

from TE_Calculator import TECalculator, build_col_map, extract_source
from pub_config import VECTOR_BASES, PublicationConfig

plt.style.use('seaborn-v0_8-whitegrid')

# ═════════════════════════════════════════════════════════════════════════
# Sweep specification
# ═════════════════════════════════════════════════════════════════════════
H_VALUES: tuple = (1, 2, 3, 4)

# KSG neighbor count is fixed at k=3 inside TE_Calculator (NPEET calls),
# matching the frozen publication spec. Used here only for the sample check.
KSG_K = 3

PAIRS: List[Dict] = [
    {'source': 'rad_lw_up', 'target': 'met_temp', 'tau': 0,
     'label': 'LW$_{up}$ → T (Stefan-Boltzmann control)'},
    {'source': 'met_rh', 'target': 'H2O_ppm', 'tau': 0,
     'label': 'RH → H$_2$O (Clausius-Clapeyron control)'},
    {'source': 'met_temp', 'target': 'dD', 'tau': 0,
     'label': 'T → δD (equilibrium pathway)'},
    {'source': 'isen_500_gph', 'target': 'd_excess', 'tau': 1,
     'label': 'GPH$_{500}$ → d-excess (far-field)'},
    {'source': 'met_wspd', 'target': 'd_excess', 'tau': 1,
     'label': 'Wind speed → d-excess (episodic)'},
    {'source': 'flux_3m_h2o', 'target': 'd_excess', 'tau': 0,
     'label': 'H$_2$O flux 3m → d-excess (episodic)'},
    {'source': 'met_wdir', 'target': 'd_excess', 'tau': 1,
     'label': 'Wind direction → d-excess (vector source)'},
]

# B&W line styles keyed by h.
H_STYLES: Dict[int, Dict] = {
    1: {'color': 'black', 'linestyle': '-', 'linewidth': 1.4},
    2: {'color': 'black', 'linestyle': '--', 'linewidth': 1.0},
    3: {'color': '0.35', 'linestyle': '-.', 'linewidth': 1.0},
    4: {'color': '0.55', 'linestyle': ':', 'linewidth': 1.2},
}


def load_base_data(cfg: PublicationConfig) -> pd.DataFrame:
    """
    Load the unlagged circular beta-normalized dataset.

    Parameters
    ----------
    cfg : PublicationConfig
        Frozen publication spec (provides file path and time column).

    Returns
    -------
    df : pd.DataFrame
        Data with parsed datetime time column.
    """
    # Repo-layout edit: data/ lives at the repo root, one level above manuscript/.
    path = Path(__file__).resolve().parents[1] / cfg.data_file_base
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run build_circular_data.py first.")
    df = pd.read_csv(path)
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col])
    return df


def sweep_pair(df: pd.DataFrame, pair: Dict, cfg: PublicationConfig,
               col_map: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Compute rolling TE for one source-target pair at every history length.

    Parameters
    ----------
    df : pd.DataFrame
        Base dataset.
    pair : dict
        Pair spec with 'source', 'target', 'tau', 'label'.
    cfg : PublicationConfig
        Frozen spec (window length, time column).
    col_map : dict
        Logical-name -> data-column(s) map (vector sources expand to 2D).

    Returns
    -------
    ts : pd.DataFrame
        Long-format rows: pair label, source, target, tau, h, window time,
        te_bits.
    """
    records = []
    for h in H_VALUES:
        calc = TECalculator(n_cores=1, window_days=cfg.window_days,
                            tau=pair['tau'], history_length=h)
        _, window_points = calc.determine_data_frequency(df, cfg.time_col)
        centers, _ = calc.create_rolling_windows(df, window_points)
        half = window_points // 2

        target_all = df[pair['target']].to_numpy(dtype=float)
        times = df[cfg.time_col].to_numpy()

        for center in centers:
            start, end = center - half, center + half
            source_win = extract_source(df, pair['source'], col_map, start, end)
            te = calc.calculate_te_single_window(source_win, target_all[start:end])
            records.append({
                'pair': pair['label'], 'source': pair['source'],
                'target': pair['target'], 'tau': pair['tau'], 'h': h,
                'time': times[center], 'te_bits': te,
                'window_points': window_points,
            })
    return pd.DataFrame.from_records(records)


def _top_decile_mask(series: np.ndarray, q: float = 0.9) -> np.ndarray:
    """
    Boolean mask of windows at or above the q-quantile of a TE series.

    Parameters
    ----------
    series : np.ndarray
        Per-window TE values (time-ordered).
    q : float, optional
        Quantile defining "episode" windows. Defaults to 0.9 (top decile).

    Returns
    -------
    mask : np.ndarray of bool
        True where the window is in the top (1 - q) fraction.
    """
    return series >= np.quantile(series, q)


def _episode_jaccard(base: np.ndarray, series: np.ndarray) -> float:
    """
    Jaccard overlap of top-decile windows between two TE series.

    Asks directly whether the episodes interpreted at h = 1 survive deeper
    conditioning: 1.0 means identical episode windows, 0.0 means disjoint.

    Parameters
    ----------
    base, series : np.ndarray
        Time-aligned per-window TE values (base is the h = 1 reference).

    Returns
    -------
    jaccard : float
        Intersection over union of the two top-decile window sets.
    """
    m1, m2 = _top_decile_mask(base), _top_decile_mask(series)
    union = np.sum(m1 | m2)
    return float(np.sum(m1 & m2) / union) if union else np.nan


def summarize(ts: pd.DataFrame, col_map: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Reduce the sweep to one row per pair and history length.

    Reports magnitude statistics, Spearman rank correlation of each h > 1
    series against the h = 1 series (temporal-structure persistence), the
    top-decile episode-overlap Jaccard index against h = 1 (episode-timing
    persistence, robust to zero-floor ties), and the KSG rule-of-thumb sample
    check for the joint dimension at each h.

    Parameters
    ----------
    ts : pd.DataFrame
        Long-format sweep output of :func:`sweep_pair`.
    col_map : dict
        Logical-name -> data-column(s) map (for source dimensionality).

    Returns
    -------
    summary : pd.DataFrame
        One row per (pair, h).
    """
    rows = []
    for pair_label, grp in ts.groupby('pair', sort=False):
        base = grp[grp['h'] == 1].sort_values('time')['te_bits'].to_numpy()
        source = grp['source'].iloc[0]
        src_dim = len(col_map.get(source, [source]))
        window_points = int(grp['window_points'].iloc[0])
        for h, sub in grp.groupby('h', sort=True):
            series = sub.sort_values('time')['te_bits'].to_numpy()
            rho = 1.0 if h == 1 else float(spearmanr(base, series).statistic)
            joint_dim = src_dim + 1 + h  # source cols + target present + history
            n_min = KSG_K * 2 ** joint_dim
            rows.append({
                'pair': pair_label, 'source': source,
                'target': sub['target'].iloc[0], 'tau': sub['tau'].iloc[0],
                'h': h, 'n_windows': len(series),
                'mean_te_bits': float(np.mean(series)),
                'median_te_bits': float(np.median(series)),
                'std_te_bits': float(np.std(series)),
                'max_te_bits': float(np.max(series)),
                'mean_ratio_vs_h1': float(np.mean(series) / np.mean(base))
                if np.mean(base) > 0 else np.nan,
                'spearman_vs_h1': rho,
                'episode_jaccard_vs_h1': 1.0 if h == 1
                else _episode_jaccard(base, series),
                'joint_dim': joint_dim,
                'window_points': window_points,
                'ksg_n_min_rule': n_min,
                'undersampled_flag': window_points < n_min,
            })
    return pd.DataFrame(rows)


def plot_sweep(ts: pd.DataFrame, out_dir: Path) -> None:
    """
    Small-multiple TE time series, one panel per pair, line style by h.

    Parameters
    ----------
    ts : pd.DataFrame
        Long-format sweep output.
    out_dir : Path
        Output directory for PNG and PDF.
    """
    pairs = list(dict.fromkeys(ts['pair']))
    n = len(pairs)
    ncols, nrows = 2, int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, pair_label in zip(axes, pairs):
        grp = ts[ts['pair'] == pair_label]
        for h in H_VALUES:
            sub = grp[grp['h'] == h].sort_values('time')
            ax.plot(sub['time'], sub['te_bits'], label=f'h = {h}',
                    **H_STYLES[h])
        ax.set_title(pair_label, fontsize=13)
        ax.tick_params(labelsize=10)
    for ax in axes[n:]:
        ax.set_visible(False)

    axes[0].legend(fontsize=10, frameon=True, loc='upper right')
    fig.supylabel('Transfer entropy (bits)', fontsize=16)
    fig.supxlabel('Date', fontsize=16)
    fig.suptitle('TE history-length sensitivity (30-day rolling windows, '
                 '6-h resolution, KSG k=3)', fontsize=18)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out_dir / f'history_sensitivity.{ext}', dpi=300,
                    bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    """Run the full sweep, write artifacts, and print the summary table."""
    cfg = PublicationConfig()
    # Repo-layout edit: publication_output/ lives at the repo root.
    out_dir = Path(__file__).resolve().parents[1] / cfg.output_base / 'history_sensitivity'
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_base_data(cfg)
    col_map = build_col_map([p['source'] for p in PAIRS], VECTOR_BASES)

    frames = []
    for pair in PAIRS:
        print(f"Sweeping h={H_VALUES} for {pair['source']} -> "
              f"{pair['target']} (tau={pair['tau']}) ...")
        frames.append(sweep_pair(df, pair, cfg, col_map))
    ts = pd.concat(frames, ignore_index=True)

    summary = summarize(ts, col_map)
    ts.to_csv(out_dir / 'history_sensitivity_timeseries.csv', index=False)
    summary.to_csv(out_dir / 'history_sensitivity_summary.csv', index=False)
    plot_sweep(ts, out_dir)

    with pd.option_context('display.width', 160, 'display.max_columns', None):
        print("\n=== History-length sensitivity summary ===")
        print(summary[['pair', 'h', 'mean_te_bits', 'mean_ratio_vs_h1',
                       'spearman_vs_h1', 'episode_jaccard_vs_h1', 'joint_dim',
                       'ksg_n_min_rule', 'undersampled_flag']].to_string(index=False))
    print(f"\nArtifacts written to {out_dir}")


if __name__ == '__main__':
    main()
