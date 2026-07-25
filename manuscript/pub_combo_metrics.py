#!/usr/bin/env python3
"""
pub_combo_metrics.py — Secondary-statistic primitives for combination ranking.

Pure, array-level functions consumed by ``pub_combo_stats.py``. Each combination's
per-window transfer-entropy series (from ``combo_table.parquet``) is reduced to
scalar descriptors that expose a distinct kind of "interesting": persistent
strength (integrated TE), episodic strength (variability, burstiness), genuinely
joint information (synergy gain), interaction dynamics (category flips), and
data-driven active periods (episode detection). No transfer entropy is recomputed.

All reductions are NaN-aware. Transfer entropy from the KSG estimator can be
slightly negative; burstiness treats information as non-negative and clips at zero
(documented per function), while means and sums keep the raw sign.

Author: Matthew Rybecky
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Interaction categories emitted by the composite engine.
CATEGORIES: Tuple[str, ...] = ('synergistic', 'redundant', 'obfuscating')


def gini(values: np.ndarray) -> float:
    """
    Gini coefficient of a non-negative series: temporal concentration.

    Parameters
    ----------
    values : np.ndarray
        Per-window values. NaNs are dropped; negatives are clipped to zero
        (information content is treated as non-negative for concentration).

    Returns
    -------
    float
        0 for a perfectly flat series (information spread evenly across windows),
        approaching 1 for information concentrated in a single window. NaN if no
        finite values or the total is zero.

    Notes
    -----
    Burstiness discriminator: high Gini marks episodic combinations whose signal
    lives in a few sharp events (storms, fronts), the regime this analysis mines
    for subtle conclusions.
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float('nan')
    v = np.clip(v, 0.0, None)
    total = v.sum()
    if total <= 0:
        return float('nan')
    v = np.sort(v)
    n = v.size
    index = np.arange(1, n + 1)
    return float((2.0 * (index * v).sum()) / (n * total) - (n + 1.0) / n)


def coefficient_of_variation(values: np.ndarray) -> float:
    """
    Std/mean of a series: scale-free variability.

    Parameters
    ----------
    values : np.ndarray
        Per-window values; NaNs dropped.

    Returns
    -------
    float
        Population std divided by mean, or NaN if fewer than two finite points or
        the mean is not safely positive (CV is ill-defined near a zero mean).
    """
    v = values[np.isfinite(values)]
    if v.size < 2:
        return float('nan')
    mean = v.mean()
    if abs(mean) < 1e-12:
        return float('nan')
    return float(v.std() / mean)


@dataclass
class SeriesStats:
    """Scalar descriptors of one per-window value series."""

    n: int
    total: float
    mean: float
    peak: float
    peak_index: int
    std: float
    cv: float
    gini: float


def reduce_series(values: np.ndarray) -> SeriesStats:
    """
    Reduce a per-window value series to its scalar descriptors.

    Parameters
    ----------
    values : np.ndarray
        Per-window values (bits or percent of entropy). NaN-aware.

    Returns
    -------
    SeriesStats
        Count, integrated total, mean, peak and its index, std, CV, Gini. The
        peak index is into the ORIGINAL array (for timestamp lookup); NaNs are
        ignored when locating the peak.
    """
    finite = np.isfinite(values)
    n = int(finite.sum())
    if n == 0:
        return SeriesStats(0, float('nan'), float('nan'), float('nan'), -1,
                           float('nan'), float('nan'), float('nan'))
    v = values[finite]
    peak_index = int(np.nanargmax(np.where(finite, values, -np.inf)))
    return SeriesStats(
        n=n,
        total=float(v.sum()),
        mean=float(v.mean()),
        peak=float(v.max()),
        peak_index=peak_index,
        std=float(v.std()),
        cv=coefficient_of_variation(values),
        gini=gini(values),
    )


def category_fractions(categories: np.ndarray) -> dict:
    """
    Fraction of windows in each interaction category and the flip count.

    Parameters
    ----------
    categories : np.ndarray of str
        Per-window interaction label ('synergistic'/'redundant'/'obfuscating').

    Returns
    -------
    dict
        ``frac_<category>`` for each of the three categories plus ``n_flips``,
        the number of window-to-window category changes (regime-switching
        combinations flip often).
    """
    out = {f'frac_{c}': float('nan') for c in CATEGORIES}
    cats = categories[categories != None]  # noqa: E711 (object array vs None)
    if cats.size == 0:
        out['n_flips'] = 0
        return out
    for c in CATEGORIES:
        out[f'frac_{c}'] = float((cats == c).mean())
    out['n_flips'] = int((cats[1:] != cats[:-1]).sum())
    return out


@dataclass
class Episode:
    """One data-driven active period discovered in a target's winner envelope."""

    start_index: int
    end_index: int
    peak_index: int
    peak_value: float
    n_windows: int


def _close_gaps(above: np.ndarray, valid: np.ndarray, max_gap: int) -> np.ndarray:
    """
    Bridge short below-threshold dips inside a run of active windows.

    A gap is a maximal run of ``~above`` windows flanked by ``above`` on both
    sides. It is filled (treated as active) only when it spans at most
    ``max_gap`` windows AND every window in it is ``valid``. The validity guard
    prevents merging across an excluded span (e.g. the diurnal window), so two
    real episodes on either side of it stay separate.

    Parameters
    ----------
    above : np.ndarray of bool
        Per-window above-threshold flags.
    valid : np.ndarray of bool
        Per-window eligibility flags (False windows are never bridged).
    max_gap : int
        Largest gap, in windows, to bridge. ``0`` disables merging.

    Returns
    -------
    np.ndarray of bool
        A copy of ``above`` with qualifying interior gaps set True.
    """
    if max_gap <= 0:
        return above
    merged = above.copy()
    n = above.size
    i = 0
    while i < n and not above[i]:
        i += 1  # skip leading inactive windows (no left flank to bridge from)
    while i < n:
        if above[i]:
            i += 1
            continue
        j = i
        while j < n and not above[j]:
            j += 1
        if j < n and (j - i) <= max_gap and valid[i:j].all():
            merged[i:j] = True  # interior gap flanked by active windows
        i = j
    return merged


def detect_episodes(series: np.ndarray,
                    k: float = 1.0,
                    min_windows: int = 2,
                    max_gap: int = 0,
                    valid_mask: np.ndarray = None,
                    abs_floor: float = None
                    ) -> Tuple[np.ndarray, List[Episode]]:
    """
    Discover active periods from a series with a robust, data-driven threshold.

    Parameters
    ----------
    series : np.ndarray
        Per-window envelope (typically the composite winner series). NaN-aware.
    k : float
        Threshold sensitivity. The cut is ``median + k * 1.4826 * MAD`` (the MAD
        scaled to a standard-deviation equivalent). Larger k = fewer, sharper
        episodes.
    min_windows : int
        Minimum contiguous windows above threshold to count as an episode
        (applied after gap merging).
    max_gap : int
        Largest below-threshold dip, in windows, to bridge so a physically
        continuous band is not fragmented into many short episodes. ``0``
        (default) keeps the original strict-contiguity behavior.
    valid_mask : np.ndarray of bool, optional
        Per-window eligibility. When given, the threshold is computed only over
        valid windows, episodes may form only in valid windows, and gaps that
        touch an invalid window are never bridged. Use it to exclude a known
        artifact span (e.g. the late-spring diurnal window) from detection.
    abs_floor : float, optional
        Absolute lower bound on the episode threshold, in the same units as
        ``series``. The effective cut is ``max(median + k*1.4826*MAD,
        abs_floor)``. Use it so only bursts of genuine magnitude qualify (e.g. a
        driver reaching a meaningful percent of target entropy), regardless of
        how flat that series' own baseline is. This is the "high explanatory
        power wins" gate for per-driver/per-combination episodicity.

    Returns
    -------
    mask : np.ndarray of bool
        True where the series is inside a retained episode (aligned to input).
    episodes : list of Episode
        Retained episodes in time order, with index bounds and peak.

    Notes
    -----
    Replaces hand-picked regime dates: the episodes emerge from the information
    transfer itself. A robust (median/MAD) threshold resists the late-spring
    diurnal inflation that would drag a mean/std cut upward; excluding that span
    via ``valid_mask`` removes it from the threshold statistics entirely.
    """
    mask = np.zeros(series.shape, dtype=bool)
    valid = (np.ones(series.shape, dtype=bool) if valid_mask is None
             else np.asarray(valid_mask, dtype=bool))
    eligible = np.isfinite(series) & valid
    finite = series[eligible]
    if finite.size == 0:
        return mask, []

    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    threshold = median + k * 1.4826 * mad
    if abs_floor is not None:
        threshold = max(threshold, float(abs_floor))

    above = eligible & (series > threshold)
    above = _close_gaps(above, valid, max_gap)
    episodes: List[Episode] = []
    i, n = 0, series.size
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1]:
            j += 1
        if (j - i + 1) >= min_windows:
            seg = np.where(np.isfinite(series[i:j + 1]), series[i:j + 1], -np.inf)
            peak_local = int(np.argmax(seg))
            episodes.append(Episode(
                start_index=i, end_index=j,
                peak_index=i + peak_local,
                peak_value=float(series[i + peak_local]),
                n_windows=j - i + 1))
            mask[i:j + 1] = True
        i = j + 1
    return mask, episodes


def episode_alignment(values: np.ndarray, episode_mask: np.ndarray,
                      valid_mask: np.ndarray = None) -> float:
    """
    Ratio of a combination's mean value inside episodes to outside.

    Parameters
    ----------
    values : np.ndarray
        The combination's per-window series (NaN-aware).
    episode_mask : np.ndarray of bool
        Target-level active-period mask from :func:`detect_episodes`.
    valid_mask : np.ndarray of bool, optional
        Per-window eligibility. When given, the "outside" baseline is restricted
        to valid windows, so an excluded artifact span (e.g. the diurnal window)
        does not inflate the denominator and depress every ratio.

    Returns
    -------
    float
        ``mean(value | in episode) / mean(value | out of episode)``. Greater than
        one means the combination concentrates its information in the target's
        active periods. NaN if either side lacks finite data or the outside mean
        is not safely positive.
    """
    finite = np.isfinite(values)
    valid = (np.ones(values.shape, dtype=bool) if valid_mask is None
             else np.asarray(valid_mask, dtype=bool))
    inside = finite & episode_mask
    outside = finite & ~episode_mask & valid
    if inside.sum() == 0 or outside.sum() == 0:
        return float('nan')
    mean_out = values[outside].mean()
    if abs(mean_out) < 1e-12:
        return float('nan')
    return float(values[inside].mean() / mean_out)
