"""Temporal alignment of record: Gaussian-weighted resampling to a uniform grid.

The functions here reimplement the Stage 1 alignment from the original
analysis pipeline (``generate_extended_datasets.py``, not shipped in this
repo): Gaussian-weighted means for
environmental (intensive) variables, window sums for flux (extensive)
variables, circular means for directional variables, and conservative linear
interpolation across short gaps only. The math is unchanged; this module
adds a single ``resample_to_grid`` entry point so any loaded dataset can be
aligned with one call.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


def epoch_seconds(times) -> np.ndarray:
    """Convert datetimes to float seconds since the epoch, unit-robust.

    The pipeline of record assumed nanosecond datetime64 storage
    (``astype(int64) / 1e9``); pandas 3 defaults to microseconds, which
    silently shifted that conversion by a factor of 1000 and emptied every
    resampling window. Forcing nanoseconds first makes the conversion
    correct under every pandas version. The resampling math is unchanged.

    Parameters
    ----------
    times : pd.Series, pd.DatetimeIndex, or array-like of datetimes
        Timestamps to convert.

    Returns
    -------
    np.ndarray
        Seconds since 1970-01-01 as float64.
    """
    values = pd.to_datetime(times).to_numpy(dtype='datetime64[ns]')
    return values.astype(np.int64) / 1e9


def gaussian_weights(time_points: np.ndarray, center_time: float,
                     sigma: float) -> np.ndarray:
    """Normalized Gaussian weights of ``time_points`` around ``center_time``.

    Parameters
    ----------
    time_points : np.ndarray
        Sample times in seconds.
    center_time : float
        Kernel center in seconds.
    sigma : float
        Kernel width in seconds.

    Returns
    -------
    np.ndarray
        Weights summing to 1 (or all zero if the kernel has no support).
    """
    weights = np.exp(-0.5 * ((time_points - center_time) / sigma) ** 2)
    s = weights.sum()
    if s > 0:
        weights = weights / s
    return weights


def circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    """Weighted circular mean of angles in degrees.

    Parameters
    ----------
    angles : np.ndarray
        Angles in degrees; NaN allowed.
    weights : np.ndarray
        Non-negative weights, same length as ``angles``.

    Returns
    -------
    float
        Mean direction in degrees [0, 360), or NaN if no valid samples.
    """
    if len(angles) == 0 or np.isnan(angles).all():
        return np.nan
    valid = ~np.isnan(angles)
    if not valid.any():
        return np.nan
    angles_rad = np.deg2rad(angles[valid])
    w = weights[valid]
    if w.sum() == 0:
        return np.nan
    mean_sin = np.average(np.sin(angles_rad), weights=w)
    mean_cos = np.average(np.cos(angles_rad), weights=w)
    return np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360


def gaussian_resample_series(time_data: pd.Series, value_data: pd.Series,
                             target_times: pd.DatetimeIndex,
                             sigma_seconds: float, interval_seconds: float,
                             is_circular: bool = False,
                             is_flux: bool = False) -> pd.Series:
    """Resample one series onto ``target_times`` (alignment of record).

    Environmental variables receive a Gaussian-weighted mean over a
    ``4 * sigma`` window; circular variables a Gaussian-weighted circular
    mean; flux variables a plain sum over the half-interval window
    (extensive aggregation).

    Parameters
    ----------
    time_data : pd.Series
        Source timestamps (datetime64).
    value_data : pd.Series
        Source values aligned with ``time_data``.
    target_times : pd.DatetimeIndex
        Uniform output grid.
    sigma_seconds : float
        Gaussian kernel width in seconds.
    interval_seconds : float
        Grid interval in seconds (window for flux sums).
    is_circular : bool
        Treat values as directions in degrees.
    is_flux : bool
        Treat values as extensive (sum, not mean).

    Returns
    -------
    pd.Series
        Resampled values indexed by ``target_times`` (NaN where no support).
    """
    time_numeric = epoch_seconds(time_data)
    target_numeric_all = epoch_seconds(target_times)
    results = np.full(len(target_times), np.nan)
    half_interval = interval_seconds / 2

    for i in range(len(target_times)):
        target_numeric = target_numeric_all[i]

        if is_flux:
            mask = np.abs(time_numeric - target_numeric) <= half_interval
        else:
            window_seconds = sigma_seconds * 4
            mask = np.abs(time_numeric - target_numeric) <= window_seconds

        if not mask.any():
            continue

        window_times = time_numeric[mask]
        window_values = value_data.iloc[mask].values
        valid = ~np.isnan(window_values)
        if not valid.any():
            continue

        window_times = window_times[valid]
        window_values = window_values[valid]

        if is_flux:
            results[i] = np.sum(window_values)
        elif is_circular:
            w = gaussian_weights(window_times, target_numeric, sigma_seconds)
            results[i] = circular_mean(window_values, w)
        else:
            w = gaussian_weights(window_times, target_numeric, sigma_seconds)
            results[i] = np.average(window_values, weights=w)

    return pd.Series(results, index=target_times)


def identify_gaps(series: pd.Series,
                  time_index: pd.DatetimeIndex) -> list:
    """Locate interior NaN runs in a resampled series.

    Parameters
    ----------
    series : pd.Series
        Resampled values (NaN marks missing grid points).
    time_index : pd.DatetimeIndex
        Grid timestamps aligned with ``series``.

    Returns
    -------
    list of tuple
        ``(start_idx, end_idx, gap_hours)`` per interior gap.
    """
    gaps = []
    is_nan = series.isna()
    in_gap = False
    gap_start = None

    for i, (is_missing, _t) in enumerate(zip(is_nan, time_index)):
        if is_missing and not in_gap:
            in_gap = True
            gap_start = i
        elif not is_missing and in_gap:
            gap_end = i
            if gap_start > 0:
                gap_hours = (time_index[gap_end]
                             - time_index[gap_start]).total_seconds() / 3600
                gaps.append((gap_start, gap_end, gap_hours))
            in_gap = False
    return gaps


def interpolate_conservative(series: pd.Series, time_index: pd.DatetimeIndex,
                             max_gap_hours: float,
                             is_flux: bool = False) -> Tuple[pd.Series, int]:
    """Linearly bridge gaps no longer than ``max_gap_hours``.

    Longer gaps stay NaN; leading/trailing NaN runs are never filled.

    Parameters
    ----------
    series : pd.Series
        Resampled values.
    time_index : pd.DatetimeIndex
        Grid timestamps aligned with ``series``.
    max_gap_hours : float
        Longest gap to bridge.
    is_flux : bool
        Unused by the interpolation itself; kept for signature parity with
        the pipeline of record.

    Returns
    -------
    result : pd.Series
        Series with short gaps filled.
    n_filled : int
        Number of gaps bridged.
    """
    result = series.copy()
    gaps = identify_gaps(series, time_index)
    n_filled = 0

    for gap_start, gap_end, gap_hours in gaps:
        if gap_hours <= max_gap_hours:
            start_val = series.iloc[gap_start - 1] if gap_start > 0 else np.nan
            end_val = series.iloc[gap_end] if gap_end < len(series) else np.nan

            if not np.isnan(start_val) and not np.isnan(end_val):
                n_points = gap_end - gap_start
                interp_vals = np.linspace(start_val, end_val,
                                          n_points + 2)[1:-1]
                result.iloc[gap_start:gap_end] = interp_vals
                n_filled += 1

    return result, n_filled


def create_target_time_grid(start_time: pd.Timestamp, end_time: pd.Timestamp,
                            interval_hours: int,
                            center: str = 'standard') -> pd.DatetimeIndex:
    """Build the uniform output grid between two timestamps.

    Parameters
    ----------
    start_time, end_time : pd.Timestamp
        Span of the source data.
    interval_hours : int
        Grid interval in hours.
    center : str
        'standard' anchors on the hour; 'noon'/'midnight' anchor diurnal
        grids as in the pipeline of record.

    Returns
    -------
    pd.DatetimeIndex
        Grid timestamps within [start_time, end_time].
    """
    if center == 'noon':
        anchor = start_time.replace(hour=12, minute=0, second=0,
                                    microsecond=0)
    elif center == 'midnight':
        anchor = start_time.replace(hour=0, minute=0, second=0,
                                    microsecond=0)
    else:
        anchor = start_time.replace(minute=0, second=0, microsecond=0)

    grid = pd.date_range(start=anchor, end=end_time,
                         freq=f'{interval_hours}h')
    grid = grid[(grid >= start_time) & (grid <= end_time)]
    return grid


def _sorted_by_time(df: pd.DataFrame,
                    time_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Sort a dataset by its datetime column.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with a parseable datetime column.
    time_col : str
        Name of the datetime column.

    Returns
    -------
    data : pd.DataFrame
        Rows sorted by time, index reset.
    times : pd.Series
        Parsed datetimes aligned with ``data``.
    """
    times = pd.to_datetime(df[time_col])
    order = times.argsort()
    return (df.iloc[order].reset_index(drop=True),
            times.iloc[order].reset_index(drop=True))


def _resample_columns(data: pd.DataFrame, times: pd.Series,
                      time_col: str, grid: pd.DatetimeIndex,
                      interval_hours: int, circular_cols: Sequence[str],
                      flux_cols: Sequence[str], sigma_fraction: float,
                      max_gap_hours: float) -> Dict[str, np.ndarray]:
    """Resample every numeric column of one dataset onto ``grid``.

    Parameters
    ----------
    data : pd.DataFrame
        Time-sorted input data.
    times : pd.Series
        Parsed datetimes aligned with ``data``.
    time_col : str
        Datetime column name (excluded from resampling).
    grid : pd.DatetimeIndex
        Target uniform grid.
    interval_hours : int
        Grid interval in hours.
    circular_cols, flux_cols : sequence of str
        Columns aggregated by circular mean / window sum.
    sigma_fraction : float
        Gaussian kernel sigma as a fraction of the interval.
    max_gap_hours : float
        Longest gap bridged by linear interpolation.

    Returns
    -------
    dict of str to np.ndarray
        Grid-aligned values per numeric column.
    """
    sigma_seconds = sigma_fraction * interval_hours * 3600.0
    interval_seconds = interval_hours * 3600.0
    out: Dict[str, np.ndarray] = {}
    numeric_cols = [c for c in data.columns if c != time_col
                    and pd.api.types.is_numeric_dtype(data[c])]
    for col in numeric_cols:
        resampled = gaussian_resample_series(
            times, data[col], grid, sigma_seconds, interval_seconds,
            is_circular=col in circular_cols,
            is_flux=col in flux_cols)
        filled, _ = interpolate_conservative(resampled, grid, max_gap_hours,
                                             is_flux=col in flux_cols)
        out[col] = filled.values
    return out


def resample_to_grid(df: pd.DataFrame, time_col: str = 'time',
                     interval_hours: int = 6,
                     circular_cols: Sequence[str] = (),
                     flux_cols: Sequence[str] = (),
                     sigma_fraction: float = 0.25,
                     max_gap_hours: float = 3.0) -> pd.DataFrame:
    """Align every numeric column of a dataset onto one uniform time grid.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with a parseable datetime column.
    time_col : str
        Name of the datetime column.
    interval_hours : int
        Output grid interval in hours.
    circular_cols : sequence of str
        Columns treated as directions in degrees (circular mean).
    flux_cols : sequence of str
        Columns treated as extensive quantities (window sum).
    sigma_fraction : float
        Gaussian kernel sigma as a fraction of the interval (of record: 0.25).
    max_gap_hours : float
        Longest gap bridged by linear interpolation (of record: 3).

    Returns
    -------
    pd.DataFrame
        Grid-aligned dataset with ``time_col`` first.
    """
    data, times = _sorted_by_time(df, time_col)
    grid = create_target_time_grid(times.iloc[0], times.iloc[-1],
                                   interval_hours)
    out = pd.DataFrame({time_col: grid})
    columns = _resample_columns(data, times, time_col, grid, interval_hours,
                                circular_cols, flux_cols, sigma_fraction,
                                max_gap_hours)
    for col, values in columns.items():
        out[col] = values
    return out


def align_datasets(datasets: Sequence[Tuple[pd.DataFrame, str]],
                   interval_hours: int = 6, time_col: str = 'time',
                   circular_cols: Sequence[str] = (),
                   flux_cols: Sequence[str] = (),
                   sigma_fraction: float = 0.25,
                   max_gap_hours: float = 3.0) -> pd.DataFrame:
    """Merge several unaligned datasets onto one uniform time grid.

    Builds a single grid spanning the union of all input time ranges,
    resamples every numeric column of every dataset onto it (each file may
    have its own native resolution and its own datetime column name), and
    joins the columns into one analysis-ready frame.

    Parameters
    ----------
    datasets : sequence of (pd.DataFrame, str)
        Input datasets, each paired with the name of its datetime column.
    interval_hours : int
        Output grid interval in hours.
    time_col : str
        Name of the datetime column in the merged output.
    circular_cols, flux_cols : sequence of str
        Columns (across all inputs) aggregated by circular mean /
        window sum.
    sigma_fraction : float
        Gaussian kernel sigma as a fraction of the interval (of record: 0.25).
    max_gap_hours : float
        Longest gap bridged by linear interpolation (of record: 3).

    Returns
    -------
    pd.DataFrame
        Merged grid-aligned dataset with ``time_col`` first.

    Raises
    ------
    ValueError
        If no datasets are given or two inputs share a data column name.
    """
    if not datasets:
        raise ValueError("no datasets to align")

    parsed = [_sorted_by_time(df, tcol) + (tcol,) for df, tcol in datasets]
    start = min(times.iloc[0] for _, times, _ in parsed)
    end = max(times.iloc[-1] for _, times, _ in parsed)
    grid = create_target_time_grid(start, end, interval_hours)

    out = pd.DataFrame({time_col: grid})
    for i, (data, times, tcol) in enumerate(parsed):
        columns = _resample_columns(data, times, tcol, grid, interval_hours,
                                    circular_cols, flux_cols, sigma_fraction,
                                    max_gap_hours)
        for col, values in columns.items():
            if col in out.columns:
                raise ValueError(
                    f"column '{col}' appears in more than one input file; "
                    "rename it in one of the files before merging")
            out[col] = values
    return out
