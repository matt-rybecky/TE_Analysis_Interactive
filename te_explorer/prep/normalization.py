"""Beta normalization of record: entropy-calibrated robust z-score.

This reimplements the Stage 3 normalization from the original analysis
pipeline (``generate_extended_datasets.py``, not shipped in this repo): each
column is robust z-scored
(``0.6745 * (x - median) / MAD``, IQR/1.349 fallback) and multiplied by one
global entropy-calibration factor ``scale = 2**(JMI - H)``, where JMI is the
joint mutual information of the calibration inputs about the calibration
target and H is the entropy of the z-scored target (KSG estimator, bits).
The calibration anchors the entropy scale to a deterministic physical
relationship among the calibration variables.

One deliberate departure from the pipeline script (author ruling
2026-07-23): the calibration KSG neighbor count defaults to k=3, unified
with the transfer-entropy engine, where the original script used k=10.
Error handling is also stricter here: failed calibration raises instead of
silently falling back to ``scale = 1.0``. The normalization math itself is
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from te_explorer import config as _config  # noqa: F401  (NPEET path injection)
from npeet import entropy_estimators as ee

MAD_SCALE_FACTOR = 0.6745


@dataclass(frozen=True)
class BetaCalibration:
    """Result of the beta-normalization entropy calibration.

    Attributes
    ----------
    target : str
        Calibration target variable.
    inputs : tuple of str
        Calibration input variables.
    jmi_bits : float
        Joint MI of the z-scored inputs about the z-scored target, bits.
    entropy_bits : float
        Entropy of the z-scored target, bits.
    scale : float
        Global entropy-calibration factor ``2**(jmi_bits - entropy_bits)``.
    k : int
        KSG neighbor count used for both estimates.
    """

    target: str
    inputs: tuple
    jmi_bits: float
    entropy_bits: float
    scale: float
    k: int


def calculate_entropy(data: np.ndarray, k: int = 3,
                      base: float = 2.0) -> float:
    """KSG entropy of a 1-D sample, in ``base`` units.

    Parameters
    ----------
    data : np.ndarray
        Sample values; NaN removed internally.
    k : int
        KSG neighbor count.
    base : float
        Logarithm base (2.0 = bits).

    Returns
    -------
    float
        Entropy estimate; 0.0 for a constant sample, NaN if too short.
    """
    data = np.asarray(data).flatten()
    clean = data[~np.isnan(data)]
    if len(clean) < k * 5 or np.std(clean) == 0:
        return 0.0 if np.std(clean) == 0 else np.nan
    return ee.entropy(clean.reshape(-1, 1), k=k, base=base)


def calculate_joint_mi(inputs: Sequence[np.ndarray], target: np.ndarray,
                       k: int = 3, base: float = 2.0) -> float:
    """KSG joint mutual information I(inputs; target), in ``base`` units.

    Parameters
    ----------
    inputs : sequence of np.ndarray
        Input samples, stacked as a joint variable.
    target : np.ndarray
        Target sample.
    k : int
        KSG neighbor count.
    base : float
        Logarithm base (2.0 = bits).

    Returns
    -------
    float
        Non-negative joint MI estimate, or NaN if too few valid rows.
    """
    joint = np.column_stack(inputs)
    target = np.asarray(target).reshape(-1, 1)
    valid = ~np.any(np.isnan(joint), axis=1) & ~np.isnan(target.flatten())
    joint_clean = joint[valid]
    target_clean = target[valid]
    if len(joint_clean) < k * 5:
        return np.nan
    return max(0.0, ee.mi(joint_clean, target_clean, k=k, base=base))


def robust_zscore(series: pd.Series, entropy_scale: float = 1.0) -> pd.Series:
    """Robust z-score of record: ``scale * 0.6745 * (x - median) / MAD``.

    Falls back to IQR/1.349 when the MAD is zero; a zero-variance column
    maps to all zeros.

    Parameters
    ----------
    series : pd.Series
        Input values; NaN preserved.
    entropy_scale : float
        Global beta-calibration factor applied after the z-score.

    Returns
    -------
    pd.Series
        Normalized values.
    """
    valid_values = series.dropna().values
    if len(valid_values) < 2:
        return series

    median_val = np.median(valid_values)
    mad_val = np.median(np.abs(valid_values - median_val))

    if mad_val == 0 or np.isclose(mad_val, 0):
        q75, q25 = np.percentile(valid_values, [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            mad_val = iqr / 1.349
        else:
            return pd.Series(np.zeros(len(series)), index=series.index)

    return entropy_scale * MAD_SCALE_FACTOR * (series - median_val) / mad_val


def _zscore_array(values: np.ndarray) -> np.ndarray:
    """Robust z-score used inside the calibration (of record)."""
    median_val = np.median(values[~np.isnan(values)])
    mad_val = np.median(np.abs(values[~np.isnan(values)] - median_val))
    if mad_val == 0 or np.isclose(mad_val, 0):
        q75, q25 = np.percentile(values[~np.isnan(values)], [75, 25])
        mad_val = (q75 - q25) / 1.349
        if mad_val == 0:
            raise ValueError("zero-variance calibration variable")
    return MAD_SCALE_FACTOR * (values - median_val) / mad_val


def compute_beta_calibration(df: pd.DataFrame, target: str,
                             inputs: Sequence[str], k: int = 3,
                             base: float = 2.0) -> BetaCalibration:
    """Compute the global entropy-calibration factor (of record).

    Parameters
    ----------
    df : pd.DataFrame
        Aligned dataset containing the calibration variables.
    target : str
        Calibration target column (of record: ``met_rh``).
    inputs : sequence of str
        Calibration input columns (of record: ``met_temp``,
        ``met_vapor_pressure``).
    k : int
        KSG neighbor count (author ruling 2026-07-23: 3).
    base : float
        Logarithm base (2.0 = bits).

    Returns
    -------
    BetaCalibration
        Calibration record including ``scale = base**(JMI - H)``.

    Raises
    ------
    ValueError
        If calibration variables are missing, constant, or the estimates
        are undefined.
    """
    required = list(inputs) + [target]
    missing = [v for v in required if v not in df.columns]
    if missing:
        raise ValueError(f"missing calibration variables: {missing}")

    norm_data = {var: _zscore_array(df[var].values.astype(float))
                 for var in required}

    input_arrays = [norm_data[v] for v in inputs]
    target_array = norm_data[target]

    jmi = calculate_joint_mi(input_arrays, target_array, k=k, base=base)
    h_current = calculate_entropy(target_array, k=k, base=base)

    if np.isnan(jmi) or jmi <= 0 or np.isnan(h_current):
        raise ValueError(
            f"beta calibration failed: JMI={jmi}, H={h_current} "
            "(too few valid rows or degenerate variables)")

    scale = base ** (jmi - h_current)
    return BetaCalibration(target=target, inputs=tuple(inputs),
                           jmi_bits=jmi, entropy_bits=h_current,
                           scale=scale, k=k)


def beta_normalize(df: pd.DataFrame, scale: float,
                   time_col: str = 'time') -> pd.DataFrame:
    """Apply the beta normalization to every numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Aligned dataset.
    scale : float
        Global calibration factor from :func:`compute_beta_calibration`.
    time_col : str
        Datetime column, passed through unchanged.

    Returns
    -------
    pd.DataFrame
        Normalized dataset with the same columns.
    """
    out = df.copy()
    for col in df.columns:
        if col == time_col or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        out[col] = robust_zscore(df[col], entropy_scale=scale)
    return out
