"""Circular encoding of record: shared-scale sin/cos components.

Directional variables (wind direction in degrees) carry a 0/360 wrap that a
linear normalization destroys: northerly flow lands at both extremes of the
column. The encoding of record (``build_circular_data.py``) decomposes the
direction into sin/cos components and normalizes both with one shared
robust scale so the unit circle stays a circle (no direction privileged):

    s = sin(theta) - median(sin),   c = cos(theta) - median(cos)
    shared_mad = median(|[s, c]|)
    col_sin = scale * 0.6745 * s / shared_mad
    col_cos = scale * 0.6745 * c / shared_mad

where ``scale`` is the global beta-calibration factor already applied to
every other column. The math is unchanged from the script of record.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

MAD_SCALE_FACTOR = 0.6745


def circular_components(degrees: np.ndarray,
                        scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Beta-normalized sin/cos components of a circular variable.

    Parameters
    ----------
    degrees : np.ndarray
        Direction in degrees [0, 360); NaN allowed.
    scale : float
        Global entropy-calibration factor (the beta scale).

    Returns
    -------
    sin_beta, cos_beta : np.ndarray
        Circle-preserving normalized components (NaN where input is NaN).
    """
    theta = np.deg2rad(degrees)
    sin_raw = np.sin(theta)
    cos_raw = np.cos(theta)

    valid = np.isfinite(sin_raw) & np.isfinite(cos_raw)
    s = sin_raw - np.median(sin_raw[valid])
    c = cos_raw - np.median(cos_raw[valid])

    # One shared spread for both axes preserves circular geometry.
    pooled = np.concatenate([s[valid], c[valid]])
    shared_mad = np.median(np.abs(pooled))
    factor = scale * MAD_SCALE_FACTOR / shared_mad

    return s * factor, c * factor


def encode_circular(df: pd.DataFrame, degrees: pd.DataFrame,
                    cols: Sequence[str], scale: float) -> pd.DataFrame:
    """Replace directional columns with their sin/cos component pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized dataset in which the directional columns are replaced.
    degrees : pd.DataFrame
        Dataset holding the same columns in raw degrees (the aligned,
        un-normalized data), row-aligned with ``df``.
    cols : sequence of str
        Directional columns to encode.
    scale : float
        Global beta-calibration factor.

    Returns
    -------
    pd.DataFrame
        Dataset with each ``col`` dropped and ``col_sin``/``col_cos`` added.

    Raises
    ------
    ValueError
        If a requested column is missing from either frame.
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns or col not in degrees.columns:
            raise ValueError(f"circular column not found: {col}")
        sin_beta, cos_beta = circular_components(
            degrees[col].to_numpy(dtype=float), scale)
        out = out.drop(columns=[col])
        out[f'{col}_sin'] = sin_beta
        out[f'{col}_cos'] = cos_beta
    return out
