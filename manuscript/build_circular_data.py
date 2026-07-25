#!/usr/bin/env python3
"""
build_circular_data.py — Circular (sin/cos) wind-direction beta dataset.

Repairs the wind-direction representation for the publication transfer-entropy
run. The shipped ``final_6hr_beta.csv`` stores ``met_wdir`` as a *linear* robust
z-score of degrees, which preserves the 0 deg / 360 deg discontinuity: northerly
flow (degrees near both ends of the range) is thrown to opposite extremes of the
column, so the KSG estimator mismeasures exactly the episodes where wind
direction carries information.

This script rebuilds the base data file with wind direction decomposed into its
two circular components, ``met_wdir_sin`` and ``met_wdir_cos``, derived from the
6-hour *circular-mean* direction in degrees (``data/final_6hr.csv``). The
components are normalized into the existing beta pipeline with a single shared
scale so the unit circle stays a circle (no direction is privileged):

    s = sin(theta) - median(sin),   c = cos(theta) - median(cos)
    shared_mad = median(|[s, c]|)                  # one spread for both axes
    met_wdir_sin = alpha * 0.6745 * s / shared_mad
    met_wdir_cos = alpha * 0.6745 * c / shared_mad

alpha is the global entropy-scaling factor already baked into every column of
``final_6hr_beta.csv`` (recovered here from the raw/beta pair, not hardcoded).
0.6745 is the MAD-to-sigma factor used by the upstream ``robust_zscore_normalize``.

All other columns are copied byte-for-byte from ``final_6hr_beta.csv`` so the
non-wind results stay directly comparable; only ``met_wdir`` is dropped and the
two component columns are added.

Output: ``data/final_6hr_beta_circular.csv``.

Run (no model compute; ~1 s):
    .venv/bin/python3 build_circular_data.py

Author: Matthew Rybecky
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAD_SCALE_FACTOR = 0.6745  # MAD -> standard-deviation factor (upstream pipeline)

RAW_FILE = Path('data/final_6hr.csv')          # circular-mean wdir in degrees
BETA_FILE = Path('data/final_6hr_beta.csv')    # existing beta dataset
OUT_FILE = Path('data/final_6hr_beta_circular.csv')
WDIR_COL = 'met_wdir'
SIN_COL = 'met_wdir_sin'
COS_COL = 'met_wdir_cos'


def recover_alpha(raw: pd.DataFrame, beta: pd.DataFrame,
                  probe_cols=('met_temp', 'met_rh', 'rad_net')) -> float:
    """
    Recover the global entropy-scaling factor alpha from the raw/beta pair.

    For every column the upstream transform is
    ``beta = alpha * 0.6745 * (raw - median(raw)) / MAD(raw)``. Solving for alpha
    on any non-trivial column recovers the same global constant.

    Parameters
    ----------
    raw, beta : pd.DataFrame
        Un-normalized and beta-normalized datasets (row-aligned).
    probe_cols : tuple of str
        Columns used to recover and cross-check alpha.

    Returns
    -------
    alpha : float
        Global entropy-scaling factor.
    """
    estimates = []
    for col in probe_cols:
        x = raw[col].to_numpy(dtype=float)
        b = beta[col].to_numpy(dtype=float)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        z = MAD_SCALE_FACTOR * (x - med) / mad
        mask = np.isfinite(z) & np.isfinite(b) & (np.abs(z) > 1e-6)
        estimates.append(np.nanmedian(b[mask] / z[mask]))
    estimates = np.array(estimates)
    if estimates.std() > 1e-6:
        raise ValueError(f"alpha inconsistent across probe columns: {estimates}")
    return float(np.nanmedian(estimates))


def circular_components(degrees: np.ndarray, alpha: float):
    """
    Beta-normalized sin/cos components of a circular variable (shared scale).

    Parameters
    ----------
    degrees : np.ndarray
        Direction in degrees [0, 360); NaN allowed.
    alpha : float
        Global entropy-scaling factor.

    Returns
    -------
    sin_beta, cos_beta : np.ndarray
        Circle-preserving beta-normalized components (NaN where input is NaN).
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
    scale = alpha * MAD_SCALE_FACTOR / shared_mad

    return s * scale, c * scale


def verify(raw: pd.DataFrame, out: pd.DataFrame) -> None:
    """Print circularity / continuity checks for the new components."""
    deg = raw[WDIR_COL].to_numpy(dtype=float)
    valid = np.isfinite(deg)
    sin_raw, cos_raw = np.sin(np.deg2rad(deg)), np.cos(np.deg2rad(deg))
    unit = sin_raw[valid] ** 2 + cos_raw[valid] ** 2
    logger.info("Verification:")
    logger.info(f"  raw degrees range: [{np.nanmin(deg):.2f}, {np.nanmax(deg):.2f}] "
                f"(n_valid={valid.sum()})")
    logger.info(f"  sin^2+cos^2 (raw): mean={unit.mean():.6f} (expect 1.0)")
    logger.info(f"  {SIN_COL}: range [{out[SIN_COL].min():.3f}, {out[SIN_COL].max():.3f}]")
    logger.info(f"  {COS_COL}: range [{out[COS_COL].min():.3f}, {out[COS_COL].max():.3f}]")
    # Northerly continuity: 334.84 deg and 31.70 deg are ~57 deg apart through
    # north; in the components their distance should be small, not extreme.
    near_n = valid & ((deg > 300) | (deg < 60))
    if near_n.any():
        comp = np.column_stack([out[SIN_COL].to_numpy(), out[COS_COL].to_numpy()])
        spread = np.nanmax(comp[near_n], axis=0) - np.nanmin(comp[near_n], axis=0)
        logger.info(f"  northerly (>300 or <60 deg) component spread: "
                    f"sin={spread[0]:.2f}, cos={spread[1]:.2f} "
                    f"(small cos spread => north no longer split)")
    assert abs(unit.mean() - 1.0) < 1e-9, "sin^2+cos^2 != 1"
    assert WDIR_COL not in out.columns, "old linear met_wdir should be dropped"
    assert out[SIN_COL].notna().sum() == valid.sum(), "component NaN mismatch"


def main() -> None:
    for f in (RAW_FILE, BETA_FILE):
        if not f.exists():
            raise FileNotFoundError(f"Required input missing: {f}")

    raw = pd.read_csv(RAW_FILE)
    beta = pd.read_csv(BETA_FILE)
    if not (raw['time'].values == beta['time'].values).all():
        raise ValueError("final_6hr.csv and final_6hr_beta.csv are not row-aligned")

    alpha = recover_alpha(raw, beta)
    logger.info(f"Recovered global alpha = {alpha:.6f}")

    sin_beta, cos_beta = circular_components(
        raw[WDIR_COL].to_numpy(dtype=float), alpha)

    out = beta.drop(columns=[WDIR_COL]).copy()
    out[SIN_COL] = sin_beta
    out[COS_COL] = cos_beta

    verify(raw, out)

    out.to_csv(OUT_FILE, index=False)
    logger.info(f"Wrote {OUT_FILE} ({len(out)} rows; dropped {WDIR_COL}, "
                f"added {SIN_COL}, {COS_COL})")


if __name__ == '__main__':
    main()
