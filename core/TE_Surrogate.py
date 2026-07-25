#!/usr/bin/env python3
"""
Transfer Entropy Surrogate Testing V1.0.0

Surrogate-based significance testing for transfer entropy calculations.

Mathematical Framework:
- Generate N surrogate time series that preserve certain statistical properties
  of the original source variable but destroy the temporal coupling with the target
- Calculate TE for each surrogate -> target
- Compare original TE to surrogate TE distribution
- Significance: p-value = fraction of surrogates with TE >= original TE
- If original TE > 95th percentile of surrogate distribution -> significant at alpha=0.05

Surrogate Types:
- IAAFT: Preserves amplitude distribution and power spectrum
- Random Shuffle: Completely destroys temporal structure
- Block Shuffle: Partially preserves short-range autocorrelation
- Phase Randomization: Preserves power spectrum, randomizes phases

References:
- Schreiber & Schmitz (2000). Surrogate time series. Physica D, 142(3-4), 346-382.
- Theiler et al. (1992). Testing for nonlinearity in time series. Physica D, 58(1-4), 77-94.

Author: Matthew Rybecky
Version: V1.0.0
"""

import pandas as pd
import numpy as np
import sys
import time
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple, Optional, Callable
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add NPEET to Python path
npeet_path = Path(__file__).parent / 'NPEET'
if npeet_path.exists():
    sys.path.insert(0, str(npeet_path))

from npeet import entropy_estimators as ee

# Shared vector-source helpers (1D scalar / 2D vector handling).
from TE_Calculator import _row_valid_mask, _as_2d


# ==========================================================================
# Surrogate Generation Functions (standalone for multiprocessing pickling)
# ==========================================================================

def generate_iaaft_surrogate(data: np.ndarray, max_iterations: int = 100,
                              tolerance: float = 1e-6) -> np.ndarray:
    """
    Generate IAAFT (Iterative Amplitude Adjusted Fourier Transform) surrogate.

    Preserves both the amplitude distribution and power spectrum of the original
    time series while destroying the phase relationships (temporal coupling).

    Parameters
    ----------
    data : np.ndarray
        Original time series
    max_iterations : int
        Maximum number of IAAFT iterations
    tolerance : float
        Convergence tolerance for iteration

    Returns
    -------
    surrogate : np.ndarray
        IAAFT surrogate time series

    References
    ----------
    Schreiber & Schmitz (2000). Surrogate time series. Physica D, 142(3-4), 346-382.
    """
    n = len(data)
    surrogate = data.copy()

    # Edge cases
    if n <= 1:
        return surrogate

    nan_mask = np.isnan(data)
    has_nan = np.any(nan_mask)

    # Fill NaN with linear interpolation for FFT processing
    if has_nan:
        non_nan_values = data[~nan_mask]
        if len(non_nan_values) <= 1:
            return surrogate
        indices = np.arange(n)
        data_filled = np.interp(indices, indices[~nan_mask], non_nan_values)
    else:
        data_filled = data.copy()

    # Check for constant series
    if np.std(data_filled) == 0:
        return surrogate

    # Pre-compute targets
    sorted_values = np.sort(data_filled)       # Target amplitude distribution
    data_mean = np.mean(data_filled)
    original_fft = np.fft.rfft(data_filled - data_mean)
    original_magnitudes = np.abs(original_fft)  # Target power spectrum

    # Initialize with random shuffle of data
    surrogate_iter = data_filled.copy()
    np.random.shuffle(surrogate_iter)

    converged = False
    for iteration in range(max_iterations):
        prev_surrogate = surrogate_iter.copy()

        # --- SPECTRAL ADJUSTMENT ---
        # Replace magnitudes with original spectrum, keep current phases
        surr_fft = np.fft.rfft(surrogate_iter - data_mean)
        surr_phases = np.angle(surr_fft)

        # Fix DC component (must be real)
        surr_phases[0] = 0.0
        # Fix Nyquist component if n is even (must be real)
        if n % 2 == 0:
            surr_phases[-1] = 0.0 if surr_fft[-1].real >= 0 else np.pi

        new_fft = original_magnitudes * np.exp(1j * surr_phases)
        spectral_adjusted = np.fft.irfft(new_fft, n=n) + data_mean

        # --- AMPLITUDE ADJUSTMENT ---
        # Rank-order map to match original amplitude distribution
        rank_order = np.argsort(np.argsort(spectral_adjusted))
        surrogate_iter = sorted_values[rank_order]

        # --- CONVERGENCE CHECK ---
        diff = np.mean((surrogate_iter - prev_surrogate) ** 2)
        norm = np.mean(surrogate_iter ** 2)
        if norm > 0:
            relative_diff = diff / norm
        else:
            relative_diff = 0.0

        if relative_diff < tolerance:
            converged = True
            break

    if not converged:
        print(f"  IAAFT warning: did not converge after {max_iterations} "
              f"iterations (rel_diff={relative_diff:.2e})")

    # Restore NaN at original positions
    if has_nan:
        surrogate_iter[nan_mask] = np.nan

    return surrogate_iter


def generate_random_shuffle_surrogate(data: np.ndarray) -> np.ndarray:
    """
    Generate random shuffle (permutation) surrogate.

    Completely destroys temporal structure while preserving the marginal
    distribution of the source variable.

    Parameters
    ----------
    data : np.ndarray
        Original time series

    Returns
    -------
    surrogate : np.ndarray
        Randomly shuffled surrogate time series
    """
    n = len(data)
    surrogate = data.copy()

    # Edge cases: nothing to shuffle
    if n <= 1:
        return surrogate

    nan_mask = np.isnan(data)
    non_nan_values = data[~nan_mask]

    # All NaN or constant series: return copy
    if len(non_nan_values) <= 1:
        return surrogate

    # Shuffle only the non-NaN values, preserve NaN positions
    np.random.shuffle(non_nan_values)
    surrogate[~nan_mask] = non_nan_values

    return surrogate


def _estimate_block_size(data: np.ndarray) -> int:
    """
    Estimate optimal block size from autocorrelation e-folding time.

    Computes the autocorrelation function via FFT and finds the first lag
    where ACF drops below 1/e (~0.368), representing the decorrelation
    timescale. Result is clamped to [2, n//3].

    Parameters
    ----------
    data : np.ndarray
        Time series (NaN-free)

    Returns
    -------
    block_size : int
        Estimated block size in data points
    """
    n = len(data)
    centered = data - np.mean(data)

    # Compute linear (non-circular) ACF via zero-padded FFT
    padded = np.zeros(2 * n)
    padded[:n] = centered
    fft_padded = np.fft.rfft(padded)
    acf_full = np.fft.irfft(fft_padded * np.conj(fft_padded))
    acf = acf_full[:n]

    # Normalize so lag-0 = 1.0
    if acf[0] > 0:
        acf = acf / acf[0]
    else:
        return 2

    # Find first lag where ACF < 1/e
    e_folding = 1.0 / np.e
    below = np.where(acf < e_folding)[0]

    if len(below) > 0 and below[0] > 0:
        block_size = int(below[0])
    else:
        # Highly persistent series: fallback
        block_size = n // 10

    # Clamp to [2, n//3] to ensure at least 3 blocks
    block_size = max(2, min(block_size, n // 3))

    return block_size


def generate_block_shuffle_surrogate(data: np.ndarray,
                                      block_size: int = None) -> np.ndarray:
    """
    Generate block shuffle surrogate.

    Shuffles blocks of consecutive data points, partially preserving
    short-range autocorrelation structure while destroying long-range
    temporal coupling.

    Parameters
    ----------
    data : np.ndarray
        Original time series
    block_size : int, optional
        Size of blocks to shuffle. If None, automatically determined
        from autocorrelation structure.

    Returns
    -------
    surrogate : np.ndarray
        Block-shuffled surrogate time series
    """
    n = len(data)
    surrogate = data.copy()

    # Edge cases
    if n <= 1:
        return surrogate

    # Auto-estimate block size from non-NaN values if not provided
    if block_size is None:
        non_nan_values = data[~np.isnan(data)]
        if len(non_nan_values) <= 1:
            return surrogate
        block_size = _estimate_block_size(non_nan_values)

    # Clamp block_size to valid range
    block_size = max(1, min(block_size, n))

    # Divide data indices into blocks
    n_full_blocks = n // block_size
    remainder = n % block_size

    blocks = []
    for i in range(n_full_blocks):
        start = i * block_size
        blocks.append(data[start:start + block_size])

    # Add remainder block if any
    if remainder > 0:
        blocks.append(data[n_full_blocks * block_size:])

    # Randomly permute block order
    perm = np.random.permutation(len(blocks))
    shuffled_blocks = [blocks[i] for i in perm]

    # Concatenate and return
    surrogate = np.concatenate(shuffled_blocks)

    return surrogate


def generate_phase_randomization_surrogate(data: np.ndarray) -> np.ndarray:
    """
    Generate phase randomization surrogate.

    Preserves the power spectrum of the original time series while
    randomizing the Fourier phases, destroying temporal coupling.

    Parameters
    ----------
    data : np.ndarray
        Original time series

    Returns
    -------
    surrogate : np.ndarray
        Phase-randomized surrogate time series
    """
    n = len(data)
    surrogate = data.copy()

    # Edge cases
    if n <= 1:
        return surrogate

    nan_mask = np.isnan(data)
    has_nan = np.any(nan_mask)

    # Fill NaN with linear interpolation for FFT processing
    if has_nan:
        non_nan_values = data[~nan_mask]
        if len(non_nan_values) <= 1:
            return surrogate
        indices = np.arange(n)
        data_filled = np.interp(indices, indices[~nan_mask], non_nan_values)
    else:
        data_filled = data.copy()

    # Check for constant series
    if np.std(data_filled) == 0:
        return surrogate

    # Center data, compute real FFT
    data_mean = np.mean(data_filled)
    centered = data_filled - data_mean
    fft_coeff = np.fft.rfft(centered)

    # Extract magnitudes (power spectrum)
    magnitudes = np.abs(fft_coeff)

    # Generate random phases for each frequency bin
    n_freq = len(fft_coeff)
    random_phases = np.random.uniform(0, 2 * np.pi, n_freq)

    # DC component (index 0) must be real -> phase = 0
    random_phases[0] = 0.0

    # Nyquist component (last index if n is even) must be real -> phase 0 or pi
    if n % 2 == 0:
        random_phases[-1] = np.random.choice([0.0, np.pi])

    # Reconstruct FFT coefficients with original magnitudes and random phases
    new_fft = magnitudes * np.exp(1j * random_phases)

    # Inverse FFT back to time domain
    surrogate_centered = np.fft.irfft(new_fft, n=n)

    # Restore mean
    surrogate = surrogate_centered + data_mean

    # Restore NaN at original positions
    if has_nan:
        surrogate[nan_mask] = np.nan

    return surrogate


def _fit_circle(xy: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Algebraic (Kasa) least-squares circle fit.

    Parameters
    ----------
    xy : np.ndarray, shape (n, 2)
        Points assumed to lie near a circle.

    Returns
    -------
    center : np.ndarray, shape (2,)
    radius : float
    """
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    radius = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0)))
    return np.array([cx, cy]), radius


def _project_to_circle(points: np.ndarray, center: np.ndarray,
                       radius: float) -> np.ndarray:
    """Move each point radially onto the circle (center, radius)."""
    v = points - center
    norm = np.hypot(v[:, 0], v[:, 1])
    norm[norm == 0] = 1.0
    return center + radius * v / norm[:, None]


def generate_iaaft_surrogate_mv(data: np.ndarray, max_iterations: int = 100,
                                 tolerance: float = 1e-6) -> np.ndarray:
    """
    Multivariate IAAFT surrogate preserving cross-spectrum (common-phase).

    For a multi-component source (e.g. circular wind direction as sin/cos) the
    null must destroy coupling with the target while preserving (i) each
    component's amplitude distribution and power spectrum and (ii) the
    cross-spectrum between components — i.e. the unit-circle geometry. This is
    achieved by letting a single common phase per frequency evolve (taken from
    component 0) while holding the original inter-component relative phases
    fixed; amplitudes are rank-mapped to each component's marginal each iteration.

    Parameters
    ----------
    data : np.ndarray, shape (n, k)
        Multivariate source window.
    max_iterations : int
        Maximum IAAFT iterations.
    tolerance : float
        Relative-change convergence tolerance.

    Returns
    -------
    surrogate : np.ndarray, shape (n, k)
        Cross-spectrum-preserving surrogate (NaN restored at original positions).

    References
    ----------
    Schreiber & Schmitz (2000). Surrogate time series. Physica D, 142, 346-382
    (multivariate extension).
    """
    n, k = data.shape
    if n <= 1:
        return data.copy()

    nan_mask = np.isnan(data)
    filled = data.copy()
    for c in range(k):
        col = data[:, c]
        m = nan_mask[:, c]
        if m.any():
            good = ~m
            if good.sum() <= 1:
                return data.copy()
            idx = np.arange(n)
            filled[:, c] = np.interp(idx, idx[good], col[good])

    if np.any(np.std(filled, axis=0) == 0):
        return data.copy()

    means = filled.mean(axis=0)
    fft0 = np.fft.rfft(filled - means, axis=0)
    mags = np.abs(fft0)                          # per-component target spectra
    rel_phase = np.angle(fft0) - np.angle(fft0)[:, [0]]  # preserved cross-phase
    sorted_vals = np.sort(filled, axis=0)        # per-component marginals
    even = (n % 2 == 0)

    surr = filled[np.random.permutation(n), :].copy()
    for _ in range(max_iterations):
        prev = surr.copy()
        psi0 = np.angle(np.fft.rfft(surr - surr.mean(axis=0), axis=0))[:, 0]
        new_phase = psi0[:, None] + rel_phase
        new_phase[0, :] = 0.0                    # DC real
        if even:
            new_phase[-1, :] = 0.0               # Nyquist real
        spectral = np.fft.irfft(mags * np.exp(1j * new_phase), n=n, axis=0) + means
        for c in range(k):
            rank_order = np.argsort(np.argsort(spectral[:, c]))
            surr[:, c] = sorted_vals[:, c][rank_order]
        diff = np.mean((surr - prev) ** 2)
        norm = np.mean(surr ** 2)
        if norm > 0 and diff / norm < tolerance:
            break

    # For a genuinely circular source (k=2, points on a circle, e.g. wind
    # direction sin/cos) re-project the surrogate radially onto the data circle
    # so every surrogate point is a valid direction. Per-component amplitude
    # adjustment alone preserves marginals/spectra but lets points drift off the
    # circle; re-projection restores the hard nonlinear constraint.
    if k == 2:
        center, radius = _fit_circle(filled)
        resid = np.abs(np.hypot(filled[:, 0] - center[0],
                                filled[:, 1] - center[1]) - radius)
        if radius > 0 and np.median(resid) < 0.1 * radius:
            surr = _project_to_circle(surr, center, radius)

    surr[nan_mask] = np.nan
    return surr


def _generate_surrogate_mv(data: np.ndarray, surrogate_type: str) -> np.ndarray:
    """
    Multivariate surrogate dispatch for a 2D source (preserves coupling).

    IAAFT uses the common-phase cross-spectrum-preserving algorithm. Shuffle
    methods apply one shared permutation across all components so the joint
    geometry is preserved; phase randomization applies one common phase series.
    """
    n, k = data.shape
    if surrogate_type == 'iaaft':
        return generate_iaaft_surrogate_mv(data)
    if surrogate_type in ('random_shuffle', 'block_shuffle'):
        if surrogate_type == 'random_shuffle':
            perm = np.random.permutation(n)
        else:  # shared block permutation from component 0's autocorrelation
            bs = _estimate_block_size(data[~np.isnan(data).any(axis=1), 0])
            order = np.random.permutation(int(np.ceil(n / bs)))
            perm = np.concatenate([np.arange(b * bs, min((b + 1) * bs, n))
                                   for b in order])[:n]
        return data[perm, :]
    if surrogate_type == 'phase_randomization':
        means = np.nanmean(data, axis=0)
        filled = np.where(np.isnan(data), means, data)
        fft = np.fft.rfft(filled - means, axis=0)
        nf = fft.shape[0]
        phases = np.random.uniform(0, 2 * np.pi, nf)
        phases[0] = 0.0
        if n % 2 == 0:
            phases[-1] = np.random.choice([0.0, np.pi])
        rel = np.angle(fft) - np.angle(fft)[:, [0]]
        out = np.fft.irfft(np.abs(fft) * np.exp(1j * (phases[:, None] + rel)),
                           n=n, axis=0) + means
        out[np.isnan(data)] = np.nan
        return out
    raise ValueError(f"Unknown surrogate type: '{surrogate_type}'")


def generate_surrogate(data: np.ndarray, surrogate_type: str = 'iaaft',
                        **kwargs) -> np.ndarray:
    """
    Generate a single surrogate time series using the specified method.

    Accepts a 1D scalar source or a 2D ``(n, k)`` vector source. Vector sources
    (e.g. circular wind direction as sin/cos) are routed to the multivariate,
    cross-spectrum-preserving surrogate so the components stay coupled.

    Parameters
    ----------
    data : np.ndarray
        Original time series, shape ``(n,)`` or ``(n, k)``.
    surrogate_type : str
        Type of surrogate: 'iaaft', 'random_shuffle', 'block_shuffle',
        'phase_randomization'
    **kwargs
        Additional arguments passed to scalar surrogate generators.

    Returns
    -------
    surrogate : np.ndarray
        Surrogate time series, same shape as ``data``.
    """
    arr = np.asarray(data)
    if arr.ndim == 2 and arr.shape[1] > 1:
        return _generate_surrogate_mv(arr, surrogate_type)

    generators = {
        'iaaft': generate_iaaft_surrogate,
        'random_shuffle': generate_random_shuffle_surrogate,
        'block_shuffle': generate_block_shuffle_surrogate,
        'phase_randomization': generate_phase_randomization_surrogate
    }

    if surrogate_type not in generators:
        raise ValueError(f"Unknown surrogate type: '{surrogate_type}'. "
                         f"Must be one of {list(generators.keys())}")

    return generators[surrogate_type](arr.ravel() if arr.ndim == 2 else arr,
                                      **kwargs)


# ==========================================================================
# Standalone TE Calculation Functions (for multiprocessing pickling)
# ==========================================================================

def calculate_te_single_window_standalone(source_data: np.ndarray,
                                           target_data: np.ndarray,
                                           tau: int = 1,
                                           history_length: int = 1) -> float:
    """
    Calculate transfer entropy for a single window (standalone for multiprocessing).

    TE(X->Y) = I(Y_t; X_{t-tau} | Y_{t-1}, ..., Y_{t-h})

    Parameters
    ----------
    source_data : np.ndarray
        Source variable data for window
    target_data : np.ndarray
        Target variable data for window
    tau : int
        Time lag parameter
    history_length : int
        Number of past target values to condition on (h >= 1)

    Returns
    -------
    te : float
        Transfer entropy value in bits
    """
    try:
        h = max(0, history_length)

        # Source may be 1D (scalar) or 2D (vector, e.g. circular wind direction).
        source_data = np.asarray(source_data)
        source_mask = _row_valid_mask(source_data)
        target_mask = ~np.isnan(target_data)
        mask = source_mask & target_mask

        if np.sum(mask) < 10:
            return 0.0

        source_clean = source_data[mask]
        target_clean = target_data[mask]

        if len(source_clean) < 10 or len(target_clean) < 10:
            return 0.0

        offset = max(tau, h) if tau > 0 else max(h, 1)
        min_length = max(2, offset + 2)
        if len(target_clean) < min_length or len(source_clean) < min_length:
            return 0.0

        n_out = len(target_clean) - offset

        target_present = target_clean[offset:]

        if tau == 0:
            source_lagged = source_clean[offset:]
        else:
            source_lagged = source_clean[offset - tau:offset - tau + n_out]

        if n_out < 5:
            return 0.0

        target_present = target_present.reshape(-1, 1)
        source_variable = _as_2d(source_lagged)

        if h == 0:
            te_bits = ee.mi(source_variable, target_present, k=3, base=2)
        else:
            # Target history: stack h past values as columns
            target_past = np.column_stack([
                target_clean[offset - j:offset - j + n_out] for j in range(1, h + 1)
            ])  # shape (n_out, h)
            te_bits = ee.mi(source_variable, target_present, target_past, k=3, base=2)

        if np.isnan(te_bits) or np.isinf(te_bits) or te_bits < 0:
            return 0.0

        return te_bits

    except Exception:
        return 0.0


def calculate_jte_single_window_standalone(source_data_list: List[np.ndarray],
                                            target_data: np.ndarray,
                                            tau: int = 1,
                                            tau_list: List[int] = None,
                                            history_length: int = 1) -> float:
    """
    Calculate Joint Transfer Entropy for a single window (standalone for multiprocessing).

    JTE_{(X1,...,Xn)->Y}(tau) = I(Y_t ; X1_{t-tau1}, ..., Xn_{t-taun} | Y_{t-1}, ..., Y_{t-h})

    Parameters
    ----------
    source_data_list : List[np.ndarray]
        List of source variable arrays for the window
    target_data : np.ndarray
        Target variable array for the window
    tau : int
        Default time lag parameter (used when tau_list is None)
    tau_list : List[int], optional
        Per-source tau values for differential lagging.
        If None, uses scalar tau for all sources.
    history_length : int
        Number of past target values to condition on (h >= 1)

    Returns
    -------
    jte : float
        Joint transfer entropy in bits
    """
    try:
        h = max(0, history_length)

        # Resolve per-source tau values
        n_sources = len(source_data_list)
        if tau_list is not None:
            taus = tau_list
        else:
            taus = [tau] * n_sources

        # Sources may be 1D (scalar) or 2D (vector); require all components valid.
        target_mask = ~np.isnan(target_data)
        combined_mask = target_mask.copy()
        for source_data in source_data_list:
            combined_mask = combined_mask & _row_valid_mask(source_data)

        if np.sum(combined_mask) < 10:
            return 0.0

        target_clean = target_data[combined_mask]
        sources_clean = [src[combined_mask] for src in source_data_list]

        if len(target_clean) < 10:
            return 0.0

        max_tau = max(taus)
        offset = max(max_tau, h) if max_tau > 0 else max(h, 1)
        min_length = max(2, offset + 2)
        if len(target_clean) < min_length:
            return 0.0

        # Create arrays for JTE with per-source differential lags and history h
        n_out = len(target_clean) - offset

        target_present = target_clean[offset:]  # Y_t

        # Each source lagged by its own tau
        sources_past = []
        for src, t in zip(sources_clean, taus):
            if t == 0:
                sources_past.append(src[offset:offset + n_out])
            else:
                sources_past.append(src[offset - t:offset - t + n_out])

        if n_out < 5:
            return 0.0

        sources_stacked = np.column_stack(sources_past)
        target_present_2d = target_present.reshape(-1, 1)

        if h == 0:
            jte_bits = ee.mi(sources_stacked, target_present_2d, k=3, base=2)
        else:
            # Target history: stack h past values as columns
            target_past = np.column_stack([
                target_clean[offset - j:offset - j + n_out] for j in range(1, h + 1)
            ])  # shape (n_out, h)
            jte_bits = ee.cmi(sources_stacked, target_present_2d, target_past, k=3, base=2)

        if np.isnan(jte_bits) or np.isinf(jte_bits) or jte_bits < 0:
            return 0.0

        return jte_bits

    except Exception:
        return 0.0


# ==========================================================================
# Per-Window Surrogate Testing Functions (standalone for multiprocessing)
# ==========================================================================

def surrogate_test_single_window(args: Tuple) -> Dict:
    """
    Perform surrogate significance testing for TE in a single time window.

    For each window:
    1. Calculate original TE
    2. Generate N surrogates of the source variable only
    3. Calculate TE for each surrogate -> target
    4. Determine significance by comparing original to surrogate distribution

    Parameters
    ----------
    args : Tuple
        (window_data, source_var, target_var, tau, n_surrogates, window_idx,
         time_point, surrogate_type)

    Returns
    -------
    result : Dict
        Window results including original TE, surrogate distribution statistics,
        p-value, and significance threshold
    """
    # Unpack args tuple — history_length is optional 9th element
    if len(args) == 9:
        (window_data, source_var, target_var, tau, n_surrogates, window_idx,
         time_point, surrogate_type, history_length) = args
    else:
        (window_data, source_var, target_var, tau, n_surrogates, window_idx,
         time_point, surrogate_type) = args
        history_length = 1

    try:
        source_window = window_data[source_var].values
        target_window = window_data[target_var].values

        # Calculate original TE for this window
        original_te = calculate_te_single_window_standalone(
            source_window, target_window, tau, history_length=history_length
        )

        # Generate surrogates of source and calculate TE for each
        surrogate_te_values = []
        for surr_idx in range(n_surrogates):
            surrogate_source = generate_surrogate(source_window, surrogate_type)
            surr_te = calculate_te_single_window_standalone(
                surrogate_source, target_window, tau, history_length=history_length
            )
            surrogate_te_values.append(surr_te)

        # Calculate significance statistics from surrogate distribution
        if len(surrogate_te_values) > 0:
            surrogate_array = np.array(surrogate_te_values)
            surrogate_clean = surrogate_array[np.isfinite(surrogate_array)]

            if len(surrogate_clean) > 0:
                surrogate_mean = np.mean(surrogate_clean)
                surrogate_std = np.std(surrogate_clean)
                threshold_95 = np.percentile(surrogate_clean, 95)
                threshold_99 = np.percentile(surrogate_clean, 99)
                p_value = np.sum(surrogate_clean >= original_te) / len(surrogate_clean)
                lower_ci = np.percentile(surrogate_clean, 2.5)
                upper_ci = np.percentile(surrogate_clean, 97.5)
                n_valid = len(surrogate_clean)
            else:
                surrogate_mean = surrogate_std = 0.0
                threshold_95 = threshold_99 = 0.0
                p_value = 1.0
                lower_ci = upper_ci = 0.0
                n_valid = 0
        else:
            surrogate_mean = surrogate_std = 0.0
            threshold_95 = threshold_99 = 0.0
            p_value = 1.0
            lower_ci = upper_ci = 0.0
            n_valid = 0

        return {
            'window_idx': window_idx,
            'time_point': time_point,
            'original_te': original_te,
            'surrogate_mean': surrogate_mean,
            'surrogate_std': surrogate_std,
            'threshold_95': threshold_95,
            'threshold_99': threshold_99,
            'p_value': p_value,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'n_surrogates': len(surrogate_te_values),
            'n_valid_surrogates': n_valid,
            'significant_95': original_te > threshold_95,
            'significant_99': original_te > threshold_99,
            'surrogate_type': surrogate_type
        }

    except Exception:
        return {
            'window_idx': window_idx,
            'time_point': time_point,
            'original_te': 0.0,
            'surrogate_mean': 0.0,
            'surrogate_std': 0.0,
            'threshold_95': 0.0,
            'threshold_99': 0.0,
            'p_value': 1.0,
            'lower_ci': 0.0,
            'upper_ci': 0.0,
            'n_surrogates': 0,
            'n_valid_surrogates': 0,
            'significant_95': False,
            'significant_99': False,
            'surrogate_type': surrogate_type
        }


def surrogate_jte_test_single_window(args: Tuple) -> Dict:
    """
    Perform surrogate significance testing for JTE in a single time window.

    Generates surrogates for all source variables simultaneously and
    calculates JTE for each surrogate set.

    Parameters
    ----------
    args : Tuple
        (window_data, source_vars, target_var, tau, n_surrogates, window_idx,
         time_point, surrogate_type, tau_list)

    Returns
    -------
    result : Dict
        Window results including original JTE, surrogate distribution statistics,
        p-value, and significance threshold
    """
    # Unpack args tuple — history_length is optional 10th element
    if len(args) == 10:
        (window_data, source_vars, target_var, tau, n_surrogates, window_idx,
         time_point, surrogate_type, tau_list, history_length) = args
    else:
        (window_data, source_vars, target_var, tau, n_surrogates, window_idx,
         time_point, surrogate_type, tau_list) = args
        history_length = 1

    try:
        source_windows = [window_data[var].values for var in source_vars]
        target_window = window_data[target_var].values

        # Calculate original JTE
        original_jte = calculate_jte_single_window_standalone(
            source_windows, target_window, tau, tau_list=tau_list,
            history_length=history_length
        )

        # Generate surrogates for all sources and calculate JTE
        surrogate_jte_values = []
        for surr_idx in range(n_surrogates):
            surrogate_sources = [
                generate_surrogate(src, surrogate_type) for src in source_windows
            ]
            surr_jte = calculate_jte_single_window_standalone(
                surrogate_sources, target_window, tau, tau_list=tau_list,
                history_length=history_length
            )
            surrogate_jte_values.append(surr_jte)

        # Calculate significance statistics
        if len(surrogate_jte_values) > 0:
            surrogate_array = np.array(surrogate_jte_values)
            surrogate_clean = surrogate_array[np.isfinite(surrogate_array)]

            if len(surrogate_clean) > 0:
                surrogate_mean = np.mean(surrogate_clean)
                surrogate_std = np.std(surrogate_clean)
                threshold_95 = np.percentile(surrogate_clean, 95)
                threshold_99 = np.percentile(surrogate_clean, 99)
                p_value = np.sum(surrogate_clean >= original_jte) / len(surrogate_clean)
                lower_ci = np.percentile(surrogate_clean, 2.5)
                upper_ci = np.percentile(surrogate_clean, 97.5)
                n_valid = len(surrogate_clean)
            else:
                surrogate_mean = surrogate_std = 0.0
                threshold_95 = threshold_99 = 0.0
                p_value = 1.0
                lower_ci = upper_ci = 0.0
                n_valid = 0
        else:
            surrogate_mean = surrogate_std = 0.0
            threshold_95 = threshold_99 = 0.0
            p_value = 1.0
            lower_ci = upper_ci = 0.0
            n_valid = 0

        return {
            'window_idx': window_idx,
            'time_point': time_point,
            'original_jte': original_jte,
            'surrogate_mean': surrogate_mean,
            'surrogate_std': surrogate_std,
            'threshold_95': threshold_95,
            'threshold_99': threshold_99,
            'p_value': p_value,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci,
            'n_surrogates': len(surrogate_jte_values),
            'n_valid_surrogates': n_valid,
            'significant_95': original_jte > threshold_95,
            'significant_99': original_jte > threshold_99,
            'surrogate_type': surrogate_type
        }

    except Exception:
        return {
            'window_idx': window_idx,
            'time_point': time_point,
            'original_jte': 0.0,
            'surrogate_mean': 0.0,
            'surrogate_std': 0.0,
            'threshold_95': 0.0,
            'threshold_99': 0.0,
            'p_value': 1.0,
            'lower_ci': 0.0,
            'upper_ci': 0.0,
            'n_surrogates': 0,
            'n_valid_surrogates': 0,
            'significant_95': False,
            'significant_99': False,
            'surrogate_type': surrogate_type
        }


# ==========================================================================
# Main Surrogate Analyzer Class
# ==========================================================================

class SurrogateAnalyzer:
    """
    Surrogate-based significance testing for transfer entropy.

    Tests whether observed transfer entropy values are statistically
    significant by comparing them against a null distribution generated
    from surrogate time series that preserve certain statistical properties
    while destroying the temporal coupling between variables.

    Parameters
    ----------
    confidence_level : float, optional
        Confidence level for significance testing (default 0.95)
    n_cores : int, optional
        Number of CPU cores for parallel processing
    """

    def __init__(self, confidence_level: float = 0.95, n_cores: int = None):
        """
        Initialize surrogate analyzer.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for significance testing (default 0.95)
        n_cores : int, optional
            Number of CPU cores for parallel processing.
            Defaults to system cores - 2.
        """
        if n_cores is None:
            self.n_cores = max(1, cpu_count() - 2)
        else:
            self.n_cores = n_cores

        self.confidence_level = confidence_level

    def determine_data_frequency(self, df: pd.DataFrame, time_col: str,
                                  window_days: int) -> Tuple[str, int]:
        """
        Determine data frequency and calculate window size in data points.

        Parameters
        ----------
        df : pd.DataFrame
            Input time series data
        time_col : str
            Time column name
        window_days : int
            Window size in days

        Returns
        -------
        freq : str
            Data frequency string
        window_points : int
            Number of data points per window
        """
        if time_col in df.columns:
            times = pd.to_datetime(df[time_col])
        else:
            times = df.index

        time_diffs = times.diff().dropna()
        median_diff = time_diffs.median()
        hours = median_diff.total_seconds() / 3600

        if abs(hours - 1) < 0.1:
            freq = '1H'
            window_points = window_days * 24
        elif abs(hours - 4) < 0.1:
            freq = '4H'
            window_points = window_days * 6
        elif abs(hours - 6) < 0.1:
            freq = '6H'
            window_points = window_days * 4
        elif abs(hours - 12) < 0.1:
            freq = '12H'
            window_points = window_days * 2
        else:
            freq = f'inferred_{hours:.1f}H'
            points_per_day = 24 / hours
            window_points = int(window_days * points_per_day)

        return freq, window_points

    def create_rolling_windows(self, df: pd.DataFrame, window_points: int,
                                time_col: str) -> List[Tuple]:
        """
        Create rolling windows with their data and metadata.

        Parameters
        ----------
        df : pd.DataFrame
            Input time series data
        window_points : int
            Number of points per window
        time_col : str
            Time column name

        Returns
        -------
        windows : List[Tuple]
            List of (window_data, center_idx, timestamp) tuples
        """
        n_points = len(df)
        half_window = window_points // 2

        if time_col in df.columns:
            times = pd.to_datetime(df[time_col])
        else:
            times = df.index

        windows = []
        for center_idx in range(half_window, n_points - half_window):
            start_idx = center_idx - half_window
            end_idx = center_idx + half_window
            window_data = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
            window_timestamp = times.iloc[center_idx]
            windows.append((window_data, center_idx, window_timestamp))

        return windows

    def calculate_surrogate_confidence_intervals(
            self, data_file: str, target_var: str,
            input_vars: List[str], window_days: int = 30,
            tau: int = 1, time_col: str = 'time',
            n_surrogates: int = 100,
            surrogate_type: str = 'iaaft',
            tau_dict: Optional[Dict[str, int]] = None,
            progress_callback: Optional[Callable] = None,
            history_length: int = 1) -> Dict:
        """
        Calculate surrogate-based significance testing for transfer entropy.

        For each input->target pair and each rolling window:
        1. Calculate original TE
        2. Generate n_surrogates surrogate source time series
        3. Calculate TE for each surrogate
        4. Determine p-value and significance

        Parameters
        ----------
        data_file : str
            Path to data CSV file
        target_var : str
            Target variable name
        input_vars : List[str]
            Input variable names
        window_days : int
            Rolling window size in days
        tau : int
            Time lag parameter
        time_col : str
            Time column name
        n_surrogates : int
            Number of surrogate time series per window
        surrogate_type : str
            Type of surrogate ('iaaft', 'random_shuffle', 'block_shuffle',
            'phase_randomization')
        progress_callback : callable, optional
            Function to report progress

        Returns
        -------
        results : Dict
            Surrogate significance testing results with same structure as
            MCMCNoiseAnalyzer for plotting compatibility
        """
        start_time = time.time()

        if progress_callback:
            progress_callback(
                f"Loading data for {surrogate_type} surrogate testing..."
            )

        # Validate surrogate type
        valid_types = ['iaaft', 'random_shuffle', 'block_shuffle',
                       'phase_randomization']
        if surrogate_type not in valid_types:
            raise ValueError(
                f"Invalid surrogate type '{surrogate_type}'. "
                f"Must be one of {valid_types}"
            )

        # Load data
        data_path = Path(data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        data = pd.read_csv(data_path)

        # Determine data frequency and create windows
        freq, window_points = self.determine_data_frequency(
            data, time_col, window_days
        )
        windows = self.create_rolling_windows(data, window_points, time_col)

        if len(windows) == 0:
            raise ValueError(f"No valid windows for {window_days}-day windows")

        print(f"Surrogate testing: {len(windows)} windows, "
              f"{n_surrogates} surrogates, type={surrogate_type}")

        # Initialize results structure (compatible with MCMCNoiseAnalyzer)
        ci_results = {target_var: {}}

        # Process each input variable
        for input_idx, input_var in enumerate(input_vars):
            # Resolve per-variable tau from tau_dict if available
            var_tau = tau_dict.get(input_var, tau) if tau_dict else tau

            if progress_callback:
                progress_callback(
                    f"Surrogate testing {input_var} -> {target_var}, "
                    f"τ={var_tau} ({input_idx + 1}/{len(input_vars)})"
                )

            print(f"  Surrogate per-pair: {input_var} -> {target_var}, τ={var_tau}" +
                  (f" (from tau_dict)" if tau_dict else " (global)"))

            # Create arguments for parallel processing
            surrogate_args = []
            for window_idx, (window_data, center_idx, timestamp) in enumerate(windows):
                surrogate_args.append((
                    window_data, input_var, target_var, var_tau, n_surrogates,
                    window_idx, timestamp, surrogate_type, history_length
                ))

            if progress_callback:
                progress_callback(
                    f"Running parallel surrogate testing for "
                    f"{len(windows)} windows"
                )

            # Calculate surrogate significance for all windows in parallel
            with Pool(processes=self.n_cores) as pool:
                window_results = pool.map(
                    surrogate_test_single_window, surrogate_args
                )

            # Sort by window index
            window_results.sort(key=lambda x: x['window_idx'])

            # Convert to time series CI format (compatible with MCMCNoiseAnalyzer)
            time_series_ci = []
            for result in window_results:
                time_series_ci.append({
                    'time_point': result['time_point'],
                    'original_te': result['original_te'],
                    'lower_ci': result['lower_ci'],
                    'upper_ci': result['upper_ci'],
                    'monte_carlo_mean': result['surrogate_mean'],
                    'monte_carlo_std': result['surrogate_std'],
                    'n_monte_carlo_samples': result['n_surrogates'],
                    # Surrogate-specific fields
                    'p_value': result['p_value'],
                    'threshold_95': result['threshold_95'],
                    'threshold_99': result['threshold_99'],
                    'significant_95': result['significant_95'],
                    'significant_99': result['significant_99']
                })

            # Store results
            ci_results[target_var][input_var] = {
                'time_series_ci': time_series_ci,
                'metadata': {
                    'n_time_points': len(time_series_ci),
                    'n_surrogates_per_window': n_surrogates,
                    'confidence_level': self.confidence_level,
                    'window_days': window_days,
                    'tau': var_tau,
                    'tau_dict': tau_dict,
                    'history_length': history_length,
                    'method': 'surrogate_testing',
                    'surrogate_type': surrogate_type,
                    'n_cores_used': self.n_cores,
                    'data_frequency': freq,
                    'window_points': window_points
                }
            }

        # Add overall metadata
        elapsed_time = time.time() - start_time
        ci_results['metadata'] = {
            'data_file': Path(data_file).name,
            'target_var': target_var,
            'input_vars': input_vars,
            'n_surrogates_per_window': n_surrogates,
            'confidence_level': self.confidence_level,
            'window_days': window_days,
            'tau': tau,
            'method': 'surrogate_testing',
            'surrogate_type': surrogate_type,
            'tau_dict': tau_dict,
            'history_length': history_length,
            'calculation_time': elapsed_time,
            'n_cores_used': self.n_cores,
            'n_time_windows': len(windows),
            'timestamp': datetime.now().isoformat(),
            'version': 'V1.0.0'
        }

        if progress_callback:
            progress_callback(
                f"Surrogate analysis complete in {elapsed_time:.1f} seconds"
            )

        return ci_results

    def calculate_jte_surrogate_confidence_intervals(
            self, data_file: str, target_var: str,
            input_vars: List[str], window_days: int = 30,
            tau: int = 1, time_col: str = 'time',
            n_surrogates: int = 100,
            surrogate_type: str = 'iaaft',
            tau_dict: Optional[Dict[str, int]] = None,
            progress_callback: Optional[Callable] = None,
            history_length: int = 1) -> Dict:
        """
        Calculate surrogate-based significance testing for Joint Transfer Entropy.

        Parameters
        ----------
        data_file : str
            Path to data CSV file
        target_var : str
            Target variable name
        input_vars : List[str]
            Input variable names (all used jointly)
        window_days : int
            Rolling window size in days
        tau : int
            Time lag parameter
        time_col : str
            Time column name
        n_surrogates : int
            Number of surrogate sets per window
        surrogate_type : str
            Type of surrogate
        progress_callback : callable, optional
            Function to report progress

        Returns
        -------
        results : Dict
            JTE surrogate significance testing results
        """
        start_time = time.time()

        if progress_callback:
            progress_callback(
                f"JTE Surrogate: Loading data for {surrogate_type} testing..."
            )

        if len(input_vars) < 2:
            raise ValueError("JTE requires at least 2 input variables")

        valid_types = ['iaaft', 'random_shuffle', 'block_shuffle',
                       'phase_randomization']
        if surrogate_type not in valid_types:
            raise ValueError(
                f"Invalid surrogate type '{surrogate_type}'. "
                f"Must be one of {valid_types}"
            )

        # Load data
        data_path = Path(data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        data = pd.read_csv(data_path)

        # Determine data frequency and create windows
        freq, window_points = self.determine_data_frequency(
            data, time_col, window_days
        )
        windows = self.create_rolling_windows(data, window_points, time_col)

        if len(windows) == 0:
            raise ValueError(f"No valid windows for {window_days}-day windows")

        # Build per-source tau_list from tau_dict
        tau_list = [tau_dict.get(var, tau) for var in input_vars] if tau_dict else None

        if tau_list:
            print(f"  JTE Surrogate: Using differential lags tau_list={tau_list} for {input_vars}")
        else:
            print(f"  JTE Surrogate: Using global tau={tau} for all {len(input_vars)} sources")

        if progress_callback:
            progress_callback(
                f"JTE Surrogate: {len(windows)} windows, "
                f"{len(input_vars)} joint inputs"
            )

        # Create arguments for parallel processing
        surrogate_args = []
        for window_idx, (window_data, center_idx, timestamp) in enumerate(windows):
            surrogate_args.append((
                window_data, input_vars, target_var, tau, n_surrogates,
                window_idx, timestamp, surrogate_type, tau_list, history_length
            ))

        # Calculate JTE surrogate significance for all windows in parallel
        with Pool(processes=self.n_cores) as pool:
            window_results = pool.map(
                surrogate_jte_test_single_window, surrogate_args
            )

        # Sort by window index
        window_results.sort(key=lambda x: x['window_idx'])

        # Convert to time series format
        time_series_ci = []
        for result in window_results:
            time_series_ci.append({
                'time_point': result['time_point'],
                'original_jte': result['original_jte'],
                'lower_ci': result['lower_ci'],
                'upper_ci': result['upper_ci'],
                'monte_carlo_mean': result['surrogate_mean'],
                'monte_carlo_std': result['surrogate_std'],
                'n_monte_carlo_samples': result['n_surrogates'],
                'p_value': result['p_value'],
                'threshold_95': result['threshold_95'],
                'threshold_99': result['threshold_99'],
                'significant_95': result['significant_95'],
                'significant_99': result['significant_99']
            })

        elapsed_time = time.time() - start_time

        combination_key = "+".join(sorted(input_vars))
        jte_surr_results = {
            'jte_time_series_ci': time_series_ci,
            'metadata': {
                'data_file': Path(data_file).name,
                'target_var': target_var,
                'input_vars': input_vars,
                'combination_key': combination_key,
                'n_time_points': len(time_series_ci),
                'n_surrogates_per_window': n_surrogates,
                'confidence_level': self.confidence_level,
                'window_days': window_days,
                'tau': tau,
                'tau_dict': tau_dict,
                'tau_list': tau_list,
                'history_length': history_length,
                'method': 'jte_surrogate_testing',
                'surrogate_type': surrogate_type,
                'n_cores_used': self.n_cores,
                'data_frequency': freq,
                'window_points': window_points,
                'calculation_time': elapsed_time,
                'timestamp': datetime.now().isoformat(),
                'version': 'V1.0.0'
            }
        }

        if progress_callback:
            progress_callback(
                f"JTE Surrogate analysis complete in {elapsed_time:.1f} seconds"
            )

        return jte_surr_results

    def extract_ci_time_series(self, ci_results: Dict) -> Dict[str, pd.DataFrame]:
        """
        Extract confidence interval time series for plotting.

        Compatible with MCMCNoiseAnalyzer.extract_ci_time_series() output format
        so existing plotting code can handle both methods.

        Parameters
        ----------
        ci_results : Dict
            Results from calculate_surrogate_confidence_intervals()

        Returns
        -------
        ci_time_series : Dict[str, pd.DataFrame]
            DataFrames with original_te, lower_ci, upper_ci columns
            per relationship
        """
        ci_time_series = {}

        for target_var, target_data in ci_results.items():
            if target_var == 'metadata':
                continue

            for input_var, input_data in target_data.items():
                if 'time_series_ci' not in input_data:
                    continue

                time_series_ci = input_data['time_series_ci']

                df_data = {
                    'date': [p.get('time_point') for p in time_series_ci],
                    'original_te': [p['original_te'] for p in time_series_ci],
                    'lower_ci': [p['lower_ci'] for p in time_series_ci],
                    'upper_ci': [p['upper_ci'] for p in time_series_ci],
                    'monte_carlo_mean': [p['monte_carlo_mean'] for p in time_series_ci],
                    'monte_carlo_std': [p['monte_carlo_std'] for p in time_series_ci],
                    # Surrogate-specific columns
                    'p_value': [p.get('p_value', np.nan) for p in time_series_ci],
                    'threshold_95': [p.get('threshold_95', np.nan) for p in time_series_ci],
                    'significant_95': [p.get('significant_95', False) for p in time_series_ci]
                }

                df = pd.DataFrame(df_data)
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['relationship'] = f"{input_var} -> {target_var}"

                ci_time_series[f"{input_var}_to_{target_var}"] = df

        return ci_time_series

    def extract_jte_ci_time_series(self, jte_ci_results: Dict) -> pd.DataFrame:
        """
        Extract JTE surrogate confidence interval time series for plotting.

        Parameters
        ----------
        jte_ci_results : Dict
            Results from calculate_jte_surrogate_confidence_intervals()

        Returns
        -------
        df : pd.DataFrame
            DataFrame with JTE results and significance information
        """
        time_series_ci = jte_ci_results.get('jte_time_series_ci', [])
        metadata = jte_ci_results.get('metadata', {})

        df_data = {
            'date': [p.get('time_point') for p in time_series_ci],
            'original_jte': [p['original_jte'] for p in time_series_ci],
            'lower_ci': [p['lower_ci'] for p in time_series_ci],
            'upper_ci': [p['upper_ci'] for p in time_series_ci],
            'monte_carlo_mean': [p['monte_carlo_mean'] for p in time_series_ci],
            'monte_carlo_std': [p['monte_carlo_std'] for p in time_series_ci],
            'p_value': [p.get('p_value', np.nan) for p in time_series_ci],
            'threshold_95': [p.get('threshold_95', np.nan) for p in time_series_ci],
            'significant_95': [p.get('significant_95', False) for p in time_series_ci]
        }

        df = pd.DataFrame(df_data)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['combination'] = metadata.get('combination_key', 'Unknown')
        df['target'] = metadata.get('target_var', 'Unknown')

        return df


if __name__ == "__main__":
    print("Transfer Entropy Surrogate Testing V1.0.0")
    print("Surrogate types: IAAFT, Random Shuffle, Block Shuffle, "
          "Phase Randomization")
    print("Import this module and use SurrogateAnalyzer class")
