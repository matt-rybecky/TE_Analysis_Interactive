#!/usr/bin/env python3
"""
Transfer Entropy Calculator V1.0.0

This module implements rolling window transfer entropy calculations with configurable 
windows sliding by single time steps. Designed for atmospheric time series analysis
with parallel processing for multiple target variables.

Mathematical Framework:
- Rolling TE: T_{X→Y}(t) = TE calculated on window [t-window_size/2, t+window_size/2]
- Window slides: t = window_size/2+1, window_size/2+2, ..., N-window_size/2
- Configurable τ: variable time lag between source and target

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
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add NPEET to Python path
npeet_path = Path(__file__).parent / 'NPEET'
if npeet_path.exists():
    sys.path.insert(0, str(npeet_path))

from npeet import entropy_estimators as ee


# ═══════════════════════════════════════════════════════════════════════════
# Vector-source support (circular wind direction and other multi-column inputs)
# ═══════════════════════════════════════════════════════════════════════════

def build_col_map(input_vars: List[str],
                  vector_bases: Dict[str, Tuple[str, ...]] = None) -> Dict[str, List[str]]:
    """
    Map each logical candidate name to its underlying data column(s).

    A scalar candidate maps to itself. A vector candidate (a base listed in
    ``vector_bases``, e.g. ``met_wdir`` -> sin/cos) maps to one data column per
    component, preserving any ``__tK`` lag suffix. This lets one logical variable
    enter the transfer-entropy estimate as a multi-column (e.g. 2D circular)
    source while combination enumeration, categorization, and tables continue to
    operate on the single logical name.

    Parameters
    ----------
    input_vars : list of str
        Logical candidate names (``base`` or ``base__tK``).
    vector_bases : dict, optional
        Map ``base -> (component, ...)``; e.g. ``{'met_wdir': ('sin', 'cos')}``.
        Component column names are ``f"{base}_{component}{suffix}"``.

    Returns
    -------
    col_map : dict
        ``logical_name -> [data_column, ...]`` (scalar -> single-element list).
    """
    vector_bases = vector_bases or {}
    col_map: Dict[str, List[str]] = {}
    for name in input_vars:
        if '__t' in name:
            base, tag = name.rsplit('__t', 1)
            suffix = '__t' + tag
        else:
            base, suffix = name, ''
        if base in vector_bases:
            col_map[name] = [f"{base}_{c}{suffix}" for c in vector_bases[base]]
        else:
            col_map[name] = [name]
    return col_map


def extract_source(df: pd.DataFrame, name: str, col_map: Dict[str, List[str]],
                   start: int, end: int) -> np.ndarray:
    """
    Extract a window of one logical source as a 1D (scalar) or 2D (vector) array.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame holding the underlying data columns.
    name : str
        Logical candidate name.
    col_map : dict
        Output of :func:`build_col_map` (``None``/missing -> treat as scalar).
    start, end : int
        Row slice bounds (``iloc[start:end]``).

    Returns
    -------
    arr : np.ndarray
        Shape ``(n,)`` for a scalar source, ``(n, k)`` for a k-component vector.
    """
    cols = (col_map or {}).get(name, [name])
    arr = df[cols].iloc[start:end].to_numpy(dtype=float)
    return arr[:, 0] if arr.shape[1] == 1 else arr


def _row_valid_mask(arr: np.ndarray) -> np.ndarray:
    """Per-row finite mask: a row is valid only if all its components are finite."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return ~np.isnan(arr)
    return ~np.isnan(arr).any(axis=1)


def _as_2d(arr: np.ndarray) -> np.ndarray:
    """Reshape a 1D array to a single column; leave 2D arrays unchanged."""
    arr = np.asarray(arr)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


class TECalculator:
    """
    Rolling window transfer entropy calculator for time series analysis.
    
    Implements configurable-size windows with single time step sliding to analyze
    temporal dynamics of information transfer between atmospheric variables.
    """
    
    def __init__(self, n_cores: int = None, window_days: int = 30, tau: int = 1,
                 history_length: int = 1):
        """
        Initialize rolling window TE calculator.

        Parameters
        ----------
        n_cores : int, optional
            Number of CPU cores for parallel processing.
            Defaults to system cores - 2.
        window_days : int, optional
            Size of rolling window in days. Defaults to 30 days.
        tau : int, optional
            Time lag parameter for TE calculation. Defaults to 1.
        history_length : int, optional
            Number of past target values to condition on (h >= 1).
            h=1 gives traditional TE: I(Y_t; X_{t-τ} | Y_{t-1}).
            h>1 extends conditioning: I(Y_t; X_{t-τ} | Y_{t-1}, ..., Y_{t-h}).
            Defaults to 1.
        """

        # Set up multiprocessing
        if n_cores is None:
            self.n_cores = max(1, cpu_count() - 2)  # Leave 2 cores for system
        else:
            self.n_cores = n_cores

        # Store window configuration
        self.window_days = window_days
        self.tau = tau
        self.history_length = max(0, history_length)
        
        
    def determine_data_frequency(self, df: pd.DataFrame, time_col: str = 'timestamp') -> Tuple[str, int]:
        """
        Determine data frequency and calculate window size in data points.
        
        Parameters
        ----------
        df : pd.DataFrame
            Time series data with datetime index or column
        time_col : str
            Name of time column if not using index
            
        Returns
        -------
        freq : str
            Data frequency ('1H', '4H', '6H', '12H')
        window_points : int
            Number of data points in 30-day window
        """
        # Get time series
        if time_col in df.columns:
            times = pd.to_datetime(df[time_col])
        else:
            times = df.index
            
        # Calculate median time difference
        time_diffs = times.diff().dropna()
        median_diff = time_diffs.median()
        
        # Determine frequency and calculate window size based on configured days
        hours = median_diff.total_seconds() / 3600
        
        if abs(hours - 1) < 0.1:
            freq = '1H'
            window_points = self.window_days * 24  # points per day * days
        elif abs(hours - 4) < 0.1:
            freq = '4H'
            window_points = self.window_days * 6   # 6 points per day * days
        elif abs(hours - 6) < 0.1:
            freq = '6H'
            window_points = self.window_days * 4   # 4 points per day * days
        elif abs(hours - 12) < 0.1:
            freq = '12H'
            window_points = self.window_days * 2   # 2 points per day * days
        else:
            # Default assumption - calculate based on detected frequency
            freq = f'inferred_{hours:.1f}H'
            points_per_day = 24 / hours
            window_points = int(self.window_days * points_per_day)
            
        return freq, window_points
        
    def create_rolling_windows(self, df: pd.DataFrame, window_points: int) -> Tuple[List[int], List[int]]:
        """
        Create rolling window indices with edge exclusion.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input time series data
        window_points : int
            Number of points in each rolling window
            
        Returns
        -------
        window_centers : List[int]
            Center indices for each window
        valid_indices : List[int]
            Valid data indices (excluding edges)
        """
        n_points = len(df)
        half_window = window_points // 2
        
        # Exclude edges - start at half_window, end at n_points - half_window
        window_centers = list(range(half_window, n_points - half_window))
        valid_indices = list(range(half_window, n_points - half_window))
        
        return window_centers, valid_indices
        
    def calculate_te_single_window(self, source_data: np.ndarray, target_data: np.ndarray,
                                     tau_override: int = None) -> float:
        """
        Calculate transfer entropy for single window with configurable time lag τ.

        Parameters
        ----------
        source_data : np.ndarray
            Source variable data for window
        target_data : np.ndarray
            Target variable data for window
        tau_override : int, optional
            Override the default tau for this specific calculation.
            Used for differential lagging where each input has its own tau.

        Returns
        -------
        te : float
            Transfer entropy value in bits
        """
        try:
            # Use override tau if provided, otherwise use instance default
            tau = tau_override if tau_override is not None else self.tau

            # TE(source→target) = I(target_t; source_{t-τ} | target_{t-1})
            # Where τ is the time lag parameter

            # Remove NaN values. Source may be 1D (scalar) or 2D (vector, e.g.
            # circular wind direction as sin/cos); a row is valid only if all
            # its components and the target are finite.
            source_data = np.asarray(source_data)
            source_mask = _row_valid_mask(source_data)
            target_mask = ~np.isnan(target_data)
            mask = source_mask & target_mask

            if np.sum(mask) < 10:  # Need minimum points
                return 0.0

            source_clean = source_data[mask]
            target_clean = target_data[mask]

            if len(source_clean) < 10 or len(target_clean) < 10:
                return 0.0

            # Need enough data points for the specified lag and history length
            h = self.history_length
            offset = max(tau, h) if tau > 0 else max(h, 1)
            min_length = max(2, offset + 2)
            if len(target_clean) < min_length or len(source_clean) < min_length:
                return 0.0

            # Create arrays for TE calculation with configurable tau and history length h
            # h>0: TE(X→Y) = I(Y_t; X_{t-τ} | Y_{t-1}, ..., Y_{t-h})
            # h=0: MI(Y_t; X_{t-τ})  — no conditioning on target past
            n_out = len(target_clean) - offset

            target_present = target_clean[offset:]  # Y_t, shape (n_out,)

            # Row-lag the source (works for 1D scalar or 2D vector sources).
            if tau == 0:
                source_lagged = source_clean[offset:]
            else:
                source_lagged = source_clean[offset - tau:offset - tau + n_out]

            if n_out < 5:  # Need minimum for KSG
                return 0.0

            # Ensure arrays are 2D for NPEET (vector source keeps its columns).
            target_present = target_present.reshape(-1, 1)
            source_variable = _as_2d(source_lagged)

            if h == 0:
                # No conditioning — compute plain mutual information I(X; Y)
                te_bits = ee.mi(source_variable, target_present, k=3, base=2)
            else:
                # Target history: stack h past values as columns
                target_past = np.column_stack([
                    target_clean[offset - j:offset - j + n_out] for j in range(1, h + 1)
                ])  # shape (n_out, h)

                # Conditional MI: I(source; target_present | target_past)
                te_bits = ee.mi(source_variable, target_present, target_past, k=3, base=2)

            # Validate result and ensure non-negative
            if np.isnan(te_bits) or np.isinf(te_bits) or te_bits < 0:
                return 0.0

            return te_bits

        except Exception:
            return 0.0

    def calculate_entropy_rolling(self, df: pd.DataFrame, target_var: str,
                                  time_col: str = 'timestamp') -> Tuple[np.ndarray, List[int]]:
        """
        Calculate Shannon entropy of target variable for each rolling window.

        H(Y) represents the total information content (uncertainty) in the target variable
        within each window. This serves as an upper bound reference for transfer entropy,
        since TE(X→Y) ≤ H(Y|Y_past) ≤ H(Y).

        For Clausius-Clapeyron normalized data, H(temperature) should approximately equal
        the Joint MI of RH and vapor pressure with temperature, providing a calibration
        reference for information metrics.

        Parameters
        ----------
        df : pd.DataFrame
            Input time series data
        target_var : str
            Target variable name
        time_col : str
            Time column name

        Returns
        -------
        entropy_timeseries : np.ndarray
            Shannon entropy values for each window (in bits)
        window_centers : List[int]
            Window center indices
        """
        # Determine data frequency and window size
        freq, window_points = self.determine_data_frequency(df, time_col)

        # Create rolling windows
        window_centers, valid_indices = self.create_rolling_windows(df, window_points)

        if len(window_centers) == 0:
            raise ValueError(f"No valid windows - dataset too small for {self.window_days}-day windows")

        if target_var not in df.columns:
            raise ValueError(f"Target variable '{target_var}' not found in data")

        # Initialize entropy array
        half_window = window_points // 2
        entropy_timeseries = np.zeros(len(window_centers))

        logger.info(f"Calculating Shannon entropy for {target_var} over {len(window_centers)} windows...")

        # Calculate entropy for each window
        for i, center_idx in enumerate(window_centers):
            start_idx = center_idx - half_window
            end_idx = center_idx + half_window

            # Extract window data
            window_data = df[target_var].iloc[start_idx:end_idx].values

            # Calculate entropy using KSG estimator
            entropy_val = self._calculate_entropy_single_window(window_data)
            entropy_timeseries[i] = entropy_val

        logger.info(f"Entropy calculation complete: mean={np.mean(entropy_timeseries):.4f} bits")
        return entropy_timeseries, window_centers

    def _calculate_entropy_single_window(self, data: np.ndarray) -> float:
        """
        Calculate Shannon entropy for a single window using KSG estimator.

        Parameters
        ----------
        data : np.ndarray
            Data array for window

        Returns
        -------
        entropy : float
            Shannon entropy in bits
        """
        try:
            # Remove NaN values
            data_clean = data[~np.isnan(data)]

            if len(data_clean) < 10:  # Need minimum points for KSG
                return 0.0

            # Reshape for NPEET (requires 2D array)
            data_2d = data_clean.reshape(-1, 1)

            # Calculate entropy using NPEET KSG estimator
            entropy_bits = ee.entropy(data_2d, k=3, base=2)

            # Validate result
            if np.isnan(entropy_bits) or np.isinf(entropy_bits):
                return 0.0

            return max(0.0, entropy_bits)  # Entropy should be non-negative

        except Exception:
            return 0.0
            
    def process_input_target_pair(self, args: Tuple) -> Tuple[str, str, np.ndarray, int]:
        """
        Process single input→target combination in parallel.

        Parameters
        ----------
        args : tuple
            (input_var, target_var, df, window_centers, window_points, tau_override)
            tau_override is optional - if provided, uses per-variable tau

        Returns
        -------
        input_var : str
            Input variable name
        target_var : str
            Target variable name
        te_timeseries : np.ndarray
            TE time series for this input→target combination
        tau_used : int
            The tau value used for this calculation
        """
        # Handle both old format (5 args) and new format (6 args with tau_override)
        if len(args) == 6:
            input_var, target_var, df, window_centers, window_points, tau_override = args
        else:
            input_var, target_var, df, window_centers, window_points = args
            tau_override = None

        tau_used = tau_override if tau_override is not None else self.tau

        if input_var == target_var:
            # Skip self-loops
            return input_var, target_var, np.zeros(len(window_centers)), tau_used

        # Initialize results
        half_window = window_points // 2
        te_timeseries = np.zeros(len(window_centers))

        # Calculate TE for each window
        for i, center_idx in enumerate(window_centers):
            # Extract window data
            start_idx = center_idx - half_window
            end_idx = center_idx + half_window

            source_window = df[input_var].iloc[start_idx:end_idx].values
            target_window = df[target_var].iloc[start_idx:end_idx].values

            # Calculate TE for this window with optional tau override
            te_val = self.calculate_te_single_window(source_window, target_window, tau_override)
            te_timeseries[i] = te_val

        return input_var, target_var, te_timeseries, tau_used
        
    def calculate_jte_single_window(self, source_data_list: List[np.ndarray],
                                     target_data: np.ndarray,
                                     tau_list: List[int] = None) -> float:
        """
        Calculate Joint Transfer Entropy for a single window.

        JTE_{(X₁,...,Xₙ)→Y}(τ₁,...,τₙ) = I(Y_t ; X₁_{t-τ₁}, ..., Xₙ_{t-τₙ} | Y_{t-1})

        Supports differential lagging where each source variable can have
        a different time lag τ. When tau_list is None, uses self.tau for all.

        Parameters
        ----------
        source_data_list : List[np.ndarray]
            List of source variable arrays for the window
        target_data : np.ndarray
            Target variable array for the window
        tau_list : List[int], optional
            Per-source tau values. Must match length of source_data_list.
            If None, uses self.tau for all sources.

        Returns
        -------
        jte : float
            Joint transfer entropy in bits
        """
        try:
            n_sources = len(source_data_list)

            # Resolve per-source tau values
            if tau_list is None:
                taus = [self.tau] * n_sources
            else:
                taus = tau_list

            # Build combined valid mask across all sources and target. Sources
            # may be 1D (scalar) or 2D (vector, e.g. circular wind direction);
            # a row counts only where the target and every source component are
            # finite.
            target_mask = ~np.isnan(target_data)
            combined_mask = target_mask.copy()

            for source_data in source_data_list:
                combined_mask = combined_mask & _row_valid_mask(source_data)

            if np.sum(combined_mask) < 10:
                return 0.0

            # Clean target data
            target_clean = target_data[combined_mask]
            sources_clean = [src[combined_mask] for src in source_data_list]

            if len(target_clean) < 10:
                return 0.0

            # The effective offset accounts for both max tau and history length
            max_tau = max(taus)
            h = self.history_length
            offset = max(max_tau, h) if max_tau > 0 else max(h, 1)
            min_length = max(2, offset + 2)
            if len(target_clean) < min_length:
                return 0.0

            # Create arrays for JTE with per-source differential lags and history length h
            # h>0: JTE = I(Y_t; X1_{t-τ1},...,Xn_{t-τn} | Y_{t-1},...,Y_{t-h})
            # h=0: Joint MI = I(Y_t; X1_{t-τ1},...,Xn_{t-τn})
            n_out = len(target_clean) - offset

            target_present = target_clean[offset:]  # Y_t

            # Each source is lagged by its own tau, all trimmed to same length
            sources_past = []
            for src, t in zip(sources_clean, taus):
                if t == 0:
                    sources_past.append(src[offset:offset + n_out])
                else:
                    sources_past.append(src[offset - t:offset - t + n_out])

            if n_out < 5:
                return 0.0

            # Stack all sources into 2D array (n_samples x n_sources)
            sources_stacked = np.column_stack(sources_past)

            # Reshape target arrays for NPEET
            target_present_2d = target_present.reshape(-1, 1)

            if h == 0:
                # No conditioning — joint mutual information I(sources; target)
                jte_bits = ee.mi(sources_stacked, target_present_2d, k=3, base=2)
            else:
                # Target history: stack h past values as columns
                target_past = np.column_stack([
                    target_clean[offset - j:offset - j + n_out] for j in range(1, h + 1)
                ])  # shape (n_out, h)

                # Conditional MI: I(sources; target_present | target_past)
                jte_bits = ee.cmi(sources_stacked, target_present_2d, target_past, k=3, base=2)

            if np.isnan(jte_bits) or np.isinf(jte_bits) or jte_bits < 0:
                return 0.0

            return jte_bits

        except Exception as e:
            return 0.0

    def run_jte_rolling_analysis(self, df: pd.DataFrame, target_var: str,
                                  input_vars: List[str], time_col: str = 'timestamp',
                                  progress_callback=None,
                                  tau_dict: Dict[str, int] = None,
                                  col_map: Dict[str, List[str]] = None) -> Dict[str, any]:
        """
        Run rolling window Joint Transfer Entropy analysis.

        Calculates both JTE (joint) and individual TEs for synergy analysis.
        Supports differential lagging where each input can have a different τ.

        Parameters
        ----------
        df : pd.DataFrame
            Input time series data
        target_var : str
            Target variable name
        input_vars : List[str]
            List of input variable names (sources)
        time_col : str
            Time column name
        progress_callback : callable, optional
            Function to report progress
        tau_dict : Dict[str, int], optional
            Dictionary mapping input variable names to their specific tau values.
            Enables differential lagging. If None, uses self.tau for all inputs.
        col_map : Dict[str, List[str]], optional
            Map of logical input name to its underlying data column(s). Vector
            inputs (e.g. circular wind direction) map to several columns and
            enter the estimator as a multi-column source. If None, every input
            is treated as a single column of the same name.

        Returns
        -------
        results : Dict
            Contains 'jte_timeseries', 'individual_te', 'timestamps',
            'input_vars', 'target_var', and 'synergy_stats'
        """
        col_map = col_map or {}

        # Determine data frequency and window size
        freq, window_points = self.determine_data_frequency(df, time_col)

        if progress_callback:
            progress_callback(f"JTE Analysis - Data frequency: {freq}, window: {window_points} points")

        # Create rolling windows
        window_centers, valid_indices = self.create_rolling_windows(df, window_points)

        if len(window_centers) == 0:
            raise ValueError(f"No valid windows - dataset too small for {self.window_days}-day windows")

        # Validate variables
        if target_var not in df.columns:
            raise ValueError(f"Target variable '{target_var}' not found in data")

        # A logical input is usable if it is a real column or expands (via
        # col_map) to data columns present in the frame.
        def _available(var: str) -> bool:
            cols = col_map.get(var, [var])
            return all(c in df.columns for c in cols)

        valid_inputs = [var for var in input_vars
                        if var != target_var and _available(var)]

        if len(valid_inputs) < 2:
            raise ValueError("JTE requires at least 2 input variables")

        # Build per-source tau list for JTE and individual tau overrides
        if tau_dict:
            tau_list = [tau_dict.get(var, self.tau) for var in valid_inputs]
            tau_str = ", ".join([f"{v}:τ={t}" for v, t in zip(valid_inputs, tau_list)])
            print(f"  JTE differential lags: {tau_str}")
        else:
            tau_list = None

        if progress_callback:
            progress_callback(f"Calculating JTE for {len(valid_inputs)} inputs → {target_var}")

        # Initialize results
        half_window = window_points // 2
        n_windows = len(window_centers)
        jte_timeseries = np.zeros(n_windows)
        individual_te = {var: np.zeros(n_windows) for var in valid_inputs}

        start_time = time.time()

        # Calculate for each window
        for i, center_idx in enumerate(window_centers):
            start_idx = center_idx - half_window
            end_idx = center_idx + half_window

            # Extract target window
            target_window = df[target_var].iloc[start_idx:end_idx].values

            # Extract all source windows (scalar -> 1D, vector -> 2D)
            source_windows = [extract_source(df, var, col_map, start_idx, end_idx)
                              for var in valid_inputs]

            # Calculate JTE (joint) with per-source differential lags
            jte_val = self.calculate_jte_single_window(source_windows, target_window, tau_list)
            jte_timeseries[i] = jte_val

            # Calculate individual TEs for synergy analysis (each with its own tau)
            for j, var in enumerate(valid_inputs):
                tau_override = tau_dict.get(var, self.tau) if tau_dict else None
                te_val = self.calculate_te_single_window(source_windows[j], target_window, tau_override)
                individual_te[var][i] = te_val

            # Progress update every 10%
            if progress_callback and (i + 1) % max(1, n_windows // 10) == 0:
                pct = (i + 1) / n_windows * 100
                progress_callback(f"JTE calculation: {pct:.0f}% complete")

        elapsed = time.time() - start_time

        if progress_callback:
            progress_callback(f"JTE calculation completed in {elapsed:.1f}s")

        # Get timestamps
        timestamps = df[time_col].iloc[window_centers]

        # Calculate synergy statistics
        sum_individual = np.sum([individual_te[var] for var in valid_inputs], axis=0)
        synergy = jte_timeseries - sum_individual

        synergy_stats = {
            'mean_jte': float(np.mean(jte_timeseries)),
            'mean_sum_individual_te': float(np.mean(sum_individual)),
            'mean_synergy': float(np.mean(synergy)),
            'synergy_fraction': float(np.mean(synergy) / np.mean(sum_individual)) if np.mean(sum_individual) > 0 else 0.0,
            'pct_synergistic': float(np.sum(synergy > 0) / len(synergy) * 100),
            'pct_redundant': float(np.sum(synergy < 0) / len(synergy) * 100)
        }

        results = {
            'jte_timeseries': jte_timeseries,
            'individual_te': individual_te,
            'sum_individual_te': sum_individual,
            'synergy_timeseries': synergy,
            'timestamps': timestamps,
            'input_vars': valid_inputs,
            'target_var': target_var,
            'synergy_stats': synergy_stats,
            'calculation_time': elapsed
        }

        return results

    def run_partial_analysis(self, df: pd.DataFrame, target_var: str,
                             input_vars: List[str], time_col: str = 'timestamp',
                             progress_callback=None,
                             tau_dict: Dict[str, int] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Run rolling window TE analysis for specific input→target combinations only.

        This method calculates TE for a single target with specified inputs,
        enabling incremental building of the full TE matrix.

        Parameters
        ----------
        df : pd.DataFrame
            Input time series data
        target_var : str
            Single target variable name
        input_vars : List[str]
            List of input variable names to calculate
        time_col : str
            Time column name
        progress_callback : callable, optional
            Function to report progress: callback(message: str)
        tau_dict : Dict[str, int], optional
            Dictionary mapping input variable names to their specific tau values.
            Enables differential lagging where each input can have different time lag.

        Returns
        -------
        results : Dict[str, Dict[str, np.ndarray]]
            Results for calculated combinations: {target_var: {combo_name: te_timeseries}}
        """
        # Determine data frequency and window size
        freq, window_points = self.determine_data_frequency(df, time_col)

        if progress_callback:
            progress_callback(f"Data frequency: {freq}, window: {window_points} points")

        # Create rolling windows
        window_centers, valid_indices = self.create_rolling_windows(df, window_points)

        if len(window_centers) == 0:
            raise ValueError(f"No valid windows - dataset too small for {self.window_days}-day windows")

        # Validate variables exist in data
        if target_var not in df.columns:
            raise ValueError(f"Target variable '{target_var}' not found in data")

        valid_inputs = [var for var in input_vars if var in df.columns and var != target_var]

        if not valid_inputs:
            raise ValueError("No valid input variables found in data")

        # Log tau configuration if using differential lags
        if tau_dict:
            tau_str = ", ".join([f"{v}:τ={tau_dict.get(v, self.tau)}" for v in valid_inputs[:3]])
            if len(valid_inputs) > 3:
                tau_str += f", +{len(valid_inputs)-3} more"
            if progress_callback:
                progress_callback(f"Differential lags: {tau_str}")

        if progress_callback:
            progress_callback(f"Calculating {len(valid_inputs)} combinations for target: {target_var}")

        # Create args for parallel processing with tau_dict support
        args_list = []
        for input_var in valid_inputs:
            tau_for_input = tau_dict.get(input_var, self.tau) if tau_dict else None
            args_list.append((input_var, target_var, df, window_centers, window_points, tau_for_input))

        start_time = time.time()

        # Run parallel processing
        with Pool(processes=self.n_cores) as pool:
            parallel_results = pool.map(self.process_input_target_pair, args_list)

        elapsed = time.time() - start_time

        if progress_callback:
            progress_callback(f"Completed {len(valid_inputs)} combinations in {elapsed:.1f}s")

        # Organize results
        results = {target_var: {}}
        for input_var, tgt_var, te_timeseries, tau_used in parallel_results:
            if input_var != tgt_var:  # Skip self-loops
                results[target_var][f"{input_var}_to_{target_var}"] = te_timeseries

        return results


if __name__ == "__main__":
    # Example usage
    print("Transfer Entropy Calculator V1.0.0")
    print("Import this module and use TECalculator class")