"""Pure helper functions for the TE Analysis tab.

These functions have no tkinter imports and no class state; they are
module-level so they can be unit-tested independently of the GUI.

Covers:
- DataFrame loading and result reshaping for the calculation threads.
- Entropy array interpolation.
- Descriptive label builders for τ, h, worker count, file info, and
  start-of-calculation status messages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import interpolate as scipy_interp


# ────────────────────────────────────────────────────────────────────────────
# Data I/O
# ────────────────────────────────────────────────────────────────────────────

def load_dataframe(data_path: Path, time_col: str) -> pd.DataFrame:
    """Load and sort a CSV file by the time column.

    Parameters
    ----------
    data_path : Path
        Full path to the CSV file.
    time_col : str
        Name of the datetime column.

    Returns
    -------
    df : pd.DataFrame
        Loaded DataFrame sorted by *time_col* with a reset integer index.
    """
    df = pd.read_csv(data_path)
    df[time_col] = pd.to_datetime(df[time_col])
    return df.sort_values(time_col).reset_index(drop=True)


def reshape_te_results(raw: dict, target_var: str) -> dict:
    """Convert engine output to a plottable nested dict.

    The engine returns ``{target: {input_to_target: ndarray}}``.
    This function strips the ``_to_target`` suffix from each key and
    converts arrays to plain Python lists for JSON compatibility.

    Parameters
    ----------
    raw : dict
        Raw output of ``TECalculator.run_partial_analysis``.
    target_var : str
        The target variable that was analyzed.

    Returns
    -------
    te_matrix : dict
        ``{target_var: {input_var_name: [float, ...]}}``
    """
    te_matrix: dict = {}
    if target_var in raw:
        te_matrix[target_var] = {}
        for combo_name, te_values in raw[target_var].items():
            input_name = combo_name.split("_to_")[0]
            te_matrix[target_var][input_name] = np.asarray(te_values).tolist()
    return te_matrix


# ────────────────────────────────────────────────────────────────────────────
# Array utilities
# ────────────────────────────────────────────────────────────────────────────

def interp_to_length(values: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly interpolate *values* to *target_length*.

    Parameters
    ----------
    values : np.ndarray
        Source array with at least one element.
    target_length : int
        Desired output length.

    Returns
    -------
    result : np.ndarray
        Interpolated array of length *target_length*.
    """
    if len(values) == target_length:
        return values
    if len(values) < 2:
        fill_val = values[0] if len(values) > 0 else 0.0
        return np.full(target_length, fill_val)
    x_old = np.linspace(0, 1, len(values))
    x_new = np.linspace(0, 1, target_length)
    f = scipy_interp.interp1d(x_old, values, kind="linear", fill_value="extrapolate")
    return f(x_new)


# ────────────────────────────────────────────────────────────────────────────
# Descriptive label builders
# ────────────────────────────────────────────────────────────────────────────

def infer_file_info(file_name: str) -> str:
    """Return a human-readable description based on filename frequency hints.

    Parameters
    ----------
    file_name : str
        Filename (not a path) of the source CSV.

    Returns
    -------
    info : str
    """
    name_lower = file_name.lower()
    if "12h" in name_lower:
        return "12-hourly data (fastest calculation)"
    if "6h" in name_lower:
        return "6-hourly data"
    if "4h" in name_lower:
        return "4-hourly data"
    if "1h" in name_lower:
        return "Hourly data (slowest calculation)"
    return "Atmospheric time series data"


def build_tau_description(tau: int, h: int) -> str:
    """Build the τ info label string shown in the configuration panel.

    Parameters
    ----------
    tau : int
        Time lag value (steps of the data interval).
    h : int
        History length (number of past target values conditioning the estimate).

    Returns
    -------
    description : str
        Human-readable TE formula description.
    """
    if h == 0:
        if tau == 0:
            return "τ=0: MI (Y_t; X_t) - no conditioning"
        return f"τ={tau}: MI (Y_t; X_{{t-{tau}}}) - no conditioning"
    if h == 1:
        cond_str = "Y_{t-1}"
    elif h <= 5:
        cond_str = ", ".join(f"Y_{{t-{j}}}" for j in range(1, h + 1))
    else:
        cond_str = f"Y_{{t-1}}, ..., Y_{{t-{h}}}"
    if tau == 0:
        return f"τ=0: Instantaneous TE (Y_t; X_t | {cond_str})"
    return f"τ={tau}: {tau}-step lag TE (Y_t; X_{{t-{tau}}} | {cond_str})"


def build_history_description(h: int) -> str:
    """Build the history-length info label string.

    Parameters
    ----------
    h : int
        History length.

    Returns
    -------
    description : str
    """
    if h == 0:
        return "h=0: No conditioning on target past (mutual information)"
    if h == 1:
        return "h=1: Condition on Y_{t-1} (traditional TE)"
    if h <= 5:
        cond = ", ".join(f"Y_{{t-{j}}}" for j in range(1, h + 1))
        return f"h={h}: Condition on ({cond})"
    return f"h={h}: Condition on (Y_{{t-1}}, ..., Y_{{t-{h}}})"


def build_workers_info(n: int, n_max: int) -> str:
    """Build the parallel worker count info label text.

    Parameters
    ----------
    n : int
        Chosen worker count (already clamped to ``[1, n_max]``).
    n_max : int
        Total available cores.

    Returns
    -------
    info : str
    """
    if n == 1:
        return "Single-threaded (slowest)"
    if n <= n_max - 2:
        return f"Recommended range (leaves {n_max - n} cores for system)"
    if n == n_max:
        return "Using all cores (may affect system responsiveness)"
    return f"Cores available: {n_max}"


def pack_jte_results(jte_raw: dict, params: dict) -> dict:
    """Serialize the raw JTE calculation output into a storable results dict.

    Parameters
    ----------
    jte_raw : dict
        Output of ``TECalculator.run_jte_rolling_analysis``.
    params : dict
        Calculation parameters dict from ``_combo_params``.

    Returns
    -------
    jte_results : dict
        Serializable dict with all fields needed for plotting and caching.
    """
    return {
        "jte_timeseries": jte_raw["jte_timeseries"],
        "individual_te": jte_raw["individual_te"],
        "sum_individual_te": jte_raw["sum_individual_te"],
        "synergy_timeseries": jte_raw["synergy_timeseries"],
        "synergy_stats": jte_raw["synergy_stats"],
        "timestamps": jte_raw["timestamps"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "metadata": {
            "target_var": params["target_var"],
            "input_vars": params["input_vars"],
            "window_days": params["window_days"],
            "tau": params["tau"],
            "tau_dict": params.get("tau_dict"),
            "history_length": params["history_length"],
        },
    }


def build_start_message(
    jte_mode: bool,
    diff_lag_enabled: bool,
    selected_inputs: List[str],
    target: str,
    tau_dict: Optional[Dict[str, int]],
) -> str:
    """Build the progress status message shown when a calculation begins.

    Parameters
    ----------
    jte_mode : bool
        Whether Joint TE is being calculated.
    diff_lag_enabled : bool
        Whether differential lagging is active.
    selected_inputs : list of str
        Selected input variable names.
    target : str
        Target variable name.
    tau_dict : dict or None
        Per-variable tau map (used for the label when ``diff_lag_enabled``).

    Returns
    -------
    message : str
    """
    n = len(selected_inputs)
    if jte_mode:
        return f"Calculating JTE for {n} inputs → {target}..."
    if diff_lag_enabled and tau_dict:
        tau_str = ", ".join(
            f"{v[:6]}:τ={t}" for v, t in list(tau_dict.items())[:3]
        )
        return f"Calculating {n} combinations with differential lags ({tau_str}...)..."
    return f"Calculating {n} combinations for {target}..."
