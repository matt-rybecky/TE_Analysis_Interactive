"""Pure figure-building functions for Transfer Entropy results.

All functions accept a matplotlib ``Figure`` or ``Axes`` and data dictionaries;
none import tkinter. They return the mutated ``Figure`` so the caller can embed
it in any backend (screen, file, or both).

Notes
-----
These functions mirror the plotting behavior of ``TE_Main.create_stacked_plot``
and ``TE_Main.create_jte_plot`` exactly (same data paths, same synergy math).
Visual style is intentionally kept at the original interactive level (colors,
grid, tight_layout) rather than the publication B&W style, because this module
serves an interactive explorer, not a manuscript pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


# ────────────────────────────────────────────────────────────────────────────
# Public: standard TE stacked-area plot
# ────────────────────────────────────────────────────────────────────────────

def draw_te_stacked(
    fig: Figure,
    timestamps: pd.DatetimeIndex,
    te_matrix: Dict[str, Dict[str, List[float]]],
    target_var: str,
    input_vars: List[str],
    *,
    tau: int = 1,
    history_length: int = 1,
    window_days: int = 30,
    dataset_name: str = "",
    tau_dict: Optional[Dict[str, int]] = None,
    entropy_values: Optional[np.ndarray] = None,
) -> Figure:
    """Draw a stacked-area TE plot onto *fig*.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to draw on (cleared before drawing).
    timestamps : pd.DatetimeIndex
        Window-center timestamps, length N.
    te_matrix : dict
        Nested dict ``{target_var: {input_var: [float, ...]}}`` from the
        calculation thread.
    target_var : str
        Name of the target variable.
    input_vars : list of str
        Input variable names to include (self-loops already removed by caller).
    tau : int
        Global time lag used in the calculation (for title/label only).
    history_length : int
        History length h (for title only when h > 1).
    window_days : int
        Rolling window length in days (for title only).
    dataset_name : str
        Filename of the source CSV (for title only).
    tau_dict : dict or None
        Per-variable tau map; when provided, legend entries include
        ``(tau=K)`` annotations.
    entropy_values : np.ndarray or None
        If provided, H(Y) reference line is drawn.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The same *fig* object, now populated.
    """
    fig.clear()
    ax = fig.add_subplot(111)

    # PHYSICS STEP 1: gather per-input TE arrays, skipping missing entries.
    plot_data: Dict[str, np.ndarray] = {}
    target_bucket = te_matrix.get(target_var, {})
    for var in input_vars:
        if var == target_var:
            continue
        if var in target_bucket:
            plot_data[var] = np.asarray(target_bucket[var], dtype=float)

    if not plot_data:
        ax.text(
            0.5, 0.5,
            f"No TE data found.\nTarget: {target_var}\nInputs: {input_vars}",
            transform=ax.transAxes, ha="center", va="center", fontsize=14,
        )
        _format_date_axis(ax)
        fig.tight_layout()
        return fig

    # PHYSICS STEP 2: stack fills from zero upward.
    colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
    bottom = np.zeros(len(timestamps))
    for i, (var_name, te_vals) in enumerate(plot_data.items()):
        label = _make_label(var_name, tau_dict)
        ax.fill_between(
            timestamps, bottom, bottom + te_vals,
            label=label, color=colors[i], alpha=0.7,
        )
        bottom += te_vals

    # PHYSICS STEP 3: optional H(Y) reference line.
    if entropy_values is not None:
        ax.plot(
            timestamps, entropy_values,
            color="black", linestyle="--", linewidth=2.5,
            label=f"H({target_var})", zorder=10,
        )

    # Formatting.
    h_str = f", h={history_length}" if history_length > 1 else ""
    title = (
        f"Transfer Entropy Contributions to {target_var}\n"
        f"({dataset_name}, {window_days}-day windows, τ={tau}{h_str})"
    )
    ax.set_title(title, fontsize=22)
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Transfer Entropy (bits)", fontsize=18)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=13)
    ax.grid(True, alpha=0.3)

    _format_date_axis(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────────
# Public: JTE plot
# ────────────────────────────────────────────────────────────────────────────

def draw_jte(
    fig: Figure,
    jte_data: dict,
    *,
    dataset_name: str = "",
    entropy_values: Optional[np.ndarray] = None,
) -> Figure:
    """Draw a Joint Transfer Entropy plot onto *fig*.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to draw on (cleared before drawing).
    jte_data : dict
        Result dict produced by ``calculate_jte_combination``, keyed:
        ``timestamps``, ``jte_timeseries``, ``sum_individual_te``,
        ``synergy_timeseries``, ``synergy_stats``, ``individual_te``,
        ``metadata``.
    dataset_name : str
        Filename of the source CSV (for title only).
    entropy_values : np.ndarray or None
        If provided, H(Y) reference line is drawn.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The same *fig* object, now populated.
    """
    fig.clear()
    ax = fig.add_subplot(111)

    timestamps = pd.to_datetime(jte_data.get("timestamps", []))
    jte_ts = np.asarray(jte_data.get("jte_timeseries", []), dtype=float)
    sum_ind = np.asarray(jte_data.get("sum_individual_te", []), dtype=float)
    synergy_stats = jte_data.get("synergy_stats", {})
    individual_te: Dict[str, list] = jte_data.get("individual_te", {})

    meta = jte_data.get("metadata", {})
    target_var = meta.get("target_var", "Unknown")
    input_vars: List[str] = meta.get("input_vars", [])
    window_days: int = meta.get("window_days", 30)
    tau: int = meta.get("tau", 1)
    history_length: int = meta.get("history_length", 1)
    tau_dict: Optional[Dict[str, int]] = meta.get("tau_dict")

    if len(timestamps) == 0 or len(jte_ts) == 0:
        ax.text(
            0.5, 0.5, "No JTE data available.",
            transform=ax.transAxes, ha="center", va="center", fontsize=16,
        )
        fig.tight_layout()
        return fig

    # PHYSICS STEP 1: stacked individual TEs as background reference.
    n_inputs = max(len(individual_te), 1)
    colors = plt.cm.Set2(np.linspace(0, 1, n_inputs))
    bottom = np.zeros(len(timestamps))
    for i, (var_name, te_vals) in enumerate(individual_te.items()):
        te_arr = np.asarray(te_vals, dtype=float)
        label = f"TE: {_make_label(var_name, tau_dict)}"
        ax.plot(timestamps, bottom + te_arr,
                linestyle="--", linewidth=1, color=colors[i], alpha=0.6)
        ax.fill_between(timestamps, bottom, bottom + te_arr,
                        color=colors[i], alpha=0.15, label=label)
        bottom += te_arr

    # PHYSICS STEP 2: sum-of-individual reference line.
    ax.plot(timestamps, sum_ind,
            linestyle=":", linewidth=1.5, color="gray", alpha=0.8,
            label="Σ Individual TE")

    # PHYSICS STEP 3: prominent JTE line.
    ax.plot(timestamps, jte_ts,
            linestyle="-", linewidth=2.5, color="#1f77b4",
            label="Joint TE", zorder=10)

    all_values = np.concatenate([jte_ts, sum_ind])

    # PHYSICS STEP 4: optional H(Y) reference line.
    if entropy_values is not None:
        ax.plot(timestamps, entropy_values,
                color="black", linestyle="--", linewidth=2.5,
                label=f"H({target_var})", zorder=11)
        all_values = np.concatenate([all_values, entropy_values])

    y_max = np.max(all_values) * 1.15
    y_min = min(0.0, np.min(all_values) * 1.1)
    ax.set_ylim(y_min, y_max)

    # Title and labels.
    _apply_jte_labels(ax, target_var, input_vars, dataset_name,
                      window_days, tau, history_length)

    # Synergy annotation panel.
    _draw_synergy_panel(ax, synergy_stats)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=13, framealpha=0.9)

    _format_date_axis(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    return fig


# ────────────────────────────────────────────────────────────────────────────
# Private helpers
# ────────────────────────────────────────────────────────────────────────────

def _make_label(var_name: str, tau_dict: Optional[Dict[str, int]]) -> str:
    """Return a legend label, appending tau annotation when differential lags are active."""
    if tau_dict and var_name in tau_dict:
        return f"{var_name} (τ={tau_dict[var_name]})"
    return var_name


def _format_date_axis(ax) -> None:
    """Apply auto date locator and concise formatter to *ax*."""
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", rotation=30, labelsize=14)


def _apply_jte_labels(
    ax,
    target_var: str,
    input_vars: List[str],
    dataset_name: str,
    window_days: int,
    tau: int,
    history_length: int,
) -> None:
    """Set title and axis labels for a JTE plot."""
    inputs_str = ", ".join(input_vars[:3])
    if len(input_vars) > 3:
        inputs_str += f", +{len(input_vars) - 3} more"
    h_str = f", h={history_length}" if history_length > 1 else ""
    title = (
        f"Joint Transfer Entropy: ({inputs_str}) → {target_var}\n"
        f"({dataset_name}, {window_days}-day windows, τ={tau}{h_str})"
    )
    ax.set_title(title, fontsize=22)
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Transfer Entropy (bits)", fontsize=18)


def _draw_synergy_panel(ax, synergy_stats: dict) -> None:
    """Draw the synergy analysis text box to the right of *ax*."""
    mean_jte = synergy_stats.get("mean_jte", 0.0)
    mean_sum_te = synergy_stats.get("mean_sum_individual_te", 0.0)
    mean_synergy = synergy_stats.get("mean_synergy", 0.0)
    synergy_frac = synergy_stats.get("synergy_fraction", 0.0) * 100.0

    if mean_synergy > 0.001:
        interpretation = "SYNERGISTIC\nInputs provide\ncomplementary info"
        interp_color = "#2ca02c"
    elif mean_synergy < -0.001:
        interpretation = "REDUNDANT\nInputs share\noverlapping info"
        interp_color = "#d62728"
    else:
        interpretation = "INDEPENDENT\nInputs contribute\nindependently"
        interp_color = "#7f7f7f"

    synergy_text = (
        f"━━━ Synergy Analysis ━━━\n"
        f"Mean JTE:      {mean_jte:.4f} bits\n"
        f"Mean Σ Ind TE: {mean_sum_te:.4f} bits\n"
        f"Mean Synergy:  {mean_synergy:+.4f} bits\n"
        f"Synergy %:     {synergy_frac:+.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"{interpretation}"
    )
    props = {"boxstyle": "round,pad=0.5", "facecolor": "wheat", "alpha": 0.8}
    text_color = (
        interp_color
        if interpretation.startswith(("SYNERGISTIC", "REDUNDANT"))
        else "black"
    )
    ax.text(
        1.02, 0.5, synergy_text,
        transform=ax.transAxes, fontsize=12,
        verticalalignment="center", fontfamily="monospace",
        bbox=props, color=text_color,
    )
