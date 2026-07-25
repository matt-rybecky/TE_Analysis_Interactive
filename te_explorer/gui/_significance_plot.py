"""Rendering and export helpers for the IAAFT significance tab.

All functions here are pure or near-pure: they take explicit arguments and
have no reference to tkinter or to the SignificanceTab class itself.  The tab
calls them after a successful analysis run.

Separation rationale: plot rendering + export together exceeded the 60-line
method limit and pushed ``significance_tab.py`` past 500 lines.  Extracting
them here keeps both files inside their limits while preserving the original
math and visual structure.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import te_explorer.config  # noqa: F401 - side effect: injects core/ into sys.path
from TE_Surrogate import SurrogateAnalyzer


# --------------------------------------------------------------------------
# Figure population
# --------------------------------------------------------------------------

def render_significance_plot(fig: Figure, sig_results: Dict) -> None:
    """Clear ``fig`` and draw the IAAFT significance result.

    Routes to :func:`render_te_subplots` (per-pair TE) or
    :func:`render_jte_panel` (joint TE) depending on ``sig_results`` metadata.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to draw into (will be cleared first).
    sig_results : dict
        Results dict from ``SurrogateAnalyzer.calculate_surrogate_*``.
    """
    fig.clear()
    metadata = sig_results.get('metadata', {})
    method = metadata.get('method', '')
    is_jte = 'jte' in method

    if is_jte:
        render_jte_panel(fig, sig_results, metadata)
    else:
        render_te_subplots(fig, sig_results, metadata)


def render_te_subplots(fig: Figure, sig_results: Dict,
                        metadata: Dict) -> None:
    """Populate ``fig`` with per-pair TE + significance band subplots.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure (already cleared).
    sig_results : dict
        Full results dict from ``calculate_surrogate_confidence_intervals``.
    metadata : dict
        Top-level ``sig_results['metadata']`` block.
    """
    analyzer = SurrogateAnalyzer()
    ci_series = analyzer.extract_ci_time_series(sig_results)

    if not ci_series:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No significance data available",
                ha='center', va='center', transform=ax.transAxes)
        return

    n = len(ci_series)
    n_cols = min(2, n)
    n_rows = int(np.ceil(n / n_cols))
    suptitle = build_suptitle(metadata, is_jte=False)

    for i, (key, df) in enumerate(ci_series.items()):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        input_var = key.split('_to_')[0] if '_to_' in key else None
        rel_name = (df['relationship'].iloc[0]
                    if 'relationship' in df.columns else key)
        subplot_title = build_subplot_title(rel_name, metadata, input_var)
        draw_significance_axes(ax, df, is_jte=False, title=subplot_title)

    fig.suptitle(suptitle, fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])


def render_jte_panel(fig: Figure, sig_results: Dict,
                      metadata: Dict) -> None:
    """Populate ``fig`` with a single JTE + significance band panel.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure (already cleared).
    sig_results : dict
        Full results dict from ``calculate_jte_surrogate_confidence_intervals``.
    metadata : dict
        Top-level ``sig_results['metadata']`` block.
    """
    analyzer = SurrogateAnalyzer()
    df = analyzer.extract_jte_ci_time_series(sig_results)
    df = df.rename(columns={'original_jte': 'original_te'})
    combo = metadata.get('combination_key', 'unknown')
    target = metadata.get('target_var', 'unknown')
    df['relationship'] = f"JTE({combo} → {target})"

    ax = fig.add_subplot(111)
    suptitle = build_suptitle(metadata, is_jte=True)
    draw_significance_axes(ax, df, is_jte=True, title=suptitle)
    fig.tight_layout()


def draw_significance_axes(ax: Any, df: pd.DataFrame, *,
                            is_jte: bool, title: str) -> None:
    """Draw TE curve and significance band onto a single axes.

    The significance band fills from zero *up to* the 95th-percentile
    threshold line.  The shaded region is the "not significant" territory,
    so the data line reads against the band edge (visualization standard,
    amended 2026-07-09).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    df : pd.DataFrame
        DataFrame with columns ``date``, ``original_te``, ``threshold_95``.
    is_jte : bool
        True when displaying joint TE (affects y-axis label).
    title : str
        Axes title.
    """
    has_dates = 'date' in df.columns and df['date'].notna().any()
    x = df['date'] if has_dates else np.arange(len(df))

    # PHYSICS STEP 2: draw TE/JTE time series
    te_label = "Joint Transfer Entropy" if is_jte else "Transfer Entropy"
    ax.plot(x, df['original_te'], color='black', linewidth=1.4,
            label=te_label, zorder=3)

    # PHYSICS STEP 3: draw significance band (shaded below threshold)
    if 'threshold_95' in df.columns:
        threshold = df['threshold_95']
        ax.plot(x, threshold, color='black', linewidth=0.9,
                linestyle='--', label='IAAFT 95th percentile', zorder=2)
        ax.fill_between(x, 0, threshold,
                        color='gray', alpha=0.20,
                        label='Not significant (below band)', zorder=1)

    ax.set_title(title, fontsize=10, pad=6)
    ax.set_ylabel("JTE (bits)" if is_jte else "TE (bits)", fontsize=10)
    ax.set_xlabel("Date" if has_dates else "Time Points", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc='best')

    if has_dates:
        format_date_axis(ax)


def format_date_axis(ax: Any) -> None:
    """Apply auto date locator and concise formatter to the x-axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format.
    """
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis='x', rotation=30, labelsize=8)


# --------------------------------------------------------------------------
# Title builders
# --------------------------------------------------------------------------

def build_suptitle(metadata: Dict, *, is_jte: bool) -> str:
    """Build a figure-level descriptive title string.

    Parameters
    ----------
    metadata : dict
        Top-level metadata from the surrogate results dict.
    is_jte : bool
        True when building a JTE-mode title.

    Returns
    -------
    title : str
    """
    te_type = "Joint Transfer Entropy" if is_jte else "Transfer Entropy"
    data_file = metadata.get('data_file', 'Unknown')
    window_days = metadata.get('window_days', '?')
    surr_type = metadata.get('surrogate_type', 'iaaft')

    tau_dict = metadata.get('tau_dict')
    tau_list = metadata.get('tau_list')
    if tau_dict:
        parts = [f"{v}:τ={t}" for v, t in tau_dict.items()]
        lag_str = "Diff. lags [" + ", ".join(parts) + "]"
    elif tau_list:
        lag_str = f"τ_list={tau_list}"
    else:
        lag_str = f"τ={metadata.get('tau', '?')}"

    h = metadata.get('history_length', 1)
    if h and h > 1:
        lag_str += f", h={h}"

    return (f"{te_type} - Surrogate ({surr_type})\n"
            f"Data: {data_file} | Window: {window_days}d | {lag_str}")


def build_subplot_title(rel_name: str, metadata: Dict,
                         input_var: Optional[str]) -> str:
    """Build a per-subplot title annotated with the effective tau.

    Parameters
    ----------
    rel_name : str
        Human-readable relationship label (e.g. ``'tempC -> dD'``).
    metadata : dict
        Top-level metadata; checked for ``tau_dict`` and ``tau``.
    input_var : str or None
        Source variable name for tau_dict lookup.

    Returns
    -------
    title : str
    """
    tau_dict = metadata.get('tau_dict')
    if tau_dict and input_var and input_var in tau_dict:
        return f"{rel_name}  (τ={tau_dict[input_var]})"
    return f"{rel_name}  (τ={metadata.get('tau', '?')})"


# --------------------------------------------------------------------------
# CSV export
# --------------------------------------------------------------------------

def build_export_dataframe(sig_results: Dict, *, is_jte: bool) -> pd.DataFrame:
    """Assemble all pairs into one DataFrame for CSV export.

    Parameters
    ----------
    sig_results : dict
        Full surrogate results dict.
    is_jte : bool
        True when results are from ``calculate_jte_surrogate_*``.

    Returns
    -------
    df : pd.DataFrame
        Combined time series, all pairs concatenated.
    """
    analyzer = SurrogateAnalyzer()
    if is_jte:
        df = analyzer.extract_jte_ci_time_series(sig_results)
        df = df.rename(columns={'original_jte': 'original_te'})
        meta = sig_results.get('metadata', {})
        combo = meta.get('combination_key', 'unknown')
        target = meta.get('target_var', 'unknown')
        df['relationship'] = f"JTE({combo} -> {target})"
        df['relationship_key'] = combo
        return df

    ci_series = analyzer.extract_ci_time_series(sig_results)
    frames: List[pd.DataFrame] = []
    for key, pair_df in ci_series.items():
        pair_df = pair_df.copy()
        pair_df['relationship_key'] = key
        frames.append(pair_df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_significance_plot(fig: Figure, sig_results: Dict,
                            output_dir: Path) -> str:
    """Save the significance figure to PNG and PDF.

    Fixed canvas size is preserved: ``bbox_inches`` is not set to ``'tight'``
    (visualization standard).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Rendered figure to save.
    sig_results : dict
        Results dict (used to derive output filename).
    output_dir : pathlib.Path
        Directory to write files into (created if absent).

    Returns
    -------
    stem : str
        Base filename stem (without extension) for both output files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = sig_results.get('metadata', {})
    target = meta.get('target_var', 'unknown')
    stem = f"iaaft_significance_{target}_{stamp}"
    for ext in ('png', 'pdf'):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300)
    return stem


def export_significance_csv(sig_results: Dict, output_dir: Path) -> Path:
    """Write significance time series to a CSV file.

    Parameters
    ----------
    sig_results : dict
        Full surrogate results dict.
    output_dir : pathlib.Path
        Directory to write into (created if absent).

    Returns
    -------
    out_path : pathlib.Path
        Full path of the written CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = sig_results.get('metadata', {})
    method = meta.get('method', '')
    is_jte = 'jte' in method
    target = meta.get('target_var', 'unknown')
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "iaaft_jte_significance" if is_jte else "iaaft_significance"
    out_path = output_dir / f"{prefix}_{target}_{stamp}.csv"

    df = build_export_dataframe(sig_results, is_jte=is_jte)
    df.to_csv(out_path, index=False)
    return out_path
