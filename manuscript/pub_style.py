"""Definitive publication figure style for the manuscript and SI.

Implements the style guide in ``Manuscript/docs/ARTIFACT_HANDOFF.md``:
pure black and white, shape-first series differentiation, print-true
dimensions and font sizes. Every figure script imports this module and
applies no style overrides of its own.

Usage
-----
    from pub_style import (setup_style, series_style, context_style,
                           shade_periods, draw_envelope, date_axis,
                           panel_label, save_figure)

    figsize = setup_style(width='column')          # or width='full'
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, **series_style(0, markevery=28),
            label='Temp (t=1) + LW-up (t=0)')
"""

from math import ceil
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Print-true dimensions (inches). JGR: Atmospheres column and full width.
COLUMN_WIDTH = 3.7
FULL_WIDTH = 7.5
DEFAULT_HEIGHT = {'column': 2.8, 'full': 3.4}

# The only sanctioned grays. Data is black; gray is for context and bands.
CONTEXT_GRAY = '0.45'   # secondary/context series (e.g. external obs)
BAND_GRAY = '0.85'      # surrogate/significance envelopes
SHADE_GRAY = '0.93'     # alternating period shading
HATCH = '///'           # for overlapping filled regions

# Shape-first series differentiation: line style + marker + marker fill.
# Grayscale is deliberately absent: four series maximum, all pure black.
_SERIES = [
    dict(linestyle='-',  marker='o', markerfacecolor='black'),
    dict(linestyle='--', marker='s', markerfacecolor='white'),
    dict(linestyle='-.', marker='^', markerfacecolor='black'),
    dict(linestyle=':',  marker='D', markerfacecolor='white'),
]


def setup_style(width: str = 'full', height: float | None = None) -> tuple:
    """Apply the definitive rcParams and return the print-true figsize.

    Parameters
    ----------
    width : str
        'column' (3.7 in) or 'full' (7.5 in).
    height : float, optional
        Figure height in inches; defaults per width class.

    Returns
    -------
    tuple
        (width_in, height_in) to pass as ``figsize``.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'axes.labelsize': 10,
        'axes.titlesize': 10,      # titles are banned; size set defensively
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'lines.linewidth': 1.3,
        'lines.markersize': 4.5,
        'lines.markeredgewidth': 0.8,
        'axes.linewidth': 0.8,
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'axes.prop_cycle': cycler(color=['black']),
        'axes.grid': False,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'savefig.dpi': 300,
        # No tight bbox: it resizes the canvas to content, so the journal
        # rescales the figure and real font sizes drift. Fixed canvas +
        # constrained layout keeps every figure print-true.
        'figure.constrained_layout.use': True,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.dpi': 120,
    })
    w = COLUMN_WIDTH if width == 'column' else FULL_WIDTH
    h = height if height is not None else DEFAULT_HEIGHT[
        'column' if width == 'column' else 'full']
    return (w, h)


def series_style(i: int, markevery: int | None = None,
                 markers: bool = True) -> dict:
    """Plot kwargs for data series ``i`` (0-3), shape-first, pure black.

    ``markers=False`` differentiates by line style alone (solid, dashed,
    dash-dot, dotted) — the author-ruled form (2026-07-09) for the
    regime/selection figure family, where marker points clutter the
    high-frequency TE series.

    Raises
    ------
    ValueError
        If a fifth series is requested: the style guide caps axes at four
        data series; split or prune the figure instead.
    """
    if i >= len(_SERIES):
        raise ValueError('Style guide: at most 4 data series per axes; '
                         'split or prune the figure.')
    if not markers:
        return dict(linestyle=_SERIES[i]['linestyle'], color='black')
    kw = dict(_SERIES[i], color='black', markeredgecolor='black')
    if markevery is not None:
        kw['markevery'] = markevery
    return kw


def context_style() -> dict:
    """Kwargs for a context/secondary series: the one sanctioned gray."""
    return dict(color=CONTEXT_GRAY, linewidth=0.9, linestyle='-',
                zorder=1)


def shade_periods(ax, boundaries, alternate: bool = True) -> None:
    """Alternating very-light-gray spans between consecutive boundaries."""
    for j in range(len(boundaries) - 1):
        if alternate and j % 2 == 0:
            continue
        ax.axvspan(boundaries[j], boundaries[j + 1],
                   color=SHADE_GRAY, zorder=0, linewidth=0)


def draw_envelope(ax, x, lower, upper, label=None) -> None:
    """Light-gray band (e.g. surrogate envelope); never darker than data."""
    ax.fill_between(x, lower, upper, color=BAND_GRAY, linewidth=0,
                    zorder=1, label=label)


def date_axis(ax, minor_days: int | None = None) -> None:
    """Month-labelled, collision-free dated x-axis."""
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    if minor_days:
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=minor_days))


def light_grid(ax) -> None:
    """Optional grid — a per-figure author choice (2026-07-08), default off.

    When a figure opts in, the grid is light gray, thin, and beneath the
    data, so it never competes with series or bands.
    """
    ax.grid(True, color=SHADE_GRAY, linewidth=0.5)
    ax.set_axisbelow(True)


def panel_label(ax, text: str) -> None:
    """Small-multiple / multi-panel identifier, e.g. '(a)'. Not a title."""
    ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=8,
            va='top', ha='left')


def save_legend(entries: list, stem, ncol: int = 2) -> list:
    """Standalone legend-key artifact (author ruling 2026-07-09).

    Legends live OUTSIDE the figure: the plot field carries no legend
    (clear, readable, no possible overlap), and each figure ships with a
    companion ``<stem>_legend.{png,pdf}`` — a table-like key strip with
    the exact line samples at print-true sizes — that LaTeX stacks with
    the figure at compile time.

    Parameters
    ----------
    entries : list of (label, style_kwargs)
        One row per series, in display order; ``style_kwargs`` are the
        exact plot kwargs (color, linestyle, linewidth, markers). A
        style dict containing ``'patch': True`` renders as a filled
        swatch (shaded bands) using its remaining keys.
    stem : path-like
        Written as ``<stem>.png/.pdf`` (pass e.g. ``fig_x_legend``).
    ncol : int
        Key columns; rows follow.
    """
    handles = [Patch(**{k: v for k, v in kw.items() if k != 'patch'})
               if kw.get('patch') else Line2D([0], [1], **kw)
               for _, kw in entries]
    labels = [label for label, _ in entries]
    n_rows = ceil(len(entries) / ncol)
    fig = plt.figure(figsize=(FULL_WIDTH, 0.18 * n_rows + 0.14))
    fig.legend(handles, labels, loc='center', ncol=ncol, frameon=False,
               handlelength=2.6, columnspacing=1.6)
    return save_figure(fig, stem)


def save_figure(fig, stem) -> list:
    """Save PNG + PDF at 300 DPI and close. Returns the paths written."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ('png', 'pdf'):
        p = stem.with_suffix(f'.{ext}')
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths
