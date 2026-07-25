#!/usr/bin/env python3
"""
pub_selected_figures.py — Staged selection library for the results section.

The figure form of record for the results section (author, 2026-07-09):
TE series over the shaded changepoint transition ranges, LINES ONLY
(series differentiated by line style, no markers), pub_style B&W,
smoothed curves (the 12-day rolling-median baseline component), taus
attached to every legend name. Targets dD and d-excess only.

COMPOSITION RULE (author, 2026-07-09, binding): a plot showing a
combination shows either ALL of that combination's input components or
NONE of them (composites compared to composites). Never a partial member
set. Maximum 4 lines per plot (a triple + its 3 members = exactly 4).
The earlier hand-curated SELECTIONS violated this rule (partial member
sets) and are retired; keepers re-enter through the staged process.

CROSS-TARGET KEY FIGURES (author picks, 2026-07-09, guaranteed for the
paper): both isotope targets on one axis, labels ``<entity> -> <target>``.
  - ``cross_lwdn_alltaus``: LW-down at all taus for dD AND d-excess —
    LW-down(tau=1) -> d-excess is a high-% episodic departure while
    LW-down explains nothing for dD; the contrast is the argument (KEY
    FIGURE). Physical framing (author, 2026-07-09): the longwave
    contrast is a real, significant finding — surface temperature
    (LW-up) is isotope-indistinguishable, but secondary cloud/sky/
    greenhouse effects (LW-down, lagged) are far more significant in
    the kinetic realm, rising well above baseline for d-excess.
  - ``cross_lwup_t0``: LW-up(tau=0) -> dD and -> d-excess together —
    near-identical signals, the foil to the LW-down contrast.
KEY-FIGURE FORM (author rulings 2026-07-09, latest supersedes):
  - NO SHADING of any kind (changepoint bounds and burst spans both
    removed; "the plots speak for themselves").
  - LW-down figure carries the FULL lag sweep (the definitive-time-lag
    argument): tau 0/1/2 for both isotopes plus tau=3 for d-excess
    (author, 2026-07-09, completeness) — 7 data lines, explicitly
    author-authorized past the 4-line cap. Styling (all-black,
    uniform-weight ruling, 2026-07-09; both gray-for-dD and bold/
    non-bold retired): every line black at one weight, differentiated
    by DASH PATTERN alone. The entity list leads with the main findings,
    which take the cleanest patterns: LW-down(t1) and (t2) -> d-excess
    are solid and dashed; the rest follow the ordered palette (dotted,
    dash-dot, dot-dot-dash, ...).
  - IAAFT SIGNIFICANCE (presented in the final paper): per-window 95th
    percentile from the Stage 2 fixed-entity runs (``pub_stage2.py``)
    of EVERY input shown, combined as the per-window MAXIMUM, smoothed
    like the data (12-day median), and presented as a SHADED BAND below
    the level (author ruling 2026-07-09; the earlier dash-dot line form
    is retired). Series absent from the frozen sweep (LW-down tau=2/3:
    the local-state group swept tau {0,1} only) come from the same
    Stage 2 CSVs, smoothed identically. Until the corresponding Stage 2
    run lands, missing series/levels are skipped with a warning.

STAGED WORKSHOP (author, 2026-07-09): the library is built and reviewed
in stages —
  1. ``singles``: every single-variable input to each target, complete
     coverage (every base at every allowed lag), grouped physically into
     panels of at most 4 lines. The author picks the singles of record.
  2. ``pairs``:   2-variable combinations (each pair + both members).
  3. ``triples``: 3-variable combinations (each triple + all 3 members).
Stages 2-3 are built after the stage-1 picks. ``selection_index_<target>
.csv`` lists one row per drawn entity (panel key, label, winter mean and
peak % of H(target)) as the pull sheet.

Pure post-processing; run pub_driver_series and pub_periods first.

Usage:
    python3 pub_selected_figures.py                # current stage: singles
    python3 pub_selected_figures.py --smooth 4

Author: Matthew Rybecky
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from pub_config import PublicationConfig, TAU_DELIM, base_of
from pub_driver_series import entity_label
from pub_labels import target_label
from pub_period_attribution import load_layers
from pub_style import (BAND_GRAY, date_axis, save_figure, save_legend,
                       series_style, setup_style, shade_periods)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FIG_HEIGHT = 3.25       # inches; one panel + above-axes legend rows

# Smoothed curves only (author ruling 2026-07-09): the 12-day median
# baseline component. Raw pct_jte still feeds the index statistics.
VARIANTS = {'': ('baseline', False)}

# Targets in the selection library (H2O_ppm removed, author 2026-07-09).
LIBRARY_TARGETS = ('d_excess', 'dD')

MAX_LINES = 4           # composition rule: hard cap per plot

# Stage 1: physical grouping of the single-variable inputs. Each group's
# bases expand to every allowed lag from pub_config; groups are sized so
# no panel exceeds MAX_LINES (asserted at build time).
# Confirmed cross-target key figures (author picks 2026-07-09).
# Entities are (target, candidate) pairs; lag-contrast pairs are ordered
# adjacent so the 2-column legend rows pair them.
# Cross figures list entities in STYLE-PRIORITY order (author ruling
# 2026-07-09): the main findings lead, so they take the cleanest dash
# patterns. LW-down: LW-down(t1) and (t2) -> d-excess are the headline
# pair (solid, dashed); the rest follow in the ordered palette.
CROSS_SELECTIONS = [
    # 'smooth': True keeps the 12-day median for THIS figure (author,
    # 2026-07-19): seven raw lines are unreadable; the caption and prose
    # disclose the smoothing, and reported statistics stay per-window.
    {'key': 'cross_lwdn_alltaus', 'smooth': True,
     'entities': [('d_excess', 'rad_lw_down__t1'),   # headline 1
                  ('d_excess', 'rad_lw_down__t2'),   # headline 2
                  ('d_excess', 'rad_lw_down__t0'),
                  ('d_excess', 'rad_lw_down__t3'),
                  ('dD', 'rad_lw_down__t1'),
                  ('dD', 'rad_lw_down__t0'),
                  ('dD', 'rad_lw_down__t2')]},
    {'key': 'cross_lwup_t0',
     'entities': [('d_excess', 'rad_lw_up__t0'), ('dD', 'rad_lw_up__t0')]},
    # Basic-meteorology comparisons (author 2026-07-10): the driver at
    # tau=0 ONLY for both isotopes, with the IAAFT band. The lagged
    # values are not explanatory here (Stage 2 verified: temp/rh tau=1
    # sit largely below the surrogate threshold) and are dropped; the
    # story is in the tau=0 variables. Protagonist target leads the style
    # priority: temperature -> dD (equilibrium anchor); RH -> d-excess
    # (kinetic). Two lines each, parallel to cross_lwup_t0.
    {'key': 'cross_temp_t0',
     'entities': [('d_excess', 'met_temp__t0'), ('dD', 'met_temp__t0')]},
    {'key': 'cross_rh_t0',
     'entities': [('d_excess', 'met_rh__t0'), ('dD', 'met_rh__t0')]},
    # H2O flux at 3 m, tau=0, both isotopes (author 2026-07-10: the one
    # flux single worth presenting — THE sublimation flux, closest EC
    # height to the surface; 10/20 m are attenuated/redundant, GPH and CO2
    # are combination-only). Screen: d-excess bursts to 21% at the
    # mid-January sublimation episode (Jan 23), alongside the LW-down
    # story. tau=0 only, parallel to the other final singles.
    {'key': 'cross_h2o3m_t0',
     'entities': [('d_excess', 'flux_3m_h2o__t0'),
                  ('dD', 'flux_3m_h2o__t0')]},
]
BASELINE_WINDOWS = 48   # 12-day median smoothing (pub_driver_series default)
# All-black, uniform-weight ruling (author, 2026-07-09): series are
# differentiated by dash pattern ALONE (no bold/non-bold). Ordered from
# cleanest to most complex; the entity list leads with the main findings.
STYLE_PALETTE = [
    (0, ()),                          # solid       — headline 1
    (0, (5, 2)),                      # dashed      — headline 2
    (0, (1, 2)),                      # dotted
    (0, (5, 2, 1, 2)),                # dash-dot
    (0, (1, 2, 1, 2, 5, 2)),          # dot-dot-dash
    (0, (5, 2, 5, 2, 1, 2)),          # dash-dash-dot
    (0, (1, 2, 1, 2, 1, 2, 5, 2)),    # dot-dot-dot-dash
]
CROSS_LINEWIDTH = 1.3
# Consistency rule (author 2026-07-10): on cross-target comparisons where
# each target appears once, the line style encodes the TARGET, kept fixed
# across figures so it never swaps — d-excess solid, dD dotted. (Many-line
# figures like the LW-down lag sweep fall back to the priority palette.)
TARGET_STYLE = {'d_excess': (0, ()), 'dD': (0, (1, 2))}

SINGLE_GROUPS = [
    ('temp_rh', ['met_temp', 'met_rh']),
    ('lw', ['rad_lw_down', 'rad_lw_up']),
    ('sw_net', ['rad_sw_down', 'rad_net']),
    ('wspd', ['met_wspd']),
    ('wdir', ['met_wdir']),
    ('h2o_flux_3m_10m', ['flux_3m_h2o', 'flux_10m_h2o']),
    ('h2o_flux_20m', ['flux_20m_h2o']),
    ('co2_density_3m_10m', ['flux_3m_co2_density', 'flux_10m_co2_density']),
    ('co2_density_20m', ['flux_20m_co2_density']),
    ('gph225', ['isen_225_gph']),
    ('gph500', ['isen_500_gph']),
]


# ═════════════════════════════════════════════════════════════════════════
# Stage builders
# ═════════════════════════════════════════════════════════════════════════
def build_singles_selections(cfg: PublicationConfig,
                             target: str) -> List[Dict]:
    """Stage 1: complete single-variable coverage, grouped, <= 4 lines.

    Every ``base__tK`` candidate of the target appears in exactly one
    panel; grouping follows SINGLE_GROUPS and panel size is asserted
    against the composition rule's line cap.
    """
    candidates = cfg.candidates(target)
    by_base: Dict[str, List[str]] = {}
    for c in candidates:
        by_base.setdefault(base_of(c), []).append(c)

    selections: List[Dict] = []
    covered: List[str] = []
    for key, bases in SINGLE_GROUPS:
        entities = [c for b in bases for c in by_base.get(b, [])]
        if not entities:
            continue
        if len(entities) > MAX_LINES:
            raise ValueError(f"singles group '{key}' has {len(entities)} "
                             f"lines (> {MAX_LINES}); split the group")
        selections.append({'key': f'singles_{key}', 'source': 'singles',
                           'entities': entities})
        covered.extend(entities)
    missing = sorted(set(candidates) - set(covered))
    if missing:
        raise ValueError(f"{target}: singles coverage incomplete: {missing}")
    return selections


CHARACTER_TOP_N = 4   # lines per character panel (composition-rule cap)


def build_character_selections(layers: Dict) -> List[Dict]:
    """Curated single-driver panels by PERSISTENT vs EPISODIC character.

    From the period-attribution decomposition (author request 2026-07-10:
    show what persistently explains, alongside the episodic drivers):
      - persistent: the top singles by mean baseline (the 12-day median
        continuous component) — steady explanatory power across periods;
      - episodic: the top singles by peak episodic excess among those NOT
        already in the persistent set — burst-driven, one-period spikes.
    Both are lag-level singles; taus attached at draw time. <= 4 lines.
    """
    base, exc = layers['baseline'], layers['excess']
    singles = [c for c in base.columns if '+' not in c]
    persist = base[singles].mean().sort_values(ascending=False)
    persistent = list(persist.head(CHARACTER_TOP_N).index)
    remaining = [s for s in singles if s not in persistent]
    episodic = list(exc[remaining].max()
                    .sort_values(ascending=False).head(CHARACTER_TOP_N).index)
    return [
        {'key': 'persistent_singles', 'source': 'character',
         'entities': persistent},
        {'key': 'episodic_singles', 'source': 'character',
         'entities': episodic},
    ]


# ═════════════════════════════════════════════════════════════════════════
# Drawing
# ═════════════════════════════════════════════════════════════════════════
def period_boundaries(periods: pd.DataFrame) -> List[pd.Timestamp]:
    """Shading edges: every period start plus the final period end."""
    return [*periods['start'], periods['end'].iloc[-1]]


def draw_selection(target: str, frame: pd.DataFrame, periods: pd.DataFrame,
                   sel: Dict, out: Path) -> None:
    """One panel: selected entities on a shared axis, periods shaded.

    No in-field legend (author ruling 2026-07-09): the key ships as the
    companion ``<out>_legend`` artifact, stacked in by LaTeX.
    """
    figsize = setup_style('full', height=FIG_HEIGHT)
    fig, ax = plt.subplots(figsize=figsize)
    shade_periods(ax, period_boundaries(periods))
    entries = []
    for e in sel['entities']:
        if e not in frame.columns:
            logger.warning(f"{target}/{sel['key']}: missing entity {e}")
            continue
        style = series_style(len(entries), markers=False)
        ax.plot(frame.index, frame[e], **style)
        entries.append((entity_label(e), style))
    ax.set_ylabel('Joint TE (% of H(target))')
    ax.set_xlabel('Date (2022-2023)')
    ax.set_xlim(frame.index[0], frame.index[-1])
    date_axis(ax)
    for p in [*save_figure(fig, out),
              *save_legend(entries, Path(f'{out}_legend'))]:
        logger.info(f"wrote {p}")


def stage2_series(base: Path, target: str, entity: str,
                  column: str) -> pd.Series | None:
    """One column of an entity's Stage 2 CSV, or None if not yet run."""
    path = base / 'stage2' / target / f'stage2_{entity}.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=['timestamp']).set_index('timestamp')
    return df[column]


def smooth_median(series: pd.Series) -> pd.Series:
    """The presentation smoothing of record: 12-day centered median."""
    min_periods = max(3, BASELINE_WINDOWS // 3)
    return series.rolling(BASELINE_WINDOWS, center=True,
                          min_periods=min_periods).median()


def cross_curve(frames: Dict[str, pd.DataFrame], base: Path,
                analysis_end: pd.Timestamp, target: str,
                entity: str) -> pd.Series | None:
    """A cross-figure data curve: frozen-artifact baseline, else Stage 2.

    Entities inside the frozen tau sets come from
    ``driver_series_baseline.parquet``; entities outside them (e.g.
    LW-down tau=2) come from their Stage 2 CSV (``pct_h``), smoothed
    identically (12-day centered median) and winter-truncated.
    """
    frame = frames[target]
    if entity in frame.columns:
        return frame[entity]
    s = stage2_series(base, target, entity, 'pct_h')
    if s is None:
        return None
    return smooth_median(s[s.index < analysis_end])


def cross_significance(base: Path, analysis_end: pd.Timestamp,
                       sel: Dict, smooth: bool = True) -> pd.Series | None:
    """Per-window IAAFT 95% level: max over ALL shown inputs.

    Author ruling 2026-07-09: every input on the plot gets Stage 2
    surrogates; the plotted level is the per-window maximum of their
    95th percentiles (% of H(target)), smoothed like the data. None
    until every input's Stage 2 CSV exists (partial levels would
    understate the family maximum). ``smooth=False`` (author ruling
    2026-07-19, unsmoothed figures) returns the raw per-window level.
    """
    levels = []
    for target, e in sel['entities']:
        s = stage2_series(base, target, e, 'surr_p95_pct')
        if s is None:
            logger.warning(f"{sel['key']}: no Stage 2 CSV yet for "
                           f"{target}/{e}; significance level skipped")
            return None
        levels.append(s[s.index < analysis_end])
    combined = pd.concat(levels, axis=1).max(axis=1)
    return smooth_median(combined) if smooth else combined


def cross_curve_raw(base: Path, analysis_end: pd.Timestamp, target: str,
                    entity: str) -> pd.Series | None:
    """A raw (unsmoothed) per-window data curve from Stage 2.

    Author ruling 2026-07-19: the 12-day median hides structure and
    flattens the reported per-window peaks below the values the text
    cites; unsmoothed figures draw pct_h directly.
    """
    s = stage2_series(base, target, entity, 'pct_h')
    if s is None:
        return None
    return s[s.index < analysis_end]


def cross_style(draw_index: int) -> Dict:
    """All black, uniform weight; dash pattern by style-priority index."""
    if draw_index >= len(STYLE_PALETTE):
        raise ValueError(f"style priority {draw_index} exceeds the "
                         f"{len(STYLE_PALETTE)}-pattern palette")
    return dict(color='black', linestyle=STYLE_PALETTE[draw_index],
                linewidth=CROSS_LINEWIDTH)


def target_cross_style(target: str) -> Dict:
    """Fixed per-target style for one-line-per-target comparisons."""
    return dict(color='black', linestyle=TARGET_STYLE[target],
                linewidth=CROSS_LINEWIDTH)


# Significance presentation (author ruling 2026-07-09): the region BELOW
# the per-window IAAFT 95% level is shaded (the style contract's light
# envelope gray), wherever the level is plotted.
SIG_BAND_STYLE = dict(patch=True, facecolor=BAND_GRAY, edgecolor='none')


def draw_cross(curves: List[tuple], sig: pd.Series | None, out: Path) -> None:
    """Cross-target panel: sig band below the 95% level; no in-field legend.

    The key ships as the companion ``<out>_legend`` artifact (author
    ruling 2026-07-09), stacked with the figure by LaTeX.
    """
    figsize = setup_style('full', height=FIG_HEIGHT)
    fig, ax = plt.subplots(figsize=figsize)
    ref_index = None
    entries = []
    for label, series, style in curves:
        ax.plot(series.index, series, **style)
        entries.append((label, style))
        ref_index = series.index if ref_index is None else ref_index
    if sig is not None:
        ax.fill_between(sig.index, 0.0, sig, color=BAND_GRAY, linewidth=0,
                        zorder=0)
        entries.append(('Below IAAFT 95% level (max over shown inputs)',
                        SIG_BAND_STYLE))
    ax.set_ylabel('TE (% of H(target))')
    ax.set_xlabel('Date (2022-2023)')
    if ref_index is not None:
        ax.set_xlim(ref_index[0], ref_index[-1])
    date_axis(ax)
    for p in [*save_figure(fig, out),
              *save_legend(entries, Path(f'{out}_legend'))]:
        logger.info(f"wrote {p}")


# Finalized single-driver panel (author 2026-07-10): the four tau=0
# cross-target comparisons in one figure — the dominance-shift story
# (which driver explains which isotope), setting up the composites. Each
# tuple is (CROSS_SELECTIONS key, panel driver name).
# RESTACKED 2026-07-19 (author): a single vertical column (4x1) for
# clarity, and UNSMOOTHED — raw per-window Stage 2 curves and band, so
# the figure shows the per-window peaks the text reports.
SINGLE_DRIVER_PANELS = [
    ('cross_temp_t0', 'Air temperature'),
    ('cross_rh_t0', 'Relative humidity'),
    ('cross_lwup_t0', 'Upwelling longwave'),
    ('cross_h2o3m_t0', 'H2O flux (3 m)'),
]
PANEL_HEIGHT = 7.6      # inches; four stacked rows at full width


def _cross_sel(key: str) -> Dict:
    for s in CROSS_SELECTIONS:
        if s['key'] == key:
            return s
    raise KeyError(key)


def draw_single_driver_panel(base: Path, analysis_end: pd.Timestamp,
                             out: Path) -> None:
    """4x1 column of the tau=0 single-driver comparisons + shared legend.

    Every panel uses the fixed target style (d-excess solid, dD dotted)
    and its own IAAFT band; the driver is named in the panel label, so a
    single shared legend (target convention + band) serves all four.
    Independent y-axes (each driver has its own range); shared date axis.
    Raw per-window curves and band, no smoothing (author, 2026-07-19).
    """
    figsize = setup_style('full', height=PANEL_HEIGHT)
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=figsize)
    tags = ['(a)', '(b)', '(c)', '(d)']
    for ax, (key, name), tag in zip(axs, SINGLE_DRIVER_PANELS, tags):
        sel = _cross_sel(key)
        ref = None
        for target, e in sel['entities']:
            series = cross_curve_raw(base, analysis_end, target, e)
            if series is None:
                logger.warning(f"panel {key}: {target}/{e} unavailable")
                continue
            ax.plot(series.index, series, **target_cross_style(target))
            ref = series.index if ref is None else ref
        sig = cross_significance(base, analysis_end, sel, smooth=False)
        if sig is not None:
            ax.fill_between(sig.index, 0.0, sig, color=BAND_GRAY,
                            linewidth=0, zorder=0)
        if ref is not None:
            ax.set_xlim(ref[0], ref[-1])
        ax.set_ylim(bottom=0.0)
        ax.text(0.02, 0.94, f'{tag} {name}', transform=ax.transAxes,
                va='top', ha='left', fontsize=8)
        date_axis(ax)
    fig.supylabel('TE (% of H(target))', fontsize=10)
    axs[-1].set_xlabel('Date (2022-2023)')
    entries = [(target_label('d_excess'), target_cross_style('d_excess')),
               (target_label('dD'), target_cross_style('dD')),
               ('Below IAAFT 95% level', SIG_BAND_STYLE)]
    for p in [*save_figure(fig, out),
              *save_legend(entries, Path(f'{out}_legend'), ncol=3)]:
        logger.info(f"wrote {p}")


def process_cross(base: Path, analysis_end: pd.Timestamp) -> None:
    """Draw the confirmed cross-target key figures.

    Raw per-window curves and bands from Stage 2, no smoothing (author
    ruling 2026-07-19: the 12-day median hides structure and flattens
    the per-window peaks the text reports). Selections carrying
    ``'smooth': True`` (the 7-line LW-down lag sweep) keep the 12-day
    median for readability, disclosed in caption and prose.
    """
    for stale in base.glob('fig_select_cross_*'):
        stale.unlink()
    for sel in CROSS_SELECTIONS:
        smooth = sel.get('smooth', False)
        # When each target appears at most once, style encodes the target
        # (d-excess solid, dD dotted — fixed across figures); otherwise
        # fall back to the many-line priority palette.
        shown = [t for t, _ in sel['entities']]
        by_target = len(shown) == len(set(shown))
        curves = []
        for target, e in sel['entities']:
            series = cross_curve_raw(base, analysis_end, target, e)
            if series is None:
                logger.warning(f"{sel['key']}: {target}/{e} unavailable "
                               "(needs the Stage 2 run); line skipped")
                continue
            if smooth:
                series = smooth_median(series)
            style = (target_cross_style(target) if by_target
                     else cross_style(len(curves)))
            curves.append((f"{entity_label(e)} → {target_label(target)}",
                           series, style))
        sig = cross_significance(base, analysis_end, sel, smooth=smooth)
        draw_cross(curves, sig, base / f"fig_select_{sel['key']}")
    logger.info(f"cross-target: {len(CROSS_SELECTIONS)} key figures")
    draw_single_driver_panel(base, analysis_end,
                             base / 'fig_single_drivers')


# ═════════════════════════════════════════════════════════════════════════
# Index (pull sheet)
# ═════════════════════════════════════════════════════════════════════════
def write_index(target: str, target_dir: Path, selections: List[Dict],
                raw: pd.DataFrame) -> None:
    """One row per drawn entity: panel key, label, winter mean/peak %.

    Statistics come from the unsmoothed pct_jte series so the choice of
    presentation smoothing cannot bias the pull. Sorted by mean.
    """
    rows = []
    for s in selections:
        for e in s['entities']:
            if e not in raw.columns:
                continue
            rows.append({'panel': s['key'], 'source': s['source'],
                         'entity': e, 'label': entity_label(e),
                         'mean_pct': round(float(raw[e].mean()), 2),
                         'peak_pct': round(float(raw[e].max()), 2)})
    (pd.DataFrame(rows).sort_values('mean_pct', ascending=False)
     .to_csv(target_dir / f'selection_index_{target}.csv', index=False))


# ═════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════
def process_target(target: str, cfg: PublicationConfig, base: Path,
                   smooth: int, analysis_end: pd.Timestamp) -> None:
    """Draw the current stage's panels for one target."""
    target_dir = base / target
    for stale in [*target_dir.glob('fig_select_*'),
                  *target_dir.glob('selection_index_*')]:
        stale.unlink()
    if target not in LIBRARY_TARGETS:
        logger.info(f"{target}: not in the selection library; cleared only")
        return
    layers = load_layers(target_dir, analysis_end)
    if layers is None:
        return
    selections = (build_singles_selections(cfg, target)
                  + build_character_selections(layers))
    write_index(target, target_dir, selections, layers['raw'])

    for suffix, (layer, smoothed) in VARIANTS.items():
        frame = layers[layer]
        if smoothed and smooth > 1:
            frame = frame.rolling(smooth, center=True, min_periods=1).mean()
        for sel in selections:
            draw_selection(
                target, frame, layers['periods'], sel,
                target_dir / f"fig_select_{sel['key']}_{target}{suffix}")
    logger.info(f"{target}: {len(selections)} panels "
                f"({sum(len(s['entities']) for s in selections)} lines); "
                "persistent="
                + ','.join(entity_label(e) for s in selections
                           if s['key'] == 'persistent_singles'
                           for e in s['entities'])
                + "; episodic="
                + ','.join(entity_label(e) for s in selections
                           if s['key'] == 'episodic_singles'
                           for e in s['entities']))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Staged selection library (current stage: singles).")
    parser.add_argument('--output-base', default=None)
    parser.add_argument('--smooth', type=int, default=4,
                        help='Rolling-mean windows if a smoothed-raw variant '
                             'is reinstated (baseline needs none)')
    parser.add_argument('--analysis-end', default=None)
    args = parser.parse_args()

    cfg = PublicationConfig()
    base = Path(args.output_base) if args.output_base else Path(cfg.output_base)
    analysis_end = pd.Timestamp(args.analysis_end if args.analysis_end
                                else cfg.analysis_end)

    for target in cfg.targets:
        process_target(target, cfg, base, args.smooth, analysis_end)
    process_cross(base, analysis_end)


if __name__ == '__main__':
    main()
