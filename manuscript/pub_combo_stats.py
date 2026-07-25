#!/usr/bin/env python3
"""
pub_combo_stats.py — Data-driven secondary statistics and targeted-run shortlist.

Pure post-processing on the publication run's saved artifacts. For every candidate
combination AND every single input, reduces the per-window transfer-entropy series
to descriptors that each expose a different kind of "interesting", then selects a
shortlist of combinations worth a deeper, variable-limited follow-up run. No
transfer entropy is recomputed.

Per combination the descriptors are computed on BOTH bases the engine records:
``plotted_value`` (JTE for synergistic/redundant windows, max individual TE for
obfuscating windows: what the composite competition and figures use) and raw
``jte_bits`` (pure joint TE). Each basis is reported as a percent of the target's
per-window entropy (the manuscript's primary metric) and in native bits.

Discovered, not assumed: active periods ("episodes") come from a robust threshold
on the target's own winner envelope, so the regime windows that drive the
alignment statistic are data-driven rather than hand-picked dates.

Inputs per target (from ``publication_output/<target>/``):
  combo_table.parquet, single_te_table.csv, target_entropy_per_window.csv,
  composite_results.csv.
Outputs per target:
  combo_secondary_stats.csv, episodes.csv, combo_shortlist.csv, combo_shortlist.md.
Combined: combo_secondary_stats_all.csv, combo_shortlist_all.csv.

Usage:
    python3 pub_combo_stats.py
    python3 pub_combo_stats.py --output-base publication_output --top 8

Author: Matthew Rybecky
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pub_config import PublicationConfig
from pub_combo_metrics import (Episode, SeriesStats, category_fractions,
                               detect_episodes, episode_alignment, reduce_series)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SINGLE_CATEGORY = 'single'


# ═════════════════════════════════════════════════════════════════════════
# Loading and unification
# ═════════════════════════════════════════════════════════════════════════
def _melt_singles(single_path: Path) -> pd.DataFrame:
    """
    Reshape the wide single-input TE table to long combo-style rows.

    Each single input becomes a size-one "combination" so masked-but-strong
    inputs (e.g. circular wind direction) compete in the same ranking. For a
    single input the joint and plotted values are the input's own TE.
    """
    wide = pd.read_csv(single_path, parse_dates=['timestamp'])
    long = wide.melt(id_vars='timestamp', var_name='combo', value_name='jte_bits')
    long['plotted_value'] = long['jte_bits']
    long['category'] = SINGLE_CATEGORY
    return long


def load_target_frames(target_dir: Path, analysis_end: Optional[pd.Timestamp] = None
                       ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame,
                                           pd.Series, pd.Series, int]]:
    """
    Load and unify the per-target artifacts.

    Parameters
    ----------
    target_dir : Path
        Per-target artifact directory.
    analysis_end : pd.Timestamp, optional
        If given, every frame is truncated to timestamps strictly before this
        date, so all downstream statistics operate on the winter record only
        (the late-spring snowmelt/diurnal regime is dropped, not merely flagged).

    Returns
    -------
    tuple or None
        (long, single_wide, entropy, winner_counts, n_windows) where ``long`` is
        the combo+single per-window frame with ``pct_plotted``/``pct_jte`` added,
        ``single_wide`` is timestamp-indexed single TEs (for synergy gain),
        ``entropy`` is timestamp-indexed H_target_bits, ``winner_counts`` maps
        each combo to its composite-win count, and ``n_windows`` is the window
        total. None if a required artifact is missing.
    """
    combo_path = target_dir / 'combo_table.parquet'
    single_path = target_dir / 'single_te_table.csv'
    ent_path = target_dir / 'target_entropy_per_window.csv'
    comp_path = target_dir / 'composite_results.csv'
    for p in (combo_path, single_path, ent_path, comp_path):
        if not p.exists():
            logger.warning(f"Missing {p.name} in {target_dir}; skipping")
            return None

    combo = pd.read_parquet(combo_path)
    combo['timestamp'] = pd.to_datetime(combo['timestamp'])
    singles_long = _melt_singles(single_path)
    long = pd.concat([combo[['timestamp', 'combo', 'jte_bits', 'plotted_value',
                             'category']], singles_long], ignore_index=True)

    ent = pd.read_csv(ent_path, parse_dates=['timestamp']).set_index('timestamp')
    comp = pd.read_csv(comp_path, parse_dates=['timestamp'])
    single_wide = pd.read_csv(single_path, parse_dates=['timestamp']
                              ).set_index('timestamp')
    if analysis_end is not None:
        long = long[long['timestamp'] < analysis_end]
        ent = ent[ent.index < analysis_end]
        comp = comp[comp['timestamp'] < analysis_end]
        single_wide = single_wide[single_wide.index < analysis_end]

    h = ent['H_target_bits']
    hvals = long['timestamp'].map(h).where(lambda s: s > 0)
    long = long.copy()
    long['pct_plotted'] = long['plotted_value'] / hvals * 100.0
    long['pct_jte'] = long['jte_bits'] / hvals * 100.0

    winner_counts = comp['combo'].value_counts()
    n_windows = int(comp['timestamp'].nunique())
    return long, single_wide, h, winner_counts, n_windows


# ═════════════════════════════════════════════════════════════════════════
# Episodes and per-combination statistics
# ═════════════════════════════════════════════════════════════════════════
def build_episodes(long: pd.DataFrame, entropy: pd.Series, k: float,
                   max_gap: int, min_windows: int,
                   diurnal_start: Optional[pd.Timestamp]
                   ) -> Tuple[pd.Series, List[Episode], pd.DatetimeIndex,
                              pd.Series]:
    """
    Discover the target's active periods from its composite winner envelope.

    The envelope is the per-window maximum ``pct_plotted`` across all combinations
    (the composite winner value normalized by entropy). Windows on or after
    ``diurnal_start`` are excluded from the threshold statistics and from episode
    formation, so the late-spring diurnal artifact neither inflates the threshold
    nor registers as a transport episode. Short below-threshold dips up to
    ``max_gap`` windows are bridged so a physically continuous band is not
    fragmented.

    Returns the episode mask, the episode list, the ordered timestamp axis, and
    the per-window validity mask (all timestamp-indexed / aligned to that axis).
    """
    envelope = (long.groupby('timestamp')['pct_plotted'].max()
                .reindex(entropy.index))
    valid = (pd.Series(True, index=entropy.index) if diurnal_start is None
             else pd.Series(entropy.index < diurnal_start, index=entropy.index))
    mask_arr, episodes = detect_episodes(
        envelope.to_numpy(), k=k, min_windows=min_windows,
        max_gap=max_gap, valid_mask=valid.to_numpy())
    mask = pd.Series(mask_arr, index=entropy.index)
    return mask, episodes, entropy.index, valid


def _synergy_gain_bits(members: List[str], timestamps: np.ndarray,
                       jte_bits: np.ndarray,
                       single_wide: pd.DataFrame) -> float:
    """
    Mean bits of joint TE beyond the best single member, over shared windows.

    NaN for size-one entries or when no member has single-input TE recorded.
    """
    cols = [m for m in members if m in single_wide.columns]
    if len(members) < 2 or not cols:
        return float('nan')
    member_max = single_wide.reindex(timestamps)[cols].max(axis=1).to_numpy()
    gain = jte_bits - member_max
    gain = gain[np.isfinite(gain)]
    return float(gain.mean()) if gain.size else float('nan')


def _stats_block(stats: SeriesStats, prefix: str) -> Dict[str, float]:
    """Flatten a SeriesStats into prefixed percent-basis columns."""
    return {
        f'{prefix}_integrated_pct': stats.total,
        f'{prefix}_mean_pct': stats.mean,
        f'{prefix}_peak_pct': stats.peak,
        f'{prefix}_std_pct': stats.std,
        f'{prefix}_cv': stats.cv,
        f'{prefix}_gini': stats.gini,
    }


def combo_stats_row(combo: str, group: pd.DataFrame, single_wide: pd.DataFrame,
                    episode_mask: pd.Series, valid_mask: pd.Series,
                    winner_counts: pd.Series,
                    n_windows: int) -> Dict[str, object]:
    """
    Reduce one combination's per-window series to a single statistics row.

    Parameters
    ----------
    combo : str
        Combination key (``+``-joined candidate names) or a single input name.
    group : pd.DataFrame
        Rows for this combination, any order; sorted here by timestamp.
    single_wide : pd.DataFrame
        Timestamp-indexed single-input TE (bits) for the synergy-gain term.
    episode_mask : pd.Series
        Target active-period mask, timestamp-indexed.
    valid_mask : pd.Series
        Per-window eligibility (excludes the diurnal span), timestamp-indexed;
        used so the episode-alignment baseline ignores the excluded windows.
    winner_counts : pd.Series
        Composite-win counts per combination.
    n_windows : int
        Total number of windows (denominator for win share).

    Returns
    -------
    dict
        One flat record: identity, both-basis percent descriptors, native-bit
        integrals/peaks, synergy gain, win share, category dynamics, and the
        episode-alignment ratio.
    """
    g = group.sort_values('timestamp')
    ts = g['timestamp'].to_numpy()
    members = combo.split('+')
    is_single = group['category'].iloc[0] == SINGLE_CATEGORY

    pct_plotted = g['pct_plotted'].to_numpy()
    pct_jte = g['pct_jte'].to_numpy()
    plotted_stats = reduce_series(pct_plotted)
    jte_stats = reduce_series(pct_jte)

    aligned_mask = episode_mask.reindex(ts).to_numpy().astype(bool)
    aligned_valid = valid_mask.reindex(ts).to_numpy().astype(bool)
    peak_ts = ts[jte_stats.peak_index] if jte_stats.peak_index >= 0 else None

    row: Dict[str, object] = {
        'combo': combo,
        'n_members': 1 if is_single else combo.count('+') + 1,
        'is_single': bool(is_single),
        'n_windows': jte_stats.n,
        'jte_integrated_bits': float(np.nansum(g['jte_bits'].to_numpy())),
        'jte_peak_bits': float(np.nanmax(g['jte_bits'].to_numpy())),
        'peak_date': pd.Timestamp(peak_ts).date() if peak_ts is not None else None,
        'synergy_gain_bits': _synergy_gain_bits(
            members, ts, g['jte_bits'].to_numpy(), single_wide),
        'win_share': float(winner_counts.get(combo, 0)) / n_windows if n_windows
        else float('nan'),
        'episode_ratio': episode_alignment(pct_jte, aligned_mask, aligned_valid),
    }
    row.update(_stats_block(plotted_stats, 'plotted'))
    row.update(_stats_block(jte_stats, 'jte'))
    row.update(category_fractions(g['category'].to_numpy()))
    return row


def build_stats_table(long: pd.DataFrame, single_wide: pd.DataFrame,
                      episode_mask: pd.Series, valid_mask: pd.Series,
                      winner_counts: pd.Series,
                      n_windows: int) -> pd.DataFrame:
    """Assemble the per-combination statistics table, sorted by integrated JTE %."""
    rows = [combo_stats_row(combo, grp, single_wide, episode_mask, valid_mask,
                            winner_counts, n_windows)
            for combo, grp in long.groupby('combo', sort=False)]
    df = pd.DataFrame(rows)
    return df.sort_values('jte_integrated_pct', ascending=False,
                          ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════
# Shortlist selection
# ═════════════════════════════════════════════════════════════════════════
# Each criterion: (label, sort column, ascending, optional pre-filter).
def _shortlist_criteria(stats: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
    """
    Build the candidate pools for each selection lens.

    Returns a list of (reason, ranked-subframe) pairs. Each lens surfaces a
    distinct kind of interesting combination; the union becomes the shortlist.
    """
    peak_floor = stats['jte_peak_pct'].median()
    strong = stats[stats['jte_peak_pct'] >= peak_floor]
    multi = stats[~stats['is_single']]
    masked = stats[stats['win_share'] <= 0.0]
    return [
        ('persistent', stats.sort_values('jte_integrated_pct', ascending=False)),
        ('episodic', strong.sort_values('jte_gini', ascending=False)),
        ('synergistic', multi.sort_values('synergy_gain_bits', ascending=False)),
        ('masked', masked.sort_values('jte_integrated_pct', ascending=False)),
        ('switching', strong.sort_values('n_flips', ascending=False)),
        ('episode_aligned', strong.sort_values('episode_ratio', ascending=False)),
    ]


def select_shortlist(stats: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Union the top ``top_n`` of each selection lens into a tagged shortlist.

    Parameters
    ----------
    stats : pd.DataFrame
        Output of :func:`build_stats_table`.
    top_n : int
        How many to take from each lens before the union.

    Returns
    -------
    pd.DataFrame
        Shortlisted combinations with a ``reasons`` column (semicolon-joined
        lenses that selected it) and the headline statistic columns, ordered by
        number of reasons then integrated JTE %.
    """
    reasons: Dict[str, List[str]] = {}
    for reason, ranked in _shortlist_criteria(stats):
        picked = ranked.dropna(subset=[ranked.columns[0]]).head(top_n)
        for combo in picked['combo']:
            reasons.setdefault(combo, []).append(reason)

    if not reasons:
        return pd.DataFrame()
    keep = stats[stats['combo'].isin(reasons)].copy()
    keep['reasons'] = keep['combo'].map(lambda c: ';'.join(reasons[c]))
    keep['n_reasons'] = keep['combo'].map(lambda c: len(reasons[c]))
    cols = ['combo', 'n_members', 'reasons', 'n_reasons', 'jte_mean_pct',
            'jte_integrated_pct', 'jte_peak_pct', 'jte_gini', 'jte_cv',
            'synergy_gain_bits', 'win_share', 'episode_ratio', 'peak_date']
    return keep[cols].sort_values(['n_reasons', 'jte_integrated_pct'],
                                  ascending=[False, False], ignore_index=True)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table (no dependency)."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    rule = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for _, r in df.iterrows():
        cells = [f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])
                 for c in cols]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule] + body)


def write_shortlist_md(shortlist: pd.DataFrame, episodes: List[Episode],
                       timestamps: pd.DatetimeIndex, target: str,
                       path: Path) -> None:
    """Write a human-readable shortlist with the discovered episode windows."""
    lines = [f"# Targeted-run shortlist: {target}", ""]
    if episodes:
        lines.append("## Discovered active periods (data-driven)")
        for i, ep in enumerate(episodes, 1):
            start = pd.Timestamp(timestamps[ep.start_index]).date()
            end = pd.Timestamp(timestamps[ep.end_index]).date()
            peak = pd.Timestamp(timestamps[ep.peak_index]).date()
            lines.append(f"{i}. {start} to {end} "
                         f"(peak {peak}, {ep.n_windows} windows)")
        lines.append("")
    lines.append("## Shortlisted combinations")
    lines.append(_df_to_markdown(shortlist) if not shortlist.empty
                 else "_none selected_")
    path.write_text("\n".join(lines) + "\n")


# ═════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════
def process_target(target: str, base: Path, top_n: int, episode_k: float,
                   episode_gap: int, episode_min_windows: int,
                   diurnal_start: Optional[pd.Timestamp],
                   analysis_end: Optional[pd.Timestamp] = None
                   ) -> Optional[pd.DataFrame]:
    """Run the full statistics + shortlist pipeline for one target."""
    target_dir = base / target
    loaded = load_target_frames(target_dir, analysis_end)
    if loaded is None:
        return None
    long, single_wide, entropy, winner_counts, n_windows = loaded

    episode_mask, episodes, ts_axis, valid_mask = build_episodes(
        long, entropy, episode_k, episode_gap, episode_min_windows,
        diurnal_start)
    stats = build_stats_table(long, single_wide, episode_mask, valid_mask,
                              winner_counts, n_windows)
    stats.to_csv(target_dir / 'combo_secondary_stats.csv', index=False)

    _write_episodes(episodes, ts_axis, target_dir / 'episodes.csv')

    shortlist = select_shortlist(stats, top_n)
    shortlist.to_csv(target_dir / 'combo_shortlist.csv', index=False)
    write_shortlist_md(shortlist, episodes, ts_axis, target,
                       target_dir / 'combo_shortlist.md')
    logger.info(f"{target}: {len(stats)} combinations ranked, "
                f"{len(shortlist)} shortlisted, {len(episodes)} episodes")

    if not shortlist.empty:
        shortlist.insert(0, 'target', target)
        return shortlist
    return None


def _write_episodes(episodes: List[Episode], timestamps: pd.DatetimeIndex,
                    path: Path) -> None:
    """Persist discovered episodes as a dated table."""
    rows = [{
        'episode': i,
        'start': pd.Timestamp(timestamps[ep.start_index]).date(),
        'end': pd.Timestamp(timestamps[ep.end_index]).date(),
        'peak_date': pd.Timestamp(timestamps[ep.peak_index]).date(),
        'peak_value_pct': round(ep.peak_value, 3),
        'n_windows': ep.n_windows,
    } for i, ep in enumerate(episodes, 1)]
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data-driven secondary statistics and targeted-run shortlist.")
    parser.add_argument('--output-base', default=None,
                        help='Root of per-target output dirs (default: pub_config)')
    parser.add_argument('--top', type=int, default=8,
                        help='Per-lens count before the shortlist union (default 8)')
    parser.add_argument('--episode-k', type=float, default=1.0,
                        help='Episode threshold sensitivity (median + k*1.4826*MAD)')
    parser.add_argument('--episode-gap', type=int, default=8,
                        help='Max below-threshold dip in windows to bridge so a '
                             'continuous band is not fragmented (8 = 2 days at '
                             '6-hr resolution; 0 disables merging)')
    parser.add_argument('--episode-min-windows', type=int, default=3,
                        help='Minimum windows (after merging) for an episode '
                             '(default 3 drops sub-day blips)')
    parser.add_argument('--diurnal-start', default='2023-05-01',
                        help='Exclude windows on/after this date (YYYY-MM-DD) '
                             'from episode detection and the alignment baseline; '
                             "'none' to keep the full record. Matches the "
                             'figure/table diurnal convention.')
    parser.add_argument('--analysis-end', default=None,
                        help='Truncate all frames before this date (YYYY-MM-DD); '
                             'default from pub_config (2023-05-01). Isolates winter.')
    args = parser.parse_args()

    cfg = PublicationConfig()
    base = Path(args.output_base) if args.output_base else Path(cfg.output_base)
    diurnal_start = (None if str(args.diurnal_start).lower() == 'none'
                     else pd.Timestamp(args.diurnal_start))
    analysis_end = pd.Timestamp(args.analysis_end if args.analysis_end
                                else cfg.analysis_end)

    shortlists, stats_all = [], []
    for target in cfg.targets:
        shortlist = process_target(target, base, args.top, args.episode_k,
                                   args.episode_gap, args.episode_min_windows,
                                   diurnal_start, analysis_end)
        stats_path = base / target / 'combo_secondary_stats.csv'
        if stats_path.exists():
            tdf = pd.read_csv(stats_path)
            tdf.insert(0, 'target', target)
            stats_all.append(tdf)
        if shortlist is not None:
            shortlists.append(shortlist)

    if stats_all:
        pd.concat(stats_all, ignore_index=True).to_csv(
            base / 'combo_secondary_stats_all.csv', index=False)
    if shortlists:
        pd.concat(shortlists, ignore_index=True).to_csv(
            base / 'combo_shortlist_all.csv', index=False)
        logger.info(f"Wrote combined shortlist ({sum(len(s) for s in shortlists)} "
                    f"rows across {len(shortlists)} targets)")
    else:
        logger.warning("No shortlists produced — run the analysis first.")


if __name__ == '__main__':
    main()
