#!/usr/bin/env python3
"""
pub_period_attribution.py — 1-3 winning combinations per winter period.

Third stage of the period-attribution layer. All lag-level entities (every
``base__tK`` single and every 2-/3-way combination, all tau permutations on
equal footing) compete inside each winter period from pub_periods. Per period:

  SCORE  = in-period mean pure joint TE (percent of target entropy).
  FLOOR  = in-period mean noise floor (``--floor-bits`` through H(t)).
  GAIN   = score minus the best-scoring entity whose BASE set is a proper
           subset (any taus): every member must earn its place, so a
           combination qualifies only when GAIN >= delta (a fraction of the
           floor, ``--delta-frac``). Kills freeloader members; the raw top of
           every period is otherwise always an unjustified triple.

Winners are picked greedily by score: skip entities below the floor, repeated
base compositions at other taus (search all taus equally, present each
composition once, winning taus attached), base overlaps with already-picked
winners (fully disjoint), and unjustified combinations. Selection stops at
``--max-winners`` (3) or when the next score falls below ``--runner-frac`` of
winner 1, so 1-3 clear, directed winners emerge per period.

One additional EPISODIC slot per period: the entity with the largest
integrated excess above its own baseline inside qualifying bursts (raw above
the noise floor, >= ``--min-burst-windows`` after gap bridging), reported
separately from the mean-TE winners.

Outputs per target: period_winners.csv, period_attribution.csv (audit: every
entity x period), periods.md. Pure post-processing; run pub_driver_series and
pub_periods first.

Usage:
    python3 pub_period_attribution.py
    python3 pub_period_attribution.py --floor-bits 0.12 --delta-frac 0.5

Author: Matthew Rybecky
"""

from __future__ import annotations

import argparse
import logging
from itertools import combinations
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

import numpy as np
import pandas as pd

from pub_combo_metrics import _close_gaps
from pub_config import PublicationConfig, base_of
from pub_driver_series import entity_bases, entity_kind, entity_label
from pub_labels import target_label

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# Loading
# ═════════════════════════════════════════════════════════════════════════
def load_layers(target_dir: Path, analysis_end: pd.Timestamp
                ) -> Optional[Dict[str, pd.DataFrame]]:
    """Raw/baseline/excess matrices, periods, and the entropy series."""
    needed = {'raw': 'driver_series_raw.parquet',
              'baseline': 'driver_series_baseline.parquet',
              'excess': 'driver_series_excess.parquet'}
    out: Dict[str, pd.DataFrame] = {}
    for key, name in needed.items():
        path = target_dir / name
        if not path.exists():
            logger.warning(f"{target_dir.name}: missing {name}; "
                           f"run pub_driver_series first")
            return None
        out[key] = pd.read_parquet(path)
    ppath = target_dir / 'periods.csv'
    if not ppath.exists():
        logger.warning(f"{target_dir.name}: missing periods.csv; "
                       f"run pub_periods first")
        return None
    out['periods'] = pd.read_csv(ppath, parse_dates=['start', 'end'])
    ent = pd.read_csv(target_dir / 'target_entropy_per_window.csv',
                      parse_dates=['timestamp']).set_index('timestamp')
    h = ent['H_target_bits'].reindex(out['raw'].index)
    out['entropy'] = h[h.index < analysis_end].to_frame()
    return out


def floor_pct_series(entropy: pd.Series, floor_bits: float) -> pd.Series:
    """Noise floor in percent of target entropy, per window."""
    return floor_bits / entropy.where(entropy > 0) * 100.0


# ═════════════════════════════════════════════════════════════════════════
# Qualifying episodic bursts
# ═════════════════════════════════════════════════════════════════════════
def _keep_runs(mask: np.ndarray, min_windows: int) -> np.ndarray:
    """Zero out True-runs shorter than ``min_windows``."""
    out = mask.copy()
    n = mask.size
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        if (j - i + 1) < min_windows:
            out[i:j + 1] = False
        i = j + 1
    return out


def burst_mask(raw: pd.DataFrame, excess: pd.DataFrame, floor_pct: pd.Series,
               min_windows: int, gap: int) -> pd.DataFrame:
    """
    Qualifying episodic windows per entity, over the full winter record.

    A window qualifies when the entity's excess is positive AND its raw series
    clears the absolute noise floor; dips up to ``gap`` windows are bridged and
    runs shorter than ``min_windows`` dropped. Computed on the full record
    (before period slicing) so bursts spanning a period boundary are kept and
    split naturally between the adjoining periods.
    """
    above = (excess.gt(0.0) & raw.ge(floor_pct, axis=0)).to_numpy()
    valid = np.ones(above.shape[0], dtype=bool)
    out = np.zeros_like(above)
    for c in range(above.shape[1]):
        bridged = _close_gaps(above[:, c], valid, gap)
        out[:, c] = _keep_runs(bridged, min_windows)
    return pd.DataFrame(out, index=raw.index, columns=raw.columns)


# ═════════════════════════════════════════════════════════════════════════
# Winner selection
# ═════════════════════════════════════════════════════════════════════════
def _subset_best(bases: FrozenSet[str],
                 best_by_set: Dict[FrozenSet[str], float]) -> float:
    """Best score over all proper base-subsets (any taus)."""
    subs = [frozenset(s) for r in range(1, len(bases))
            for s in combinations(sorted(bases), r)]
    return max((best_by_set.get(s, -np.inf) for s in subs), default=-np.inf)


def pick_winners(scores: pd.Series, floor: float, delta: float,
                 max_winners: int, runner_frac: float) -> List[Dict]:
    """
    Greedy selection of justified, base-disjoint winners for one period.

    Parameters
    ----------
    scores : pd.Series
        In-period mean pct_jte per lag-level entity.
    floor : float
        In-period mean noise floor (percent of entropy); hard minimum score.
    delta : float
        Justification margin a combination needs over its best base-subset.
    max_winners : int
        Winner-slot cap (1-3 emerge naturally below this).
    runner_frac : float
        A later winner must score at least this fraction of winner 1.

    Returns
    -------
    list of dict
        One record per winner: entity, score, gain (NaN for singles), raw rank.
    """
    sets = {e: frozenset(entity_bases(e)) for e in scores.index}
    best_by_set: Dict[FrozenSet[str], float] = {}
    for e, s in scores.items():
        if np.isfinite(s) and s > best_by_set.get(sets[e], -np.inf):
            best_by_set[sets[e]] = float(s)
    ranked = scores.dropna().sort_values(ascending=False)
    winners: List[Dict] = []
    used_sets: List[FrozenSet[str]] = []
    used_bases: set = set()
    for rank, (entity, score) in enumerate(ranked.items(), 1):
        if len(winners) >= max_winners or score < floor:
            break
        if winners and score < runner_frac * winners[0]['score']:
            break
        b = sets[entity]
        if b in used_sets or b & used_bases:
            continue  # same composition at other taus / not base-disjoint
        gain = (float(score - _subset_best(b, best_by_set))
                if len(b) > 1 else float('nan'))
        if len(b) > 1 and not (gain >= delta):
            continue  # a member fails to earn its place
        winners.append({'entity': entity, 'score': float(score),
                        'gain': gain, 'raw_rank': rank})
        used_sets.append(b)
        used_bases |= b
    return winners


def episodic_slot(span_excess: pd.DataFrame, span_raw: pd.DataFrame,
                  span_mask: pd.DataFrame) -> Optional[Dict]:
    """The period's single strongest burst entity by integrated excess."""
    integrated = (span_excess * span_mask).sum()
    if integrated.max() <= 0:
        return None
    entity = integrated.idxmax()
    masked = span_raw[entity].where(span_mask[entity])
    return {'entity': entity,
            'ep_integrated_pct': round(float(integrated[entity]), 3),
            'ep_windows': int(span_mask[entity].sum()),
            'ep_peak_pct': round(float(masked.max()), 3),
            'ep_peak_date': masked.idxmax().date()}


# ═════════════════════════════════════════════════════════════════════════
# Per-period assembly
# ═════════════════════════════════════════════════════════════════════════
def attribute_period(period_row: pd.Series, layers: Dict[str, pd.DataFrame],
                     mask: pd.DataFrame, floor_pct: pd.Series, delta_frac: float,
                     max_winners: int, runner_frac: float) -> pd.DataFrame:
    """Winner slots (1..max_winners + episodic) for one period, as rows."""
    span = slice(period_row['start'], period_row['end'])
    raw = layers['raw'].loc[span]
    scores = raw.mean()
    floor = float(floor_pct.loc[span].mean())

    rows = []
    for slot, w in enumerate(pick_winners(scores, floor, delta_frac * floor,
                                          max_winners, runner_frac), 1):
        rows.append({'period': period_row['period'], 'slot': str(slot),
                     'entity': w['entity'], 'mean_pct': round(w['score'], 3),
                     'gain_pct': (round(w['gain'], 3)
                                  if np.isfinite(w['gain']) else np.nan),
                     'raw_rank': w['raw_rank'], 'floor_pct': round(floor, 3)})
    ep = episodic_slot(layers['excess'].loc[span], raw, mask.loc[span])
    if ep is not None:
        rows.append({'period': period_row['period'], 'slot': 'episodic',
                     'floor_pct': round(floor, 3), **ep})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(3, 'label', df['entity'].map(entity_label))
        df.insert(4, 'kind', df['entity'].map(entity_kind))
    return df


def audit_table(layers: Dict[str, pd.DataFrame], mask: pd.DataFrame
                ) -> pd.DataFrame:
    """Every entity x period: mean, baseline mean, and burst integral (audit)."""
    frames = []
    for _, p in layers['periods'].iterrows():
        span = slice(p['start'], p['end'])
        raw = layers['raw'].loc[span]
        frames.append(pd.DataFrame({
            'period': p['period'], 'entity': raw.columns,
            'mean_pct': raw.mean().round(3).to_numpy(),
            'cont_mean_pct': layers['baseline'].loc[span].mean()
            .round(3).to_numpy(),
            'ep_integrated_pct': (layers['excess'].loc[span]
                                  * mask.loc[span]).sum().round(3).to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════
# Reporting
# ═════════════════════════════════════════════════════════════════════════
def write_markdown(target: str, periods: pd.DataFrame, winners: pd.DataFrame,
                   floor_bits: float, delta_frac: float, path: Path) -> None:
    """Concise per-period winner tables, taus attached."""
    lines = [f"# Winter periods and winning combinations: "
             f"{target_label(target)}", "",
             "Per changepoint-tiled period: 1-3 base-disjoint winners by "
             "in-period mean joint TE (a combination qualifies only when it "
             f"beats its best base-subset by {delta_frac:g}x the noise floor; "
             f"floor = {floor_bits} bits through H(t)), plus the strongest "
             "episodic burst entity. All taus searched equally; each base "
             "composition presented once, winning taus shown.", ""]
    for _, p in periods.iterrows():
        grp = winners[winners['period'] == p['period']]
        lines += [f"## Period {p['period']}: {p['start'].date()} to "
                  f"{p['end'].date()} ({p['n_days']:.0f} days, floor "
                  f"{grp['floor_pct'].iloc[0] if not grp.empty else float('nan'):.1f}%)",
                  "",
                  "| slot | winner | mean TE % | gain % | episodic |",
                  "| --- | --- | --- | --- | --- |"]
        if grp.empty:
            lines += ["| — | _no entity clears the floor_ | | | |"]
        for _, r in grp.iterrows():
            if r['slot'] == 'episodic':
                epi = (f"∫excess {r['ep_integrated_pct']:.0f}%, peak "
                       f"{r['ep_peak_pct']:.1f}% on {r['ep_peak_date']} "
                       f"({int(r['ep_windows'])}w)")
                lines.append(f"| episodic | {r['label']} | | | {epi} |")
            else:
                gain = (f"+{r['gain_pct']:.1f}"
                        if np.isfinite(r['gain_pct']) else "—")
                lines.append(f"| {r['slot']} | {r['label']} "
                             f"| {r['mean_pct']:.1f} | {gain} | |")
        lines += [""]
    path.write_text("\n".join(lines) + "\n")


# ═════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════
def process_target(target: str, base: Path, floor_bits: float,
                   delta_frac: float, max_winners: int, runner_frac: float,
                   min_burst_windows: int, gap: int,
                   analysis_end: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Select winners per period for one target and write the tables."""
    target_dir = base / target
    layers = load_layers(target_dir, analysis_end)
    if layers is None:
        return None
    floor_pct = floor_pct_series(layers['entropy']['H_target_bits'], floor_bits)
    mask = burst_mask(layers['raw'], layers['excess'], floor_pct,
                      min_burst_windows, gap)

    frames = [attribute_period(p, layers, mask, floor_pct, delta_frac,
                               max_winners, runner_frac)
              for _, p in layers['periods'].iterrows()]
    winners = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    winners.to_csv(target_dir / 'period_winners.csv', index=False)
    audit_table(layers, mask).to_csv(target_dir / 'period_attribution.csv',
                                     index=False)
    write_markdown(target, layers['periods'], winners, floor_bits, delta_frac,
                   target_dir / 'periods.md')
    n_mean = int((winners['slot'] != 'episodic').sum())
    logger.info(f"{target}: {n_mean} winners + "
                f"{len(winners) - n_mean} episodic slots across "
                f"{len(layers['periods'])} periods")
    winners.insert(0, 'target', target)
    return winners


def main() -> None:
    parser = argparse.ArgumentParser(
        description="1-3 winning combinations per winter period.")
    parser.add_argument('--output-base', default=None)
    parser.add_argument('--floor-bits', type=float, default=0.12,
                        help='KSG noise floor in bits (p95 from the '
                             'uncertainty_noise_floor session: 0.10-0.12)')
    parser.add_argument('--delta-frac', type=float, default=0.5,
                        help='Justification margin as a fraction of the floor')
    parser.add_argument('--max-winners', type=int, default=3,
                        help='Winner slots per period (1-3 emerge naturally)')
    parser.add_argument('--runner-frac', type=float, default=0.6,
                        help='Later winners need this fraction of winner 1')
    parser.add_argument('--min-burst-windows', type=int, default=3,
                        help='Minimum qualifying burst length (3 = 18 h)')
    parser.add_argument('--gap', type=int, default=2,
                        help='Bridge sub-threshold dips up to this many windows')
    parser.add_argument('--analysis-end', default=None,
                        help='Winter cutoff (YYYY-MM-DD); default from pub_config')
    args = parser.parse_args()

    cfg = PublicationConfig()
    base = Path(args.output_base) if args.output_base else Path(cfg.output_base)
    analysis_end = pd.Timestamp(args.analysis_end if args.analysis_end
                                else cfg.analysis_end)

    all_winners = []
    for target in cfg.targets:
        winners = process_target(target, base, args.floor_bits,
                                 args.delta_frac, args.max_winners,
                                 args.runner_frac, args.min_burst_windows,
                                 args.gap, analysis_end)
        if winners is not None and not winners.empty:
            all_winners.append(winners)
    if all_winners:
        pd.concat(all_winners, ignore_index=True).to_csv(
            base / 'period_winners_all.csv', index=False)


if __name__ == '__main__':
    main()
