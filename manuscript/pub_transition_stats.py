#!/usr/bin/env python3
"""
pub_transition_stats.py — Statistics across the melting-point transition.

Source-of-truth artifact for the manuscript's melting-point recast
(author, 2026-07-19): the divergence argument rests on the air
temperature control and on the step in the mean isotopic composition,
not on upwelling longwave (whose transfer simply falls for BOTH targets
as the snowpack approaches isothermality).

Boundary: 2023-03-13, the day the snow surface first reaches 0 C in the
SPLASH ASFS-30 record (Cox et al. 2025; cited in the manuscript as
Cox2025_asfs30).

Computed, per target (dD, d-excess):
  - Mean isotope composition before/after the boundary, full record and
    +/- 30-day windows (the step-change comparison).
  - Per-window TE statistics (mean pct_h, significant windows, post
    peak, last significant window) before/after the boundary for the
    air temperature and upwelling longwave channels (Stage 2 CSVs).

Outputs:
  publication_output/transition_stats.json
  publication_output/transition_stats.md

Usage:
    python3 pub_transition_stats.py

Author: Matthew Rybecky
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from pub_config import PublicationConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Snow surface first reaches 0 C (SPLASH ASFS-30; Cox2025_asfs30).
BOUNDARY = pd.Timestamp('2023-03-13')
STEP_DAYS = 30                    # +/- window for the step comparison
ISOTOPE_CSV = Path('data/final_6hr.csv')
TARGETS = ('d_excess', 'dD')
ENTITIES = ('met_temp__t0', 'rad_lw_up__t0')
OUT_BASE = Path('publication_output')


def isotope_means(path: Path, end: pd.Timestamp) -> Dict:
    """Mean composition before/after the boundary, full and +/- 30 d."""
    df = pd.read_csv(path, parse_dates=['time']).set_index('time')
    df = df[df.index < end]
    out: Dict = {}
    for col in TARGETS:
        s = df[col].dropna()
        step = pd.Timedelta(days=STEP_DAYS)
        out[col] = {
            'mean_pre_full': round(float(s[s.index < BOUNDARY].mean()), 2),
            'mean_post_full': round(float(s[s.index >= BOUNDARY].mean()), 2),
            'mean_pre_30d': round(float(
                s[(s.index >= BOUNDARY - step)
                  & (s.index < BOUNDARY)].mean()), 2),
            'mean_post_30d': round(float(
                s[(s.index >= BOUNDARY)
                  & (s.index < BOUNDARY + step)].mean()), 2),
        }
    return out


def te_transition(base: Path, end: pd.Timestamp) -> Dict:
    """Pre/post TE statistics for the temperature and LW-up channels."""
    out: Dict = {}
    for entity in ENTITIES:
        for target in TARGETS:
            path = base / 'stage2' / target / f'stage2_{entity}.csv'
            d = pd.read_csv(path, parse_dates=['timestamp']
                            ).set_index('timestamp')
            d = d[d.index < end]
            pre, post = d[d.index < BOUNDARY], d[d.index >= BOUNDARY]
            last_sig = d[d['sig95']].index.max()
            out[f'{entity}->{target}'] = {
                'pre_mean_pct_h': round(float(pre['pct_h'].mean()), 1),
                'pre_sig': f"{int(pre['sig95'].sum())}/{len(pre)}",
                'post_mean_pct_h': round(float(post['pct_h'].mean()), 1),
                'post_sig': f"{int(post['sig95'].sum())}/{len(post)}",
                'post_peak_pct_h': round(float(post['pct_h'].max()), 1),
                'last_sig_window': f'{last_sig:%Y-%m-%d}',
            }
    return out


def write_markdown(stats: Dict, path: Path) -> None:
    """Readable companion to the JSON artifact."""
    lines = ['# Melting-point transition statistics',
             f"Boundary: {stats['boundary']} (snow surface first 0 C, "
             'Cox2025_asfs30); step windows +/- '
             f"{STEP_DAYS} d; record < {stats['analysis_end']}.", '',
             '## Isotope means (permil)']
    for col, m in stats['isotope_means'].items():
        lines.append(f"- {col}: full {m['mean_pre_full']} -> "
                     f"{m['mean_post_full']}; 30-day {m['mean_pre_30d']} "
                     f"-> {m['mean_post_30d']}")
    lines += ['', '## TE channels across the boundary (% of H)']
    for key, t in stats['te_channels'].items():
        lines.append(f"- {key}: pre mean {t['pre_mean_pct_h']}% "
                     f"(sig {t['pre_sig']}) -> post mean "
                     f"{t['post_mean_pct_h']}% (sig {t['post_sig']}, peak "
                     f"{t['post_peak_pct_h']}%); last significant window "
                     f"{t['last_sig_window']}")
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    cfg = PublicationConfig()
    end = pd.Timestamp(cfg.analysis_end)
    stats = {
        'boundary': f'{BOUNDARY:%Y-%m-%d}',
        'analysis_end': f'{end:%Y-%m-%d}',
        'step_days': STEP_DAYS,
        'isotope_means': isotope_means(ISOTOPE_CSV, end),
        'te_channels': te_transition(OUT_BASE, end),
    }
    (OUT_BASE / 'transition_stats.json').write_text(
        json.dumps(stats, indent=2) + '\n')
    write_markdown(stats, OUT_BASE / 'transition_stats.md')
    logger.info(f"wrote {OUT_BASE / 'transition_stats.json'} and .md")


if __name__ == '__main__':
    main()
