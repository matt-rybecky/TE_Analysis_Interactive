#!/usr/bin/env python3
"""pub_data_coverage.py — isotope-record coverage + descriptive statistics.

Source-of-truth artifact for the manuscript's data description (Study Area
coverage sentence, A2) and the isotope presentation statistics. Reads the
ANALYZED base stream `data/final_1hr.csv`: the real-unit, spike-removed
1-hour record that the transfer-entropy analysis runs on (robust-z
normalized into `final_1hr_beta.csv`). Provenance verified 2026-07-13:
byte-identical to the isotope pipeline's
`data/processed/extended_june/unnormalized/final_1hr.csv` (Feb 2026
vintage). The underlying calibration is VSMOW/SLAP-corrected Los Gatos
analyzer output at 1-minute cadence
(`isotope-analysis-pipeline/.../atmospheric_isotopes_vsmow_final_cleaned.csv`).
That 1-minute product IS the analyzed stream: aggregating it with the
analysis kernel reproduces this file to ~1e-12 (the 2022-12-12 malfunction
window is already interpolated in it), and it is published with per-point
uncertainties in `WRITING/Manuscript/data/published/` (verified 2026-07-16).
This corrects an earlier note that the 1-minute product predated the spike
removal. The older `outputs/summary_statistics/summary_statistics_1hr.*`
is a stale vintage (N=3841, Nov-Apr) and is superseded by this script.

Per isotope variable (dD, d18O, d-excess, H2O): coverage over the regular
hourly grid (N valid, uptime %, gap count, longest gap within the
measurement span) and descriptive statistics (min, max, mean, std,
median, IQR). A QC block lists the most extreme low/high values with
their timestamps so any residual spike is caught before a number enters
the paper.

Run:
    .venv/bin/python3 pub_data_coverage.py

Outputs (publication_output/data_coverage/):
    isotope_coverage_stats.{csv,md,tex}   coverage + descriptive stats
    isotope_qc_extremes.{csv,md}          k lowest/highest with timestamps

Author: Matthew Rybecky
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pub_config import PublicationConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = Path('data/final_1hr.csv')
TIME_COL = 'time'
OUT_DIR = Path('publication_output/data_coverage')
GRID_FREQ = '1h'
N_EXTREMES = 8   # k lowest and k highest listed per variable in the QC block

# Isotope variables in presentation order: column -> (label, units, tex_label).
ISOTOPES: Dict[str, Tuple[str, str, str]] = {
    'dD':       ('dD', 'permil', r'$\delta$D'),
    'd18O':     ('d18O', 'permil', r'$\delta^{18}$O'),
    'd_excess': ('d-excess', 'permil', 'd-excess'),
    'H2O_ppm':  ('H2O', 'ppm', r'H$_2$O'),
}


def load_grid(path: Path, end: Optional[str] = None) -> pd.DataFrame:
    """Load the base stream onto a strictly regular hourly grid.

    Parameters
    ----------
    path : Path
        `final_1hr.csv` (real-unit, spike-removed analyzed stream).
    end : str, optional
        Exclusive upper timestamp bound. Passed `pub_config.analysis_end`
        so the statistics describe exactly the winter-truncated record the
        transfer-entropy analysis uses (late-spring is excluded).

    Returns
    -------
    pandas.DataFrame
        Indexed by a complete 1-hour DatetimeIndex from first to last
        timestamp, so missing hours are explicit NaN rows.
    """
    df = pd.read_csv(path, parse_dates=[TIME_COL]).set_index(TIME_COL)
    df = df.sort_index()
    if end is not None:
        df = df[df.index < pd.Timestamp(end)]
    full = pd.date_range(df.index.min(), df.index.max(), freq=GRID_FREQ)
    if not df.index.equals(full):
        logger.info(f"reindexing {len(df)} rows onto {len(full)} hourly grid "
                    f"points (filling {len(full) - len(df)} missing hours)")
        df = df.reindex(full)
    return df


def longest_gap_hours(valid: pd.Series) -> int:
    """Longest run of consecutive missing hours within the measured span.

    Parameters
    ----------
    valid : pandas.Series
        Boolean per hourly grid point (True = observation present),
        restricted to the measurement span before calling.

    Returns
    -------
    int
        Length in hours of the longest consecutive gap (0 if none).
    """
    longest = run = 0
    for present in valid:
        run = 0 if present else run + 1
        longest = max(longest, run)
    return longest


def coverage(df: pd.DataFrame, col: str) -> Dict:
    """Coverage statistics for one variable over its measurement span."""
    present = df[col].notna()
    if not present.any():
        return {'n_total_grid': len(df), 'n_valid': 0, 'coverage_pct': 0.0}
    first, last = present[present].index[[0, -1]]
    span = df.loc[first:last, col]
    span_present = span.notna()
    n_span = len(span)
    n_valid = int(span_present.sum())
    gap = ~span_present
    return {
        'n_total_grid': len(df),
        'first_valid': first, 'last_valid': last,
        'span_hours': n_span, 'n_valid': n_valid,
        'coverage_pct': 100.0 * n_valid / n_span,
        'n_gaps': int((gap & ~gap.shift(fill_value=False)).sum()),
        'longest_gap_hours': longest_gap_hours(span_present.values),
    }


def describe(series: pd.Series) -> Dict:
    """Descriptive statistics for one variable (NaN dropped)."""
    s = series.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    return {'min': s.min(), 'max': s.max(), 'mean': s.mean(),
            'std': s.std(), 'median': s.median(),
            'q1': q1, 'q3': q3, 'iqr': q3 - q1}


def extremes(df: pd.DataFrame, col: str, k: int) -> pd.DataFrame:
    """The k lowest and k highest values of a variable, with timestamps."""
    s = df[col].dropna()
    rows: List[Dict] = []
    for tag, idx in (('low', s.nsmallest(k).index),
                     ('high', s.nlargest(k).index)):
        for t in idx:
            rows.append({'variable': col, 'end': tag,
                         'timestamp': t, 'value': s.loc[t]})
    return pd.DataFrame(rows)


def build_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the per-variable stats table and the QC-extremes table."""
    stats, qc = [], []
    for col, (label, units, _) in ISOTOPES.items():
        row = {'variable': col, 'label': label, 'units': units}
        row.update(coverage(df, col))
        row.update(describe(df[col]))
        stats.append(row)
        qc.append(extremes(df, col, N_EXTREMES))
    return pd.DataFrame(stats), pd.concat(qc, ignore_index=True)


def write_markdown(stats: pd.DataFrame, path: Path) -> None:
    """Human-readable coverage + statistics table."""
    lines = ['# Isotope record — coverage and descriptive statistics',
             '',
             '*Source: `data/final_1hr.csv` (analyzed, spike-removed base '
             'stream), winter-truncated at `pub_config.analysis_end` to '
             'match the transfer-entropy analysis. Generated by '
             '`pub_data_coverage.py`.*', '',
             '| Variable | Units | N | Coverage % | Longest gap (h) | '
             'Min | Max | Mean | Std | Median | IQR |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | '
             '--- | --- |']
    for _, r in stats.iterrows():
        lines.append(
            f"| {r['label']} | {r['units']} | {r['n_valid']:,} | "
            f"{r['coverage_pct']:.1f} | {r['longest_gap_hours']} | "
            f"{r['min']:.2f} | {r['max']:.2f} | {r['mean']:.2f} | "
            f"{r['std']:.2f} | {r['median']:.2f} | {r['iqr']:.2f} |")
    span = stats.iloc[0]
    lines += ['', f"Measurement span: {span['first_valid']} to "
              f"{span['last_valid']} ({span['span_hours']:,} hourly grid "
              f"points).", '']
    path.write_text('\n'.join(lines))


def write_latex(stats: pd.DataFrame, path: Path) -> None:
    """Booktabs table for the manuscript SI/Study Area.

    Each variable label carries its unit (author, 2026-07-19), so every
    statistic column reads in the row's unit without a separate column.
    """
    head = [r'\begin{table}[t]', r'\centering',
            r'\caption{Isotope record coverage and descriptive statistics '
            r'(analyzed 1-hour stream, winter-truncated).}',
            r'\label{tab:coverage}',
            r'\begin{tabular}{lrrrrrrr}',
            r'\hline',
            r'Variable & $N$ & Cov.\ (\%) & Min & Max & Mean & Std & '
            r'Median \\', r'\hline']
    body = []
    for _, r in stats.iterrows():
        unit = r'\permil{}' if r['units'] == 'permil' else 'ppm'
        body.append(
            f"{ISOTOPES[r['variable']][2]} ({unit}) & {r['n_valid']:,} & "
            f"{r['coverage_pct']:.1f} & {r['min']:.2f} & {r['max']:.2f} & "
            f"{r['mean']:.2f} & {r['std']:.2f} & {r['median']:.2f} \\\\")
    tail = [r'\hline', r'\end{tabular}', r'\end{table}']
    path.write_text('\n'.join(head + body + tail))


def main() -> None:
    cfg = PublicationConfig()
    df = load_grid(DATA_FILE, end=cfg.analysis_end)
    logger.info(f"winter-analyzed window: < {cfg.analysis_end} "
                f"(pub_config.analysis_end); {len(df):,} hourly grid points")
    stats, qc = build_tables(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats.to_csv(OUT_DIR / 'isotope_coverage_stats.csv', index=False)
    qc.to_csv(OUT_DIR / 'isotope_qc_extremes.csv', index=False)
    write_markdown(stats, OUT_DIR / 'isotope_coverage_stats.md')
    write_latex(stats, OUT_DIR / 'isotope_coverage_stats.tex')

    for _, r in stats.iterrows():
        logger.info(f"{r['label']:9s} N={r['n_valid']:,} "
                    f"cov={r['coverage_pct']:.1f}% gap<={r['longest_gap_hours']}h "
                    f"min={r['min']:.2f} max={r['max']:.2f} "
                    f"mean={r['mean']:.2f}")
    logger.info(f"QC extremes ({N_EXTREMES} low/high per variable) -> "
                f"{OUT_DIR / 'isotope_qc_extremes.md'}")
    _write_qc_markdown(qc, OUT_DIR / 'isotope_qc_extremes.md')
    logger.info(f"wrote artifacts to {OUT_DIR}/")


def _write_qc_markdown(qc: pd.DataFrame, path: Path) -> None:
    """QC block: extreme values with timestamps, for author eyeball."""
    lines = ['# Isotope QC — extreme values with timestamps', '',
             '*Confirm none are residual instrument spikes before any '
             'becomes a reported number.*', '']
    for col, (label, units, _) in ISOTOPES.items():
        sub = qc[qc['variable'] == col]
        lines.append(f'## {label} ({units})')
        lines.append('| End | Timestamp | Value |')
        lines.append('| --- | --- | --- |')
        for _, r in sub.iterrows():
            lines.append(f"| {r['end']} | {r['timestamp']} | "
                         f"{r['value']:.3f} |")
        lines.append('')
    path.write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
