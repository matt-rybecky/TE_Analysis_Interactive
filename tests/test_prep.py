"""Headless test of the Data Preparation pipeline (tab 1, no GUI).

Exercises the full first-tab workflow on synthetic unaligned data:
load multiple files at different native resolutions, align them onto one
uniform grid, beta-normalize, and circular-encode. Asserts the output is
non-empty, on the requested grid, and numerically sane.

Run from the repository root:

    python3 tests/test_prep.py          # plain run, prints PASS/FAIL
    python3 -m pytest tests/            # pytest, if installed
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from te_explorer.prep import (beta_normalize, compute_beta_calibration,
                              encode_circular, resample_to_grid)
from te_explorer.gui.prep_tab import detect_time_column


def make_met_file() -> pd.DataFrame:
    """Synthetic 5-minute meteorology: temp, vapor pressure, rh, wind dir.

    rh is a deterministic-ish function of temp and vapor pressure so the
    beta calibration has real joint information to find.
    """
    rng = np.random.default_rng(42)
    times = pd.date_range('2023-01-01 00:02', '2023-01-11 00:00',
                          freq='5min')
    n = len(times)
    hours = (times - times[0]).total_seconds() / 3600.0
    temp = 5.0 + 8.0 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 0.8, n)
    vp = 6.0 + 0.25 * temp + rng.normal(0, 0.3, n)
    svp = 6.1 * np.exp(0.067 * temp)
    rh = np.clip(100.0 * vp / svp + rng.normal(0, 2.0, n), 1, 100)
    wdir = (200 + 60 * np.sin(2 * np.pi * hours / 36)
            + rng.normal(0, 15, n)) % 360
    return pd.DataFrame({'timestamp': times, 'met_temp': temp,
                         'met_vapor_pressure': vp, 'met_rh': rh,
                         'met_wdir': wdir})


def make_iso_file() -> pd.DataFrame:
    """Synthetic 47-minute isotope record with a 7-hour gap."""
    rng = np.random.default_rng(7)
    times = pd.date_range('2023-01-01 00:13', '2023-01-11 00:00',
                          freq='47min')
    n = len(times)
    hours = (times - times[0]).total_seconds() / 3600.0
    dD = -120.0 + 10 * np.sin(2 * np.pi * hours / 48) + rng.normal(0, 2, n)
    dxs = 10.0 + 3 * np.cos(2 * np.pi * hours / 48) + rng.normal(0, 1, n)
    df = pd.DataFrame({'timestamp': times, 'dD': dD, 'd_excess': dxs})
    gap = (df['timestamp'] > '2023-01-05 06:00') \
        & (df['timestamp'] < '2023-01-05 13:00')
    df.loc[gap, ['dD', 'd_excess']] = np.nan
    return df


def test_detect_time_column() -> None:
    """The loader finds the datetime column whatever it is named."""
    met = make_met_file()
    met_csv = met.copy()
    met_csv['timestamp'] = met_csv['timestamp'].astype(str)
    assert detect_time_column(met_csv, 'time') == 'timestamp'


def test_align_single_file() -> None:
    """One unaligned file resamples onto a full, non-empty 1-hour grid."""
    met = make_met_file()
    aligned = resample_to_grid(met, time_col='timestamp', interval_hours=1,
                               circular_cols=('met_wdir',))
    assert len(aligned) > 200, f"grid too short: {len(aligned)} rows"
    data_cols = [c for c in aligned.columns if c != 'timestamp']
    assert data_cols, "no data columns survived alignment"
    coverage = aligned[data_cols].notna().mean()
    assert (coverage > 0.95).all(), (
        f"aligned output nearly empty; coverage:\n{coverage}")
    # Gaussian mean must stay inside the data's physical range.
    assert aligned['met_temp'].min() >= met['met_temp'].min() - 1e-6
    assert aligned['met_temp'].max() <= met['met_temp'].max() + 1e-6
    # Circular mean must stay on the circle's value range.
    assert aligned['met_wdir'].between(0, 360).all()


def test_align_preserves_gap() -> None:
    """A 7-hour gap must NOT be interpolated across (max gap 3 h)."""
    iso = make_iso_file()
    aligned = resample_to_grid(iso, time_col='timestamp', interval_hours=1)
    in_gap = (aligned['timestamp'] > '2023-01-05 08:00') \
        & (aligned['timestamp'] < '2023-01-05 11:00')
    assert in_gap.any(), "test grid missed the gap window"
    assert aligned.loc[in_gap, 'dD'].isna().all(), (
        "7-hour gap was bridged; conservative interpolation is broken")


def test_full_chain_multifile() -> None:
    """Two unaligned files -> one aligned, normalized, encoded dataset."""
    from te_explorer.prep import align_datasets

    met, iso = make_met_file(), make_iso_file()
    aligned = align_datasets([(met, 'timestamp'), (iso, 'timestamp')],
                             interval_hours=1, time_col='time',
                             circular_cols=('met_wdir',))
    expected = {'time', 'met_temp', 'met_vapor_pressure', 'met_rh',
                'met_wdir', 'dD', 'd_excess'}
    assert expected.issubset(aligned.columns), (
        f"merged columns wrong: {list(aligned.columns)}")
    assert len(aligned) > 200
    coverage = aligned.drop(columns='time').notna().mean()
    assert (coverage > 0.9).all(), f"poor coverage after merge:\n{coverage}"

    cal = compute_beta_calibration(
        aligned, 'met_rh', ['met_temp', 'met_vapor_pressure'], k=3)
    assert cal.jmi_bits > 0 and cal.entropy_bits > 0
    assert 0 < cal.scale < 10, f"implausible beta scale {cal.scale}"

    norm = beta_normalize(aligned, cal.scale, time_col='time')
    med = norm['met_temp'].median()
    assert abs(med) < 0.05 * cal.scale, (
        f"normalized median not near zero: {med}")

    encoded = encode_circular(norm, aligned, ['met_wdir'], cal.scale)
    assert 'met_wdir' not in encoded.columns
    assert {'met_wdir_sin', 'met_wdir_cos'}.issubset(encoded.columns)
    assert encoded['met_wdir_sin'].notna().sum() > 200


def main() -> int:
    """Plain runner: execute every test_* function, report PASS/FAIL."""
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report, keep running
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
