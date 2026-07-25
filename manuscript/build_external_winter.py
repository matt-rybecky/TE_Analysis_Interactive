#!/usr/bin/env python3
"""
build_external_winter.py — Winter 6-h series from the external SPLASH datasets.

Reduces the downloaded Kettle Ponds corroboration data (see
``data/external/README.md``) to analysis-ready 6-hour CSVs aligned with the
TE grid (00/06/12/18 UTC), for overlay on the d-excess period figures:

  SPLASH ASFS-30 sledseb (10-min, double-rotation variant):
    snow depth (SR-50A acoustic), radiometric skin temperature, air
    temperature, the sled up/down longwave fluxes, and the bulk-aerodynamic
    latent heat flux (``bulk_Hl``, flagged by ``bulk_qc``). QC flags 0
    (good) and 1 (caution) are kept; 2 (bad) and 3 (engineering) are
    dropped. The archive's eddy-covariance latent flux (``Hl``) is an empty
    placeholder in both processing variants (verified 2026-07-05); the bulk
    flux is the sled's only latent-flux product and serves as the
    consistency check beside the measured 10 m EC record below.

  SPLASH Kettle Ponds 10 m EC fluxes (Meyers, 30-min, ``KP10mFLX-*.zip``):
    eddy-covariance latent heat flux (``LE_10m``) parsed from the FLX1
    text files directly inside the zips. Quirks handled here: the header
    row is whitespace-mangled (read positionally), the fill value is -999
    for both data and QC columns, and timestamps are MST (UTC-7, no DST)
    and are shifted to UTC to match the TE grid. QC flags 0 and 1 are
    kept; 2 is dropped. The flux converts to a sublimation/deposition
    rate in mm SWE per day via the latent heat of sublimation
    (2.838e6 J/kg); positive = sublimation (upward vapor flux).

  Kettle Ponds CL51 ceilometer cloud products (1-min):
    cloud fraction (share of minutes with any detected layer or full
    obscuration, status >= 1), obscured fraction (status == 4), and the
    median first cloud-base height over cloudy minutes (under status 4 the
    instrument reports vertical visibility there; retained as an effective
    base). For the lagged-LW-down (cloud passage) episode analysis.

Outputs (``data/external/processed/``):
  splash_winter_6h.csv      — datetime, snow_depth_cm, skin_temp_C,
                              air_temp_C, lw_down_wm2, lw_up_wm2,
                              bulk_hl_wm2
  ceilometer_winter_6h.csv  — datetime, cloud_frac, obscured_frac,
                              cloud_base_m, n_minutes
  kp10m_flux_winter_6h.csv  — datetime, le_wm2, subl_mm_day, n_halfhours

Usage:
    python3 build_external_winter.py
    python3 build_external_winter.py --start 2022-11-01 --end 2023-05-31

Author: Matthew Rybecky
"""

from __future__ import annotations

import argparse
import io
import logging
import zipfile
from pathlib import Path
from typing import Dict, List

import netCDF4 as nc
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SPLASH_VARS = {'snow_depth': 'snow_depth_cm',
               'skin_temp_surface': 'skin_temp_C',
               'temp': 'air_temp_C',
               'down_long_hemisp': 'lw_down_wm2',
               'up_long_hemisp': 'lw_up_wm2',
               'bulk_Hl': 'bulk_hl_wm2'}
# QC variable is <var>_qc except for the bulk fluxes, which share one flag.
SPLASH_QC_NAME = {'bulk_Hl': 'bulk_qc'}
QC_KEEP = (0, 1)  # good, caution

# Latent heat of sublimation (J/kg): W/m2 -> mm SWE/day conversion for the
# winter snow surface. 1 kg/m2 = 1 mm water equivalent.
L_SUBLIMATION = 2.838e6
MST_TO_UTC = pd.Timedelta(hours=7)  # MST is UTC-7, no DST
FLX1_FILL = -999.0


# ═════════════════════════════════════════════════════════════════════════
# SPLASH ASFS-30 sledseb
# ═════════════════════════════════════════════════════════════════════════
def read_sledseb_day(path: Path) -> pd.DataFrame:
    """One daily 10-min sledseb file to a QC-masked frame (raw resolution)."""
    ds = nc.Dataset(path)
    t = ds.variables['time']
    times = pd.to_datetime(nc.num2date(t[:], t.units,
                                       only_use_cftime_datetimes=False))
    out: Dict[str, np.ndarray] = {}
    for var, col in SPLASH_VARS.items():
        vals = np.ma.filled(ds.variables[var][:], np.nan).astype(float)
        qc_name = SPLASH_QC_NAME.get(var, f'{var}_qc')
        if qc_name in ds.variables:
            qc = np.ma.filled(ds.variables[qc_name][:], 2)
            vals = np.where(np.isin(qc, QC_KEEP), vals, np.nan)
        out[col] = vals
    ds.close()
    return pd.DataFrame(out, index=times)


def build_splash(winter_dir: Path, start: pd.Timestamp,
                 end: pd.Timestamp) -> pd.DataFrame:
    """All winter sledseb days, QC-masked, resampled to the 6-h TE grid."""
    frames: List[pd.DataFrame] = []
    for path in sorted(winter_dir.glob('sledseb.*.nc')):
        day = pd.Timestamp(path.name.split('.')[-3][:8])
        if start <= day <= end:
            frames.append(read_sledseb_day(path))
    raw = pd.concat(frames).sort_index()
    agg = raw.resample('6h').agg({
        'snow_depth_cm': 'median', 'skin_temp_C': 'mean',
        'air_temp_C': 'mean', 'lw_down_wm2': 'mean', 'lw_up_wm2': 'mean',
        'bulk_hl_wm2': 'mean'})
    logger.info(f"SPLASH: {len(frames)} days -> {len(agg)} 6-h bins "
                f"({agg.index[0]}..{agg.index[-1]})")
    return agg


# ═════════════════════════════════════════════════════════════════════════
# SPLASH Kettle Ponds 10 m EC fluxes (Meyers FLX1)
# ═════════════════════════════════════════════════════════════════════════
def read_flx1_zip(path: Path) -> pd.DataFrame:
    """One KP10mFLX zip to a QC-masked 30-min latent-heat-flux frame (UTC).

    Parameters
    ----------
    path : Path
        ``KP10mFLX-<year>.zip`` containing a single whitespace-separated
        ``KPA<yy>_*.FLX1`` member.

    Returns
    -------
    pd.DataFrame
        Index UTC timestamps; column ``le_wm2`` with -999 fills and
        QC-rejected (flag 2) values as NaN.

    Notes
    -----
    The FLX1 header row does not split cleanly (``time(MST)``), so columns
    are read positionally: 0 date, 1 time, 5 ``LE_10m``, 6 ``qc_LE``.
    Timestamps are MST (UTC-7, no DST) and are shifted to UTC here.
    """
    with zipfile.ZipFile(path) as zf:
        member = next(n for n in zf.namelist() if n.endswith('.FLX1'))
        with zf.open(member) as fh:
            df = pd.read_csv(io.TextIOWrapper(fh), sep=r'\s+', skiprows=1,
                             header=None, usecols=[0, 1, 5, 6],
                             names=['date', 'time', 'le_wm2', 'qc_le'])
    stamps = pd.to_datetime(df['date'] + ' ' + df['time']) + MST_TO_UTC
    le = df['le_wm2'].where(df['le_wm2'] != FLX1_FILL)
    le = le.where(df['qc_le'].isin(QC_KEEP))
    out = pd.DataFrame({'le_wm2': le.to_numpy()}, index=stamps).sort_index()
    logger.info(f"KP10m: {member}: {out['le_wm2'].notna().sum()} good "
                f"half-hours of {len(out)}")
    return out


def build_kp10m(ec_dir: Path, start: pd.Timestamp,
                end: pd.Timestamp) -> pd.DataFrame:
    """All KP10mFLX years, reduced to the 6-h TE grid with sublimation rate.

    ``subl_mm_day`` is the latent heat flux converted through the latent
    heat of sublimation; positive = sublimation (mass loss), negative =
    deposition. ``n_halfhours`` counts the good 30-min values per bin.
    """
    frames = [read_flx1_zip(p) for p in sorted(ec_dir.glob('KP10mFLX-*.zip'))]
    raw = pd.concat(frames).sort_index()
    raw = raw.loc[(raw.index >= start) & (raw.index < end + pd.Timedelta(days=1))]
    grouped = raw['le_wm2'].resample('6h')
    agg = pd.DataFrame({'le_wm2': grouped.mean(),
                        'n_halfhours': grouped.count()})
    agg['subl_mm_day'] = agg['le_wm2'] / L_SUBLIMATION * 86400.0
    agg = agg[['le_wm2', 'subl_mm_day', 'n_halfhours']]
    logger.info(f"KP10m: {len(agg)} 6-h bins "
                f"({agg.index[0]}..{agg.index[-1]})")
    return agg


# ═════════════════════════════════════════════════════════════════════════
# CL51 ceilometer cloud products
# ═════════════════════════════════════════════════════════════════════════
def read_ceilo_day(path: Path) -> pd.DataFrame:
    """One daily 1-min cloud-product file: status and first cloud base."""
    ds = nc.Dataset(path)
    t = ds.variables['time']
    times = pd.to_datetime(nc.num2date(t[:], t.units,
                                       only_use_cftime_datetimes=False))
    status = np.ma.filled(ds.variables['cloud_status'][:], np.nan
                          ).astype(float)
    base1 = np.ma.filled(ds.variables['cloud_data'][:, 0], np.nan
                         ).astype(float)
    ds.close()
    return pd.DataFrame({'status': status, 'base1_m': base1}, index=times)


def build_ceilometer(winter_dir: Path, start: pd.Timestamp,
                     end: pd.Timestamp) -> pd.DataFrame:
    """All winter ceilometer days, reduced to 6-h cloud statistics."""
    frames: List[pd.DataFrame] = []
    for path in sorted(winter_dir.glob('ckp.cl51.cloud_prod.*.nc')):
        day = pd.Timestamp(path.stem.split('.')[-1])
        if start <= day <= end:
            frames.append(read_ceilo_day(path))
    raw = pd.concat(frames).sort_index()
    cloudy = raw['status'] >= 1
    grouped = raw.resample('6h')
    out = pd.DataFrame({
        'cloud_frac': grouped['status'].apply(
            lambda s: float((s >= 1).mean()) if s.notna().any() else np.nan),
        'obscured_frac': grouped['status'].apply(
            lambda s: float((s == 4).mean()) if s.notna().any() else np.nan),
        'cloud_base_m': raw.loc[cloudy, 'base1_m'].resample('6h').median(),
        'n_minutes': grouped['status'].count(),
    })
    logger.info(f"Ceilometer: {len(frames)} days -> {len(out)} 6-h bins "
                f"({out.index[0]}..{out.index[-1]})")
    return out


# ═════════════════════════════════════════════════════════════════════════
# Orchestration
# ═════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Winter 6-h series from the external SPLASH datasets.")
    parser.add_argument('--external-dir', default='data/external')
    parser.add_argument('--start', default='2022-11-01',
                        help='First day to include (YYYY-MM-DD)')
    parser.add_argument('--end', default='2023-05-31',
                        help='Last day to include (YYYY-MM-DD)')
    args = parser.parse_args()

    ext = Path(args.external_dir)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    out_dir = ext / 'processed'
    out_dir.mkdir(exist_ok=True)

    splash = build_splash(ext / 'splash_asfs30' / 'winter', start, end)
    splash.round(3).to_csv(out_dir / 'splash_winter_6h.csv',
                           index_label='datetime')

    ceilo = build_ceilometer(ext / 'ceilometer_cloud' / 'winter', start, end)
    ceilo.round(3).to_csv(out_dir / 'ceilometer_winter_6h.csv',
                          index_label='datetime')

    kp10m = build_kp10m(ext / 'splash_10m_ec', start, end)
    kp10m.round(4).to_csv(out_dir / 'kp10m_flux_winter_6h.csv',
                          index_label='datetime')
    logger.info(f"Wrote {out_dir / 'splash_winter_6h.csv'}, "
                f"{out_dir / 'ceilometer_winter_6h.csv'}, and "
                f"{out_dir / 'kp10m_flux_winter_6h.csv'}")


if __name__ == '__main__':
    main()
