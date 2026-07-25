# te-explorer

Interactive explorer for rolling-window transfer entropy in time series,
released to accompany a research manuscript on phase-change
signatures in water vapor isotopes (SAIL/SPLASH campaigns, East River
watershed, Colorado, winter 2022-23).

The application implements exactly the mathematics used in the manuscript:

- **Rolling-window transfer entropy and joint transfer entropy** with the
  Kraskov-Stoegbauer-Grassberger (KSG) k-nearest-neighbor estimator
  (k=3, base 2, results in bits), including differential lags per source
  and vector-valued sources (sin/cos pairs for directional variables).
- **IAAFT surrogate significance testing** (iterative amplitude-adjusted
  Fourier transform), per window, drawn as a 95th-percentile band.
- **The data preparation of record**: Gaussian-weighted temporal alignment
  onto a uniform grid, entropy-calibrated beta normalization (robust
  z-score scaled by 2^(JMI - H)), and shared-scale circular encoding of
  directional variables.

Every analysis parameter is adjustable in the GUI and defaults to the
publication value: 30-day windows, history length h=1, 2000 IAAFT
surrogates per window. The KSG neighbor count k=3 is fixed in the engine.
The app is dataset-agnostic: load any CSV with a datetime column and
numeric variables.

## Install

From the repository root, in a fresh virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(Windows: `.venv\Scripts\activate` instead of the `source` line.)

tkinter ships with Python but is not pip-installable; on Debian/Ubuntu
install the system package `python3-tk` if `python3 -c "import tkinter"`
fails. The KSG estimator (NPEET, MIT license) is vendored at `core/NPEET`
and needs no installation itself; its scikit-learn dependency is covered
by `requirements.txt`. Requires Python 3.9 or newer.

## Quickstart

```bash
python3 -m te_explorer
```

Three tabs, in workflow order:

1. **Data Preparation** — load one or more CSVs (each may have its own
   native resolution and datetime column), align and merge them onto one
   uniform time grid (choose the interval; mark circular and flux
   columns), apply the beta normalization (pick the calibration target
   and inputs; the log reports JMI, H, and the scale factor), optionally
   encode directional columns as sin/cos pairs, and save the
   analysis-ready file into `data/`.
2. **Transfer Entropy Analysis** — pick a file from `data/`, a target and
   one or more source variables with their lags, and compute the rolling
   TE/JTE series. Results can be displayed in bits or as a percentage of
   the target's per-window entropy.
3. **IAAFT Significance** — rerun a chosen entity with per-window IAAFT
   surrogates and draw the significance band under the TE curve; export
   the series and band as CSV or figures.

The shipped `data/` files are the manuscript's analysis inputs, so the
explorer works out of the box; `data/final_6hr_beta_circular.csv` is the
publication input.

## Tests

```bash
python3 tests/test_prep.py
```

Headless test of the data-preparation pipeline (no GUI): synthetic
unaligned multi-file datasets are aligned, merged, beta-normalized, and
circular-encoded, with assertions on grid coverage, gap preservation, and
calibration sanity. Runs in a few seconds. The tests are also pytest-compatible if you
install pytest separately (not a runtime dependency).

## Reproducing the manuscript

`manuscript/` contains the exact scripts of record behind every figure and
table, run from the repository root; see `manuscript/README.md` for the
per-artifact commands. The frozen per-window results of record ship in
`publication_output/stage2/`, so all figures and tables rebuild in seconds
without recomputation.

## Repository map

- `te_explorer/` — the application (GUI in `gui/`, data preparation in
  `prep/`, configuration in `config.py`).
- `core/` — the computation engines of record (`TE_Calculator.py`,
  `TE_Surrogate.py`), byte-identical to the analysis repository, plus the
  vendored NPEET estimator.
- `manuscript/` — scripts of record for the manuscript artifacts.
- `data/` — analysis inputs (see `manuscript/README.md` for the
  preparation chain and provenance).
- `publication_output/stage2/` — frozen per-entity rolling TE/JTE + IAAFT
  results of record.

## Data provenance

Isotope record: TWVIA water vapor isotopes (dD, d18O, derived d-excess),
calibrated VSMOW-SLAP, SAIL campaign. Meteorology and radiation: ARM SAIL.
Fluxes: SoS/SAIL eddy covariance. External corroboration
(`data/external/processed/`): SPLASH ASFS-30 surface energy balance and
NOAA ceilometer products on the 6-hour analysis grid. Raw external
archives (5+ GB) are not shipped; `manuscript/build_external_winter.py`
documents their processing.

## Citation

See `CITATION.cff`. Please cite the accompanying manuscript (DOI pending)
and this software release.

## License

MIT (see `LICENSE`). The vendored NPEET package retains its own MIT
license at `core/NPEET/LICENSE.md`.
