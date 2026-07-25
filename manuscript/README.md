# manuscript/ — Scripts of Record

**Purpose:** the exact code that generated the figures, tables, and statistics
in the accompanying JGR: Atmospheres manuscript. These scripts are provenance,
preserved with minimal modification from the analysis repository (TE_V1.0.0).
The only edits are documented in-file: a two-line `sys.path` bootstrap to
`../core` in `pub_stage2.py` and `pub_history_sensitivity.py`, and two
repo-layout path adjustments in `pub_history_sensitivity.py`. The math is
unchanged everywhere.

**Run convention:** every command executes from the repository root, e.g.

```bash
cd te-explorer
python3 manuscript/pub_fig_overview.py
```

so that the relative paths `data/...` and `publication_output/...` resolve
exactly as they did in the original analysis tree. Outputs land in
`publication_output/`.

## Publication settings (frozen)

All manuscript results use: 30-day rolling windows, history length h=1,
KSG k=3 (base 2, bits), 2000 IAAFT surrogates per window, 95th-percentile
significance band, analysis truncated at 2023-05-01. These are encoded in
`pub_config.py`; no script hand-types them.

## Manuscript artifacts

| Artifact | Script | Command (from repo root) | Runtime |
|---|---|---|---|
| Figure 1 (isotope overview) | `pub_fig_overview.py` | `python3 manuscript/pub_fig_overview.py` | <1 min |
| Table 1 (variables) | `pub_table1.py` | `python3 manuscript/pub_table1.py` | <1 min |
| Figure 2 (Stefan-Boltzmann validation) | `pub_validation.py` | `python3 manuscript/pub_validation.py` (reads the shipped Stage 2 CSV; full regeneration: `python3 manuscript/pub_stage2.py --target met_temp --entities rad_lw_up__t0 --n-cores 8` first, ~12-24 min) | <1 min |
| Figure 3 (single drivers) + Figure 4 (LW-down lag sweep) | `pub_selected_figures.py` | `python3 manuscript/pub_selected_figures.py` (both figures come from the shipped Stage 2 CSVs; the script's per-target selection library additionally needs frozen parquet layers not shipped here and skips itself with a logged warning) | ~1 min |
| Figure 5 (sublimation corroboration) | `pub_fig_sublimation.py` | `python3 manuscript/pub_fig_sublimation.py` (reads shipped `data/external/processed/` CSVs) | ~1 min |
| Table 3 (results) | `pub_table_results.py` | `python3 manuscript/pub_table_results.py` | <1 min |
| Table S1 (composites) | `pub_table_s1.py` | `python3 manuscript/pub_table_s1.py` | <1 min |
| Isotope coverage table | `pub_data_coverage.py` | `python3 manuscript/pub_data_coverage.py` | <1 min |
| Transition statistics (March 13 melting-point recast) | `pub_transition_stats.py` | `python3 manuscript/pub_transition_stats.py` | <1 min |
| SI history sensitivity (h=1..4) | `pub_history_sensitivity.py` | `python3 manuscript/pub_history_sensitivity.py` | minutes-hours (recomputes TE) |

## The Stage 2 engine

`pub_stage2.py` is the targeted rolling TE/JTE + IAAFT runner behind every
results figure and table. The frozen per-entity outputs of record ship in
`publication_output/stage2/<target>/stage2_<entity>.csv`, so all figures and
tables above rebuild in seconds without recomputation. To regenerate any
entity from scratch (~12-24 min per entity on 8 cores):

```bash
python3 manuscript/pub_stage2.py --target d_excess \
    --entities rad_lw_down__t0 rad_lw_down__t1 rad_lw_down__t2 --n-cores 8
```

## Data preparation chain

The shipped analysis inputs were built as:
`data/final_6hr.csv` (real-unit record) → beta normalization →
`data/final_6hr_beta.csv` → `build_circular_data.py` (wind as sin/cos) →
`data/final_6hr_beta_circular.csv` → `build_augmented_data.py`
(pre-lagged columns) → `data/final_6hr_beta_circular_lagged.csv`.

`build_external_winter.py` is provenance-only: it built
`data/external/processed/*.csv` from raw SPLASH/ARM/ceilometer archives
(5+ GB) that are not shipped. Its processed outputs are included, so
`pub_fig_sublimation.py` runs without it.

## Support modules (imported, not run directly)

`pub_config.py` (frozen publication configuration), `pub_style.py`
(figure style), `pub_labels.py` (variable labels), `pub_driver_series.py`,
`pub_combo_stats.py`, `pub_combo_metrics.py`, `pub_period_attribution.py`,
and `pub_selected_figures.py` double as an import library for
`pub_validation.py` and `pub_fig_sublimation.py`. `pub_driver_series.py`,
`pub_combo_stats.py`, `pub_combo_metrics.py`, and `pub_period_attribution.py`
ship only as import dependencies of the kept scripts.
