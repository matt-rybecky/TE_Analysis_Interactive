#!/usr/bin/env python3
"""
pub_config.py — Frozen specification for the publication composite JTE run.

Single source of truth for the manuscript's transfer-entropy analysis: the
candidate input groups (each with its own tau set), the three targets, and the
per-target input exclusions. Candidate names are ``base__tK`` so the base
variable is recoverable by splitting on ``__t`` (drives the distinct-base
combination rule and table grouping).

Frozen spec (revised 2026-06-27):
  - Excluded from all inputs: met_pressure (same instrument as the isotopes;
    biased) and isen_825_gph (questionable in complex mountain terrain).
  - H2O_ppm additionally excludes met_temp and met_rh (deterministic
    Clausius-Clapeyron relation, used for validation, not a result).
  - Fluxes (H2O and CO2 density at 3/10/20 m) use tau {0,1}; wind and GPH keep
    tau {0,1,2}; local state uses tau {0,1}.

Changing the inputs, tau sets, targets, window, k, or surrogate count is the
only thing that requires re-running the calculation.

Author: Matthew Rybecky
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

# ═════════════════════════════════════════════════════════════════════════
# Candidate groups: (base variables, tau set). tau in 6-hour steps.
# ═════════════════════════════════════════════════════════════════════════
GROUPS: List[Tuple[List[str], Tuple[int, ...]]] = [
    # Local thermodynamic/radiative state — fast-acting.
    (['met_temp', 'met_rh', 'rad_sw_down', 'rad_lw_down', 'rad_lw_up',
      'rad_net'], (0, 1)),
    # Wind / transport — lag-uncertain over a longer window.
    (['met_wspd', 'met_wdir'], (0, 1, 2)),
    # Eddy-covariance fluxes (H2O and CO2 density) at 3/10/20 m.
    (['flux_3m_h2o', 'flux_10m_h2o', 'flux_20m_h2o',
      'flux_3m_co2_density', 'flux_10m_co2_density', 'flux_20m_co2_density'],
     (0, 1)),
    # Far-field air-mass advection (isentropic geopotential height).
    (['isen_225_gph', 'isen_500_gph'], (0, 1, 2)),
]

# Vector (multi-column) bases: a single logical input backed by several data
# columns. Circular wind direction enters as sin/cos so the KSG estimator sees a
# continuous 2D source instead of a discontinuous linear angle (the 0/360 wrap).
# Data columns are f"{base}_{component}" (e.g. met_wdir_sin, met_wdir_cos),
# built by build_circular_data.py.
VECTOR_BASES: Dict[str, Tuple[str, ...]] = {'met_wdir': ('sin', 'cos')}

# Targets (isotope/humidity); never used as inputs.
TARGETS: List[str] = ['d_excess', 'dD', 'H2O_ppm']

# Per-target base-variable exclusions (deterministic / circular sources).
TARGET_EXCLUDE: Dict[str, Tuple[str, ...]] = {
    'H2O_ppm': ('met_temp', 'met_rh'),
}

TAU_DELIM = '__t'


@dataclass
class PublicationConfig:
    """
    Configuration for the frozen publication composite JTE run.

    Parameters
    ----------
    data_file_base : str
        Beta-normalized source CSV (targets read from here, unchanged).
    data_file_lagged : str
        Augmented CSV with duplicated ``base__tK`` candidate columns.
    output_base : str
        Root directory for per-target artifact subdirectories.
    time_col : str
        Datetime column name.
    window_days : int
        Rolling window length in days.
    history_length : int
        TE history length h.
    surrogate_type : str
        Surrogate algorithm for Phase 2.
    n_surrogates : int
        IAAFT surrogates per window (Phase 2).
    max_combo_size : int
        Maximum combination size (2- and 3-way).
    targets : list of str
        Target variable names.

    Notes
    -----
    The KSG neighbor count k is the NPEET default (k=3). Logarithm base 2 (bits).
    """

    data_file_base: str = 'data/final_6hr_beta_circular.csv'
    data_file_lagged: str = 'data/final_6hr_beta_circular_lagged.csv'
    output_base: str = 'publication_output'
    time_col: str = 'time'
    # All post-processing (stats, tables, figures) is truncated to before this
    # date: the late-spring snowmelt/diurnal regime takes over after May 1, so
    # the analysis isolates the local winter signal (sublimation and its drivers).
    analysis_end: str = '2023-05-01'
    window_days: int = 30
    history_length: int = 1
    surrogate_type: str = 'iaaft'
    n_surrogates: int = 2000
    max_combo_size: int = 3
    targets: List[str] = field(default_factory=lambda: list(TARGETS))

    def __post_init__(self) -> None:
        if self.window_days <= 0:
            raise ValueError("window_days must be positive")
        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")
        if self.n_surrogates < 1:
            raise ValueError("n_surrogates must be >= 1")
        if self.max_combo_size < 2:
            raise ValueError("max_combo_size must be >= 2")

    @staticmethod
    def _excluded(target: Optional[str]) -> set:
        return set(TARGET_EXCLUDE.get(target, ())) if target else set()

    def candidates(self, target: Optional[str] = None) -> List[str]:
        """Candidate names ``base__tK`` for a target (None = full union)."""
        excl = self._excluded(target)
        names: List[str] = []
        for bases, taus in GROUPS:
            for base in bases:
                if base in excl:
                    continue
                names.extend(f"{base}{TAU_DELIM}{t}" for t in taus)
        return names

    def tau_map(self, target: Optional[str] = None) -> Dict[str, int]:
        """Map each candidate name to its tau for a target."""
        return {n: int(n.rsplit(TAU_DELIM, 1)[1])
                for n in self.candidates(target)}

    def vector_bases(self) -> Dict[str, List[str]]:
        """Vector bases mapped to their component list (JSON-serializable)."""
        return {b: list(c) for b, c in VECTOR_BASES.items()}

    def column_map(self, target: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Map each logical candidate ``base__tK`` to its data column(s).

        Scalar candidates map to themselves; vector candidates (e.g. wind
        direction) map to one ``base_component__tK`` column per component.
        """
        cmap: Dict[str, List[str]] = {}
        for name in self.candidates(target):
            base = base_of(name)
            suffix = name[len(base):]  # '__tK'
            if base in VECTOR_BASES:
                cmap[name] = [f"{base}_{c}{suffix}" for c in VECTOR_BASES[base]]
            else:
                cmap[name] = [name]
        return cmap

    def data_columns(self, target: Optional[str] = None) -> List[str]:
        """All ``base__tK`` / ``base_component__tK`` columns the augmented file holds."""
        cols: List[str] = []
        for expanded in self.column_map(target).values():
            cols.extend(expanded)
        return cols

    def source_columns(self, target: Optional[str] = None) -> List[str]:
        """
        Unique underlying (un-lagged) data columns the augmented file copies
        from, e.g. ``met_temp``, ``met_wdir_sin``, ``met_wdir_cos``.
        """
        seen: List[str] = []
        for col in self.data_columns(target):
            src = col.rsplit(TAU_DELIM, 1)[0]
            if src not in seen:
                seen.append(src)
        return seen

    def base_columns(self, target: Optional[str] = None) -> List[str]:
        """Unique base variables for a target, in group order."""
        excl = self._excluded(target)
        seen: List[str] = []
        for bases, _ in GROUPS:
            for base in bases:
                if base not in excl and base not in seen:
                    seen.append(base)
        return seen

    def distinct_base_combos(self,
                             target: Optional[str] = None) -> List[Tuple[str, ...]]:
        """
        All 2- and 3-way candidate combinations whose members are distinct base
        variables (never the same variable at two lags in one combo).
        """
        names = sorted(self.candidates(target))
        combos: List[Tuple[str, ...]] = []
        for size in range(2, min(self.max_combo_size, len(names)) + 1):
            for combo in combinations(names, size):
                if len({base_of(n) for n in combo}) == size:
                    combos.append(combo)
        return combos


def base_of(name: str) -> str:
    """Return the base variable name for a ``base__tK`` candidate."""
    return name.rsplit(TAU_DELIM, 1)[0]


if __name__ == '__main__':
    cfg = PublicationConfig()
    print(f"Union candidates: {len(cfg.candidates())}")
    for tgt in cfg.targets:
        print(f"  {tgt:9}: {len(cfg.candidates(tgt))} candidates, "
              f"{len(cfg.base_columns(tgt))} bases, "
              f"{len(cfg.distinct_base_combos(tgt))} combos")
