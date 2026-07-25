#!/usr/bin/env python3
"""
pub_labels.py — Human-readable labels for publication outputs.

Maps candidate names (``base__tK``) and targets to readable text for figures and
tables. Keeps presentation strings in one place so figures and tables stay
consistent.

Author: Matthew Rybecky
"""

from pub_config import base_of

# Full base-variable labels (figure axes, table cells).
BASE_LABELS = {
    'met_temp': 'Temperature',
    'met_rh': 'Relative humidity',
    'met_pressure': 'Pressure',
    # Validation-only base (Figure 2 Clausius-Clapeyron control); not in
    # the 16-variable analysis set.
    'met_vapor_pressure': 'Vapor pressure',
    'met_wspd': 'Wind speed',
    'met_wdir': 'Wind direction',
    'rad_sw_down': 'SW' + '↓',
    'rad_lw_down': 'LW' + '↓',
    'rad_lw_up': 'LW' + '↑',
    'rad_net': 'Net radiation',
    'flux_3m_h2o': 'H' + '₂' + 'O flux 3m',
    'flux_10m_h2o': 'H' + '₂' + 'O flux 10m',
    'flux_20m_h2o': 'H' + '₂' + 'O flux 20m',
    # CO2 molar density (state variable from the SOS flux tower), NOT an
    # EC flux: pipeline maps sos 'co2_density' -> flux_{h}_co2_density
    # (confirmed 2026-07-08). The 'flux_' prefix only records the tower.
    'flux_3m_co2_density': 'CO' + '₂' + ' density 3m',
    'flux_10m_co2_density': 'CO' + '₂' + ' density 10m',
    'flux_20m_co2_density': 'CO' + '₂' + ' density 20m',
    # Pressure levels (hPa), NOT isentropic K despite the 'isen_' code
    # prefix: thesis ch3 and the GPH values (~11.1 km, ~5.7 km) confirm.
    'isen_225_gph': 'GPH 225 hPa',
    'isen_500_gph': 'GPH 500 hPa',
    'isen_825_gph': 'GPH 825 hPa',
}

# Short labels for the participation-ribbon rows.
BASE_SHORT = {
    'met_temp': 'Temp', 'met_rh': 'RH', 'met_pressure': 'Press',
    'met_vapor_pressure': 'VP',
    'met_wspd': 'Wspd', 'met_wdir': 'Wdir',
    'rad_sw_down': 'SW' + '↓', 'rad_lw_down': 'LW' + '↓',
    'rad_lw_up': 'LW' + '↑', 'rad_net': 'Rnet',
    'flux_3m_h2o': 'H2O 3m', 'flux_10m_h2o': 'H2O 10m', 'flux_20m_h2o': 'H2O 20m',
    'flux_3m_co2_density': 'CO2 3m', 'flux_10m_co2_density': 'CO2 10m',
    'flux_20m_co2_density': 'CO2 20m',
    'isen_225_gph': 'GPH225', 'isen_500_gph': 'GPH500', 'isen_825_gph': 'GPH825',
}

# Target labels.
TARGET_LABELS = {
    'd_excess': 'd-excess',
    'dD': 'δD',
    'H2O_ppm': 'H' + '₂' + 'O (ppm)',
}

# Interaction-category grayscale (B&W safe) and display names.
CATEGORY_GRAY = {'synergistic': 0.20, 'redundant': 0.55, 'obfuscating': 0.85}
CATEGORY_NAME = {'synergistic': 'Synergistic', 'redundant': 'Redundant',
                 'obfuscating': 'Obfuscating'}


def tau_of(name: str) -> int:
    """Return the integer tau encoded in a ``base__tK`` candidate name."""
    return int(name.rsplit('__t', 1)[1])


def candidate_label(name: str, short: bool = False) -> str:
    """Readable label for one candidate, e.g. 'Wind speed (tau=1)'."""
    base = base_of(name)
    text = (BASE_SHORT if short else BASE_LABELS).get(base, base)
    return f"{text} (τ={tau_of(name)})"


def combo_label(combo_key: str, short: bool = False) -> str:
    """Readable label for a '+'-joined combination of candidates."""
    return ' + '.join(candidate_label(p, short=short)
                      for p in combo_key.split('+'))


def target_label(target: str) -> str:
    """Readable label for a target variable."""
    return TARGET_LABELS.get(target, target)
