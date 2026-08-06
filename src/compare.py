# -*- coding: utf-8 -*-
"""
compare.py
~~~~~~~~~~
Session-to-session pace comparison.

Answers "how did this driver/stint change between two sessions (or two
drivers within a session)": both lap tables are normalised for tyre age with
pace.normalised_pace_by_stint and then joined on matching drivers and stints,
producing a per-row pace delta. A negative delta means session A was faster.

Data Honesty contract: every delta is estimated, derived from the normalised
base pace (fresh-tyre lap time) which itself carries a confidence label; rows
without enough clean laps are dropped rather than compared on noise.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure src/ is importable no matter how this module is reached.
sys.path.insert(0, str(Path(__file__).parent))

from pace import normalised_pace_by_stint

ESTIMATED = "estimated"


def compare_sessions(laps_a: pd.DataFrame, laps_b: pd.DataFrame,
                     driver: Optional[str] = None) -> pd.DataFrame:
    """
    Per-driver/stint pace delta between two sessions.

    Both inputs are normalised for tyre age independently, then joined on
    (Driver, Stint). ``Delta`` is ``BasePace_A - BasePace_B`` in seconds per
    lap: negative = session A faster, positive = session B faster. Rows where
    either session lacks a fit (too few clean laps) are excluded.

    Args:
        laps_a, laps_b: FastF1-style lap tables for the two sessions.
        driver: restrict the comparison to one driver when given.

    Returns:
        DataFrame with Driver, Stint, BasePace_A, BasePace_B, Delta (est),
        Level, Confidence, Estimated.
    """
    na = normalised_pace_by_stint(laps_a)
    nb = normalised_pace_by_stint(laps_b)
    if na.empty or nb.empty:
        return pd.DataFrame()

    for df_ in (na, nb):
        df_['BasePace'] = pd.to_numeric(df_['BasePace'], errors='coerce')
    na = na.dropna(subset=['BasePace'])
    nb = nb.dropna(subset=['BasePace'])
    if na.empty or nb.empty:
        return pd.DataFrame()

    keys = ['Driver']
    if 'Stint' in na.columns and 'Stint' in nb.columns:
        keys.append('Stint')

    if driver is not None:
        na = na[na['Driver'] == driver]
        nb = nb[nb['Driver'] == driver]

    merged = na.merge(
        nb, on=keys, suffixes=('_A', '_B'),
        how='inner', sort=False,
    )
    if merged.empty:
        return pd.DataFrame()

    rows = []
    for _, r in merged.iterrows():
        base_a = float(r['BasePace_A'])
        base_b = float(r['BasePace_B'])
        rows.append({
            **{k: r[k] for k in keys},
            'BasePace_A': round(base_a, 3),
            'BasePace_B': round(base_b, 3),
            'Delta': round(base_a - base_b, 3),
            'Level': r.get('Level_A', 'low'),
            'Confidence': r.get('Confidence_A', 'low'),
            'Estimated': ESTIMATED,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values('Delta').reset_index(drop=True)
    return out
