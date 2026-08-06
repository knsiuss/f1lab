# -*- coding: utf-8 -*-
"""
sector.py
~~~~~~~~~
Sector / corner insights from lap-level sector times.

Answers "where does each driver's pace live on the lap": per-driver sector
best, average and consistency, plus each driver's deficit to the session-best
sector time. Sector times come straight from the lap table (Sector1/2/3
columns), so no telemetry is required.

Data Honesty contract: analysis uses only clean laps (out/in laps, Safety Car
windows and inaccurate laps removed via pace.clean_laps_for_analysis), every
figure is labelled estimated, and a confidence level is reported from the
number of clean laps behind each row.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure src/ is importable no matter how this module is reached.
sys.path.insert(0, str(Path(__file__).parent))

from pace import clean_laps_for_analysis, data_quality

ESTIMATED = "estimated"
SECTORS = ["Sector1", "Sector2", "Sector3"]


def _sec_to_float(series: pd.Series) -> pd.Series:
    """Coerce a sector column (Timedelta or seconds) to float."""
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return pd.to_numeric(series, errors="coerce")


def _clean_with_sectors(laps: pd.DataFrame) -> pd.DataFrame:
    clean = clean_laps_for_analysis(laps)
    if clean.empty:
        return clean
    for s in SECTORS:
        if s in clean.columns:
            clean[s] = _sec_to_float(clean[s])
    return clean


def sector_summary(laps: pd.DataFrame,
                   driver: Optional[str] = None) -> pd.DataFrame:
    """Per-driver sector best, mean and consistency (std), estimated.

    One row per driver; each present sector gets Best/Mean/Std columns plus a
    data-quality level and confidence from the clean-lap count.

    Returns:
        DataFrame (possibly empty) with Driver, N, sector stats and the
        Data Honesty labels.
    """
    clean = _clean_with_sectors(laps)
    if clean.empty or 'Driver' not in clean.columns:
        return pd.DataFrame()

    if driver is not None:
        clean = clean[clean['Driver'] == driver]
        if clean.empty:
            return pd.DataFrame()

    rows = []
    for d, grp in clean.groupby('Driver', dropna=False):
        n = len(grp)
        quality = data_quality(n)
        row = {"Driver": d, "N": n,
               "Level": quality["level"], "Confidence": quality["confidence"],
               "Estimated": ESTIMATED}
        for s in SECTORS:
            if s in grp.columns:
                vals = grp[s].dropna()
                if vals.empty:
                    continue
                row[f"{s}Best"] = round(float(vals.min()), 3)
                row[f"{s}Mean"] = round(float(vals.mean()), 3)
                row[f"{s}Std"] = round(float(vals.std()), 4)
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            [f"{SECTORS[0]}Best"] if f"{SECTORS[0]}Best" in out.columns else 'Driver'
        ).reset_index(drop=True)
    return out


def sector_deficits(laps: pd.DataFrame,
                    driver: Optional[str] = None) -> pd.DataFrame:
    """Per-driver deficit to the session-best sector time (seconds, est).

    A negative/large deficit in one sector says "that driver is quick there";
    the column names the fastest driver and the deficit of every other driver
    relative to it. Best sector times are taken from clean laps only.

    Returns:
        DataFrame (possibly empty) with Driver, S1Def/S2Def/S3Def (None when
        the sector is missing or the driver has no clean sector), Quality,
        Confidence, Estimated.
    """
    clean = _clean_with_sectors(laps)
    if clean.empty or 'Driver' not in clean.columns:
        return pd.DataFrame()

    best = {}
    for s in SECTORS:
        if s in clean.columns:
            vals = clean[s].dropna()
            if not vals.empty:
                best[s] = float(vals.min())

    if not best:
        return pd.DataFrame()

    rows = []
    for d, grp in clean.groupby('Driver', dropna=False):
        n = len(grp)
        quality = data_quality(n)
        row = {"Driver": d, "N": n,
               "Level": quality["level"], "Confidence": quality["confidence"],
               "Estimated": ESTIMATED}
        for s in SECTORS:
            if s not in best or s not in grp.columns:
                row[f"{s}Def"] = None
                continue
            vals = grp[s].dropna()
            if vals.empty:
                row[f"{s}Def"] = None
            else:
                row[f"{s}Def"] = round(float(vals.min()) - best[s], 3)
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty and f"{SECTORS[0]}Def" in out.columns:
        out = out.sort_values(f"{SECTORS[0]}Def").reset_index(drop=True)
    return out
