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
SECTOR_SUFFIXES = ["1", "2", "3"]


def _sector_source(clean: pd.DataFrame) -> dict:
    """Map each sector index to the column that actually holds its time.

    FastF1 lap tables name these ``Sector1``/``Sector2``/``Sector3`` on some
    builds and ``Sector1Time``/``Sector2Time``/``Sector3Time`` on others, so we
    resolve the present spelling once rather than hardcode one.
    """
    cols = {}
    for s in SECTOR_SUFFIXES:
        for cand in (f"Sector{s}", f"Sector{s}Time"):
            if cand in clean.columns:
                cols[s] = cand
                break
    return cols


def _sec_to_float(series: pd.Series) -> pd.Series:
    """Coerce a sector column (Timedelta or seconds) to float."""
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return pd.to_numeric(series, errors="coerce")


def _clean_with_sectors(laps: pd.DataFrame) -> pd.DataFrame:
    clean = clean_laps_for_analysis(laps)
    if clean.empty:
        return clean
    for col in _sector_source(clean).values():
        clean[col] = _sec_to_float(clean[col])
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
    src = _sector_source(clean)
    for d, grp in clean.groupby('Driver', dropna=False):
        n = len(grp)
        quality = data_quality(n)
        row = {"Driver": d, "N": n,
               "Level": quality["level"], "Confidence": quality["confidence"],
               "Estimated": ESTIMATED}
        for s, col in src.items():
            vals = grp[col].dropna()
            if vals.empty:
                continue
            row[f"Sector{s}Best"] = round(float(vals.min()), 3)
            row[f"Sector{s}Mean"] = round(float(vals.mean()), 3)
            row[f"Sector{s}Std"] = round(float(vals.std()), 4)
        rows.append(row)

    out = pd.DataFrame(rows)
    first_key = f"Sector{SECTOR_SUFFIXES[0]}Best"
    if not out.empty:
        out = out.sort_values(
            [first_key] if first_key in out.columns else 'Driver'
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

    src = _sector_source(clean)
    best = {}
    for s, col in src.items():
        vals = clean[col].dropna()
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
        for s, col in src.items():
            if s not in best:
                row[f"Sector{s}Def"] = None
                continue
            vals = grp[col].dropna()
            if vals.empty:
                row[f"Sector{s}Def"] = None
            else:
                row[f"Sector{s}Def"] = round(float(vals.min()) - best[s], 3)
        rows.append(row)

    out = pd.DataFrame(rows)
    first_def = f"Sector{SECTOR_SUFFIXES[0]}Def"
    if not out.empty and first_def in out.columns:
        out = out.sort_values(first_def).reset_index(drop=True)
    return out
