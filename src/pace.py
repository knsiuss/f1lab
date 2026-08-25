# -*- coding: utf-8 -*-
"""
pace.py
~~~~~~~
Normalised pace, long-run and degradation analysis from lap-level data.

Pure pandas logic that operates on FastF1-style lap tables so it can be unit
tested with synthetic data and reused from any UI. Every derived number is
labelled ``estimated`` and carries a data-quality/confidence level derived from
the number of clean laps behind it. When there are too few laps we say
"insufficient data" instead of inventing a value.

The Data Honesty contract is central here:

    * Nothing is fabricated, hidden or overstated.
    * Every fitted/projected value is flagged ``estimated``.
    * Sample size is always reported next to any pace or degradation figure.
    * ``data_quality`` returns a level the caller is expected to surface.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

# Banner applied to every derived metric.
ESTIMATED = "estimated"

# TrackStatus values treated as "all clear" (green). Laps behind a Safety Car
# or VSC are excluded from pace fits: they are not representative of race pace.
# This is FastF1's green convention; override explicitly if it differs.
GREEN_TRACK_STATUS = {1}

# Minimum clean laps before a pace/degradation estimate is shown at all.
MIN_LAPS_FOR_FIT = 2

# Sample-size thresholds for the data-quality labels in data_quality(). These
# define the honesty contract shared across modules: "good" needs at least
# GOOD_LAPS clean laps, "moderate" at least MODERATE_LAPS; anything above
# MIN_LAPS_FOR_FIT is "low", anything below that is "insufficient".
GOOD_LAPS = 15
MODERATE_LAPS = 8

_DEFAULT_QUALITY = {"level": "insufficient", "confidence": "low", "n": 0,
                    "estimated": ESTIMATED}


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _to_seconds(series: pd.Series) -> pd.Series:
    """Coerce a LapTime-like column (Timedelta or numeric seconds) to float."""
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return pd.to_numeric(series, errors="coerce")


def clean_laps_for_analysis(
    laps: pd.DataFrame,
    green_track_status: set = GREEN_TRACK_STATUS,
    drop_out_laps: bool = True,
    drop_in_laps: bool = True,
) -> pd.DataFrame:
    """
    Return the subset of ``laps`` suitable for pace analysis.

    Excludes in/out laps, inaccurate laps, laps behind Safety Car / VSC windows
    and laps without a usable LapTime. ``LapTime`` is returned as float seconds
    under a ``LapTime_s`` column so downstream fits are stable.

    Args:
        laps: FastF1-style lap table (``LapTime``, ``IsOutLap``, ``IsInLap``,
            ``IsAccurate``, ``TrackStatus``, ...).
        green_track_status: seen as green and therefore kept.
        drop_out_laps: drop out-laps when the column is present.
        drop_in_laps: drop in-laps when the column is present.

    Returns:
        Copy of ``laps`` reduced to clean laps with ``LapTime_s`` as seconds.
    """
    out = laps.copy()
    if out.empty:
        out["LapTime_s"] = pd.Series(dtype=float)
        return out

    out["LapTime_s"] = _to_seconds(out["LapTime"])
    mask = out["LapTime_s"].notna()

    if drop_out_laps and "IsOutLap" in out.columns:
        mask &= ~out["IsOutLap"].fillna(False)
    if drop_in_laps and "IsInLap" in out.columns:
        mask &= ~out["IsInLap"].fillna(False)
    if "IsAccurate" in out.columns:
        mask &= out["IsAccurate"].astype(bool).fillna(True)
    if "TrackStatus" in out.columns:
        mask &= pd.to_numeric(out["TrackStatus"], errors="coerce").isin(
            green_track_status
        )

    return out.loc[mask].reset_index(drop=True)


def _clean_pool(laps: pd.DataFrame) -> pd.DataFrame:
    """Clean laps plus the numeric tyre-age column needed for fits."""
    clean = clean_laps_for_analysis(laps)
    if clean.empty:
        return clean
    if "TyreLife" in clean.columns:
        clean["TyreLife"] = pd.to_numeric(clean["TyreLife"], errors="coerce")
    else:
        clean["TyreLife"] = np.nan
    return clean


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

def data_quality(n: int) -> Dict[str, str]:
    """Classify the number of clean laps into a data-quality/confidence level."""
    if n >= GOOD_LAPS:
        level, conf = "good", "high"
    elif n >= MODERATE_LAPS:
        level, conf = "moderate", "medium"
    elif n >= MIN_LAPS_FOR_FIT:
        level, conf = "low", "low"
    else:
        level, conf = "insufficient", "low"
    return {"level": level, "confidence": conf, "n": n, "estimated": ESTIMATED}


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------

def _fit_base_degradation(clean: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Least-squares fit of lap time vs tyre age. None when not enough laps."""
    x = clean["TyreLife"].astype(float).to_numpy()
    y = clean["LapTime_s"].astype(float).to_numpy()
    if len(x) < MIN_LAPS_FOR_FIT or np.all(np.isnan(x)):
        return None

    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < MIN_LAPS_FOR_FIT:
        return None

    slope, intercept = np.polyfit(x, y, 1)
    yhat = intercept + slope * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "base_pace": float(intercept),
        "deg_per_lap": float(slope),
        "r2": float(r2),
        "n": int(len(x)),
    }



def normalised_pace_by_stint(
    laps: pd.DataFrame, group_by_compound: bool = True
) -> pd.DataFrame:
    """
    Per-driver/stint normalised (tyre-age removed) pace.

    For each stint, when enough clean laps exist, ``BasePace`` is the fitted
    fresh-tyre lap time (estimated). Otherwise only the raw mean is shown and
    the quality is downgraded. ``DegPerLap`` is the estimated degradation.

    Returns:
        DataFrame with columns Driver, Stint, Compound, N, MeanPace,
        BasePace (est), DegPerLap (est), Level, Confidence.
    """
    clean = _clean_pool(laps)
    if clean.empty:
        return pd.DataFrame()

    keys = ["Driver", "Stint"] if "Stint" in clean.columns else ["Driver"]
    if group_by_compound and "Compound" in clean.columns:
        keys.append("Compound")

    rows = []
    for (group_key, grp) in clean.groupby(keys, dropna=False):
        if isinstance(group_key, tuple):
            row = dict(zip(keys, group_key))
        else:
            row = {keys[0]: group_key}
        mean_pace = float(grp["LapTime_s"].mean())
        fit = _fit_base_degradation(grp)
        n = len(grp)
        quality = data_quality(n)
        if fit is not None:
            base = round(fit["base_pace"], 3)
            deg = round(fit["deg_per_lap"], 4)
        else:
            base, deg = None, None
        rows.append({
            **row,
            "N": n,
            "MeanPace": round(mean_pace, 3),
            "BasePace": base,
            "DegPerLap": deg,
            "Level": quality["level"],
            "Confidence": quality["confidence"],
            "Estimated": quality["estimated"],
        })
    return pd.DataFrame(rows)



def long_run_quality(laps: pd.DataFrame, min_stint: int = 10) -> pd.DataFrame:
    """
    Identify representative long runs per driver.

    One representative stint per driver+compound (the longest one with enough
    clean laps). Reports its length, mean pace and an estimated degradation,
    all with a data-quality level. ``min_stint`` controls how long a run must be
    to count as a "long run".

    Returns:
        DataFrame of candidate long runs, one row per driver+compound.
    """
    clean = _clean_pool(laps)
    if clean.empty or "Stint" not in clean.columns:
        return pd.DataFrame()

    keys = ["Driver", "Stint"]
    if "Compound" in clean.columns:
        keys.append("Compound")

    rows = []
    for (key, grp) in clean.groupby(keys, dropna=False):
        n = len(grp)
        if n < min_stint:
            continue
        row = dict(zip(keys, key)) if isinstance(key, tuple) else {keys[0]: key}
        fit = _fit_base_degradation(grp)
        quality = data_quality(n)
        row.update({
            "Length": n,
            "MeanPace": round(float(grp["LapTime_s"].mean()), 3),
            "BasePace": round(fit["base_pace"], 3) if fit else None,
            "DegPerLap": round(fit["deg_per_lap"], 4) if fit else None,
            "Level": quality["level"],
            "Confidence": quality["confidence"],
            "Estimated": ESTIMATED,
        })
        rows.append(row)
    return pd.DataFrame(rows)