# -*- coding: utf-8 -*-
"""
pitwindow.py
~~~~~~~~~~~~
Pit-window radar: the competitive lap window for a pit stop.

For each driver we take their fitted fresh-tyre base pace and per-lap
degradation (from pace.normalised_pace_by_stint, i.e. estimated from their own
clean laps) and grid-search the stop lap(s) that minimise total race time under
the exact same strategy engine used everywhere else (whatif.simulate_scenario).
The result is an optimal stop lap plus a "radar" window of laps within a small
time tolerance of that optimum.

Data Honesty contract: everything is estimated from the driver's own clean laps
where available, labelled estimated, and carries a confidence label from the
sample size. Drivers with no degradation fit fall back to model defaults at low
confidence rather than being silently assumed perfect or assumed bad.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

# Ensure src/ is importable no matter how this module is reached.
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_CONFIG
from whatif import Scenario, simulate_scenario
from pace import clean_laps_for_analysis, data_quality, normalised_pace_by_stint

ESTIMATED = "estimated"

DEFAULT_COMPOUND = "DEFAULT"

# Below this fitted per-lap degradation a pit window is not meaningful: with
# (near-)no ageing penalty the total-time curve is flat, so any stop lap is
# "optimal" and an early stop would be an artefact of noise. We report no window
# (the row still appears, flagged) instead of inventing an early stop lap.
MIN_MEANINGFUL_DEG = 0.02


def _resolve_total_laps(laps: pd.DataFrame,
                        total_laps: Optional[int]) -> int:
    """Race length to simulate: explicit arg, else the lap table's max lap."""
    if total_laps:
        return int(total_laps)
    if "LapNumber" in laps.columns:
        mx = laps["LapNumber"].dropna()
        if not mx.empty:
            return int(mx.max())
    return int(MODEL_CONFIG["default_total_laps"])


def _driver_estimates(laps: pd.DataFrame) -> Tuple[Dict[str, float],
                                                   Dict[str, float],
                                                   Dict[str, int]]:
    """Per-driver fitted fresh-tyre base pace and degradation, plus lap counts.

    Returns ``(base_map, deg_map, counts)`` keyed by Driver: the median fitted
    fresh-tyre lap time and median per-lap degradation, and each driver's clean
    lap count used for the confidence label.
    """
    base_map, deg_map = {}, {}
    fits = normalised_pace_by_stint(laps)
    if fits is not None and not fits.empty:
        for drv, sub in fits.groupby("Driver", dropna=False):
            b = pd.to_numeric(sub["BasePace"], errors="coerce").dropna()
            d = pd.to_numeric(sub.get("DegPerLap"), errors="coerce").dropna()
            if not b.empty:
                base_map[drv] = float(b.median())
            if not d.empty:
                deg_map[drv] = float(d.median())
    clean = clean_laps_for_analysis(laps)
    counts = clean["Driver"].value_counts().to_dict() if not clean.empty else {}
    return base_map, deg_map, counts


def pit_window(laps: pd.DataFrame,
               total_laps: Optional[int] = None,
               n_stops: int = 1,
               deg_rates: Optional[Dict[str, float]] = None,
               pit_loss: Optional[float] = None,
               fuel_gain: Optional[float] = None,
               tolerance: float = 1.0) -> pd.DataFrame:
    """
    Per-driver pit-stop radar window (module: :mod:`pitwindow`).

    Grid-searches stop-lap positions for an ``n_stops``-stop strategy, reusing
    the exact simulate_scenario engine, and reports each driver's optimal stop
    lap plus the range of stop laps that fall within ``tolerance`` seconds of
    the optimum.

    Args:
        laps: FastF1-style lap table (Driver, LapNumber, LapTime, ...).
        total_laps: simulated race length; defaults to the table's max lap.
        n_stops: number of pit stops in the modelled strategy.
        deg_rates / pit_loss / fuel_gain: strategy-engine knobs; default from
            config.MODEL_CONFIG (as everywhere else).
        tolerance: window spread (seconds) around the objective optimum.

    Returns:
        DataFrame (possibly empty) with Driver, OptimalLap, WindowStart,
        WindowEnd, BasePace, DegPerLap, Level, Confidence, Estimated.
    """
    deg_rates = deg_rates or MODEL_CONFIG["degradation_rates"]
    pit_loss = pit_loss if pit_loss is not None else MODEL_CONFIG["pit_loss_sec"]
    if fuel_gain is None:
        fuel_gain = float(MODEL_CONFIG["fuel_gain_per_lap"])

    if laps is None or laps.empty or "Driver" not in laps.columns:
        return pd.DataFrame()

    N = _resolve_total_laps(laps, total_laps)
    if N <= n_stops + 1:
        return pd.DataFrame()

    base_map, deg_map, counts = _driver_estimates(laps)
    if not base_map:
        return pd.DataFrame()

    search_laps = range(1, N - n_stops + 1)
    compounds = [DEFAULT_COMPOUND] * (n_stops + 1)
    default_deg = deg_rates["DEFAULT"]

    rows = []
    for drv in sorted(base_map.keys()):
        base = base_map[drv]
        deg = deg_map.get(drv)
        pool = {DEFAULT_COMPOUND: deg if deg is not None else default_deg}
        quality = data_quality(counts.get(drv, 0))

        totals = []
        for stop_lap in search_laps:
            scenario = Scenario(
                name=f"radar@{stop_lap}",
                compounds=compounds,
                stop_laps=[stop_lap] * n_stops,
            )
            res = simulate_scenario(
                scenario, N,
                base_lap_time=base, deg_rates=pool,
                fuel_gain=fuel_gain, pit_loss=pit_loss,
            )
            tot = (res or {}).get("total_time")
            if tot is not None:
                totals.append((stop_lap, float(tot)))

        if not totals:
            continue

        # Data-honesty gate: without meaningful tyre ageing there is no real pit
        # window, so we report the driver with a blank window rather than an
        # artefact early stop from a flat total-time curve.
        meaningful = deg is not None and deg >= MIN_MEANINGFUL_DEG
        if meaningful:
            optimal_lap, opt_total = min(totals, key=lambda t: t[1])
            window = sorted(lap for lap, tot in totals if tot <= opt_total + tolerance)
            opt, ws, we = optimal_lap, window[0], window[-1]
        else:
            opt, ws, we = None, None, None

        rows.append({
            "Driver": drv,
            "OptimalLap": opt,
            "WindowStart": ws,
            "WindowEnd": we,
            "BasePace": round(base, 3),
            "DegPerLap": round(deg if deg is not None else default_deg, 4),
            "Level": quality["level"],
            "Confidence": quality["confidence"],
            "Estimated": ESTIMATED,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("OptimalLap").reset_index(drop=True)
    return out