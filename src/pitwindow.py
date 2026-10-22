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

Multi-stop strategies
---------------------
``n_stops`` is the number of *real* pit stops: the search enumerates stop
plans with distinct, increasing stop laps (``stop_1 < stop_2 < ...``) and
every plan is simulated with the stop count the plan declares, so a two-stop
strategy is charged two pit losses and can never collapse into the one-stop
answer. The stop-plan rules live in :func:`whatif.stop_lap_problem` and both
modules use them.

Data Honesty contract: everything is estimated from the driver's own clean laps
where available, labelled estimated, and carries a confidence label from the
sample size. Drivers with no degradation fit fall back to model defaults at low
confidence rather than being silently assumed perfect or assumed bad.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import itertools
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# Upper bound on the number of stop plans evaluated per driver. A two-stop
# search over a 57-lap race is C(56, 2) = 1540 plans; a three-stop search
# would be 29k, so the candidate laps are thinned with a stride and the
# stride is reported in the output (a coarser search, said out loud).
MAX_CANDIDATES = 2000


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


def _stop_plans(total_laps: int, n_stops: int) -> Tuple[List[Tuple[int, ...]], int]:
    """Candidate stop plans: ``n_stops`` distinct, increasing laps in the race.

    A stop must leave at least one lap on the old tyre and one on the new
    tyre, so the candidate laps are ``1 .. total_laps - 1`` (the same rule
    :func:`whatif.stop_lap_problem` enforces). Returns the plans and the
    stride used to thin them when the search space would be too large.
    """
    laps = list(range(1, max(2, total_laps)))
    if n_stops <= 1:
        return [(lap,) for lap in laps], 1
    if n_stops > len(laps):
        return [], 1
    total = 1
    for i in range(n_stops):
        total = total * (len(laps) - i) // (i + 1)
    stride = 1
    while total // stride > MAX_CANDIDATES:
        stride += 1
    if stride > 1:
        thinned = laps[::stride]
        if thinned[-1] != laps[-1]:
            thinned.append(laps[-1])
        laps = thinned
    return [tuple(plan) for plan in itertools.combinations(laps, n_stops)], stride


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

    For ``n_stops >= 2`` the search enumerates plans with distinct, increasing
    stop laps, so the answer is a real two-stop (or n-stop) strategy and is
    charged that many pit losses. ``OptimalLap`` then reports the *first*
    stop lap of the optimal plan, ``OptimalLap2`` its last stop lap, and
    ``WindowStart``/``WindowEnd`` the range of first stop laps inside the
    tolerance (``WindowStart2``/``WindowEnd2`` the same for the last stop).

    Args:
        laps: FastF1-style lap table (Driver, LapNumber, LapTime, ...).
        total_laps: simulated race length; defaults to the table's max lap.
        n_stops: number of pit stops in the modelled strategy.
        deg_rates / pit_loss / fuel_gain: strategy-engine knobs; default from
            config.MODEL_CONFIG (as everywhere else).
        tolerance: window spread (seconds) around the objective optimum.

    Returns:
        DataFrame (possibly empty) with Driver, OptimalLap, WindowStart,
        WindowEnd, BasePace, DegPerLap, Level, Confidence, Estimated and the
        extra multi-stop columns (NStops, OptimalLap2, WindowStart2,
        WindowEnd2, OptimalStops, Candidates, SearchStride).
    """
    deg_rates = deg_rates or MODEL_CONFIG["degradation_rates"]
    pit_loss = pit_loss if pit_loss is not None else MODEL_CONFIG["pit_loss_sec"]
    if fuel_gain is None:
        fuel_gain = float(MODEL_CONFIG["fuel_gain_per_lap"])

    if laps is None or laps.empty or "Driver" not in laps.columns:
        return pd.DataFrame()

    n_stops = max(1, int(n_stops))
    N = _resolve_total_laps(laps, total_laps)
    if N <= n_stops + 1:
        return pd.DataFrame()

    base_map, deg_map, counts = _driver_estimates(laps)
    if not base_map:
        return pd.DataFrame()

    plans, stride = _stop_plans(N, n_stops)
    if not plans:
        return pd.DataFrame()
    compounds = [DEFAULT_COMPOUND] * (n_stops + 1)
    default_deg = deg_rates["DEFAULT"]

    rows = []
    for drv in sorted(base_map.keys()):
        base = base_map[drv]
        deg = deg_map.get(drv)
        pool = {DEFAULT_COMPOUND: deg if deg is not None else default_deg}
        quality = data_quality(counts.get(drv, 0))

        totals = []
        for stops in plans:
            scenario = Scenario(
                name="radar@" + "-".join(str(s) for s in stops),
                compounds=compounds,
                stop_laps=list(stops),
            )
            res = simulate_scenario(
                scenario, N,
                base_lap_time=base, deg_rates=pool,
                fuel_gain=fuel_gain, pit_loss=pit_loss,
            )
            tot = (res or {}).get("total_time")
            if tot is not None:
                totals.append((stops, float(tot)))

        if not totals:
            continue

        # Data-honesty gate: without meaningful tyre ageing there is no real pit
        # window, so we report the driver with a blank window rather than an
        # artefact early stop from a flat total-time curve.
        meaningful = deg is not None and deg >= MIN_MEANINGFUL_DEG
        if meaningful:
            optimal_stops, opt_total = min(totals, key=lambda t: t[1])
            inside = [stops for stops, tot in totals
                      if tot <= opt_total + tolerance]
            firsts = sorted(s[0] for s in inside)
            lasts = sorted(s[-1] for s in inside)
            # The optimum is the argmin plan; the window is the spread of the
            # plans that stay within the tolerance of it.
            opt = optimal_stops[0]
            ws, we = firsts[0], firsts[-1]
            opt_last = optimal_stops[-1]
            ws2 = we2 = None
            if n_stops >= 2:
                ws2, we2 = lasts[0], lasts[-1]
        else:
            opt = ws = we = opt_last = ws2 = we2 = None

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
            "NStops": n_stops,
            "OptimalLap2": opt_last if n_stops >= 2 else None,
            "WindowStart2": ws2,
            "WindowEnd2": we2,
            "OptimalStops": ("-".join(str(s) for s in optimal_stops)
                             if meaningful else None),
            "Candidates": len(totals),
            "SearchStride": stride,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("OptimalLap").reset_index(drop=True)
    return out
