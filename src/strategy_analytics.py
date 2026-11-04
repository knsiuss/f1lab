# -*- coding: utf-8 -*-
"""
strategy_analytics.py
~~~~~~~~~~~~~~~~~~~~~
Tyre-strategy and pit-stop calculations for the Tyre Strategy page.

Pure pandas logic over FastF1-style frames (the session lap table, the
tyre-stint table from :func:`fastf1_extended.get_tyre_stints` and the
pit-stop table from :func:`fastf1_extended.get_pit_stops`). No Streamlit,
no network: everything here is unit-testable with synthetic frames.

Data-honesty contract (shared with :mod:`pace`):

* measured and estimated values are never mixed -- pit-stop durations
  carry a ``Measured`` flag from the extractor, and every result dict
  reports how many of its inputs were measured;
* small samples are reported with their size, and pace/degradation
  figures below the fit threshold return ``None`` instead of a number;
* every function returns source metadata (``source``, ``status``,
  ``n``) alongside the numbers, so the page can label them.

Function names follow the project's snake_case convention; they map to
the brief's camelCase list (``calculateTotalPitStops`` etc.) one to one.
"""

from typing import Dict, List, Optional

# Ensure src/ is importable no matter how this module is reached (as
# ``strategy_analytics`` from a page, or as ``src.strategy_analytics``
# from the test suite) -- the same convention as src/analysis.py.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from pace import ESTIMATED, MIN_LAPS_FOR_FIT, clean_laps_for_analysis, data_quality  # noqa: E402

# Compounds the app knows how to colour; anything else is reported as
# UNKNOWN rather than guessed.
KNOWN_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"}


def _compound(value) -> str:
    """Normalise a compound label ("Soft" -> "SOFT", junk -> "UNKNOWN")."""
    label = str(value or "").strip().upper()
    return label if label in KNOWN_COMPOUNDS else "UNKNOWN"


def _team_map(laps: pd.DataFrame) -> Dict[str, str]:
    """Driver -> team from the lap table (most recent row wins)."""
    if laps is None or laps.empty or "Driver" not in laps.columns \
            or "Team" not in laps.columns:
        return {}
    teams = laps["Team"].replace("", pd.NA)
    return teams.groupby(laps["Driver"]).last().dropna().to_dict()


# ---------------------------------------------------------------------------
# Pit stops
# ---------------------------------------------------------------------------

def calculate_total_pit_stops(pit_stops: pd.DataFrame) -> Dict:
    """Total stops in the frame plus the per-driver breakdown."""
    if pit_stops is None or pit_stops.empty or "Driver" not in pit_stops.columns:
        return {"total": 0, "per_driver": {}, "source": "pit_stops",
                "status": "unavailable", "n": 0}
    per_driver = pit_stops.groupby("Driver").size().astype(int).to_dict()
    measured = int(pit_stops["Measured"].sum()) \
        if "Measured" in pit_stops.columns else len(pit_stops)
    return {
        "total": int(len(pit_stops)),
        "per_driver": per_driver,
        "measured": measured,
        "estimated": int(len(pit_stops)) - measured,
        "source": "pit_stops",
        "status": "verified" if measured == len(pit_stops) else "partial",
        "n": int(len(pit_stops)),
    }


def calculate_average_pit_duration(pit_stops: pd.DataFrame) -> Dict:
    """Mean/median/fastest/slowest stop duration, measured split out."""
    empty = {"average": None, "median": None, "fastest": None, "slowest": None,
             "valid": 0, "measured": 0, "source": "pit_stops",
             "status": "unavailable", "n": 0}
    if pit_stops is None or pit_stops.empty or "PitTime" not in pit_stops.columns:
        return empty
    times = pd.to_numeric(pit_stops["PitTime"], errors="coerce").dropna()
    if times.empty:
        return empty
    measured = int(pit_stops.loc[times.index, "Measured"].sum()) \
        if "Measured" in pit_stops.columns else int(len(times))
    return {
        "average": round(float(times.mean()), 2),
        "median": round(float(times.median()), 2),
        "fastest": round(float(times.min()), 2),
        "slowest": round(float(times.max()), 2),
        "valid": int(len(times)),
        "measured": measured,
        "source": "pit_stops",
        "status": "verified" if measured == len(times) else "partial",
        "n": int(len(times)),
    }


def calculate_fastest_pit_stop(pit_stops: pd.DataFrame,
                               laps: Optional[pd.DataFrame] = None) -> Dict:
    """The fastest single stop: driver, team, lap and duration."""
    if pit_stops is None or pit_stops.empty or "PitTime" not in pit_stops.columns:
        return {"driver": None, "team": None, "lap": None, "duration": None,
                "measured": False, "source": "pit_stops",
                "status": "unavailable", "n": 0}
    times = pd.to_numeric(pit_stops["PitTime"], errors="coerce")
    if times.notna().sum() == 0:
        return {"driver": None, "team": None, "lap": None, "duration": None,
                "measured": False, "source": "pit_stops",
                "status": "unavailable", "n": 0}
    row = pit_stops.loc[times.idxmin()]
    team = _team_map(laps).get(row["Driver"]) if laps is not None else None
    return {
        "driver": str(row["Driver"]),
        "team": team,
        "lap": int(row["Lap"]) if "Lap" in pit_stops.columns
               and pd.notna(row.get("Lap")) else None,
        "duration": round(float(row["PitTime"]), 2),
        "measured": bool(row["Measured"]) if "Measured" in pit_stops.columns
                    else False,
        "source": "pit_stops",
        "status": "verified",
        "n": int(times.notna().sum()),
    }


def calculate_pit_stop_table(pit_stops: pd.DataFrame,
                             laps: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per driver: stops, durations, window and average stop lap."""
    if pit_stops is None or pit_stops.empty or "Driver" not in pit_stops.columns:
        return pd.DataFrame()
    teams = _team_map(laps) if laps is not None else {}
    rows = []
    for driver, group in pit_stops.groupby("Driver"):
        times = pd.to_numeric(group["PitTime"], errors="coerce").dropna()
        stop_laps = pd.to_numeric(group["Lap"], errors="coerce").dropna() \
            if "Lap" in group.columns else pd.Series(dtype=float)
        rows.append({
            "Driver": str(driver),
            "Team": teams.get(driver, ""),
            "Pit Stops": int(len(group)),
            "Avg Duration (s)": round(float(times.mean()), 2) if not times.empty else None,
            "Fastest Stop (s)": round(float(times.min()), 2) if not times.empty else None,
            "Slowest Stop (s)": round(float(times.max()), 2) if not times.empty else None,
            "Pit Window": (f"L{int(stop_laps.min())}–L{int(stop_laps.max())}"
                           if not stop_laps.empty else "n/a"),
            "Avg Pit Lap": round(float(stop_laps.mean()), 1)
                           if not stop_laps.empty else None,
            "Measured": int(group["Measured"].sum())
                        if "Measured" in group.columns else int(len(group)),
        })
    table = pd.DataFrame(rows)
    return table.sort_values("Avg Duration (s)", na_position="last") \
        .reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tyre stints and compound usage
# ---------------------------------------------------------------------------

def calculate_stint_length(stints: pd.DataFrame) -> pd.DataFrame:
    """Validate stints and (re)compute ``Laps = End - Start + 1``."""
    if stints is None or stints.empty:
        return pd.DataFrame()
    out = stints.copy()
    out["Compound"] = out["Compound"].map(_compound) \
        if "Compound" in out.columns else "UNKNOWN"
    start = pd.to_numeric(out.get("StartLap"), errors="coerce")
    end = pd.to_numeric(out.get("EndLap"), errors="coerce")
    out = out[start.notna() & end.notna() & (end >= start)]
    out["StartLap"] = start.loc[out.index].astype(int)
    out["EndLap"] = end.loc[out.index].astype(int)
    out["Laps"] = out["EndLap"] - out["StartLap"] + 1
    return out.sort_values(["Driver", "StartLap"]).reset_index(drop=True)


def calculate_compound_usage(laps: pd.DataFrame,
                             drivers: Optional[List[str]] = None) -> Dict:
    """Share of recorded laps per compound (laps, not stints)."""
    empty = {"usage": {}, "total_laps": 0, "by_driver": {},
             "source": "laps", "status": "unavailable", "n": 0}
    if laps is None or laps.empty or "Compound" not in laps.columns:
        return empty
    frame = laps
    if drivers:
        frame = frame[frame["Driver"].isin(drivers)]
    if frame.empty:
        return empty
    compounds = frame["Compound"].map(_compound)
    counts = compounds.value_counts()
    total = int(counts.sum())
    usage = {str(k): {"laps": int(v), "pct": round(100.0 * v / total, 1)}
             for k, v in counts.items()}
    by_driver = {}
    for driver, group in frame.groupby("Driver"):
        sub = group["Compound"].map(_compound).value_counts()
        by_driver[str(driver)] = {str(k): int(v) for k, v in sub.items()}
    return {
        "usage": usage,
        "total_laps": total,
        "by_driver": by_driver,
        "source": "laps",
        "status": "verified",
        "n": total,
    }


# ---------------------------------------------------------------------------
# Pace by compound and degradation
# ---------------------------------------------------------------------------

def calculate_average_compound_pace(laps: pd.DataFrame,
                                    mode: str = "filtered") -> pd.DataFrame:
    """Average/median/fastest lap per compound with sample sizes.

    ``mode="raw"`` uses every timed lap; ``mode="filtered"`` uses clean
    laps only (in/out laps, SC/VSC and inaccurate laps removed). Raw
    averages mix race phases and fuel loads, so the mode is carried in
    the result and each row reports its sample size and driver coverage.
    """
    if laps is None or laps.empty or "Compound" not in laps.columns:
        return pd.DataFrame()
    if mode == "filtered":
        frame = clean_laps_for_analysis(laps)
    else:
        frame = laps.copy()
        if "LapTime" in frame.columns:
            frame["LapTime_s"] = pd.to_numeric(
                frame["LapTime"].dt.total_seconds()
                if hasattr(frame["LapTime"], "dt") else frame["LapTime"],
                errors="coerce")
    if frame.empty or "LapTime_s" not in frame.columns:
        return pd.DataFrame()
    frame = frame.copy()
    frame["Compound"] = frame["Compound"].map(_compound)
    frame = frame[frame["Compound"] != "UNKNOWN"]
    rows = []
    for compound, group in frame.groupby("Compound"):
        times = pd.to_numeric(group["LapTime_s"], errors="coerce").dropna()
        if times.empty:
            continue
        quality = data_quality(len(times))
        rows.append({
            "Compound": compound,
            "Mean (s)": round(float(times.mean()), 3),
            "Median (s)": round(float(times.median()), 3),
            "Fastest (s)": round(float(times.min()), 3),
            "Laps": int(len(times)),
            "Drivers": int(group["Driver"].nunique())
                       if "Driver" in group.columns else 0,
            "Level": quality["level"],
            "Estimated": ESTIMATED,
        })
    return pd.DataFrame(rows).sort_values("Mean (s)").reset_index(drop=True)


def calculate_tyre_degradation(laps: pd.DataFrame) -> pd.DataFrame:
    """Per-stint fitted pace trend (seconds per lap of tyre age).

    Thin wrapper over :func:`pace.normalised_pace_by_stint` so the page
    and the tests share one definition: the slope of a linear fit of lap
    time against tyre age on clean laps. Labelled estimated -- it is a
    pace trend, not a controlled degradation model (traffic, fuel load
    and track evolution are not separated out).
    """
    from pace import normalised_pace_by_stint
    if laps is None or laps.empty:
        return pd.DataFrame()
    return normalised_pace_by_stint(laps, group_by_compound=True)


def degradation_trend(laps: pd.DataFrame) -> pd.DataFrame:
    """Median clean-lap time per (compound, tyre age) for the trend chart."""
    if laps is None or laps.empty:
        return pd.DataFrame()
    clean = clean_laps_for_analysis(laps)
    if clean.empty or "Compound" not in clean.columns \
            or "TyreLife" not in clean.columns:
        return pd.DataFrame()
    clean = clean.copy()
    clean["Compound"] = clean["Compound"].map(_compound)
    clean = clean[clean["Compound"] != "UNKNOWN"]
    clean["TyreAge"] = pd.to_numeric(clean["TyreLife"], errors="coerce")
    clean = clean[clean["TyreAge"].notna()]
    if clean.empty:
        return pd.DataFrame()
    trend = (clean.groupby(["Compound", "TyreAge"])["LapTime_s"]
             .median().reset_index()
             .rename(columns={"LapTime_s": "MedianLapTime_s"}))
    return trend.sort_values(["Compound", "TyreAge"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy simulation, comparison and pit-window optimisation
# ---------------------------------------------------------------------------

def simulate_race_strategy(compounds: List[str], stop_laps: List[int],
                           total_laps: int, base_lap_time: float = 95.0,
                           pit_loss: float = 25.0,
                           deg_rates: Optional[Dict[str, float]] = None,
                           **kwargs) -> Dict:
    """Simulate one stop plan through the shared strategy engine."""
    from whatif import Scenario, simulate_scenario
    scenario = Scenario(name="custom", compounds=list(compounds),
                        stop_laps=[int(s) for s in stop_laps])
    sim_kwargs = dict(base_lap_time=float(base_lap_time),
                      pit_loss=float(pit_loss))
    if deg_rates:
        sim_kwargs["deg_rates"] = deg_rates
    sim_kwargs.update(kwargs)
    result = simulate_scenario(scenario, int(total_laps), **sim_kwargs)
    result["source"] = "whatif.simulate_scenario"
    return result


def compare_strategies(plans: List[Dict], total_laps: int,
                       **sim_kwargs) -> pd.DataFrame:
    """Simulate up to three plans and rank them by projected time.

    Each plan is a dict with ``name``, ``compounds`` and ``stop_laps``.
    The fastest row is marked; it is the fastest *under the stated
    assumptions*, never a recommendation.
    """
    from whatif import Scenario, compare_scenarios
    scenarios = [Scenario(name=str(p.get("name", f"Plan {i + 1}")),
                          compounds=list(p.get("compounds", [])),
                          stop_laps=[int(s) for s in p.get("stop_laps", [])])
                 for i, p in enumerate(plans[:3])]
    if not scenarios:
        return pd.DataFrame()
    return compare_scenarios(scenarios, int(total_laps), **sim_kwargs)


def optimize_pit_lap(laps: pd.DataFrame, total_laps: Optional[int] = None,
                     n_stops: int = 1) -> pd.DataFrame:
    """Per-driver optimal pit lap + window from the shared grid search."""
    from pitwindow import pit_window
    if laps is None or laps.empty:
        return pd.DataFrame()
    return pit_window(laps, total_laps=total_laps, n_stops=int(n_stops))


# ---------------------------------------------------------------------------
# Rule-based insights
# ---------------------------------------------------------------------------

def generate_strategy_insights(laps: pd.DataFrame,
                               stints: pd.DataFrame,
                               pit_stops: pd.DataFrame) -> List[Dict]:
    """Rule-based findings, each with its supporting metrics.

    No prose is generated: every item is a number the data supports
    (longest/shortest stint, fastest stop, most-used compound, largest
    pace-trend estimate). Items without enough data are omitted.
    """
    insights: List[Dict] = []

    usage = calculate_compound_usage(laps)
    if usage["usage"]:
        compound, info = max(usage["usage"].items(),
                             key=lambda kv: kv[1]["laps"])
        insights.append({
            "title": "Most used compound",
            "value": compound,
            "detail": (f"{info['laps']} of {usage['total_laps']} laps "
                       f"({info['pct']}%)"),
        })

    if stints is not None and not stints.empty and "Laps" in stints.columns:
        longest = stints.loc[stints["Laps"].idxmax()]
        insights.append({
            "title": "Longest stint",
            "value": f"{int(longest['Laps'])} laps",
            "detail": (f"{longest['Driver']} on {longest['Compound']}, "
                       f"L{int(longest['StartLap'])}–L{int(longest['EndLap'])}"),
        })
        shortest = stints.loc[stints["Laps"].idxmin()]
        insights.append({
            "title": "Shortest stint",
            "value": f"{int(shortest['Laps'])} laps",
            "detail": (f"{shortest['Driver']} on {shortest['Compound']}, "
                       f"L{int(shortest['StartLap'])}–L{int(shortest['EndLap'])}"),
        })

    fastest = calculate_fastest_pit_stop(pit_stops, laps)
    if fastest["driver"]:
        insights.append({
            "title": "Fastest pit stop",
            "value": f"{fastest['duration']:.2f}s",
            "detail": (f"{fastest['driver']}"
                       + (f" ({fastest['team']})" if fastest["team"] else "")
                       + (f", lap {fastest['lap']}" if fastest["lap"] else "")
                       + (" (measured)" if fastest["measured"]
                          else " (estimated)")),
        })

    deg = calculate_tyre_degradation(laps)
    if deg is not None and not deg.empty and "DegPerLap" in deg.columns:
        # A slope from a handful of laps is noise, not degradation: only
        # fits backed by a moderate sample (>= 8 clean laps, the shared
        # data-quality threshold) may become an insight.
        fitted = deg.dropna(subset=["DegPerLap"])
        fitted = fitted[fitted["N"] >= 8] if "N" in fitted.columns else fitted
        if not fitted.empty:
            worst = fitted.loc[fitted["DegPerLap"].idxmax()]
            insights.append({
                "title": "Largest pace-trend estimate",
                "value": f"+{float(worst['DegPerLap']):.3f}s/lap",
                "detail": (f"{worst['Driver']} on {worst['Compound']} "
                           f"(stint {worst['Stint']}, {int(worst['N'])} clean laps)"),
            })

    pace = calculate_average_compound_pace(laps, mode="filtered")
    if pace is not None and not pace.empty:
        best = pace.iloc[0]
        insights.append({
            "title": "Best average compound pace",
            "value": f"{best['Compound']} {best['Mean (s)']:.2f}s",
            "detail": (f"filtered clean laps, n={int(best['Laps'])}, "
                       f"{int(best['Drivers'])} drivers — averages mix race "
                       "phases, so this is not proof the compound itself is best"),
        })
    return insights
