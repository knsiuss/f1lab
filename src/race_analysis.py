# -*- coding: utf-8 -*-
"""Race-analysis analytics: pure functions over a session lap table.

One deterministic home for the calculations the Analysis Center (and
the weekend report) share: race overview, lap-mode filtering, results,
telemetry alignment by distance, sector theoretical best, tyre
degradation per stint, pit summary, position changes, pace
distribution, gap statistics, data-quality validation and automated
insights.

Every function degrades gracefully when a column is absent (the
OpenF1 fallback session has no race-control flags and no telemetry):
missing input yields an empty frame / ``None``, never a fabricated
number. Status labels follow :mod:`sources` (Verified / Available /
Partial / Estimated / Unavailable).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from pace import clean_laps_for_analysis  # noqa: E402

VALID_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET",
                    "_UNKNOWN"}


def _laps_with_times(laps):
    """Lap rows with a usable numeric lap time (seconds).

    ``LapTime`` may be a Timedelta (FastF1) or numeric seconds; both
    are coerced via :func:`pace._to_seconds` so a Timedelta never
    reads as nanoseconds.
    """
    if laps is None or laps.empty or "LapTime" not in laps.columns:
        return pd.DataFrame()
    from pace import _to_seconds
    work = laps.copy()
    work["_time"] = _to_seconds(work["LapTime"])
    return work.dropna(subset=["_time"])


def validate_laps(laps) -> pd.DataFrame:
    """Data-quality issues in a lap table (one row per issue found).

    Checks: duplicate (driver, lap) rows, non-positive lap numbers,
    non-positive lap times, unknown tyre compounds, and laps with a
    missing lap time. Returns columns ``Driver, LapNumber, Issue``;
    an empty frame means no issues were found.
    """
    issues = []
    if laps is None or laps.empty:
        return pd.DataFrame(columns=["Driver", "LapNumber", "Issue"])
    work = laps.copy()
    if "Driver" in work.columns and "LapNumber" in work.columns:
        dupes = work[work.duplicated(subset=["Driver", "LapNumber"],
                                     keep=False)]
        for _, row in dupes.iterrows():
            issues.append({"Driver": row.get("Driver"),
                           "LapNumber": row.get("LapNumber"),
                           "Issue": "duplicate lap"})
    if "LapNumber" in work.columns:
        bad = work[pd.to_numeric(work["LapNumber"], errors="coerce") <= 0]
        for _, row in bad.iterrows():
            issues.append({"Driver": row.get("Driver"),
                           "LapNumber": row.get("LapNumber"),
                           "Issue": "non-positive lap number"})
    if "LapTime" in work.columns:
        times = pd.to_numeric(work["LapTime"], errors="coerce")
        bad = work[(times.notna()) & (times <= 0)]
        for _, row in bad.iterrows():
            issues.append({"Driver": row.get("Driver"),
                           "LapNumber": row.get("LapNumber"),
                           "Issue": "non-positive lap time"})
        missing = work[times.isna()]
        for _, row in missing.iterrows():
            issues.append({"Driver": row.get("Driver"),
                           "LapNumber": row.get("LapNumber"),
                           "Issue": "missing lap time"})
    if "Compound" in work.columns:
        unknown = work[~work["Compound"].astype(str).str.upper()
                       .isin(VALID_COMPOUNDS)]
        for _, row in unknown.iterrows():
            issues.append({"Driver": row.get("Driver"),
                           "LapNumber": row.get("LapNumber"),
                           "Issue": "unknown tyre compound"})
    return pd.DataFrame(issues,
                        columns=["Driver", "LapNumber", "Issue"])


def data_status(laps) -> str:
    """Overall data status of a lap table (see :mod:`sources`)."""
    if laps is None or laps.empty:
        return "Unavailable"
    issues = validate_laps(laps)
    if issues.empty:
        return "Verified"
    if len(issues) <= max(1, int(0.05 * len(laps))):
        return "Partial"
    return "Partial"


def race_overview(laps, pit_stops=None) -> dict:
    """Headline numbers of a session, all measured from the lap table."""
    timed = _laps_with_times(laps)
    if timed.empty:
        return {"status": "Unavailable"}
    fastest = timed.loc[timed["_time"].idxmin()]
    total_laps = int(pd.to_numeric(timed["LapNumber"],
                                   errors="coerce").nunique()) \
        if "LapNumber" in timed.columns else 0
    return {
        "status": "Available",
        "fastest_lap_s": float(fastest["_time"]),
        "fastest_driver": str(fastest.get("Driver", "")),
        "fastest_lap_number": (int(fastest["LapNumber"])
                               if pd.notna(fastest.get("LapNumber"))
                               else None),
        "average_lap_s": float(timed["_time"].mean()),
        "total_laps": total_laps,
        "pit_stops": int(len(pit_stops)) if pit_stops is not None
        and not pit_stops.empty else 0,
    }


def lap_mode_filter(laps, mode: str = "all") -> pd.DataFrame:
    """Lap subset for an analysis mode.

    ``all`` keeps every timed lap; ``green`` keeps green-flag laps
    when a track-status column exists (otherwise it falls back to
    ``all`` and the caller must say so); ``clean`` uses
    :func:`pace.clean_laps_for_analysis`; ``stint`` is the caller's
    job per stint (this returns ``all``).
    """
    timed = _laps_with_times(laps)
    if timed.empty:
        return timed
    if mode == "clean":
        return clean_laps_for_analysis(laps)
    if mode == "green" and "TrackStatus" in timed.columns:
        flags = timed["TrackStatus"].astype(str).str.strip()
        return timed[flags.isin(["1", "1.0", "Green"])]
    return timed


def results_table(laps, pit_stops=None, stints=None,
                  points_by_driver=None) -> pd.DataFrame:
    """Classification-style results table from the lap table.

    Sorted by best finishing position (last recorded), then team.
    Pit-stop counts and tyre strategy come from the optional frames;
    columns with no data are left empty, never guessed.
    """
    if laps is None or laps.empty or "Driver" not in laps.columns:
        return pd.DataFrame()
    work = laps.copy()
    if "Position" in work.columns:
        work["_pos"] = pd.to_numeric(work["Position"], errors="coerce")
    else:
        work["_pos"] = np.nan
    pit_counts = {}
    if pit_stops is not None and not pit_stops.empty \
            and "Driver" in pit_stops.columns:
        pit_counts = pit_stops.groupby("Driver").size().to_dict()
    strategy = {}
    if stints is not None and not stints.empty \
            and "Driver" in stints.columns and "Compound" in stints.columns:
        for driver, group in stints.groupby("Driver"):
            compounds = [str(c).upper() for c in
                         group.sort_values("Stint")["Compound"]
                         if str(c).upper() not in ("UNKNOWN", "NAN", "")]
            strategy[driver] = " -> ".join(compounds)
    rows = []
    for driver, group in work.groupby("Driver"):
        timed = group.dropna(subset=["_time"]) if "_time" in group \
            else group
        timed = _laps_with_times(group)
        positions = group["_pos"].dropna() if "_pos" in group \
            else pd.Series(dtype=float)
        rows.append({
            "Driver": str(driver),
            "Team": str(group["Team"].iloc[-1]) if "Team" in group.columns
            and not group.empty else "",
            "Position": int(positions.iloc[-1]) if not positions.empty
            else None,
            "Laps": int(pd.to_numeric(group.get("LapNumber"),
                                       errors="coerce").nunique())
            if "LapNumber" in group.columns else 0,
            "BestLap_s": float(timed["_time"].min()) if not timed.empty
            else None,
            "AvgLap_s": float(timed["_time"].mean()) if not timed.empty
            else None,
            "PitStops": int(pit_counts.get(driver, 0)),
            "TyreStrategy": strategy.get(driver, ""),
            "Points": (float(points_by_driver.get(str(driver), 0.0))
                       if points_by_driver else None),
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["Position", "BestLap_s"], na_position="last").reset_index(drop=True)


def align_telemetry(driver_frames: dict, grid_points: int = 200) -> dict:
    """Resample each driver's telemetry onto a common distance grid.

    ``driver_frames`` maps driver -> frame with ``Distance`` and the
    channels ``Speed, Throttle, Brake, Gear, RPM, DRS``. Alignment is
    by circuit distance (never raw timestamps from unrelated laps).
    Returns ``{driver: DataFrame(distance, channels...)}``; drivers
    without a usable distance column are skipped.
    """
    aligned = {}
    channels = ["Speed", "Throttle", "Brake", "Gear", "RPM", "DRS"]
    for driver, frame in (driver_frames or {}).items():
        if frame is None or frame.empty or "Distance" not in frame.columns:
            continue
        work = frame.copy()
        work["_dist"] = pd.to_numeric(work["Distance"], errors="coerce")
        work = work.dropna(subset=["_dist"]).sort_values("_dist")
        if len(work) < 2:
            continue
        grid = np.linspace(work["_dist"].min(), work["_dist"].max(),
                           int(grid_points))
        sampled = {"Distance": grid}
        for channel in channels:
            sampled[channel] = (np.interp(grid, work["_dist"],
                                          pd.to_numeric(
                                              work.get(channel, 0),
                                              errors="coerce").fillna(0))
                                if channel in work.columns else np.zeros_like(grid))
        aligned[str(driver)] = pd.DataFrame(sampled)
    return aligned


def telemetry_summary(frame) -> dict:
    """Max/average speed and braking-point count of one aligned trace."""
    if frame is None or frame.empty or "Speed" not in frame.columns:
        return {"status": "Unavailable"}
    speed = pd.to_numeric(frame["Speed"], errors="coerce").fillna(0)
    brake = pd.to_numeric(frame.get("Brake", 0), errors="coerce").fillna(0)
    braking = int(((brake > 0) & (brake.shift(fill_value=0) == 0)).sum())
    return {
        "status": "Available",
        "max_speed": float(speed.max()),
        "avg_speed": float(speed.mean()),
        "braking_points": braking,
    }


def theoretical_best(sector_summary) -> pd.DataFrame:
    """Theoretical best lap: sum of each driver's best sectors (S1+S2+S3)."""
    if sector_summary is None or sector_summary.empty:
        return pd.DataFrame()
    rows = []
    for _, row in sector_summary.iterrows():
        bests = [row.get(f"Sector{part}Best") for part in ("1", "2", "3")]
        if any(pd.isna(b) for b in bests):
            continue
        rows.append({"Driver": row.get("Driver"),
                     "TheoreticalBest_s": float(sum(bests)),
                     "S1": float(bests[0]), "S2": float(bests[1]),
                     "S3": float(bests[2])})
    return pd.DataFrame(rows).sort_values("TheoreticalBest_s")


def stint_timeline(stints) -> pd.DataFrame:
    """Stint lengths (``end - start + 1``) with compound and driver."""
    if stints is None or stints.empty:
        return pd.DataFrame()
    work = stints.copy()
    for col in ("StartLap", "EndLap"):
        work[col] = pd.to_numeric(work.get(col), errors="coerce")
    work = work.dropna(subset=["StartLap", "EndLap"])
    work["StintLength"] = work["EndLap"] - work["StartLap"] + 1
    return work


def degradation_slope(laps, min_laps: int = 5) -> pd.DataFrame:
    """Per-stint lap-time slope (s/lap) from a linear fit.

    Only stints with at least ``min_laps`` timed laps are fitted;
    traffic, fuel load and safety cars are **not** controlled for, so
    the slope is a tyre-wear *proxy*, labelled estimated by callers.
    """
    # Clean laps only: a pit-lap outlier would dominate the fit and
    # masquerade as tyre wear.
    timed = clean_laps_for_analysis(laps) if laps is not None \
        and not laps.empty else pd.DataFrame()
    if timed.empty or "Stint" not in timed.columns \
            or "LapNumber" not in timed.columns \
            or "LapTime_s" not in timed.columns:
        return pd.DataFrame()
    rows = []
    for (driver, stint), group in timed.groupby(["Driver", "Stint"]):
        group = group.dropna(subset=["LapNumber"]).sort_values("LapNumber")
        if len(group) < int(min_laps):
            continue
        x = pd.to_numeric(group["LapNumber"], errors="coerce").to_numpy()
        y = group["LapTime_s"].to_numpy()
        slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 1 else 0.0
        rows.append({"Driver": driver, "Stint": stint, "Laps": len(group),
                     "Slope_s_per_lap": slope})
    return pd.DataFrame(rows)


def pit_summary(pit_stops) -> dict:
    """Field pit summary: counts, durations, window, fastest."""
    if pit_stops is None or pit_stops.empty \
            or "PitTime" not in pit_stops.columns:
        return {"status": "Unavailable"}
    times = pd.to_numeric(pit_stops["PitTime"], errors="coerce").dropna()
    laps = pd.to_numeric(pit_stops.get("Lap"), errors="coerce").dropna()
    return {
        "status": "Available",
        "total": int(len(pit_stops)),
        "average_s": float(times.mean()) if not times.empty else None,
        "fastest_s": float(times.min()) if not times.empty else None,
        "window": (int(laps.min()), int(laps.max())) if not laps.empty
        else None,
    }


def position_changes(laps, pit_laps=None) -> pd.DataFrame:
    """Start-vs-final position gain/loss per driver.

    ``Gain`` is positive when the driver finished better than they
    started. Overtakes are **not** inferred from position changes
    alone (pit cycles can move positions); see
    :func:`conservative_overtakes` for the stricter event list.
    """
    if laps is None or laps.empty or "Position" not in laps.columns:
        return pd.DataFrame()
    work = laps.copy()
    work["_pos"] = pd.to_numeric(work["Position"], errors="coerce")
    work["_lap"] = pd.to_numeric(work.get("LapNumber"), errors="coerce")
    work = work.dropna(subset=["_pos", "_lap"]).sort_values(
        ["Driver", "_lap"])
    rows = []
    for driver, group in work.groupby("Driver"):
        start = int(group["_pos"].iloc[0])
        final = int(group["_pos"].iloc[-1])
        rows.append({"Driver": driver, "Start": start, "Final": final,
                     "Gain": start - final})
    return pd.DataFrame(rows)


def conservative_overtakes(laps, pit_laps=None) -> pd.DataFrame:
    """Position gains on a single lap, excluding pit laps.

    A gain counts only when the driver improved between two
    consecutive recorded laps and neither lap is a pit lap (when pit
    data is available), so a stop cannot masquerade as an overtake.
    """
    if laps is None or laps.empty or "Position" not in laps.columns:
        return pd.DataFrame()
    work = laps.copy()
    work["_pos"] = pd.to_numeric(work["Position"], errors="coerce")
    work["_lap"] = pd.to_numeric(work.get("LapNumber"), errors="coerce")
    work = work.dropna(subset=["_pos", "_lap"]).sort_values(
        ["Driver", "_lap"])
    pit_laps = pit_laps or {}
    events = []
    for driver, group in work.groupby("Driver"):
        stops = set(pit_laps.get(driver, []) or [])
        previous = None
        for _, row in group.iterrows():
            lap = int(row["_lap"])
            if previous is not None and row["_pos"] < previous[1] \
                    and lap not in stops and previous[0] not in stops:
                events.append({"Driver": driver, "Lap": lap,
                               "From": previous[1],
                               "To": int(row["_pos"])})
            previous = (lap, int(row["_pos"]))
    return pd.DataFrame(events)


def pace_distribution(laps) -> pd.DataFrame:
    """Per-driver pace spread: median, mean, std and IQR outliers."""
    clean = clean_laps_for_analysis(laps) if laps is not None \
        and not laps.empty else pd.DataFrame()
    if clean.empty or "LapTime_s" not in clean.columns:
        return pd.DataFrame()
    rows = []
    for driver, group in clean.groupby("Driver"):
        times = group["LapTime_s"].dropna()
        if times.empty:
            continue
        q1, q3 = times.quantile(0.25), times.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((times < q1 - 1.5 * iqr)
                        | (times > q3 + 1.5 * iqr)).sum())
        rows.append({"Driver": driver, "Laps": int(len(times)),
                     "Median_s": float(times.median()),
                     "Mean_s": float(times.mean()),
                     "Std_s": float(times.std()) if len(times) > 1 else 0.0,
                     "Outliers": outliers})
    return pd.DataFrame(rows).sort_values("Median_s")


def gap_stats(gap_frame) -> pd.DataFrame:
    """Largest/smallest/final gap and closing rate per driver."""
    if gap_frame is None or gap_frame.empty or "Gap" not in gap_frame.columns:
        return pd.DataFrame()
    rows = []
    for driver, group in gap_frame.groupby("Driver"):
        group = group.sort_values("LapNumber") if "LapNumber" \
            in group.columns else group
        gaps = pd.to_numeric(group["Gap"], errors="coerce").dropna()
        if gaps.empty:
            continue
        laps_arr = pd.to_numeric(group["LapNumber"], errors="coerce") \
            if "LapNumber" in group.columns else pd.Series(range(len(gaps)))
        closing = None
        if len(gaps) > 1:
            span = float(laps_arr.iloc[-1] - laps_arr.iloc[0])
            closing = float((gaps.iloc[0] - gaps.iloc[-1]) / span) \
                if span else None
        rows.append({"Driver": driver, "LargestGap_s": float(gaps.max()),
                     "SmallestGap_s": float(gaps.min()),
                     "FinalGap_s": float(gaps.iloc[-1]),
                     "ClosingRate_s_per_lap": closing})
    return pd.DataFrame(rows)


def automated_insights(laps, pit_stops=None, stints=None) -> list:
    """Deterministic rule-based insights, each with its evidence.

    Every insight is computed from the given lap table; texts are
    generated from the numbers, never written by hand per driver.
    """
    out = []
    timed = _laps_with_times(laps)
    if timed.empty:
        return out
    fastest = timed.loc[timed["_time"].idxmin()]
    out.append({
        "title": "Fastest lap",
        "text": (f"{fastest.get('Driver', '')} set the fastest lap "
                 f"({fastest['_time']:.3f}s on lap "
                 f"{fastest.get('LapNumber', '?')})."),
        "evidence": f"{fastest['_time']:.3f}s",
    })
    dist = pace_distribution(laps)
    if not dist.empty:
        steadiest = dist.sort_values("Std_s").iloc[0]
        out.append({
            "title": "Most consistent",
            "text": (f"{steadiest['Driver']} was the most consistent "
                     f"scorer of clean laps (std {steadiest['Std_s']:.3f}s "
                     f"over {int(steadiest['Laps'])} laps)."),
            "evidence": f"std {steadiest['Std_s']:.3f}s",
        })
        quickest = dist.iloc[0]
        out.append({
            "title": "Best race pace",
            "text": (f"{quickest['Driver']} had the best median clean-lap "
                     f"pace ({quickest['Median_s']:.3f}s)."),
            "evidence": f"median {quickest['Median_s']:.3f}s",
        })
    changes = position_changes(laps)
    if not changes.empty:
        best_gain = changes.sort_values("Gain", ascending=False).iloc[0]
        worst_loss = changes.sort_values("Gain").iloc[0]
        if best_gain["Gain"] > 0:
            out.append({
                "title": "Biggest position gain",
                "text": (f"{best_gain['Driver']} gained "
                         f"{int(best_gain['Gain'])} places "
                         f"(P{int(best_gain['Start'])} to "
                         f"P{int(best_gain['Final'])})."),
                "evidence": f"+{int(best_gain['Gain'])} places",
            })
        if worst_loss["Gain"] < 0:
            out.append({
                "title": "Biggest position loss",
                "text": (f"{worst_loss['Driver']} lost "
                         f"{abs(int(worst_loss['Gain']))} places "
                         f"(P{int(worst_loss['Start'])} to "
                         f"P{int(worst_loss['Final'])})."),
                "evidence": f"{int(worst_loss['Gain'])} places",
            })
    pit = pit_summary(pit_stops)
    if pit.get("status") == "Available" and pit.get("window"):
        out.append({
            "title": "Pit window",
            "text": (f"Stops were spread from lap {pit['window'][0]} to "
                     f"lap {pit['window'][1]} "
                     f"({pit['total']} stops, fastest "
                     f"{pit['fastest_s']:.1f}s)."),
            "evidence": f"{pit['total']} stops",
        })
    slopes = degradation_slope(laps)
    if not slopes.empty:
        worst = slopes.sort_values("Slope_s_per_lap",
                                   ascending=False).iloc[0]
        out.append({
            "title": "Tyre wear proxy",
            "text": (f"{worst['Driver']} (stint {worst['Stint']}) showed "
                     f"the steepest lap-time slope "
                     f"({worst['Slope_s_per_lap']:.3f}s/lap, estimated, "
                     f"not controlled for traffic or fuel)."),
            "evidence": f"{worst['Slope_s_per_lap']:.3f}s/lap",
        })
    return out
