# -*- coding: utf-8 -*-
"""
lapcompare.py
~~~~~~~~~~~~~
Ghost-lap comparison of two laps from OpenF1 car telemetry: line two laps up
by distance and show where one driver gains or loses time against the other.

Data honesty contract
---------------------
OpenF1's ``car_data`` carries speed, throttle, brake, gear and DRS and **no
distance channel**. The distance axis is therefore *derived*: the measured
speed is integrated over the sample clock with the trapezoidal rule and every
channel is then resampled onto that distance grid (10 m by default). Every
figure that depends on distance is ``estimated`` (module constant
``ESTIMATED``) and the UI carries a confidence level next to it.

``car_data`` also has no lap number, no tyre compound or age and no track
position: a comparison can say *where* on the distance axis time was gained or
lost, never *why* (traffic, tyre state, fuel load and driving line are not in
the feed). The lap boundaries come from the ``laps`` table, not from the car
data.

Sign convention
---------------
``compare_laps`` returns ``Delta = TimeA - TimeB`` at every distance: a
positive value means driver B reached that point of the lap first, i.e.
**positive = driver B faster**, negative = driver A faster. ``mini_sectors``
keeps the same convention in its ``delta`` field and names the winner
``"A"``, ``"B"`` or ``"level"``.

All numeric outputs are plain ``float`` / ``int`` values, ready for a table.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

ESTIMATED = "estimated"

# OpenF1 car_data.p speed is in km/h; the integration needs m/s.
KMH_TO_MS = 1000.0 / 3600.0
# The distance grid the two laps are compared on.
DEFAULT_STEP_M = 10.0
# Equal-distance segments in the mini-sector table.
MINI_SECTORS = 8
# A segment whose mean time difference is inside this dead band is "level".
MIN_DELTA_S = 0.001

TRACE_COLUMNS = ("speed", "throttle", "brake", "gear", "drs")
DELTA_COLUMNS = ("Distance", "TimeA", "TimeB", "Delta")


def _frame(value) -> pd.DataFrame:
    """Any row source (DataFrame, list of dicts, sqlite rows) as a DataFrame."""
    if isinstance(value, pd.DataFrame):
        return value
    if value is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(list(value))
    except TypeError:
        return pd.DataFrame()


def _empty_trace() -> pd.DataFrame:
    """A trace with no rows, already shaped like ``lap_trace`` output."""
    return pd.DataFrame(
        {name: pd.Series(dtype=float) for name in TRACE_COLUMNS},
        index=pd.Index([], dtype=float, name="Time"),
    )


def _empty_distance() -> pd.DataFrame:
    """A distance trace with no rows, already shaped like a resampled lap."""
    return pd.DataFrame(
        {name: pd.Series(dtype=float) for name in TRACE_COLUMNS},
        index=pd.Index([], dtype=float, name="Distance"),
    )


def _empty_delta() -> pd.DataFrame:
    """A comparison with no rows."""
    return pd.DataFrame({name: pd.Series(dtype=float) for name in DELTA_COLUMNS})


def _channel(frame: pd.DataFrame, names, default: float = 0.0) -> np.ndarray:
    """One numeric channel of a frame, tolerant of a missing column.

    A missing channel is filled with ``default`` (never invented from another
    channel and never faked as a measurement).
    """
    for name in names:
        if name in frame.columns:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            return np.nan_to_num(
                values, nan=float(default), posinf=float(default), neginf=float(default)
            )
    return np.full(len(frame), float(default), dtype=float)


def _clock(frame: pd.DataFrame) -> Optional[np.ndarray]:
    """The lap clock of a trace: its index when numeric, else a Time column."""
    if len(frame) == 0:
        return None
    try:
        index = pd.to_numeric(pd.Series(frame.index, dtype=float), errors="coerce")
    except (TypeError, ValueError):
        index = None
    if index is not None and bool(index.notna().all()):
        return index.to_numpy(dtype=float)
    if "Time" in frame.columns:
        values = pd.to_numeric(frame["Time"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).all():
            return values
    return None


def _top_speed(trace) -> Optional[float]:
    """Highest measured speed of a trace, in km/h."""
    if not isinstance(trace, pd.DataFrame) or trace.empty or "speed" not in trace.columns:
        return None
    values = pd.to_numeric(trace["speed"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(round(float(values.max()), 1))


def best_lap(frame, driver=None) -> Optional[Dict[str, float]]:
    """The fastest usable lap of one driver, from the session's lap table.

    Parameters
    ----------
    frame : DataFrame or row source
        Rows of the ``laps`` table (driver_number, lap_number, start_t,
        lap_duration, is_pit_out, ...).
    driver : driver number, optional
        When given, only that driver's laps are considered.

    Returns
    -------
    dict or None
        ``{"number", "start", "end", "duration"}`` with the lap clock window
        the telemetry is cut with, or ``None`` when the driver has no usable
        lap. A lap is usable when it has a start clock and a positive
        duration; a pit-out lap is skipped because it is not a comparable lap
        of racing. An in-lap cannot be identified from this feed (the table
        carries no in-lap flag), which the page states.
    """
    laps = _frame(frame)
    if laps.empty or "lap_duration" not in laps.columns or "start_t" not in laps.columns:
        return None
    if driver is not None and "driver_number" in laps.columns:
        numbers = pd.to_numeric(laps["driver_number"], errors="coerce")
        try:
            laps = laps[numbers == float(driver)]
        except (TypeError, ValueError):
            return None
    if laps.empty:
        return None

    duration = pd.to_numeric(laps["lap_duration"], errors="coerce")
    start = pd.to_numeric(laps["start_t"], errors="coerce")
    usable = duration.notna() & (duration > 0) & start.notna()
    if "is_pit_out" in laps.columns:
        pit_out = pd.to_numeric(laps["is_pit_out"], errors="coerce").fillna(0.0)
        usable &= pit_out <= 0
    if not bool(usable.any()):
        return None

    candidates = laps[usable]
    fastest = duration[usable].idxmin()
    row = candidates.loc[fastest]
    number = row.get("lap_number")
    lap_start = float(start.loc[fastest])
    lap_duration = float(duration.loc[fastest])
    return {
        "number": int(number) if pd.notna(number) else 0,
        "start": lap_start,
        "end": lap_start + lap_duration,
        "duration": lap_duration,
    }


def lap_trace(car_df, lap_start, lap_end) -> pd.DataFrame:
    """The telemetry of one lap, indexed by seconds into the lap.

    ``car_df`` is a ``car_data`` window for one driver (``t``, ``speed``,
    ``throttle``, ``brake``, ``n_gear``, ``drs``). Rows outside
    ``[lap_start, lap_end]`` are dropped and the clock is rebased to zero at
    the first kept sample, so two laps line up on their own start line.

    Returns an empty, correctly shaped frame when the window holds fewer than
    two samples instead of raising.
    """
    empty = _empty_trace()
    frame = _frame(car_df)
    if frame.empty or lap_start is None or lap_end is None:
        return empty
    clock_column = "t" if "t" in frame.columns else ("Time" if "Time" in frame.columns else None)
    if clock_column is None:
        return empty
    try:
        start = float(lap_start)
        end = float(lap_end)
    except (TypeError, ValueError):
        return empty
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return empty

    clock = pd.to_numeric(frame[clock_column], errors="coerce").to_numpy(dtype=float)
    inside = np.isfinite(clock) & (clock >= start) & (clock <= end)
    if int(inside.sum()) < 2:
        return empty

    order = np.argsort(clock[inside], kind="stable")
    ordered = frame[inside].iloc[order]
    seconds = np.sort(clock[inside])
    values = {
        "speed": _channel(ordered, ("speed",)),
        "throttle": _channel(ordered, ("throttle",)),
        "brake": _channel(ordered, ("brake",)),
        "gear": _channel(ordered, ("n_gear", "gear")),
        "drs": _channel(ordered, ("drs",)),
    }
    return pd.DataFrame(values, index=pd.Index(seconds - start, name="Time"))


def resample_to_distance(trace, step_m: float = DEFAULT_STEP_M) -> pd.DataFrame:
    """Integrate speed into distance and resample every channel onto it.

    The trapezoidal rule is applied to the measured speed over the sample
    clock (``distance[i] = distance[i-1] + (v[i-1] + v[i]) / 2 * dt``), so the
    distance axis is **estimated**, not measured: OpenF1 publishes no distance
    channel. Continuous channels (speed, throttle, brake) are interpolated
    linearly; state channels (gear, DRS) are *held* at the last sample at or
    before the grid point, because a gear does not exist between two gears.

    Returns an empty frame (not an exception) when the trace is empty, too
    short to cover one step, or has no usable speed.
    """
    empty = _empty_distance()
    frame = trace if isinstance(trace, pd.DataFrame) else _frame(trace)
    if frame.empty or "speed" not in frame.columns:
        return empty
    try:
        step = float(step_m)
    except (TypeError, ValueError):
        return empty
    if not np.isfinite(step) or step <= 0:
        return empty

    clock = _clock(frame)
    if clock is None or clock.size < 2 or not np.isfinite(clock).all():
        return empty
    order = np.argsort(clock, kind="stable")
    ordered = frame.iloc[order]
    times = clock[order]
    if np.any(np.diff(times) < 0):
        return empty

    speed = np.clip(_channel(ordered, ("speed",)), 0.0, None)
    velocity = speed * KMH_TO_MS
    delta_t = np.diff(times)
    covered = (velocity[1:] + velocity[:-1]) / 2.0 * delta_t
    distance = np.concatenate([[0.0], np.cumsum(covered)])
    total = float(distance[-1])
    if not np.isfinite(total) or total < step:
        return empty

    grid = np.arange(0.0, total, step)
    if grid.size < 2:
        return empty

    elapsed = np.interp(grid, distance, times - times[0])
    speed_grid = np.interp(grid, distance, speed)
    throttle = np.interp(grid, distance, np.clip(_channel(ordered, ("throttle",)), 0.0, 100.0))
    brake = np.interp(grid, distance, _channel(ordered, ("brake",)))

    # State channels are held, not averaged.
    held = np.searchsorted(distance, grid, side="right") - 1
    held = np.clip(held, 0, distance.size - 1)
    gear = _channel(ordered, ("n_gear", "gear"))[held]
    drs = _channel(ordered, ("drs",))[held]

    return pd.DataFrame(
        {
            "Time": elapsed,
            "speed": speed_grid,
            "throttle": throttle,
            "brake": brake,
            "gear": gear,
            "drs": drs,
        },
        index=pd.Index(np.round(grid, 6), name="Distance"),
    )


def _common_step(*frames) -> float:
    """The coarser of the two distance grids, so neither lap is oversampled."""
    steps: List[float] = []
    for frame in frames:
        if not isinstance(frame, pd.DataFrame) or len(frame) < 2:
            continue
        try:
            diff = np.diff(np.asarray(frame.index, dtype=float))
        except (TypeError, ValueError):
            continue
        diff = diff[np.isfinite(diff) & (diff > 0)]
        if diff.size:
            steps.append(float(np.median(diff)))
    return max(steps) if steps else DEFAULT_STEP_M


def _interp_time(frame: pd.DataFrame, grid: np.ndarray) -> Optional[np.ndarray]:
    """Lap time of a resampled lap at every grid point (distance-indexed)."""
    if not isinstance(frame, pd.DataFrame) or frame.empty or "Time" not in frame.columns:
        return None
    try:
        distance = np.asarray(frame.index, dtype=float)
    except (TypeError, ValueError):
        return None
    values = pd.to_numeric(frame["Time"], errors="coerce").to_numpy(dtype=float)
    if distance.size != values.size or distance.size < 2:
        return None
    if not (np.isfinite(distance).all() and np.isfinite(values).all()):
        return None
    order = np.argsort(distance, kind="stable")
    distance, values = distance[order], values[order]
    if np.any(np.diff(distance) <= 0):
        return None
    return np.interp(grid, distance, values)


def compare_laps(trace_a, trace_b) -> pd.DataFrame:
    """Line two resampled laps up by distance and return the running delta.

    Parameters
    ----------
    trace_a, trace_b : DataFrame
        Output of :func:`resample_to_distance` (distance index, a ``Time``
        column). The two laps are compared over the distance they both cover.

    Returns
    -------
    DataFrame
        ``Distance`` (m), ``TimeA`` / ``TimeB`` (s elapsed into the lap) and
        ``Delta`` = ``TimeA - TimeB`` in seconds: **positive means driver B
        reached that distance first (B faster)**. Empty (no rows, no
        exception) when either trace is empty, shorter than one grid step, or
        the lap lines cannot be interpolated.
    """
    empty = _empty_delta()
    if not isinstance(trace_a, pd.DataFrame) or not isinstance(trace_b, pd.DataFrame):
        return empty
    if trace_a.empty or trace_b.empty:
        return empty
    if "Time" not in trace_a.columns or "Time" not in trace_b.columns:
        return empty
    try:
        end_a = float(trace_a.index[-1])
        end_b = float(trace_b.index[-1])
    except (TypeError, ValueError, IndexError):
        return empty
    overlap = min(end_a, end_b)
    step = _common_step(trace_a, trace_b)
    if not np.isfinite(overlap) or overlap < step:
        return empty

    grid = np.arange(0.0, overlap + 1e-9, step)
    grid = grid[grid <= overlap + 1e-9]
    if grid.size < 2:
        return empty

    time_a = _interp_time(trace_a, grid)
    time_b = _interp_time(trace_b, grid)
    if time_a is None or time_b is None:
        return empty

    return pd.DataFrame(
        {
            "Distance": grid,
            "TimeA": time_a,
            "TimeB": time_b,
            "Delta": time_a - time_b,
        }
    )


def mini_sectors(delta, n: int = MINI_SECTORS) -> List[dict]:
    """Split the compared distance into ``n`` equal segments.

    Each segment carries the equally spaced distance window, the average time
    difference over that window and the driver it favours (``"A"``, ``"B"`` or
    ``"level"`` when the average sits inside :data:`MIN_DELTA_S`). A segment
    with no samples is omitted rather than guessed. Returns an empty list for
    an empty or unusable comparison.
    """
    try:
        segments_n = int(n)
    except (TypeError, ValueError):
        return []
    if segments_n <= 0:
        return []
    frame = delta if isinstance(delta, pd.DataFrame) else _frame(delta)
    if frame.empty or "Distance" not in frame.columns or "Delta" not in frame.columns:
        return []

    distance = pd.to_numeric(frame["Distance"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(frame["Delta"], errors="coerce").to_numpy(dtype=float)
    usable = np.isfinite(distance) & np.isfinite(values)
    if not bool(usable.any()):
        return []
    distance, values = distance[usable], values[usable]
    low = float(distance.min())
    span = float(distance.max()) - low
    if span <= 0:
        return []
    width = span / segments_n
    bucket = np.clip(((distance - low) / width).astype(int), 0, segments_n - 1)

    out: List[dict] = []
    for segment in range(segments_n):
        selected = values[bucket == segment]
        if selected.size == 0:
            continue
        mean = float(np.mean(selected))
        if mean > MIN_DELTA_S:
            winner = "B"
        elif mean < -MIN_DELTA_S:
            winner = "A"
        else:
            winner = "level"
        out.append(
            {
                "segment": int(segment + 1),
                "from_m": float(round(low + segment * width, 1)),
                "to_m": float(round(low + (segment + 1) * width, 1)),
                "delta": float(round(mean, 3)),
                "winner": winner,
                "points": int(selected.size),
            }
        )
    return out


def _segment_swing(segment: dict, name_a: str, name_b: str) -> dict:
    """Name whoever gained in a mini-sector, with the gain as a positive value."""
    delta = float(segment["delta"])
    gainer = name_a if delta < 0 else name_b
    return {
        "segment": int(segment["segment"]),
        "driver": gainer,
        "delta_s": float(round(abs(delta), 3)),
        "from_m": float(segment["from_m"]),
        "to_m": float(segment["to_m"]),
    }


def summary(a, b) -> dict:
    """The two-lap headline: total gap, top speed and the biggest swing.

    ``a`` and ``b`` are dicts with ``driver`` (a display name) and ``trace``
    (the resampled lap). The gap is ``TimeA - TimeB`` at the furthest distance
    the two laps share, so a **positive gap means driver B was faster** over
    the compared distance. Top speeds are measured; the distance and every
    figure derived from it are ``estimated``. ``biggest_gain`` is the segment
    that favours driver A most and ``biggest_loss`` the one that favours B
    most; each names the driver who actually gained there and carries the
    gain as a positive number of seconds. Returns ``available: False`` when
    the two laps cannot be compared, never an exception.
    """
    result: Dict[str, object] = {
        "available": False,
        "estimated": ESTIMATED,
        "driver_a": "",
        "driver_b": "",
        "gap_s": None,
        "compared_distance_m": None,
        "top_speed_a_kmh": None,
        "top_speed_b_kmh": None,
        "top_speed_diff_kmh": None,
        "biggest_gain": None,
        "biggest_loss": None,
        "segments": [],
    }
    side_a = a if isinstance(a, dict) else {}
    side_b = b if isinstance(b, dict) else {}
    name_a = str(side_a.get("driver") or "A")
    name_b = str(side_b.get("driver") or "B")
    result["driver_a"] = name_a
    result["driver_b"] = name_b
    trace_a = side_a.get("trace")
    trace_b = side_b.get("trace")

    result["top_speed_a_kmh"] = _top_speed(trace_a)
    result["top_speed_b_kmh"] = _top_speed(trace_b)
    if result["top_speed_a_kmh"] is not None and result["top_speed_b_kmh"] is not None:
        result["top_speed_diff_kmh"] = float(
            round(result["top_speed_b_kmh"] - result["top_speed_a_kmh"], 1)
        )

    delta = compare_laps(trace_a, trace_b)
    if delta.empty:
        return result

    result["available"] = True
    result["gap_s"] = float(round(float(delta["Delta"].iloc[-1]), 3))
    result["compared_distance_m"] = float(round(float(delta["Distance"].iloc[-1]), 1))
    segments = mini_sectors(delta, MINI_SECTORS)
    result["segments"] = segments
    driven = [segment for segment in segments if segment["winner"] != "level"]
    if driven:
        gain = min(driven, key=lambda segment: segment["delta"])
        loss = max(driven, key=lambda segment: segment["delta"])
        result["biggest_gain"] = _segment_swing(gain, name_a, name_b)
        result["biggest_loss"] = _segment_swing(loss, name_a, name_b)
    return result
