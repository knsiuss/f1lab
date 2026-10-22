# -*- coding: utf-8 -*-
"""
replay_payload.py
~~~~~~~~~~~~~~~~~
Assemble the cached session into the single JSON payload the player renders.

One payload covers every panel on screen. The expensive per-sample channels go
out as base64 typed arrays rather than JSON numbers, and track position travels
as a fraction around the lap instead of coordinates, so the browser can
interpolate along the racing line rather than cutting across it between frames.

Lap metadata -- boundaries, sectors, stints, gaps, race order -- stays as plain
JSON because it is small, and because the player derives state at the playhead
by searching it, which keeps scrubbing free of precomputed per-frame state.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import base64
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from telemetry import (
        DEFAULT_FRAME_HZ, FORMATION_MARGIN_S, LapWindow, TelemetryStore,
        lap_windows, team_colour,
    )
    from styles import COMPOUND_COLORS
except ImportError:
    from .telemetry import (
        DEFAULT_FRAME_HZ, FORMATION_MARGIN_S, LapWindow, TelemetryStore,
        lap_windows, team_colour,
    )
    from .styles import COMPOUND_COLORS

# Channel quantisation. Speeds are halved so a full uint8 still covers the
# fastest F1 car; pedal and DRS channels already fit natively.
SPEED_DIVISOR = 2
MAX_PEDAL = 100
MAX_SPEED_KMH = 400

# Full uint16 is a fraction of one lap, i.e. sub-decimetre precision on a 5 km
# circuit, so car ordering within a lap stays exact. The player renders the
# fraction without rounding it, so that precision survives to the screen.
FRACTION_SCALE = 65535


def encode_channel(values: np.ndarray) -> str:
    """Base64 a numeric array so the player can decode it as a typed array."""
    return base64.b64encode(values.tobytes()).decode("ascii")


def player_html(template: str, payload: Dict[str, Any]) -> str:
    """Fill the player template with the payload, inlined as JSON.

    ``<``, ``>`` and ``&`` are escaped so that no value in the payload can
    close the script block early. This is belt and braces: the player escapes
    every value again before it reaches the DOM, because the payload carries
    strings from a third-party API.
    """
    blob = json.dumps(payload)
    blob = (blob.replace("<", "\\u003c").replace(">", "\\u003e")
                .replace("&", "\\u0026"))
    return template.replace("__PAYLOAD__", blob)


def _grid_for(store: TelemetryStore, session_key: int,
              hz: float) -> Tuple[np.ndarray, float]:
    """Uniform sample times across the race, from its cached extent.

    Telemetry can start well before the first timed lap -- OpenF1 occasionally
    opens a session with pre-race data -- so the grid is clamped to the
    formation window. Without that, most of the timeline is a dead lead-in
    where nothing is on track.
    """
    extent = store.time_extent(session_key)
    if extent is None:
        return np.zeros(0), hz
    low = max(float(extent[0]), -FORMATION_MARGIN_S)
    if float(extent[1]) <= low:
        return np.zeros(0), hz
    return np.arange(low, float(extent[1]), 1.0 / hz), hz


def _lap_records(laps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lap rows in the shape the player wants: clock, durations, trap speeds."""
    return [
        {
            "number": int(lap["lap_number"]),
            "start": lap.get("start_t"),
            "duration": lap.get("lap_duration"),
            "s1": lap.get("s1"), "s2": lap.get("s2"), "s3": lap.get("s3"),
            "st": lap.get("st_speed"), "i1": lap.get("i1_speed"),
            "i2": lap.get("i2_speed"),
            "out": bool(lap.get("is_pit_out")),
        }
        for lap in laps if lap.get("lap_number")
    ]


def _pit_spans(pits: List[Dict[str, Any]], fallback_laps: List[Dict[str, Any]],
               seconds_per_lap: float) -> List[Tuple[float, float]]:
    """Pit windows as ``(start, end)`` seconds, for pinning position to the line.

    Pit timestamps and lap starts are stored in different tables, so a missing
    clock is reconstructed from the lap it happened on. The lap is looked up
    **by its number**, not by its position in the list: a driver whose first
    lap is not lap 1 would otherwise be given another driver's pit window.
    """
    spans: List[Tuple[float, float]] = []
    starts_by_lap = {lap["number"]: lap["start"] for lap in fallback_laps
                     if lap.get("start") is not None and lap.get("number") is not None}
    starts = [lap["start"] for lap in fallback_laps if lap.get("start") is not None]
    for pit in pits:
        duration = float(pit.get("pit_duration") or 0.0)
        start = pit.get("t")
        if start is None:
            start = starts_by_lap.get(int(pit.get("lap_number") or 1))
        if start is None and starts:
            index = int(pit.get("lap_number") or 1) - 1
            start = starts[min(max(index, 0), len(starts) - 1)]
        if start is None:
            continue
        spans.append((float(start), float(start) + (duration or seconds_per_lap)))
    return spans


def progress_series(grid: np.ndarray, windows: List[LapWindow],
                    pits: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Estimated lap fraction for every grid time, vectorised.

    :func:`telemetry.progress_at` is the readable reference; this is the same
    rule applied to a whole session's grid at once.
    """
    if grid.size == 0 or not windows:
        return np.zeros(grid.size)

    starts = np.array([w.start for w in windows])
    durations = np.array([w.duration for w in windows])
    index = np.searchsorted(starts, grid, side="right") - 1

    fractions = np.zeros(grid.size)
    safe_durations = np.where(durations > 0, durations, 1.0)
    inside = (index >= 0) & (index < len(windows))
    fractions[inside] = np.clip(
        (grid[inside] - starts[index[inside]]) / safe_durations[index[inside]],
        0.0, 1.0,
    )
    # Past the final lap the session is over; hold the flag rather than 0.
    fractions[index >= len(windows)] = 1.0

    for pit_start, pit_end in pits:
        fractions[(grid >= pit_start) & (grid < pit_end)] = 0.0
    return fractions


def _sample_channels(rows: List[Dict[str, Any]], grid: np.ndarray,
                     names: Sequence[str]) -> Dict[str, np.ndarray]:
    """Resample telemetry channels onto the grid by holding the last reading."""
    if not rows or grid.size == 0:
        return {name: np.zeros(grid.size, dtype=np.int16) for name in names}
    times = np.array([r["t"] for r in rows], dtype=float)
    out: Dict[str, np.ndarray] = {}
    index = np.searchsorted(times, grid, side="right") - 1
    index = np.clip(index, 0, times.size - 1)
    for name in names:
        out[name] = np.array([r[name] for r in rows], dtype=float)[index]
    return out


def build_payload(store: TelemetryStore, session_key: int,
                  frame_hz: float = DEFAULT_FRAME_HZ) -> Dict[str, Any]:
    """Assemble the full replay payload for one cached session.

    Returns ``{}`` when the session was never ingested, so a caller can render
    an empty state without a special case.
    """
    session = store.session(session_key)
    if not session:
        return {}

    track = store.track(int(session["circuit_key"]), int(session["year"]))
    drivers = store.drivers(session_key)
    control = store.race_control(session_key)
    grid, hz = _grid_for(store, session_key, frame_hz)

    xs = np.asarray((track or {}).get("xs") or [], dtype=float)
    ys = np.asarray((track or {}).get("ys") or [], dtype=float)

    driver_blocks, channels = _assemble_drivers(store, session_key, drivers, grid)

    return {
        "meta": _session_meta(session, grid, hz, driver_blocks),
        "track": _track_block(track, xs, ys),
        "grid": {"n": int(grid.size), "hz": hz, "t0": float(grid[0]) if grid.size else 0.0},
        "channels": channels,
        "drivers": driver_blocks,
        "raceControl": _control_block(control),
        "compounds": COMPOUND_COLORS,
    }


def _session_meta(session: Dict[str, Any], grid: np.ndarray, hz: float,
                  drivers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Header facts: clock window, lap count, fastest laps and trap speeds."""
    fastest: List[Dict[str, Any]] = []
    trap: List[Dict[str, Any]] = []
    for driver in drivers:
        if driver["fastestLap"]:
            fastest.append({"driver": driver["number"],
                            "lap": driver["fastestLap"]["number"],
                            "time": driver["fastestLap"]["duration"]})
        if driver["topSpeed"]:
            trap.append({"driver": driver["number"], "speed": driver["topSpeed"]})
    fastest.sort(key=lambda entry: entry["time"])
    trap.sort(key=lambda entry: entry["speed"], reverse=True)

    span = float(grid[-1] - grid[0]) if grid.size else 0.0
    with_telemetry = sum(1 for d in drivers if d.get("hasTelemetry"))
    return {
        "sessionKey": session["session_key"],
        "year": session["year"],
        "name": session["session_name"],
        "circuit": session["circuit_short_name"],
        "country": session["country_name"],
        "start": session["date_start"],
        "end": session["date_end"],
        # The *span* of the timeline, not its absolute end: the player counts
        # the clock from the first frame, and the grid can start before the
        # session origin (the formation window), so an absolute end here
        # silently truncated the last seconds of every session.
        "duration": span,
        "endTime": float(grid[-1]) if grid.size else 0.0,
        "totalLaps": max((d["lastLap"] for d in drivers), default=0),
        "fastest": fastest[:5],
        "speedTrap": trap[:10],
        "coverage": {"drivers": len(drivers), "withTelemetry": with_telemetry},
    }


def _track_block(track: Optional[Dict[str, Any]], xs: np.ndarray,
                 ys: np.ndarray) -> Dict[str, Any]:
    """Circuit outline plus corners, ready for the canvas renderer."""
    if xs.size == 0:
        return {"xs": [], "ys": [], "corners": []}
    return {
        "xs": [round(float(v), 1) for v in xs],
        "ys": [round(float(v), 1) for v in ys],
        "corners": (track or {}).get("corners") or [],
    }


def _assemble_drivers(store: TelemetryStore, session_key: int,
                      drivers: List[Dict[str, Any]],
                      grid: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Per-driver metadata plus the flattened per-sample channels."""
    blocks: List[Dict[str, Any]] = []
    fractions: List[np.ndarray] = []
    speeds: List[np.ndarray] = []
    throttles: List[np.ndarray] = []
    brakes: List[np.ndarray] = []
    drs: List[np.ndarray] = []
    gears: List[np.ndarray] = []

    t_from = float(grid[0]) if grid.size else 0.0
    t_to = float(grid[-1]) if grid.size else 0.0

    for driver in drivers:
        number = int(driver["driver_number"])
        laps = _lap_records(store.table("laps", session_key, number, "lap_number"))
        windows = lap_windows([
            {"number": lap["number"], "start": lap["start"], "duration": lap["duration"]}
            for lap in laps
        ])
        seconds = float(np.median([w.duration for w in windows])) if windows else 90.0
        pits = store.table("pits", session_key, number)
        spans = _pit_spans(pits, laps, seconds)

        # Quantise here rather than after concatenation: the channels are laid
        # out driver-major, and casting a 0..1 fraction to uint16 first would
        # truncate every value to zero before it ever reaches the encoder.
        fractions.append(_quantise_fraction(progress_series(grid, windows, spans)))

        # Only the samples inside the replay window are read: roughly half of
        # a session's telemetry sits before the grid starts. The store also
        # returns the last row before the window, so the player can hold that
        # reading instead of showing zeros.
        sampled = _sample_channels(
            store.car_data_window(session_key, number, t_from, t_to), grid,
            ("speed", "throttle", "brake", "drs", "n_gear"))
        speeds.append(_quantise_speed(sampled["speed"]))
        throttles.append(_quantise_pedal(sampled["throttle"]))
        brakes.append(_quantise_pedal(sampled["brake"]))
        drs.append(np.clip(sampled["drs"], 0, 255).astype(np.uint8))
        gears.append(sampled["n_gear"].astype(np.int8))

        blocks.append(_driver_block(driver, number, laps, windows, seconds,
                                    store, session_key, spans,
                                    store.has_car_data(session_key, number)))

    return blocks, {
        "frac": encode_channel(_concat(fractions, np.uint16)),
        "speed": encode_channel(_concat(speeds, np.uint8)),
        "throttle": encode_channel(_concat(throttles, np.uint8)),
        "brake": encode_channel(_concat(brakes, np.uint8)),
        "drs": encode_channel(_concat(drs, np.uint8)),
        "gear": encode_channel(_concat(gears, np.int8)),
    }


def _concat(channels: List[np.ndarray], dtype: Any) -> np.ndarray:
    """Stack per-driver channels driver-major: all of one driver, then the next."""
    if not channels:
        return np.zeros(0, dtype=dtype)
    return np.concatenate(channels).astype(dtype)


def _quantise_speed(values: np.ndarray) -> np.ndarray:
    """Speeds to uint8 at 2 km/h per step, clipped to the car maximum."""
    return np.clip(values / SPEED_DIVISOR, 0, MAX_SPEED_KMH // SPEED_DIVISOR).astype(np.uint8)


def _quantise_pedal(values: np.ndarray) -> np.ndarray:
    """Throttle and brake to uint8 percent. OpenF1 occasionally reports >100."""
    return np.clip(values, 0, MAX_PEDAL).astype(np.uint8)


def _quantise_fraction(values: np.ndarray) -> np.ndarray:
    """Lap fractions to uint16, the precision car ordering needs."""
    return np.clip(values * FRACTION_SCALE, 0, FRACTION_SCALE).astype(np.uint16)


def _driver_block(driver: Dict[str, Any], number: int, laps: List[Dict[str, Any]],
                  windows: List[LapWindow], seconds_per_lap: float,
                  store: TelemetryStore, session_key: int,
                  pits: List[Tuple[float, float]],
                  has_telemetry: bool = True) -> Dict[str, Any]:
    """One driver: identity, lap metadata, stints, gaps and race order.

    ``hasTelemetry`` says whether the driver has *any* car telemetry for this
    session. OpenF1's coverage is partial, and a driver without telemetry has
    to be shown as "no telemetry" rather than as a car reading zero speed: the
    two look identical on a trace and mean opposite things.
    """
    completed = [lap for lap in laps if lap.get("duration")]
    fastest = min(completed, key=lambda lap: lap["duration"], default=None)
    trap = max((lap["st"] for lap in laps if lap.get("st")), default=None)
    positions = store.table("positions", session_key, number)

    return {
        "number": number,
        "acronym": driver.get("name_acronym") or driver.get("broadcast_name") or str(number),
        "name": driver.get("full_name") or driver.get("broadcast_name") or "",
        "team": driver.get("team_name") or "",
        "colour": team_colour(driver.get("team_name"), driver.get("team_colour")),
        "firstLap": windows[0].start if windows else 0.0,
        "lastLap": max((lap["number"] for lap in laps), default=0),
        "medianLap": round(seconds_per_lap, 3),
        "fastestLap": fastest,
        "topSpeed": trap,
        "hasTelemetry": bool(has_telemetry),
        "laps": laps,
        "stints": _stint_block(store, session_key, number),
        "intervals": [[r["t"], r.get("gap_to_leader"), r.get("gap_ahead")]
                      for r in store.table("intervals", session_key, number)],
        "positions": [[r["t"], r["position"]] for r in positions],
        # How fresh the position feed is, and how much of it there is. The
        # feed is sampled a handful of times per race, so the tower shows a
        # position that can be tens of minutes old.
        "positionCount": len(positions),
        "positionUpdated": positions[-1]["t"] if positions else None,
        "positionByLap": _position_by_lap(laps, positions),
        "pits": [{"start": start, "end": end} for start, end in pits],
    }


def _position_by_lap(laps: List[Dict[str, Any]],
                     positions: List[Dict[str, Any]]) -> List[Optional[int]]:
    """Race position at the start of each lap, for the position-swap chart.

    Position is held from the last reading before the lap began, which is how a
    timing screen resolves it too. The result is therefore an *estimate* of
    the position at each lap boundary, not a measurement of it: OpenF1 samples
    the position feed only a few times per race.
    """
    times = [p["t"] for p in positions]
    values = [p["position"] for p in positions]
    resolved: List[Optional[int]] = []
    for lap in laps:
        if lap.get("start") is None or not times:
            resolved.append(None)
            continue
        index = int(np.searchsorted(times, lap["start"], side="right")) - 1
        resolved.append(values[index] if index >= 0 else values[0])
    return resolved


def _stint_block(store: TelemetryStore, session_key: int, number: int) -> List[Dict[str, Any]]:
    """Tyre stints with a resolved colour, keyed by lap range."""
    return [
        {
            "number": int(stint["stint_number"] or 0),
            "from": stint.get("lap_start"),
            "to": stint.get("lap_end"),
            "compound": stint.get("compound") or "UNKNOWN",
            "colour": COMPOUND_COLORS.get(
                (stint.get("compound") or "").upper(), COMPOUND_COLORS["SOFT"]),
            "age": stint.get("tyre_age_at_start") or 0,
        }
        for stint in store.table("stints", session_key, number, "stint_number")
    ]


def _control_block(control: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Race-control feed in the order it happened, for a scrolling log."""
    return [
        {"t": row["t"], "lap": row.get("lap_number"),
         "flag": row.get("flag"), "category": row.get("category"),
         "scope": row.get("scope"), "message": row.get("message")}
        for row in sorted(control, key=lambda row: row["t"])
    ]