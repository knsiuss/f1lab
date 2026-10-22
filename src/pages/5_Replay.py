# -*- coding: utf-8 -*-
"""Race replay: OpenF1 telemetry in a browser player, plus the position chart.

The page is composed from the shared design system (``ui``): one page header,
one section per block, KPI tiles for the headline facts, and note panels for
the disclosures. No page-local inline markup.

The page is written for someone who does not follow the sport: it says in
plain words what the replay shows and what it does not show, and it ends with
a short guide to the controls and to the words used.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from typing import Any

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

import glossary as glossary_terms
from config import FASTF1_CONFIG
from lapcompare import (
    MIN_DELTA_S, MINI_SECTORS, best_lap, compare_laps, lap_trace,
    mini_sectors, resample_to_distance, summary,
)
from replay import positions_by_lap, build_position_replay
from replay_payload import build_payload, player_html
from shared import show_plotly_chart
from styles import (
    chart_height,
    fig_layout as _fig_layout,
)
import styles
from telemetry import (
    DEFAULT_FRAME_HZ, TelemetryStore, ensure_track, ingest_session,
    latest_ingested, safe_race_sessions,
)
import ui
from ui import (
    caption, confidence_tag, empty, estimate_note, kpis, note, page_header,
)

logger = logging.getLogger(__name__)

PLAYER_TEMPLATE = Path(__file__).parent.parent / 'assets' / 'replay_player.html'
# The player is an iframe, not a chart; its height comes from the shared
# layout system like the chart sizes do.
PLAYER_HEIGHT = styles.PLAYER_HEIGHT

# OpenF1 holds sessions from 2023 onwards; earlier seasons have no replay source.
EARLIEST_REPLAY_SEASON = 2023

# The payload is deterministic for a given session and resolution, and building
# it costs seconds and megabytes, so it is memoised.
PAYLOAD_TTL_S = 3600


@st.cache_resource(ttl=3600)
def get_store() -> TelemetryStore:
    """Process-wide SQLite handle.

    The store hands out one connection per thread, so two browser sessions
    never share a transaction and one session's rollback cannot discard
    another session's rows.
    """
    return TelemetryStore()


@st.cache_data(ttl=PAYLOAD_TTL_S, show_spinner=False)
def get_payload(session_key: int, frame_hz: float) -> dict:
    """Assemble the player payload, memoised per session and resolution.

    A race is ~770k telemetry samples; reading and resampling them costs
    seconds and produces a multi-megabyte payload. It does not change while
    the cached session is unchanged, so it is built once.
    """
    return build_payload(get_store(), session_key, frame_hz=frame_hz)


@st.cache_data(ttl=3600, show_spinner=False)
def get_seasons() -> list:
    """Seasons OpenF1 can serve, newest first. Cached so the page stays cheap.

    Probed rather than derived from the CSVs: replay coverage starts in 2023
    and does not depend on which seasons the app happens to hold.
    """
    upper = max(EARLIEST_REPLAY_SEASON, FASTF1_CONFIG.get_supported_years()[-1])
    seasons = []
    for year in range(upper, EARLIEST_REPLAY_SEASON - 1, -1):
        # The API being unreachable is not an empty season, but it must not
        # take the page down: the page falls back to its empty state instead
        # of crashing with a traceback.
        rows = safe_race_sessions(year)
        if rows:
            seasons.append(year)
    return seasons


@st.cache_data(ttl=900, show_spinner=False)
def get_races(year: int) -> list:
    """Race calendar for a season, as ``(label, session_key)`` pairs."""
    return [(f"{r['circuit_short_name']} — {r['session_name']}", r['session_key'])
            for r in safe_race_sessions(year)]


@st.cache_resource(ttl=900, show_spinner=False)
def get_position_chart(session_key: int, frame_hz: float):
    """The position chart, built once per session.

    The animated figure holds one trace per driver per lap (a 57-lap race with
    22 drivers is ~1,300 traces), and Plotly validates every property of every
    trace: building it costs seconds, while drawing the finished figure costs
    milliseconds. The data behind it only changes when the cached payload
    changes, so the figure is kept and reused.

    Returns ``(figure, position_rows, pivot)``; the figure is ``None`` when the
    session has no position readings.
    """
    payload = get_payload(session_key, frame_hz)
    frame = _position_frame(payload)
    if frame.empty:
        return None, frame, None
    pivot, _ = positions_by_lap(frame)
    if pivot.empty:
        return None, frame, pivot
    team_map = {d.get("acronym"): d.get("team") for d in payload.get("drivers", [])}
    fig = build_position_replay(pivot, team_map, title="")
    fig.update_layout(**_fig_layout(
        height=chart_height('feature'),
        xaxis=dict(),
        yaxis=dict(),
        title=None,
    ))
    return fig, frame, pivot


@st.cache_data(ttl=PAYLOAD_TTL_S, show_spinner=False)
def get_lap_catalogue(session_key: int) -> pd.DataFrame:
    """Every measured lap of the session, the source of the two fastest laps.

    The lap table is a few thousand rows and never changes for a session that
    is already on disk, so it is read once and shared by the comparison.
    """
    rows = get_store().table("laps", session_key, order="lap_number")
    return pd.DataFrame(rows)


@st.cache_data(ttl=PAYLOAD_TTL_S, show_spinner=False)
def get_lap_compare(session_key: int, driver_a: int, driver_b: int) -> dict:
    """Two best laps lined up by distance, memoised per session and pair.

    The expensive part of the comparison tab: two ``car_data`` windows are
    read from SQLite and each lap's distance is integrated from its speed.
    Memoised on ``(session_key, driver_a, driver_b)`` on purpose, so changing
    the pair rebuilds once and every other rerun of the fragment is free.
    """
    store = get_store()
    catalogue = get_lap_catalogue(session_key)
    metadata = {
        int(d["driver_number"]): d
        for d in store.drivers(session_key)
        if d.get("driver_number") is not None
    }
    result = {
        "available": False,
        "reason": "",
        "a": {},
        "b": {},
        "delta": pd.DataFrame(),
        "segments": [],
        "summary": {},
    }

    sides = {}
    for number in (int(driver_a), int(driver_b)):
        driver = metadata.get(number, {})
        label = _driver_label({
            "acronym": driver.get("name_acronym") or driver.get("broadcast_name"),
            "team": driver.get("team_name"),
        }) or str(number)
        lap = best_lap(catalogue, number)
        if lap is None:
            result["reason"] = (
                f"{label} has no measured lap in this session, so there is no "
                "lap to line up."
            )
            return result
        rows = store.car_data_window(session_key, number, lap["start"], lap["end"])
        trace = lap_trace(rows, lap["start"], lap["end"])
        if trace.empty:
            result["reason"] = (
                f"OpenF1 published no car data for {label}'s best lap, so that "
                "lap cannot be compared."
            )
            return result
        distance = resample_to_distance(trace)
        if distance.empty:
            result["reason"] = (
                f"{label}'s best lap is too short in the car data to line up "
                "by distance."
            )
            return result
        sides[number] = {
            "number": number,
            "label": label,
            "name": (driver.get("full_name") or driver.get("broadcast_name")
                     or label),
            "team": driver.get("team_name") or "",
            "lap": lap,
            "trace": trace,
            "distance": distance,
        }

    side_a = sides[int(driver_a)]
    side_b = sides[int(driver_b)]
    delta = compare_laps(side_a["distance"], side_b["distance"])
    if delta.empty:
        result["reason"] = (
            "The two laps do not cover enough of the same distance to compare: "
            "one of them is too short in the car data."
        )
        return result
    result.update({
        "available": True,
        "a": side_a,
        "b": side_b,
        "delta": delta,
        "segments": mini_sectors(delta, MINI_SECTORS),
        "summary": summary(
            {"driver": side_a["name"], "trace": side_a["distance"],
             "lap": side_a["lap"]},
            {"driver": side_b["name"], "trace": side_b["distance"],
             "lap": side_b["lap"]},
        ),
    })
    return result


# --------------------------------------------------------------------------
# Pure helpers (the facts the page shows, separated from the rendering)
# --------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """A duration in seconds as ``h:mm:ss`` or ``m:ss``."""
    try:
        total = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        return '--:--'
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _format_lap_time(seconds: Any) -> str:
    """A lap time in seconds as ``m:ss.mmm``."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return '--.---'
    return f"{int(value // 60)}:{value % 60:06.3f}"


def _position_frame(payload: dict) -> pd.DataFrame:
    """Lap-by-lap race position, as measured and held forward to each lap."""
    rows = [
        {"Driver": d.get("acronym"), "LapNumber": lap.get("number"),
         "Position": pos}
        for d in payload.get("drivers", [])
        for lap, pos in zip(d.get("laps", []), d.get("positionByLap", []))
        if lap.get("start") is not None and pos is not None
    ]
    return pd.DataFrame(rows, columns=["Driver", "LapNumber", "Position"])


def _kpi_rows(payload: dict) -> list:
    """The headline facts of the session, as KPI tiles.

    The coverage and the position-feed counts are part of this row on purpose:
    a reader must see how much of the field the data actually covers.
    """
    meta = payload.get("meta") or {}
    drivers = payload.get("drivers", [])
    coverage = meta.get("coverage") or {}
    drivers_total = int(coverage.get("drivers", len(drivers)))
    with_telemetry = int(coverage.get("withTelemetry", drivers_total))
    readings = sum(int(d.get("positionCount", 0)) for d in drivers)
    names = {int(d["number"]): d.get("acronym") for d in drivers if "number" in d}

    fastest = (meta.get("fastest") or [None])[0]
    trap = (meta.get("speedTrap") or [None])[0]
    fastest_name = names.get(int(fastest["driver"]), "") if fastest else ""
    trap_name = names.get(int(trap["driver"]), "") if trap else ""

    return [
        {'label': 'Laps', 'value': str(meta.get("totalLaps") or 0),
         'sub': 'laps in the race', 'tone': 'accent'},
        {'label': 'Race time', 'value': _format_duration(meta.get("duration") or 0),
         'sub': 'from start to finish'},
        {'label': 'Drivers', 'value': f"{with_telemetry}/{drivers_total}",
         'sub': 'with car data',
         'tone': 'warn' if with_telemetry < drivers_total else 'good'},
        {'label': 'Fastest lap',
         'value': _format_lap_time(fastest.get("time")) if fastest else '--.---',
         'sub': fastest_name or 'no lap recorded', 'tone': 'good'},
        {'label': 'Top speed',
         'value': f"{trap.get('speed')} km/h" if trap else '--',
         'sub': trap_name or 'fastest speed recorded'},
        {'label': 'Position readings', 'value': str(readings),
         'sub': 'sent by the position feed', 'tone': 'warn'},
    ]


def _replay_verdict(payload: dict) -> str:
    """What the replay adds up to, in one plain sentence."""
    meta = payload.get("meta") or {}
    drivers = payload.get("drivers", [])
    coverage = meta.get("coverage") or {}
    drivers_total = int(coverage.get("drivers", len(drivers)))
    with_telemetry = int(coverage.get("withTelemetry", drivers_total))
    laps = meta.get("totalLaps") or 0
    readings = sum(int(d.get("positionCount", 0)) for d in drivers)
    names = {int(d["number"]): d.get("acronym") for d in drivers if "number" in d}
    fastest = (meta.get("fastest") or [None])[0]
    fastest_name = names.get(int(fastest["driver"]), "") if fastest else ""
    fastest_time = _format_lap_time(fastest.get("time")) if fastest else ""

    text = (f"The replay follows {with_telemetry} of the {drivers_total} cars "
            f"around the circuit over {laps} laps.")
    if fastest_name and fastest_time:
        text += (f" The fastest lap in the feed was {fastest_name} in "
                 f"{fastest_time}.")
    text += (f" The public position feed sent {readings} readings, so the "
             f"running order on screen can be a little behind the real race.")
    return text


def _position_verdict(pivot) -> str:
    """What the position chart adds up to, in one plain sentence."""
    if pivot is None or pivot.empty:
        return ("There are not enough position readings in this session to say "
                "how the race order changed.")

    first, last = pivot.iloc[0], pivot.iloc[-1]
    moves = []
    for driver in pivot.columns:
        start, end = first.get(driver), last.get(driver)
        if pd.isna(start) or pd.isna(end):
            continue
        moves.append((driver, int(start) - int(end)))

    if not moves:
        return ("There are not enough position readings in this session to say "
                "how the race order changed.")

    gained = [m for m in moves if m[1] > 0]
    lost = [m for m in moves if m[1] < 0]
    stayed = len(moves) - len(gained) - len(lost)

    if len(moves) == 1:
        text = "One car is shown, from the first lap to the last. "
    else:
        text = f"{len(moves)} cars are shown, from the first lap to the last. "
    if gained:
        best = max(gained, key=lambda m: m[1])
        text += f"{best[0]} gained the most places ({best[1]}). "
    if lost:
        worst = min(lost, key=lambda m: m[1])
        text += f"{worst[0]} lost the most places ({abs(worst[1])}). "
    if stayed == 1:
        text += "One car finished where it started. "
    elif stayed > 1:
        text += f"{stayed} cars finished where they started. "
    if not gained and not lost:
        if len(moves) == 1:
            text = ("One car is shown, and it did not change position between "
                    "the first lap and the last. ")
        else:
            text = (f"{len(moves)} cars are shown, and none of them changed "
                    "position between the first lap and the last. ")

    return (text + "The order is read from the public position feed, so it "
            "can be a little behind the real race.")


def _driver_label(driver: dict) -> str:
    """A driver as the comparison names it: code plus team, not a bare code."""
    acronym = str(driver.get("acronym") or "").strip()
    team = str(driver.get("team") or "").strip()
    if acronym and team:
        return f"{acronym} - {team}"
    return acronym or team or "unknown driver"


def _surname(name: Any) -> str:
    """The last word of a driver's name, the way a broadcast says it.

    OpenF1 broadcasts the surname in capitals ("Lando NORRIS"); a sentence in
    the app is not shouted, so a fully upper-case surname is title-cased.
    """
    parts = str(name or "").strip().split()
    if not parts:
        return "That driver"
    last = parts[-1]
    return last.title() if last.isupper() else last


def _display_name(name: Any) -> str:
    """A driver's full name in sentence case ("Max VERSTAPPEN" -> "Max Verstappen")."""
    words = str(name or "").strip().split()
    if not words:
        return "That driver"
    return " ".join(word.title() if word.isupper() and len(word) > 1 else word
                    for word in words)


def _phase_label(segment: Any, total: Any) -> str:
    """A plain-language name for one of the equal-distance mini-sectors."""
    try:
        position = (float(segment) - 0.5) / max(1.0, float(total))
    except (TypeError, ValueError):
        return "one part of the lap"
    if position < 1.0 / 3.0:
        return "the first third of the lap"
    if position < 2.0 / 3.0:
        return "the middle of the lap"
    return "the last third of the lap"


def _fastest_drivers(catalogue: pd.DataFrame, numbers, limit: int = 2) -> list:
    """The fastest drivers of the session by their best usable lap.

    Only drivers with car data are considered: a driver whose laps have no
    telemetry would be a dead end for a comparison. A driver with no measured
    lap simply does not rank.
    """
    ranked = []
    for number in numbers:
        lap = best_lap(catalogue, number)
        if lap is not None:
            ranked.append((lap["duration"], int(number)))
    ranked.sort()
    return [number for _, number in ranked[:max(2, int(limit))]]


def _compare_confidence(result: dict) -> str:
    """How much measured distance a comparison rests on.

    The speed is measured, but the distance axis is integrated from it, so the
    level is capped at ``moderate``: there is no distance channel to check the
    integration against. Only a comparison with a real stretch of shared
    distance and a dense grid earns that; a short one is ``low``.
    """
    delta = result.get("delta")
    if not isinstance(delta, pd.DataFrame) or delta.empty:
        return "insufficient"
    points = int(len(delta))
    try:
        covered = float(delta["Distance"].iloc[-1])
    except (KeyError, IndexError, TypeError, ValueError):
        return "low"
    if points >= 100 and covered >= 2000.0:
        return "moderate"
    return "low"


def _compare_verdict(result: dict) -> str:
    """What the two laps add up to, in one or two plain sentences."""
    info = result.get("summary") or {}
    if not info.get("available"):
        return ("The two laps could not be lined up closely enough to say who "
                "was faster.")
    side_a, side_b = result.get("a") or {}, result.get("b") or {}
    gap = float(info.get("gap_s") or 0.0)
    distance = float(info.get("compared_distance_m") or 0.0)
    segments = len(result.get("segments") or []) or MINI_SECTORS

    if abs(gap) <= MIN_DELTA_S:
        text = (f"The two laps are level over the {distance:.0f} m they share: "
                "they reach every point at the same moment.")
    elif gap > 0:
        text = (f"{_display_name(side_b.get('name'))} was faster over "
                f"the {distance:.0f} m the two laps share, by {abs(gap):.2f} s.")
    else:
        text = (f"{_display_name(side_a.get('name'))} was faster over "
                f"the {distance:.0f} m the two laps share, by {abs(gap):.2f} s.")

    gain = info.get("biggest_gain") or {}
    if float(gain.get("delta_s") or 0.0) > MIN_DELTA_S:
        text += (f" {_surname(gain.get('driver'))} gains "
                 f"{float(gain['delta_s']):.2f} s in "
                 f"{_phase_label(gain.get('segment'), segments)}.")
    loss = info.get("biggest_loss") or {}
    # The loss segment is the one that favours the other driver most. When one
    # lap is quicker everywhere, the same driver wins every segment: naming
    # them twice would read as a contradiction, so only the gain is said.
    if (float(loss.get("delta_s") or 0.0) > MIN_DELTA_S
            and loss.get("segment") != gain.get("segment")
            and _surname(loss.get("driver")) != _surname(gain.get("driver"))):
        text += (f" {_surname(loss.get('driver'))} gains "
                 f"{float(loss['delta_s']):.2f} s in "
                 f"{_phase_label(loss.get('segment'), segments)}.")
    return text


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _render_player(payload: dict) -> None:
    """Mount the replay player.

    Rendered on every rerun on purpose: Streamlit drops any element the page
    does not re-emit, so skipping it would blank the player whenever an
    unrelated widget elsewhere in the app is touched.
    """
    template = PLAYER_TEMPLATE.read_text(encoding='utf-8')
    components.html(player_html(template, payload), height=PLAYER_HEIGHT,
                    scrolling=False)


def _load_payload(store: TelemetryStore, session_key: int, frame_hz: float):
    """Return ``(payload, was_cached)`` for a session that is on disk.

    Circuit geometry is repaired separately from telemetry: it costs one request
    against ingest's thirty, so a transient miss should never mean re-downloading
    a finished session.
    """
    cached = store.has_session(session_key)
    session = store.session(session_key) or {}
    circuit_key, year = session.get("circuit_key"), session.get("year")
    if circuit_key and year:
        ensure_track(store, int(circuit_key), int(year))

    payload = get_payload(session_key, frame_hz)
    # A payload built before the geometry arrived would keep an empty track
    # map for the whole cache lifetime; rebuild it once the circuit is there.
    if (not payload.get("track", {}).get("xs") and circuit_key and year
            and store.track(int(circuit_key), int(year))):
        get_payload.clear()
        payload = get_payload(session_key, frame_hz)
    return payload, cached


def _render_position_chart(payload: dict, session_key: int, frame_hz: float) -> None:
    """The position-swap chart below the player, driven by the same cache."""
    fig, frame, pivot = get_position_chart(session_key, frame_hz)
    if frame.empty or pivot is None or pivot.empty:
        empty("No position data for this session",
              "OpenF1 recorded no position readings, so the race order cannot "
              "be drawn.")
        ui.verdict(_position_verdict(pivot))
        return

    show_plotly_chart(fig)

    readings = sum(int(d.get("positionCount", 0)) for d in payload.get("drivers", []))
    estimate_note(
        "The position at each lap boundary is the last position OpenF1 actually "
        f"measured, held forward to the boundary. The feed recorded {readings} "
        "readings for this session, so this chart is a coarse staircase, not a "
        "per-lap measurement."
    )
    st.download_button(
        "Download lap-by-lap positions (CSV)",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=f"replay_{session_key}_positions.csv",
        mime="text/csv",
        key="replay_positions_download",
    )
    ui.verdict(_position_verdict(pivot))


@st.fragment
def _tab_positions(payload: dict, session_key: int, frame_hz: float) -> None:
    """The position chart on its own, so a rerun inside it costs one panel."""
    ui.section("Position by lap",
            note_text="estimated from the OpenF1 position feed",
            hint="Where each car stood at the end of every lap. A line that "
                 "moves up means the car gained places; a line that moves down "
                 "means it lost them.")
    _render_position_chart(payload, session_key, frame_hz)


@st.fragment
def _tab_compare(payload: dict, session_key: int) -> None:
    """Two laps lined up by distance: the picker, the chart and the table."""
    ui.section("Compare two laps",
            note_text="estimated - the distance axis is worked out from speed",
            hint="Both drivers' fastest usable lap, lined up by how far into "
                 "the lap they are rather than by the clock. Where the line is "
                 "above zero, the second driver is ahead at that point of the "
                 "circuit.")
    drivers = [d for d in payload.get("drivers", [])
               if d.get("number") is not None]
    with_telemetry = [int(d["number"]) for d in drivers if d.get("hasTelemetry")]
    if len(with_telemetry) < 2:
        empty("Not enough car data to compare two laps",
              "Fewer than two drivers in this session have car data, so there "
              "are no two laps to line up. The Position by lap tab still shows "
              "the race order.")
        return

    catalogue = get_lap_catalogue(session_key)
    fastest = _fastest_drivers(catalogue, with_telemetry)
    if len(fastest) < 2:
        empty("Not enough measured laps to compare",
              "Two drivers need a measured lap with a start clock before their "
              "laps can be compared, and this session's lap table does not "
              "hold two.")
        return

    labels = {int(d["number"]): _driver_label(d) for d in drivers}
    options = [int(d["number"]) for d in drivers]
    default_a, default_b = fastest[0], fastest[1]
    left, right = st.columns(2)
    with left:
        number_a = st.selectbox(
            "First driver",
            options,
            index=options.index(default_a) if default_a in options else 0,
            format_func=lambda number: labels.get(int(number), str(number)),
            key="replay_compare_a",
        )
    with right:
        number_b = st.selectbox(
            "Second driver",
            options,
            index=options.index(default_b) if default_b in options else 0,
            format_func=lambda number: labels.get(int(number), str(number)),
            key="replay_compare_b",
        )

    if int(number_a) == int(number_b):
        empty("Pick two different drivers",
              "The comparison needs one driver on each side. Change one of the "
              "two selections.")
        return

    result = get_lap_compare(session_key, int(number_a), int(number_b))
    if not result.get("available"):
        empty("These two laps cannot be compared",
              result.get("reason") or
              "The car data needed for one of the two laps is missing from "
              "this session.")
        return
    _render_lap_compare(result)


def _render_lap_compare(result: dict) -> None:
    """The distance-delta chart, the mini-sector table and the wording."""
    side_a, side_b = result["a"], result["b"]
    delta = result["delta"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=delta["Distance"].to_numpy(),
        y=delta["Delta"].to_numpy(),
        mode="lines",
        name="Time difference",
        line=dict(color=styles.ACCENT, width=2),
        hovertemplate="%{x:.0f} m: %{y:+.3f} s<extra></extra>",
    ))
    fig.add_hline(y=0.0, line=dict(color=styles.ZERO_LINE, width=1))
    fig.update_layout(**_fig_layout(
        height=chart_height('block'),
        xaxis=dict(),
        yaxis=dict(),
        title=None,
    ))
    fig.update_layout(
        xaxis_title=dict(text="Distance into the lap (m)", font=styles.LABEL_FONT),
        yaxis_title=dict(
            text=f"Time difference (s) - above zero {side_b['label']} is ahead",
            font=styles.LABEL_FONT,
        ),
    )
    show_plotly_chart(fig)

    estimate_note(
        "Distance is estimated: OpenF1's car data has no distance channel, so "
        "each lap's distance is integrated from its measured speed and the two "
        "laps are lined up every 10 m. Speed and lap time are measured; the "
        "distance axis, and every number read off it, is an estimate."
    )

    rows = []
    for segment in result["segments"]:
        winner = segment["winner"]
        if winner == "A":
            faster = {"value": side_a["label"], "tone": "good"}
        elif winner == "B":
            faster = {"value": side_b["label"], "tone": "good"}
        else:
            faster = {"value": "level", "tone": "dim"}
        rows.append([
            str(segment["segment"]),
            f"{segment['from_m']:.0f} - {segment['to_m']:.0f} m",
            faster,
            f"{abs(float(segment['delta'])):.3f} s",
        ])
    ui.data_table(
        ["Mini-sector", "Distance window", "Faster driver", "Average gap"],
        rows,
    )

    ui.explain(
        "Mini-sector",
        "One of eight equal slices of the distance both laps cover. The gap "
        "shown is the average time difference over that slice, so it says "
        "where time changed hands without pointing at a single corner.",
    )
    note(
        "The car data holds no tyre, fuel, engine or traffic information, so "
        "this shows where time was gained or lost, not why.",
        kind="warn",
    )
    caption(
        f"Best usable laps: {side_a['label']} lap {side_a['lap']['number']} "
        f"({_format_lap_time(side_a['lap']['duration'])}) against "
        f"{side_b['label']} lap {side_b['lap']['number']} "
        f"({_format_lap_time(side_b['lap']['duration'])}). A pit-out lap is "
        "skipped and an in-lap cannot be identified, because the car data "
        "carries no lap, tyre or pit information. Telemetry: OpenF1 "
        "(CC BY-NC-SA 4.0, non-commercial)."
    )
    confidence_tag(_compare_confidence(result))
    ui.verdict(_compare_verdict(result))


def _body():
    """Everything on the page except the closing reading guide."""
    seasons = get_seasons()
    if not seasons:
        empty("No replay data available",
              "OpenF1's archive starts with the 2023 season, and the API may be "
              "unreachable right now.")
        return

    sidebar_year = st.session_state.get('selected_year', FASTF1_CONFIG.default_year)
    # The replay's own season selector wins over the sidebar: they do not
    # cover the same range, and the header must name the season that is
    # actually on screen.
    chosen = st.session_state.get('replay_year')
    if chosen in seasons:
        pick_year = chosen
    else:
        pick_year = sidebar_year if sidebar_year in seasons else seasons[0]

    races = get_races(pick_year)
    if not races:
        empty(f"No races available for {pick_year}",
              "OpenF1 has not published this season's calendar yet.")
        return

    labels = [name for name, _ in races]
    keys = [key for _, key in races]
    store = get_store()
    # Open on the newest already-cached race: the first visit would otherwise
    # start a multi-minute download before anything is playable.
    cached_key = latest_ingested(store, pick_year)
    default_index = keys.index(cached_key) if cached_key in keys else 0

    # The race widget is keyed per season, so switching seasons starts from
    # that season's newest cached race instead of fighting a stale value.
    race_key = f"replay_race_{pick_year}"
    current = st.session_state.get(race_key)
    if current not in labels:
        current = labels[default_index]
    session_key = dict(races)[current]

    page_header(
        "Race Replay",
        "The race as an animation: where every car was on track, and the "
        "running order taken from the public position feed.",
        chips=[
            ('season', str(pick_year)),
            ('', current),
            ('', 'OpenF1'),
            ('', 'estimated track position'),
        ],
    )

    # === Session ===
    ui.section("Session",
            note_text="the telemetry is cached on disk after the first download",
            hint="Choose the season, the race and how often the car data is "
                 "recorded. Only the 2023 season onwards can be replayed.")

    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        pick_year = st.selectbox("Season", seasons,
                                 index=seasons.index(pick_year),
                                 key="replay_year")
    with c2:
        current = st.selectbox("Grand Prix", labels,
                                index=labels.index(current) if current in labels
                                else default_index,
                                key=race_key)
    with c3:
        frame_hz = st.select_slider(
            "How often the car data is recorded",
            options=[1.0, 2.0],
            value=DEFAULT_FRAME_HZ,
            format_func=lambda v: "once a second" if v <= 1.0 else "twice a second",
            key="replay_hz")

    # === Telemetry ===
    # The download is user-triggered: a race is tens of megabytes and a few
    # minutes, which must never start on its own when the page opens.
    if not store.has_session(session_key):
        ui.section("Telemetry",
                hint="The car data for this race has to be fetched once. It is "
                     "then kept on disk, so the replay opens straight away "
                     "next time.")
        note("This session is not cached yet. OpenF1 serves telemetry one "
             "driver at a time, so the first download takes a few minutes; it "
             "is then kept on disk and replays instantly.")
        if not st.button("Download telemetry", type="primary",
                         key="replay_ingest"):
            return
        with st.spinner("Downloading telemetry from OpenF1 ..."):
            ingest_session(store, session_key)
        get_payload.clear()

    try:
        payload, cached = _load_payload(store, session_key, frame_hz)
    except Exception as exc:
        logger.error("Replay ingest failed: %s", exc)
        empty("Could not load telemetry for this session", str(exc))
        return

    if not payload or not payload.get("drivers"):
        empty("No telemetry for this session",
              "OpenF1 has no car data for this race.")
        return

    coverage = payload["meta"].get("coverage") or {}
    drivers_total = coverage.get("drivers", len(payload["drivers"]))
    with_telemetry = coverage.get("withTelemetry", drivers_total)
    readings = sum(int(d.get("positionCount", 0)) for d in payload["drivers"])

    # === Session facts ===
    kpis(_kpi_rows(payload))

    # === What the replay is ===
    ui.section("What this replay shows",
            hint="So that nobody mistakes it for the official timing screen.")
    note(
        "What it shows: where each car was on the circuit, the running order "
        "from the public position feed, the lap times and the gaps from the "
        "timing data, the tyre each driver was on, and - for the drivers that "
        "have car data - the speed, the throttle and the brake at each moment. "
        + glossary_terms.short('replay'),
        kind="info",
    )
    note(
        "What it does not show: this is not the official timing feed. The "
        "running order is taken from a public position feed that sends a "
        "reading only now and then, so the order on screen can be minutes "
        "behind the real race. The time each car is shown at on the track map "
        "is an estimate, and the lap times and gaps are not corrected for "
        "safety cars or penalties.",
        kind="warn",
    )

    # === Replay ===
    ui.section("Replay",
            note_text="play, scrub and pick a driver inside the player",
            hint="The animation of the race. Use the controls under the track "
                 "to move through the race and to follow one car.")
    _render_player(payload)

    ui.steps([
        "Press PLAY (or the space bar) to start the replay, and press it again "
        "to pause.",
        "The -10s and +10s buttons move ten seconds back or forward; the left "
        "and right arrow keys do the same.",
        "Drag the slider under the track to jump to any point in the race.",
        "The speed box on the right plays the race slower or faster, from a "
        "quarter speed up to eight times faster.",
        "NEXT LAP jumps to the start of the next lap.",
        "Click a driver in the running order on the right to follow that car: "
        "its speed, throttle and brake are drawn at the bottom.",
    ], title="How to use the player")

    if not payload["track"]["xs"]:
        note("Circuit geometry could not be loaded, so the track map is blank. "
             "Timing and telemetry are unaffected.", kind="warn")
    estimate_note(
        "Track position is an estimate: a car is placed by interpolating lap "
        "progress around the circuit, because circuit x/y telemetry is not "
        "reachable from this app. Expect drift through slow corners and under "
        "safety cars."
    )
    note(
        "The running order comes from the public position feed, not from the "
        f"official timing. It sent only {readings} readings for this session, "
        "so the order shown in the running order panel can be minutes old.",
        kind="info",
    )
    if with_telemetry < drivers_total:
        note(
            f"Telemetry coverage is partial: {with_telemetry} of {drivers_total} "
            "drivers have car data for this session. A driver without it is "
            "labelled in the player rather than shown reading zero.",
            kind="warn",
        )
    caption(
        f"{'Loaded from cache' if cached else 'Downloaded'} - "
        "The lap times, the gaps, the runs of laps on each tyre and the "
        "running order are measured from the timing data. The position of "
        "each car on the track "
        "map is an estimate. Telemetry and results: OpenF1 (CC BY-NC-SA 4.0, "
        "non-commercial). Circuit geometry: MultiViewer."
    )
    ui.verdict(_replay_verdict(payload))

    # === Lap-level views ===
    # Both panels are built on every rerun: lazy tabs left the new panel blank
    # while the second script run arrived, which reads as a broken page. Each
    # panel is a fragment, so a widget inside one redraws only that panel.
    lap_tabs = st.tabs(["Position by lap", "Compare two laps"])
    with lap_tabs[0]:
        _tab_positions(payload, session_key, frame_hz)
    with lap_tabs[1]:
        _tab_compare(payload, session_key)


def page():
    """The page: the replay, then the reading guide."""
    _body()

    ui.steps([
        "Pick the race you want to watch. If it has not been downloaded yet, "
        "press 'Download telemetry' and wait a few minutes; after that it "
        "opens straight away.",
        "Watch the running order on the right and the track map on the left. "
        "The cars move because the replay is playing, not because the race is "
        "happening now.",
        "The running order is not the official timing. It comes from a public "
        "position feed that sends a reading only now and then, so it can be a "
        "little behind the real race.",
        "The chart at the bottom shows where each car stood at the end of "
        "every lap, measured from the same feed.",
        "The second tab lines two drivers' fastest laps up by distance and "
        "shows where one gained time on the other. That distance is worked out "
        "from the speed, so it is an estimate and is labelled as one.",
        "Anything marked as estimated is worked out from the data, not "
        "measured directly.",
    ], title="How to read this page")

    ui.glossary(['replay', 'stint', 'pit stop', 'fastest lap', 'delta',
                 'estimated'],
             title="What the words mean")


page()
