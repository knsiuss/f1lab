# -*- coding: utf-8 -*-
"""Jolpica-F1 service: historical results, standings and circuits.

Jolpica (``https://api.jolpi.ca/ergast/f1/``) is the free successor to
the retired Ergast API. It is the app's primary source for
championship standings and official-style classifications
(:mod:`sources.SOURCE_PRIORITY`), with standings computed from the
OpenF1 season results as the documented fallback.

Data honesty:
    * historical tables are cached on disk (``f1_cache/jolpica/``) so a
      season costs one fetch, not one per page load;
    * a network or parse failure returns ``None`` -- the caller shows
      an explicit unavailable state, never invented standings;
    * the current (in-progress) season is re-checked every 5 minutes,
      finished seasons are immutable and cached for a day;
    * every returned frame carries ``source_name='Jolpica'`` and a
      ``data_status`` from :mod:`sources`.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests

from sources import provenance

logger = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"
SOURCE_NAME = "Jolpica"
CACHE_ROOT = Path(__file__).parent.parent / "f1_cache" / "jolpica"
FINAL_TTL_S = 24 * 3600
LIVE_TTL_S = 300
REQUEST_TIMEOUT_S = 20


def _cache_file(path: str) -> Path:
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()
    return CACHE_ROOT / f"{digest}.json"


def _read_cache(path: str, ttl: int) -> Optional[Dict[str, Any]]:
    cached = _cache_file(path)
    try:
        if cached.exists() and time.time() - cached.stat().st_mtime < ttl:
            return json.loads(cached.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # pragma: no cover - corrupt cache
        logger.warning("Jolpica cache unreadable for %s: %s", path, exc)
    return None


def _write_cache(path: str, payload: Dict[str, Any]) -> None:
    try:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        _cache_file(path).write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - read-only filesystem
        logger.warning("Jolpica cache unwritable for %s: %s", path, exc)


def _get(path: str, ttl: int) -> Optional[Dict[str, Any]]:
    """One Jolpica document, from disk cache or the network."""
    cached = _read_cache(path, ttl)
    if cached is not None:
        return cached
    try:
        response = requests.get(f"{BASE_URL}/{path}",
                                headers={"Accept": "application/json"},
                                timeout=REQUEST_TIMEOUT_S)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Jolpica request failed for %s: %s", path, exc)
        return None
    _write_cache(path, payload)
    return payload


def _ttl_for(year: int) -> int:
    """Finished seasons never change; the live season is re-checked."""
    return FINAL_TTL_S if int(year) < datetime.now().year else LIVE_TTL_S


def _stamp(frame: pd.DataFrame, dataset: str, status: str) -> pd.DataFrame:
    """Attach provenance columns (source, fetch time, status) to a frame."""
    if frame is None or frame.empty:
        return frame
    record = provenance(dataset, SOURCE_NAME, status)
    frame = frame.copy()
    frame["source_name"] = record["source_name"]
    frame["data_status"] = record["data_status"]
    frame["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return frame


def _lists(payload: Optional[Dict[str, Any]], table: str, key: str) -> list:
    """The standings/results list of a Jolpica document (``[]`` if absent)."""
    if not payload:
        return []
    try:
        node = payload["MRData"][table]
        lists = node.get("StandingsLists") or node.get("RaceTable", {}) \
            .get("Races") or []
        if isinstance(lists, dict):
            lists = [lists]
        rows: list = []
        for entry in lists:
            rows.extend(entry.get(key, []) or [])
        return rows
    except (KeyError, TypeError, AttributeError) as exc:
        logger.warning("Jolpica payload missing %s: %s", table, exc)
        return []


def seasons() -> Optional[pd.DataFrame]:
    """All seasons Jolpica knows (``None`` when the API is unreachable)."""
    payload = _get("seasons.json?limit=1000", FINAL_TTL_S)
    if payload is None:
        return None
    try:
        rows = payload["MRData"]["SeasonTable"]["Seasons"]
    except (KeyError, TypeError):
        return None
    frame = pd.DataFrame([{
        "Season": int(r["season"]),
        "Url": r.get("url", ""),
    } for r in rows if str(r.get("season", "")).isdigit()])
    return _stamp(frame, "race_results", "Available") if not frame.empty else None


def circuits(year: int) -> Optional[pd.DataFrame]:
    """Circuits of a season with coordinates (for weather/radar lookups)."""
    payload = _get(f"{int(year)}/circuits.json?limit=100", _ttl_for(year))
    if payload is None:
        return None
    try:
        rows = payload["MRData"]["CircuitTable"]["Circuits"]
    except (KeyError, TypeError):
        return None
    frame = pd.DataFrame([{
        "CircuitId": c.get("circuitId", ""),
        "Circuit": c.get("circuitName", ""),
        "Lat": float((c.get("Location") or {}).get("lat") or float("nan")),
        "Lon": float((c.get("Location") or {}).get("long") or float("nan")),
        "Locality": (c.get("Location") or {}).get("locality", ""),
        "Country": (c.get("Location") or {}).get("country", ""),
    } for c in rows])
    return _stamp(frame, "race_results", "Available") if not frame.empty else None


def driver_standings(year: int, round_number: Optional[int] = None):
    """Driver championship standings (Jolpica), best first."""
    scope = f"{int(year)}/{int(round_number)}" if round_number \
        else str(int(year))
    payload = _get(f"{scope}/driverStandings.json?limit=100", _ttl_for(year))
    rows = _lists(payload, "StandingsTable", "DriverStandings")
    if not rows:
        return None
    records = []
    for row in rows:
        driver = row.get("Driver") or {}
        constructors = row.get("Constructors") or []
        records.append({
            "Position": int(row.get("position") or 0),
            "Driver": f"{driver.get('givenName', '')} "
                      f"{driver.get('familyName', '')}".strip(),
            "Code": driver.get("code", ""),
            "Number": driver.get("permanentNumber", ""),
            "Constructor": constructors[0].get("name", "") if constructors else "",
            "Points": float(row.get("points") or 0),
            "Wins": int(row.get("wins") or 0),
        })
    frame = pd.DataFrame(records).sort_values("Position")
    return _stamp(frame, "standings", "Verified")


def constructor_standings(year: int, round_number: Optional[int] = None):
    """Constructor championship standings (Jolpica), best first."""
    scope = f"{int(year)}/{int(round_number)}" if round_number \
        else str(int(year))
    payload = _get(f"{scope}/constructorStandings.json?limit=100",
                   _ttl_for(year))
    rows = _lists(payload, "StandingsTable", "ConstructorStandings")
    if not rows:
        return None
    records = []
    for row in rows:
        constructor = row.get("Constructor") or {}
        records.append({
            "Position": int(row.get("position") or 0),
            "Constructor": constructor.get("name", ""),
            "Points": float(row.get("points") or 0),
            "Wins": int(row.get("wins") or 0),
        })
    frame = pd.DataFrame(records).sort_values("Position")
    return _stamp(frame, "standings", "Verified")


def _race_table(year: int, kind: str, round_number: int):
    """The ``Races`` list of a results document (``[]`` when absent)."""
    payload = _get(f"{int(year)}/{int(round_number)}/{kind}.json?limit=100",
                   _ttl_for(year))
    if not payload:
        return []
    try:
        return payload["MRData"]["RaceTable"]["Races"] or []
    except (KeyError, TypeError, AttributeError) as exc:
        logger.warning("Jolpica payload missing RaceTable: %s", exc)
        return []


def _driver_name(record: Dict[str, Any]) -> str:
    driver = (record or {}).get("Driver") or {}
    return f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip()


def _constructor_name(record: Dict[str, Any]) -> str:
    """Constructor name, from the list shape or the dict shape."""
    raw = (record or {}).get("Constructors", (record or {}).get("Constructor"))
    if isinstance(raw, list):
        return raw[0].get("name", "") if raw else ""
    if isinstance(raw, dict):
        return raw.get("name", "")
    return ""


def _status_text(record: Dict[str, Any]) -> str:
    """Status as text, whether the API sends a string or a dict."""
    raw = (record or {}).get("status")
    if isinstance(raw, dict):
        return raw.get("status", "")
    return str(raw or "")


def _time_text(record: Dict[str, Any], key: str = "Time") -> str:
    """Time text, whether the API sends a string or a dict."""
    raw = (record or {}).get(key)
    if isinstance(raw, dict):
        return raw.get("time", "")
    return str(raw or "")


def race_results(year: int, round_number: int):
    """Grand Prix classification of one round (Jolpica)."""
    races = _race_table(year, "results", round_number)
    rows = []
    for race in races:
        for result in race.get("Results", []) or []:
            rows.append({
                "Round": round_number,
                "Driver": _driver_name(result),
                "Code": ((result.get("Driver") or {}).get("code", "")),
                "Constructor": _constructor_name(result),
                "Position": int(result.get("position") or 0),
                "Grid": int(result.get("grid") or 0),
                "Laps": int(result.get("laps") or 0),
                "Points": float(result.get("points") or 0),
                "Status": _status_text(result),
                "Time": _time_text(result),
            })
    frame = pd.DataFrame(rows)
    return _stamp(frame, "race_results", "Verified") if not frame.empty else None


def sprint_results(year: int, round_number: int):
    """Sprint classification of one round (Jolpica)."""
    races = _race_table(year, "sprint", round_number)
    rows = []
    for race in races:
        for result in race.get("SprintResults", []) or []:
            rows.append({
                "Round": round_number,
                "Driver": _driver_name(result),
                "Code": ((result.get("Driver") or {}).get("code", "")),
                "Constructor": _constructor_name(result),
                "Position": int(result.get("position") or 0),
                "Grid": int(result.get("grid") or 0),
                "Points": float(result.get("points") or 0),
                "Status": _status_text(result),
            })
    frame = pd.DataFrame(rows)
    return _stamp(frame, "race_results", "Verified") if not frame.empty else None


def qualifying_results(year: int, round_number: int):
    """Qualifying classification of one round (Jolpica)."""
    races = _race_table(year, "qualifying", round_number)
    rows = []
    for race in races:
        for result in race.get("QualifyingResults", []) or []:
            rows.append({
                "Round": round_number,
                "Driver": _driver_name(result),
                "Code": ((result.get("Driver") or {}).get("code", "")),
                "Constructor": _constructor_name(result),
                "Position": int(result.get("position") or 0),
                "Q1": _time_text(result, "Q1"),
                "Q2": _time_text(result, "Q2"),
                "Q3": _time_text(result, "Q3"),
            })
    frame = pd.DataFrame(rows)
    return _stamp(frame, "race_results", "Verified") if not frame.empty else None


def pit_stops(year: int, round_number: int):
    """Pit stops of one round (Jolpica); durations may be absent."""
    races = _race_table(year, "pitstops", round_number)
    rows = []
    for race in races:
        for stop in race.get("PitStops", []) or []:
            rows.append({
                "Round": round_number,
                "DriverId": stop.get("driverId", ""),
                "Lap": int(stop.get("lap") or 0),
                "Stop": int(stop.get("stop") or 0),
                "Duration": stop.get("duration", ""),
                "Time": stop.get("time", ""),
            })
    frame = pd.DataFrame(rows)
    return _stamp(frame, "pit_stops", "Available") if not frame.empty else None


def constructor_standings_with_fallback(year: int, season_frame=None):
    """Constructor standings: Jolpica first, computed totals as fallback.

    Returns ``(frame, source_label)``; ``(None, 'Unavailable')`` when
    neither Jolpica nor the season results can provide the table.
    """
    table = constructor_standings(year)
    if table is not None and not table.empty:
        return table, SOURCE_NAME
    if season_frame is None or season_frame.empty \
            or "Team" not in season_frame.columns:
        return None, "Unavailable"
    totals = season_frame.groupby("Team")["Points"].apply(
        lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()
    ).sort_values(ascending=False)
    wins = (season_frame[pd.to_numeric(season_frame.get("Position"),
                                       errors="coerce") == 1]
            .groupby("Team").size())
    frame = pd.DataFrame({
        "Position": range(1, len(totals) + 1),
        "Constructor": totals.index,
        "Points": totals.values,
        "Wins": [int(wins.get(team, 0)) for team in totals.index],
    })
    frame["source_name"] = "computed_from_results"
    frame["data_status"] = "Estimated"
    frame["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return frame, "computed_from_results (fallback)"
