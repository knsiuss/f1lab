# -*- coding: utf-8 -*-
"""Team-level season views: pit stops, reliability and a season summary.

Everything here is derived from the OpenF1 feeds the app already caches
(``/pit``, ``/session_result``, ``/drivers``) so a page can answer "who has the
quickest pit crew?" and "who keeps retiring?" with measured numbers instead of
impressions.

Two rules from the project's data-honesty guide apply throughout:

* a stop whose duration was not published is skipped and counted, never
  replaced by a default value;
* every aggregate carries the size of the sample it was computed from and a
  confidence label from ``pace.data_quality``.
"""
from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional

import pandas as pd

import openf1_client
import openf1_results
from pace import data_quality

logger = logging.getLogger(__name__)

ESTIMATED = 'estimated'

# A pit stop shorter than this is a data artefact, not a stop.
MIN_PLAUSIBLE_STOP_S = 1.0
# A stop longer than this is a stop-and-go, a repair, or a red flag.
MAX_PLAUSIBLE_STOP_S = 180.0


def _rows(endpoint: str, **params) -> List[Dict[str, Any]]:
    """Read one OpenF1 collection, returning [] instead of raising."""
    try:
        return openf1_client.cached_openf1(endpoint, **params)
    except Exception as exc:  # network down, rate limited, unexpected shape
        logger.info('OpenF1 %s unavailable (%s)', endpoint, exc)
        return []


def _driver_teams(meeting_key: int) -> Dict[int, Dict[str, str]]:
    """driver_number -> {'driver': name, 'team': team, 'colour': hex}."""
    mapping: Dict[int, Dict[str, str]] = {}
    # drivers_for_meeting returns {driver_number: record}, not a list.
    records = openf1_results.drivers_for_meeting(meeting_key) or {}
    rows = records.values() if hasattr(records, 'values') else records
    for row in rows:
        number = row.get('driver_number')
        if number is None:
            continue
        mapping[int(number)] = {
            'driver': row.get('full_name') or row.get('broadcast_name') or '',
            'team': row.get('team_name') or '',
            'colour': row.get('team_colour') or '',
        }
    return mapping


def _quality(n: int) -> Dict[str, Any]:
    """Confidence label for a sample size (pace.data_quality, never guessed)."""
    return data_quality(int(n or 0))


def _clean_duration(value) -> Optional[float]:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds < MIN_PLAUSIBLE_STOP_S or seconds > MAX_PLAUSIBLE_STOP_S:
        return None
    return seconds


def pit_stops(year: int, max_races: Optional[int] = None) -> pd.DataFrame:
    """Every measured pit stop of the season, one row per stop.

    Columns: round, track, team, driver, lap, duration_s.
    """
    rows: List[Dict[str, Any]] = []
    sessions = openf1_results.race_sessions(year) or []
    if max_races:
        sessions = sessions[-int(max_races):]

    for session in sessions:
        session_key = session.get('session_key')
        meeting_key = session.get('meeting_key')
        if session_key is None or meeting_key is None:
            continue
        teams = _driver_teams(int(meeting_key))
        for stop in _rows('pit', session_key=int(session_key)):
            duration = _clean_duration(stop.get('pit_duration'))
            if duration is None:
                continue
            number = stop.get('driver_number')
            info = teams.get(int(number), {}) if number is not None else {}
            rows.append({
                'round': session.get('round_number'),
                'track': session.get('location') or session.get('circuit_short_name'),
                'team': info.get('team', ''),
                'driver': info.get('driver', ''),
                'lap': stop.get('lap_number'),
                'duration_s': duration,
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame['duration_s'] = pd.to_numeric(frame['duration_s'], errors='coerce')
    return frame


def pit_stop_league(year: int, max_races: Optional[int] = None) -> pd.DataFrame:
    """Per-team pit stop comparison: median, mean, best and the sample size.

    A team with fewer than two measured stops is reported with its stops as
    they are but the confidence label stays 'insufficient'.
    """
    stops = pit_stops(year, max_races=max_races)
    columns = ['team', 'stops', 'median_s', 'mean_s', 'best_s', 'worst_s',
               'level', 'confidence']
    if stops.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for team, group in stops.groupby('team'):
        values = [value for value in group['duration_s'] if pd.notna(value)]
        if not values:
            continue
        rows.append({
            'team': team,
            'stops': len(values),
            'median_s': round(statistics.median(values), 2),
            'mean_s': round(statistics.fmean(values), 2),
            'best_s': round(min(values), 2),
            'worst_s': round(max(values), 2),
            'level': _quality(len(values))['level'],
            'confidence': _quality(len(values))['confidence'],
        })
    frame = pd.DataFrame(rows).sort_values('median_s').reset_index(drop=True)
    return frame


def reliability(year: int, max_races: Optional[int] = None) -> pd.DataFrame:
    """Per team: starts, finishes, retirements and the finishing rate.

    A retirement is a result row flagged dnf/dns/dsq - the numbers come from
    the classification, so a car that retired after being classified still
    counts as a finish.
    """
    columns = ['team', 'starts', 'finishes', 'retirements', 'finish_rate',
               'level', 'confidence']
    sessions = openf1_results.race_sessions(year) or []
    if max_races:
        sessions = sessions[-int(max_races):]

    tally: Dict[str, Dict[str, int]] = {}
    for session in sessions:
        meeting_key = session.get('meeting_key')
        session_key = session.get('session_key')
        if meeting_key is None or session_key is None:
            continue
        teams = _driver_teams(int(meeting_key))
        for row in openf1_results.session_result(int(session_key)) or []:
            number = row.get('driver_number')
            team = teams.get(int(number), {}).get('team') if number is not None else None
            if not team:
                continue
            entry = tally.setdefault(team, {'starts': 0, 'finishes': 0})
            entry['starts'] += 1
            retired = bool(row.get('dnf') or row.get('dns') or row.get('dsq'))
            if not retired:
                entry['finishes'] += 1

    if not tally:
        return pd.DataFrame(columns=columns)

    rows = []
    for team, entry in tally.items():
        starts = entry['starts']
        finishes = entry['finishes']
        rows.append({
            'team': team,
            'starts': starts,
            'finishes': finishes,
            'retirements': starts - finishes,
            'finish_rate': round(100.0 * finishes / starts, 1) if starts else 0.0,
            'level': _quality(starts)['level'],
            'confidence': _quality(starts)['confidence'],
        })
    return (pd.DataFrame(rows)
            .sort_values(['finish_rate', 'team'], ascending=[False, True])
            .reset_index(drop=True))


def team_season_summary(year: int) -> pd.DataFrame:
    """Points, wins and podiums per team from the season results frame.

    Points include sprints; wins and podiums are Grand Prix results only (the
    page says so next to the table).
    """
    columns = ['team', 'points', 'wins', 'podiums', 'best_finish']
    frame = openf1_results.build_season_results(year)
    if frame is None or frame.empty or 'Team' not in frame.columns:
        return pd.DataFrame(columns=columns)

    points = frame.groupby('Team')['Points'].sum()
    races = frame[frame.get('SessionType', 'Race') == 'Race'] if 'SessionType' in frame.columns else frame
    wins = races[races['Position'] == 1].groupby('Team').size() if 'Position' in races.columns else pd.Series(dtype=int)
    podiums = (races[races['Position'].isin([1, 2, 3])].groupby('Team').size()
               if 'Position' in races.columns else pd.Series(dtype=int))
    best = (races.groupby('Team')['Position'].min()
            if 'Position' in races.columns else pd.Series(dtype=float))

    rows = []
    for team, value in points.sort_values(ascending=False).items():
        rows.append({
            'team': team,
            'points': int(value),
            'wins': int(wins.get(team, 0)),
            'podiums': int(podiums.get(team, 0)),
            'best_finish': (int(best.get(team)) if pd.notna(best.get(team, None))
                            else None),
        })
    return pd.DataFrame(rows, columns=columns)


def skipped_stop_count(year: int, max_races: Optional[int] = None) -> int:
    """How many published stop rows carried no usable duration."""
    skipped = 0
    sessions = openf1_results.race_sessions(year) or []
    if max_races:
        sessions = sessions[-int(max_races):]
    for session in sessions:
        session_key = session.get('session_key')
        if session_key is None:
            continue
        for stop in _rows('pit', session_key=int(session_key)):
            if _clean_duration(stop.get('pit_duration')) is None:
                skipped += 1
    return skipped
