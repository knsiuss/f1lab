# -*- coding: utf-8 -*-
"""
openf1_results.py
~~~~~~~~~~~~~~~~~~
Season results, sprint results and qualifying grids from the OpenF1 API.

This module replaces the local CSV exports (``data/Formula1_{year}Season_*.csv``)
as the app's source of championship data. OpenF1 exposes the same information
the CSVs carried -- classification, points, grid, laps, retirement status --
through its session endpoints, so the frame returned here keeps the historical
column names and every downstream module keeps working unchanged.

What OpenF1 gives us, endpoint by endpoint:

    /sessions         every session of a season (race, sprint, qualifying) with
                      its meeting, circuit and start date
    /session_result   the classification of one session: position, laps,
                      points, gap to leader, DNF/DNS/DSQ flags
    /starting_grid    the grid order of a meeting (taken from qualifying)
    /drivers          driver number -> full name, team, team colour
    /laps             per-driver lap times (used for the optional fastest-lap
                      enrichment, never for the base frame)

Data honesty:
    * a session with no rows (cancelled event, missing classification) is
      skipped and counted, never invented;
    * a driver without telemetry for a session simply has no row -- the frame
      says "no data" rather than "0 points";
    * the fastest-lap columns are only filled when the lap data was actually
      fetched, so an empty cell means "not loaded", not "no fastest lap".

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from openf1_client import CACHE_TTL_S, cached_openf1  # noqa: E402

logger = logging.getLogger(__name__)

# OpenF1's historical archive starts with the 2023 season; earlier seasons have
# no results there and the app must say so instead of showing an empty table.
FIRST_COVERED_SEASON = 2023

# A session that finished long ago will never change again, so its response
# can be cached for a day. A session that is still running (or finished
# minutes ago) is re-fetched quickly, so the in-progress season does not show
# a stale classification.
FINAL_SESSION_TTL_S = 24 * 3600
LIVE_SESSION_TTL_S = 300

# Column order of the frame, mirroring the historical CSV exports so every
# consumer (pages, analysis, rival, report) keeps its expectations.
RESULT_COLUMNS = [
    'Track', 'Round', 'SessionType', 'SessionKey', 'MeetingKey',
    'Position', 'No', 'Driver', 'Team',
    'Starting Grid', 'Laps', 'Time/Retired', 'Status',
    'Points', 'GapToLeader', 'Duration',
    'Set Fastest Lap', 'Fastest Lap Time', 'FastestLapNumber',
]


# ---------------------------------------------------------------------------
# Raw endpoint access (cached)
# ---------------------------------------------------------------------------

def season_sessions(year: int) -> List[Dict[str, Any]]:
    """Every session OpenF1 holds for a season.

    The session list of a finished season never changes, so it is cached for
    a day; the season in progress is re-checked more often, because a new
    session appears there.
    """
    import datetime as _dt
    ttl = (FINAL_SESSION_TTL_S
           if int(year) < _dt.datetime.now().year
           else LIVE_SESSION_TTL_S)
    return cached_openf1('sessions', ttl=ttl, year=year)


def _sessions_of_type(sessions: List[Dict[str, Any]], session_type: str
                      ) -> List[Dict[str, Any]]:
    """Sessions of one type, oldest first, cancelled events removed."""
    picked = [s for s in sessions
              if s.get('session_type') == session_type and not s.get('is_cancelled')]
    return sorted(picked, key=lambda s: s.get('date_start') or '')


def _is_sprint(session: Dict[str, Any]) -> bool:
    """True for a sprint session.

    OpenF1 reports a sprint with ``session_type == 'Race'`` and
    ``session_name == 'Sprint'``, so the name -- not the type -- is what tells
    a sprint apart from a Grand Prix.
    """
    return str(session.get('session_name') or '').strip().lower() == 'sprint'


def race_sessions(year: int) -> List[Dict[str, Any]]:
    """Grand Prix sessions of a season, oldest first (sprints excluded)."""
    return [s for s in _sessions_of_type(season_sessions(year), 'Race')
            if not _is_sprint(s)]


def sprint_sessions(year: int) -> List[Dict[str, Any]]:
    """Sprint sessions of a season, oldest first."""
    return [s for s in _sessions_of_type(season_sessions(year), 'Race')
            if _is_sprint(s)]


def session_result(session_key: int, ttl: int = CACHE_TTL_S) -> List[Dict[str, Any]]:
    """Classification rows of one session."""
    return cached_openf1('session_result', ttl=ttl, session_key=session_key)


def starting_grid(meeting_key: int, session_key: Optional[int] = None,
                  ttl: int = CACHE_TTL_S) -> Dict[int, int]:
    """Grid positions of a meeting as ``{driver_number: position}``.

    A sprint weekend publishes two grids (sprint qualifying and qualifying), so
    passing ``session_key`` picks the grid set *before* that session: OpenF1
    session keys grow with the calendar, so the largest key that does not
    exceed the target is the last grid set before it. Without ``session_key``
    the latest grid of the weekend is returned - the Grand Prix grid.
    """
    rows = cached_openf1('starting_grid', ttl=ttl, meeting_key=meeting_key)
    if not rows:
        return {}
    by_session: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        key = row.get('session_key')
        by_session.setdefault(int(key) if key is not None else -1,
                              []).append(row)
    keys = sorted(by_session)
    chosen = keys[-1]
    if session_key is not None and keys:
        earlier = [key for key in keys if key <= int(session_key)]
        chosen = earlier[-1] if earlier else keys[0]
    grid: Dict[int, int] = {}
    for row in by_session[chosen]:
        number = row.get('driver_number')
        position = row.get('position')
        if number is None or position is None:
            continue
        grid[int(number)] = int(position)
    return grid


def _meeting_name(session: Dict[str, Any]) -> str:
    """Meeting name of a session, in preference to the calendar's event name.

    ``/sessions`` carries no ``meeting_name`` key, so whenever the FastF1
    calendar was unreachable every ``Track`` label came out blank. The meetings
    feed costs one request per season and is cached like the rest of it.
    """
    name = session.get('meeting_name')
    if name:
        return str(name)
    key = session.get('meeting_key')
    year = session.get('year')
    if key is None or year is None:
        return ''
    try:
        rows = cached_openf1('meetings', ttl=CACHE_TTL_S, year=int(year))
    except Exception as exc:
        logger.debug('Meeting names unavailable for %s: %s', year, exc)
        return ''
    for row in rows or []:
        if row.get('meeting_key') == key:
            return str(row.get('meeting_name') or '')
    return ''


def drivers_for_session(session_key: int, ttl: int = CACHE_TTL_S
                         ) -> Dict[int, Dict[str, Any]]:
    """Driver metadata of a session as ``{driver_number: record}``."""
    rows = cached_openf1('drivers', ttl=ttl, session_key=session_key)
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        number = row.get('driver_number')
        if number is None:
            continue
        out[int(number)] = row
    return out


def drivers_for_meeting(meeting_key: int, ttl: int = CACHE_TTL_S
                         ) -> Dict[int, Dict[str, Any]]:
    """Driver metadata of a whole meeting as ``{driver_number: record}``.

    One request covers every session of the weekend, which matters: a season
    of results is ~55 requests and the published rate limit (30/minute) is
    what a cold start costs.
    """
    rows = cached_openf1('drivers', ttl=ttl, meeting_key=meeting_key)
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        number = row.get('driver_number')
        if number is None:
            continue
        out.setdefault(int(number), row)
    return out


def fastest_laps_for_session(session_key: int, ttl: int = CACHE_TTL_S
                              ) -> Dict[int, Dict[str, Any]]:
    """Fastest lap per driver of one session.

    This is the one expensive call in the module (a race session is ~1,100 lap
    rows, half a megabyte), so it is never part of the base season frame: the
    caller asks for the session it actually wants to show.
    """
    rows = cached_openf1('laps', ttl=ttl, session_key=session_key)
    best: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        number = row.get('driver_number')
        duration = row.get('lap_duration')
        if number is None or duration is None:
            continue
        number = int(number)
        current = best.get(number)
        if current is None or float(duration) < float(current['time']):
            best[number] = {'time': float(duration),
                             'lap': int(row.get('lap_number') or 0)}
    return best


# ---------------------------------------------------------------------------
# Calendar matching
# ---------------------------------------------------------------------------

def _calendar_events(year: int) -> List[Dict[str, Any]]:
    """FastF1 calendar rows as ``[{'round', 'name', 'date'}]``.

    The FastF1 event name ("Australian Grand Prix") is the canonical track
    label in this app: it is what the race dropdowns show, so a Track column
    built from it matches the dropdowns exactly and no name translation is
    needed anywhere.
    """
    try:
        from season_config import get_season_calendar
        calendar = get_season_calendar(year)
    except Exception as exc:  # pragma: no cover - FastF1 unavailable
        logger.warning('Calendar unavailable for %s: %s', year, exc)
        return []
    events = []
    for row in calendar or []:
        date = row.get('date')
        events.append({'round': row.get('round'), 'name': row.get('name'),
                       'date': _as_date(date)})
    events = [e for e in events if e['name']]
    events.sort(key=lambda e: (e['date'] or datetime.min, e['round'] or 0))
    return events


def _as_date(value: Any) -> Optional[datetime]:
    """Coerce a calendar date into a naive UTC ``datetime``."""
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert('UTC').tz_localize(None)
    return stamp.to_pydatetime()


def _session_date(session: Dict[str, Any]) -> Optional[datetime]:
    """UTC date of a session's start."""
    raw = session.get('date_start')
    if not raw:
        return None
    try:
        stamp = datetime.fromisoformat(str(raw).replace('Z', '+00:00'))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).replace(tzinfo=None)


def match_event(events: List[Dict[str, Any]], session: Dict[str, Any]
                ) -> Optional[Dict[str, Any]]:
    """Calendar event a session belongs to (nearest race weekend).

    Race and sprint sessions are matched to the Grand Prix weekend they belong
    to, so both share one Track label. When the calendar is unavailable the
    OpenF1 meeting name is used, which keeps the frame usable (and honest:
    the label still names the right event).
    """
    when = _session_date(session)
    best = None
    best_delta = None
    for event in events:
        event_date = event.get('date')
        if when is None or event_date is None:
            continue
        delta = abs((when - event_date).total_seconds())
        if best_delta is None or delta < best_delta:
            best, best_delta = event, delta
    if best is not None and best_delta is not None:
        # A session more than a week away from any race weekend is not a race.
        if best_delta <= 7 * 86400:
            return best
    return None


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

def _driver_name(record: Dict[str, Any]) -> str:
    """Full name in "First Last" form, matching the historical CSV exports."""
    first = (record.get('first_name') or '').strip()
    last = (record.get('last_name') or '').strip()
    if first and last:
        return f'{first} {last}'
    full = (record.get('full_name') or '').strip()
    if full:
        return ' '.join(part.capitalize() for part in full.split())
    return (record.get('broadcast_name') or '').strip()


def format_duration(seconds: Optional[float]) -> str:
    """Seconds as an F1-style race time ("1:42:06.304")."""
    if seconds is None:
        return ''
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        return ''
    if total <= 0:
        return ''
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total % 60
    if hours > 0:
        return f'{hours}:{minutes:02d}:{secs:06.3f}'
    return f'{minutes}:{secs:06.3f}'


def _time_retired(row: Dict[str, Any]) -> str:
    """Reconstruct the historical "Time/Retired" column.

    The CSV carried the winner's race time, a gap ("+12.345"), a lap count
    ("+1 LAP") or a retirement status. OpenF1 splits that across
    ``duration``, ``gap_to_leader`` and the dnf/dns/dsq flags, so the column is
    rebuilt here with the same meanings.
    """
    if row.get('dns'):
        return 'DNS'
    if row.get('dsq'):
        return 'DSQ'
    if row.get('dnf'):
        return 'DNF'
    gap = row.get('gap_to_leader')
    if isinstance(gap, str):
        text = gap.strip()
        if text:
            return text
    if gap in (None, 0, 0.0):
        return format_duration(row.get('duration'))
    try:
        return f'+{float(gap):.3f}'
    except (TypeError, ValueError):
        return ''


def _status(row: Dict[str, Any]) -> str:
    """Finished / DNF / DNS / DSQ for one classification row."""
    if row.get('dns'):
        return 'DNS'
    if row.get('dsq'):
        return 'DSQ'
    if row.get('dnf'):
        return 'DNF'
    return 'Finished'


def _session_ttl(session: Dict[str, Any]) -> int:
    """Cache lifetime for a session's data.

    A finished session is immutable, so it is cached for a day; a session that
    started within the last few hours is re-checked quickly so the in-progress
    season never shows a stale classification.

    A session that has **not started yet** (the rest of the season's calendar)
    is cached for a day as well: it has no result yet, and that answer will not
    change for hours. Refetching it every few minutes would make every page
    load of the in-progress season pull ~50 requests for nothing.
    """
    when = _session_date(session)
    if when is None:
        return CACHE_TTL_S
    age = (datetime.now(timezone.utc).replace(tzinfo=None) - when).total_seconds()
    if age < 0:
        # Not run yet: the (empty) result is stable until the session starts.
        return FINAL_SESSION_TTL_S
    if age > 6 * 3600:
        return FINAL_SESSION_TTL_S
    return LIVE_SESSION_TTL_S


def build_session_rows(session: Dict[str, Any],
                       event: Optional[Dict[str, Any]],
                       session_type: str,
                       drivers: Optional[Dict[int, Dict[str, Any]]] = None
                       ) -> List[Dict[str, Any]]:
    """Rows of one session in the historical result schema.

    Args:
        session: the OpenF1 session record.
        event: the calendar event the session belongs to.
        session_type: ``Race`` or ``Sprint``.
        drivers: the driver map of the season, carried from one session to the
            next. A meeting's driver list is only fetched when it holds a
            driver number the map does not know yet (a debut, a replacement),
            which turns 24 requests per season into one.
    """
    session_key = session.get('session_key')
    ttl = _session_ttl(session)
    result = session_result(int(session_key), ttl=ttl) if session_key else []
    if not result:
        return []
    meeting_key = session.get('meeting_key')
    grid = (starting_grid(int(meeting_key), session_key=int(session_key),
                          ttl=ttl)
            if meeting_key and session_key else {})

    known = drivers if drivers is not None else {}
    numbers = {int(r['driver_number']) for r in result
               if r.get('driver_number') is not None}
    if meeting_key and not numbers.issubset(known.keys()):
        # One request per meeting covers every session of the weekend.
        for number, meta in drivers_for_meeting(int(meeting_key),
                                                 ttl=ttl).items():
            known.setdefault(number, meta)

    track = (event or {}).get('name') or _meeting_name(session) or ''
    round_no = (event or {}).get('round')
    rows: List[Dict[str, Any]] = []
    for row in result:
        number = row.get('driver_number')
        if number is None:
            continue
        number = int(number)
        meta = known.get(number, {})
        rows.append({
            'Track': track,
            'Round': round_no,
            'SessionType': session_type,
            'SessionKey': session_key,
            'MeetingKey': session.get('meeting_key'),
            'Position': row.get('position'),
            'No': number,
            'Driver': _driver_name(meta) or f'#{number}',
            'Team': meta.get('team_name') or '',
            'Starting Grid': grid.get(number),
            'Laps': row.get('number_of_laps'),
            'Time/Retired': _time_retired(row),
            'Status': _status(row),
            'Points': row.get('points'),
            'GapToLeader': row.get('gap_to_leader'),
            'Duration': row.get('duration'),
            'Set Fastest Lap': '',
            'Fastest Lap Time': '',
            'FastestLapNumber': None,
        })
    return rows


def build_season_results(year: int) -> Optional[pd.DataFrame]:
    """Race + sprint results of a season in the historical result schema.

    Returns ``None`` when the season has no data at all (an unsupported year,
    or the service is unreachable) so callers can degrade gracefully instead
    of showing an empty table that looks like "no races yet".
    """
    try:
        sessions = season_sessions(int(year))
    except Exception as exc:
        logger.error('OpenF1 session list unavailable for %s: %s', year, exc)
        return None
    if not sessions:
        logger.info('OpenF1 has no sessions for %s', year)
        return None

    events = _calendar_events(int(year))
    rows: List[Dict[str, Any]] = []
    # The driver map is carried across the whole season: a meeting's driver
    # list is only fetched when it holds a number the map does not know yet.
    drivers: Dict[int, Dict[str, Any]] = {}
    for session_type, sessions in (('Race', race_sessions(int(year))),
                                    ('Sprint', sprint_sessions(int(year)))):
        for session in sessions:
            try:
                event = match_event(events, session)
                rows.extend(build_session_rows(session, event, session_type,
                                               drivers=drivers))
            except Exception as exc:
                logger.warning('Skipping session %s: %s',
                               session.get('session_key'), exc)
    if not rows:
        return None

    frame = pd.DataFrame(rows)
    for column in RESULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[RESULT_COLUMNS]
    frame['Round'] = pd.to_numeric(frame['Round'], errors='coerce')
    frame['Position'] = pd.to_numeric(frame['Position'], errors='coerce')
    frame['Points'] = pd.to_numeric(frame['Points'], errors='coerce').fillna(0.0)
    frame['Laps'] = pd.to_numeric(frame['Laps'], errors='coerce')
    frame['Starting Grid'] = pd.to_numeric(frame['Starting Grid'], errors='coerce')
    # Race rows first, sprint rows after, then by round: the frame is in
    # calendar order, which is what the walk-forward backtest relies on.
    frame['_type_order'] = frame['SessionType'].map({'Race': 0, 'Sprint': 1})
    frame = (frame.sort_values(['Round', '_type_order', 'Position'])
             .drop(columns=['_type_order'])
             .reset_index(drop=True))
    return frame


def team_colours(year: int) -> Dict[str, str]:
    """Team name -> livery colour for a season, straight from the API.

    The OpenF1 drivers feed reports the official team colour per driver, so
    the app never has to hardcode a season's palette: a new team (or a new
    livery) is coloured correctly the moment the season is loaded, without a
    code change.

    Returns an empty mapping when the season has no data yet; callers fall
    back to their own palette then.
    """
    try:
        sessions = race_sessions(int(year)) or sprint_sessions(int(year))
    except Exception as exc:
        logger.warning('Team colours unavailable for %s: %s', year, exc)
        return {}
    for session in reversed(sessions):
        meeting_key = session.get('meeting_key')
        if meeting_key is None:
            continue
        try:
            drivers = drivers_for_meeting(int(meeting_key))
        except Exception:
            continue
        colours: Dict[str, str] = {}
        for meta in drivers.values():
            name = meta.get('team_name')
            colour = meta.get('team_colour')
            if name and colour:
                colours[str(name)] = f'#{str(colour).lstrip("#").upper()}'
        if colours:
            return colours
    return {}


def coverage_note(year: int) -> str:
    """One-line explanation for a season without results.

    The pages show this instead of a bare "no data" so the user knows
    whether the season has not started yet or is outside the archive.
    """
    try:
        year = int(year)
    except (TypeError, ValueError):
        return 'No results for this season.'
    if year < FIRST_COVERED_SEASON:
        return (f"OpenF1's archive starts with the {FIRST_COVERED_SEASON} season, "
                f"so there are no championship results for {year}.")
    return f'No completed races for {year} yet.'


def grid_for_event(year: int, event_name: str) -> Dict[str, int]:
    """Qualifying grid of one Grand Prix as ``{driver_name: position}``.

    The grid comes from the meeting's qualifying session, which is the same
    order the race started in.
    """
    if not event_name:
        return {}
    try:
        sessions = season_sessions(int(year))
        events = _calendar_events(int(year))
    except Exception as exc:
        logger.warning('Grid unavailable for %s %s: %s', year, event_name, exc)
        return {}

    meeting_key = None
    meeting_session = None
    for session in _sessions_of_type(sessions, 'Race'):
        event = match_event(events, session)
        if event and event.get('name') == event_name:
            meeting_key = session.get('meeting_key')
            meeting_session = session
            break
    if meeting_key is None:
        # Fall back to a direct name match against the OpenF1 meeting names.
        for session in sessions:
            if session.get('meeting_name') == event_name:
                meeting_key = session.get('meeting_key')
                meeting_session = session
                break
    if meeting_key is None:
        return {}

    try:
        # A finished meeting's grid never changes, so it is cached for a day
        # like the rest of its data. The race session key picks the grid that
        # was set before it, not the sprint's.
        race_key = (meeting_session or {}).get('session_key')
        grid = starting_grid(int(meeting_key),
                             session_key=int(race_key) if race_key else None,
                             ttl=_session_ttl(meeting_session or {}))
    except Exception as exc:
        logger.warning('Starting grid unavailable for %s: %s', event_name, exc)
        return {}
    if not grid:
        return {}

    # Driver metadata (number -> name/team) for the meeting: one request for
    # the whole weekend, cached like the rest of the meeting's data.
    ttl = _session_ttl(meeting_session or {})
    drivers: Dict[int, Dict[str, Any]] = {}
    try:
        drivers = drivers_for_meeting(int(meeting_key), ttl=ttl)
    except Exception:
        drivers = {}
    if not drivers:
        # Fall back to a session-level lookup.
        for session in sessions:
            if session.get('meeting_key') != meeting_key:
                continue
            key = session.get('session_key')
            if key is None:
                continue
            try:
                drivers.update(drivers_for_session(int(key), ttl=ttl))
            except Exception:
                continue
            if drivers:
                break
    return {(_driver_name(meta) or f'#{number}'): position
            for number, position in grid.items()
            for meta in [drivers.get(number, {})]}
