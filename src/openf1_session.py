"""Rebuild a FastF1-shaped session from the OpenF1 public feed.

Why this module exists
----------------------
FastF1 builds its lap table from the official timing archive at
``livetiming.formula1.com``.  That endpoint now answers ``401 Unauthorized``
for the static timing files, so ``fastf1.get_session(...).load()`` cannot
produce laps in this environment ("Failed to load timing data!" /
"Finished loading data for 0 drivers").  OpenF1 publishes the same race
information through a free, keyless REST API, so the analysis pages fall
back to it: this module turns the OpenF1 tables into the small surface the
app actually uses (``laps``, ``results``, ``race_control_messages``).

Column mapping
--------------
=========================  ==========================================
FastF1 column              OpenF1 source
=========================  ==========================================
``LapTime``                ``laps.lap_duration`` (seconds)
``Sector1Time``..``3``     ``laps.duration_sector_1..3``
``LapNumber``              ``laps.lap_number``
``Time``                   ``laps.date_start``
``Driver`` / ``Team``      ``drivers.full_name`` / ``drivers.team_name``
``Compound`` / ``Stint``   ``stints.compound`` / ``stints.stint_number``
``TyreLife``               ``stints.tyre_age_at_start`` + laps on the stint
``IsInLap`` / ``IsOutLap`` ``pit.lap_number`` / ``laps.is_pit_out_lap``
``Position``               Jolpica lap timings (races only)
``IsAccurate``             assumed true (the feed publishes only timed laps)
=========================  ==========================================

One thing is genuinely missing: the **track status** (safety car, virtual
safety car).  OpenF1's ``/session_status`` table is empty for real sessions,
so the ``TrackStatus`` column is left out rather than faked, and the pages
tell the reader that a few slow laps may therefore be part of a pace
average.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from openf1_client import cached_openf1

logger = logging.getLogger(__name__)

#: App session types -> OpenF1 ``session_name`` values.
SESSION_NAME_BY_TYPE = {
    'race': 'Race',
    'sprint': 'Sprint',
    'qualifying': 'Qualifying',
    'sprint qualifying': 'Sprint Qualifying',
    'practice': 'Practice 1',
    'practice 1': 'Practice 1',
    'practice 2': 'Practice 2',
    'practice 3': 'Practice 3',
    'fp1': 'Practice 1',
    'fp2': 'Practice 2',
    'fp3': 'Practice 3',
}

#: OpenF1 compound names -> the FastF1-style names the app expects.
COMPOUNDS = {
    'soft': 'SOFT',
    'medium': 'MEDIUM',
    'hard': 'HARD',
    'intermediate': 'INTERMEDIATE',
    'wet': 'WET',
    'unknown': 'UNKNOWN',
}

#: Cache lifetime for a session that already happened (24 h is plenty).
FINISHED_TTL_S = 24 * 3600

#: A session more than this far from the calendar date belongs to another
#: weekend (practice is on Friday, the race on Sunday, so three and a half
#: days is a whole weekend).
WEEKEND_WINDOW_S = 3.5 * 86400


def _normalise(text: Any) -> str:
    return str(text or '').strip().lower()


def _as_naive_utc(value: Any) -> Optional[datetime]:
    """A datetime as a naive UTC ``datetime`` (the calendar's own format)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        stamp = pd.Timestamp(value)
    else:
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert('UTC').tz_localize(None)
    return stamp.to_pydatetime()


def _event_rows(year: int) -> List[Dict[str, Any]]:
    """Calendar rows ``[{'round', 'name', 'date'}]`` for the season."""
    from openf1_results import _calendar_events  # local import: avoids a cycle

    return _calendar_events(year)


def _event_for(year: int, race: str) -> Optional[Dict[str, Any]]:
    """The calendar row of a Grand Prix (matched by name, then by date)."""
    rows = _event_rows(year)
    if not rows:
        return None
    wanted = _normalise(race)
    for row in rows:
        if _normalise(row.get('name')) == wanted:
            return row
    # Fall back to a "contains" match so a small naming difference does not
    # hide the event ("Australian Grand Prix" vs "Australian GP").
    for row in rows:
        name = _normalise(row.get('name'))
        if wanted and (wanted in name or name in wanted):
            return row
    return None


def find_session(year: int, race: str,
                 session_type: str) -> Optional[Dict[str, Any]]:
    """The OpenF1 session dict for a Grand Prix + session type.

    The OpenF1 session is matched to the Grand Prix weekend by the nearest
    calendar date, the same way the results table is built, so both use one
    Track label.
    """
    wanted_name = SESSION_NAME_BY_TYPE.get(_normalise(session_type))
    if not wanted_name:
        return None
    sessions = cached_openf1('sessions', ttl=FINISHED_TTL_S, year=year)
    if not sessions:
        return None
    event = _event_for(year, race)
    event_date = _as_naive_utc(event.get('date')) if event else None

    candidates = [s for s in sessions
                  if _normalise(s.get('session_name')) == _normalise(wanted_name)]
    if not candidates:
        return None

    if event_date is None:
        return candidates[-1]

    def _start(session: Dict[str, Any]) -> Optional[datetime]:
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

    best = None
    best_delta = None
    for session in candidates:
        when = _start(session)
        if when is None:
            continue
        delta = abs((when - event_date).total_seconds())
        if best_delta is None or delta < best_delta:
            best, best_delta = session, delta
    if best is None:
        return None
    if best_delta is not None and best_delta > WEEKEND_WINDOW_S:
        # The nearest session of this type belongs to another weekend.  It is
        # better to show nothing than to show another Grand Prix's data under
        # this race's name.
        logger.info('No %s session for %s %s (nearest is %s days away)',
                    wanted_name, year, race, round(best_delta / 86400, 1))
        return None
    return best


def _driver_table(session_key: int) -> Dict[int, Dict[str, Any]]:
    """Driver number -> driver record for the session."""
    rows = cached_openf1('drivers', ttl=FINISHED_TTL_S, session_key=session_key)
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows or []:
        number = row.get('driver_number')
        if number is None:
            continue
        try:
            out[int(number)] = row
        except (TypeError, ValueError):
            continue
    return out


def _driver_name(record: Dict[str, Any]) -> str:
    full = (record.get('full_name') or '').strip()
    if full:
        return ' '.join(part.capitalize() for part in full.split())
    broadcast = (record.get('broadcast_name') or '').strip()
    if broadcast:
        return broadcast
    acronym = (record.get('name_acronym') or '').strip()
    return acronym or str(record.get('driver_number') or '')


def _jolpica_positions(year: int, round_number: Any
                       ) -> tuple[Dict[tuple, int], set]:
    """(driver id, lap) -> position, from the Jolpica (Ergast) lap feed.

    Only races have a lap-by-lap position feed there; qualifying and sprint
    laps come back empty, which is why the result is optional.  The set of
    driver ids in the feed is returned as well, so the app can match its own
    driver names to those ids.
    """
    if round_number in (None, ''):
        return {}, set()
    try:
        url = (f'https://api.jolpi.ca/ergast/f1/{int(year)}/{int(round_number)}'
               '/laps.json?limit=2000')
        import requests  # imported here: only the race page needs it

        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return {}, set()
        payload = response.json()
    except Exception as exc:  # pragma: no cover - network/offline
        logger.info('Jolpica lap positions unavailable: %s', exc)
        return {}, set()

    races = ((payload.get('MRData') or {}).get('RaceTable') or {}).get('Races') or []
    if not races:
        return {}, set()
    out: Dict[tuple, int] = {}
    known: set = set()
    for lap in races[0].get('Laps') or []:
        try:
            lap_number = int(lap.get('number'))
        except (TypeError, ValueError):
            continue
        for timing in lap.get('Timings') or []:
            driver_id = timing.get('driverId')
            position = timing.get('position')
            if not driver_id or position in (None, ''):
                continue
            known.add(str(driver_id))
            try:
                out[(str(driver_id), lap_number)] = int(position)
            except (TypeError, ValueError):
                continue
    return out, known


def _jolpica_driver_id(record: Dict[str, Any], known: set) -> str:
    """The Jolpica driver id of an OpenF1 driver record.

    Jolpica identifies drivers by surname (``hamilton``, ``max_verstappen``),
    so the app's own driver names are matched on their last word.
    """
    full = (record.get('full_name') or '').strip().lower()
    candidates = []
    if full:
        parts = [part for part in full.split() if part]
        candidates.append('_'.join(parts))
        candidates.append(parts[-1])
    for candidate in candidates:
        if candidate in known:
            return candidate
    return ''


def _stint_lookup(session_key: int) -> Dict[int, List[Dict[str, Any]]]:
    """Driver number -> stints of the session."""
    rows = cached_openf1('stints', ttl=FINISHED_TTL_S, session_key=session_key)
    out: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows or []:
        number = row.get('driver_number')
        if number is None:
            continue
        try:
            out.setdefault(int(number), []).append(row)
        except (TypeError, ValueError):
            continue
    for stints in out.values():
        stints.sort(key=lambda s: s.get('lap_start') or 0)
    return out


def _pit_events(session_key: int) -> Dict[int, List[Dict[str, Any]]]:
    """Driver number -> pit events of the session.

    Each event carries the lap the car entered the pits, the timestamp of the
    stop and how long the stop took.
    """
    rows = cached_openf1('pit', ttl=FINISHED_TTL_S, session_key=session_key)
    out: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows or []:
        number = row.get('driver_number')
        lap = row.get('lap_number')
        if number is None or lap is None:
            continue
        when = None
        raw = row.get('date')
        if raw:
            try:
                when = pd.to_datetime(raw, utc=True)
            except (TypeError, ValueError):
                when = None
        duration = row.get('stop_duration')
        if duration is None:
            duration = row.get('pit_duration')
        try:
            duration = float(duration) if duration is not None else None
        except (TypeError, ValueError):
            duration = None
        try:
            out.setdefault(int(number), []).append(
                {'lap': int(lap), 'date': when, 'duration': duration})
        except (TypeError, ValueError):
            continue
    for events in out.values():
        events.sort(key=lambda e: e['lap'])
    return out


def _derived_positions(frame: pd.DataFrame, pit_seconds: List[float]
                       ) -> List[Optional[int]]:
    """The running order at the end of every lap, derived from lap times.

    The public feed does not publish the official running order lap by lap,
    so the order is rebuilt from the elapsed race time: every driver's lap
    times added up, plus the time lost in the pit lane.  A car that has
    driven fewer laps (or a slower total) sits further back, which is the
    same arithmetic the official gap uses.  The result is an **estimate**
    and the pages label it as one.
    """
    if frame.empty or 'LapTime' not in frame.columns:
        return []
    work = frame[['Driver', 'LapNumber', 'LapTime']].copy()
    work['_stop'] = list(pit_seconds)
    work['LapTime'] = pd.to_timedelta(work['LapTime'], errors='coerce')
    work['_lap_s'] = work['LapTime'].dt.total_seconds()
    work = work.sort_values(['Driver', 'LapNumber'], kind='stable')
    work['_elapsed'] = (work.groupby('Driver')['_lap_s'].cumsum()
                        + work.groupby('Driver')['_stop'].cumsum())
    lookup: Dict[tuple, int] = {}
    for lap_num, group in work.groupby('LapNumber', sort=True):
        ranked = group.sort_values('_elapsed')
        for index, (_, row) in enumerate(ranked.iterrows(), start=1):
            lookup[(row['Driver'], int(lap_num))] = index
    out: List[Optional[int]] = []
    for driver, lap in zip(frame['Driver'], frame['LapNumber']):
        if lap is None or pd.isna(lap):
            out.append(None)
            continue
        out.append(lookup.get((driver, int(lap))))
    return out


def build_laps(session_key: int, year: int = 0,
               round_number: Any = None,
               official_positions: bool = False) -> pd.DataFrame:
    """The lap table of a session, in the shape the app expects.

    ``official_positions`` allows the official lap-by-lap running order to be
    used.  That order only exists for the race of a weekend: the public lap
    feed it comes from holds the race laps, so applying it to a sprint or a
    qualifying session would put the wrong order on the wrong laps.

    Returns an empty frame when the session has no lap data (a weekend that
    has not run yet, for example).
    """
    rows = cached_openf1('laps', ttl=FINISHED_TTL_S, session_key=session_key)
    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    drivers = _driver_table(session_key)
    stints = _stint_lookup(session_key)
    pits = _pit_events(session_key)

    frame = pd.DataFrame()
    frame['DriverNumber'] = pd.to_numeric(raw.get('driver_number'), errors='coerce')
    frame['LapNumber'] = pd.to_numeric(raw.get('lap_number'), errors='coerce')
    frame['LapTime'] = pd.to_timedelta(
        pd.to_numeric(raw.get('lap_duration'), errors='coerce'), unit='s')
    frame['Time'] = pd.to_datetime(raw.get('date_start'), errors='coerce',
                                   utc=True)
    for index, column in enumerate(('Sector1Time', 'Sector2Time', 'Sector3Time')):
        source = f'duration_sector_{index + 1}'
        frame[column] = pd.to_timedelta(
            pd.to_numeric(raw.get(source), errors='coerce'), unit='s')

    def _name(number: Any) -> str:
        try:
            return _driver_name(drivers.get(int(number), {}))
        except (TypeError, ValueError):
            return ''

    def _team(number: Any) -> str:
        try:
            return str(drivers.get(int(number), {}).get('team_name') or '')
        except (TypeError, ValueError):
            return ''

    frame['Driver'] = frame['DriverNumber'].map(_name)
    frame['Team'] = frame['DriverNumber'].map(_team)
    frame['IsAccurate'] = True
    frame['Deleted'] = False

    # Tyre: the stint that covers the lap gives compound, stint number and
    # how old the tyre was at the start of the stint.
    compounds: List[str] = []
    stint_numbers: List[Any] = []
    tyre_lives: List[Any] = []
    for number, lap in zip(frame['DriverNumber'], frame['LapNumber']):
        compound, stint_no, tyre_life = 'UNKNOWN', None, None
        try:
            for stint in stints.get(int(number), []):
                start = stint.get('lap_start')
                end = stint.get('lap_end')
                if start is None:
                    continue
                if lap is None or pd.isna(lap):
                    continue
                if lap >= start and (end is None or lap <= end):
                    compound = COMPOUNDS.get(_normalise(stint.get('compound')),
                                            'UNKNOWN')
                    stint_no = stint.get('stint_number')
                    age = stint.get('tyre_age_at_start')
                    try:
                        age = int(age) if age is not None else 0
                    except (TypeError, ValueError):
                        age = 0
                    tyre_life = age + int(lap - start) + 1
                    break
        except (TypeError, ValueError):
            pass
        compounds.append(compound)
        stint_numbers.append(stint_no)
        tyre_lives.append(tyre_life)
    frame['Compound'] = compounds
    frame['Stint'] = stint_numbers
    frame['TyreLife'] = tyre_lives

    # Pit laps: the lap the car entered the pits (in-lap) and the lap it
    # drove out on (out-lap).  The out-lap flag comes from the lap feed
    # itself, the in-lap from the pit feed.
    out_flags = raw.get('is_pit_out_lap')
    if out_flags is None:
        out_flags = pd.Series([False] * len(frame))
    frame['IsOutLap'] = [bool(value) for value in out_flags]
    in_flags: List[bool] = []
    pit_in: List[Any] = []
    pit_out: List[Any] = []
    pit_seconds: List[float] = []
    for number, lap in zip(frame['DriverNumber'], frame['LapNumber']):
        flag = False
        when = None
        out_time = None
        stop_seconds = 0.0
        try:
            events = pits.get(int(number), [])
        except (TypeError, ValueError):
            events = []
        for event in events:
            if lap is None or pd.isna(lap):
                continue
            if int(lap) == event['lap']:
                flag = True
                when = event['date']
                if event['duration'] is not None:
                    stop_seconds = float(event['duration'])
            if int(lap) == event['lap'] + 1:
                if event['date'] is not None and event['duration'] is not None:
                    out_time = event['date'] + timedelta(
                        seconds=float(event['duration']))
        in_flags.append(flag)
        pit_in.append(when)
        pit_out.append(out_time)
        pit_seconds.append(stop_seconds)
    frame['IsInLap'] = in_flags
    frame['PitInTime'] = pit_in
    frame['PitOutTime'] = pit_out

    # FastF1 stores the lap and pit times as elapsed times from the start of
    # the session, not as wall-clock timestamps, and the rest of the app
    # measures a pit stop as "the time the car left the lane minus the time
    # it entered".  The same convention is used here so the numbers mean the
    # same thing whichever feed the session came from.
    origin = frame['Time'].min()
    if pd.notna(origin):
        frame['Time'] = frame['Time'] - origin
        frame['PitInTime'] = [
            (value - origin) if value is not None and pd.notna(value) else None
            for value in frame['PitInTime']]
        frame['PitOutTime'] = [
            (value - origin) if value is not None and pd.notna(value) else None
            for value in frame['PitOutTime']]

    # Position: the official lap-by-lap order when the feed publishes it
    # (races only), otherwise the order derived from the elapsed race time.
    positions, jolpica_ids = (_jolpica_positions(year, round_number)
                              if (round_number and official_positions)
                              else ({}, set()))
    official: List[Optional[int]] = []
    if positions:
        by_driver: Dict[int, str] = {}
        for number, record in drivers.items():
            jolpica_id = _jolpica_driver_id(record, jolpica_ids)
            if jolpica_id:
                by_driver[number] = jolpica_id
        official = [
            positions.get((by_driver.get(int(number), ''), int(lap)))
            if number is not None and lap is not None
            and not pd.isna(number) and not pd.isna(lap) else None
            for number, lap in zip(frame['DriverNumber'], frame['LapNumber'])
        ]
    covered = sum(1 for value in official if value is not None)
    if official and covered >= 0.8 * len(frame):
        frame['Position'] = official
        position_source = 'official'
    else:
        # The public feed only publishes part of the running order, so a
        # half-filled column would be worse than no column: the order is
        # rebuilt from the elapsed race time and labelled as an estimate.
        frame['Position'] = _derived_positions(frame, pit_seconds)
        position_source = 'estimated'

    frame = frame.sort_values(['DriverNumber', 'LapNumber']).reset_index(drop=True)
    frame.attrs['position_source'] = position_source
    frame.attrs['track_status'] = 'unavailable'
    return frame


def build_results(session_key: int) -> pd.DataFrame:
    """The classification of a session, in the shape the app expects."""
    rows = cached_openf1('session_result', ttl=FINISHED_TTL_S,
                         session_key=session_key)
    if not rows:
        return pd.DataFrame()
    drivers = _driver_table(session_key)
    frame = pd.DataFrame()
    frame['Position'] = pd.to_numeric(
        pd.Series([r.get('position') for r in rows]), errors='coerce')
    frame['DriverNumber'] = pd.to_numeric(
        pd.Series([r.get('driver_number') for r in rows]), errors='coerce')
    frame['Abbreviation'] = [
        str(drivers.get(int(r['driver_number']), {}).get('name_acronym') or '')
        if r.get('driver_number') is not None else ''
        for r in rows]
    frame['Driver'] = [
        _driver_name(drivers.get(int(r['driver_number']), {}))
        if r.get('driver_number') is not None else ''
        for r in rows]
    frame['Team'] = [
        str(drivers.get(int(r['driver_number']), {}).get('team_name') or '')
        if r.get('driver_number') is not None else ''
        for r in rows]
    frame['Laps'] = pd.to_numeric(
        pd.Series([r.get('number_of_laps') for r in rows]), errors='coerce')
    frame['Points'] = pd.to_numeric(
        pd.Series([r.get('points') for r in rows]), errors='coerce')
    frame['Status'] = [
        'DNF' if r.get('dnf') else ('DNS' if r.get('dns') else
                                    ('DSQ' if r.get('dsq') else 'Finished'))
        for r in rows]
    frame['GapToLeader'] = [r.get('gap_to_leader') for r in rows]
    frame = frame.sort_values('Position').reset_index(drop=True)
    return frame


@dataclass
class OpenF1Session:
    """The small slice of a FastF1 session the app reads.

    ``source`` tells the pages where the numbers came from so the reader is
    told the truth: lap times come from the OpenF1 feed, not from FastF1.
    """

    laps: pd.DataFrame = field(default_factory=pd.DataFrame)
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    race_control_messages: Optional[Any] = None
    source: str = 'OpenF1'
    session_key: Optional[int] = None
    track_status_available: bool = False
    position_source: str = 'estimated'

    def __bool__(self) -> bool:
        return not self.laps.empty


def build_session(year: int, race: str, session_type: str
                  ) -> Optional[OpenF1Session]:
    """Build a session from OpenF1, or ``None`` when there is no data."""
    session = find_session(year, race, session_type)
    if not session:
        logger.info('No OpenF1 session for %s %s %s', year, race, session_type)
        return None
    session_key = session.get('session_key')
    if session_key is None:
        return None
    event = _event_for(year, race)
    is_race = _normalise(session.get('session_name')) == 'race'
    laps = build_laps(int(session_key), year=year,
                      round_number=event.get('round') if event else None,
                      official_positions=is_race)
    if laps.empty:
        logger.info('OpenF1 session %s has no lap data yet', session_key)
        return None
    results = build_results(int(session_key))
    return OpenF1Session(laps=laps, results=results, session_key=int(session_key),
                         position_source=str(laps.attrs.get('position_source',
                                                            'estimated')))
