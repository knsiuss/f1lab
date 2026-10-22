# -*- coding: utf-8 -*-
"""Gap radar, team radio and race control, straight from OpenF1.

Three feeds that only exist while (or just after) a session runs:

``/intervals``
    the gap of every car to the leader and the interval to the car ahead,
    published every few seconds while the session is live.
``/team_radio``
    the radio messages OpenF1 publishes, with the recording URL.
``/race_control``
    the steward messages: flags, investigations, penalties.

The three are deliberately separate readers with the same shape: a frame with
one row per event, an honest empty frame when the feed published nothing, and
no invented values. FastF1 sessions carry none of this (the timing archive is
closed), so the module is keyed by the OpenF1 ``session_key`` and the caller
must say so when it has no key.

Data honesty
------------
* A gap that OpenF1 did not publish is a missing row, never a zero.
* Radio and race-control timestamps are parsed as UTC and labelled as such.
* ``feed_status`` returns the row count of each feed so a page can say "this
  session published no radio" instead of showing an empty box.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

import openf1_client

log = logging.getLogger(__name__)

# A live session's gaps change every few seconds; finished sessions are cached
# for a day by the client's own TTL policy, this is only the in-process slice.
CACHE_TTL_S = 300

GAP_COLUMNS = ['Driver', 'DriverNumber', 'Date', 'GapToLeader', 'Interval']
RADIO_COLUMNS = ['Date', 'Driver', 'DriverNumber', 'RecordingUrl']
CONTROL_COLUMNS = ['Date', 'Lap', 'Category', 'Flag', 'Message', 'Driver']

NOT_PUBLISHED = 'not published'


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def _rows(endpoint: str, **params) -> List[Dict[str, Any]]:
    """Rows of one OpenF1 endpoint, empty list when it publishes nothing."""
    try:
        rows = openf1_client.cached_openf1(endpoint, ttl=CACHE_TTL_S, **params)
    except Exception as exc:  # a feed nobody publishes is not an error
        log.debug('OpenF1 %s unavailable: %s', endpoint, exc)
        return []
    return list(rows or [])


def _driver_names(session_key: int) -> Dict[int, str]:
    """``{driver_number: name}`` for one session."""
    names: Dict[int, str] = {}
    for row in _rows('drivers', session_key=session_key):
        number = row.get('driver_number')
        if number is None:
            continue
        name = (row.get('full_name') or row.get('broadcast_name')
                or row.get('name_acronym') or str(number))
        names.setdefault(int(number), str(name))
    return names


def _to_utc(series: pd.Series) -> pd.Series:
    """Parse OpenF1 timestamps (ISO 8601 with an offset) as UTC."""
    return pd.to_datetime(series, errors='coerce', utc=True)


def _number(value: Any) -> Optional[float]:
    """A gap as a float, ``None`` when the feed left it blank."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


# ---------------------------------------------------------------------------
# Gap radar
# ---------------------------------------------------------------------------

def gap_history(session_key: int) -> pd.DataFrame:
    """Every published gap of one session, one row per car per update.

    ``GapToLeader`` is the time behind the leader (masked to the leader, who has
    no gap), ``Interval`` the time behind the car directly ahead. Rows where
    both are blank are dropped: they carry no information.
    """
    rows = _rows('intervals', session_key=session_key)
    if not rows:
        return pd.DataFrame(columns=GAP_COLUMNS)

    names = _driver_names(session_key)
    records: List[Dict[str, Any]] = []
    for row in rows:
        number = row.get('driver_number')
        if number is None:
            continue
        gap = _number(row.get('gap_to_leader'))
        interval = _number(row.get('interval'))
        if gap is None and interval is None:
            continue
        records.append({
            'DriverNumber': int(number),
            'Driver': names.get(int(number), str(number)),
            'Date': row.get('date'),
            'GapToLeader': gap,
            'Interval': interval,
        })

    if not records:
        return pd.DataFrame(columns=GAP_COLUMNS)

    frame = pd.DataFrame(records)
    frame['Date'] = _to_utc(frame['Date'])
    frame = frame.dropna(subset=['Date']).sort_values(['Date', 'DriverNumber'])
    return frame[GAP_COLUMNS].reset_index(drop=True)


def latest_gaps(session_key: int) -> pd.DataFrame:
    """The most recent gap of every car, ordered by the gap to the leader."""
    history = gap_history(session_key)
    if history.empty:
        return pd.DataFrame(columns=GAP_COLUMNS)

    latest = (history.dropna(subset=['GapToLeader'])
              .sort_values('Date')
              .groupby('Driver', as_index=False)
              .last())
    return latest.sort_values('GapToLeader')[GAP_COLUMNS].reset_index(drop=True)


def gap_series(session_key: int, max_drivers: int = 8) -> pd.DataFrame:
    """Pivot of the gap to the leader: one column per car, one row per update.

    Only the ``max_drivers`` closest cars are kept, because a chart with
    twenty-two lines is unreadable - and the backmarkers are the ones a reader
    is least interested in.
    """
    history = gap_history(session_key)
    if history.empty:
        return pd.DataFrame()

    closest = (history.dropna(subset=['GapToLeader'])
               .sort_values('Date')
               .groupby('Driver')['GapToLeader']
               .last()
               .sort_values()
               .head(max_drivers)
               .index)
    subset = history[history['Driver'].isin(closest)]
    pivot = subset.pivot_table(index='Date', columns='Driver',
                               values='GapToLeader', aggfunc='last')
    return pivot.sort_index().reset_index()


def gap_summary(session_key: int) -> Dict[str, Any]:
    """Facts about the gap feed, for the sentence above the chart."""
    history = gap_history(session_key)
    if history.empty:
        return {'rows': 0, 'drivers': 0, 'minutes': 0.0,
                'last_update': None, 'leader': None}

    span = history['Date'].max() - history['Date'].min()
    latest = latest_gaps(session_key)
    leader = latest['Driver'].iloc[0] if not latest.empty else None
    return {
        'rows': int(len(history)),
        'drivers': int(history['Driver'].nunique()),
        'minutes': round(span.total_seconds() / 60.0, 1) if pd.notna(span) else 0.0,
        'last_update': history['Date'].max(),
        'leader': leader,
    }


# ---------------------------------------------------------------------------
# Team radio and race control
# ---------------------------------------------------------------------------

def team_radio(session_key: int) -> pd.DataFrame:
    """Published radio messages, newest last."""
    rows = _rows('team_radio', session_key=session_key)
    if not rows:
        return pd.DataFrame(columns=RADIO_COLUMNS)

    names = _driver_names(session_key)
    records: List[Dict[str, Any]] = []
    for row in rows:
        number = row.get('driver_number')
        records.append({
            'Date': row.get('date'),
            'DriverNumber': int(number) if number is not None else None,
            'Driver': names.get(int(number), str(number)) if number is not None else '-',
            'RecordingUrl': row.get('recording_url') or '',
        })

    frame = pd.DataFrame(records)
    frame['Date'] = _to_utc(frame['Date'])
    frame = frame.dropna(subset=['Date']).sort_values('Date')
    return frame[RADIO_COLUMNS].reset_index(drop=True)


def race_control(session_key: int) -> pd.DataFrame:
    """Steward messages (flags, investigations, penalties), newest last."""
    rows = _rows('race_control', session_key=session_key)
    if not rows:
        return pd.DataFrame(columns=CONTROL_COLUMNS)

    names = _driver_names(session_key)
    records: List[Dict[str, Any]] = []
    for row in rows:
        number = row.get('driver_number')
        records.append({
            'Date': row.get('date'),
            'Lap': row.get('lap_number'),
            'Category': str(row.get('category') or ''),
            'Flag': str(row.get('flag') or ''),
            'Message': str(row.get('message') or ''),
            'Driver': names.get(int(number), str(number)) if number is not None else '',
        })

    frame = pd.DataFrame(records)
    frame['Date'] = _to_utc(frame['Date'])
    frame = frame.dropna(subset=['Date']).sort_values('Date')
    return frame[CONTROL_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# One status object for the page
# ---------------------------------------------------------------------------

def feed_status(session_key: Optional[int]) -> Dict[str, Any]:
    """What these feeds actually published for a session.

    ``available`` is False when the caller has no OpenF1 session key (a FastF1
    session) or when every feed came back empty, so the page can show one
    honest sentence instead of three empty boxes.
    """
    if not session_key:
        return {
            'available': False,
            'reason': ('This session has no OpenF1 key: the gap radar, the '
                       'radio and the race-control feed are only published '
                       'for OpenF1 sessions.'),
            'intervals': 0, 'radio': 0, 'control': 0,
        }

    gaps = gap_history(session_key)
    radio = team_radio(session_key)
    control = race_control(session_key)
    counts = {'intervals': int(len(gaps)), 'radio': int(len(radio)),
              'control': int(len(control))}
    return {
        'available': any(counts.values()),
        'reason': '' if any(counts.values()) else
                  ('OpenF1 published no live gaps, radio or race-control '
                   'messages for this session.'),
        **counts,
    }
