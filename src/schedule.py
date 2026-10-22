# -*- coding: utf-8 -*-
"""When the next session starts, and how long until then.

The calendar comes from the FastF1 event schedule (already cached by
``season_config``) and falls back to the registered per-year calendar when the
schedule cannot be fetched. Nothing here is invented: when the calendar is
unknown the helpers return ``None`` / empty lists so a page can stay honest
instead of guessing a date.

Times are handled as timezone-aware UTC ``datetime`` objects and only turned
into strings at the edge (WIB for the default reading, the browser's own zone
when the page passes one).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

import season_config

WIB = 'Asia/Jakarta'

# A weekend counts as "live" from a few hours before its first session until
# three days after its last one - long enough to cover the press day and the
# post-race analysis, short enough not to overlap the next weekend.
LIVE_WINDOW_BEFORE = timedelta(hours=6)
LIVE_WINDOW_AFTER = timedelta(hours=72)

# FastF1 session names are lowercase ('practice 1', 'sprint', 'qualifying').
SESSION_LABELS = {
    'practice 1': 'Practice 1',
    'practice 2': 'Practice 2',
    'practice 3': 'Practice 3',
    'sprint qualifying': 'Sprint qualifying',
    'sprint shootout': 'Sprint qualifying',
    'sprint': 'Sprint',
    'qualifying': 'Qualifying',
    'race': 'Race',
}


def _as_utc(value) -> Optional[datetime]:
    """Return a timezone-aware UTC datetime, or None for NaT/None/''."""
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            value = pd.Timestamp(value)
        except (TypeError, ValueError):
            return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, 'to_pydatetime'):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _session_label(name) -> str:
    text = str(name or '').strip()
    if not text:
        return ''
    return SESSION_LABELS.get(text.lower(), text.title())


def _sessions_from_row(row) -> list:
    """Pull the (label, start) pairs out of one schedule row."""
    sessions = []
    for index in range(1, 6):
        name = row.get(f'Session{index}')
        start = (_as_utc(row.get(f'Session{index}DateUtc'))
                 or _as_utc(row.get(f'Session{index}Date')))
        label = _session_label(name)
        if label and start is not None:
            sessions.append({'type': label, 'start': start})
    sessions.sort(key=lambda item: item['start'])
    return sessions


def race_weekends(year) -> list:
    """Every weekend of the season with its sessions, oldest first.

    Falls back to the registered calendar (race date only) when the schedule
    cannot be fetched. Returns an empty list when neither is available.
    """
    schedule = None
    try:
        schedule = season_config.get_event_schedule_cached(year)
    except Exception:
        schedule = None

    weekends = []
    if schedule is not None and not getattr(schedule, 'empty', True):
        try:
            for _, row in schedule.iterrows():
                if str(row.get('EventFormat', '')).lower() == 'testing':
                    continue
                sessions = _sessions_from_row(row)
                weekends.append({
                    'round': int(row.get('RoundNumber') or 0),
                    'name': str(row.get('EventName', '') or ''),
                    'location': str(row.get('Location', '') or ''),
                    'country': str(row.get('Country', '') or ''),
                    'date': _as_utc(row.get('EventDate')),
                    'sessions': sessions,
                })
        except Exception:
            weekends = []

    if not weekends:
        try:
            for event in season_config.get_season_calendar(year) or []:
                start = _as_utc(event.get('date'))
                weekends.append({
                    'round': int(event.get('round') or 0),
                    'name': str(event.get('name', '') or ''),
                    'location': str(event.get('location', '') or ''),
                    'country': str(event.get('country', '') or ''),
                    'date': start,
                    # The fallback calendar only knows the race day.
                    'sessions': ([{'type': 'Race', 'start': start}]
                                 if start is not None else []),
                })
        except Exception:
            weekends = []

    weekends = [weekend for weekend in weekends if weekend['sessions']]
    weekends.sort(key=lambda weekend: weekend['sessions'][0]['start'])
    return weekends


def upcoming_sessions(year, now_utc=None) -> list:
    """Every future session of the season as {track, type, start}, soonest first."""
    now = _as_utc(now_utc) or datetime.now(timezone.utc)
    rows = []
    for weekend in race_weekends(year):
        for session in weekend['sessions']:
            if session['start'] > now:
                rows.append({
                    'round': weekend['round'],
                    'track': weekend['name'],
                    'location': weekend['location'],
                    'type': session['type'],
                    'start': session['start'],
                    'weekend_start': weekend['sessions'][0]['start'],
                    'weekend_end': weekend['sessions'][-1]['start'],
                })
    rows.sort(key=lambda row: row['start'])
    return rows


def next_session(year, now_utc=None) -> Optional[dict]:
    """The next session that has not started yet, or None."""
    rows = upcoming_sessions(year, now_utc)
    return rows[0] if rows else None


def countdown(start_utc, now_utc=None) -> Optional[dict]:
    """Break the wait down into days/hours/minutes with a readable label."""
    start = _as_utc(start_utc)
    if start is None:
        return None
    now = _as_utc(now_utc) or datetime.now(timezone.utc)
    seconds = int((start - now).total_seconds())
    future = seconds > 0
    seconds = abs(seconds)
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60

    if not future:
        label = 'under way'
    elif days:
        label = f'{days} day{"s" if days != 1 else ""} {hours} h'
    elif hours:
        label = f'{hours} h {minutes:02d} min'
    else:
        label = f'{minutes} min'
    return {
        'days': days, 'hours': hours, 'minutes': minutes,
        'total_seconds': seconds, 'is_future': future, 'label': label,
    }


def _format_local(start, tz_name) -> str:
    try:
        from zoneinfo import ZoneInfo
        local = start.astimezone(ZoneInfo(tz_name))
    except Exception:
        return ''
    return local.strftime('%a %d %b, %H:%M')


def session_times(start_utc, tz_name: str = WIB) -> dict:
    """Readable times for a session in WIB and in a second zone.

    The second zone defaults to WIB as well (the dashboard's default reader is
    in Indonesia); a page can pass the browser's zone instead.
    """
    start = _as_utc(start_utc)
    if start is None:
        return {'utc': '', 'wib': '', 'local': '', 'local_zone': tz_name}
    return {
        'utc': start.strftime('%a %d %b, %H:%M UTC'),
        'wib': _format_local(start, WIB),
        'local': _format_local(start, tz_name) if tz_name else '',
        'local_zone': tz_name or '',
    }


def live_weekend(year, now_utc=None) -> Optional[dict]:
    """The weekend happening now (or just finished), if any."""
    now = _as_utc(now_utc) or datetime.now(timezone.utc)
    for weekend in race_weekends(year):
        start = weekend['sessions'][0]['start'] - LIVE_WINDOW_BEFORE
        end = weekend['sessions'][-1]['start'] + LIVE_WINDOW_AFTER
        if start <= now <= end:
            return weekend
    return None


def next_weekend(year, now_utc=None) -> Optional[dict]:
    """The weekend of the next session (or the live one)."""
    live = live_weekend(year, now_utc)
    if live is not None:
        return live
    session = next_session(year, now_utc)
    if session is None:
        return None
    for weekend in race_weekends(year):
        if weekend['round'] == session['round']:
            return weekend
    return None
