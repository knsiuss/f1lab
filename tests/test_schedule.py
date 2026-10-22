# -*- coding: utf-8 -*-
"""Behaviour tests for schedule.py (countdowns, session times, live weekends).

No network and no pandas magic: the calendar is injected through a fake
``race_weekends`` so the arithmetic is tested on fixed datetimes.
"""
from datetime import datetime, timedelta, timezone

import pytest

import schedule as race_schedule

UTC = timezone.utc


def _weekend(round_number, start, sessions=('Race',)):
    """A weekend whose sessions start ``start`` and follow an hour apart."""
    built = []
    for index, label in enumerate(sessions):
        built.append({'type': label, 'start': start + timedelta(hours=index)})
    return {
        'round': round_number,
        'name': f'Round {round_number} Grand Prix',
        'location': 'Somewhere',
        'country': 'Someland',
        'date': built[-1]['start'],
        'sessions': built,
    }


@pytest.fixture
def fake_calendar(monkeypatch):
    start = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)
    weekends = [
        _weekend(16, start - timedelta(days=14), ('Qualifying', 'Race')),
        _weekend(17, start, ('Practice 1', 'Qualifying', 'Race')),
        _weekend(18, start + timedelta(days=21), ('Race',)),
    ]
    monkeypatch.setattr(race_schedule, 'race_weekends', lambda year: weekends)
    return weekends


def test_countdown_splits_days_hours_minutes():
    now = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)
    result = race_schedule.countdown(now + timedelta(days=3, hours=4, minutes=5),
                                     now)
    assert result['days'] == 3
    assert result['hours'] == 4
    assert result['minutes'] == 5
    assert result['is_future'] is True
    assert '3 days' in result['label']


def test_countdown_under_an_hour_reads_in_minutes():
    now = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)
    result = race_schedule.countdown(now + timedelta(minutes=42), now)
    assert result['label'] == '42 min'


def test_countdown_flags_a_session_that_already_started():
    now = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    result = race_schedule.countdown(now - timedelta(minutes=5), now)
    assert result['is_future'] is False
    assert result['label'] == 'under way'


def test_countdown_of_a_missing_date_is_none():
    assert race_schedule.countdown(None) is None


def test_session_times_converts_utc_to_wib():
    start = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)
    times = race_schedule.session_times(start)
    # WIB is UTC+7, so a 12:00 UTC start is 19:00 in Jakarta.
    assert '19:00' in times['wib']
    assert times['utc'].startswith('Sun 11 Oct')


def test_next_session_skips_the_past(fake_calendar):
    now = datetime(2026, 10, 11, 10, 0, tzinfo=UTC)
    session = race_schedule.next_session(2026, now)
    assert session['round'] == 17
    assert session['type'] == 'Practice 1'   # 12:00 is the next start
    assert session['start'] == datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


def test_next_session_is_none_when_the_season_is_over(fake_calendar):
    now = datetime(2027, 1, 1, tzinfo=UTC)
    assert race_schedule.next_session(2026, now) is None


def test_live_weekend_covers_the_race_day(fake_calendar):
    during = datetime(2026, 10, 11, 13, 0, tzinfo=UTC)
    live = race_schedule.live_weekend(2026, during)
    assert live is not None and live['round'] == 17


def test_live_weekend_is_none_between_weekends(fake_calendar):
    between = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
    assert race_schedule.live_weekend(2026, between) is None


def test_next_weekend_prefers_the_live_one(fake_calendar):
    during = datetime(2026, 10, 11, 13, 0, tzinfo=UTC)
    weekend = race_schedule.next_weekend(2026, during)
    assert weekend['round'] == 17


def test_upcoming_sessions_are_sorted(fake_calendar):
    now = datetime(2026, 10, 5, tzinfo=UTC)
    rows = race_schedule.upcoming_sessions(2026, now)
    starts = [row['start'] for row in rows]
    assert starts == sorted(starts)
    assert len(rows) == 4   # 3 sessions of round 17 + 1 of round 18


def test_race_weekends_without_a_calendar_is_empty(monkeypatch):
    monkeypatch.setattr(race_schedule.season_config,
                        'get_event_schedule_cached', lambda year: None)
    monkeypatch.setattr(race_schedule.season_config,
                        'get_season_calendar', lambda year: [])
    assert race_schedule.race_weekends(2026) == []
