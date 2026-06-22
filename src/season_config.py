# -*- coding: utf-8 -*-
"""
season_config.py
~~~~~~~~~~~~~~~~
Dynamic season configuration via FastF1 API with fallback to hardcoded data.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import fastf1
import pandas as pd
from datetime import datetime

# Simple in-memory cache for event schedule (avoid repeated API calls)
_schedule_cache = {}

# Fallback: hardcoded 2025 data when FastF1 API is unavailable
FALLBACK_CALENDAR_2025 = [
    {'round': 1, 'name': 'Australian Grand Prix', 'location': 'Melbourne', 'country': 'Australia'},
    {'round': 2, 'name': 'Chinese Grand Prix', 'location': 'Shanghai', 'country': 'China'},
    {'round': 3, 'name': 'Japanese Grand Prix', 'location': 'Suzuka', 'country': 'Japan'},
    {'round': 4, 'name': 'Bahrain Grand Prix', 'location': 'Sakhir', 'country': 'Bahrain'},
    {'round': 5, 'name': 'Saudi Arabian Grand Prix', 'location': 'Jeddah', 'country': 'Saudi Arabia'},
    {'round': 6, 'name': 'Miami Grand Prix', 'location': 'Miami', 'country': 'USA'},
    {'round': 7, 'name': 'Emilia Romagna Grand Prix', 'location': 'Imola', 'country': 'Italy'},
    {'round': 8, 'name': 'Monaco Grand Prix', 'location': 'Monte Carlo', 'country': 'Monaco'},
    {'round': 9, 'name': 'Spanish Grand Prix', 'location': 'Barcelona', 'country': 'Spain'},
    {'round': 10, 'name': 'Canadian Grand Prix', 'location': 'Montreal', 'country': 'Canada'},
    {'round': 11, 'name': 'Austrian Grand Prix', 'location': 'Spielberg', 'country': 'Austria'},
    {'round': 12, 'name': 'British Grand Prix', 'location': 'Silverstone', 'country': 'Great Britain'},
    {'round': 13, 'name': 'Belgian Grand Prix', 'location': 'Spa', 'country': 'Belgium'},
    {'round': 14, 'name': 'Hungarian Grand Prix', 'location': 'Budapest', 'country': 'Hungary'},
    {'round': 15, 'name': 'Dutch Grand Prix', 'location': 'Zandvoort', 'country': 'Netherlands'},
    {'round': 16, 'name': 'Italian Grand Prix', 'location': 'Monza', 'country': 'Italy'},
    {'round': 17, 'name': 'Azerbaijan Grand Prix', 'location': 'Baku', 'country': 'Azerbaijan'},
    {'round': 18, 'name': 'Singapore Grand Prix', 'location': 'Marina Bay', 'country': 'Singapore'},
    {'round': 19, 'name': 'United States Grand Prix', 'location': 'Austin', 'country': 'USA'},
    {'round': 20, 'name': 'Mexico City Grand Prix', 'location': 'Mexico City', 'country': 'Mexico'},
    {'round': 21, 'name': 'Sao Paulo Grand Prix', 'location': 'Interlagos', 'country': 'Brazil'},
    {'round': 22, 'name': 'Las Vegas Grand Prix', 'location': 'Las Vegas', 'country': 'USA'},
    {'round': 23, 'name': 'Qatar Grand Prix', 'location': 'Lusail', 'country': 'Qatar'},
    {'round': 24, 'name': 'Abu Dhabi Grand Prix', 'location': 'Yas Marina', 'country': 'UAE'},
]

FALLBACK_RACE_NAMES_2025 = {r['name']: r['name'] for r in FALLBACK_CALENDAR_2025}


def _get_schedule_cached(year):
    """Get event schedule with in-memory cache to avoid repeated API calls."""
    if year in _schedule_cache:
        return _schedule_cache[year]
    try:
        schedule = fastf1.get_event_schedule(year)
        _schedule_cache[year] = schedule
        return schedule
    except Exception:
        return None


def get_season_calendar(year):
    """Get the full calendar for the given year using FastF1 API.

    Falls back to hardcoded 2025 data if the API is unavailable.
    """
    schedule = _get_schedule_cached(year)
    if schedule is None or schedule.empty:
        return _fallback_calendar(year)
    calendar = []
    for _, row in schedule.iterrows():
        event_format = row.get('EventFormat', '')
        if event_format != 'testing':
            calendar.append({
                'round': row['RoundNumber'],
                'name': row['EventName'],
                'location': row['Location'],
                'country': row['Country'],
                'date': row['EventDate']
            })
    return calendar


def get_completed_races(year):
    """Get list of completed race names for the given year.

    Falls back to hardcoded 2025 list if the API is unavailable.
    """
    schedule = _get_schedule_cached(year)
    if schedule is None or schedule.empty:
        return _fallback_completed(year)
    try:
        if schedule['EventDate'].dt.tz is None:
            schedule['EventDate'] = schedule['EventDate'].dt.tz_localize('UTC')
        now = pd.Timestamp.now(tz='UTC')
        completed = schedule[schedule['EventDate'] < now]
        completed = completed[completed['EventFormat'] != 'testing']
        return completed['EventName'].tolist()
    except Exception:
        return _fallback_completed(year)


def get_race_names(year):
    """Get dictionary mapping display names to FastF1 names.

    Falls back to hardcoded 2025 data if the API is unavailable.
    """
    try:
        calendar = get_season_calendar(year)
        if calendar:
            return {r['name']: r['name'] for r in calendar}
    except Exception:
        pass
    return _fallback_race_names(year)


# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------

def _fallback_calendar(year):
    """Return hardcoded calendar for known years."""
    if year == 2025:
        return FALLBACK_CALENDAR_2025
    return []


def _fallback_completed(year):
    """Return all races as completed for known years (assumes season over)."""
    if year == 2025:
        return [r['name'] for r in FALLBACK_CALENDAR_2025]
    return []


def _fallback_race_names(year):
    """Return hardcoded race names for known years."""
    if year == 2025:
        return FALLBACK_RACE_NAMES_2025
    return {}
