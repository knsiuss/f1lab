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

# Simple in-memory cache for event schedule (avoid repeated API calls)
_schedule_cache = {}

# Fallback calendars used when the FastF1 API is unavailable.
# Keyed by year so a future season only needs to register its races here and
# every _fallback_* helper below picks them up automatically.
FALLBACK_CALENDARS = {
    2025: [
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
    ],
    # The 2026 calendar, as published by the OpenF1 meetings feed. It is only
    # used when FastF1 cannot be reached; the live calendar always wins.
    2026: [
        {'round': 1, 'name': 'Australian Grand Prix', 'location': 'Melbourne', 'country': 'Australia'},
        {'round': 2, 'name': 'Chinese Grand Prix', 'location': 'Shanghai', 'country': 'China'},
        {'round': 3, 'name': 'Japanese Grand Prix', 'location': 'Suzuka', 'country': 'Japan'},
        {'round': 4, 'name': 'Miami Grand Prix', 'location': 'Miami Gardens', 'country': 'United States'},
        {'round': 5, 'name': 'Canadian Grand Prix', 'location': 'Montreal', 'country': 'Canada'},
        {'round': 6, 'name': 'Monaco Grand Prix', 'location': 'Monte Carlo', 'country': 'Monaco'},
        {'round': 7, 'name': 'Barcelona Grand Prix', 'location': 'Barcelona', 'country': 'Spain'},
        {'round': 8, 'name': 'Austrian Grand Prix', 'location': 'Spielberg', 'country': 'Austria'},
        {'round': 9, 'name': 'British Grand Prix', 'location': 'Silverstone', 'country': 'United Kingdom'},
        {'round': 10, 'name': 'Belgian Grand Prix', 'location': 'Spa-Francorchamps', 'country': 'Belgium'},
        {'round': 11, 'name': 'Hungarian Grand Prix', 'location': 'Budapest', 'country': 'Hungary'},
        {'round': 12, 'name': 'Dutch Grand Prix', 'location': 'Zandvoort', 'country': 'Netherlands'},
        {'round': 13, 'name': 'Italian Grand Prix', 'location': 'Monza', 'country': 'Italy'},
        {'round': 14, 'name': 'Spanish Grand Prix', 'location': 'Madrid', 'country': 'Spain'},
        {'round': 15, 'name': 'Azerbaijan Grand Prix', 'location': 'Baku', 'country': 'Azerbaijan'},
        {'round': 16, 'name': 'Bahrain Grand Prix', 'location': 'Kuala Lumpur', 'country': 'Bahrain'},
        {'round': 17, 'name': 'Singapore Grand Prix', 'location': 'Marina Bay', 'country': 'Singapore'},
        {'round': 18, 'name': 'United States Grand Prix', 'location': 'Austin', 'country': 'United States'},
        {'round': 19, 'name': 'Mexico City Grand Prix', 'location': 'Mexico City', 'country': 'Mexico'},
        {'round': 20, 'name': 'Sao Paulo Grand Prix', 'location': 'Sao Paulo', 'country': 'Brazil'},
        {'round': 21, 'name': 'Las Vegas Grand Prix', 'location': 'Las Vegas', 'country': 'United States'},
        {'round': 22, 'name': 'Qatar Grand Prix', 'location': 'Lusail', 'country': 'Qatar'},
        {'round': 23, 'name': 'Abu Dhabi Grand Prix', 'location': 'Yas Marina', 'country': 'United Arab Emirates'},
    ],
}

# Per-year display-name maps derived from the calendars above.
FALLBACK_RACE_NAMES = {
    year: {r['name']: r['name'] for r in races}
    for year, races in FALLBACK_CALENDARS.items()
}


def get_event_schedule_cached(year):
    """Get the event schedule with in-memory cache to avoid repeated API calls.

    Public accessor: pages use this (e.g. future-race gating) instead of
    reaching into a module private.
    """
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

    Falls back to the registered per-year fallback if the API is unavailable.
    """
    schedule = get_event_schedule_cached(year)
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

    A race counts as completed once its session start has passed, so a race
    day is not reported as finished while the cars are still on track.

    Falls back to the registered per-year fallback if the API is unavailable.
    """
    schedule = get_event_schedule_cached(year)
    if schedule is None or schedule.empty:
        return _fallback_completed(year)
    try:
        # Work on a copy: the cached schedule is shared by every caller and
        # must not be mutated in place (a tz_localize on the cached frame
        # would leak into every later read).
        events = schedule.copy()
        if events['EventDate'].dt.tz is None:
            events['EventDate'] = events['EventDate'].dt.tz_localize('UTC')
        now = pd.Timestamp.now(tz='UTC')
        completed = events[events['EventDate'] < now]
        completed = completed[completed['EventFormat'] != 'testing']
        return completed['EventName'].tolist()
    except Exception:
        return _fallback_completed(year)


def get_race_names(year):
    """Get dictionary mapping display names to FastF1 names.

    Falls back to the registered per-year fallback if the API is unavailable.
    """
    try:
        calendar = get_season_calendar(year)
        if calendar:
            return {r['name']: r['name'] for r in calendar}
    except Exception:
        pass
    return _fallback_race_names(year)


def latest_race_index(year, races, tracks) -> int:
    """Index of the most recent race that actually has data.

    The race dropdowns list the whole calendar, so the default would be the
    **first** race of the season. For the predictor that means a forecast with
    no history behind it ("insufficient", zero rounds), and for the analysis
    it means the opening race instead of the latest one. Defaulting to the
    most recent race that has rows is what a reader expects.

    Args:
        year: the season (used to resolve display names to track values).
        races: the dropdown entries (FastF1 event names).
        tracks: the track values that are present in the loaded data.

    Returns:
        An index into ``races``; 0 when nothing matches.
    """
    if not races:
        return 0
    wanted = {str(t) for t in (tracks or [])}
    lowered = {t.lower() for t in wanted}
    for index in range(len(races) - 1, -1, -1):
        race = str(races[index])
        if race in wanted or race.lower() in lowered:
            return index
        # Only a name-based resolution is trusted here: the positional
        # fallback of resolve_csv_track can alias a race onto a track that
        # belongs to a different round when the frame is incomplete.
        resolved = resolve_csv_track(year, race, tracks)
        if resolved and str(resolved).lower() == race.lower():
            return index
    return 0


def resolve_csv_track(year, race, csv_tracks):
    """Map a calendar display name to the Track value used in the season data.

    The race dropdowns show FastF1 event names (e.g. 'Australian Grand Prix').
    The season frame from OpenF1 uses the same event name in its ``Track``
    column, so the exact match below is the normal path; the positional and
    case-insensitive fallbacks keep older frames (short track names such as
    'Australia') working too.

    Strategy: exact match first, then positional mapping (the season frame
    stores tracks in calendar order, so round N maps to the Nth track), then a
    case-insensitive match. Returns None when nothing matches (e.g. a future
    race with no results yet) so callers degrade gracefully.
    """
    if not race:
        return None
    tracks = [str(t) for t in (csv_tracks or [])]
    if race in tracks:
        return race
    try:
        names = [r['name'] for r in get_season_calendar(year)]
    except Exception:
        names = []
    if race in names:
        idx = names.index(race)
        if 0 <= idx < len(tracks):
            return tracks[idx]
    lowered = {t.lower(): t for t in tracks}
    return lowered.get(str(race).lower())


# The frame no longer comes from a CSV export, but the helper keeps its name
# so every existing call site keeps working.
resolve_track_name = resolve_csv_track


# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------

def _fallback_calendar(year):
    """Return the registered fallback calendar for ``year``."""
    return FALLBACK_CALENDARS.get(year, [])


def _fallback_completed(year):
    """Completed races from the offline fallback calendar.

    The fallback calendar carries no dates, so only a finished season can be
    resolved honestly: a season that is still running would otherwise be
    reported as "every race done". For the current season the answer is
    unknown, and :func:`calendar_is_live` lets the UI say so instead of
    guessing.
    """
    import datetime as _dt
    races = FALLBACK_CALENDARS.get(year, [])
    if int(year) < _dt.datetime.now().year:
        return [r['name'] for r in races]
    return []


def calendar_is_live(year) -> bool:
    """True when the real FastF1 calendar is available for the year.

    When it is not, the number of completed races is unknown rather than
    zero, and the UI should say so.
    """
    schedule = get_event_schedule_cached(int(year))
    return schedule is not None and not schedule.empty


def _fallback_race_names(year):
    """Return the registered fallback race names for ``year``."""
    return FALLBACK_RACE_NAMES.get(year, {})
