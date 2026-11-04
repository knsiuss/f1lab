import streamlit as st
import pandas as pd
import fastf1
import logging
import time
from typing import Dict

# Local imports
import artifacts
from config import STREAMLIT_CONFIG, FASTF1_CONFIG, TEAM_COLORS
from loader import clean_data
from openf1_results import (
    build_season_results, coverage_note, grid_for_event, team_colours,
)

logger = logging.getLogger(__name__)

# Prefer a published archive over the network. ``activate`` wraps
# ``openf1_client.cached_json`` -- the one function every data module reaches
# OpenF1 through -- so a query that belongs to a session the nightly refresh
# marked finished is served from disk at any age, while live sessions and cold
# misses behave exactly as before. With no manifest on this machine (a fresh
# checkout) it is a no-op. See docs/DATA_CACHE.md.
artifacts.activate()

# Colour used when a team has no livery colour from the API and no entry in
# the fallback palette.
UNKNOWN_TEAM_COLOUR = '#6B7280'


@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl)
def season_team_colours(year) -> Dict[str, str]:
    """Team -> livery colour for a season.

    The API's colour wins: it is the official livery of the season the user
    is looking at, so a team that is new (or has changed its livery) is
    correct without a code change. ``config.TEAM_COLORS`` is only the fallback
    for teams the API has not reported.
    """
    colours: Dict[str, str] = dict(TEAM_COLORS)
    try:
        colours.update(team_colours(int(year)))
    except Exception as e:
        logger.warning(f"Could not load team colours for {year}: {e}")
    return colours


def team_colour(team, year=None) -> str:
    """The livery colour of a team for the given season."""
    if not team:
        return UNKNOWN_TEAM_COLOUR
    name = str(team)
    if year is not None:
        colours = season_team_colours(year)
        if name in colours:
            return colours[name]
    return TEAM_COLORS.get(name, UNKNOWN_TEAM_COLOUR)


@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl)
def load_race_data(year):
    """Cached season results for a year (OpenF1).

    The frame keeps the historical CSV schema -- ``Track``, ``Position``,
    ``Driver``, ``Team``, ``Starting Grid``, ``Laps``, ``Time/Retired``,
    ``Points`` plus ``SessionType`` -- so every consumer keeps working while
    the local CSV exports are no longer needed.

    Returns ``None`` when the season has no data (an unsupported year or the
    service being unreachable) so pages can say so instead of rendering an
    empty table that looks like "no races yet".
    """
    try:
        # The spinner only shows on a cache miss, which is exactly when the
        # user waits: the first load of a season pulls dozens of requests from
        # OpenF1 under its rate limit.
        with st.spinner(f'Loading the {year} season from OpenF1 '
                        f'(the first load takes a minute)...'):
            frame = build_season_results(int(year))
        if frame is None or frame.empty:
            return None
        return clean_data(frame)
    except Exception as e:
        logger.error(f"Error loading season data for {year}: {e}")
        return None


@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl)
def get_total_points_combined(year):
    """Total points (race + sprint) per driver, plus season totals."""
    total_points_combined = {}
    total_laps_all = 0
    total_all_points = 0
    try:
        df = load_race_data(year)
        if df is None or df.empty:
            return total_points_combined, total_laps_all, total_all_points

        if 'SessionType' in df.columns:
            race_only = df[df['SessionType'] == 'Race']
        else:
            race_only = df
        if 'Laps' in race_only.columns:
            total_laps_all = int(pd.to_numeric(race_only['Laps'],
                                               errors='coerce').fillna(0).sum())
        if 'Driver' in df.columns and 'Points' in df.columns:
            total_points_combined = df.groupby('Driver')['Points'].sum().to_dict()
        total_all_points = float(sum(total_points_combined.values()))
    except Exception as e:
        logger.warning(f"Could not compute total points for {year}: {e}")
    return total_points_combined, total_laps_all, total_all_points


def season_coverage_note(year) -> str:
    """Why a season has no results (out of archive vs not started yet)."""
    return coverage_note(year)


def season_progress(year) -> tuple:
    """How far the season has gone: ``(races with results, races in the calendar)``.

    One source for the sidebar and the dashboard, so the two can never show
    different numbers. The count of completed races comes from the data that
    is actually loaded (a race whose result has not been published yet is not
    counted as done), the total from the calendar.
    """
    frame = load_race_data(year)
    completed = 0
    if frame is not None and not frame.empty:
        races = frame[frame['SessionType'] == 'Race'] if 'SessionType' in frame.columns else frame
        if 'Round' in races.columns:
            completed = int(pd.to_numeric(races['Round'], errors='coerce').dropna().nunique())
        elif 'Track' in races.columns:
            completed = int(races['Track'].nunique())
    try:
        from season_config import get_season_calendar
        total = len(get_season_calendar(int(year)))
    except Exception:
        total = 0
    return completed, total


def load_race_grid(year, race=None, race_df=None, legacy_race_df=None):
    """Qualifying grid of one Grand Prix as ``{driver_name: position}``.

    The grid comes from the meeting's qualifying session via OpenF1. The
    historical ``(data_dir, year, race, race_df)`` call shape is still
    accepted so existing call sites keep working -- the data directory is
    ignored, because the app no longer reads local CSV exports.
    """
    if isinstance(year, str):
        # Legacy positional call: load_race_grid(data_dir, year, race, df)
        year, race = race, race_df
    try:
        return grid_for_event(int(year), str(race or ''))
    except Exception as e:
        logger.warning(f"Could not load grid for {year} {race}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Session loading: FastF1 first, OpenF1 as the fallback
# ---------------------------------------------------------------------------

#: FastF1's timing archive is not always reachable (the static timing files
#: now need a subscription).  Every failed attempt costs about ten seconds,
#: so after a failure the app goes straight to the OpenF1 feed for a while
#: instead of paying that cost again on every page load.
FASTF1_FAILURE_COOLDOWN_S = 900
_FASTF1_STATE = {'failures': 0, 'blocked_until': 0.0}

#: The official timing archive is either reachable or it is not, and when it
#: is not, a full FastF1 load still burns about ten seconds before it gives
#: up.  One small probe costs under two seconds and answers the question for
#: every session of the process.
FASTF1_ARCHIVE_PROBE_TTL_S = 900
_FASTF1_ARCHIVE_BASES = ('https://livetiming.formula1.com',
                         'https://livetiming-mirror.fastf1.dev')
_FASTF1_ARCHIVE = {'available': None, 'checked_at': 0.0}


def _fastf1_blocked() -> bool:
    """True while the app skips FastF1 and uses the OpenF1 feed."""
    return time.monotonic() < _FASTF1_STATE['blocked_until']


def _fastf1_archive_reachable(year: int, race: str,
                              session_type: str = 'Race') -> bool:
    """Check cheaply whether FastF1's timing archive answers.

    The session files behind ``livetiming.formula1.com`` now need a
    subscription, so the request comes back with ``401`` and FastF1 keeps
    retrying for about ten seconds before it gives up.  Building the session
    object (without loading it) is nearly free and gives the exact archive
    path, so one small ``Index.json`` request is enough to know whether the
    attempt is worth it.

    The answer is cached for the process, because the archive either answers
    for everybody or for nobody.
    """
    now = time.monotonic()
    state = _FASTF1_ARCHIVE
    if (state['available'] is not None
            and now - state['checked_at'] < FASTF1_ARCHIVE_PROBE_TTL_S):
        return bool(state['available'])

    state['checked_at'] = now
    path = None
    try:
        # Building the session object does not download any timing data.
        path = fastf1.get_session(year, race, session_type).api_path
    except Exception as e:
        logger.warning('Could not build the FastF1 session path: %s', e)
        state['available'] = True  # cannot tell, let FastF1 try
        return True

    try:
        import requests
    except Exception:  # pragma: no cover - requests ships with FastF1
        state['available'] = True
        return True

    for base in _FASTF1_ARCHIVE_BASES:
        try:
            response = requests.get(f'{base}{path}Index.json', timeout=6)
        except Exception:
            continue
        if response.status_code == 200:
            state['available'] = True
            return True

    state['available'] = False
    logger.info('The FastF1 timing archive did not answer for %s %s; '
                'the OpenF1 feed will be used', race, session_type)
    return False


def _note_fastf1_failure() -> None:
    """Record a FastF1 failure and pause further FastF1 attempts."""
    _FASTF1_STATE['failures'] += 1
    _FASTF1_STATE['blocked_until'] = (time.monotonic()
                                      + FASTF1_FAILURE_COOLDOWN_S)
    logger.warning('FastF1 is unavailable; using the OpenF1 feed for %s minutes',
                   FASTF1_FAILURE_COOLDOWN_S // 60)


@st.cache_data(ttl=STREAMLIT_CONFIG.session_cache_ttl,
                show_spinner=False)
def _openf1_session(year: int, race: str, session_type: str):
    """Rebuild a session from the OpenF1 feed (cached)."""
    from openf1_session import build_session

    try:
        return build_session(int(year), str(race), str(session_type))
    except Exception as e:  # pragma: no cover - network/offline
        logger.error('OpenF1 fallback failed for %s %s %s: %s',
                     year, race, session_type, e)
        return None


@st.cache_resource(ttl=STREAMLIT_CONFIG.session_cache_ttl)
def load_fastf1_session(year: int, race: str, session_type: str,
                         load_telemetry: bool = False,
                         weather: bool = True, messages: bool = True):
    """Load a session with caching, falling back to the OpenF1 feed.

    FastF1's own timing archive answers ``401`` without a subscription, so a
    FastF1 session can come back without any laps.  When that happens the
    same session is rebuilt from the OpenF1 feed
    (:mod:`openf1_session`), which publishes lap times, sectors, stints and
    pit stops for free.  The returned object then carries ``source`` =
    ``'OpenF1'`` so the pages can say where the numbers came from.

    Args:
        year, race, session_type: which session to load.
        load_telemetry: also load the car telemetry (slow, and only the setup
            fingerprint needs it).  The OpenF1 fallback has no telemetry, so
            this is a FastF1-only feature.
        weather: load the weather feed. Every extra feed is one more HTTP
            request, so a page that only needs the lap table (the weekend
            report) turns it off and saves the round trip.
        messages: load the race control messages (the analysis page needs
            them, the report does not).
    """
    if not _fastf1_blocked():
        if not _fastf1_archive_reachable(year, race, session_type):
            logger.info('Skipping FastF1 for %s %s %s: the timing archive '
                        'did not answer', year, race, session_type)
        else:
            try:
                session = fastf1.get_session(year, race, session_type)
                if load_telemetry:
                    session.load()
                else:
                    session.load(telemetry=False, laps=True, weather=weather,
                                 messages=messages)
                # Touch the lap table: a session whose load failed (a missing
                # telemetry feed, an offline weekend) raises here instead of
                # much later, in whichever page happened to read it first.
                if session.laps is None:
                    logger.error(
                        f"FastF1 session {year} {race} {session_type} has no laps")
                    _note_fastf1_failure()
                else:
                    return session
            except Exception as e:
                logger.error(f"Error loading FastF1 session: {e}")
                _note_fastf1_failure()
    return _openf1_session(year, race, session_type)




def format_f1_time(td: pd.Timedelta) -> str:
    """Formats a pandas Timedelta object into an F1-style lap time string."""
    if pd.isna(td):
        return "N/A"

    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:06.3f}"
    else:
        return f"{minutes:d}:{seconds:06.3f}"


def setup_fastf1_cache():
    cache_dir = FASTF1_CONFIG.cache_dir
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    return cache_dir


def show_plotly_chart(fig, width='stretch', apply_theme=True, **kwargs):
    """Display a Plotly chart with the app theme and a hidden toolbar.

    Only the *colours* are applied here. The geometry (margins, legend
    position, hover mode) belongs to the chart itself -- ``styles.fig_layout``
    builds it -- because a helper that re-applies its own margins silently
    undoes the layout every page asked for.
    """
    if apply_theme:
        fig.update_layout(
            font=dict(family="Inter, system-ui, sans-serif", color="#E8EAF0"),
            paper_bgcolor='#12151C',
            plot_bgcolor='#171B24',
            hoverlabel=dict(
                bgcolor="rgba(18,21,28,0.95)",
                font_size=12,
                font_family="Inter, system-ui, sans-serif",
                bordercolor="rgba(255,255,255,0.14)"
            ),
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)",
                         zerolinecolor="rgba(255,255,255,0.13)")
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)",
                         zerolinecolor="rgba(255,255,255,0.13)")

    config = {
        'displayModeBar': False,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'zoomIn2d', 'zoomOut2d',
                                   'autoScale2d', 'resetScale2d', 'toImage'],
        'staticPlot': False
    }
    st.plotly_chart(fig, width=width, config=config, **kwargs)


def empty_state(title: str, hint: str = '') -> None:
    """Consistent, calm empty state panel (season not started / no data yet).

    Same markup as :func:`ui.empty`, so pages that still use this helper look
    identical to the ones that moved to the shared component.
    """
    from ui import empty
    empty(title, hint)

