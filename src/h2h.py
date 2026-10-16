# -*- coding: utf-8 -*-
"""
h2h.py
FastF1-backed season results and driver head-to-head analysis.

The season CSVs in data/ are static snapshots that go stale every race
weekend, so the race-results frame is built straight from the FastF1 timing
API (which keeps its own disk cache). Everything downstream of the load is a
pure function over that frame: this module owns the numbers, the page owns
how they look.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import fastf1
import numpy as np
import pandas as pd
import streamlit as st

# Ensure src/ is importable no matter how this module is reached
# (as `h2h` from a page, or as `src.h2h` from the test suite).
sys.path.insert(0, str(Path(__file__).parent))

from config import FASTF1_CONFIG, STREAMLIT_CONFIG
from season_config import get_completed_races

logger = logging.getLogger(__name__)

# FastF1 event Location -> short display code for axes and table rows.
LOCATION_CODES = {
    'Melbourne': 'AUS', 'Shanghai': 'CHN', 'Suzuka': 'JPN', 'Sakhir': 'BHR',
    'Jeddah': 'SAU', 'Miami': 'MIA', 'Imola': 'EMI', 'Monte Carlo': 'MON',
    'Barcelona': 'ESP', 'Montreal': 'CAN', 'Spielberg': 'AUT',
    'Silverstone': 'GBR', 'Spa-Francorchamps': 'BEL', 'Budapest': 'HUN',
    'Zandvoort': 'NED', 'Monza': 'ITA', 'Baku': 'AZE',
    'Marina Bay': 'SGP', 'Austin': 'USA', 'Mexico City': 'MEX',
    'Interlagos': 'BRA', 'Las Vegas': 'LAS', 'Lusail': 'QAT',
    'Yas Island': 'ABU',
}

# FastF1 reports 3-letter IOC-style country codes; emoji flags need ISO alpha-2.
_IOC_TO_ISO = {
    'ALG': 'DZ', 'ARG': 'AR', 'AUS': 'AU', 'AUT': 'AT', 'AZE': 'AZ',
    'BEL': 'BE', 'BRA': 'BR', 'CAN': 'CA', 'CHI': 'CL', 'CHN': 'CN',
    'COL': 'CO', 'CRO': 'HR', 'CZE': 'CZ', 'DEN': 'DK', 'ESP': 'ES',
    'FIN': 'FI', 'FRA': 'FR', 'GER': 'DE', 'GBR': 'GB', 'GRE': 'GR',
    'HUN': 'HU', 'IDN': 'ID', 'IND': 'IN', 'IRE': 'IE', 'IRL': 'IE',
    'ITA': 'IT', 'JPN': 'JP', 'LTU': 'LT', 'MAS': 'MY', 'MEX': 'MX',
    'MON': 'MC', 'NED': 'NL', 'NZL': 'NZ', 'POL': 'PL', 'POR': 'PT',
    'QAT': 'QA', 'RSA': 'ZA', 'RUS': 'RU', 'SAU': 'SA', 'SGP': 'SG',
    'SLO': 'SI', 'SUI': 'CH', 'SWE': 'SE', 'THA': 'TH', 'TUR': 'TR',
    'UAE': 'AE', 'USA': 'US', 'VEN': 'VE',
}

_FALLBACK_TEAM_COLOR = '#888888'

# Teammates share one team colour, so the second side's tint keeps the two
# apart in dual-bar layouts.
_TEAMMATE_TINT = 0.45


def _ensure_cache() -> None:
    """Enable the FastF1 disk cache at the configured location.

    Same location as ``shared.setup_fastf1_cache``; enabling twice is a no-op,
    so a standalone ``streamlit run`` of this page still gets a warm cache.
    """
    try:
        FASTF1_CONFIG.cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(FASTF1_CONFIG.cache_dir))
    except Exception as exc:  # cache is an optimisation, never a blocker
        logger.warning(f"Could not enable FastF1 cache: {exc}")


_ensure_cache()


def country_flag(country_code: Optional[str]) -> str:
    """Regional-indicator flag emoji for an IOC/ISO country code.

    Returns '' for unknown or malformed codes so callers render nothing
    instead of a broken glyph.
    """
    code = str(country_code or '').strip().upper()
    code = _IOC_TO_ISO.get(code, code)
    if len(code) != 2 or not code.isalpha():
        return ''
    return chr(0x1F1E6 + ord(code[0]) - ord('A')) + chr(0x1F1E6 + ord(code[1]) - ord('A'))


def team_color(raw: Optional[str]) -> str:
    """FastF1 TeamColor -> '#RRGGBB'; tolerates a missing '#' and junk values."""
    value = str(raw or '').strip().lstrip('#')
    if len(value) == 6 and all(c in '0123456789abcdefABCDEF' for c in value):
        return f'#{value.upper()}'
    return _FALLBACK_TEAM_COLOR


def _tint(color: str, amount: float) -> str:
    """Blend a '#RRGGBB' colour toward white by amount (0..1)."""
    value = color.lstrip('#')
    rgb = [int(value[i:i + 2], 16) for i in (0, 2, 4)]
    mixed = [round(c + (255 - c) * amount) for c in rgb]
    return '#' + ''.join(f'{c:02X}' for c in mixed)


def with_alpha(color: str, alpha: float) -> str:
    """'#RRGGBB' -> 'rgba(...)' so a team colour can fade inside a gradient."""
    value = color.lstrip('#')
    if len(value) != 6:
        return f'rgba(255, 255, 255, {alpha})'
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'


def driver_color(color: str, is_teammate_of_first: bool) -> str:
    """Teammates share a team colour, so lighten the second driver's side."""
    return _tint(color, _TEAMMATE_TINT) if is_teammate_of_first else color


def race_code(event_name: Optional[str], location: Optional[str]) -> str:
    """Short code for a race tick (e.g. 'Australian Grand Prix' -> 'AUS')."""
    return LOCATION_CODES.get(str(location or ''), str(event_name or '')[:3].upper())


def short_race_name(event_name: Optional[str]) -> str:
    """Event name without the ' Grand Prix' suffix ('Bahrain', 'Monaco')."""
    name = str(event_name or '').strip()
    suffix = ' Grand Prix'
    return name[:-len(suffix)].strip() if name.endswith(suffix) else name


def result_label(position, status: Optional[str]) -> str:
    """'P3' for a classified result, else a compact DNF/DNS/DSQ code."""
    if pd.notna(position):
        return f"P{int(position)}"

    text = str(status or '').lower()
    if 'disqual' in text or 'exclud' in text:
        return 'DSQ'
    if 'did not start' in text or 'withdrew' in text:
        return 'DNS'
    return 'DNF'


def empty_results_frame() -> pd.DataFrame:
    """Normalised empty frame with the full column set."""
    return pd.DataFrame(columns=[
        'Round', 'Track', 'EventName', 'Code', 'SessionType', 'Driver', 'Number',
        'Team', 'TeamColor', 'CountryCode', 'Position', 'Grid', 'Points',
        'Laps', 'TimeSec', 'Status',
    ])


def _normalise_results(results: pd.DataFrame, session_type: str) -> pd.DataFrame:
    """Map a FastF1 ``session.results`` frame onto the shared schema.

    Args:
        results: Raw FastF1 results frame.
        session_type: 'Race' or 'Qualifying', kept for downstream filtering.

    Returns:
        Normalised frame with one row per driver. ``TimeSec`` is the gap to
        the winner in seconds (0 for the winner, NaN when no time was
        recorded, e.g. a DNF).

    Raises:
        KeyError: If FastF1 drops an expected column (API drift guard).
    """
    required = {'FullName', 'TeamName', 'TeamColor', 'CountryCode', 'Position',
                'GridPosition', 'Points', 'DriverNumber', 'Status'}
    missing = required.difference(results.columns)
    if missing:
        raise KeyError(f"FastF1 results missing columns: {sorted(missing)}")

    position = pd.to_numeric(results['Position'], errors='coerce')

    # A NaT-only 'Time' column (nothing classified) infers as datetime64,
    # which to_timedelta rejects outright; NaN coerces to NaT instead.
    time_column = results.get('Time')
    if (time_column is None or
            (time_column.isna().all()
             and pd.api.types.is_datetime64_any_dtype(time_column))):
        time_column = pd.Series(np.nan, index=results.index)
    seconds = pd.to_timedelta(time_column, errors='coerce').dt.total_seconds()

    # FastF1 stores the winner's full race time and everyone else's gap to it,
    # so only non-winners carry an offset. Seconds stay NaN for unclassified
    # rows, where a gap is meaningless.
    gap = np.where(position == 1, 0.0, seconds)
    gap = pd.Series(gap, index=results.index).where(position.notna())

    return pd.DataFrame({
        'Driver': results['FullName'].astype(str),
        'Number': results['DriverNumber'].astype(str),
        'Team': results['TeamName'].astype(str),
        'TeamColor': results['TeamColor'].map(team_color),
        'CountryCode': results['CountryCode'].astype(str),
        'Position': position,
        'Grid': pd.to_numeric(results['GridPosition'], errors='coerce'),
        'Points': pd.to_numeric(results['Points'], errors='coerce').fillna(0.0),
        'Laps': pd.to_numeric(results.get('Laps'), errors='coerce'),
        'TimeSec': gap,
        'Status': results['Status'].astype(str),
        'SessionType': session_type,
    })


@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl, show_spinner=False)
def _cached_session_results(year: int, event_name: str, session_type: str) -> pd.DataFrame:
    """One session's normalised results; FastF1's disk cache absorbs reloads."""
    try:
        session = fastf1.get_session(year, event_name, session_type)
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        results = session.results
        if results is None or results.empty:
            return empty_results_frame()

        rows = _normalise_results(results, session_type)
        event = getattr(session, 'event', None)
        round_no = event.get('RoundNumber') if event is not None else None
        rows['Round'] = int(round_no) if pd.notna(round_no) else 0
        rows['EventName'] = event_name
        rows['Track'] = event_name
        rows['Code'] = race_code(event_name, event.get('Location') if event is not None else '')
        return rows
    except Exception as exc:
        logger.warning("Could not load %s %s %s: %s", year, event_name, session_type, exc)
        return empty_results_frame()


def _assemble(frames: list) -> pd.DataFrame:
    """Concatenate per-session frames and sort chronologically by round."""
    usable = [f for f in frames if f is not None and not f.empty]
    if not usable:
        return empty_results_frame()
    combined = pd.concat(usable, ignore_index=True)
    return combined.sort_values(['Round', 'Driver']).reset_index(drop=True)


def load_season_results(year: int, include_quali: bool = False,
                        progress: Optional[callable] = None):
    """Load the completed season's results from FastF1.

    Args:
        year: Championship year.
        include_quali: Also load qualifying sessions (doubles the API calls,
            so it stays opt-in -- grid position is the default qui comparison).
        progress: Optional callback(event_name, index, total) for UI feedback.

    Returns:
        Tuple (races, qualifying). Both are normalised frames; qualifying is
        empty when include_quali is False.
    """
    quali_types = ('Qualifying',) if include_quali else ()
    events = get_completed_races(year) or []
    events = [e for e in events if str(e).lower() not in ('pre-season testing', 'testing')]

    race_frames, quali_frames = [], []
    for index, event_name in enumerate(events):
        if progress:
            progress(event_name, index, len(events))
        race_frames.append(_cached_session_results(year, event_name, 'Race'))
        for session_type in quali_types:
            quali_frames.append(_cached_session_results(year, event_name, session_type))

    return _assemble(race_frames), _assemble(quali_frames)


# ---------------------------------------------------------------------------
# Season-level helpers
# ---------------------------------------------------------------------------

def season_points(race_df: pd.DataFrame) -> pd.Series:
    """Driver championship points (races only), descending."""
    if race_df is None or race_df.empty:
        return pd.Series(dtype=float)
    return race_df.groupby('Driver')['Points'].sum().sort_values(ascending=False)


def driver_summary(race_df: pd.DataFrame, driver: str,
                   quali_df: Optional[pd.DataFrame] = None) -> dict:
    """Totals for one driver over the races present in ``race_df``.

    Args:
        race_df: Normalised race results.
        driver: Driver name to aggregate.
        quali_df: Optional normalised qualifying results; poles stay 0 when
            qualifying was not loaded.

    Returns:
        Dict with points, wins, podiums, races, poles, best finish, average
        finish (classified races only) and the number of unclassified races.
    """
    empty = {'points': 0, 'wins': 0, 'podiums': 0, 'races': 0, 'poles': 0,
             'best': None, 'avg_finish': None, 'dnfs': 0}
    if race_df is None or race_df.empty:
        return empty

    rows = race_df[race_df['Driver'] == driver]
    if rows.empty:
        return empty

    classified = rows[rows['Position'].notna()]
    poles = 0
    if quali_df is not None and not quali_df.empty:
        q_rows = quali_df[quali_df['Driver'] == driver]
        poles = int((q_rows['Position'] == 1).sum())

    return {
        'points': int(rows['Points'].sum()),
        'wins': int((rows['Position'] == 1).sum()),
        'podiums': int(rows['Position'].between(1, 3).sum()),
        'races': int(rows['Round'].nunique()),
        'poles': poles,
        'best': int(classified['Position'].min()) if not classified.empty else None,
        'avg_finish': round(float(classified['Position'].mean()), 2) if not classified.empty else None,
        'dnfs': int(rows['Position'].isna().sum()),
    }


def team_rosters(race_df: pd.DataFrame,
                 points: Optional[pd.Series] = None) -> pd.DataFrame:
    """One row per team with its drivers ordered by season points.

    Mid-season driver swaps can attach a third name to a team; the roster
    keeps them all, and ``head_to_head`` only ever pairs drivers that share
    races, so the main pairing is unaffected.
    """
    if race_df is None or race_df.empty or 'Team' not in race_df.columns:
        return pd.DataFrame(columns=['Team', 'TeamColor', 'Drivers', 'Points'])

    if points is None:
        points = season_points(race_df)
    teams = []
    for team, group in race_df.groupby('Team'):
        present = set(group['Driver'])
        drivers = [d for d in points.index if d in present]
        drivers += sorted(present.difference(drivers))
        teams.append({
            'Team': team,
            'TeamColor': team_color(group['TeamColor'].iloc[0]),
            'Drivers': drivers,
            'Points': float(points.reindex(drivers).fillna(0.0).sum()),
        })
    return (pd.DataFrame(teams)
            .sort_values('Points', ascending=False)
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------

def _driver_side(race_df: pd.DataFrame, quali_df: Optional[pd.DataFrame],
                 driver: str, color: str) -> dict:
    """Identity + season totals for one side of the comparison."""
    rows = race_df[race_df['Driver'] == driver]
    first = rows.iloc[0] if not rows.empty else {}
    name = str(first.get('Driver', driver) or driver)
    parts = name.split()
    number = first.get('Number')

    side = {
        'name': name,
        'short': parts[-1] if len(parts) > 1 else name,
        'first': parts[0] if parts else name,
        'last': ' '.join(parts[1:]).upper() if len(parts) > 1 else name.upper(),
        'team': str(first.get('Team', '')),
        'color': color,
        'flag': country_flag(first.get('CountryCode')),
        'number': str(int(float(number))) if pd.notna(number) else '—',
    }
    side.update(driver_summary(race_df, driver, quali_df))
    return side


def head_to_head(race_df: pd.DataFrame, quali_df: Optional[pd.DataFrame],
                 driver1: str, driver2: str) -> Optional[dict]:
    """Full head-to-head between two drivers over their shared races.

    Args:
        race_df: Normalised race results (see ``load_season_results``).
        quali_df: Normalised qualifying results, or None/empty to fall back to
            grid position as the qualifying comparison.
        driver1: Driver shown on the left / as side 1.
        driver2: Driver shown on the right / as side 2.

    Returns:
        Dict with both drivers' card data, race and qualifying records,
        points totals and the per-race table, or None when the pair shares no
        races.
    """
    if race_df is None or race_df.empty or driver1 == driver2:
        return None

    d1 = race_df[race_df['Driver'] == driver1]
    d2 = race_df[race_df['Driver'] == driver2]
    shared_rounds = sorted(set(d1['Round']) & set(d2['Round']))
    if not shared_rounds:
        return None

    d1 = d1[d1['Round'].isin(shared_rounds)].set_index('Round')
    d2 = d2[d2['Round'].isin(shared_rounds)].set_index('Round')

    races = []
    race_w1 = race_w2 = ties = 0
    cum1 = cum2 = 0
    for round_no in shared_rounds:
        r1, r2 = d1.loc[round_no], d2.loc[round_no]
        p1 = r1['Position'] if pd.notna(r1['Position']) else np.nan
        p2 = r2['Position'] if pd.notna(r2['Position']) else np.nan
        cum1 += int(r1['Points'])
        cum2 += int(r2['Points'])

        # Absolute finish-time difference, not a leader-relative gap, so the
        # column reads the same regardless of who won the race.
        gap = None
        if pd.notna(r1['TimeSec']) and pd.notna(r2['TimeSec']):
            gap = round(abs(float(r1['TimeSec']) - float(r2['TimeSec'])), 3)

        winner = 0
        if pd.notna(p1) and pd.notna(p2):
            if p1 < p2:
                race_w1 += 1
                winner = 1
            elif p2 < p1:
                race_w2 += 1
                winner = 2
            else:
                ties += 1
        elif pd.isna(p1) and pd.isna(p2):
            # Both unclassified: neither driver produced a result to
            # compare, so the exchange stays level.
            ties += 1

        races.append({
            'Round': int(round_no),
            'Code': r1['Code'],
            'Track': r1['Track'],
            'Pos1': p1, 'Pos2': p2,
            'Grid1': r1['Grid'], 'Grid2': r2['Grid'],
            'Pts1': int(r1['Points']), 'Pts2': int(r2['Points']),
            'Status1': r1['Status'], 'Status2': r2['Status'],
            'Cum1': cum1, 'Cum2': cum2,
            'Gap': gap,
            'Winner': winner,
        })

    race_table = pd.DataFrame(races)

    quali_record = _quali_record(quali_df, driver1, driver2, shared_rounds)
    if quali_record is None:
        quali_record = _record_from_grids(d1, d2, shared_rounds)

    color1 = team_color(d1['TeamColor'].iloc[0])
    same_team = d1['Team'].iloc[0] == d2['Team'].iloc[0]
    side1 = _driver_side(race_df, quali_df, driver1, color1)
    side2 = _driver_side(race_df, quali_df, driver2,
                         driver_color(color1, same_team))

    return {
        'driver1': side1,
        'driver2': side2,
        'races_shared': len(shared_rounds),
        'race_record': {'wins1': race_w1, 'wins2': race_w2, 'ties': ties},
        'quali_record': quali_record,
        'points': {'1': cum1, '2': cum2},
        'summary_rows': _summary_rows(side1, side2, race_w1, race_w2,
                                      quali_record),
        'wins_by_race': _wins_by_race(race_table),
        'track_delta': _track_points_delta(race_table),
        'race_table': race_table,
    }


def _summary_rows(side1: dict, side2: dict, race_w1: int, race_w2: int,
                  quali_record: dict) -> list:
    """Rows for the dual-bar summary panel: season totals plus H2H records."""
    rows = [
        ('Points', side1['points'], side2['points']),
        ('Wins', side1['wins'], side2['wins']),
        ('Podiums', side1['podiums'], side2['podiums']),
        ('Race H2H', race_w1, race_w2),
        ('Quali H2H', quali_record['wins1'], quali_record['wins2']),
    ]
    return [{'label': label, 'value1': v1, 'value2': v2,
             'display1': str(v1), 'display2': str(v2)}
            for label, v1, v2 in rows]


def _wins_by_race(race_table: pd.DataFrame) -> pd.DataFrame:
    """Per-race win flags (0/1) for the paired-driver wins chart."""
    return pd.DataFrame({
        'Code': race_table['Code'],
        'Wins1': race_table['Winner'] == 1,
        'Wins2': race_table['Winner'] == 2,
    }).astype({'Wins1': int, 'Wins2': int})


def _track_points_delta(race_table: pd.DataFrame) -> pd.DataFrame:
    """Points difference (driver1 - driver2) per race, for the diverging chart."""
    return pd.DataFrame({
        'Code': race_table['Code'],
        'Delta': race_table['Pts1'] - race_table['Pts2'],
    })


def _record_from_grids(d1: pd.DataFrame, d2: pd.DataFrame,
                       shared_rounds: list) -> dict:
    """Qualifying record approximated from grid slots over the shared races.

    Grid position carries penalties, so this is a proxy for the qualifying
    battle rather than a Q result; callers label it as such.
    """
    wins1 = wins2 = ties = 0
    for round_no in shared_rounds:
        g1, g2 = d1.loc[round_no, 'Grid'], d2.loc[round_no, 'Grid']
        if pd.isna(g1) or pd.isna(g2):
            ties += 1
        elif g1 < g2:
            wins1 += 1
        elif g2 < g1:
            wins2 += 1
        else:
            ties += 1
    return {'wins1': int(wins1), 'wins2': int(wins2), 'ties': int(ties),
            'source': 'grid', 'compared': int(len(shared_rounds))}


def _quali_record(quali_df: Optional[pd.DataFrame], driver1: str, driver2: str,
                  shared_rounds: list) -> Optional[dict]:
    """Qualifying record from real Q sessions, or None when none were loaded."""
    if quali_df is None or quali_df.empty:
        return None

    q1 = quali_df[quali_df['Driver'] == driver1].set_index('Round')['Position']
    q2 = quali_df[quali_df['Driver'] == driver2].set_index('Round')['Position']
    shared = [r for r in shared_rounds if r in q1.index and r in q2.index]
    if not shared:
        return None

    wins1 = wins2 = ties = 0
    for round_no in shared:
        p1, p2 = q1[round_no], q2[round_no]
        if pd.isna(p1) or pd.isna(p2):
            ties += 1
        elif p1 < p2:
            wins1 += 1
        elif p2 < p1:
            wins2 += 1
        else:
            ties += 1
    return {'wins1': int(wins1), 'wins2': int(wins2), 'ties': int(ties),
            'source': 'quali', 'compared': int(len(shared))}
