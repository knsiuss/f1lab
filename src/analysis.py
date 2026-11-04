# -*- coding: utf-8 -*-
"""
analysis.py
Championship standings and race statistics.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Ensure src/ is importable no matter how this module is reached (as `analysis`
# from a page, or as `src.analysis` from the test suite).
sys.path.insert(0, str(Path(__file__).parent))

from pace import data_quality, clean_laps_for_analysis, ESTIMATED  # noqa: E402

logger = logging.getLogger(__name__)


def _race_only(df: pd.DataFrame) -> pd.DataFrame:
    """Race-session rows of a season frame.

    The season frame carries race *and* sprint rows. A sprint result is not a
    race result, so every helper in this module that reasons about race
    positions filters on ``SessionType`` itself instead of trusting the caller
    to have done it -- a frame with two rows per (driver, track) otherwise
    produces duplicate index labels and interleaved totals.

    Frames without a ``SessionType`` column (older exports, unit-test
    fixtures) are returned untouched.
    """
    if df is None or df.empty or 'SessionType' not in df.columns:
        return df
    return df[df['SessionType'] == 'Race']


def _driver_teams(races: pd.DataFrame) -> dict:
    """Each driver's team in their **most recent** race (race rows only)."""
    if races is None or races.empty or 'Driver' not in races.columns \
            or 'Team' not in races.columns:
        return {}
    ordered = races
    if 'Round' in ordered.columns:
        ordered = ordered.assign(
            _round=pd.to_numeric(ordered['Round'], errors='coerce'))
        ordered = ordered.sort_values('_round', kind='stable')
    teams = ordered['Team'].replace('', np.nan)
    return teams.groupby(ordered['Driver']).last().dropna().to_dict()


def calculate_driver_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate driver statistics from race results."""
    logger.info("Calculating driver statistics...")
    
    try:
        # Validate input
        if df is None or df.empty:
            logger.error("Cannot calculate driver stats from empty DataFrame")
            return pd.DataFrame()
        
        required_cols = ['Driver', 'Points', 'Position', 'Finished']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return pd.DataFrame()
        
        # Aggregate driver statistics. 'Set Fastest Lap' is optional: it is
        # aggregated separately so a missing column degrades to zeros instead
        # of voiding the whole table (a lambda cannot guard column selection).
        driver_stats = df.groupby('Driver').agg({
            'Points': ['sum', 'mean', 'count'],
            'Position': ['mean', 'min', 'max'],
            'Finished': 'sum',
        }).round(2)
        driver_stats.columns = ['Total_Points', 'Avg_Points', 'Races', 'Avg_Position', 'Best_Position', 'Worst_Position', 'Finishes']

        # Calculate wins and podiums
        wins = df[df['Position'] == 1].groupby('Driver').size()
        podiums = df[df['Position'].between(1, 3)].groupby('Driver').size()

        driver_stats['Wins'] = wins
        driver_stats['Podium'] = podiums
        # Only the count-like columns are filled: a driver whose positions are
        # all non-numeric (retired every round) has no average/best/worst
        # position, and filling that with 0 would read as a better-than-P1
        # result. Those columns stay NaN instead.
        count_columns = ['Total_Points', 'Avg_Points', 'Races', 'Finishes',
                         'Wins', 'Podium']
        for column in count_columns:
            if column in driver_stats.columns:
                driver_stats[column] = driver_stats[column].fillna(0)

        # Fastest laps: the season frame carries empty cells unless the lap
        # data was fetched for that session. An empty column means "not
        # loaded", so the statistic is unavailable (None) for every driver --
        # never a zero, which would claim nobody ever set a fastest lap.
        fastest_available = False
        if 'Set Fastest Lap' in df.columns:
            flags = df['Set Fastest Lap'].astype(str).str.strip().str.lower()
            fastest_available = bool((flags == 'yes').any())
        if fastest_available:
            fl = df[df['Set Fastest Lap'].astype(str).str.strip().str.lower() == 'yes'] \
                .groupby('Driver').size()
            driver_stats['Fastest_Laps'] = fl.reindex(driver_stats.index).fillna(0).astype(int)
        else:
            driver_stats['Fastest_Laps'] = pd.Series(
                [None] * len(driver_stats), index=driver_stats.index, dtype=object)
        
        # Calculate percentages with division by zero protection
        driver_stats['Win_Rate'] = np.where(
            driver_stats['Races'] > 0,
            (driver_stats['Wins'] / driver_stats['Races'] * 100).round(1),
            0
        )
        driver_stats['Finish_Rate'] = np.where(
            driver_stats['Races'] > 0,
            (driver_stats['Finishes'] / driver_stats['Races'] * 100).round(1),
            0
        )
        
        logger.info(f"Calculated stats for {len(driver_stats)} drivers")
        return driver_stats
        
    except Exception as e:
        logger.exception(f"Error calculating driver statistics: {e}")
        return pd.DataFrame()



def calculate_teammate_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Head-to-Head statistics for teammates.
    
    Identifies teammate pairs for each team and compares their performance across all races.
    
    Args:
        df: Cleaned race DataFrame.
        
    Returns:
        pd.DataFrame: Teammate comparison stats.
    """
    try:
        teams = df['Team'].unique()
        comparisons = []

        # Race rows only: a sprint result is not a race result, and two rows
        # per (driver, track) would break the "one row per race" assumption
        # the comparison below is built on.
        df = _race_only(df)

        for team in teams:
            team_df = df[df['Team'] == team]
            drivers = team_df['Driver'].unique()
            
            # Need at least 2 drivers to compare
            if len(drivers) < 2:
                continue
                
            # Sort drivers by points to have a consistent order (or just pick first 2)
            # For simplicity, we take the top 2 drivers by race count or points to avoid reserve drivers skewing
            # But let's just take all pairs or the main pair.
            # Let's iterate through unique pairs
            
            processed_pairs = set()
            
            for i in range(len(drivers)):
                for j in range(i + 1, len(drivers)):
                    d1 = drivers[i]
                    d2 = drivers[j]
                    
                    pair_id = tuple(sorted((d1, d2)))
                    if pair_id in processed_pairs:
                        continue
                    processed_pairs.add(pair_id)
                    
                    # Get common races
                    d1_races = team_df[team_df['Driver'] == d1]
                    d2_races = team_df[team_df['Driver'] == d2]
                    
                    common_tracks = set(d1_races['Track']) & set(d2_races['Track'])
                    
                    d1_race_wins = 0
                    d2_race_wins = 0
                    d1_quali_wins = 0
                    d2_quali_wins = 0
                    
                    for track in common_tracks:
                        # Race H2H
                        pos1 = d1_races[d1_races['Track'] == track]['Position'].iloc[0]
                        pos2 = d2_races[d2_races['Track'] == track]['Position'].iloc[0]
                        
                        # Only compare if both finished (or use raw position if that's the standard)
                        # Standard H2H usually counts classification.
                        if pd.notna(pos1) and pd.notna(pos2):
                            if pos1 < pos2: d1_race_wins += 1
                            elif pos2 < pos1: d2_race_wins += 1
                            
                        # Quali H2H (Approximation using Starting Grid)
                        grid1 = d1_races[d1_races['Track'] == track]['Starting Grid'].iloc[0]
                        grid2 = d2_races[d2_races['Track'] == track]['Starting Grid'].iloc[0]
                        
                        if pd.notna(grid1) and pd.notna(grid2):
                            if grid1 < grid2: d1_quali_wins += 1
                            elif grid2 < grid1: d2_quali_wins += 1
                    
                    comparisons.append({
                        'Team': team,
                        'Driver 1': d1,
                        'Driver 2': d2,
                        'Races Together': len(common_tracks),
                        'Race H2H': f"{d1_race_wins} - {d2_race_wins}",
                        'Quali H2H': f"{d1_quali_wins} - {d2_quali_wins}",
                        'Pts 1': d1_races['Points'].sum(),
                        'Pts 2': d2_races['Points'].sum(),
                        'D1 Race Wins': d1_race_wins, # Store as int for helper logic
                        'D2 Race Wins': d2_race_wins
                    })
                    
        return pd.DataFrame(comparisons)
        
    except Exception as e:
        logger.error(f"Error calculating teammate comparison: {e}")
        return pd.DataFrame()


def _race_order(df: pd.DataFrame) -> list:
    """Chronological race order.

    The ``Round`` column is authoritative when the frame has one (the season
    frame is calendar-ordered by round); otherwise the first-appearance order
    of the Track column is used, which is the layout the historical exports
    had.
    """
    if df is None or df.empty or 'Track' not in df.columns:
        return []
    if 'Round' in df.columns:
        rounds = pd.to_numeric(df['Round'], errors='coerce')
        if rounds.notna().any():
            ordered = (df.assign(_round=rounds)
                         .dropna(subset=['_round'])
                         .sort_values('_round', kind='stable'))
            tracks = list(dict.fromkeys(ordered['Track'].dropna().astype(str).tolist()))
            if tracks:
                return tracks
    return list(dict.fromkeys(df['Track'].dropna().astype(str).tolist()))


def calculate_matchup(df: pd.DataFrame, driver1: str, driver2: str):
    """Head-to-head record between two drivers across their shared races.

    Finishing position is used as the race comparison; Starting Grid stands in
    for the qualifying comparison. Returns a dict with per-track ``details``
    (DataFrame) and a ``summary`` dict, or ``None`` when the pair shares no races.

    Race rows only: a sprint result is not a race result.
    """
    if df is None or df.empty or 'Driver' not in df.columns or 'Track' not in df.columns:
        return None

    df = _race_only(df)

    d1 = df[df['Driver'] == driver1]
    d2 = df[df['Driver'] == driver2]

    order = _race_order(df)
    common = set(d1['Track']) & set(d2['Track'])
    tracks = [t for t in order if t in common]
    if not tracks:
        return None

    rows = []
    race_w1 = race_w2 = quali_w1 = quali_w2 = 0
    for track in tracks:
        r1 = d1[d1['Track'] == track].iloc[0]
        r2 = d2[d2['Track'] == track].iloc[0]

        p1 = pd.to_numeric(r1.get('Position'), errors='coerce')
        p2 = pd.to_numeric(r2.get('Position'), errors='coerce')
        g1 = pd.to_numeric(r1.get('Starting Grid'), errors='coerce')
        g2 = pd.to_numeric(r2.get('Starting Grid'), errors='coerce')
        pts1 = pd.to_numeric(r1.get('Points'), errors='coerce')
        pts2 = pd.to_numeric(r2.get('Points'), errors='coerce')
        pts1 = float(pts1) if pd.notna(pts1) else 0.0
        pts2 = float(pts2) if pd.notna(pts2) else 0.0

        if pd.notna(p1) and pd.notna(p2):
            if p1 < p2:
                race_w1 += 1
            elif p2 < p1:
                race_w2 += 1
        if pd.notna(g1) and pd.notna(g2):
            if g1 < g2:
                quali_w1 += 1
            elif g2 < g1:
                quali_w2 += 1

        rows.append({
            'Track': track,
            driver1: p1, driver2: p2,
            f'{driver1} Grid': g1, f'{driver2} Grid': g2,
            f'{driver1} Pts': pts1, f'{driver2} Pts': pts2,
        })

    details = pd.DataFrame(rows)
    return {
        'details': details,
        'summary': {
            'driver1': driver1,
            'driver2': driver2,
            'races_together': len(tracks),
            'race_record': {'driver1': race_w1, 'driver2': race_w2},
            'quali_record': {'driver1': quali_w1, 'driver2': quali_w2},
            'points': {
                'driver1': int(details[f'{driver1} Pts'].sum()),
                'driver2': int(details[f'{driver2} Pts'].sum()),
            },
            'avg_position': {
                'driver1': round(details[driver1].mean(), 2),
                'driver2': round(details[driver2].mean(), 2),
            },
        },
    }


def _is_finished_status(status) -> bool:
    """True when a ``Status`` value means the driver finished the race.

    Frames without a ``Status`` column (unit-test fixtures, older exports) are
    treated as all-finished so the rolling logic keeps working for them.
    """
    if status is None or (isinstance(status, float) and pd.isna(status)):
        return True
    return str(status).strip().lower() in ('finished', '')


def calculate_form_trend(df: pd.DataFrame, drivers: list, window: int = 3) -> dict:
    """Rolling-average finishing position ("form") plus per-race positions.

    Returns a dict with ``form`` (driver x race, rolling mean position, lower is
    better) and ``positions`` (driver x race finishing position), both indexed by
    race name in chronological order.

    Race rows only: a sprint result is not a race result. Only *finished* races
    enter a rolling average: a DNF/DNS/DSQ is not an ordinary finishing
    position, so when the frame carries a ``Status`` column those rows are
    excluded and the driver shows a gap (NaN) for that round instead of a
    fabricated point. Frames without ``Status`` keep the historical behaviour
    (every numeric ``Position`` counts).

    The rolling mean is taken over the driver's own finished sequence (the last
    ``window`` finishes), so a missed round does not dilute the average; the
    frame is then aligned back onto the full calendar so gaps stay visible.
    """
    empty = {'form': pd.DataFrame(), 'positions': pd.DataFrame()}
    if df is None or df.empty or 'Driver' not in df.columns:
        return empty

    df = _race_only(df)
    has_status = 'Status' in df.columns

    order = _race_order(df)
    idx = {t: i for i, t in enumerate(order)}
    series = {}
    positions = {}

    for d in drivers:
        dd = df[df['Driver'] == d].copy()
        if dd.empty:
            continue
        if has_status:
            dd = dd[dd['Status'].map(_is_finished_status)]
            if dd.empty:
                continue
        dd['_pos'] = pd.to_numeric(dd['Position'], errors='coerce')
        dd['_ord'] = dd['Track'].map(idx)
        dd = dd.dropna(subset=['_pos', '_ord']).sort_values('_ord')
        if dd.empty:
            continue
        keyed = dd.set_index('_ord')
        series[d] = keyed['_pos'].rolling(window, min_periods=1).mean()
        positions[d] = keyed['_pos']

    def align(frame: pd.DataFrame) -> pd.DataFrame:
        """Relabel + align onto the full calendar (missing = NaN gap)."""
        frame = frame.copy()
        frame.index = [order[int(i)] if int(i) < len(order) else int(i)
                       for i in frame.index]
        if order:
            frame = frame.reindex(order)
        return frame

    form = align(pd.DataFrame(series)).round(2) if series else pd.DataFrame()
    pos = align(pd.DataFrame(positions)) if positions else pd.DataFrame()
    return {'form': form, 'positions': pos}


# ---------------------------------------------------------------------------
# Form Analysis calculations (Head-to-Head "Form" tab)
# ---------------------------------------------------------------------------

# Race-name -> three-letter code for the heatmap axis. Keyed on the lowercased
# track name without the "Grand Prix" suffix; anything unlisted falls back to
# the first word of the name so a new calendar still gets a short label.
RACE_ABBREV = {
    'australian': 'AUS', 'chinese': 'CHN', 'japanese': 'JPN',
    'bahrain': 'BHR', 'saudi arabian': 'SAU', 'miami': 'MIA',
    'emilia romagna': 'EMI', 'monaco': 'MON', 'spanish': 'ESP',
    'canadian': 'CAN', 'austrian': 'AUT', 'british': 'GBR',
    'belgian': 'BEL', 'hungarian': 'HUN', 'dutch': 'NLD',
    'italian': 'ITA', 'azerbaijan': 'AZE', 'singapore': 'SGP',
    'united states': 'USA', 'mexico city': 'MEX', 'sao paulo': 'BRA',
    'são paulo': 'BRA', 'las vegas': 'LVG', 'qatar': 'QAT',
    'abu dhabi': 'ABU', 'barcelona': 'ESP', 'madrid': 'MAD',
}


def race_abbreviation(track) -> str:
    """Three-letter race code for ``track`` (falls back to first word)."""
    text = str(track or '').strip()
    if not text:
        return ''
    core = text.replace('Grand Prix', '').strip().lower()
    if core in RACE_ABBREV:
        return RACE_ABBREV[core]
    first = core.split()[0] if core.split() else ''
    return first[:3].upper() or text[:3].upper()


def form_finished_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Finished race rows with the columns the Form tab needs.

    Columns: Track, Ord, Driver, Team, Position, Points, Status, Round, Grid.
    Frames without ``Status`` treat every numeric ``Position`` as finished.
    """
    if df is None or df.empty or 'Driver' not in df.columns \
            or 'Track' not in df.columns:
        return pd.DataFrame(columns=[
            'Track', 'Ord', 'Driver', 'Team', 'Position', 'Points',
            'Status', 'Round', 'Grid',
        ])
    races = _race_only(df)
    if 'Status' in races.columns:
        races = races[races['Status'].map(_is_finished_status)]
    order = _race_order(races)
    idx = {t: i for i, t in enumerate(order)}
    out = pd.DataFrame({
        'Track': races['Track'].astype(str),
        'Ord': races['Track'].map(idx),
        'Driver': races['Driver'].astype(str),
        'Team': (races['Team'].astype(str) if 'Team' in races.columns
                 else ''),
        'Position': pd.to_numeric(races['Position'], errors='coerce')
        if 'Position' in races.columns else np.nan,
        'Points': pd.to_numeric(races['Points'], errors='coerce').fillna(0.0)
        if 'Points' in races.columns else 0.0,
        'Status': (races['Status'].astype(str) if 'Status' in races.columns
                   else 'Finished'),
        'Round': pd.to_numeric(races['Round'], errors='coerce')
        if 'Round' in races.columns else np.nan,
        'Grid': pd.to_numeric(races['Starting Grid'], errors='coerce')
        if 'Starting Grid' in races.columns else np.nan,
    })
    return out.dropna(subset=['Position', 'Ord']).sort_values(
        ['Driver', 'Ord'], kind='stable').reset_index(drop=True)


def _window_means(positions, recent: int = 5, previous: int = 5):
    """(last-``recent`` mean, previous-``previous`` mean, delta) for a series.

    ``delta`` is positive when recent form is worse (higher average position).
    Returns ``(None, None, None)`` when the driver has too few finishes for a
    full previous window; callers use the available-finishes fallback instead.
    """
    values = [float(v) for v in positions if pd.notna(v)]
    if len(values) < recent + previous:
        return None, None, None
    last = values[-recent:]
    prior = values[-recent - previous:-recent]
    last_mean, prior_mean = float(np.mean(last)), float(np.mean(prior))
    return last_mean, prior_mean, last_mean - prior_mean


def form_driver_stats(df: pd.DataFrame, drivers) -> pd.DataFrame:
    """Per-driver form statistics over finished races.

    Columns: Driver, Finishes, AvgPos, StdPos, BestPos, WorstPos, Wins,
    Podiums, PodiumRate, PointsPerFinish, RacePoints, Last5Avg, Prev5Avg,
    Momentum, BucketP1/P2P3/P4P10/P11Plus. ``StdPos`` is the sample standard
    deviation; drivers with fewer than 3 finishes have no std (None).
    """
    frames = form_finished_frame(df)
    rows = []
    for driver in drivers:
        group = frames[frames['Driver'] == driver] if not frames.empty \
            else pd.DataFrame()
        positions = group['Position'].tolist() if not group.empty else []
        n = len(positions)
        std = float(np.std(positions, ddof=1)) if n >= 3 else None
        last5 = float(np.mean(positions[-5:])) if n >= 1 else None
        last_mean, prior_mean, delta = _window_means(positions)
        rows.append({
            'Driver': driver,
            'Finishes': n,
            'AvgPos': float(np.mean(positions)) if n else None,
            'StdPos': std,
            'BestPos': int(min(positions)) if n else None,
            'WorstPos': int(max(positions)) if n else None,
            'Wins': int(sum(1 for p in positions if float(p) == 1)),
            'Podiums': int(sum(1 for p in positions if float(p) <= 3)),
            'PodiumRate': (100.0 * sum(1 for p in positions if float(p) <= 3) / n
                           if n else None),
            'RacePoints': float(group['Points'].sum()) if n else 0.0,
            'PointsPerFinish': (float(group['Points'].sum()) / n if n
                                else None),
            'Last5Avg': last5,
            'Prev5Avg': prior_mean,
            'Momentum': delta,
            'LastTracks': (group['Track'].tolist()[-5:] if n else []),
            'PrevTracks': (group['Track'].tolist()[-10:-5]
                           if last_mean is not None else []),
            'BucketP1': sum(1 for p in positions if float(p) == 1),
            'BucketP2P3': sum(1 for p in positions if 2 <= float(p) <= 3),
            'BucketP4P10': sum(1 for p in positions if 4 <= float(p) <= 10),
            'BucketP11Plus': sum(1 for p in positions if float(p) >= 11),
        })
    return pd.DataFrame(rows)


def _fallback_delta(positions):
    """First-half vs second-half delta when a 5v5 split is impossible.

    Used only when no selected driver has 10+ finishes (early season): the 3
    most recent finishes are compared with the 3 finishes before them.
    """
    values = [float(v) for v in positions if pd.notna(v)]
    if len(values) < 6:
        return None, None, None
    last = float(np.mean(values[-3:]))
    prior = float(np.mean(values[-6:-3]))
    return last, prior, last - prior


def form_kpis(df: pd.DataFrame, drivers) -> dict:
    """The four Form KPI cards, each a dict or ``None`` when insufficient.

    * most_consistent: lowest ``StdPos`` (needs 3+ finishes).
    * best_recent: lowest ``Last5Avg``.
    * biggest_drop / strongest_recovery: largest ``delta`` (last-5 minus
      previous-5, 5v5; 3v3 fallback early in the season).
    """
    stats = form_driver_stats(df, drivers)
    if stats.empty:
        return {}
    frames = form_finished_frame(df)

    out: dict = {}
    consistent = stats.dropna(subset=['StdPos'])
    if not consistent.empty:
        row = consistent.sort_values('StdPos', kind='stable').iloc[0]
        field = stats['StdPos'].dropna()
        out['most_consistent'] = {
            'driver': str(row['Driver']),
            'std': float(row['StdPos']),
            'field_median': float(field.median()) if not field.empty else None,
            'finishes': int(row['Finishes']),
        }

    recent = stats.dropna(subset=['Last5Avg'])
    if not recent.empty:
        row = recent.sort_values('Last5Avg', kind='stable').iloc[0]
        trend = ''
        if pd.notna(row['Momentum']):
            trend = 'improving' if row['Momentum'] < 0 else 'slipping'
        out['best_recent'] = {
            'driver': str(row['Driver']),
            'avg': float(row['Last5Avg']),
            'finishes': int(row['Finishes']),
            'trend': trend,
        }

    def delta_for(driver: str):
        positions = (frames[frames['Driver'] == driver]['Position'].tolist()
                     if not frames.empty else [])
        last, prior, delta = _window_means(positions)
        if delta is None:
            last, prior, delta = _fallback_delta(positions)
        return last, prior, delta, positions

    candidates = []
    for driver in drivers:
        last, prior, delta, _positions = delta_for(driver)
        if delta is None:
            continue
        group = frames[frames['Driver'] == driver] if not frames.empty \
            else pd.DataFrame()
        tracks = group['Track'].tolist() if not group.empty else []
        candidates.append({
            'driver': str(driver),
            'last': float(last), 'prior': float(prior), 'delta': float(delta),
            'prev_tracks': tracks[-10:-5] if len(tracks) >= 10 else tracks[-6:-3],
            'last_tracks': tracks[-5:] if len(tracks) >= 10 else tracks[-3:],
        })
    if candidates:
        drop = max(candidates, key=lambda c: c['delta'])
        recovery = min(candidates, key=lambda c: c['delta'])
        if drop['delta'] > 0:
            out['biggest_drop'] = drop
        if recovery['delta'] < 0:
            out['strongest_recovery'] = recovery
    return out


def form_annotations(df: pd.DataFrame, drivers) -> list:
    """Chart annotations: best/worst finish, drop, recovery, consistency."""
    stats = form_driver_stats(df, drivers)
    items: list = []
    if stats.empty:
        return items
    frames = form_finished_frame(df)

    def single_extreme(best: bool):
        if frames.empty:
            return None
        sub = frames[frames['Driver'].isin(list(drivers))].copy()
        if sub.empty:
            return None
        row = sub.sort_values(
            'Position', ascending=best, kind='stable').iloc[0]
        return {'driver': str(row['Driver']), 'track': str(row['Track']),
                'position': int(row['Position'])}

    best = single_extreme(True)
    worst = single_extreme(False)
    if best:
        items.append({'kind': 'best', 'label': 'Best performance',
                      'text': f"{best['driver']} · P{best['position']} · "
                              f"{best['track']}"})
    if worst:
        items.append({'kind': 'worst', 'label': 'Worst performance',
                      'text': f"{worst['driver']} · P{worst['position']} · "
                              f"{worst['track']}"})
    kpis = form_kpis(df, drivers)
    if 'biggest_drop' in kpis:
        drop = kpis['biggest_drop']
        items.append({'kind': 'drop', 'label': 'Largest drop',
                      'text': f"{drop['driver']} · P{drop['prior']:.1f} → "
                              f"P{drop['last']:.1f} · "
                              f"{_track_span(drop['prev_tracks'])}→"
                              f"{_track_span(drop['last_tracks'])}"})
    if 'strongest_recovery' in kpis:
        rec = kpis['strongest_recovery']
        items.append({'kind': 'recovery', 'label': 'Strongest recovery',
                      'text': f"{rec['driver']} · P{rec['prior']:.1f} → "
                              f"P{rec['last']:.1f} · "
                              f"{_track_span(rec['prev_tracks'])}→"
                              f"{_track_span(rec['last_tracks'])}"})
    if 'most_consistent' in kpis:
        cons = kpis['most_consistent']
        items.append({'kind': 'consistent', 'label': 'Most consistent',
                      'text': f"{cons['driver']} · std {cons['std']:.2f} "
                              f"over {cons['finishes']} finished races"})
    return items


def _track_span(tracks) -> str:
    """Short ``AUS-CHN`` span for a list of race names ('' when empty)."""
    codes = [race_abbreviation(t) for t in (tracks or []) if t]
    if not codes:
        return ''
    if len(codes) == 1:
        return codes[0]
    return f"{codes[0]}-{codes[-1]}"


def form_takeaways(df: pd.DataFrame, drivers) -> list:
    """Three to five insights, each calculated from the finished races."""
    stats = form_driver_stats(df, drivers)
    items: list = []
    if stats.empty:
        return items
    kpis = form_kpis(df, drivers)

    if 'most_consistent' in kpis:
        c = kpis['most_consistent']
        median = c.get('field_median')
        tail = (f", against a field median of {median:.2f}"
                if median is not None else '')
        items.append((
            'Most consistent',
            f"{c['driver']} has the lowest finishing-position spread "
            f"(std {c['std']:.2f} over {c['finishes']} finished races{tail}).",
        ))
    if 'best_recent' in kpis:
        b = kpis['best_recent']
        items.append((
            'Strongest recent momentum',
            f"{b['driver']} averages P{b['avg']:.1f} over the last "
            f"{min(5, b['finishes'])} finished races"
            + (f" and is {b['trend']}." if b['trend'] else "."),
        ))
    podium = stats.dropna(subset=['PodiumRate'])
    if not podium.empty:
        row = podium.sort_values(
            ['PodiumRate', 'Wins'], ascending=False, kind='stable').iloc[0]
        items.append((
            'Best podium conversion',
            f"{row['Driver']} converts {row['PodiumRate']:.0f}% of "
            f"finished races into podiums ({int(row['Podiums'])} of "
            f"{int(row['Finishes'])}).",
        ))
    volatile = stats.dropna(subset=['StdPos'])
    if not volatile.empty:
        row = volatile.sort_values('StdPos', ascending=False,
                                   kind='stable').iloc[0]
        items.append((
            'Highest volatility',
            f"{row['Driver']} shows the widest spread of finishes "
            f"(std {row['StdPos']:.2f}), the most race-to-race swing in "
            f"this group.",
        ))
    if 'biggest_drop' in kpis:
        d = kpis['biggest_drop']
        items.append((
            'Biggest decline',
            f"{d['driver']} slipped from P{d['prior']:.1f} to "
            f"P{d['last']:.1f} between the two comparison windows "
            f"({_track_span(d['prev_tracks'])} to {_track_span(d['last_tracks'])}).",
        ))
    elif 'strongest_recovery' in kpis:
        r = kpis['strongest_recovery']
        items.append((
            'Biggest decline',
            f"No driver faded across the comparison windows; the sharpest "
            f"move is {r['driver']} improving from P{r['prior']:.1f} to "
            f"P{r['last']:.1f}.",
        ))
    # Cap at five; keep the strongest five in the order above.
    return items[:5]


# Heatmap cell categories in display order: key -> (label, bucket).
HEATMAP_BUCKETS = [
    ('p1', 'P1 (Win)', lambda code, pos: code == 'p1'),
    ('p2p3', 'P2 - P3 (Podium)', lambda code, pos: code == 'p2p3'),
    ('p4p6', 'P4 - P6', lambda code, pos: code == 'p4p6'),
    ('p7p10', 'P7 - P10', lambda code, pos: code == 'p7p10'),
    ('p11p15', 'P11 - P15', lambda code, pos: code == 'p11p15'),
    ('p16p20', 'P16 - P20', lambda code, pos: code == 'p16p20'),
    ('dnf', 'DNF', lambda code, pos: code == 'dnf'),
    ('dns', 'DNS', lambda code, pos: code == 'dns'),
    ('dsq', 'DSQ', lambda code, pos: code == 'dsq'),
    ('none', 'No entry', lambda code, pos: code == 'none'),
]


def _heatmap_code(status, position) -> str:
    """One heatmap cell: p1/p2p3/.../dnf/dns/dsq/none."""
    label = str(status or '').strip().upper()
    if label and label != 'FINISHED':
        if label in ('DNF', 'DNS', 'DSQ'):
            return label.lower()
        return 'none'
    try:
        pos = float(position)
    except (TypeError, ValueError):
        return 'none'
    if pd.isna(pos):
        return 'none'
    if pos == 1:
        return 'p1'
    if pos <= 3:
        return 'p2p3'
    if pos <= 6:
        return 'p4p6'
    if pos <= 10:
        return 'p7p10'
    if pos <= 15:
        return 'p11p15'
    return 'p16p20'


def form_heatmap(df: pd.DataFrame, drivers) -> dict:
    """Race-by-race matrix with DNF/DNS/DSQ kept distinct from finishes.

    Returns ``codes`` (driver x race bucket codes), ``text`` (P12/DNF/...),
    ``hover`` (full race detail per cell), ``races`` (ordered race names),
    ``abbr`` (race codes), and ``stats`` (avg/wins/podiums per driver).
    """
    if df is None or df.empty or 'Driver' not in df.columns \
            or 'Track' not in df.columns:
        return {'codes': pd.DataFrame(), 'text': pd.DataFrame(),
                'hover': {}, 'races': [], 'abbr': [], 'stats': pd.DataFrame()}
    races = _race_only(df)
    order = _race_order(races)

    lookup = {}
    if 'Status' in races.columns:
        statuses = races['Status'].astype(str)
    else:
        statuses = pd.Series(['Finished'] * len(races), index=races.index)
    for track, driver, status, position in zip(
            races['Track'].astype(str), races['Driver'].astype(str),
            statuses, (races['Position'] if 'Position' in races.columns
                       else [None] * len(races))):
        key = (driver, str(track))
        # First row wins: one row per (driver, track) after the race filter.
        lookup.setdefault(key, (str(status), position))

    codes, text = {}, {}
    for driver in drivers:
        code_row, text_row = [], []
        for track in order:
            status, position = lookup.get((driver, track), ('', None))
            code = _heatmap_code(status, position)
            code_row.append(code)
            if code == 'none':
                # "Did not enter this round" renders as an empty cell; the
                # hover text below says so instead of inventing a position.
                text_row.append('')
                continue
            if code in ('dnf', 'dns', 'dsq'):
                text_row.append(code.upper())
            else:
                text_row.append(f"P{int(float(position))}")
        codes[driver] = code_row
        text[driver] = text_row
    # Hover detail needs one entry per cell (Plotly reads 2D arrays).
    hover_grid = pd.DataFrame(
        {driver: [form_heatmap_hover(
            df, driver, track) for track in order] for driver in drivers},
        index=order).T if order and drivers else pd.DataFrame()

    code_frame = pd.DataFrame(codes, index=order).T
    text_frame = pd.DataFrame(text, index=order).T
    stats = form_driver_stats(df, drivers)[
        ['Driver', 'AvgPos', 'Wins', 'Podiums', 'Finishes']]
    return {'codes': code_frame, 'text': text_frame, 'hover': hover_grid,
            'races': list(order),
            'abbr': [race_abbreviation(t) for t in order], 'stats': stats}


def form_heatmap_hover(df: pd.DataFrame, driver: str, track: str) -> str:
    """Full race detail for one heatmap cell hover."""
    races = _race_only(df)
    rows = races[(races['Driver'] == driver)
                 & (races['Track'].astype(str) == str(track))]
    if rows.empty:
        return f"{driver}<br>{track}<br>No entry in this round"
    row = rows.iloc[0]
    status = str(row.get('Status', 'Finished') or 'Finished')
    label = status if status.upper() not in ('', 'FINISHED') else 'Finished'
    if status.upper() in ('DNF', 'DNS', 'DSQ'):
        finish = status.upper()
    else:
        try:
            pos = int(float(row.get('Position')))
            finish = f"P{pos}"
        except (TypeError, ValueError):
            finish = label
    parts = [str(driver), str(track), finish]
    try:
        pts = float(row.get('Points'))
        if pd.notna(pts):
            parts.append(f"{pts:.0f} pts")
    except (TypeError, ValueError):
        pass
    try:
        grid = int(float(row.get('Starting Grid')))
        parts.append(f"Grid P{grid}")
    except (TypeError, ValueError):
        pass
    team = str(row.get('Team') or '').strip()
    if team:
        parts.append(team)
    return '<br>'.join(parts)



def calculate_points_trajectory(df: pd.DataFrame, drivers: list) -> pd.DataFrame:
    """Cumulative championship points per driver across the season.

    Returns a DataFrame indexed by race name (chronological) with one column per
    driver holding their running season total.

    Race rows only: a sprint result is not a race result, and a frame with one
    row per (driver, track, session) would interleave the sprint points into
    the race step. For the true championship curve -- race *and* sprint points
    counted at the round they happened -- use :func:`championship_trajectory`.
    """
    if df is None or df.empty or 'Driver' not in df.columns:
        return pd.DataFrame()

    df = _race_only(df)

    order = _race_order(df)
    idx = {t: i for i, t in enumerate(order)}
    series = {}

    for d in drivers:
        dd = df[df['Driver'] == d].copy()
        dd['_pts'] = pd.to_numeric(dd['Points'], errors='coerce').fillna(0)
        dd['_ord'] = dd['Track'].map(idx)
        dd = dd.dropna(subset=['_ord']).sort_values('_ord')
        if dd.empty:
            continue
        series[d] = dd.set_index('_ord')['_pts'].cumsum()

    traj = pd.DataFrame(series)
    labels = [order[int(i)] if int(i) < len(order) else int(i) for i in traj.index]
    traj.index = labels
    return traj.round(1)


def championship_trajectory(df: pd.DataFrame, drivers: list) -> pd.DataFrame:
    """Cumulative **championship** points per driver (race + sprint).

    The season frame holds a race row and a sprint row for the same weekend.
    Both count towards the championship, so the points of a round are summed
    per (driver, track) first and the running total is taken over the rounds.
    That keeps one point per round on the curve -- no interleaved sprint jump
    in the middle of a race step.

    Returns a DataFrame indexed by event name (chronological) with one column
    per driver holding their running championship total.
    """
    if df is None or df.empty or 'Driver' not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    if 'Track' not in work.columns:
        return pd.DataFrame()
    work['_pts'] = pd.to_numeric(work['Points'], errors='coerce').fillna(0)
    # One row per (driver, track): race + sprint of the same weekend collapse.
    per_track = (work.groupby(['Driver', 'Track'], as_index=False)['_pts']
                 .sum())
    order = _race_order(per_track)
    idx = {t: i for i, t in enumerate(order)}

    series = {}
    for d in drivers:
        dd = per_track[per_track['Driver'] == d].copy()
        dd['_ord'] = dd['Track'].map(idx)
        dd = dd.dropna(subset=['_ord']).sort_values('_ord')
        if dd.empty:
            continue
        series[d] = dd.set_index('_ord')['_pts'].cumsum()

    traj = pd.DataFrame(series)
    if traj.empty:
        return traj
    labels = [order[int(i)] if int(i) < len(order) else int(i) for i in traj.index]
    traj.index = labels
    return traj.round(1)


# ---------------------------------------------------------------------------
# Season frame helpers (used by the dashboard pages)
# ---------------------------------------------------------------------------

def latest_race_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the most recent race of a season frame.

    "Most recent" means the highest round (the frame is calendar-ordered), and
    only the race session of that round -- a sprint of the same weekend is not
    part of the race result. Frames without a usable ``Round`` fall back to the
    last track in first-appearance order.

    Returns an empty frame when there is no race to show.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    races = _race_only(df)
    if races.empty or 'Track' not in races.columns:
        return pd.DataFrame()

    if 'Round' in races.columns:
        rounds = pd.to_numeric(races['Round'], errors='coerce')
        if rounds.notna().any():
            latest = rounds.max()
            return races[rounds == latest].copy()

    tracks = list(dict.fromkeys(races['Track'].dropna().astype(str).tolist()))
    if not tracks:
        return pd.DataFrame()
    return races[races['Track'].astype(str) == tracks[-1]].copy()


def points_progression(df: pd.DataFrame, drivers: list = None) -> pd.DataFrame:
    """Running championship total per driver, one row per round.

    The points of a round (race *and* sprint of the same weekend) are summed
    per (driver, track) first, so the curve has one step per round and no
    interleaved sprint jump inside a race step.

    Returns a DataFrame with columns Driver, Track, RaceNum, Points (the
    running championship total) and, when the frame has one, Round.
    """
    if df is None or df.empty or 'Driver' not in df.columns:
        return pd.DataFrame()
    if 'Track' not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work['_pts'] = pd.to_numeric(work['Points'], errors='coerce').fillna(0)
    keys = ['Driver', 'Track']
    if 'Round' in work.columns:
        keys.append('Round')
    per_track = work.groupby(keys, as_index=False, dropna=False)['_pts'].sum()

    order = _race_order(per_track)
    idx = {t: i for i, t in enumerate(order)}
    per_track['_ord'] = per_track['Track'].map(idx)
    per_track = per_track.dropna(subset=['_ord']).sort_values(['Driver', '_ord'])
    if per_track.empty:
        return pd.DataFrame()

    per_track['Points'] = per_track.groupby('Driver')['_pts'].cumsum()
    per_track['RaceNum'] = per_track.groupby('Driver').cumcount() + 1
    per_track = per_track.rename(columns={'_pts': 'RoundPoints'})

    if drivers:
        per_track = per_track[per_track['Driver'].isin(drivers)]
    columns = ['Driver', 'Track', 'RaceNum', 'Points']
    if 'Round' in per_track.columns:
        columns.append('Round')
    return per_track[columns].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Session (lap) helpers
# ---------------------------------------------------------------------------

def count_lead_changes(laps: pd.DataFrame) -> int:
    """Number of times the race leader changed hands.

    FastF1's lap table is grouped by driver, so the leader has to be read in
    lap order: a driver who loses the lead and takes it back counts as two
    changes, which a driver-grouped sequence cannot see.

    Returns 0 when the lap table carries no usable position data.
    """
    if laps is None or laps.empty:
        return 0
    if 'Driver' not in laps.columns or 'Position' not in laps.columns:
        return 0

    work = laps[['Driver', 'Position']].copy()
    if 'LapNumber' in laps.columns:
        work['LapNumber'] = laps['LapNumber']
    work['Position'] = pd.to_numeric(work['Position'], errors='coerce')
    work = work.dropna(subset=['Position'])
    work = work[work['Position'] == 1]
    if work.empty:
        return 0
    if 'LapNumber' in work.columns:
        work = work.sort_values('LapNumber', kind='stable')

    leaders = work['Driver'].to_numpy()
    if len(leaders) < 2:
        return 0
    return int((leaders[1:] != leaders[:-1]).sum())


def gap_to_leader(laps: pd.DataFrame, drivers: list = None):
    """Per-driver gap to the race leader, lap by lap.

    The reference is the driver who completed the **most laps** (ties broken
    by the best finishing position), not the driver with the lowest cumulative
    lap time -- a driver who crashed on lap 5 has the lowest cumulative time of
    the field and would otherwise become the "leader" of the chart.

    The gap is only reported for laps the reference driver actually completed;
    every other lap is left empty rather than silently reset to zero.

    Returns ``(leader, frame)`` where ``frame`` has columns Driver, LapNumber
    and Gap (seconds, positive = behind the leader).
    """
    empty = (None, pd.DataFrame(columns=['Driver', 'LapNumber', 'Gap']))
    if laps is None or laps.empty:
        return empty
    if 'Driver' not in laps.columns or 'LapTime' not in laps.columns:
        return empty
    if 'LapNumber' not in laps.columns:
        return empty

    work = laps[['Driver', 'LapNumber', 'LapTime']].copy()
    work['LapNumber'] = pd.to_numeric(work['LapNumber'], errors='coerce')
    work = work.dropna(subset=['LapNumber'])
    work = work[work['LapTime'].notna()]
    if work.empty:
        return empty

    work['LapTime_s'] = work['LapTime'].dt.total_seconds()
    work = work.sort_values(['Driver', 'LapNumber'], kind='stable')
    work['CumTime'] = work.groupby('Driver')['LapTime_s'].cumsum()

    if drivers:
        selected = [d for d in drivers if d in set(work['Driver'])]
        if not selected:
            return empty
        work = work[work['Driver'].isin(selected)]

    # The reference driver: most laps completed, best position as tie-break.
    positions = (pd.to_numeric(laps['Position'], errors='coerce')
                 if 'Position' in laps.columns else pd.Series(dtype=float))
    best_position = {}
    if not positions.empty:
        tmp = laps[['Driver']].copy()
        tmp['_pos'] = positions.to_numpy()
        tmp = tmp.dropna(subset=['_pos'])
        if not tmp.empty:
            best_position = tmp.groupby('Driver')['_pos'].min().to_dict()

    completed = work.groupby('Driver')['LapNumber'].nunique()
    if completed.empty:
        return empty
    ranking = sorted(
        completed.index,
        key=lambda d: (-completed[d], best_position.get(d, float('inf'))),
    )
    leader = ranking[0]

    reference = (work[work['Driver'] == leader]
                 .set_index('LapNumber')['CumTime'].to_dict())

    rows = []
    for driver, group in work.groupby('Driver'):
        for _, r in group.iterrows():
            lap = int(r['LapNumber'])
            ref = reference.get(lap)
            if ref is None:
                continue  # the reference driver did not complete this lap
            rows.append({
                'Driver': driver,
                'LapNumber': lap,
                'Gap': round(float(r['CumTime'] - ref), 3),
            })
    frame = pd.DataFrame(rows, columns=['Driver', 'LapNumber', 'Gap'])
    return leader, frame


# Weights and scalings of the driver performance score. They live here, not
# in the page, so the numbers behind the leaderboard are tunable in one place.
PERFORMANCE_WEIGHTS = {'pace': 0.40, 'consistency': 0.35, 'race_pace': 0.25}
PACE_SCALE = 10.0
CONSISTENCY_SCALE = 20.0
RACE_PACE_SCALE = 5.0


def driver_performance_scores(laps: pd.DataFrame, min_laps: int = 2) -> pd.DataFrame:
    """Per-driver performance scores, estimated, from **clean laps only**.

    The score is a transparent blend of three normalised components:

        * pace         -- how close the driver's best clean lap is to the
                         session best;
        * consistency  -- the standard deviation of the clean lap times;
        * race pace    -- how the driver's mean clean lap compares with the
                         field median.

    Every score is an estimate: the frame carries the number of clean laps
    behind each row plus the ``Level``/``Confidence`` from
    ``pace.data_quality``. Drivers without enough clean laps get no row at
    all instead of a score built on noise.

    Returns a DataFrame with columns Driver, Team, Pace, Consistency,
    Race Pace, Overall, N (clean laps), Level, Confidence, Estimated.
    """
    if laps is None or laps.empty or 'Driver' not in laps.columns:
        return pd.DataFrame()

    clean = clean_laps_for_analysis(laps)
    if clean.empty or 'LapTime_s' not in clean.columns:
        return pd.DataFrame()

    all_times = clean['LapTime_s'].dropna()
    if all_times.empty:
        return pd.DataFrame()
    session_best = float(all_times.min())
    field_median = float(all_times.median())

    rows = []
    for driver, group in clean.groupby('Driver', dropna=False):
        times = group['LapTime_s'].dropna()
        n = len(times)
        if n < max(2, int(min_laps)):
            continue
        quality = data_quality(n)
        pace = max(0.0, 100.0 - (float(times.min()) - session_best) * PACE_SCALE)
        consistency = max(0.0, 100.0 - float(times.std()) * CONSISTENCY_SCALE)
        race_pace = max(0.0, 100.0 - (float(times.mean()) - field_median) * RACE_PACE_SCALE)
        overall = (pace * PERFORMANCE_WEIGHTS['pace']
                   + consistency * PERFORMANCE_WEIGHTS['consistency']
                   + race_pace * PERFORMANCE_WEIGHTS['race_pace'])
        team = ''
        if 'Team' in group.columns:
            team_values = group['Team'].replace('', np.nan).dropna()
            if not team_values.empty:
                team = str(team_values.iloc[-1])
        rows.append({
            'Driver': driver,
            'Team': team,
            'Pace': round(pace, 1),
            'Consistency': round(consistency, 1),
            'Race Pace': round(race_pace, 1),
            'Overall': round(overall, 1),
            'N': n,
            'Level': quality['level'],
            'Confidence': quality['confidence'],
            'Estimated': ESTIMATED,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values('Overall', ascending=False).reset_index(drop=True)


def top3_overlap(predicted_drivers: list, actual_positions: dict, n: int = 3) -> int:
    """How many drivers are in both the predicted and the actual top ``n``.

    A driver counts when they are in the predicted top ``n`` **and** their
    recorded finishing position is within the top ``n``. The actual podium is
    built from the recorded positions, not from the first rows of the
    prediction: a predicted driver without a recorded finish would otherwise
    pull the next predicted driver into the comparison and inflate the count.

    Args:
        predicted_drivers: driver names in predicted order (best first).
        actual_positions: ``{driver: finishing position}``.
        n: size of the top group (default 3).

    Returns:
        The number of drivers in both top groups.
    """
    if not predicted_drivers or not actual_positions:
        return 0
    predicted_top = set(list(predicted_drivers)[:n])
    actual_top = {driver for driver, position in actual_positions.items()
                  if pd.notna(position) and float(position) <= n}
    return len(predicted_top & actual_top)


def compare_driver_pace(laps: pd.DataFrame, driver_a: str, driver_b: str) -> dict:
    """Head-to-head pace of two drivers from **clean laps only**.

    Raw lap times include the pit-entry lap, the pit-exit lap and laps run
    behind a Safety Car, none of which are comparable to a clean lap, so both
    drivers are cleaned first (:func:`pace.clean_laps_for_analysis`).

    Returns a dict with the cleaned lap frames, the mean lap times, the
    difference (negative = A faster), the clean-lap counts and the
    ``Level``/``Confidence`` of the weaker sample. ``level`` is ``insufficient``
    when either driver has fewer than two clean laps.
    """
    result = {
        'driver_a': driver_a, 'driver_b': driver_b,
        'laps_a': pd.DataFrame(), 'laps_b': pd.DataFrame(),
        'avg_a': None, 'avg_b': None, 'diff': None,
        'n_a': 0, 'n_b': 0,
        'level': 'insufficient', 'confidence': 'low', 'estimated': ESTIMATED,
    }
    if laps is None or laps.empty or 'Driver' not in laps.columns:
        return result

    clean = clean_laps_for_analysis(laps)
    if clean.empty:
        return result

    a = clean[clean['Driver'] == driver_a]
    b = clean[clean['Driver'] == driver_b]
    result['laps_a'] = a
    result['laps_b'] = b

    n_a, n_b = len(a), len(b)
    result['n_a'], result['n_b'] = n_a, n_b
    if n_a == 0 or n_b == 0:
        return result

    avg_a = float(a['LapTime_s'].mean())
    avg_b = float(b['LapTime_s'].mean())
    result['avg_a'], result['avg_b'] = avg_a, avg_b
    result['diff'] = avg_a - avg_b

    # The comparison is only as trustworthy as the smaller of the two samples.
    quality = data_quality(min(n_a, n_b))
    result['level'] = quality['level']
    result['confidence'] = quality['confidence']
    return result
