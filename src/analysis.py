# -*- coding: utf-8 -*-
"""
analysis.py
Championship standings and race statistics.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

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

        if 'Set Fastest Lap' in df.columns:
            fl = df[df['Set Fastest Lap'].astype(str).str.lower() == 'yes'] \
                .groupby('Driver').size()
            driver_stats['Fastest_Laps'] = fl
        else:
            driver_stats['Fastest_Laps'] = 0
        
        # Calculate wins and podiums
        wins = df[df['Position'] == 1].groupby('Driver').size()
        podiums = df[df['Position'].between(1, 3)].groupby('Driver').size()
        
        driver_stats['Wins'] = wins
        driver_stats['Podium'] = podiums
        driver_stats = driver_stats.fillna(0)
        
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
    """Chronological race order from file layout (first-appearance order).

    The season CSVs are grouped by track in calendar order, so first-appearance
    order of the Track column is a reliable race sequence without needing dates.
    """
    if 'Track' not in df.columns:
        return []
    return list(dict.fromkeys(df['Track'].dropna().astype(str).tolist()))


def calculate_matchup(df: pd.DataFrame, driver1: str, driver2: str):
    """Head-to-head record between two drivers across their shared races.

    Finishing position is used as the race comparison; Starting Grid stands in
    for the qualifying comparison. Returns a dict with per-track ``details``
    (DataFrame) and a ``summary`` dict, or ``None`` when the pair shares no races.
    """
    if df is None or df.empty or 'Driver' not in df.columns or 'Track' not in df.columns:
        return None

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


def calculate_form_trend(df: pd.DataFrame, drivers: list, window: int = 3) -> dict:
    """Rolling-average finishing position ("form") plus per-race positions.

    Returns a dict with ``form`` (driver x race, rolling mean position, lower is
    better) and ``positions`` (driver x race finishing position), both indexed by
    race name in chronological order.
    """
    empty = {'form': pd.DataFrame(), 'positions': pd.DataFrame()}
    if df is None or df.empty or 'Driver' not in df.columns:
        return empty

    order = _race_order(df)
    idx = {t: i for i, t in enumerate(order)}
    series = {}
    positions = {}

    for d in drivers:
        dd = df[df['Driver'] == d].copy()
        dd['_pos'] = pd.to_numeric(dd['Position'], errors='coerce')
        dd['_ord'] = dd['Track'].map(idx)
        dd = dd.dropna(subset=['_pos', '_ord']).sort_values('_ord')
        if dd.empty:
            continue
        keyed = dd.set_index('_ord')
        series[d] = keyed['_pos'].rolling(window, min_periods=1).mean()
        positions[d] = keyed['_pos']

    def relabel(frame: pd.DataFrame) -> pd.DataFrame:
        labels = [order[int(i)] if int(i) < len(order) else int(i) for i in frame.index]
        frame = frame.copy()
        frame.index = labels
        return frame

    form = relabel(pd.DataFrame(series)).round(2)
    return {'form': form, 'positions': relabel(pd.DataFrame(positions))}


def calculate_points_trajectory(df: pd.DataFrame, drivers: list) -> pd.DataFrame:
    """Cumulative championship points per driver across the season.

    Returns a DataFrame indexed by race name (chronological) with one column per
    driver holding their running season total.
    """
    if df is None or df.empty or 'Driver' not in df.columns:
        return pd.DataFrame()

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
