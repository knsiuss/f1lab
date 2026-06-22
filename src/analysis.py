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
from typing import Optional

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
        
        # Aggregate driver statistics
        driver_stats = df.groupby('Driver').agg({
            'Points': ['sum', 'mean', 'count'],
            'Position': ['mean', 'min', 'max'],
            'Finished': 'sum',
            'Set Fastest Lap': lambda x: (x == 'Yes').sum() if 'Set Fastest Lap' in df.columns else 0
        }).round(2)
        
        driver_stats.columns = ['Total_Points', 'Avg_Points', 'Races', 'Avg_Position', 'Best_Position', 'Worst_Position', 'Finishes', 'Fastest_Laps']
        
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

def calculate_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate constructor statistics from race results."""
    logger.info("Calculating team statistics...")
    team_agg = df.groupby('Team').agg({
        'Points': ['sum', 'mean'],
        'Driver': 'nunique',
        'Position': 'mean',
        'Finished': 'sum',
        'Set Fastest Lap': lambda x: (x == 'Yes').sum() if 'Set Fastest Lap' in df.columns else 0,
        'Starting Grid': 'mean'
    }).round(2)
    
    team_agg.columns = ['_'.join(col).strip() for col in team_agg.columns.values]
    
    team_analysis = team_agg.rename(columns={
        'Points_sum': 'Total_Points',
        'Points_mean': 'Avg_Points_Per_Entry',
        'Driver_nunique': 'Drivers',
        'Position_mean': 'Avg_Position',
        'Finished_sum': 'Finishes',
        'Set Fastest Lap_<lambda>': 'Fastest_Laps',
        'Starting Grid_mean': 'Avg_Grid'
    })
    
    team_analysis['Points_Per_Driver'] = (team_analysis['Total_Points'] / team_analysis['Drivers']).round(1)
    
    races_count = df['Track'].nunique()
    team_analysis['Total_Entries'] = team_analysis['Drivers'] * races_count
    team_analysis['Finish_Rate'] = (team_analysis['Finishes'] / team_analysis['Total_Entries'] * 100).round(1)
    
    logger.info(f"Calculated stats for {len(team_analysis)} teams")
    return team_analysis

def calculate_combined_constructor_standings(
    df_race: pd.DataFrame, 
    df_sprint: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Calculate combined constructor standings with Race + Sprint points.
    
    Args:
        df_race: Race results DataFrame.
        df_sprint: Sprint results DataFrame (can be None).
        
    Returns:
        pd.DataFrame: Constructor standings indexed by team name.
    """
    logger.info("Calculating combined constructor standings...")

    # Race-only fallback if no sprint data
    if df_sprint is None or df_sprint.empty:
        race_points = df_race.groupby('Team')['Points'].sum().rename('Race_Points')
        race_wins = df_race[df_race['Position'] == 1].groupby('Team').size().rename('Race_Wins')
        standings = pd.DataFrame({'Race_Points': race_points}).fillna(0)
        standings['Sprint_Points'] = 0
        standings['Total_Points'] = standings['Race_Points']
        standings = standings.join(race_wins, how='left').fillna(0)
        standings['Sprint_Wins'] = 0
        standings = standings.sort_values('Total_Points', ascending=False)
        standings['Position'] = range(1, len(standings) + 1)
        standings = standings[['Position', 'Race_Points', 'Sprint_Points', 'Total_Points', 'Race_Wins', 'Sprint_Wins']]
        logger.info(f"Constructor standings calculated (race-only): {len(standings)} teams")
        return standings

    # Ensure Points column is numeric and fill NaN with 0
    df_sprint = df_sprint.copy()
    df_sprint['Points'] = pd.to_numeric(df_sprint['Points'], errors='coerce').fillna(0)
    
    race_points = df_race.groupby('Team')['Points'].sum().rename('Race_Points')
    sprint_points = df_sprint.groupby('Team')['Points'].sum().rename('Sprint_Points')
    
    standings = pd.DataFrame({'Race_Points': race_points, 'Sprint_Points': sprint_points}).fillna(0)
    standings['Total_Points'] = standings['Race_Points'] + standings['Sprint_Points']
    
    race_wins = df_race[df_race['Position'] == 1].groupby('Team').size().rename('Race_Wins')
    sprint_wins = df_sprint[df_sprint['Position'] == 1].groupby('Team').size().rename('Sprint_Wins')
    
    standings = standings.join(race_wins, how='left').fillna(0)
    standings = standings.join(sprint_wins, how='left').fillna(0)
    
    standings = standings.sort_values('Total_Points', ascending=False)
    standings['Position'] = range(1, len(standings) + 1)
    
    standings = standings[['Position', 'Race_Points', 'Sprint_Points', 'Total_Points', 'Race_Wins', 'Sprint_Wins']]
    
    logger.info(f"Constructor standings calculated: {len(standings)} teams")
    return standings

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
