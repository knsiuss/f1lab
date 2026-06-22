# -*- coding: utf-8 -*-
"""
fastf1_extended.py
Extended FastF1 data extraction utilities.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import fastf1
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Cache setup
CACHE_DIR = Path(__file__).parent.parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
# Note: f1.py may override this with FASTF1_CONFIG.cache_dir if run via dashboard
fastf1.Cache.enable_cache(str(CACHE_DIR))

# SESSION INFORMATION

def get_session_info(session: Any) -> Dict[str, Any]:
    """
    Get comprehensive session information.
    
    Args:
        session: FastF1 Session object
        
    Returns:
        Dict with session details (name, date, time, duration, etc.)
    """
    if session is None:
        logger.warning("No session provided")
        return {}
    
    try:
        event = session.event
        
        # Try multiple ways to get circuit name
        circuit_name = (
            event.get('CircuitShortName') or 
            event.get('OfficialEventName', '').replace(' Grand Prix', '') or
            event.get('EventName', '').replace(' Grand Prix', '') or
            'Unknown'
        )
        
        info = {
            'event_name': event.get('EventName', 'Unknown'),
            'official_name': event.get('OfficialEventName', 'Unknown'),
            'country': event.get('Country', 'Unknown'),
            'location': event.get('Location', 'Unknown'),
            'circuit': circuit_name,
            'round': event.get('RoundNumber', 0),
            'session_name': session.name,
            'session_type': session.session_type if hasattr(session, 'session_type') else 'Unknown',
            'date': str(session.date.date()) if session.date else 'Unknown',
            'year': event.get('EventDate', datetime.now()).year if hasattr(event.get('EventDate', datetime.now()), 'year') else 2025,
        }
        
        # Session start time - format as HH:MM:SS only
        if hasattr(session, 'session_start_time') and session.session_start_time:
            start_time = session.session_start_time
            if hasattr(start_time, 'total_seconds'):
                total_secs = int(start_time.total_seconds())
                hours = total_secs // 3600
                minutes = (total_secs % 3600) // 60
                seconds = total_secs % 60
                info['start_time_utc'] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                info['start_time_utc'] = str(start_time)[:8] if len(str(start_time)) > 8 else str(start_time)
        
        # Session duration from laps
        if hasattr(session, 'laps') and len(session.laps) > 0:
            try:
                total_time = session.laps['Time'].max()
                if pd.notna(total_time):
                    info['duration_seconds'] = total_time.total_seconds()
                    total_secs = int(total_time.total_seconds())
                    hours = total_secs // 3600
                    minutes = (total_secs % 3600) // 60
                    seconds = total_secs % 60
                    info['duration_formatted'] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            except:
                pass
        
        # Total laps (for race)
        if hasattr(session, 'total_laps'):
            info['total_laps'] = session.total_laps
        elif hasattr(session, 'laps') and len(session.laps) > 0:
            info['total_laps'] = int(session.laps['LapNumber'].max())
        
        # Number of drivers
        if hasattr(session, 'drivers'):
            info['num_drivers'] = len(session.drivers)
        
        logger.debug(f"Session info extracted: {info['event_name']} - {info['session_name']}")
        return info
        
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        return {}

def get_session_schedule(session: Any) -> Dict[str, Any]:
    """
    Get session schedule with all session times.
    
    Args:
        session: FastF1 Session object
        
    Returns:
        Dict with schedule for all weekend sessions
    """
    if session is None:
        return {}
    
    try:
        event = session.event
        schedule = {}
        
        session_keys = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
        session_names = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
        
        for key in session_keys:
            date_key = f'{key}Date'
            if date_key in event.index and pd.notna(event[date_key]):
                schedule[key] = {
                    'name': event.get(key, key),
                    'date': str(event[date_key]),
                }
        
        return schedule
        
    except Exception as e:
        logger.error(f"Error getting session schedule: {e}")
        return {}

# WEATHER DATA

def get_weather_data(session: Any) -> pd.DataFrame:
    """
    Get detailed weather data for the session.
    
    Args:
        session: FastF1 Session object
        
    Returns:
        DataFrame with weather data
    """
    if session is None:
        logger.warning("No session provided for weather data")
        return pd.DataFrame()
    
    try:
        weather = session.weather_data.copy()
        logger.debug(f"Weather data: {len(weather)} records")
        return weather
    except Exception as e:
        logger.error(f"Error getting weather data: {e}")
        return pd.DataFrame()

def get_weather_summary(session: Any) -> Dict[str, Any]:
    """
    Get weather summary with average conditions.
    
    Args:
        session: FastF1 Session object
        
    Returns:
        Dict with weather summary
    """
    if session is None:
        return {}
    
    try:
        weather = session.weather_data
        
        if weather is None or len(weather) == 0:
            return {'available': False}
        
        summary = {
            'available': True,
            'air_temp_avg': round(weather['AirTemp'].mean(), 1) if 'AirTemp' in weather.columns else None,
            'air_temp_min': round(weather['AirTemp'].min(), 1) if 'AirTemp' in weather.columns else None,
            'air_temp_max': round(weather['AirTemp'].max(), 1) if 'AirTemp' in weather.columns else None,
            'track_temp_avg': round(weather['TrackTemp'].mean(), 1) if 'TrackTemp' in weather.columns else None,
            'track_temp_min': round(weather['TrackTemp'].min(), 1) if 'TrackTemp' in weather.columns else None,
            'track_temp_max': round(weather['TrackTemp'].max(), 1) if 'TrackTemp' in weather.columns else None,
            'humidity_avg': round(weather['Humidity'].mean(), 1) if 'Humidity' in weather.columns else None,
            'pressure_avg': round(weather['Pressure'].mean(), 1) if 'Pressure' in weather.columns else None,
            'wind_speed_avg': round(weather['WindSpeed'].mean(), 1) if 'WindSpeed' in weather.columns else None,
            'wind_direction_avg': round(weather['WindDirection'].mean(), 0) if 'WindDirection' in weather.columns else None,
            'rainfall': weather['Rainfall'].any() if 'Rainfall' in weather.columns else False,
        }
        
        # Determine conditions
        if summary['rainfall']:
            summary['conditions'] = 'Wet'
        elif summary['track_temp_avg'] and summary['track_temp_avg'] > 45:
            summary['conditions'] = 'Hot'
        elif summary['track_temp_avg'] and summary['track_temp_avg'] < 25:
            summary['conditions'] = 'Cool'
        else:
            summary['conditions'] = 'Dry'
        
        logger.debug(f"Weather summary: {summary['conditions']}")
        return summary
        
    except Exception as e:
        logger.error(f"Error getting weather summary: {e}")
        return {'available': False}

# TYRE STRATEGY

def get_tyre_stints(session: Any, driver: Optional[str] = None) -> pd.DataFrame:
    """
    Get tyre stint data for all or specific driver.
    
    Args:
        session: FastF1 Session object
        driver: Optional driver code (e.g., 'VER', 'NOR')
        
    Returns:
        DataFrame with tyre stint information
    """
    if session is None:
        return pd.DataFrame()
    
    try:
        laps = session.laps.copy()
        
        if driver:
            laps = laps[laps['Driver'] == driver]
        
        if len(laps) == 0:
            return pd.DataFrame()
        
        # Group by driver and stint
        stints = []
        for drv in laps['Driver'].unique():
            driver_laps = laps[laps['Driver'] == drv].sort_values('LapNumber')
            
            stint_num = 0
            prev_compound = None
            stint_start = 1
            
            for _, lap in driver_laps.iterrows():
                compound = lap.get('Compound', 'UNKNOWN')
                lap_num = lap['LapNumber']
                
                if compound != prev_compound and prev_compound is not None:
                    # End previous stint
                    stints.append({
                        'Driver': drv,
                        'Stint': stint_num,
                        'Compound': prev_compound,
                        'StartLap': stint_start,
                        'EndLap': lap_num - 1,
                        'Laps': lap_num - stint_start,
                    })
                    stint_num += 1
                    stint_start = lap_num
                
                prev_compound = compound
            
            # Add final stint
            if prev_compound:
                stints.append({
                    'Driver': drv,
                    'Stint': stint_num,
                    'Compound': prev_compound,
                    'StartLap': stint_start,
                    'EndLap': int(driver_laps['LapNumber'].max()),
                    'Laps': int(driver_laps['LapNumber'].max()) - stint_start + 1,
                })
        
        return pd.DataFrame(stints)
        
    except Exception as e:
        logger.error(f"Error getting tyre stints: {e}")
        return pd.DataFrame()

def get_tyre_degradation(session: Any, driver: str) -> Dict[str, Any]:
    """
    Calculate tyre degradation for a driver.
    
    Args:
        session: FastF1 Session object
        driver: Driver code
        
    Returns:
        Dict with degradation data per stint
    """
    if session is None:
        return {}
    
    try:
        laps = session.laps.pick_driver(driver).copy()
        laps = laps[laps['IsAccurate'] == True] if 'IsAccurate' in laps.columns else laps
        
        degradation = {}
        stints = get_tyre_stints(session, driver)
        
        for _, stint in stints.iterrows():
            stint_laps = laps[
                (laps['LapNumber'] >= stint['StartLap']) & 
                (laps['LapNumber'] <= stint['EndLap'])
            ]
            
            if len(stint_laps) > 2:
                lap_times = stint_laps['LapTime'].dt.total_seconds().values
                lap_nums = stint_laps['LapNumber'].values
                
                # Calculate degradation (time lost per lap)
                if len(lap_times) > 2:
                    # Use polyfit for trend
                    coeffs = np.polyfit(lap_nums, lap_times, 1)
                    deg_per_lap = coeffs[0]
                    
                    degradation[f"Stint {stint['Stint'] + 1}"] = {
                        'compound': stint['Compound'],
                        'laps': stint['Laps'],
                        'deg_per_lap_sec': round(deg_per_lap, 3),
                        'total_deg_sec': round(deg_per_lap * stint['Laps'], 2),
                    }
        
        return degradation
        
    except Exception as e:
        logger.error(f"Error calculating tyre degradation: {e}")
        return {}

# PIT STOPS

def get_pit_stops(session: Any) -> pd.DataFrame:
    """
    Get all pit stop data from the session.
    
    Args:
        session: FastF1 Session object
        
    Returns:
        DataFrame with pit stop information
    """
    if session is None:
        return pd.DataFrame()
    
    try:
        laps = session.laps.copy()
        
        # Find pit laps using PitInTime and PitOutTime
        pit_data = []
        
        for driver in laps['Driver'].unique():
            driver_laps = laps[laps['Driver'] == driver].sort_values('LapNumber')
            stop_num = 0
            
            for idx, lap in driver_laps.iterrows():
                pit_in = lap.get('PitInTime')
                pit_out = lap.get('PitOutTime')
                
                # Check if this is a pit lap
                if pd.notna(pit_out):
                    stop_num += 1
                    
                    # Calculate pit time (stationary time in pits)
                    pit_time = None
                    if pd.notna(pit_in) and pd.notna(pit_out):
                        if hasattr(pit_out, 'total_seconds') and hasattr(pit_in, 'total_seconds'):
                            pit_time = pit_out.total_seconds() - pit_in.total_seconds()
                        else:
                            pit_time = float(pit_out) - float(pit_in)
                        
                        # Pit time should be positive and reasonable (2-60 seconds)
                        if pit_time < 0 or pit_time > 120:
                            pit_time = None
                    
                    # Get pit duration from lap time difference if needed
                    if pit_time is None or pit_time < 2:
                        # Use a reasonable default or estimate
                        pit_duration = lap.get('PitOutTime')
                        if pd.notna(pit_duration) and hasattr(pit_duration, 'total_seconds'):
                            # Estimate based on position in pit lane
                            pit_time = 22.0  # Average pit stop time
                    
                    pit_data.append({
                        'Driver': driver,
                        'Stop': stop_num,
                        'Lap': int(lap['LapNumber']),
                        'PitTime': round(pit_time, 1) if pit_time and pit_time > 0 else 22.0,
                        'Compound': lap.get('Compound', 'UNKNOWN'),
                    })
        
        if len(pit_data) == 0:
            # Alternative: use stint changes to detect pit stops
            stints = get_tyre_stints(session)
            if not stints.empty:
                for driver in stints['Driver'].unique():
                    driver_stints = stints[stints['Driver'] == driver].sort_values('Stint')
                    for i in range(1, len(driver_stints)):
                        pit_data.append({
                            'Driver': driver,
                            'Stop': i,
                            'Lap': int(driver_stints.iloc[i]['StartLap']),
                            'PitTime': 22.0,  # Default estimate
                            'Compound': driver_stints.iloc[i]['Compound'],
                        })
        
        return pd.DataFrame(pit_data)
        
    except Exception as e:
        logger.error(f"Error getting pit stops: {e}")
        return pd.DataFrame()

# SECTOR TIMES

def get_sector_times(session: Any, driver: Optional[str] = None) -> pd.DataFrame:
    """
    Get sector times for all laps.
    
    Args:
        session: FastF1 Session object
        driver: Optional driver code
        
    Returns:
        DataFrame with sector times
    """
    if session is None:
        return pd.DataFrame()
    
    try:
        laps = session.laps.copy()
        
        if driver:
            laps = laps[laps['Driver'] == driver]
        
        if len(laps) == 0:
            return pd.DataFrame()
        
        # Build sector data
        sector_data = []
        
        for _, lap in laps.iterrows():
            row = {
                'Driver': lap['Driver'],
                'Lap': int(lap['LapNumber']),
            }
            
            # Get sector times and convert to seconds
            for i, sector_col in enumerate(['Sector1Time', 'Sector2Time', 'Sector3Time'], 1):
                if sector_col in lap.index and pd.notna(lap[sector_col]):
                    time_val = lap[sector_col]
                    if hasattr(time_val, 'total_seconds'):
                        row[f'Sector{i}'] = round(time_val.total_seconds(), 3)
                    else:
                        row[f'Sector{i}'] = round(float(time_val), 3)
                else:
                    row[f'Sector{i}'] = None
            
            # Get lap time
            if 'LapTime' in lap.index and pd.notna(lap['LapTime']):
                lap_time = lap['LapTime']
                if hasattr(lap_time, 'total_seconds'):
                    row['LapTime'] = round(lap_time.total_seconds(), 3)
                else:
                    row['LapTime'] = round(float(lap_time), 3)
            
            # Get compound
            row['Compound'] = lap.get('Compound', 'UNKNOWN')
            row['TyreLife'] = lap.get('TyreLife', 0)
            
            sector_data.append(row)
        
        df = pd.DataFrame(sector_data)
        
        # Filter out rows with no sector data
        if 'Sector1' in df.columns:
            df = df.dropna(subset=['Sector1', 'Sector2', 'Sector3'], how='all')
        
        return df
        
    except Exception as e:
        logger.error(f"Error getting sector times: {e}")
        return pd.DataFrame()

def get_best_sectors(session: Any) -> pd.DataFrame:
    """
    Get best sector times for each driver.
    
    Args:
        session: FastF1 Session object
        
    Returns:
        DataFrame with best sectors per driver
    """
    if session is None:
        return pd.DataFrame()
    
    try:
        laps = session.laps.copy()
        
        best_sectors = []
        for driver in laps['Driver'].unique():
            driver_laps = laps[laps['Driver'] == driver]
            
            best = {
                'Driver': driver,
                'Team': driver_laps['Team'].iloc[0] if 'Team' in driver_laps.columns else 'Unknown',
            }
            
            for sector in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
                if sector in driver_laps.columns:
                    min_time = driver_laps[sector].min()
                    if pd.notna(min_time):
                        best[f'Best_{sector[:-4]}'] = min_time.total_seconds()
            
            if 'LapTime' in driver_laps.columns:
                fastest = driver_laps['LapTime'].min()
                if pd.notna(fastest):
                    best['FastestLap'] = fastest.total_seconds()
            
            best_sectors.append(best)
        
        df = pd.DataFrame(best_sectors)
        
        # Calculate theoretical best
        if all(col in df.columns for col in ['Best_Sector1', 'Best_Sector2', 'Best_Sector3']):
            df['TheoreticalBest'] = df['Best_Sector1'] + df['Best_Sector2'] + df['Best_Sector3']
        
        return df.sort_values('FastestLap') if 'FastestLap' in df.columns else df
        
    except Exception as e:
        logger.error(f"Error getting best sectors: {e}")
        return pd.DataFrame()

# SPEED TRAPS

def get_speed_data(session: Any) -> pd.DataFrame:
    """
    Get speed trap data from session.
    
    Args:
        session: FastF1 Session object
        
    Returns:
        DataFrame with speed data
    """
    if session is None:
        return pd.DataFrame()
    
    try:
        laps = session.laps.copy()
        
        speed_cols = ['Driver', 'LapNumber', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']
        available_cols = [c for c in speed_cols if c in laps.columns]
        
        if len(available_cols) <= 2:
            logger.warning("Limited speed data available")
            return pd.DataFrame()
        
        return laps[available_cols]
        
    except Exception as e:
        logger.error(f"Error getting speed data: {e}")
        return pd.DataFrame()

def get_top_speeds(session: Any) -> pd.DataFrame:
    """
    Get top speeds for each driver.
    
    Args:
        session: FastF1 Session object
        
    Returns:
        DataFrame with max speeds per driver
    """
    if session is None:
        return pd.DataFrame()
    
    try:
        laps = session.laps.copy()
        
        top_speeds = []
        for driver in laps['Driver'].unique():
            driver_laps = laps[laps['Driver'] == driver]
            
            speeds = {'Driver': driver}
            
            for col in ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']:
                if col in driver_laps.columns:
                    max_speed = driver_laps[col].max()
                    if pd.notna(max_speed):
                        speeds[f'Max_{col}'] = max_speed
            
            top_speeds.append(speeds)
        
        df = pd.DataFrame(top_speeds)
        
        # Sort by speed trap speed if available
        if 'Max_SpeedST' in df.columns:
            df = df.sort_values('Max_SpeedST', ascending=False)
        
        return df
        
    except Exception as e:
        logger.error(f"Error getting top speeds: {e}")
        return pd.DataFrame()

# TRACK STATUS

def get_track_status(session: Any) -> pd.DataFrame:
    """
    Get track status changes (flags, SC, VSC).
    
    Args:
        session: FastF1 Session object
        
    Returns:
        DataFrame with track status events
    """
    if session is None:
        return pd.DataFrame()
    
    try:
        # Try to get race control messages
        if hasattr(session, 'race_control_messages'):
            rcm = session.race_control_messages
            if rcm is not None and len(rcm) > 0:
                return rcm
        
        # Alternative: from session status
        if hasattr(session, 'session_status'):
            status = session.session_status
            if status is not None and len(status) > 0:
                return status
        
        logger.warning("No track status data available")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error getting track status: {e}")
        return pd.DataFrame()

def get_flag_events(session: Any) -> List[Dict]:
    """
    Get flag events (yellow, red, SC, VSC).
    
    Args:
        session: FastF1 Session object
        
    Returns:
        List of flag events
    """
    if session is None:
        return []
    
    try:
        events = []
        rcm = get_track_status(session)
        
        if len(rcm) == 0:
            return []
        
        flag_keywords = ['YELLOW', 'RED', 'SAFETY CAR', 'VSC', 'GREEN', 'CHEQUERED']
        
        for _, row in rcm.iterrows():
            message = str(row.get('Message', '')).upper()
            category = str(row.get('Category', '')).upper()
            
            for keyword in flag_keywords:
                if keyword in message or keyword in category:
                    events.append({
                        'time': str(row.get('Time', '')),
                        'lap': row.get('Lap', None),
                        'flag': keyword,
                        'message': row.get('Message', ''),
                    })
                    break
        
        return events
        
    except Exception as e:
        logger.error(f"Error getting flag events: {e}")
        return []

# CAR DATA (Advanced Telemetry)

def get_car_data(session: Any, driver: str, lap: Optional[int] = None) -> pd.DataFrame:
    """
    Get detailed car data (telemetry).
    
    Args:
        session: FastF1 Session object
        driver: Driver code
        lap: Optional specific lap number (default: fastest lap)
        
    Returns:
        DataFrame with car telemetry
    """
    if session is None:
        return pd.DataFrame()
    
    try:
        driver_laps = session.laps.pick_driver(driver)
        
        if lap:
            target_lap = driver_laps[driver_laps['LapNumber'] == lap]
            if len(target_lap) == 0:
                logger.warning(f"Lap {lap} not found for {driver}")
                return pd.DataFrame()
            target_lap = target_lap.iloc[0]
        else:
            target_lap = driver_laps.pick_fastest()
        
        # Get car data (more detailed than telemetry)
        if hasattr(target_lap, 'get_car_data'):
            car_data = target_lap.get_car_data()
            if car_data is not None and len(car_data) > 0:
                return car_data.add_distance() if hasattr(car_data, 'add_distance') else car_data
        
        # Fallback to telemetry
        telemetry = target_lap.get_telemetry()
        if telemetry is not None:
            return telemetry.add_distance() if hasattr(telemetry, 'add_distance') else telemetry
        
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error getting car data for {driver}: {e}")
        return pd.DataFrame()

def get_telemetry_comparison(
    session: Any, 
    driver1: str, 
    driver2: str,
    lap: Optional[int] = None
) -> Dict[str, pd.DataFrame]:
    """
    Get telemetry comparison between two drivers.
    
    Args:
        session: FastF1 Session object
        driver1: First driver code
        driver2: Second driver code
        lap: Optional specific lap
        
    Returns:
        Dict with telemetry DataFrames for both drivers
    """
    if session is None:
        return {}
    
    try:
        tel1 = get_car_data(session, driver1, lap)
        tel2 = get_car_data(session, driver2, lap)
        
        return {
            driver1: tel1,
            driver2: tel2,
        }
        
    except Exception as e:
        logger.error(f"Error comparing telemetry: {e}")
        return {}

# POSITION DATA

def get_position_data(session: Any) -> pd.DataFrame:
    """
    Get position data per lap for all drivers.
    
    Args:
        session: FastF1 Session object
        
    Returns:
        DataFrame with position per lap
    """
    if session is None:
        return pd.DataFrame()
    
    try:
        laps = session.laps.copy()
        
        position_data = laps[['Driver', 'LapNumber', 'Position', 'Time', 'LapTime']].copy()
        position_data = position_data.dropna(subset=['Position'])
        
        return position_data.sort_values(['LapNumber', 'Position'])
        
    except Exception as e:
        logger.error(f"Error getting position data: {e}")
        return pd.DataFrame()

def get_position_changes(session: Any) -> pd.DataFrame:
    """
    Calculate position changes throughout the race.
    
    Args:
        session: FastF1 Session object
        
    Returns:
        DataFrame with position changes per driver
    """
    if session is None:
        return pd.DataFrame()
    
    try:
        laps = session.laps.copy()
        results = session.results.copy() if hasattr(session, 'results') else None
        
        changes = []
        for driver in laps['Driver'].unique():
            driver_laps = laps[laps['Driver'] == driver].sort_values('LapNumber')
            
            if len(driver_laps) == 0:
                continue
            
            # Get grid position
            grid_pos = None
            if results is not None and 'Abbreviation' in results.columns:
                driver_result = results[results['Abbreviation'] == driver]
                if len(driver_result) > 0:
                    grid_pos = driver_result['GridPosition'].iloc[0]
            
            first_pos = driver_laps['Position'].iloc[0] if len(driver_laps) > 0 else None
            last_pos = driver_laps['Position'].iloc[-1] if len(driver_laps) > 0 else None
            
            if grid_pos and last_pos:
                changes.append({
                    'Driver': driver,
                    'GridPosition': int(grid_pos) if pd.notna(grid_pos) else None,
                    'FirstLapPosition': int(first_pos) if pd.notna(first_pos) else None,
                    'FinalPosition': int(last_pos) if pd.notna(last_pos) else None,
                    'PositionsGained': int(grid_pos - last_pos) if pd.notna(grid_pos) and pd.notna(last_pos) else 0,
                    'FirstLapGain': int(grid_pos - first_pos) if pd.notna(grid_pos) and pd.notna(first_pos) else 0,
                })
        
        return pd.DataFrame(changes).sort_values('FinalPosition')
        
    except Exception as e:
        logger.error(f"Error getting position changes: {e}")
        return pd.DataFrame()

def get_gaps_to_leader(session: Any) -> pd.DataFrame:
    """
    Calculate gaps to leader per lap.
    
    Args:
        session: FastF1 Session object
        
    Returns:
        DataFrame with gap to leader per driver per lap
    """
    if session is None:
        return pd.DataFrame()
    
    try:
        laps = session.laps.copy()
        
        gaps = []
        for lap_num in laps['LapNumber'].unique():
            lap_data = laps[laps['LapNumber'] == lap_num].sort_values('Position')
            
            if len(lap_data) == 0:
                continue
            
            leader = lap_data.iloc[0]
            leader_time = leader['Time']
            
            for _, row in lap_data.iterrows():
                gap = (row['Time'] - leader_time).total_seconds() if pd.notna(row['Time']) and pd.notna(leader_time) else None
                
                gaps.append({
                    'Lap': int(lap_num),
                    'Driver': row['Driver'],
                    'Position': int(row['Position']) if pd.notna(row['Position']) else None,
                    'GapToLeader': gap,
                })
        
        return pd.DataFrame(gaps)
        
    except Exception as e:
        logger.error(f"Error calculating gaps: {e}")
        return pd.DataFrame()

# RACE RESULTS

def get_race_results(session: Any) -> pd.DataFrame:
    """
    Get final race results with full details.
    
    Args:
        session: FastF1 Session object
        
    Returns:
        DataFrame with race results
    """
    if session is None:
        return pd.DataFrame()
    
    try:
        results = session.results.copy()
        
        # Add additional columns if available
        columns_of_interest = [
            'Position', 'Abbreviation', 'DriverNumber', 'FullName', 
            'TeamName', 'GridPosition', 'Status', 'Points', 'Time',
            'FastestLap', 'FastestLapTime'
        ]
        
        available = [c for c in columns_of_interest if c in results.columns]
        return results[available]
        
    except Exception as e:
        logger.error(f"Error getting race results: {e}")
        return pd.DataFrame()

def get_circuit_layout_info(session: Any) -> Dict[str, Any]:
    """
    Get detailed circuit layout information including corners and rotation.
    
    Args:
        session: FastF1 Session object
        
    Returns:
        Dict with rotation and corners data
    """
    if session is None:
        return {}
    
    try:
        info = session.get_circuit_info()
        
        corners = []
        if info is not None and info.corners is not None:
            corners = info.corners.to_dict('records')
            
        marshal_lights = []
        if info is not None and info.marshal_lights is not None:
            marshal_lights = info.marshal_lights.to_dict('records')
            
        return {
            'rotation': info.rotation if info else 0,
            'corners': corners,
            'marshal_lights': marshal_lights
        }
    except Exception as e:
        logger.error(f"Error getting circuit info: {e}")
        return {}

# UTILITY FUNCTIONS

def format_lap_time(seconds: float) -> str:
    """Format seconds to lap time string (M:SS.mmm)."""
    if pd.isna(seconds):
        return "-"

    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes}:{remaining:06.3f}"

def format_f1_time(td: pd.Timedelta) -> str:
    """Formats a pandas Timedelta into an F1-style time string (M:SS.mmm or H:MM:SS.mmm)."""
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

def get_compound_color(compound: str) -> str:
    """Get color for tyre compound."""
    colors = {
        'SOFT': '#FF3333',
        'MEDIUM': '#FFF200',
        'HARD': '#EBEBEB',
        'INTERMEDIATE': '#43B02A',
        'WET': '#0067AD',
        'UNKNOWN': '#808080',
    }
    return colors.get(compound.upper(), '#808080')

def get_driver_team(session: Any, driver: str) -> str:
    """Get team name for a driver."""
    if session is None:
        return "Unknown"
    
    try:
        results = session.results
        if results is not None and 'Abbreviation' in results.columns:
            driver_result = results[results['Abbreviation'] == driver]
            if len(driver_result) > 0:
                return driver_result['TeamName'].iloc[0]
        
        laps = session.laps
        if laps is not None:
            driver_laps = laps[laps['Driver'] == driver]
            if len(driver_laps) > 0 and 'Team' in driver_laps.columns:
                return driver_laps['Team'].iloc[0]
        
        return "Unknown"
        
    except:
        return "Unknown"

def export_session_to_csv(session: Any) -> Dict[str, str]:
    """
    Export session data to CSV strings.
    
    Args:
        session: FastF1 Session object
        
    Returns:
        Dict with CSV strings for 'laps', 'results', 'weather'
    """
    if session is None:
        return {}
    
    exports = {}
    
    try:
        # Laps
        if session.laps is not None:
            exports['laps'] = session.laps.to_csv(index=False)
            
        # Results
        if session.results is not None:
            exports['results'] = session.results.to_csv(index=False)
            
        # Weather
        if hasattr(session, 'weather_data') and session.weather_data is not None:
            exports['weather'] = session.weather_data.to_csv(index=False)
            
        return exports
    except Exception as e:
        logger.error(f"Error exporting session data: {e}")
        return {}

# RACE CONTROL & EVENTS

def get_race_control_messages(session: Any) -> pd.DataFrame:
    """
    Get race control messages (flags, penalties, SC).
    
    Args:
        session: FastF1 Session object
        
    Returns:
        DataFrame with race control messages
    """
    try:
        if not hasattr(session, 'race_control_messages'):
            return pd.DataFrame()
            
        rcm = session.race_control_messages
        if rcm is None or rcm.empty:
            return pd.DataFrame()
            
        # Select and rename columns for display
        display_cols = ['Time', 'Category', 'Message', 'Flag', 'Scope', 'Sector', 'Lap']
        valid_cols = [c for c in display_cols if c in rcm.columns]
        
        df = rcm[valid_cols].copy()
        
        # Format time
        if 'Time' in df.columns:
            df['Time'] = df['Time'].apply(lambda x: format_f1_time(x) if pd.notna(x) else "")
            
        return df
        
    except Exception as e:
        logger.error(f"Error getting race control messages: {e}")
        return pd.DataFrame()

def get_detailed_pit_analysis(session: Any) -> pd.DataFrame:
    """
    Get pit stops with detailed context (SC, Flags).
    
    Args:
        session: FastF1 Session object
        
    Returns:
        DataFrame with detailed pit stop info
    """
    try:
        # Check if pit stops data is available directly
        if hasattr(session, 'pit_stops') and session.pit_stops is not None and not session.pit_stops.empty:
            pit_data = session.pit_stops.copy()
            # Rename columns if needed to match expected output
            if 'Lap' in pit_data.columns:
                pit_data = pit_data.rename(columns={'Lap': 'LapNumber'})
                
        else:
            # Fallback to laps
            laps = session.laps
            pit_stops = laps.pick_pit_stops() if laps is not None else None
            
            if pit_stops is None or pit_stops.empty:
                return pd.DataFrame()
            
            # Select available columns
            wanted_cols = ['Driver', 'LapNumber', 'LapTime', 'PitInTime', 'PitOutTime', 'Duration']
            avail_cols = [c for c in wanted_cols if c in pit_stops.columns]
            
            pit_data = pit_stops[avail_cols].copy()
            
            # Calculate duration if missing
            if 'Duration' not in pit_data.columns and 'PitInTime' in pit_data.columns and 'PitOutTime' in pit_data.columns:
                pit_data['Duration'] = (pit_data['PitOutTime'] - pit_data['PitInTime']).dt.total_seconds()
        
        # Helper to safely get team
        def get_team_safe(drv):
            try:
                if hasattr(session, 'get_driver'):
                    return session.get_driver(drv)['TeamName']
                return "Unknown"
            except:
                return "Unknown"

        # Add Team
        if 'Team' not in pit_data.columns:
            pit_data['Team'] = pit_data['Driver'].apply(get_team_safe)
            
        # Ensure Duration is numeric
        if 'Duration' in pit_data.columns:
             pit_data['Duration'] = pd.to_numeric(pit_data['Duration'], errors='coerce')
        
        # Add Track Status context if available
        # This requires checking track status at PitInTime
        if hasattr(session, 'track_status'):
            # This is complex to map exactly, simplified for now:
            # Check if any SC/VSC was active during the lap
            # 4=SC, 6=VSC, 7=VSC ending
            pass # TODO: Implement precise track status mapping
            
        return pit_data
        
    except Exception as e:
        logger.error(f"Error getting detailed pit analysis: {e}")
        return pd.DataFrame()
