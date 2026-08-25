# -*- coding: utf-8 -*-
"""
fastf1_extended.py
Extended FastF1 data extraction utilities.

Deliberately small: it now contains only the extractors the dashboard
actually consumes (tyre stints, pit stops, race control messages) plus the
shared cache bootstrap. Everything else was dead weight and has been removed;
history keeps the rest if ever needed.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import fastf1
import pandas as pd
import logging
from typing import Any, Optional
from config import FASTF1_CONFIG, MODEL_CONFIG, PIT_STOP_CONFIG

logger = logging.getLogger(__name__)


def _ensure_cache() -> None:
    """Enable the FastF1 disk cache once, at the configured location.

    Single source of truth is FASTF1_CONFIG.cache_dir (f1_cache/), matching
    shared.setup_fastf1_cache -- importing this module no longer forks a second
    cache under cache/.
    """
    try:
        FASTF1_CONFIG.cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(FASTF1_CONFIG.cache_dir))
    except Exception as e:  # cache is an optimisation; never block imports
        logger.warning(f"Could not enable FastF1 cache: {e}")


_ensure_cache()

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

# PIT STOPS

def _td_seconds(value: Any) -> Optional[float]:
    """Coerce a Timedelta/timestamp/numeric to seconds, or None."""
    try:
        if value is None or pd.isna(value):
            return None
        if hasattr(value, 'total_seconds'):
            return float(value.total_seconds())
        return float(value)
    except (TypeError, ValueError):
        return None


def get_pit_stops(session: Any) -> pd.DataFrame:
    """
    Get all pit stop data from the session.

    A stop is detected on the pit-entry lap (non-null ``PitInTime``). The
    stop duration is measured across the pit-lane crossing boundary --
    ``PitOutTime`` of the following lap minus ``PitInTime`` of the stop lap --
    which is the only pairing that yields a positive, physical duration.
    When that measurement is unavailable or implausible the configured model
    default is used instead and ``Measured`` is False, so estimates are never
    presented as telemetry.

    Args:
        session: FastF1 Session object

    Returns:
        DataFrame with Driver, Stop, Lap, PitTime (s), Compound (fitted on
        the out lap when known) and Measured (bool).
    """
    if session is None:
        return pd.DataFrame()

    try:
        laps = session.laps
        if laps is None or laps.empty:
            return pd.DataFrame()
        laps = laps.copy()
        if not {'Driver', 'LapNumber'}.issubset(laps.columns):
            return pd.DataFrame()

        default_pit = float(MODEL_CONFIG['pit_loss_sec'])
        has_pit_in = 'PitInTime' in laps.columns
        if not has_pit_in:
            # No telemetry pit markers at all -> fall back to stint changes.
            return _pit_stops_from_stints(session, default_pit)

        pit_data = []
        for driver, group in laps.groupby('Driver', dropna=False):
            dl = group.sort_values('LapNumber').reset_index(drop=True)
            pin_s = dl['PitInTime'].map(_td_seconds)
            pout_s = (dl['PitOutTime'].map(_td_seconds)
                      if 'PitOutTime' in dl.columns else pd.Series([None] * len(dl)))
            compounds = (dl['Compound'].tolist()
                         if 'Compound' in dl.columns else ['UNKNOWN'] * len(dl))

            stop_num = 0
            for i in range(len(dl)):
                pin = pin_s.iloc[i]
                # NOTE: Series.map turns returned None into NaN, so test with
                # pd.isna rather than identity against None.
                if pin is None or pd.isna(pin):
                    continue  # not a pit-entry lap
                stop_num += 1

                pit_time, measured = None, False
                pout_next = pout_s.iloc[i + 1] if i + 1 < len(dl) else None
                if pout_next is not None and not pd.isna(pout_next):
                    cand = pout_next - pin
                    if (PIT_STOP_CONFIG['min_measured_sec'] <= cand
                            <= PIT_STOP_CONFIG['max_measured_sec']):
                        pit_time, measured = cand, True
                if pit_time is None:
                    pit_time = default_pit

                # Compound fitted for the next stint lives on the out lap.
                compound = compounds[i + 1] if i + 1 < len(dl) else compounds[i]

                pit_data.append({
                    'Driver': driver,
                    'Stop': stop_num,
                    'Lap': int(dl['LapNumber'].iloc[i]),
                    'PitTime': round(pit_time, 1),
                    'Compound': compound if pd.notna(compound) else 'UNKNOWN',
                    'Measured': measured,
                })

        if not pit_data:
            return _pit_stops_from_stints(session, default_pit)

        return pd.DataFrame(pit_data)

    except Exception as e:
        logger.error(f"Error getting pit stops: {e}")
        return pd.DataFrame()


def _pit_stops_from_stints(session: Any, default_pit: float) -> pd.DataFrame:
    """Fallback: infer stops from tyre-stint changes; all times estimated."""
    try:
        stints = get_tyre_stints(session)
        if stints.empty:
            return pd.DataFrame()
        pit_data = []
        for driver in stints['Driver'].unique():
            driver_stints = stints[stints['Driver'] == driver].sort_values('Stint')
            for i in range(1, len(driver_stints)):
                pit_data.append({
                    'Driver': driver,
                    'Stop': i,
                    'Lap': int(driver_stints.iloc[i]['StartLap']),
                    'PitTime': round(default_pit, 1),
                    'Compound': driver_stints.iloc[i]['Compound'],
                    'Measured': False,
                })
        return pd.DataFrame(pit_data)
    except Exception as e:
        logger.error(f"Error inferring pit stops from stints: {e}")
        return pd.DataFrame()

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


def format_f1_time(td) -> str:
    """Format a Timedelta/seconds value as H:MM:SS or M:SS."""
    try:
        total_seconds = float(td.total_seconds()) if hasattr(td, 'total_seconds') else float(td)
    except (TypeError, ValueError, AttributeError):
        return ""
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:06.3f}"
    return f"{minutes}:{seconds:06.3f}"
