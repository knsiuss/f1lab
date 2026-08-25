# -*- coding: utf-8 -*-
"""
loader.py
CSV data loading and preprocessing.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import logging
import pandas as pd
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load race data from CSV. Returns None on failure."""
    try:
        logger.info(f"Loading data from {file_path}")
        
        # Try different encodings if UTF-8 fails
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 encoding failed, trying latin-1 for {file_path}")
            df = pd.read_csv(file_path, encoding='latin-1')
        
        # Validate loaded data
        if df.empty:
            logger.warning(f"Loaded empty DataFrame from {file_path}")
            return None
            
        logger.info(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns from {file_path}")
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        logger.error("Please ensure the file exists at the specified path")
        return None
    except PermissionError:
        logger.error(f"Permission denied accessing file: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"Empty data file: {file_path}")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error in {file_path}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error loading data from {file_path}: {e}")
        return None

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column types and create derived fields."""
    logger.info("Cleaning data...")
    
    try:
        # Validate input
        if df is None or df.empty:
            logger.error("Cannot clean None or empty DataFrame")
            raise ValueError("DataFrame is None or empty")
        
        # Check for required columns
        required_columns = ['Points', 'Position', 'Starting Grid', 'Laps', 'Time/Retired']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            logger.warning(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Convert numeric columns with error handling
        df['Points'] = pd.to_numeric(df['Points'], errors='coerce').fillna(0)
        df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
        df['Starting Grid'] = pd.to_numeric(df['Starting Grid'], errors='coerce')
        df['Laps'] = pd.to_numeric(df['Laps'], errors='coerce')
        
        # Create finished flag - must have valid position and not DNS/DSQ/DNF
        try:
            df['Finished'] = (
                df['Position'].notna() & 
                ~df['Time/Retired'].str.contains('DNS|DSQ|DNF', na=False, regex=True)
            )
        except Exception as e:
            logger.warning(f"Error creating Finished flag: {e}. Using position-only check.")
            df['Finished'] = df['Position'].notna()
        
        # Log cleaning summary
        finishers = df['Finished'].sum()
        logger.info(f"Data cleaned: {len(df)} rows, {finishers} finishers ({finishers/len(df)*100:.1f}%)")
        logger.debug(f"Points range: {df['Points'].min()}-{df['Points'].max()}")
        logger.debug(f"Position range: {df['Position'].min()}-{df['Position'].max()}")
        
        return df
        
    except Exception as e:
        logger.exception(f"Error cleaning data: {e}")
        raise

def load_season_data(data_dir: str, year: int) -> Optional[pd.DataFrame]:
    """
    Load, tag and combine Race and Sprint results for a season (canonical).

    This is the single implementation of season CSV loading;
    ``shared.load_race_data`` wraps it with Streamlit caching. Raises the same
    ``ValueError`` as ``clean_data`` when the schema is invalid.

    Args:
        data_dir: Directory containing the CSV files.
        year: Season year used in the CSV filenames.

    Returns:
        pd.DataFrame: Cleaned combined DataFrame (race + sprint when present),
        or None when the race CSV is missing/empty.
    """
    race_path = Path(data_dir) / f'Formula1_{year}Season_RaceResults.csv'
    sprint_path = Path(data_dir) / f'Formula1_{year}Season_SprintResults.csv'

    df_race = load_data(str(race_path))
    if df_race is None:
        return None
    df_race['SessionType'] = 'Race'

    frames = [df_race]
    if sprint_path.exists():
        df_sprint = load_data(str(sprint_path))
        if df_sprint is not None and not df_sprint.empty:
            df_sprint['SessionType'] = 'Sprint'
            frames.append(df_sprint)

    return clean_data(pd.concat(frames, ignore_index=True))


def load_race_grid(data_dir: str, year: int, race: str,
                   race_df: Optional[pd.DataFrame] = None) -> Dict[str, int]:
    """Qualifying grid for one race, falling back to the race's Starting Grid.

    Canonical implementation shared by the Predictor and Report pages (they
    previously carried near-identical copies). In the fallback, grid entries
    <= 0 mean unknown and are skipped.

    Args:
        data_dir: Directory containing the CSV files.
        year: Season year used in the CSV filenames.
        race: Track name to look up.
        race_df: Cleaned season frame for the Starting-Grid fallback.

    Returns:
        dict mapping driver name to grid position (possibly empty).
    """
    try:
        qpath = Path(data_dir) / f'Formula1_{year}Season_QualifyingResults.csv'
        qdf = load_data(str(qpath))
        if (qdf is not None and not qdf.empty and 'Track' in qdf.columns
                and 'Driver' in qdf.columns):
            grid: Dict[str, int] = {}
            for _, r in qdf[qdf['Track'] == race].iterrows():
                pos = pd.to_numeric(r.get('Position'), errors='coerce')
                if pd.notna(pos):
                    grid[r['Driver']] = int(pos)
            if grid:
                return grid
    except Exception as e:
        logger.warning(f"Qualifying grid unavailable for {year} {race}: {e}")

    grid = {}
    if (race_df is not None and not race_df.empty
            and {'Track', 'Driver'}.issubset(race_df.columns)):
        sub = race_df[race_df['Track'] == race]
        for _, r in sub.iterrows():
            g = pd.to_numeric(r.get('Starting Grid'), errors='coerce')
            if pd.notna(g) and g > 0:
                grid[r['Driver']] = int(g)
    return grid
