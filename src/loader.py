# -*- coding: utf-8 -*-
"""
loader.py
CSV data loading and preprocessing.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional
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
        logger.error(f"Please ensure the file exists at the specified path")
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

def load_combined_data(data_dir: str) -> Optional[pd.DataFrame]:
    """
    Load and combine Race and Sprint results.
    
    Args:
        data_dir: Directory containing the CSV files.
        
    Returns:
        pd.DataFrame: Combined DataFrame containing both race and sprint results.
    """
    try:
        race_path = Path(data_dir) / 'Formula1_2025Season_RaceResults.csv'
        sprint_path = Path(data_dir) / 'Formula1_2025Season_SprintResults.csv'
        
        df_race = load_data(str(race_path))
        
        if df_race is None:
            return None
            
        df_race['SessionType'] = 'Race'
        
        if sprint_path.exists():
            df_sprint = load_data(str(sprint_path))
            if df_sprint is not None and not df_sprint.empty:
                df_sprint['SessionType'] = 'Sprint'
                # Ensure we don't have conflicting index or columns if concatenating
                return pd.concat([df_race, df_sprint], ignore_index=True)
        
        return df_race
        
    except Exception as e:
        logger.error(f"Error loading combined data: {e}")
        return None
