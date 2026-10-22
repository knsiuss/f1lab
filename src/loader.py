# -*- coding: utf-8 -*-
"""
loader.py
~~~~~~~~~
Schema normalisation for the season results frame.

The season data itself comes from the OpenF1 API (:mod:`openf1_results`);
this module owns the shared shape every consumer relies on: numeric columns
are coerced, a "did this driver finish?" flag is derived, and the column
order is fixed.

The app no longer reads local CSV exports -- there is no file path left in
this module.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import logging
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)

# Columns every consumer of the season frame expects to exist.
REQUIRED_COLUMNS = ['Points', 'Position', 'Starting Grid', 'Laps', 'Time/Retired']


def clean_data(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Normalize column types and create derived fields.

    Args:
        df: the season frame produced by :func:`openf1_results.build_season_results`.
        strict: raise when a required column is missing instead of warning.
            The app runs in non-strict mode so a partial response degrades
            into "some numbers are missing" rather than a crashed page.
    """
    logger.info("Cleaning data...")

    if df is None or df.empty:
        if strict:
            raise ValueError("DataFrame is None or empty")
        logger.warning("Cannot clean None or empty DataFrame")
        return df if df is not None else pd.DataFrame()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        message = f"Missing required columns: {missing}"
        if strict:
            logger.error(message)
            raise ValueError(message)
        logger.warning("%s (continuing with the columns that are present)", message)
    else:
        # Make a copy to avoid modifying the caller's frame.
        df = df.copy()

    if 'Points' in df.columns:
        df['Points'] = pd.to_numeric(df['Points'], errors='coerce').fillna(0)
    if 'Position' in df.columns:
        df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
    if 'Starting Grid' in df.columns:
        df['Starting Grid'] = pd.to_numeric(df['Starting Grid'], errors='coerce')
    if 'Laps' in df.columns:
        df['Laps'] = pd.to_numeric(df['Laps'], errors='coerce')

    # Finished flag: a driver who has a classified position and is not
    # flagged as retired. The status column is authoritative when present;
    # the historic 'Time/Retired' text is the fallback.
    df['Finished'] = _finished_flags(df)

    finishers = int(df['Finished'].sum())
    logger.info("Data cleaned: %s rows, %s finishers (%.1f%%)",
                len(df), finishers, finishers / len(df) * 100 if len(df) else 0.0)
    return df


def _finished_flags(df: pd.DataFrame) -> pd.Series:
    """Per-row "finished the race" flag."""
    position_ok = (df['Position'].notna() if 'Position' in df.columns
                   else pd.Series(True, index=df.index))
    if 'Status' in df.columns:
        status = df['Status'].astype(str).str.strip().str.upper()
        return position_ok & status.isin(['FINISHED', 'CLASSIFIED', ''])
    if 'Time/Retired' in df.columns:
        try:
            retired = df['Time/Retired'].astype(str).str.contains(
                'DNS|DSQ|DNF', na=False, regex=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Error creating Finished flag: %s", exc)
            return position_ok
        return position_ok & ~retired
    return position_ok


def race_tracks(df: pd.DataFrame) -> List[str]:
    """Ordered list of the tracks in a season frame (calendar order)."""
    if df is None or df.empty or 'Track' not in df.columns:
        return []
    if 'Round' in df.columns:
        ordered = (df[['Round', 'Track']].dropna()
                   .drop_duplicates('Track')
                   .sort_values('Round'))
        return [str(t) for t in ordered['Track'].tolist()]
    seen: List[str] = []
    for track in df['Track'].tolist():
        text = str(track)
        if text not in seen:
            seen.append(text)
    return seen
