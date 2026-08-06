# -*- coding: utf-8 -*-
"""Tests for sector/corner insights in src/sector.py."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sector import ESTIMATED, sector_deficits, sector_summary


def _laps(n=8, base=90.0, driver='VER', s1=28.0, s2=30.0, s3=32.0, stint=1):
    """Synthetic laps: lap_time = base + small noise; fixed sector times."""
    rows = []
    for i in range(1, n + 1):
        rows.append({
            'Driver': driver, 'Stint': stint,
            'LapTime': pd.Timedelta(seconds=base + i * 0.05),
            'TyreLife': i,
            'Sector1': pd.Timedelta(seconds=s1),
            'Sector2': pd.Timedelta(seconds=s2),
            'Sector3': pd.Timedelta(seconds=s3),
        })
    return pd.DataFrame(rows)


def test_sector_summary_best_mean_std():
    out = sector_summary(_laps())
    assert len(out) == 1
    row = out.iloc[0]
    assert row['Driver'] == 'VER'
    assert row['Sector1Best'] == 28.0
    assert row['Sector1Std'] == 0.0  # identical sector times
    assert row['Estimated'] == ESTIMATED


def test_sector_summary_multiple_drivers_sorted_by_s1():
    df = pd.concat([_laps(driver='VER', s1=28.0), _laps(driver='LEC', s1=29.0)])
    out = sector_summary(df)
    assert list(out['Driver']) == ['VER', 'LEC']


def test_sector_summary_accepts_sector_time_spelling():
    # FastF1 builds name these Sector1Time/Sector2Time/Sector3Time; both
    # spellings must work without the module hardcoding one.
    df = _laps().rename(columns={
        'Sector1': 'Sector1Time', 'Sector2': 'Sector2Time', 'Sector3': 'Sector3Time',
    })
    out = sector_summary(df)
    assert len(out) == 1
    assert out.iloc[0]['Sector1Best'] == 28.0
    dfits = sector_deficits(df)
    assert not dfits.empty
    assert dfits.iloc[0]['Sector1Def'] == 0.0


def test_sector_summary_missing_sectors_ok():
    df = _laps().drop(columns=['Sector3'])
    out = sector_summary(df)
    assert 'Sector3Best' not in out.columns
    assert out.iloc[0]['Sector1Best'] == 28.0


def test_sector_summary_empty():
    assert sector_summary(pd.DataFrame()).empty


def test_sector_deficits_points_to_fastest():
    df = pd.concat([
        _laps(driver='VER', s1=28.0, s2=31.0, s3=32.0),
        _laps(driver='LEC', s1=28.5, s2=30.0, s3=32.0),
    ])
    out = sector_deficits(df)
    # VER is fastest in S1, LEC fastest in S2.
    ver = out[out['Driver'] == 'VER'].iloc[0]
    lec = out[out['Driver'] == 'LEC'].iloc[0]
    assert ver['Sector1Def'] == 0.0
    assert lec['Sector2Def'] == 0.0
    assert lec['Sector1Def'] > 0
    assert out['Estimated'].eq(ESTIMATED).all()


def test_sector_deficits_insufficient_sample_none():
    out = sector_deficits(_laps(n=1))
    # One clean lap is below the fit threshold but sector stats still show.
    assert not out.empty
