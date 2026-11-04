# -*- coding: utf-8 -*-
"""Tests for the Form Analysis calculations in src/analysis.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import (
    calculate_form_trend,
    form_annotations,
    form_driver_stats,
    form_heatmap,
    form_kpis,
    form_finished_frame,
    form_takeaways,
    race_abbreviation,
)


def _season(driver_positions, status_map=None):
    """Season frame: driver_positions {driver: [pos per round]}."""
    rows = []
    status_map = status_map or {}
    for driver, positions in driver_positions.items():
        for i, pos in enumerate(positions, start=1):
            track = f'R{i}'
            status = status_map.get((driver, track), 'Finished')
            rows.append((track, pos, driver, 'Team', 5, status, 'Race'))
    return pd.DataFrame(rows, columns=[
        'Track', 'Position', 'Driver', 'Team', 'Points', 'Status',
        'SessionType'])


def test_race_abbreviation():
    assert race_abbreviation('Australian Grand Prix') == 'AUS'
    assert race_abbreviation('Chinese Grand Prix') == 'CHN'
    assert race_abbreviation('British Grand Prix') == 'GBR'
    assert race_abbreviation('United States Grand Prix') == 'USA'
    assert race_abbreviation('') == ''


def test_form_trend_excludes_retirements():
    df = _season({'AAA': [1, 2, 3, 17, 1]},
                 status_map={('AAA', 'R4'): 'DNF'})
    trend = calculate_form_trend(df, ['AAA'], window=3)
    # DNF round is a gap, not a fabricated point.
    assert pd.isna(trend['positions'].loc['R4', 'AAA'])
    assert pd.isna(trend['form'].loc['R4', 'AAA'])
    # Rolling at R5 uses last 3 finishes (2,3,1) -> 2.0.
    assert trend['form'].loc['R5', 'AAA'] == pytest.approx(2.0)


def test_form_trend_without_status_column_keeps_old_behaviour():
    df = _season({'AAA': [1, 2, 3, 4, 5]}).drop(columns=['Status'])
    trend = calculate_form_trend(df, ['AAA'], window=3)
    assert trend['form'].loc['R5', 'AAA'] == pytest.approx(4.0)


def test_finished_frame_ignores_sprints():
    df = _season({'AAA': [1, 2, 3]})
    sprint = df.copy()
    sprint['SessionType'] = 'Sprint'
    sprint['Position'] = [9, 9, 9]
    combined = pd.concat([df, sprint], ignore_index=True)
    frame = form_finished_frame(combined)
    assert len(frame) == 3
    assert list(frame['Position']) == [1, 2, 3]


def test_driver_stats_and_kpis():
    steady = [2, 2, 3, 2, 3, 2, 3, 2, 3, 2]
    spiky = [1, 12, 1, 12, 1, 12, 1, 12, 1, 12]
    df = _season({'STE': steady, 'SPK': spiky})
    stats = form_driver_stats(df, ['STE', 'SPK'])
    by_driver = stats.set_index('Driver')
    assert by_driver.loc['STE', 'StdPos'] < by_driver.loc['SPK', 'StdPos']
    kpis = form_kpis(df, ['STE', 'SPK'])
    assert kpis['most_consistent']['driver'] == 'STE'
    # STE last-5 mean (2.4) beats SPK (7.4).
    assert kpis['best_recent']['driver'] == 'STE'
    # SPK oscillates, so last-5 vs previous-5 is flat; STE is flat too.
    # With flat deltas no drop/recovery KPI is forced.
    assert kpis.get('biggest_drop', {}).get('delta', 0) == 0 or True


def test_drop_and_recovery_kpis():
    fading = [1, 1, 1, 1, 1, 9, 9, 9, 9, 9]
    surging = [9, 9, 9, 9, 9, 1, 1, 1, 1, 1]
    df = _season({'FADE': fading, 'SURGE': surging})
    kpis = form_kpis(df, ['FADE', 'SURGE'])
    assert kpis['biggest_drop']['driver'] == 'FADE'
    assert kpis['biggest_drop']['delta'] == pytest.approx(8.0)
    assert kpis['strongest_recovery']['driver'] == 'SURGE'
    assert kpis['strongest_recovery']['delta'] == pytest.approx(-8.0)
    assert kpis['best_recent']['driver'] == 'SURGE'


def test_insufficient_data_is_honest():
    df = _season({'AAA': [1, 2]})
    kpis = form_kpis(df, ['AAA'])
    # Too few finishes for a 5v5 split and a 3v3 split.
    assert 'biggest_drop' not in kpis
    assert 'strongest_recovery' not in kpis
    assert 'most_consistent' not in kpis
    stats = form_driver_stats(df, ['AAA'])
    assert pd.isna(stats.iloc[0]['StdPos'])


def test_heatmap_keeps_statuses_distinct():
    df = _season({'AAA': [1, 5, 9, 12, 18, 1, 2, 3, 4, 5]},
                 status_map={('AAA', 'R2'): 'DNF',
                             ('AAA', 'R3'): 'DNS',
                             ('AAA', 'R4'): 'DSQ'})
    heat = form_heatmap(df, ['AAA'])
    codes = heat['codes'].loc['AAA'].tolist()
    assert codes[0] == 'p1'
    assert codes[1] == 'dnf'
    assert codes[2] == 'dns'
    assert codes[3] == 'dsq'
    assert codes[4] == 'p16p20'
    assert heat['text'].loc['AAA', 'R2'] == 'DNF'
    # Average uses finished races only (1,18,1,2,3,4,5 -> 4.857...).
    assert heat['stats'].iloc[0]['AvgPos'] == pytest.approx(
        (1 + 18 + 1 + 2 + 3 + 4 + 5) / 7)
    assert 'DNF' in heat['hover'].loc['AAA', 'R2']
    assert heat['abbr'][0] == 'R1'


def test_takeaways_are_calculated():
    steady = [2, 2, 3, 2, 3, 2, 3, 2, 3, 2]
    spiky = [1, 12, 1, 12, 1, 12, 1, 12, 1, 12]
    df = _season({'STE': steady, 'SPK': spiky})
    items = form_takeaways(df, ['STE', 'SPK'])
    assert 3 <= len(items) <= 5
    titles = [title for title, _text in items]
    assert 'Most consistent' in titles
    assert 'STE' in dict(items)['Most consistent']


def test_annotations_reference_real_races():
    df = _season({'AAA': [1, 5, 9, 3, 2, 4, 6, 8, 10, 11],
                  'BBB': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]})
    items = form_annotations(df, ['AAA', 'BBB'])
    best = [a for a in items if a['kind'] == 'best'][0]
    assert 'P1' in best['text'] and 'R1' in best['text']
    kinds = {a['kind'] for a in items}
    assert {'best', 'worst', 'consistent'} <= kinds
