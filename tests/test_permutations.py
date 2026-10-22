# -*- coding: utf-8 -*-
"""Behaviour tests for permutations.py - the title arithmetic.

The point of these tests is that the numbers a page shows are the numbers the
points system actually allows, and that a driver who cannot win is never
reported as able to.
"""
import pytest

import permutations as title_math


STANDINGS = {
    'Antonelli': 340,
    'Verstappen': 238,
    'Leclerc': 215,
    'Hamilton': 196,
}


def test_weekend_max_normal_and_sprint():
    assert title_math.weekend_max(False) == 26
    assert title_math.weekend_max(True) == 34


def test_remaining_points_counts_races_and_sprints():
    assert title_math.remaining_points(0) == 0
    assert title_math.remaining_points(3) == 78
    assert title_math.remaining_points(3, sprints_left=1) == 86


def test_remaining_points_ignores_negative_and_clamps_sprints():
    assert title_math.remaining_points(-2) == 0
    assert title_math.remaining_points(2, sprints_left=99) == 52 + 16


def test_clinch_round_none_while_a_rival_can_still_catch_up():
    # 20 points clear with 5 races left: 130 points remain after round 0, so
    # nothing is settled yet.
    assert title_math.clinch_round(20, 5) is None


def test_clinch_round_finds_the_settling_round():
    # 100 clear with 3 races left: once one round is gone only 52 points are
    # left for a rival, so the title is settled at round 1.
    assert title_math.clinch_round(100, 3) == 1
    # 40 clear with 5 left: 26 points remain after round 4 -> settled at round 4.
    assert title_math.clinch_round(40, 5) == 4


def test_clinch_round_at_season_end_is_not_an_early_clinch():
    assert title_math.clinch_round(5, 0) is None
    assert title_math.clinch_round(5, 1) is None


def test_title_clinch_checks_orders_by_points_and_marks_the_leader():
    rows = title_math.title_clinch_checks(STANDINGS, races_left=4)
    assert [row['driver'] for row in rows] == ['Antonelli', 'Verstappen',
                                               'Leclerc', 'Hamilton']
    leader = rows[0]
    assert leader['gap_to_leader'] == 0
    assert leader['can_win'] is True
    assert leader['verdict'] == 'Leads the championship'


def test_title_clinch_checks_can_rule_a_driver_out():
    # 104 points available (4 x 26); Verstappen is 102 behind -> still alive,
    # Hamilton at 144 behind -> out.
    rows = {row['driver']: row
            for row in title_math.title_clinch_checks(STANDINGS, races_left=4)}
    assert rows['Verstappen']['can_win'] is True
    assert rows['Hamilton']['can_win'] is False
    assert rows['Hamilton']['verdict'] == 'Cannot win the title any more'


def test_title_clinch_checks_with_no_races_left_only_the_leader_can_win():
    rows = title_math.title_clinch_checks(STANDINGS, races_left=0)
    assert [row['driver'] for row in rows if row['can_win']] == ['Antonelli']


def test_points_needed_is_one_more_than_the_gap():
    rows = {row['driver']: row
            for row in title_math.title_clinch_checks(STANDINGS, races_left=6)}
    gap = 340 - 238
    assert rows['Verstappen']['points_needed'] == gap + 1


def test_title_clinch_checks_on_empty_input():
    assert title_math.title_clinch_checks({}, races_left=3) == []
    assert title_math.title_clinch_checks(None, races_left=3) == []


def test_explain_clinch_wording():
    rows = {row['driver']: row
            for row in title_math.title_clinch_checks(STANDINGS, races_left=4)}
    text = title_math.explain_clinch(rows['Verstappen'], 'Antonelli', 4)
    assert 'Verstappen' in text and 'Antonelli' in text
    assert 'countback' in text
    out = title_math.explain_clinch(rows['Hamilton'], 'Antonelli', 4)
    assert 'not enough points remain' in out


def test_title_is_settled_only_when_nobody_can_catch_the_leader():
    assert title_math.title_is_settled(STANDINGS, races_left=0) is True
    assert title_math.title_is_settled(STANDINGS, races_left=4) is False


def test_fastest_lap_point_is_part_of_the_weekend_maximum():
    # A win plus the fastest lap is 26, not 25 - the page must not understate
    # the points still available.
    assert title_math.RACE_WEEKEND_MAX == title_math.RACE_WIN + title_math.FASTEST_LAP


@pytest.mark.parametrize('points,expected_max', [(1, 26), (2, 52), (5, 130)])
def test_remaining_points_scales_with_races(points, expected_max):
    assert title_math.remaining_points(points) == expected_max
