# -*- coding: utf-8 -*-
"""Offline tests for the ghost-lap comparison in src/lapcompare.py.

Everything is synthetic: no OpenF1 request, no SQLite store, no Streamlit.
A constant-speed trace has a distance that can be worked out by hand, which is
what pins the integration and the sign convention down.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lapcompare import (
    compare_laps,
    best_lap,
    lap_trace,
    mini_sectors,
    resample_to_distance,
    summary,
)

# 100 km/h for 60 s covers 1666.7 m, so a 10 m grid ends on 1660 m.
CONSTANT_SPEED_KMH = 100.0
CONSTANT_SECONDS = 60.0


def _car_data(kmh=CONSTANT_SPEED_KMH, seconds=CONSTANT_SECONDS, hz=4.0,
              gear=7.0, drs=0.0):
    """A synthetic car_data window: one channel per row, like OpenF1."""
    step = 1.0 / hz
    clock = np.arange(0.0, seconds + step / 2.0, step)
    count = clock.size
    return pd.DataFrame({
        "t": clock,
        "speed": np.full(count, float(kmh)),
        "throttle": np.full(count, 100.0),
        "brake": np.zeros(count),
        "n_gear": np.full(count, float(gear)),
        "drs": np.full(count, float(drs)),
    })


def _distance_lap(kmh=CONSTANT_SPEED_KMH, seconds=CONSTANT_SECONDS, step_m=10.0):
    """One constant-speed lap, resampled onto the distance grid."""
    trace = lap_trace(_car_data(kmh=kmh, seconds=seconds), 0.0, seconds)
    return resample_to_distance(trace, step_m)


def test_distance_integration_is_monotonic_and_plausible():
    lap = _distance_lap()
    assert not lap.empty
    assert lap.index.name == "Distance"
    grid = np.asarray(lap.index, dtype=float)
    assert grid[0] == pytest.approx(0.0)
    assert np.all(np.diff(grid) > 0)
    # The last whole step before the 1666.7 m the lap actually covers.
    assert float(grid[-1]) == pytest.approx(1660.0)
    assert 1650.0 <= float(grid[-1]) <= 100 / 3.6 * CONSTANT_SECONDS
    # 1660 m at 27.78 m/s is 59.76 s of the 60 s trace.
    assert float(lap["Time"].iloc[-1]) == pytest.approx(59.76, abs=0.05)
    assert np.all(np.diff(lap["Time"].to_numpy()) > 0)


def test_resample_uses_the_requested_distance_step():
    lap = _distance_lap(step_m=25.0)
    steps = np.diff(np.asarray(lap.index, dtype=float))
    assert np.allclose(steps, 25.0)
    assert float(lap.index[-1]) == pytest.approx(1650.0)


def test_resample_holds_state_channels_instead_of_averaging_them():
    car = _car_data()
    car.loc[car["t"] < 30.0, "n_gear"] = 1.0
    lap = resample_to_distance(lap_trace(car, 0.0, CONSTANT_SECONDS))
    gears = set(np.unique(lap["gear"].to_numpy()))
    # A gear does not exist between two gears: no intermediate values.
    assert gears <= {1.0, 7.0}
    assert float(lap["gear"].iloc[0]) == pytest.approx(1.0)
    assert float(lap["gear"].iloc[-1]) == pytest.approx(7.0)


def test_delta_is_positive_when_the_second_lap_is_faster():
    slow = _distance_lap(kmh=100.0)
    fast = _distance_lap(kmh=110.0)
    delta = compare_laps(slow, fast)
    assert not delta.empty
    assert list(delta.columns) == ["Distance", "TimeA", "TimeB", "Delta"]
    # Positive delta = driver B reached that distance first.
    assert np.all(delta["Delta"].to_numpy()[1:] > 0.0)
    assert float(delta["Delta"].iloc[-1]) == pytest.approx(5.433, abs=0.01)
    assert float(delta["Distance"].iloc[-1]) == pytest.approx(1660.0)
    assert np.allclose(np.diff(delta["Distance"].to_numpy()), 10.0)


def test_delta_is_zero_for_two_identical_laps():
    delta = compare_laps(_distance_lap(), _distance_lap())
    assert not delta.empty
    assert np.allclose(delta["Delta"].to_numpy(), 0.0)


def test_mini_sectors_split_the_distance_and_name_the_winner():
    delta = pd.DataFrame({"Distance": np.arange(0.0, 800.0, 10.0)})
    delta["TimeA"] = 0.0
    delta["TimeB"] = 0.0
    delta["Delta"] = np.where(delta["Distance"] < 400.0, -0.1, 0.1)

    segments = mini_sectors(delta, 8)
    assert len(segments) == 8
    assert [segment["segment"] for segment in segments] == list(range(1, 9))
    # Negative delta favours A; the first half of the distance is negative.
    assert all(segment["winner"] == "A" for segment in segments[:4])
    assert all(segment["winner"] == "B" for segment in segments[4:])
    for left, right in zip(segments, segments[1:]):
        assert left["to_m"] == pytest.approx(right["from_m"])
    assert all(segment["delta"] == pytest.approx(-0.1) for segment in segments[:4])
    assert all(segment["delta"] == pytest.approx(0.1) for segment in segments[4:])
    assert all(isinstance(segment["segment"], int) for segment in segments)


def test_mini_sectors_are_empty_rather_than_wrong():
    assert mini_sectors(pd.DataFrame()) == []
    assert mini_sectors(pd.DataFrame({"Distance": [0.0], "Delta": [0.0]})) == []
    delta = pd.DataFrame({
        "Distance": np.arange(0.0, 100.0, 10.0),
        "Delta": np.zeros(10),
    })
    assert mini_sectors(delta, 0) == []
    assert mini_sectors(pd.DataFrame({"Distance": [0.0, 10.0]}), 4) == []


def test_compare_laps_returns_empty_for_empty_or_one_lap_input():
    assert compare_laps(pd.DataFrame(), pd.DataFrame()).empty
    assert compare_laps(_distance_lap(), pd.DataFrame()).empty
    assert compare_laps(pd.DataFrame(), _distance_lap()).empty
    assert not compare_laps(_distance_lap(), _distance_lap()).empty


def test_best_lap_picks_the_fastest_usable_lap():
    laps = pd.DataFrame({
        "driver_number": [4, 4, 4, 4, 1],
        "lap_number": [1, 2, 3, 4, 1],
        "start_t": [100.0, 200.0, 300.0, 400.0, 90.0],
        "lap_duration": [95.0, 82.0, 81.0, 83.0, 94.0],
        "is_pit_out": [0, 0, 1, 0, 0],
    })
    lap = best_lap(laps, 4)
    assert lap is not None
    # Lap 3 is the quickest but it is a pit-out lap, so it is skipped.
    assert lap["number"] == 2
    assert lap["start"] == pytest.approx(200.0)
    assert lap["end"] == pytest.approx(282.0)
    assert lap["duration"] == pytest.approx(82.0)
    assert best_lap(laps, 1)["number"] == 1


def test_best_lap_is_none_without_a_usable_lap():
    assert best_lap(pd.DataFrame(), 4) is None
    assert best_lap(pd.DataFrame({"lap_number": [1]}), 4) is None
    no_start = pd.DataFrame({
        "driver_number": [4], "lap_number": [1],
        "start_t": [None], "lap_duration": [82.0],
    })
    assert best_lap(no_start, 4) is None
    all_pit_out = pd.DataFrame({
        "driver_number": [4, 4], "lap_number": [1, 2],
        "start_t": [10.0, 100.0], "lap_duration": [82.0, 83.0],
        "is_pit_out": [1, 1],
    })
    assert best_lap(all_pit_out, 4) is None
    assert best_lap(pd.DataFrame({
        "driver_number": [4], "lap_number": [1],
        "start_t": [10.0], "lap_duration": [82.0],
    }), 99) is None


def test_lap_trace_clips_to_the_lap_and_rebases_the_clock():
    car = pd.DataFrame({
        "t": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        "speed": [0.0, 100.0, 150.0, 160.0, 150.0, 120.0],
        "throttle": [0.0, 50.0, 100.0, 100.0, 100.0, 90.0],
        "brake": [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        "n_gear": [0.0, 3.0, 5.0, 6.0, 6.0, 5.0],
        "drs": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    })
    trace = lap_trace(car, 10.0, 13.0)
    assert list(trace.index) == [0.0, 1.0, 2.0, 3.0]
    assert trace.index.name == "Time"
    assert trace["speed"].tolist() == [100.0, 150.0, 160.0, 150.0]
    assert trace["gear"].tolist() == [3.0, 5.0, 6.0, 6.0]
    assert trace["brake"].tolist() == [0.0, 0.0, 0.0, 1.0]
    # No window, an empty window and a single sample all come back empty.
    assert lap_trace(car, None, 13.0).empty
    assert lap_trace(car, 50.0, 60.0).empty
    assert lap_trace(car, 12.5, 13.0).empty
    assert lap_trace(pd.DataFrame(), 0.0, 1.0).empty


def test_resample_rejects_a_trace_it_cannot_use():
    assert resample_to_distance(pd.DataFrame()).empty
    one_sample = lap_trace(_car_data(seconds=0.0), 0.0, 1.0)
    assert resample_to_distance(one_sample).empty
    assert resample_to_distance(_distance_lap(), 0.0).empty
    assert resample_to_distance(_distance_lap(), -10.0).empty


def test_summary_reports_gap_top_speed_and_biggest_swing():
    slow = _distance_lap(kmh=100.0)
    fast = _distance_lap(kmh=110.0)
    info = summary(
        {"driver": "Charles Leclerc", "trace": slow},
        {"driver": "Max Verstappen", "trace": fast},
    )
    assert info["available"] is True
    assert isinstance(info["gap_s"], float)
    assert info["gap_s"] == pytest.approx(5.433, abs=0.01)
    assert info["compared_distance_m"] == pytest.approx(1660.0)
    assert info["top_speed_a_kmh"] == pytest.approx(100.0)
    assert info["top_speed_b_kmh"] == pytest.approx(110.0)
    assert info["top_speed_diff_kmh"] == pytest.approx(10.0)
    assert len(info["segments"]) == 8
    gain = info["biggest_gain"]
    loss = info["biggest_loss"]
    assert gain["driver"] in {"Charles Leclerc", "Max Verstappen"}
    assert gain["delta_s"] > 0.0
    assert loss["delta_s"] >= gain["delta_s"]
    assert 1 <= gain["segment"] <= 8


def test_summary_without_two_laps_is_unavailable_not_an_exception():
    empty = summary({}, {})
    assert empty["available"] is False
    assert empty["gap_s"] is None
    assert empty["biggest_gain"] is None
    one_side = summary({"driver": "A", "trace": _distance_lap()}, {})
    assert one_side["available"] is False
    assert one_side["top_speed_a_kmh"] == pytest.approx(100.0)
    assert summary(None, None)["available"] is False
