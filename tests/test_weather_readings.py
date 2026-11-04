# -*- coding: utf-8 -*-
"""Tests for the Weekend Report weather additions in src/wx.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import wx  # noqa: E402


def test_rain_risk_five_bands():
    assert wx.rain_risk(0) == 'Very Low'
    assert wx.rain_risk(10) == 'Very Low'
    assert wx.rain_risk(11) == 'Low'
    assert wx.rain_risk(30) == 'Low'
    assert wx.rain_risk(31) == 'Moderate'
    assert wx.rain_risk(50) == 'Moderate'
    assert wx.rain_risk(51) == 'High'
    assert wx.rain_risk(70) == 'High'
    assert wx.rain_risk(71) == 'Very High'
    assert wx.rain_risk(100) == 'Very High'
    assert wx.rain_risk(None) is None
    assert wx.rain_risk('wet') is None


def test_weather_condition_labels():
    assert wx.weather_condition(0) == 'Clear sky'
    assert wx.weather_condition(2) == 'Partly cloudy'
    assert wx.weather_condition(3) == 'Overcast'
    assert wx.weather_condition(61) == 'Light rain'
    assert wx.weather_condition(None) is None
    assert wx.weather_condition(9999) is None


def test_wind_direction_labels():
    assert wx.wind_direction_label(0) == 'N'
    assert wx.wind_direction_label(90) == 'E'
    assert wx.wind_direction_label(180) == 'S'
    assert wx.wind_direction_label(270) == 'W'
    assert wx.wind_direction_label(None) is None


def test_weather_readings_sorted_and_normalised(monkeypatch):
    rows = [
        {'date': '2025-03-16T03:10:00+00:00',
         'air_temperature': 16.0, 'track_temperature': 19.0,
         'humidity': 80, 'pressure': 1010, 'wind_speed': 3.0,
         'wind_direction': 180, 'rainfall': 0},
        {'date': '2025-03-16T03:05:00+00:00',
         'air_temperature': 15.8, 'track_temperature': 18.8,
         'humidity': 82, 'pressure': 1009, 'wind_speed': 2.6,
         'wind_direction': 190, 'rainfall': 1},
    ]
    monkeypatch.setattr(wx, 'cached_openf1', lambda *a, **k: rows)
    readings = wx.weather_readings(123)
    assert len(readings) == 2
    assert readings[0]['time_utc'] < readings[1]['time_utc']
    assert readings[0]['rainfall'] == 1
    assert readings[0]['wind_direction'] == 190


def test_weather_readings_empty_on_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError('down')
    monkeypatch.setattr(wx, 'cached_openf1', boom)
    assert wx.weather_readings(123) == []
    assert wx.weather_readings('not-a-key') == []


def test_parse_forecast_carries_condition():
    payload = {
        'hourly': {
            'time': ['2025-03-16T04:00', '2025-03-16T05:00'],
            'temperature_2m': [15.0, 16.0],
            'precipitation_probability': [10, 40],
            'wind_speed_10m': [5.0, 6.0],
            'rain': [0.0, 0.2],
            'weather_code': [2, 61],
        }
    }
    parsed = wx.parse_forecast(payload, '2025-03-16T05:00:00Z')
    assert parsed['condition'] == 'Light rain'
    assert parsed['weather_code'] == 61


def test_interpretation_covers_all_four_cards():
    conditions = {'air_temp_mean': 24.0, 'track_temp_mean': 36.0,
                  'wind_speed_mean': 9.0, 'humidity_mean': 40.0,
                  'samples': 100, 'wet': False}
    outlook = {'probability': 12.0, 'band': 'dry', 'confidence': 'moderate',
               'when_utc': '2025-03-16T05:00:00Z'}
    cards = dict(wx.weather_interpretation(conditions, outlook,
                                           session='Practice 1'))
    assert set(cards) == {'Drivers', 'Teams', 'Tyre strategy',
                          'Race conditions'}
    assert '12%' in cards['Teams']
    assert all(text.strip() for text in cards.values())


def test_interpretation_is_honest_without_data():
    cards = dict(wx.weather_interpretation(None, None))
    assert all(('can not' in text or 'no ' in text.lower())
               for text in cards.values())
