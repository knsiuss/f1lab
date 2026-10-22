# -*- coding: utf-8 -*-
"""Tests for the session weather and rain outlook in src/wx.py.

Everything here is offline: the OpenF1 client and the Open-Meteo request are
replaced by fakes, so the suite never touches the network.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import wx  # noqa: E402

BASE = datetime(2026, 10, 11, 11, 0, tzinfo=timezone.utc)


def _rows(air, track, humidity=None, wind=None, rain=None, pressure=None):
    """Synthetic OpenF1 /weather rows, one per reading."""
    humidity = humidity or [72.0] * len(air)
    wind = wind or [1.0] * len(air)
    rain = rain or [0.0] * len(air)
    pressure = pressure or [1010.0] * len(air)
    rows = []
    for index, (air_value, track_value) in enumerate(zip(air, track)):
        stamp = (BASE + timedelta(minutes=index)).isoformat().replace('+00:00', 'Z')
        rows.append({
            'session_key': 11388,
            'meeting_key': 1296,
            'date': stamp,
            'air_temperature': air_value,
            'track_temperature': track_value,
            'humidity': humidity[index],
            'pressure': pressure[index],
            'wind_speed': wind[index],
            'rainfall': rain[index],
        })
    return rows


# --- measured session weather ------------------------------------------------

def test_dry_session_summary():
    rows = _rows([30.0, 31.0, 30.5], [40.0, 44.0, 42.0])
    out = wx.summarise_weather(rows)
    assert out['samples'] == 3
    assert out['session_key'] == 11388
    assert out['air_temp_mean'] == pytest.approx(30.5)
    assert out['air_temp_min'] == 30.0
    assert out['air_temp_max'] == 31.0
    assert out['track_temp_mean'] == pytest.approx(42.0)
    assert out['wind_speed_max'] == 1.0
    assert out['wind_speed_max_kmh'] == pytest.approx(3.6)
    assert out['wet'] is False
    assert out['rainfall_total'] == 0.0
    assert out['rain_samples'] == 0
    assert out['started_at'] == '2026-10-11T11:00:00Z'
    assert out['ended_at'] == '2026-10-11T11:02:00Z'


def test_wet_session_detected_from_rainfall_flag():
    rows = _rows([25.0] * 4, [33.0] * 4, rain=[0.0, 1.0, 1.0, 0.0])
    out = wx.summarise_weather(rows)
    assert out['wet'] is True
    assert out['rain_samples'] == 2
    assert out['rainfall_total'] == 2.0


def test_zero_track_temperature_is_dropped_but_air_zero_is_kept():
    rows = _rows([0.0, 20.0], [0.0, 40.0])
    out = wx.summarise_weather(rows)
    assert out['track_temp_zeros_dropped'] == 1
    assert out['track_temp_min'] == 40.0
    assert out['track_temp_mean'] == pytest.approx(40.0)
    # A genuine 0 C air reading is data, not a dropout.
    assert out['air_temp_min'] == 0.0
    assert out['air_temp_mean'] == pytest.approx(10.0)


def test_summary_skips_missing_and_non_numeric_readings():
    rows = _rows([20.0, 22.0], [30.0, 34.0])
    rows[0]['air_temperature'] = None
    rows[0]['wind_speed'] = None
    rows[1]['wind_speed'] = 'n/a'
    rows.append({'date': rows[0]['date'], 'air_temperature': float('nan'),
                 'track_temperature': float('inf')})
    out = wx.summarise_weather(rows)
    assert out['samples'] == 3
    assert out['air_temp_min'] == 22.0
    assert out['wind_speed_max'] is None
    assert out['track_temp_max'] == 34.0


def test_summary_accepts_a_dataframe():
    rows = _rows([20.0, 22.0], [30.0, 34.0])
    from_frame = wx.summarise_weather(pd.DataFrame(rows))
    from_list = wx.summarise_weather(rows)
    assert from_frame['air_temp_mean'] == from_list['air_temp_mean']
    assert from_frame['samples'] == from_list['samples']


@pytest.mark.parametrize('empty', [None, [], pd.DataFrame()])
def test_summary_returns_none_when_the_feed_is_empty(empty):
    assert wx.summarise_weather(empty) is None


def test_session_weather_uses_the_shared_client(monkeypatch):
    seen = {}

    def fake_client(endpoint, **params):
        seen['endpoint'] = endpoint
        seen['params'] = params
        return _rows([25.0], [35.0])

    monkeypatch.setattr(wx, 'cached_openf1', fake_client)
    out = wx.session_weather('11388')
    assert seen == {'endpoint': 'weather', 'params': {'session_key': 11388}}
    assert out['samples'] == 1


def test_session_weather_returns_none_on_failure_and_bad_key(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError('OpenF1 is down')

    monkeypatch.setattr(wx, 'cached_openf1', broken)
    assert wx.session_weather(1) is None
    assert wx.session_weather(None) is None
    assert wx.session_weather('not-a-session') is None


# --- coordinate matcher ------------------------------------------------------

@pytest.mark.parametrize('name, expected', [
    ('Singapore Grand Prix', 'singapore'),
    ('Marina Bay', 'singapore'),
    ('SINGAPORE', 'singapore'),
    ('British Grand Prix', 'britain'),
    ('Silverstone', 'britain'),
    ('São Paulo Grand Prix', 'brazil'),
    ('Sao Paulo GP', 'brazil'),
    ('United States Grand Prix', 'usa'),
    ('Emilia Romagna Grand Prix', 'imola'),
    ('Monaco Grand Prix', 'monaco'),
])
def test_coords_for_matches_event_and_city_names(name, expected):
    assert wx.coords_for(name) == wx.CIRCUIT_COORDS[expected]


@pytest.mark.parametrize('name', ['Atlantis Grand Prix', '', None, 'Narnia'])
def test_coords_for_returns_none_on_a_miss(name):
    assert wx.coords_for(name) is None


def test_every_calendar_circuit_has_coordinates():
    assert len(wx.CIRCUIT_COORDS) >= 24
    for latitude, longitude in wx.CIRCUIT_COORDS.values():
        assert -90.0 <= latitude <= 90.0
        assert -180.0 <= longitude <= 180.0


# --- rain bands --------------------------------------------------------------

@pytest.mark.parametrize('probability, band', [
    (0, 'dry'),
    (19.9, 'dry'),
    (20, 'possible'),
    (59.9, 'possible'),
    (60, 'likely'),
    (100, 'likely'),
    (140, 'likely'),
    (-5, 'dry'),
])
def test_rain_band_boundaries(probability, band):
    assert wx.rain_band(probability) == band


@pytest.mark.parametrize('value', [None, 'wet', float('nan')])
def test_rain_band_unusable_value_is_none(value):
    assert wx.rain_band(value) is None


# --- forecast parsing --------------------------------------------------------

def _payload(times, temperatures, probabilities, winds):
    return {
        'latitude': 1.2914,
        'longitude': 103.864,
        'hourly': {
            'time': times,
            'temperature_2m': temperatures,
            'precipitation_probability': probabilities,
            'wind_speed_10m': winds,
            'rain': [0.0, 1.2, 0.0][:len(times)],
        },
    }


PAYLOAD = _payload(
    ['2026-10-24T11:00', '2026-10-24T12:00', '2026-10-24T13:00'],
    [26.0, 27.5, 28.0],
    [10, 67, 80],
    [5.0, 7.2, 8.0],
)


def test_parse_forecast_takes_the_session_hour():
    out = wx.parse_forecast(PAYLOAD, '2026-10-24T12:00:00Z')
    assert out['time_utc'] == '2026-10-24T12:00'
    assert out['temperature_c'] == 27.5
    assert out['precipitation_probability'] == 67
    assert isinstance(out['precipitation_probability'], int)
    assert out['wind_speed_kmh'] == 7.2
    assert out['rain_mm'] == 1.2
    assert out['source'] == 'Open-Meteo'
    assert out['estimated'] is True


def test_parse_forecast_falls_back_to_the_nearest_hour():
    out = wx.parse_forecast(PAYLOAD, '2026-10-24T12:40:00Z')
    assert out['time_utc'] == '2026-10-24T12:00'
    far = _payload(['2026-10-27T12:00'], [20.0], [10], [4.0])
    assert wx.parse_forecast(far, '2026-10-24T12:00:00Z') is None


@pytest.mark.parametrize('payload', [
    {},
    {'hourly': []},
    {'hourly': {'time': []}},
    {'hourly': {'time': ['2026-10-24T12:00'], 'temperature_2m': [None],
                'precipitation_probability': [None], 'wind_speed_10m': [None]}},
])
def test_parse_forecast_returns_none_on_unusable_payload(payload):
    assert wx.parse_forecast(payload, '2026-10-24T12:00:00Z') is None


def test_forecast_parses_a_fake_network_payload(monkeypatch):
    wx.clear_forecast_cache()
    seen = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return PAYLOAD

    def fake_get(url, params=None, timeout=None):
        seen['url'] = url
        seen['params'] = params
        return FakeResponse()

    monkeypatch.setattr(wx.requests, 'get', fake_get)
    out = wx.forecast('Marina Bay', datetime(2026, 10, 24, 12, tzinfo=timezone.utc))
    assert out['temperature_c'] == 27.5
    assert out['precipitation_probability'] == 67
    assert seen['url'] == wx.OPEN_METEO_URL
    assert seen['params']['latitude'] == pytest.approx(1.291)
    assert seen['params']['start_date'] == '2026-10-24'
    wx.clear_forecast_cache()


def test_forecast_returns_none_when_the_network_fails(monkeypatch):
    wx.clear_forecast_cache()

    def boom(*args, **kwargs):
        raise RuntimeError('offline')

    monkeypatch.setattr(wx.requests, 'get', boom)
    assert wx.forecast('Marina Bay', '2026-10-24T12:00:00Z') is None
    # A circuit that is not in the table does not even reach the network.
    assert wx.forecast('Atlantis Grand Prix', '2026-10-24T12:00:00Z') is None
    wx.clear_forecast_cache()


# --- rain outlook ------------------------------------------------------------

def _fake_forecast(probability=67, temperature=27.0, wind=7.0):
    return {
        'precipitation_probability': probability,
        'temperature_c': temperature,
        'wind_speed_kmh': wind,
    }


def test_rain_outlook_confidence_follows_the_horizon(monkeypatch):
    monkeypatch.setattr(wx, 'forecast', lambda *a, **k: _fake_forecast())
    when = datetime(2026, 10, 24, 12, tzinfo=timezone.utc)

    near = wx.rain_outlook('Marina Bay', when,
                           now=datetime(2026, 10, 23, 12, tzinfo=timezone.utc))
    assert near['band'] == 'likely'
    assert near['probability'] == 67.0
    assert near['confidence'] == 'moderate'
    assert near['horizon_hours'] == pytest.approx(24.0)
    assert near['estimated'] is True
    assert 'tyre' in near['implication']
    assert 'safety car' in near['implication']

    far = wx.rain_outlook('Marina Bay', when,
                          now=datetime(2026, 10, 19, 12, tzinfo=timezone.utc))
    assert far['horizon_hours'] == pytest.approx(120.0)
    assert far['confidence'] == 'low'


def test_rain_outlook_dry_band_implication(monkeypatch):
    monkeypatch.setattr(wx, 'forecast', lambda *a, **k: _fake_forecast(probability=5))
    out = wx.rain_outlook('Monza', '2026-10-24T12:00:00Z')
    assert out['band'] == 'dry'
    assert 'slick tyres' in out['implication']
    assert 'safety car' in out['implication']


def test_rain_outlook_returns_none_without_a_probability(monkeypatch):
    monkeypatch.setattr(wx, 'forecast', lambda *a, **k: None)
    assert wx.rain_outlook('Monza', '2026-10-24T12:00:00Z') is None
    monkeypatch.setattr(wx, 'forecast', lambda *a, **k: _fake_forecast(probability=None))
    assert wx.rain_outlook('Monza', '2026-10-24T12:00:00Z') is None
