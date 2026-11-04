# -*- coding: utf-8 -*-
"""Tests for the live rain radar in :mod:`wx` and :mod:`radar`.

Everything here is offline: the RainViewer index request is replaced by a fake,
and the panel is never rendered -- the browser downloads the tiles, so the
Python side only ever sees the index document.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import radar  # noqa: E402
import styles  # noqa: E402
import wx  # noqa: E402

BASE = datetime(2026, 10, 23, 14, 30, tzinfo=timezone.utc)
NEWEST_AGE_S = 300               # the newest published frame lags ~5 minutes


def _frame(index, count=13, spacing=600):
    """The ``index``-th of ``count`` frames, oldest first."""
    newest = int(BASE.timestamp()) - NEWEST_AGE_S
    stamp = newest - (count - 1 - index) * spacing
    return {'time': stamp, 'path': '/v2/radar/%d' % stamp}


def _index(count=13, spacing=600, host='https://tilecache.rainviewer.com'):
    return {
        'version': '2.0',
        'generated': int(BASE.timestamp()) - 60,
        'host': host,
        'radar': {'past': [_frame(i, count, spacing) for i in range(count)],
                  'nowcast': []},
        'satellite': {'infrared': []},
    }


class _Response:
    """The part of ``requests.Response`` the radar code touches."""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise wx.requests.HTTPError('%d' % self.status_code)

    def json(self):
        return self._payload


class _Stub:
    """A ``requests.get`` that counts calls and replays canned payloads."""

    def __init__(self, payload=None, status=200, error=None):
        self.payload = payload if payload is not None else _index()
        self.status = status
        self.error = error
        self.count = 0
        self.urls = []

    def __call__(self, url, **kwargs):
        self.count += 1
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        return _Response(self.payload, self.status)


@pytest.fixture(autouse=True)
def clean_cache():
    """Every test starts (and ends) with an empty index cache."""
    wx.clear_radar_cache()
    yield
    wx.clear_radar_cache()


def _install(monkeypatch, **kwargs):
    stub = _Stub(**kwargs)
    monkeypatch.setattr(wx.requests, 'get', stub)
    return stub


# ---------------------------------------------------------------------------
# Parsing the index document
# ---------------------------------------------------------------------------

def test_frames_are_sorted_and_the_newest_is_last():
    payload = _index(count=3)
    payload['radar']['past'] = list(reversed(payload['radar']['past']))

    parsed = wx.parse_radar_index(payload, now=BASE)

    assert [frame['time'] for frame in parsed['frames']] == sorted(
        frame['time'] for frame in payload['radar']['past'])
    assert parsed['latest'] == parsed['frames'][-1]['path']
    assert parsed['count'] == 3
    assert parsed['span_min'] == pytest.approx(20.0)


def test_the_history_is_trimmed_to_the_published_window():
    parsed = wx.parse_radar_index(_index(count=30), now=BASE)
    assert parsed['count'] == wx.RADAR_FRAME_LIMIT
    assert parsed['latest'] == parsed['frames'][-1]['path']


def test_a_frame_without_a_timestamp_or_an_absolute_path_is_dropped():
    payload = _index(count=3)
    newest = payload['radar']['past'][-1]['path']
    payload['radar']['past'].append({'time': 'soon', 'path': '/v2/radar/x'})
    payload['radar']['past'].append({'time': 1, 'path': 'https://evil/x'})
    payload['radar']['past'].append('not a frame')

    parsed = wx.parse_radar_index(payload, now=BASE)

    assert parsed['count'] == 3
    assert parsed['latest'] == newest
    assert all(frame['path'].startswith('/') for frame in parsed['frames'])


def test_a_document_without_usable_frames_is_none():
    for payload in ({}, {'host': 'https://x', 'radar': {}},
                    {'host': 'https://x', 'radar': {'past': []}},
                    {'radar': {'past': [_frame(0)]}},
                    {'host': '  ', 'radar': {'past': [_frame(0)]}},
                    {'host': 'https://x', 'radar': [_frame(0)]},
                    'not a document'):
        assert wx.parse_radar_index(payload, now=BASE) is None


def test_the_age_of_the_newest_frame_is_measured_against_now():
    parsed = wx.parse_radar_index(_index(count=13), now=BASE)
    newest = datetime.fromtimestamp(parsed['frames'][-1]['time'], timezone.utc)

    assert parsed['age_min'] == pytest.approx(5.0)
    assert parsed['age_s'] == pytest.approx(300.0)
    assert parsed['observed_at'] == newest.strftime('%Y-%m-%dT%H:%M:%SZ')


def test_an_old_document_reports_a_negative_age_as_zero():
    """A clock that is behind the service must not produce a past age."""
    parsed = wx.parse_radar_index(_index(count=13), now=BASE - timedelta(hours=1))
    assert parsed['age_min'] == 0.0


def test_the_host_is_normalised_and_the_limits_are_reported():
    parsed = wx.parse_radar_index(
        _index(host='https://tilecache.rainviewer.com/'), now=BASE)
    assert parsed['host'] == 'https://tilecache.rainviewer.com'
    assert parsed['max_zoom'] == wx.RADAR_MAX_ZOOM == 7
    assert parsed['rings_km'] == [50, 100, 200]
    assert parsed['attribution'] == 'Weather data by RainViewer'
    assert parsed['measured'] is True


def test_a_missing_generated_stamp_falls_back_to_the_newest_frame():
    payload = _index(count=2)
    payload.pop('generated')
    parsed = wx.parse_radar_index(payload, now=BASE)
    assert parsed['generated'] == parsed['frames'][-1]['time']


# ---------------------------------------------------------------------------
# The five-minute poll
# ---------------------------------------------------------------------------

def test_the_index_is_requested_once_and_then_served_from_the_cache(monkeypatch):
    stub = _install(monkeypatch)

    first = wx.radar_frames(now=BASE)
    second = wx.radar_frames(now=BASE + timedelta(minutes=4))

    assert stub.count == 1
    assert stub.urls[0] == wx.RADAR_INDEX_URL
    assert first is second
    assert second['latest'] == first['latest']


def test_the_cache_expires_after_five_minutes(monkeypatch):
    stub = _install(monkeypatch)

    wx.radar_frames(now=BASE)
    wx.radar_frames(now=BASE + timedelta(seconds=wx.RADAR_INDEX_TTL_S + 1))

    assert stub.count == 2


def test_a_failed_poll_is_retried_after_a_minute_not_every_rerun(monkeypatch):
    stub = _install(monkeypatch, status=503)

    assert wx.radar_frames(now=BASE) is None
    assert wx.radar_frames(now=BASE + timedelta(seconds=30)) is None
    assert stub.count == 1

    assert wx.radar_frames(now=BASE + timedelta(seconds=61)) is None
    assert stub.count == 2


def test_a_network_error_returns_none_instead_of_raising(monkeypatch):
    _install(monkeypatch, error=wx.requests.ConnectionError('offline'))
    assert wx.radar_frames(now=BASE) is None


def test_a_document_without_frames_is_cached_as_a_failure(monkeypatch):
    stub = _install(monkeypatch, payload={'host': 'https://x', 'radar': {}})
    assert wx.radar_frames(now=BASE) is None
    assert wx.radar_frames(now=BASE + timedelta(seconds=30)) is None
    assert stub.count == 1


# ---------------------------------------------------------------------------
# The view the panel draws
# ---------------------------------------------------------------------------

def test_the_view_is_centred_on_the_circuit(monkeypatch):
    _install(monkeypatch)
    view = wx.radar_view('British Grand Prix', now=BASE)

    assert view['center'] == {'lat': pytest.approx(52.0786, abs=1e-4),
                              'lon': pytest.approx(-1.0169, abs=1e-4)}
    assert view['track_name'] == 'British Grand Prix'
    assert view['frames']
    assert view['host'] == 'https://tilecache.rainviewer.com'


def test_an_unknown_circuit_never_touches_the_network(monkeypatch):
    stub = _install(monkeypatch)
    assert wx.radar_view('Moon Grand Prix', now=BASE) is None
    assert stub.count == 0


def test_the_zoom_is_clamped_to_the_free_tier(monkeypatch):
    _install(monkeypatch)
    assert wx.radar_view('Monza', now=BASE, zoom=12)['zoom'] == wx.RADAR_MAX_ZOOM
    assert wx.radar_view('Monza', now=BASE, zoom=1)['zoom'] == wx.RADAR_MIN_ZOOM


def test_the_view_says_how_much_ground_it_covers(monkeypatch):
    _install(monkeypatch)
    view = wx.radar_view('Monza', now=BASE, zoom=7)
    # z7 over Monza is ~0.85 km a pixel, so a 760 px view is ~650 km.
    assert 0.80 < wx.metres_per_pixel(7, 45.6156) / 1000.0 < 0.90
    assert 600 < view['span_km'] < 700


def test_the_tile_scale_matches_the_published_web_mercator_numbers():
    """z0 is one 256 px tile for the whole world; z7 is 1/128 of it."""
    assert wx.metres_per_pixel(0, 0) == pytest.approx(156543.03392, rel=1e-6)
    assert wx.metres_per_pixel(7, 52.0786) == pytest.approx(752.0, abs=1.0)


# ---------------------------------------------------------------------------
# The template
# ---------------------------------------------------------------------------

def test_the_template_is_shipped_and_has_the_placeholder():
    assert radar.TEMPLATE.exists()
    text = radar.TEMPLATE.read_text(encoding='utf-8')
    assert '__PAYLOAD__' in text
    # The credit line is filled from the payload, so the panel names the
    # sources even before Python fills it.
    assert 'RainViewer' in text
    assert 'OpenStreetMap' in text


def test_the_panel_loads_no_library_and_pulls_only_the_two_tile_hosts():
    text = radar.TEMPLATE.read_text(encoding='utf-8')
    for cdn in ('cdn.jsdelivr.net', 'unpkg.com', 'cdnjs.cloudflare.com',
                'leaflet', 'googleapis'):
        assert cdn not in text.lower(), f'the panel pulls {cdn} from the web'
    # The basemap host is fixed; the radar host arrives in the payload.
    assert 'tile.openstreetmap.org' in text
    assert 'DATA.host' in text


def test_the_panel_names_the_circuit_it_is_centred_on():
    text = radar.TEMPLATE.read_text(encoding='utf-8')
    assert 'id="place"' in text
    assert 'DATA.track_name' in text


def test_the_payload_is_inlined_with_the_script_block_escaped():
    payload = {'track_name': '</script><script>alert(1)</script>',
               'frames': [{'path': '/v2/radar/1', 'time': 1, 'label': '00:00'}],
               'host': 'https://tilecache.rainviewer.com'}
    html = radar.radar_html(payload, template='<script>__PAYLOAD__</script>')

    assert '__PAYLOAD__' not in html
    assert '</script><script>' not in html
    assert '\\u003c/script\\u003e' in html


def test_the_panel_height_lives_with_the_other_sizes():
    assert radar.HEIGHT == styles.RADAR_HEIGHT
    assert isinstance(radar.HEIGHT, int) and radar.HEIGHT > 300


# ---------------------------------------------------------------------------
# The page must centre the map on the race it is showing
# ---------------------------------------------------------------------------

def test_the_radar_follows_the_race_being_read():
    """Reading one race's weather must not draw another circuit's rain.

    The page used to prefer the next session's circuit, so every race showed
    the same map (Austin, while the next session was there).
    """
    assert radar.scope_for('British Grand Prix', 'Austin') == 'British Grand Prix'
    assert radar.scope_for('Japanese Grand Prix', 'Austin') == 'Japanese Grand Prix'


def test_the_next_session_is_only_a_fallback_for_an_unknown_circuit():
    assert radar.scope_for('Moon Grand Prix', 'Austin') == 'Austin'
    assert radar.scope_for('Moon Grand Prix', None) is None
    assert radar.scope_for(None, 'Austin') == 'Austin'
    assert radar.scope_for(None, None) is None


def test_the_page_asks_radar_for_the_scope_instead_of_picking_one_itself():
    source = (ROOT / 'src' / 'pages' / '4_Report.py').read_text(encoding='utf-8')
    assert 'radar.scope_for(track, forecast_place)' in source
    assert 'forecast_place or track' not in source
