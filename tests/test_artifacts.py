# -*- coding: utf-8 -*-
"""Tests for the data-as-artifact loader (:mod:`artifacts`).

Every test is offline: the cache directory is a ``tmp_path`` and the HTTP layer
is a stub that counts its calls, so the suite never touches OpenF1 and never
reads the repository's real ``f1_cache``.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

# Imported the way the app imports them (``from openf1_client import ...``), so
# the modules under test are the same objects a page would use. Importing
# ``src.openf1_client`` here instead would load a second copy of the module and
# patch a cache root the loader never looks at.
import artifacts  # noqa: E402
import openf1_client as client  # noqa: E402

ROWS = [{'driver_number': 1, 'position': 1}]
OTHER_ROWS = [{'driver_number': 44, 'position': 2}]
OLD = 400 * 24 * 3600  # older than any policy in the loader


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class _Response:
    """The part of ``requests.Response`` the client touches."""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {}
        self.text = ''

    def raise_for_status(self):
        if self.status_code >= 400:
            raise client.requests.HTTPError('%d' % self.status_code)

    def json(self):
        return self._payload


class _Stub:
    """A ``requests.get`` that counts calls and replays canned payloads."""

    def __init__(self, *payloads, status=200):
        self.payloads = list(payloads) or [[]]
        self.status = status
        self.count = 0

    def __call__(self, url, params=None, timeout=None, headers=None):
        payload = self.payloads[min(self.count, len(self.payloads) - 1)]
        self.count += 1
        return _Response(payload, self.status)


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    """A private cache directory, with the rate limiter and sleeps removed."""
    root = tmp_path / 'openf1'
    monkeypatch.setattr(client, 'CACHE_ROOT', root)
    monkeypatch.setattr(client, '_MISSING_ROOT', root / 'missing')
    monkeypatch.setattr(artifacts, 'CACHE_ROOT', root)
    monkeypatch.setattr(client.LIMITER, 'acquire', lambda: None)
    monkeypatch.setattr(client.time, 'sleep', lambda seconds: None)
    return root


def _install(monkeypatch, *payloads, status=200):
    """Patch the client's HTTP layer; returns the call counter."""
    stub = _Stub(*payloads, status=status)
    monkeypatch.setattr(client.requests, 'get', stub)
    return stub


def _age(path, seconds):
    """Backdate a cache file."""
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def _write_manifest(cache, sessions):
    index = cache / artifacts.MANIFEST_DIR_NAME
    index.mkdir(parents=True, exist_ok=True)
    payload = {'sessions': sessions, 'generated_at': '2026-03-01T00:00:00+00:00'}
    (index / artifacts.MANIFEST_NAME).write_text(json.dumps(payload), encoding='utf-8')


def _archive_file(cache, endpoint, params, payload, age_s=OLD):
    """Write an archive file through the client's own path, then age it."""
    path = client._cache_path(artifacts.endpoint_url(endpoint), params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')
    _age(path, age_s)
    return path


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

def test_cache_path_matches_the_client(cache):
    """The loader and the client must look at the same file."""
    params = {'session_key': 5}
    assert (artifacts.cache_path('laps', params)
            == client._cache_path(artifacts.endpoint_url('laps'), params))


def test_the_loader_uses_the_app_client_module(cache):
    """One client module, or the loader would read a different cache."""
    assert artifacts.client is client


def test_the_endpoint_url_is_built_from_the_client_base(cache):
    assert artifacts.endpoint_url('pit') == '%s/pit' % client.OPENF1_BASE


# ---------------------------------------------------------------------------
# Freshness policy
# ---------------------------------------------------------------------------

def test_session_finished_waits_for_the_grace_period():
    now = datetime(2026, 3, 1, 15, 0, tzinfo=timezone.utc)
    grace = artifacts.FINISHED_GRACE_S
    assert artifacts.session_finished(
        {'date_end': (now - timedelta(seconds=60)).isoformat()}, now) is False
    assert artifacts.session_finished(
        {'date_end': (now - timedelta(seconds=grace + 60)).isoformat()}, now) is True


def test_a_session_without_dates_is_not_finished():
    assert artifacts.session_finished({}) is False


def test_a_past_year_is_finished(cache):
    assert artifacts.is_finished({'year': 2000}) is True


# ---------------------------------------------------------------------------
# load_json: the four cases the brief names
# ---------------------------------------------------------------------------

def test_missing_file_falls_back_to_the_api(cache, monkeypatch):
    stub = _install(monkeypatch, ROWS)
    assert artifacts.load_json('session_result', session_key=9999) == ROWS
    assert stub.count == 1
    assert artifacts.cache_path('session_result', {'session_key': 9999}).exists()


def test_fresh_file_is_used_without_a_network_call(cache, monkeypatch):
    first = _install(monkeypatch, ROWS)
    assert artifacts.load_json('laps', session_key=7) == ROWS
    assert first.count == 1

    second = _install(monkeypatch, OTHER_ROWS)
    assert artifacts.load_json('laps', session_key=7) == ROWS
    assert second.count == 0


def test_stale_file_triggers_a_refresh(cache, monkeypatch):
    _install(monkeypatch, ROWS)
    artifacts.load_json('pit', session_key=11)
    path = artifacts.cache_path('pit', {'session_key': 11})
    _age(path, artifacts.LIVE_MAX_AGE_S + 60)
    assert artifacts.entry_age_s(path) > artifacts.LIVE_MAX_AGE_S

    stub = _install(monkeypatch, OTHER_ROWS)
    assert artifacts.load_json('pit', session_key=11) == OTHER_ROWS
    assert stub.count == 1


def test_a_404_is_an_empty_result_not_a_crash(cache, monkeypatch):
    stub = _install(monkeypatch, [], status=404)
    assert artifacts.load_json('weather', session_key=3) == []
    assert stub.count == 1
    # The client remembers the miss for a few minutes: no second request.
    assert artifacts.load_json('weather', session_key=3) == []
    assert stub.count == 1


# ---------------------------------------------------------------------------
# Offline mode
# ---------------------------------------------------------------------------

def test_allow_network_false_returns_none_for_a_cold_cache(cache, monkeypatch):
    stub = _install(monkeypatch, ROWS)
    assert artifacts.load_json('drivers', allow_network=False, meeting_key=4) is None
    assert stub.count == 0


def test_allow_network_false_serves_the_stale_copy(cache, monkeypatch):
    _install(monkeypatch, ROWS)
    artifacts.load_json('drivers', session_key=4)
    _age(artifacts.cache_path('drivers', {'session_key': 4}), OLD)

    stub = _install(monkeypatch, OTHER_ROWS)
    assert artifacts.load_json('drivers', allow_network=False, session_key=4) == ROWS
    assert stub.count == 0


# ---------------------------------------------------------------------------
# The manifest decides which files never expire
# ---------------------------------------------------------------------------

def test_a_finished_session_is_served_forever(cache, monkeypatch):
    _archive_file(cache, 'session_result', {'session_key': 9012}, ROWS)
    _write_manifest(cache, {'9012': {'finished': True, 'meeting_key': 100}})

    stub = _install(monkeypatch, OTHER_ROWS)
    assert artifacts.load_json('session_result', session_key=9012) == ROWS
    assert stub.count == 0


def test_a_live_session_is_still_refreshed(cache, monkeypatch):
    _archive_file(cache, 'session_result', {'session_key': 9013}, ROWS)
    _write_manifest(cache, {'9013': {'finished': False, 'meeting_key': 101}})

    stub = _install(monkeypatch, OTHER_ROWS)
    assert artifacts.load_json('session_result', session_key=9013) == OTHER_ROWS
    assert stub.count == 1


def test_a_meeting_query_is_finished_only_when_every_session_is(cache, monkeypatch):
    _archive_file(cache, 'starting_grid', {'meeting_key': 200}, ROWS)
    _archive_file(cache, 'starting_grid', {'meeting_key': 201}, ROWS)

    _write_manifest(cache, {
        '1': {'finished': True, 'meeting_key': 200},
        '2': {'finished': True, 'meeting_key': 200},
        '3': {'finished': True, 'meeting_key': 201},
        '4': {'finished': False, 'meeting_key': 201},
    })

    stub = _install(monkeypatch, OTHER_ROWS)
    assert artifacts.load_json('starting_grid', meeting_key=200) == ROWS
    assert stub.count == 0

    assert artifacts.load_json('starting_grid', meeting_key=201) == OTHER_ROWS
    assert stub.count == 1


def test_without_a_manifest_the_live_rule_applies(cache, monkeypatch):
    _archive_file(cache, 'laps', {'session_key': 55}, ROWS)
    assert artifacts.read_manifest() == {}

    stub = _install(monkeypatch, OTHER_ROWS)
    assert artifacts.load_json('laps', session_key=55) == OTHER_ROWS
    assert stub.count == 1


# ---------------------------------------------------------------------------
# The whole-app hook
# ---------------------------------------------------------------------------

def test_activate_makes_the_archive_win_for_a_finished_session(cache, monkeypatch):
    _archive_file(cache, 'laps', {'session_key': 9012}, ROWS)
    _write_manifest(cache, {'9012': {'finished': True, 'meeting_key': 100}})

    artifacts.activate()
    try:
        stub = _install(monkeypatch, OTHER_ROWS)
        url = artifacts.endpoint_url('laps')
        assert client.cached_json(url, {'session_key': 9012}, ttl=300) == ROWS
        assert stub.count == 0

        # A live session is delegated to the client unchanged.
        _write_manifest(cache, {
            '9012': {'finished': True, 'meeting_key': 100},
            '9013': {'finished': False, 'meeting_key': 101},
        })
        assert client.cached_json(url, {'session_key': 9013}, ttl=300) == OTHER_ROWS
        assert stub.count == 1
    finally:
        artifacts.deactivate()

    assert client.cached_json is not artifacts._cached_json_from_archive


def test_activate_is_idempotent_and_deactivate_restores(cache):
    original = client.cached_json
    artifacts.activate()
    try:
        artifacts.activate()
        assert client.cached_json is artifacts._cached_json_from_archive
    finally:
        artifacts.deactivate()
    assert client.cached_json is original


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def test_cache_status_counts_files_and_bytes(cache):
    cache.mkdir(parents=True)
    for name in ('a', 'b', 'c'):
        (cache / ('%s.json' % name)).write_text('[]', encoding='utf-8')
    (cache / artifacts.MANIFEST_DIR_NAME).mkdir(parents=True)
    (cache / artifacts.MANIFEST_DIR_NAME / artifacts.MANIFEST_NAME).write_text(
        json.dumps({'sessions': {'1': {'finished': True}}}), encoding='utf-8')

    status = artifacts.cache_status()
    assert status['exists'] is True
    assert status['files'] == 3
    assert status['bytes'] == sum((cache / ('%s.json' % name)).stat().st_size
                                  for name in ('a', 'b', 'c'))
    assert status['oldest'] and status['newest']
    assert status['sessions_indexed'] == 1
    assert status['archive_active'] is False


def test_cache_status_of_a_cold_cache(cache):
    status = artifacts.cache_status()
    assert status['exists'] is False
    assert status['files'] == 0
    assert status['bytes'] == 0
    assert status['oldest'] is None
    assert status['newest'] is None


def test_cache_status_reports_last_run(cache):
    index = cache / artifacts.MANIFEST_DIR_NAME
    index.mkdir(parents=True)
    (index / artifacts.MANIFEST_NAME).write_text(json.dumps({
        'sessions': {'1': {'finished': True}, '2': {'finished': True}},
        'generated_at': '2026-10-23T03:17:00+00:00',
        'last_run': {'years': [2026], 'requests': 21, 'cache_hits': 4},
        'endpoints': {'laps': {'requests': 3, 'rows': 1200}},
    }), encoding='utf-8')

    status = artifacts.cache_status()
    assert status['sessions_indexed'] == 2
    assert status['manifest_generated_at'] == '2026-10-23T03:17:00+00:00'
    assert status['last_run']['requests'] == 21
    assert status['endpoints']['laps']['rows'] == 1200


def test_the_manifest_round_trips_through_write(cache):
    assert artifacts.write_manifest({'sessions': {'1': {'finished': True}}}) is not None
    assert artifacts.read_manifest()['sessions']['1']['finished'] is True
