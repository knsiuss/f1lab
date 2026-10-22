# -*- coding: utf-8 -*-
"""Tests for the OpenF1 HTTP client (src/openf1_client.py).

The client is shared by the season-results layer and the telemetry ingest, so
its failure behaviour matters twice over: a query that has no rows must not be
retried (the backoff of four attempts costs ~14 s per missing resource), and
a transport error must not be cached as an empty result.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src import openf1_client


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else []
        self.headers = {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f'{self.status_code} error')

    def json(self):
        return self._payload


def test_a_404_is_not_retried(monkeypatch):
    """A missing resource is an empty result, not a transient failure."""
    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(url)
        return _Response(404)

    monkeypatch.setattr(openf1_client.requests, 'get', fake_get)
    monkeypatch.setattr(openf1_client.LIMITER, 'acquire', lambda: None)

    with pytest.raises(openf1_client.NotFound):
        openf1_client.fetch_json('https://example.invalid/x', params={'a': 1})
    assert len(calls) == 1, 'a 404 must not be retried'


def test_openf1_turns_a_404_into_an_empty_list(monkeypatch):
    def fake_get(url, params=None, timeout=None, headers=None):
        return _Response(404)

    monkeypatch.setattr(openf1_client.requests, 'get', fake_get)
    monkeypatch.setattr(openf1_client.LIMITER, 'acquire', lambda: None)
    assert openf1_client.openf1('laps', session_key=1) == []


def test_a_transport_error_is_retried_then_raised(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(url)
        raise openf1_client.requests.ConnectionError('boom')

    monkeypatch.setattr(openf1_client.requests, 'get', fake_get)
    monkeypatch.setattr(openf1_client.LIMITER, 'acquire', lambda: None)
    monkeypatch.setattr(openf1_client.time, 'sleep', lambda s: None)

    with pytest.raises(RuntimeError):
        openf1_client.fetch_json('https://example.invalid/x')
    assert len(calls) == openf1_client.REQUEST_ATTEMPTS


def test_the_response_cache_serves_a_fresh_file(monkeypatch, tmp_path):
    monkeypatch.setattr(openf1_client, 'CACHE_ROOT', tmp_path / 'openf1')
    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(url)
        return _Response(200, [{'driver_number': 1}])

    monkeypatch.setattr(openf1_client.requests, 'get', fake_get)
    monkeypatch.setattr(openf1_client.LIMITER, 'acquire', lambda: None)

    first = openf1_client.cached_json('https://example.invalid/x', params={'a': 1})
    second = openf1_client.cached_json('https://example.invalid/x', params={'a': 1})
    assert first == second == [{'driver_number': 1}]
    assert len(calls) == 1, 'the second read must come from the cache'


def test_a_missing_query_is_not_asked_again_straight_away(monkeypatch, tmp_path):
    monkeypatch.setattr(openf1_client, 'CACHE_ROOT', tmp_path / 'openf1')
    monkeypatch.setattr(openf1_client, '_MISSING_ROOT', tmp_path / 'openf1' / 'missing')
    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append(url)
        return _Response(404)

    monkeypatch.setattr(openf1_client.requests, 'get', fake_get)
    monkeypatch.setattr(openf1_client.LIMITER, 'acquire', lambda: None)

    assert openf1_client.cached_openf1('laps', session_key=7) == []
    assert openf1_client.cached_openf1('laps', session_key=7) == []
    assert len(calls) == 1, 'a query with no rows must not be repeated'


def test_one_rate_limiter_per_process(monkeypatch):
    """The module can be imported under two names; the limiter is shared."""
    import builtins
    monkeypatch.delattr(builtins, openf1_client._LIMITER_ATTRIBUTE,
                        raising=False)
    first = openf1_client._shared_limiter()
    second = openf1_client._shared_limiter()
    assert first is second
