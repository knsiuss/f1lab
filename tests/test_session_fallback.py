# -*- coding: utf-8 -*-
"""Tests for the session fallback in :mod:`shared`.

The official timing archive that FastF1 reads now answers ``401`` without a
subscription, and a full FastF1 load then burns about ten seconds before it
gives up.  Two pieces keep the app quick and honest:

* a cheap probe (:func:`shared._fastf1_archive_reachable`) that asks the
  archive for one small file before the app commits to a full load;
* the OpenF1 fallback (:func:`shared._openf1_session`) that rebuilds the
  session when FastF1 cannot.

The tests never touch the network: the archive and the OpenF1 feed are
monkeypatched.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src import shared


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


@pytest.fixture(autouse=True)
def _clean_probe_state():
    """Every test starts with an unknown archive state."""
    shared._FASTF1_ARCHIVE.update({'available': None, 'checked_at': 0.0})
    shared._FASTF1_STATE.update({'failures': 0, 'blocked_until': 0.0})
    yield
    shared._FASTF1_ARCHIVE.update({'available': None, 'checked_at': 0.0})
    shared._FASTF1_STATE.update({'failures': 0, 'blocked_until': 0.0})


def _fake_session(path='/static/2026/2026-10-11_Singapore_Grand_Prix/'
                       '2026-10-11_Race/'):
    class _Session:
        api_path = path

    return _Session()


def test_probe_says_yes_when_the_archive_answers(monkeypatch):
    monkeypatch.setattr(shared.fastf1, 'get_session',
                        lambda *a, **k: _fake_session())
    monkeypatch.setattr('requests.get',
                        lambda *a, **k: _Response(200))

    assert shared._fastf1_archive_reachable(2026, 'Singapore Grand Prix') is True


def test_probe_says_no_when_every_host_refuses(monkeypatch):
    """A 401 from the official host and a 404 from the mirror means the
    archive is not available, so the app should not spend ten seconds
    discovering that again for every session."""
    monkeypatch.setattr(shared.fastf1, 'get_session',
                        lambda *a, **k: _fake_session())
    monkeypatch.setattr('requests.get',
                        lambda *a, **k: _Response(401))

    assert shared._fastf1_archive_reachable(2026, 'Singapore Grand Prix') is False


def test_probe_keeps_the_answer_for_the_process(monkeypatch):
    """One probe answers for every session: the archive is either there for
    everybody or for nobody."""
    calls = []

    def _get(*args, **kwargs):
        calls.append(args)
        return _Response(401)

    monkeypatch.setattr(shared.fastf1, 'get_session',
                        lambda *a, **k: _fake_session())
    monkeypatch.setattr('requests.get', _get)

    assert shared._fastf1_archive_reachable(2026, 'Singapore Grand Prix') is False
    assert shared._fastf1_archive_reachable(2026, 'Australian Grand Prix') is False
    assert len(calls) == 2  # the two hosts, not four


def test_probe_does_not_block_fastf1_when_the_path_cannot_be_built(monkeypatch):
    """If the session path cannot be built the app cannot tell, so FastF1
    gets its chance instead of being skipped on a guess."""
    def _boom(*args, **kwargs):
        raise RuntimeError('no schedule')

    monkeypatch.setattr(shared.fastf1, 'get_session', _boom)

    assert shared._fastf1_archive_reachable(2026, 'Nowhere Grand Prix') is True


def test_the_openf1_fallback_is_used_when_the_archive_is_closed(monkeypatch):
    """With the archive closed the app must go straight to the OpenF1 feed
    and hand the page an object that says where it came from."""
    sentinel = object()
    monkeypatch.setattr(shared, '_fastf1_archive_reachable',
                        lambda *a, **k: False)
    monkeypatch.setattr(shared, '_openf1_session',
                        lambda *a, **k: sentinel)

    def _never(*args, **kwargs):
        raise AssertionError('FastF1 must not be called')

    monkeypatch.setattr(shared.fastf1, 'get_session', _never)

    assert shared.load_fastf1_session(2026, 'Singapore Grand Prix', 'Race') is sentinel
