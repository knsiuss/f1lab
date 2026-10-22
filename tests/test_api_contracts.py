# -*- coding: utf-8 -*-
"""
test_api_contracts.py
~~~~~~~~~~~~~~~~~~~~~~
Offline contract tests for the OpenF1 payloads this app depends on.

The project has already been broken twice by upstream behaviour changes -- a
sprint session arriving as ``session_type='Race'`` with ``session_name='Sprint'``,
and 404 responses that used to be retried for 90 s -- so the payload shape is
pinned here: golden fixtures in ``tests/fixtures/`` are the reference, and
``tools/contract_check.py --offline`` validates them. Nothing in this module
touches the network: the live comparison is the tool's ``--live`` mode.

The contract (which fields the app reads and their types) lives in
``tools/contract_check.ENDPOINTS`` -- one source of truth shared by the tool and
these tests, so a new endpoint used in ``src/`` without a fixture fails here.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / 'tests' / 'fixtures'
SRC = ROOT / 'src'
TOOLS = ROOT / 'tools'
for _path in (str(ROOT), str(SRC), str(TOOLS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import contract_check  # noqa: E402  (tool on sys.path)
import openf1_client  # noqa: E402

from src import openf1_results as ofr  # noqa: E402
from src import openf1_session as ofs  # noqa: E402
from src import telemetry  # noqa: E402

# The endpoints the app uses, as a module-level constant. The schema itself is
# owned by the contract tool; this test module only consumes it.
ENDPOINTS = contract_check.ENDPOINTS

#: Endpoint literals a module may pass to the shared client.
_ENDPOINT_CALL_RE = re.compile(r"(?:cached_openf1|openf1)\(\s*['\"]([a-z][a-z_]+)['\"]")
_TUPLE_RE = re.compile(r"SESSION_ENDPOINTS\s*=\s*\(([^)]*)\)", re.S)

_CREDENTIAL_RE = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer|access[_-]?token|client[_-]?secret|password)")


def load(endpoint: str):
    """The fixture payload of one endpoint (a list of objects)."""
    return json.loads((FIXTURES / f'{endpoint}.json').read_text(encoding='utf-8'))


def _session(session_key: int):
    return next(row for row in load('sessions') if row['session_key'] == session_key)


def _driver_map():
    return {int(row['driver_number']): row for row in load('drivers')}


def _fake_cached(endpoint, ttl=None, **params):
    return load(endpoint)


# ---------------------------------------------------------------------------
# Coverage: a new endpoint cannot be added without a fixture
# ---------------------------------------------------------------------------
def test_every_contracted_endpoint_has_a_fixture():
    missing = [name for name in ENDPOINTS
               if not (FIXTURES / f'{name}.json').exists()]
    assert not missing, f'missing fixtures for {sorted(missing)}'


def test_no_fixture_lacks_a_contract():
    orphans = contract_check.orphan_fixtures(FIXTURES)
    assert not orphans, f'fixtures without an ENDPOINTS contract: {orphans}'


def test_fixture_readme_documents_every_endpoint():
    readme = (FIXTURES / 'README.md').read_text(encoding='utf-8')
    undocumented = [name for name in ENDPOINTS if f'`{name}`' not in readme]
    assert not undocumented, f'fixtures/README.md misses {undocumented}'


def test_src_endpoint_literals_are_all_contracted():
    """A new endpoint call in src/ must come with a contract and a fixture."""
    used = set()
    for path in SRC.rglob('*.py'):
        text = path.read_text(encoding='utf-8')
        used.update(_ENDPOINT_CALL_RE.findall(text))
        for block in _TUPLE_RE.findall(text):
            used.update(re.findall(r"['\"]([a-z][a-z_]+)['\"]", block))
    unknown = sorted(used - set(ENDPOINTS))
    assert not unknown, f'src/ uses uncontracted endpoint(s): {unknown}'


def test_contract_specs_are_well_formed():
    valid = re.compile(r'^(?:int|number|str|bool|list|object)\??$|^gap$')
    for endpoint, schema in ENDPOINTS.items():
        assert schema, f'{endpoint} has an empty contract'
        bad = {field: spec for field, spec in schema.items() if not valid.match(spec)}
        assert not bad, f'{endpoint} has invalid type spec(s): {bad}'


# ---------------------------------------------------------------------------
# Fixture shape
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('endpoint', sorted(ENDPOINTS))
def test_fixture_is_a_small_list_of_objects(endpoint):
    rows = load(endpoint)
    assert isinstance(rows, list), f'{endpoint} is not a JSON list (paging shape)'
    assert rows, f'{endpoint} fixture is empty'
    assert len(rows) <= contract_check.MAX_ROWS, (
        f'{endpoint} keeps {len(rows)} rows, fixtures must stay small')
    assert all(isinstance(row, dict) for row in rows), (
        f'{endpoint} must page a list of objects')


@pytest.mark.parametrize('endpoint', sorted(ENDPOINTS))
def test_fixture_meets_the_contract(endpoint):
    problems = contract_check.check_fixture(endpoint, load(endpoint))
    assert not problems, f'{endpoint} breaks the contract: {problems}'


def test_fixtures_contain_no_credentials():
    for path in FIXTURES.glob('*.json'):
        text = path.read_text(encoding='utf-8')
        assert not _CREDENTIAL_RE.search(text), f'{path.name} looks like it stores a secret'


def test_fixture_urls_are_public_and_https():
    for path in FIXTURES.glob('*.json'):
        for url in re.findall(r'https?://[^"\s]+', path.read_text(encoding='utf-8')):
            assert url.startswith('https://'), f'{path.name} has a plain-http URL'
            assert '.formula1.com' in url or 'multiviewer.app' in url, (
                f'{path.name} has an unexpected URL: {url}')


# ---------------------------------------------------------------------------
# The sprint must stay identifiable (session_type='Race', session_name='Sprint')
# ---------------------------------------------------------------------------
def test_sprint_is_identifiable_and_separated_from_the_grand_prix(monkeypatch):
    sessions = load('sessions')
    sprints = [row for row in sessions if row['session_name'] == 'Sprint']
    assert sprints, 'the fixture set must contain a sprint session'
    assert all(row['session_type'] == 'Race' for row in sprints), (
        'OpenF1 still reports a sprint as session_type=Race')
    assert all(ofr._is_sprint(row) for row in sprints)
    assert not any(ofr._is_sprint(row) for row in sessions
                   if row['session_name'] != 'Sprint')

    monkeypatch.setattr(ofr, 'season_sessions', lambda year: sessions)
    assert [row['session_key'] for row in ofr.race_sessions(2025)] == [9693, 9998]
    assert [row['session_key'] for row in ofr.sprint_sessions(2025)] == [9993]


def test_is_sprint_tolerates_whitespace_and_case():
    assert ofr._is_sprint({'session_name': ' Sprint '})
    assert ofr._is_sprint({'session_name': 'SPRINT'})
    assert not ofr._is_sprint({'session_name': 'Sprint Qualifying'})
    assert not ofr._is_sprint({})


# ---------------------------------------------------------------------------
# openf1_results parsers accept the fixtures
# ---------------------------------------------------------------------------
def test_result_helpers_accept_the_fixtures(monkeypatch):
    monkeypatch.setattr(ofr, 'cached_openf1', _fake_cached)

    result = ofr.session_result(9998)
    assert len(result) == len(load('session_result'))

    grid = ofr.starting_grid(1255)
    assert grid and all(isinstance(k, int) and isinstance(v, int)
                        for k, v in grid.items())

    drivers = ofr.drivers_for_meeting(1255)
    assert drivers and all(isinstance(k, int) for k in drivers)

    laps = ofr.fastest_laps_for_session(9998)
    assert isinstance(laps, dict)


def test_driver_names_resolve_for_every_driver_in_the_fixture():
    for row in load('drivers'):
        assert ofr._driver_name(row).strip(), row
        assert ofs._driver_name(row).strip(), row


def test_session_result_parser_handles_every_gap_shape(monkeypatch):
    result = load('session_result')
    gaps = {type(row['gap_to_leader']).__name__ for row in result}
    assert 'NoneType' in gaps, 'the fixture must include a DNS-style null gap'
    assert 'str' in gaps, "the fixture must include a '+1 LAP' string gap"

    monkeypatch.setattr(ofr, 'session_result', lambda *a, **kw: result)
    monkeypatch.setattr(ofr, 'starting_grid', lambda *a, **kw: {})
    session = _session(9998)
    event = {'round': 2, 'name': 'Chinese Grand Prix'}
    rows = ofr.build_session_rows(session, event, 'Race', drivers=_driver_map())

    assert len(rows) == len(result)
    assert set(ofr.RESULT_COLUMNS) <= set(rows[0])
    assert all(row['Track'] == 'Chinese Grand Prix' for row in rows)
    assert all(row['SessionType'] == 'Race' for row in rows)
    assert {row['Status'] for row in rows} <= {'Finished', 'DNF', 'DNS', 'DSQ'}
    assert any(row['Time/Retired'] == '+1 LAP' for row in rows), (
        'the +1 LAP gap must survive as text')
    assert any(row['Position'] is None for row in rows), 'a DNS row has no position'


def test_build_season_results_accepts_the_fixtures_without_a_calendar(monkeypatch):
    sessions = load('sessions')
    grid = {int(row['driver_number']): int(row['position'])
            for row in load('starting_grid')}
    drivers = _driver_map()
    monkeypatch.setattr(ofr, 'season_sessions', lambda year: sessions)
    monkeypatch.setattr(ofr, 'session_result', lambda *a, **kw: load('session_result'))
    monkeypatch.setattr(ofr, 'starting_grid', lambda *a, **kw: grid)
    monkeypatch.setattr(ofr, 'drivers_for_meeting', lambda *a, **kw: drivers)
    # No FastF1 calendar (the archive is closed): the frame must still build.
    monkeypatch.setattr(ofr, '_calendar_events', lambda year: [])

    frame = ofr.build_season_results(2025)

    assert frame is not None
    assert list(frame.columns) == ofr.RESULT_COLUMNS
    assert set(frame['SessionType']) == {'Race', 'Sprint'}
    assert len(frame) == 3 * len(load('session_result'))  # 2 races + 1 sprint
    # Without a calendar the rows are still kept (never invented away); only
    # the event label is missing, so the caller can degrade honestly.
    assert frame['Track'].notna().all()
    classified = sum(1 for row in load('session_result')
                     if row['position'] is not None)
    assert int(frame['Position'].notna().sum()) == 3 * classified


def test_build_season_results_uses_the_calendar_when_it_exists(monkeypatch):
    sessions = load('sessions')
    grid = {int(row['driver_number']): int(row['position'])
            for row in load('starting_grid')}
    events = [
        {'round': 1, 'name': 'Australian Grand Prix', 'date': datetime(2025, 3, 16)},
        {'round': 2, 'name': 'Chinese Grand Prix', 'date': datetime(2025, 3, 23)},
    ]
    monkeypatch.setattr(ofr, 'season_sessions', lambda year: sessions)
    monkeypatch.setattr(ofr, 'session_result', lambda *a, **kw: load('session_result'))
    monkeypatch.setattr(ofr, 'starting_grid', lambda *a, **kw: grid)
    monkeypatch.setattr(ofr, 'drivers_for_meeting', lambda *a, **kw: _driver_map())
    monkeypatch.setattr(ofr, '_calendar_events', lambda year: events)

    frame = ofr.build_season_results(2025)

    assert set(frame['Track']) == {'Australian Grand Prix', 'Chinese Grand Prix'}
    assert set(frame['Round'].dropna().astype(int)) == {1, 2}
    sprint = frame[frame['SessionType'] == 'Sprint']
    assert set(sprint['Track']) == {'Chinese Grand Prix'}


# ---------------------------------------------------------------------------
# openf1_session fallback rebuilds a FastF1-shaped frame from the fixtures
# ---------------------------------------------------------------------------
def test_openf1_session_builds_laps_from_the_fixtures(monkeypatch):
    monkeypatch.setattr(ofs, 'cached_openf1', _fake_cached)

    laps = ofs.build_laps(9998)

    assert not laps.empty
    for column in ('DriverNumber', 'LapNumber', 'LapTime', 'Sector1Time',
                   'Driver', 'Team', 'Compound', 'Stint', 'TyreLife',
                   'IsOutLap', 'IsInLap', 'PitInTime', 'PitOutTime', 'Position'):
        assert column in laps.columns
    assert laps['LapTime'].notna().all()
    assert laps.attrs['position_source'] == 'estimated'
    assert laps.attrs['track_status'] == 'unavailable'
    assert set(laps['Compound']) <= set(ofs.COMPOUNDS.values()) | {'UNKNOWN'}


def test_openf1_session_builds_results_from_the_fixtures(monkeypatch):
    monkeypatch.setattr(ofs, 'cached_openf1', _fake_cached)

    results = ofs.build_results(9998)

    assert len(results) == len(load('session_result'))
    for column in ('Position', 'DriverNumber', 'Abbreviation', 'Driver',
                   'Team', 'Laps', 'Points', 'Status', 'GapToLeader'):
        assert column in results.columns
    assert {'Finished', 'DNF', 'DNS', 'DSQ'} >= set(results['Status'])


# ---------------------------------------------------------------------------
# telemetry ingest helpers accept the fixtures
# ---------------------------------------------------------------------------
def test_telemetry_session_endpoints_are_contracted():
    assert set(telemetry.SESSION_ENDPOINTS) <= set(ENDPOINTS)


def test_telemetry_row_builders_accept_the_fixtures():
    laps = load('laps')
    origin = pd.Timestamp(laps[0]['date_start']) - pd.Timedelta(seconds=5)

    lap_rows = telemetry._lap_rows(laps, origin)
    assert len(lap_rows) == len(laps)
    assert len(lap_rows[0]) == 11  # driver, lap, t, time, 3 sectors, 3 speeds, out-lap

    position_rows = telemetry._sample_rows(load('position'), origin, ['position'])
    assert len(position_rows) == len(load('position'))

    car_rows = telemetry._sample_rows(
        load('car_data'), origin,
        ['speed', 'throttle', 'brake', 'drs', 'rpm', 'n_gear'])
    assert len(car_rows) == len(load('car_data'))

    pit_rows = telemetry._pit_rows(load('pit'), origin)
    assert len(pit_rows) == len(load('pit'))
    assert all(row[3] is not None for row in pit_rows)

    interval_rows = telemetry._interval_rows(load('intervals'), origin)
    assert len(interval_rows) == len(load('intervals'))
    assert any(isinstance(row[2], str) for row in interval_rows), (
        "the '+1 LAP' gap must pass through the ingest unchanged")

    control_rows = telemetry._control_rows(load('race_control'), 9998, origin)
    assert len(control_rows) == len(load('race_control'))
    assert control_rows == sorted(control_rows, key=lambda row: row[1])
    assert {row[3] for row in control_rows} <= {'Flag', 'Other', 'Drs', 'SessionStatus'}


# ---------------------------------------------------------------------------
# The contract tool itself
# ---------------------------------------------------------------------------
def test_contract_check_offline_is_green(capsys):
    assert contract_check.main(['--offline']) == 0
    assert 'RESULT: PASS' in capsys.readouterr().out


def test_contract_check_offline_fails_when_a_fixture_is_missing(tmp_path, capsys):
    for path in FIXTURES.glob('*.json'):
        if path.name != 'laps.json':
            (tmp_path / path.name).write_text(
                path.read_text(encoding='utf-8'), encoding='utf-8')
    assert contract_check.main(['--offline', '--fixtures', str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert 'FAIL laps' in out
    assert 'missing fixture laps.json' in out


def test_contract_check_flags_a_removed_required_field():
    fixture = load('sessions')
    live = [{k: v for k, v in row.items() if k != 'session_name'} for row in fixture]

    diff = contract_check.diff_endpoint('sessions', live, fixture)

    assert diff['breaking']
    assert any('session_name' in problem for problem in diff['breaking'])
    assert 'session_name' in diff['removed']


def test_contract_check_does_not_fail_on_a_removed_optional_field():
    fixture = load('sessions')
    live = [{k: v for k, v in row.items() if k != 'location'} for row in fixture]

    diff = contract_check.diff_endpoint('sessions', live, fixture)

    assert 'location' in diff['removed']
    assert not diff['breaking']


def test_contract_check_flags_a_retyped_required_field():
    fixture = load('session_result')
    live = [dict(row, points=str(row['points'])) for row in fixture]

    diff = contract_check.diff_endpoint('session_result', live, fixture)

    assert diff['breaking']
    assert any('points' in problem for problem in diff['breaking'])
    assert diff['changed'][0]['field'] == 'points'


def test_contract_check_tolerates_added_fields():
    fixture = load('weather')
    live = [dict(row, new_field='x') for row in fixture]

    diff = contract_check.diff_endpoint('weather', live, fixture)

    assert diff['added'] == ['new_field']
    assert not diff['breaking']


def test_contract_check_live_mode_uses_the_shared_client(monkeypatch):
    calls = []

    def fake_openf1(endpoint, **params):
        calls.append((endpoint, params))
        return load(endpoint)

    monkeypatch.setattr(openf1_client, 'openf1', fake_openf1)
    assert contract_check.live_client() is fake_openf1

    rows = contract_check.fetch_endpoint('sessions', {'year': 2025})

    assert calls == [('sessions', {'year': 2025})]
    assert rows == load('sessions')
    # …and the default query is the one the fixture was captured with.
    contract_check.fetch_endpoint('car_data')
    assert calls[-1] == ('car_data', {'session_key': 9998, 'driver_number': 1})
