# -*- coding: utf-8 -*-
"""
contract_check.py
~~~~~~~~~~~~~~~~~~
Guard the OpenF1 payload shapes this app depends on.

The project has already been broken twice by upstream behaviour changes (a
sprint arriving as ``session_type='Race'`` with ``session_name='Sprint'``; a
404 that used to be retried for 90 s; the FastF1 timing archive closing), and
the expensive failure mode is a page that renders an empty column instead of
raising. This tool turns the payload shape into an explicit contract:

* ``--offline`` (the CI mode) validates the golden fixtures in
  ``tests/fixtures/`` against :data:`ENDPOINTS`: no network, no API key, exit
  1 on a fixture that lost a field the app reads or changed its type.
* ``--live`` fetches the query each fixture was captured with and prints the
  added / removed / changed fields per endpoint. It exits 1 only when the live
  payload breaks a field the app depends on (a removed or retyped required
  field), not on a harmless addition.

:data:`ENDPOINTS` is the single source of truth for the contract:
``tests/test_api_contracts.py`` imports it, so a new endpoint in ``src/`` or a
new fixture without a contract fails the test suite.

Live requests go through ``src/openf1_client.py`` -- one process-wide rate
limiter (3/s, 30/min) and the shared retry/404 policy -- never a private copy.

Usage::

    PYTHONPATH=src .venv/Scripts/python.exe tools/contract_check.py --offline
    PYTHONPATH=src .venv/Scripts/python.exe tools/contract_check.py --live
    PYTHONPATH=src .venv/Scripts/python.exe tools/contract_check.py --live --year 2024

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src'
FIXTURES_DIR = ROOT / 'tests' / 'fixtures'

#: Fixtures must stay small: they are reviewed and diffed like code.
MAX_ROWS = 40

# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------
# endpoint -> field -> type spec. A spec is one of
#   int, number, str, bool, list, object   -- required, must not be null
#   <spec>?                                -- nullable
#   gap                                    -- OpenF1 reports gaps as a number,
#                                             as '+1 LAP' or as null
# Only fields the app (src/) actually reads are listed, so an unrelated field
# disappearing upstream is reported as a warning, not a failure.
ENDPOINTS: Dict[str, Dict[str, str]] = {
    'sessions': {
        'session_key': 'int',
        'meeting_key': 'int',
        'session_type': 'str',
        'session_name': 'str',
        'date_start': 'str',
        'date_end': 'str',
        'circuit_key': 'int',
        'circuit_short_name': 'str',
        'country_name': 'str',
        'year': 'int',
        'is_cancelled': 'bool',
    },
    'meetings': {
        'meeting_key': 'int',
        'meeting_name': 'str',
        'meeting_official_name': 'str',
        'location': 'str',
        'country_name': 'str',
        'country_code': 'str',
        'circuit_key': 'int',
        'circuit_short_name': 'str',
        'date_start': 'str',
        'date_end': 'str',
        'year': 'int',
        'is_cancelled': 'bool',
    },
    'session_result': {
        'session_key': 'int',
        'meeting_key': 'int',
        'driver_number': 'int',
        'position': 'int?',
        'number_of_laps': 'int?',
        'points': 'number',
        'dnf': 'bool',
        'dns': 'bool',
        'dsq': 'bool',
        'duration': 'number?',
        # float, int 0, '+1 LAP' or null -- the parser must accept all four.
        'gap_to_leader': 'gap',
    },
    'starting_grid': {
        'session_key': 'int',
        'meeting_key': 'int',
        'driver_number': 'int',
        'position': 'int',
        'lap_duration': 'number?',
    },
    'drivers': {
        'session_key': 'int',
        'meeting_key': 'int',
        'driver_number': 'int',
        'broadcast_name': 'str',
        'full_name': 'str',
        'first_name': 'str',
        'last_name': 'str',
        'name_acronym': 'str',
        'team_name': 'str',
        'team_colour': 'str',
        'country_code': 'str?',
        'headshot_url': 'str?',
    },
    'laps': {
        'session_key': 'int',
        'meeting_key': 'int',
        'driver_number': 'int',
        'lap_number': 'int',
        'date_start': 'str',
        'lap_duration': 'number?',
        'duration_sector_1': 'number?',
        'duration_sector_2': 'number?',
        'duration_sector_3': 'number?',
        'st_speed': 'int?',
        'i1_speed': 'int?',
        'i2_speed': 'int?',
        'is_pit_out_lap': 'bool',
        'segments_sector_1': 'list?',
        'segments_sector_2': 'list?',
        'segments_sector_3': 'list?',
    },
    'stints': {
        'session_key': 'int',
        'meeting_key': 'int',
        'driver_number': 'int',
        'stint_number': 'int',
        'lap_start': 'int',
        'lap_end': 'int?',
        'compound': 'str',
        'tyre_age_at_start': 'int?',
    },
    'pit': {
        'session_key': 'int',
        'meeting_key': 'int',
        'driver_number': 'int',
        'lap_number': 'int',
        'date': 'str',
        'pit_duration': 'number?',
        'lane_duration': 'number?',
        'stop_duration': 'number?',
    },
    'weather': {
        'session_key': 'int',
        'meeting_key': 'int',
        'date': 'str',
        'air_temperature': 'number',
        'track_temperature': 'number',
        'humidity': 'number',
        'pressure': 'number',
        'rainfall': 'number',
        'wind_direction': 'number',
        'wind_speed': 'number',
    },
    'intervals': {
        'session_key': 'int',
        'meeting_key': 'int',
        'driver_number': 'int',
        'date': 'str',
        'gap_to_leader': 'gap',
        'interval': 'number?',
    },
    'race_control': {
        'session_key': 'int',
        'meeting_key': 'int',
        'date': 'str',
        'lap_number': 'int?',
        'category': 'str',
        'flag': 'str?',
        'scope': 'str?',
        'sector': 'int?',
        'qualifying_phase': 'int?',
        'driver_number': 'int?',
        'message': 'str',
    },
    'position': {
        'session_key': 'int',
        'meeting_key': 'int',
        'driver_number': 'int',
        'date': 'str',
        'position': 'int',
    },
    'car_data': {
        'session_key': 'int',
        'meeting_key': 'int',
        'driver_number': 'int',
        'date': 'str',
        'speed': 'int',
        'throttle': 'int',
        'brake': 'int',
        'drs': 'int',
        'n_gear': 'int',
        'rpm': 'int',
    },
}

#: The query each fixture was captured with; ``--live`` replays exactly this.
QUERIES: Dict[str, Dict[str, Any]] = {
    'sessions': {'year': 2025},
    'meetings': {'year': 2025},
    'session_result': {'session_key': 9998},
    'starting_grid': {'meeting_key': 1255},
    'drivers': {'meeting_key': 1255},
    'laps': {'session_key': 9998},
    'stints': {'session_key': 9998},
    'pit': {'session_key': 9998},
    'weather': {'session_key': 9998},
    'intervals': {'session_key': 9998},
    'race_control': {'session_key': 9998},
    'position': {'session_key': 9998},
    'car_data': {'session_key': 9998, 'driver_number': 1},
}

_SPEC_TYPES: Dict[str, Tuple[str, ...]] = {
    'int': ('number',),
    'number': ('number',),
    'str': ('str',),
    'bool': ('bool',),
    'list': ('list',),
    'object': ('object',),
}


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------
def type_name(value: Any) -> str:
    """JSON-ish type name, with ``bool`` kept apart from ``int``."""
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'bool'
    if isinstance(value, (int, float)):
        return 'number'
    if isinstance(value, str):
        return 'str'
    if isinstance(value, list):
        return 'list'
    if isinstance(value, dict):
        return 'object'
    return type(value).__name__


def matches(value: Any, spec: str) -> bool:
    """True when *value* satisfies the contract *spec*."""
    actual = type_name(value)
    if spec == 'gap':
        return actual in ('null', 'number', 'str')
    if actual == 'null':
        return spec.endswith('?')
    return actual in _SPEC_TYPES[spec.rstrip('?')]


def row_problems(name: str, rows: Sequence[Any]) -> List[str]:
    """Every way *rows* breaks the contract of *name* (bounded output)."""
    schema = ENDPOINTS[name]
    problems: List[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f'row {index}: {type_name(row)}, expected object')
            continue
        missing = [field for field in schema if field not in row]
        if missing:
            problems.append(f'row {index}: missing field(s) {", ".join(missing)}')
            continue
        wrong = [
            f'{field}={row[field]!r} ({type_name(row[field])}), expected {spec}'
            for field, spec in schema.items()
            if not matches(row[field], spec)
        ]
        if wrong:
            problems.append(f'row {index}: ' + '; '.join(wrong))
    return problems


def field_types(rows: Sequence[Any]) -> Dict[str, set]:
    """field -> set of JSON type names seen in *rows*."""
    seen: Dict[str, set] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for field, value in row.items():
            seen.setdefault(field, set()).add(type_name(value))
    return seen


# ---------------------------------------------------------------------------
# Offline: fixtures against the contract
# ---------------------------------------------------------------------------
def fixture_path(name: str, directory: Optional[Path] = None) -> Path:
    return Path(directory or FIXTURES_DIR) / f'{name}.json'


def load_fixture(name: str, directory: Optional[Path] = None) -> List[Any]:
    return json.loads(fixture_path(name, directory).read_text(encoding='utf-8'))


def check_fixture(name: str, rows: Any) -> List[str]:
    """Validate one fixture payload. Returns [] when it is a usable golden sample."""
    if not isinstance(rows, list):
        return ['not a JSON list -- OpenF1 pages with a list of objects']
    problems: List[str] = []
    if not rows:
        return ['fixture has no rows']
    if len(rows) > MAX_ROWS:
        problems.append(f'{len(rows)} rows, keep fixtures <= {MAX_ROWS}')
    problems.extend(row_problems(name, rows))
    return problems


def check_all(directory: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Validate every contracted endpoint's fixture: name -> {rows, problems}."""
    results: Dict[str, Dict[str, Any]] = {}
    for name in ENDPOINTS:
        path = fixture_path(name, directory)
        if not path.exists():
            results[name] = {'rows': 0, 'problems': [f'missing fixture {path.name}']}
            continue
        try:
            rows = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            results[name] = {'rows': 0, 'problems': [f'invalid JSON: {exc}']}
            continue
        results[name] = {
            'rows': len(rows) if isinstance(rows, list) else 0,
            'problems': check_fixture(name, rows),
        }
    return results


def orphan_fixtures(directory: Optional[Path] = None) -> List[str]:
    """JSON files in the fixture directory that have no entry in the contract."""
    directory = Path(directory or FIXTURES_DIR)
    if not directory.is_dir():
        return []
    return sorted(
        path.stem for path in directory.glob('*.json') if path.stem not in ENDPOINTS
    )


# ---------------------------------------------------------------------------
# Live: the API against the fixture
# ---------------------------------------------------------------------------
def live_client():
    """The shared OpenF1 client (one rate limiter for the whole process)."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from openf1_client import openf1  # local import: offline mode never needs it
    return openf1


def query_for(name: str,
              year: Optional[int] = None,
              session_key: Optional[int] = None,
              meeting_key: Optional[int] = None) -> Dict[str, Any]:
    """The fixture's capture query, with the matching CLI override applied."""
    params = dict(QUERIES[name])
    for key, override in (('year', year),
                          ('session_key', session_key),
                          ('meeting_key', meeting_key)):
        if key in params and override is not None:
            params[key] = override
    return params


def fetch_endpoint(name: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
    """Fetch one endpoint through the shared client (rate limited, 404 -> [])."""
    kwargs = QUERIES[name] if params is None else params
    return live_client()(name, **kwargs) or []


def diff_endpoint(name: str,
                  live_rows: Sequence[Any],
                  fixture_rows: Sequence[Any]) -> Dict[str, Any]:
    """Field-level difference between a live response and its fixture."""
    schema = ENDPOINTS[name]
    live_types = field_types(live_rows)
    fixture_types = field_types(fixture_rows)
    added = sorted(set(live_types) - set(fixture_types))
    removed = sorted(set(fixture_types) - set(live_types))
    changed: List[Dict[str, str]] = []
    for field in sorted(set(live_types) & set(fixture_types)):
        before = sorted(fixture_types[field] - {'null'})
        after = sorted(live_types[field] - {'null'})
        if before != after:
            changed.append({
                'field': field,
                'fixture': '|'.join(before) or 'null',
                'live': '|'.join(after) or 'null',
            })

    breaking: List[str] = []
    if not live_rows:
        breaking.append('endpoint returned no rows (the fixture is not empty)')
    for field in removed:
        if field in schema:
            breaking.append(f"required field '{field}' disappeared")
    breaking.extend(row_problems(name, live_rows))
    return {
        'endpoint': name,
        'rows': len(live_rows),
        'added': added,
        'removed': removed,
        'changed': changed,
        # de-duplicated, order preserved
        'breaking': list(dict.fromkeys(breaking)),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _trim(items: Sequence[str], limit: int = 8) -> List[str]:
    if len(items) <= limit:
        return list(items)
    return list(items[:limit]) + [f'... and {len(items) - limit} more']


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def run_offline(names: Sequence[str],
                directory: Optional[Path] = None,
                as_json: bool = False) -> int:
    """Validate fixtures only. Returns the process exit code."""
    results = check_all(directory)
    selected = {name: results[name] for name in names}
    orphans = orphan_fixtures(directory)
    failed = [name for name, result in selected.items() if result['problems']]
    ok = not failed and not orphans

    if as_json:
        print(json.dumps({
            'mode': 'offline',
            'ok': ok,
            'fixtures': _rel(Path(directory or FIXTURES_DIR)),
            'endpoints': selected,
            'orphans': orphans,
        }, indent=2))
        return 0 if ok else 1

    print(f'contract_check: offline -- {len(selected)} endpoint(s), '
          f'fixtures in {_rel(Path(directory or FIXTURES_DIR))}')
    for name in names:
        result = selected[name]
        if result['problems']:
            print(f'  FAIL {name}')
            for problem in _trim(result['problems']):
                print(f'       {problem}')
        else:
            print(f'  ok   {name:14s} {result["rows"]:2d} row(s)')
    for name in orphans:
        print(f'  FAIL orphan fixture {name}.json has no contract in ENDPOINTS')
    passed = len(selected) - len(failed)
    verdict = 'PASS' if ok else 'FAIL'
    print(f'RESULT: {verdict} {passed}/{len(selected)} fixture(s) valid')
    return 0 if ok else 1


def run_live(names: Sequence[str],
             directory: Optional[Path] = None,
             year: Optional[int] = None,
             session_key: Optional[int] = None,
             meeting_key: Optional[int] = None,
             as_json: bool = False) -> int:
    """Compare live responses with the fixtures. Returns the process exit code."""
    diffs: List[Dict[str, Any]] = []
    for name in names:
        try:
            fixture_rows = load_fixture(name, directory)
        except (OSError, json.JSONDecodeError) as exc:
            print(f'{name}: cannot read fixture ({exc})', file=sys.stderr)
            return 2
        params = query_for(name, year, session_key, meeting_key)
        print(f'contract_check: live {name} {params}', file=sys.stderr)
        try:
            live_rows = fetch_endpoint(name, params)
        except Exception as exc:  # network failure is a failed check, not a crash
            diffs.append({
                'endpoint': name,
                'rows': 0,
                'added': [],
                'removed': [],
                'changed': [],
                'breaking': [f'live request failed: {exc}'],
            })
            continue
        diffs.append(diff_endpoint(name, live_rows, fixture_rows))

    breaking = [diff for diff in diffs if diff['breaking']]
    ok = not breaking

    if as_json:
        print(json.dumps({'mode': 'live', 'ok': ok, 'endpoints': diffs}, indent=2))
        return 0 if ok else 1

    print(f'contract_check: live -- {len(diffs)} endpoint(s) vs '
          f'{_rel(Path(directory or FIXTURES_DIR))}')
    for diff in diffs:
        name = diff['endpoint']
        status = 'FAIL' if diff['breaking'] else 'ok'
        print(f'  {status} {name:14s} {diff["rows"]:5d} row(s)')
        if diff['added']:
            print(f'       added:   {", ".join(diff["added"])}')
        if diff['removed']:
            required = [f for f in diff['removed'] if f in ENDPOINTS[name]]
            label = 'removed (required)' if required else 'removed'
            print(f'       {label}: {", ".join(diff["removed"])}')
        for change in diff['changed']:
            print(f'       changed: {change["field"]}: '
                  f'{change["fixture"]} -> {change["live"]}')
        for problem in _trim(diff['breaking']):
            print(f'       BREAKING {problem}')
    verdict = 'PASS' if ok else 'FAIL'
    print(f'RESULT: {verdict} {len(diffs) - len(breaking)}/{len(diffs)} endpoint(s) '
          f'compatible with the fixtures')
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='contract_check.py',
        description='Validate the OpenF1 payload contract against the golden '
                    'fixtures (offline) or against the live API.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--offline', action='store_true',
                      help='validate the fixtures only, no network (the CI mode; '
                           'this is the default)')
    mode.add_argument('--live', action='store_true',
                      help='fetch each endpoint through the shared rate limiter '
                           'and diff it against its fixture')
    parser.add_argument('--fixtures', type=Path, default=FIXTURES_DIR,
                        metavar='DIR', help='fixture directory (default: tests/fixtures)')
    parser.add_argument('--endpoint', action='append', dest='endpoints',
                        metavar='NAME', help='limit the check to this endpoint '
                                             '(repeatable)')
    parser.add_argument('--year', type=int, default=None,
                        help='override the capture query year in --live mode')
    parser.add_argument('--session-key', type=int, default=None,
                        help='override session_key in --live mode')
    parser.add_argument('--meeting-key', type=int, default=None,
                        help='override meeting_key in --live mode')
    parser.add_argument('--json', action='store_true',
                        help='print machine-readable JSON')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    names = list(args.endpoints) if args.endpoints else list(ENDPOINTS)
    unknown = [name for name in names if name not in ENDPOINTS]
    if unknown:
        print(f'unknown endpoint(s): {", ".join(unknown)}', file=sys.stderr)
        print(f'known endpoints: {", ".join(sorted(ENDPOINTS))}', file=sys.stderr)
        return 2
    if args.live:
        return run_live(names, args.fixtures, args.year, args.session_key,
                        args.meeting_key, args.json)
    return run_offline(names, args.fixtures, args.json)


if __name__ == '__main__':
    raise SystemExit(main())
