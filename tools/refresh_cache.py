#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Refresh the on-disk OpenF1 archive (the data-as-artifact job).

The app normally reads OpenF1 through :mod:`openf1_client`, whose disk cache
is filled on demand. A cold machine therefore needs the network for every
page, and an outage leaves the pages empty. This job does the downloading
ahead of time: it walks the sessions of the requested seasons and writes each
response into the same cache directory the app reads, so the app can serve a
season from disk.

    python tools/refresh_cache.py --years 2026 --session-limit 8
    python tools/refresh_cache.py --years 2023 2024 2025 --dry-run
    python tools/refresh_cache.py --years 2026 --out f1_cache/openf1

What it fetches, per started session (future sessions are skipped):

    sessions?year=...            one per year
    session_result?session_key=  per session
    drivers?session_key=         per session
    stints?session_key=          per session
    pit?session_key=             per session
    laps?session_key=            per session
    weather?session_key=         per session
    starting_grid?meeting_key=   once per meeting
    drivers?meeting_key=         once per meeting

``drivers`` is requested twice on purpose: ``openf1_session`` asks by
``session_key`` and ``openf1_results`` (the season frame) asks by
``meeting_key``. They are two different cache entries, so archiving only one
would leave the other page on the network.

Every request goes through :func:`openf1_client.cached_openf1`, which owns the
process-wide rate limiter (30 requests/minute, 3/second) and the atomic cache
write. This job adds no other HTTP path and no second limiter.

A 404 (or any other transport error) is counted and skipped: one missing
resource must never stop the run. The tally at the end reports how many
requests, cache hits, rows and bytes the run produced, and how big the archive
now is.

The job also maintains ``f1_cache/openf1/_index/manifest.json``: which
sessions are archived and finished. :mod:`artifacts` reads that manifest to
decide how long a cached file may be served straight from disk (forever for a
finished session, :data:`artifacts.LIVE_MAX_AGE_S` otherwise). Without it the
loader stays on the conservative live rule.

Exit codes: 0 when at least one year produced a session list, 1 when every
requested year failed (a failed run must not look green in CI).
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import artifacts  # noqa: E402
import openf1_client as client  # noqa: E402

logger = logging.getLogger('refresh_cache')

# OpenF1's archive starts with the 2023 season (see docs/DATA_CACHE.md).
FIRST_ARCHIVED_YEAR = 2023

# Queries scoped to one session.
SESSION_ENDPOINTS: Tuple[str, ...] = (
    'session_result',
    'drivers',
    'stints',
    'pit',
    'laps',
    'weather',
)

# Queries scoped to one meeting (weekend). Fetched once per meeting.
MEETING_ENDPOINTS: Tuple[str, ...] = (
    'starting_grid',
    'drivers',
)

STAT_KEYS: Tuple[str, ...] = ('requests', 'hits', 'empty', 'errors', 'rows', 'bytes')


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _new_stats() -> Dict[str, Dict[str, int]]:
    """A zeroed counter block per endpoint (order preserved, duplicates gone)."""
    names = ('sessions',) + SESSION_ENDPOINTS + MEETING_ENDPOINTS
    return {name: {key: 0 for key in STAT_KEYS} for name in dict.fromkeys(names)}


def _merge(target: Dict[str, Dict[str, int]],
           part: Dict[str, Dict[str, int]]) -> None:
    """Add ``part`` into ``target`` in place."""
    for name, entry in part.items():
        dst = target.setdefault(name, {key: 0 for key in STAT_KEYS})
        for key in STAT_KEYS:
            dst[key] = int(dst.get(key, 0)) + int(entry.get(key, 0))


def _totals(stats: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    """One row of sums over every endpoint."""
    total = {key: 0 for key in STAT_KEYS}
    for entry in stats.values():
        for key in STAT_KEYS:
            total[key] += int(entry.get(key, 0))
    return total


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _int(value: Any) -> Optional[int]:
    """Coerce an OpenF1 key to int, or ``None`` when it is missing."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _fmt_bytes(value: Any) -> str:
    size = float(value or 0)
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024 or unit == 'GB':
            if unit == 'B':
                return '%d B' % int(size)
            return '%.1f %s' % (size, unit)
        size /= 1024
    return '%.1f GB' % size


def _fmt_duration(seconds: float) -> str:
    total = int(round(seconds))
    if total < 60:
        return '%ds' % total
    minutes, seconds_left = divmod(total, 60)
    if minutes < 60:
        return '%dm %02ds' % (minutes, seconds_left)
    hours, minutes_left = divmod(minutes, 60)
    return '%dh %02dm' % (hours, minutes_left)


def _iso(stamp: Optional[float]) -> Optional[str]:
    if stamp is None:
        return None
    return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat(timespec='seconds')


def _use_cache_dir(path: Optional[str]) -> Path:
    """Point the client and the loader at one cache directory.

    ``--out`` exists so a test or a second archive can be filled without
    touching the app's cache. The client reads its ``CACHE_ROOT`` at call time,
    so assigning the module attribute is enough, and the loader is pinned to
    the same value so both agree which file a query maps to.
    """
    target = Path(path).expanduser() if path else Path(client.CACHE_ROOT)
    target = target.resolve()
    artifacts.CACHE_ROOT = target
    client.CACHE_ROOT = target
    if hasattr(client, '_MISSING_ROOT'):
        client._MISSING_ROOT = target / 'missing'
    return target


# ---------------------------------------------------------------------------
# One query
# ---------------------------------------------------------------------------

def _do(endpoint: str,
        params: Dict[str, Any],
        ttl: Optional[int],
        stats: Dict[str, Dict[str, int]],
        dry_run: bool) -> List[Dict[str, Any]]:
    """Fetch (or plan) one query and count what happened.

    ``ttl`` is ``None`` for "the file never expires" (a finished session) and
    ``0`` for "re-check it every run" (a session that is still running). The
    fetch itself goes through :func:`artifacts.load_json`, which delegates to
    ``openf1_client.cached_openf1`` -- same cache path, same rate limiter.
    """
    entry = stats.setdefault(endpoint, {key: 0 for key in STAT_KEYS})
    path = artifacts.cache_path(endpoint, params or None)

    if not dry_run and artifacts.is_fresh(artifacts.entry_age_s(path), ttl):
        payload = artifacts.read_cached(endpoint, **params)
        if payload is not None:
            entry['hits'] += 1
            entry['rows'] += len(payload)
            return payload

    if dry_run:
        # A dry run must not touch the network. Count the request that would
        # be made, and keep any local copy so the plan can still be enumerated.
        entry['requests'] += 1
        payload = artifacts.read_cached(endpoint, **params)
        return payload or []

    before = _file_size(path)
    try:
        payload = artifacts.load_json(endpoint, max_age=ttl, **params) or []
    except Exception as exc:  # noqa: BLE001 - one bad query must not stop the run
        entry['errors'] += 1
        logger.warning('%s %s failed: %s', endpoint, params, exc)
        return []

    entry['requests'] += 1
    entry['rows'] += len(payload)
    if not payload:
        entry['empty'] += 1
    entry['bytes'] += max(0, _file_size(path) - before)
    return payload


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def _select_sessions(sessions: List[Dict[str, Any]],
                     now: datetime,
                     limit: int) -> Tuple[List[Dict[str, Any]], int, int]:
    """The sessions worth fetching: started, in chronological order.

    Returns ``(planned, future, cancelled)``. A session that has not started
    has no data yet, so it is skipped (the next run picks it up); a cancelled
    meeting is skipped as well.
    """
    planned: List[Dict[str, Any]] = []
    future = 0
    cancelled = 0
    for session in sessions:
        if session.get('is_cancelled'):
            cancelled += 1
            continue
        start = artifacts.parse_dt(session.get('date_start'))
        if start is not None and start > now:
            future += 1
            continue
        planned.append(session)

    planned.sort(key=lambda item: str(item.get('date_start') or ''))
    if limit > 0:
        planned = planned[-limit:]
    return planned, future, cancelled


def _meeting_finished(sessions: List[Dict[str, Any]],
                      now: datetime) -> Dict[int, bool]:
    """Per meeting: has the weekend's last session ended long enough ago?

    A meeting is not finished when its first session is: ``starting_grid``
    only has rows once qualifying ran, so the grid of an FP1-only weekend
    would be an empty file. The test is therefore the meeting's *last* session
    plus the same grace period the loader uses for a single session.
    """
    last: Dict[int, datetime] = {}
    for session in sessions:
        key = _int(session.get('meeting_key'))
        end = (artifacts.parse_dt(session.get('date_end'))
               or artifacts.parse_dt(session.get('date_start')))
        if key is None or end is None:
            continue
        if key not in last or end > last[key]:
            last[key] = end
    grace = timedelta(seconds=artifacts.FINISHED_GRACE_S)
    return {key: (end + grace) <= now for key, end in last.items()}


def _session_entry(session: Dict[str, Any], finished: bool) -> Dict[str, Any]:
    """The manifest row for one archived session."""
    return {
        'year': _int(session.get('year')),
        'meeting_key': _int(session.get('meeting_key')),
        'session_name': session.get('session_name'),
        'session_type': session.get('session_type'),
        'date_start': session.get('date_start'),
        'date_end': session.get('date_end'),
        'finished': bool(finished),
    }


def _label(session: Dict[str, Any]) -> str:
    """``2026-03-01 Race`` -- short enough for a progress line."""
    start = str(session.get('date_start') or '')[:10]
    name = session.get('session_name') or session.get('session_type') or 'session'
    text = ('%s %s' % (start, name)).strip()
    return text[:42]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_table(stats: Dict[str, Dict[str, int]]) -> None:
    header = '  %-14s %9s %7s %7s %8s %10s %10s' % (
        'endpoint', 'requests', 'hits', 'empty', 'errors', 'rows', 'bytes')
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for name in stats:
        entry = stats[name]
        if not any(entry[key] for key in STAT_KEYS):
            continue
        print('  %-14s %9d %7d %7d %8d %10d %10s' % (
            name,
            entry['requests'],
            entry['hits'],
            entry['empty'],
            entry['errors'],
            entry['rows'],
            _fmt_bytes(entry['bytes']),
        ))


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def _write_index(entries: Dict[int, Dict[str, Any]],
                 year_stats: Dict[str, Dict[str, int]],
                 years: List[int],
                 run_totals: Dict[str, int],
                 started: float) -> Optional[Path]:
    """Merge this year into the manifest the loader reads.

    Sessions accumulate; the per-endpoint counters are lifetime totals for the
    whole archive, so a status page can say how much has ever been pulled.
    """
    manifest = artifacts.read_manifest()
    sessions = dict(manifest.get('sessions') or {})
    for key, entry in entries.items():
        sessions[str(key)] = entry

    endpoints = dict(manifest.get('endpoints') or {})
    for name, entry in year_stats.items():
        bucket = endpoints.setdefault(name, {key: 0 for key in STAT_KEYS})
        for key in STAT_KEYS:
            bucket[key] = int(bucket.get(key, 0)) + int(entry.get(key, 0))

    manifest['sessions'] = sessions
    manifest['endpoints'] = endpoints
    manifest['cache_root'] = str(artifacts.cache_root())
    manifest['generated_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    manifest['last_run'] = {
        'years': list(years),
        'started_at': _iso(started),
        'requests': run_totals.get('requests', 0),
        'cache_hits': run_totals.get('hits', 0),
        'errors': run_totals.get('errors', 0),
        'bytes': run_totals.get('bytes', 0),
    }
    return artifacts.write_manifest(manifest)


# ---------------------------------------------------------------------------
# One year
# ---------------------------------------------------------------------------

def _refresh_year(year: int,
                  limit: int,
                  dry_run: bool,
                  now: datetime,
                  stats: Dict[str, Dict[str, int]]
                  ) -> Tuple[bool, Dict[int, Dict[str, Any]]]:
    """Walk one season. Returns ``(ok, manifest entries)``."""
    print('\n[%d]' % year)
    if year < FIRST_ARCHIVED_YEAR:
        print('  skipped: OpenF1 has no archive before %d' % FIRST_ARCHIVED_YEAR)
        return True, {}

    # The session list of the running season still changes, so it is re-read
    # every run; a past season's list is immutable.
    year_ttl = artifacts.FINISHED_MAX_AGE_S if year < now.year else 0
    sessions = _do('sessions', {'year': year}, year_ttl, stats, dry_run)

    if not sessions:
        if stats['sessions']['errors']:
            print('  the session list could not be fetched (see the warning above)')
            return False, {}
        if dry_run and not artifacts.read_cached('sessions', year=year):
            print('  the session list is not cached yet: the run would start by '
                  'fetching /sessions?year=%d' % year)
            return True, {}
        print('  no sessions: OpenF1 has nothing for %d yet' % year)
        return True, {}

    started_count = len(_select_sessions(sessions, now, 0)[0])
    planned, future, cancelled = _select_sessions(sessions, now, limit)
    print('  sessions: %d listed, %d started, %d future, %d cancelled -> fetching %d'
          % (len(sessions), started_count, future, cancelled, len(planned)))

    meetings = _meeting_finished(sessions, now)
    seen_meetings = set()
    entries: Dict[int, Dict[str, Any]] = {}

    for index, session in enumerate(planned, 1):
        key = _int(session.get('session_key'))
        finished = artifacts.session_finished(session, now)
        session_ttl = artifacts.FINISHED_MAX_AGE_S if finished else 0
        before = _totals(stats)

        if key is not None:
            for endpoint in SESSION_ENDPOINTS:
                _do(endpoint, {'session_key': key}, session_ttl, stats, dry_run)

        meeting = _int(session.get('meeting_key'))
        if meeting is not None and meeting not in seen_meetings:
            seen_meetings.add(meeting)
            meeting_ttl = (artifacts.FINISHED_MAX_AGE_S
                           if meetings.get(meeting) else 0)
            for endpoint in MEETING_ENDPOINTS:
                _do(endpoint, {'meeting_key': meeting}, meeting_ttl, stats, dry_run)

        after = _totals(stats)
        errors = after['errors'] - before['errors']
        print('  [%2d/%2d] %-44s req %2d  hit %2d  rows %6d%s'
              % (index, len(planned), _label(session),
                 after['requests'] - before['requests'],
                 after['hits'] - before['hits'],
                 after['rows'] - before['rows'],
                 ('  errors %d' % errors) if errors else ''))

        if key is not None:
            # A session is only advertised as "finished" once every query for
            # it came back cleanly; otherwise the loader would serve a partial
            # file forever instead of retrying it on the next run.
            entries[key] = _session_entry(session, finished and errors == 0)

    print('')
    _print_table(stats)
    return True, entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='refresh_cache.py',
        description='Download OpenF1 sessions into the on-disk cache the app reads.',
        epilog='The archive is a derivative of OpenF1 data (CC BY-NC-SA 4.0): '
               'non-commercial use only. See docs/DATA_CACHE.md.',
    )
    parser.add_argument('--years', type=int, nargs='+', metavar='YEAR',
                        help='seasons to refresh (default: the current year)')
    parser.add_argument('--session-limit', type=int, default=0, metavar='N',
                        help='only the N most recent started sessions per year '
                             '(default: all of them)')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the plan without a request or a write')
    parser.add_argument('--out', metavar='DIR',
                        help='cache directory (default: f1_cache/openf1)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='log every skipped query')
    args = parser.parse_args(argv)
    if args.session_limit < 0:
        parser.error('--session-limit must be zero or positive')
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format='%(levelname)s %(message)s',
    )

    cache_dir = _use_cache_dir(args.out)
    now = datetime.now(timezone.utc)
    years = sorted(set(args.years or [now.year]))
    started = time.time()

    print('OpenF1 cache refresh')
    print('  cache dir : %s' % cache_dir)
    print('  years     : %s' % ', '.join(str(year) for year in years))
    print('  sessions  : %s'
          % (('the last %d per year' % args.session_limit)
             if args.session_limit else 'every started session'))
    print('  mode      : %s'
          % ('DRY RUN - no request, no write'
             if args.dry_run else 'live - through the shared rate limiter'))

    totals = _new_stats()
    all_entries: Dict[int, Dict[str, Any]] = {}
    failed: List[int] = []

    for year in years:
        year_stats = _new_stats()
        ok, entries = _refresh_year(year, args.session_limit, args.dry_run,
                                   now, year_stats)
        _merge(totals, year_stats)
        all_entries.update(entries)
        if not ok:
            failed.append(year)
        if not args.dry_run:
            path = _write_index(all_entries, year_stats, years,
                                _totals(totals), started)
            if path is not None:
                print('  index: %s (%d sessions)' % (path, len(all_entries)))

    elapsed = time.time() - started
    total = _totals(totals)
    status = artifacts.cache_status()

    print('\nFinal tally')
    print('  network requests : %d' % total['requests'])
    print('  cache hits       : %d' % total['hits'])
    print('  empty results    : %d' % total['empty'])
    print('  transport errors : %d' % total['errors'])
    print('  rows read        : %d' % total['rows'])
    print('  bytes written    : %s' % _fmt_bytes(total['bytes']))
    print('  archive on disk  : %d files, %s'
          % (status['files'], _fmt_bytes(status['bytes'])))
    print('  sessions indexed : %d' % status['sessions_indexed'])
    print('  elapsed          : %s' % _fmt_duration(elapsed))
    if args.dry_run:
        print('  dry run          : nothing was requested or written')

    if failed and len(failed) == len(years):
        print('\nEvery requested year failed: the archive was not refreshed.')
        return 1
    if failed:
        print('\nWARNING: no data for %s'
              % ', '.join(str(year) for year in failed))
    return 0


if __name__ == '__main__':
    sys.exit(main())
