# -*- coding: utf-8 -*-
"""
artifacts.py
~~~~~~~~~~~~
Data-as-artifact loader: the app reads OpenF1 from the local cache first.

The app used to depend on OpenF1 at runtime, so a cold start or an outage
meant empty pages. A scheduled job (:mod:`tools.refresh_cache`) now downloads
the season into the same on-disk cache the HTTP client writes
(``f1_cache/openf1/``), and this module is the read side: it answers from that
directory and only calls the service when a file is missing or stale.

Freshness policy
----------------
A response is a snapshot of something that changes at a known rate, so the
carrier of the policy is the session, not the endpoint:

    ==========================  ==========================  ===============
    what the query is about     how long the file is good   why
    ==========================  ==========================  ===============
    a finished session          forever                     the archive is
                                                            immutable once
                                                            the session ended
    a running session           12 h (LIVE_MAX_AGE_S)       it is still being
                                                            written
    a past season's /sessions   forever                     a season that has
                                                            ended gains no
                                                            sessions
    nothing known               the caller's max_age        be conservative
    ==========================  ==========================  ===============

"Finished" is not guessed from the clock alone: the refresh job records every
session it archived -- with its start, its end and whether it was over -- in
``f1_cache/openf1/_index/manifest.json``, and :func:`is_finished` reads that.
Without a manifest (a developer machine that has only ever used the app) the
rule is the conservative one: the caller's own TTL, which is exactly today's
behaviour, so nothing regresses.

Two entry points
----------------
* :func:`load_json` -- one query, local file first, API on a miss. This is the
  function new code should call.
* :func:`activate` -- one line that makes the *whole* app prefer the archive
  without touching a single data module: it hands the HTTP client a TTL policy
  under which an archived finished session never expires. Recommended for
  ``src/shared.py`` (see ``docs/DATA_CACHE.md``).

Data honesty
------------
* the loader never invents a row: a missing file with the network switched off
  is ``None``, and an endpoint that really has no rows is ``[]``;
* a stale local copy is only served when the caller explicitly asked for no
  network (``allow_network=False``), and the event is logged;
* the manifest is diagnostic, never authoritative about results: it records
  which queries were archived and when, so a status page can say how old the
  data is.

This module must stay importable without Streamlit -- the refresh job, the
tests and the CI lint all import it in a plain interpreter.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

# The cache key must match the client's byte for byte, so the key is asked of
# the client instead of being re-derived here.
import openf1_client as client  # noqa: E402

logger = logging.getLogger(__name__)

# --- freshness policy ------------------------------------------------------
# A session that is still being run is re-checked every 12 hours; one the
# refresh job has archived as finished is served from disk forever.
LIVE_MAX_AGE_S = 12 * 3600
FINISHED_MAX_AGE_S: Optional[int] = None

# A session counts as finished only this long after its end, so the
# classification of a race that ended minutes ago is never treated as final.
FINISHED_GRACE_S = 3 * 3600

# ``cached_json`` wants an integer TTL; "forever" is expressed to it as ten
# years, which is longer than any cache entry can plausibly live.
_FOREVER_TTL_S = 10 * 365 * 24 * 3600

# The manifest lives beside the response files, in a directory of its own so a
# status page can tell the two apart.
MANIFEST_DIR_NAME = '_index'
MANIFEST_NAME = 'manifest.json'

# Sentinels for "the policy decides" and "nothing is cached".
_AUTO_MAX_AGE = object()

# When set, the local cache lives here instead of in the HTTP client's default
# directory (tests point it at ``tmp_path``; ``--out`` of the refresh job uses
# it too). ``None`` means "follow the client".
CACHE_ROOT: Optional[Path] = None


# ---------------------------------------------------------------------------
# Where a query's file lives
# ---------------------------------------------------------------------------

def cache_root() -> Path:
    """Directory the artifact cache lives in."""
    if CACHE_ROOT is not None:
        return Path(CACHE_ROOT)
    return Path(client.CACHE_ROOT)


def endpoint_url(endpoint: str) -> str:
    """Full URL of a collection endpoint, as the HTTP client builds it."""
    return f'{client.OPENF1_BASE}/{endpoint}'


def cache_path(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Path:
    """File that holds the response to one query.

    The name comes from the client's key function: a loader that derived its
    own name would look in one place while the client wrote to another.
    """
    key = client._cache_key(endpoint_url(endpoint), params or None)
    return cache_root() / f'{key}.json'


def entry_age_s(path: Path, now: Optional[float] = None) -> Optional[float]:
    """Age of a cache entry in seconds, or ``None`` when it does not exist."""
    try:
        return max(0.0, (time.time() if now is None else now) - path.stat().st_mtime)
    except OSError:
        return None


def is_fresh(age_s: Optional[float], max_age_s: Optional[int]) -> bool:
    """True when an entry of ``age_s`` is still usable under ``max_age_s``.

    ``max_age_s=None`` means "never expires"; an absent entry is never fresh.
    """
    if age_s is None:
        return False
    if max_age_s is None:
        return True
    return age_s < max_age_s


# ---------------------------------------------------------------------------
# Local reads (never a network call)
# ---------------------------------------------------------------------------

def read_cached(endpoint: str, **params: Any) -> Optional[List[Dict[str, Any]]]:
    """The local copy of a query, whatever its age.

    Returns ``None`` when there is no usable file. An empty list is a real
    answer -- "the service has no rows" -- and is not a miss.
    """
    path = cache_path(endpoint, params or None)
    try:
        with path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, list) else None


# ---------------------------------------------------------------------------
# The manifest: what the refresh job archived
# ---------------------------------------------------------------------------

def manifest_path() -> Path:
    """Location of the archive manifest."""
    return cache_root() / MANIFEST_DIR_NAME / MANIFEST_NAME


def read_manifest() -> Dict[str, Any]:
    """The manifest as a dict, or ``{}`` when there is none."""
    try:
        with manifest_path().open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_manifest(payload: Dict[str, Any]) -> Optional[Path]:
    """Write the manifest atomically (temp file + replace).

    A half-written manifest would make every session look live, which is
    expensive but not wrong; the atomic replace makes sure it cannot happen.
    """
    path = manifest_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        with tmp.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning('Could not write the cache manifest %s: %s', path, exc)
        return None
    return path


# The index is read on every page load, so the parse is memoised until the
# manifest on disk changes (the refresh job rewrites it every night).
_INDEX_CACHE: Dict[str, Any] = {'key': None, 'sessions': {}, 'meetings': {}}


def _index_view() -> 'tuple[Dict[int, Dict[str, Any]], Dict[int, bool]]':
    """(session_key -> record, meeting_key -> all sessions finished)."""
    path = manifest_path()
    try:
        key = (str(path), path.stat().st_mtime_ns)
    except OSError:
        key = (str(path), None)
    if _INDEX_CACHE['key'] == key:
        return _INDEX_CACHE['sessions'], _INDEX_CACHE['meetings']

    sessions: Dict[int, Dict[str, Any]] = {}
    meetings: Dict[int, bool] = {}
    for raw_key, meta in (read_manifest().get('sessions') or {}).items():
        number = _as_int(raw_key)
        if number is None or not isinstance(meta, dict):
            continue
        sessions[number] = meta
        meeting = _as_int(meta.get('meeting_key'))
        if meeting is not None:
            # A meeting is finished only when every one of its sessions is.
            meetings[meeting] = (meetings.get(meeting, True)
                                 and bool(meta.get('finished')))
    _INDEX_CACHE.update(key=key, sessions=sessions, meetings=meetings)
    return sessions, meetings


def session_index() -> Dict[int, Dict[str, Any]]:
    """``{session_key: record}`` of every session the archive holds."""
    return dict(_index_view()[0])


def is_finished(params: Optional[Dict[str, Any]] = None) -> bool:
    """True when the archive can no longer change for this query.

    A query is pinned to a session through ``session_key``, to a weekend
    through ``meeting_key``, or to a season through ``year``. Only the third
    can be answered without the manifest -- a season that ended gains no
    sessions -- so an unknown session is treated as live.
    """
    params = params or {}
    sessions, meetings = _index_view()

    session_key = _as_int(params.get('session_key'))
    if session_key is not None:
        meta = sessions.get(session_key)
        return bool(meta.get('finished')) if meta else False

    meeting_key = _as_int(params.get('meeting_key'))
    if meeting_key is not None:
        return bool(meetings.get(meeting_key, False))

    year = _as_int(params.get('year'))
    if year is not None:
        return year < datetime.now(timezone.utc).year

    return False


def max_age_for(params: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Seconds a cached response stays fresh (``None`` = never expires)."""
    return FINISHED_MAX_AGE_S if is_finished(params) else LIVE_MAX_AGE_S


def session_finished(session: Dict[str, Any],
                     now: Optional[datetime] = None) -> bool:
    """True when a session's data can no longer change.

    Shared by the loader and the refresh job so the two can never disagree
    about which sessions are still being written.
    """
    end = parse_dt(session.get('date_end')) or parse_dt(session.get('date_start'))
    if end is None:
        return False
    now = now or datetime.now(timezone.utc)
    return end + timedelta(seconds=FINISHED_GRACE_S) <= now


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_json(
    endpoint: str,
    *,
    max_age: Any = _AUTO_MAX_AGE,
    allow_network: bool = True,
    **params: Any,
) -> Optional[List[Dict[str, Any]]]:
    """The response to one OpenF1 query, from the local cache when it can be.

    Args:
        endpoint: collection endpoint, e.g. ``'session_result'``.
        max_age: seconds the local file stays good. Omitted means "ask the
            policy": forever for a session the refresh job archived as
            finished, :data:`LIVE_MAX_AGE_S` otherwise.
        allow_network: when false, no request is made at all. A stale local
            copy is then returned if one exists (logged as such) so an offline
            demo still shows the last archived state; ``None`` means nothing
            is available.
        **params: the OpenF1 query parameters (``session_key``, ``year``, ...).

    Returns:
        The payload (``[]`` when the service has no rows for the query), or
        ``None`` when the file is missing and the network is not allowed.
    """
    path = cache_path(endpoint, params or None)
    limit = max_age_for(params) if max_age is _AUTO_MAX_AGE else max_age

    if is_fresh(entry_age_s(path), limit):
        payload = read_cached(endpoint, **params)
        if payload is not None:
            return payload

    if not allow_network:
        payload = read_cached(endpoint, **params)
        if payload is not None:
            logger.info('Serving the stale local copy of %s %s (offline mode)',
                        endpoint, params)
        return payload

    ttl = _FOREVER_TTL_S if limit is None else int(limit)
    return client.cached_openf1(endpoint, ttl=ttl, **params)


# ---------------------------------------------------------------------------
# Whole-app policy (the one-line integration)
# ---------------------------------------------------------------------------

# The client's own function, parked while :func:`activate` has the wheel.
_original_cached_json: Optional[Any] = None


def activate(cache_dir: Optional[Path] = None) -> None:
    """Make the archive outrank the API for finished sessions, app-wide.

    Every data module reaches OpenF1 through
    ``openf1_client.cached_json`` -- the season results, the session fallback,
    the telemetry ingest -- and each passes its own TTL (a day for a finished
    session). A day is short for data that can never change: a cold machine
    refetches ~55 requests a night for numbers it already has on disk, and an
    offline machine fails where the archive would have answered.

    This installs a policy in front of that function: when the query belongs to
    a session the refresh job archived as finished and the file exists, the
    file wins, however old it is. Everything else is delegated unchanged, so
    live sessions and cold misses behave exactly as before.

    Idempotent; :func:`deactivate` restores the client's own function.
    """
    global CACHE_ROOT, _original_cached_json
    if cache_dir is not None:
        CACHE_ROOT = Path(cache_dir)
    if _original_cached_json is None:
        _original_cached_json = client.cached_json
        client.cached_json = _cached_json_from_archive


def deactivate() -> None:
    """Undo :func:`activate` (tests and tools that want the raw client)."""
    global _original_cached_json
    if _original_cached_json is not None:
        client.cached_json = _original_cached_json
        _original_cached_json = None


def _cached_json_from_archive(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    ttl: int = client.CACHE_TTL_S,
    timeout: int = client.REQUEST_TIMEOUT_S,
    attempts: int = client.REQUEST_ATTEMPTS,
) -> Any:
    """``openf1_client.cached_json`` with the archive in front of the TTL."""
    endpoint = _endpoint_of(url)
    if endpoint is not None and is_finished(params):
        payload = read_cached(endpoint, **(params or {}))
        if payload is not None:
            return payload
    if _original_cached_json is None:  # pragma: no cover - activate() first
        raise RuntimeError('artifacts.activate() was not called')
    return _original_cached_json(url, params=params, ttl=ttl,
                                 timeout=timeout, attempts=attempts)


def _endpoint_of(url: str) -> Optional[str]:
    """The endpoint name of a URL built by the client, or ``None``."""
    prefix = f'{client.OPENF1_BASE}/'
    if url.startswith(prefix):
        return url[len(prefix):]
    return None


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def cache_status() -> Dict[str, Any]:
    """Counts and ages of the archive, for a status page.

    Everything here is measured on disk (or read from the manifest), so a page
    can say how old its data is without asking the service anything.
    """
    root = cache_root()
    files = 0
    total = 0
    oldest: Optional[float] = None
    newest: Optional[float] = None
    if root.is_dir():
        for entry in root.iterdir():
            if not entry.name.endswith('.json') or not entry.is_file():
                continue
            try:
                stat = entry.stat()
            except OSError:
                continue
            files += 1
            total += stat.st_size
            oldest = stat.st_mtime if oldest is None else min(oldest, stat.st_mtime)
            newest = stat.st_mtime if newest is None else max(newest, stat.st_mtime)

    manifest = read_manifest()
    return {
        'root': str(root),
        'exists': root.is_dir(),
        'files': files,
        'bytes': total,
        'oldest': _iso(oldest),
        'newest': _iso(newest),
        'sessions_indexed': len(manifest.get('sessions') or {}),
        'archive_active': _original_cached_json is not None,
        'manifest_generated_at': manifest.get('generated_at'),
        'last_run': manifest.get('last_run') or {},
        'endpoints': manifest.get('endpoints') or {},
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _as_int(value: Any) -> Optional[int]:
    """Coerce a query parameter to an int, or ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_dt(value: Any) -> Optional[datetime]:
    """Parse an OpenF1 timestamp into an aware UTC datetime."""
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _iso(stamp: Optional[float]) -> Optional[str]:
    """A POSIX timestamp as an ISO-8601 UTC string."""
    if stamp is None:
        return None
    return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat(timespec='seconds')
