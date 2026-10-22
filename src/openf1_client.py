# -*- coding: utf-8 -*-
"""
openf1_client.py
~~~~~~~~~~~~~~~~~
Shared HTTP client for the OpenF1 API.

Two parts of the app talk to OpenF1: the session-telemetry ingest
(:mod:`telemetry`) and the season-results layer (:mod:`openf1_results`).
They must respect the same published limits (3 requests/second, 30
requests/minute on the free tier), so the rate limiter, the retry policy and
the on-disk response cache live here instead of being duplicated.

Data honesty:
    * a 404 is an empty result, not a failure (a driver who never set a lap
      time, a meeting without a grid) -- it becomes ``[]``;
    * every other transport error is raised after the retries are exhausted,
      so a caller never receives a half-built payload;
    * cached responses are stored with the query that produced them, so a
      cache hit can never be served for a different session or season.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import builtins
import hashlib
import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

OPENF1_BASE = 'https://api.openf1.org/v1'
MULTIVIEWER_BASE = 'https://api.multiviewer.app/api/v1'

REQUEST_TIMEOUT_S = 30
REQUEST_ATTEMPTS = 4
RATE_LIMIT_PER_SECOND = 3
RATE_LIMIT_PER_MINUTE = 30
REQUEST_HEADERS = {
    'User-Agent': 'F1Lab/1.0 (public-data analytics workbench)',
    'Accept': 'application/json',
}

# On-disk cache for API responses: a season of results is ~55 requests, so a
# warm cache turns a two-minute cold start into an instant one. A container (or
# any host with a volume) can move it with F1LAB_CACHE_DIR, which keeps the app
# directory read-only and the cache on a mount that survives a restart.
_CACHE_ENV = os.environ.get('F1LAB_CACHE_DIR', '').strip()
CACHE_BASE = (Path(_CACHE_ENV) if _CACHE_ENV
              else Path(__file__).parent.parent / 'f1_cache')
CACHE_ROOT = CACHE_BASE / 'openf1'
CACHE_TTL_S = 6 * 3600

# "This query has no rows" notes live next to the response cache but expire
# much faster: the archive can gain a session at any time.
_MISSING_ROOT = CACHE_ROOT / 'missing'
MISSING_TTL_S = 300

# Attribute name the process-wide rate limiter is parked under, so the module
# can be imported under two names without creating two limiters.
_LIMITER_ATTRIBUTE = '_f1lab_openf1_rate_limiter'


class RateLimiter:
    """Token bucket honouring both the per-second and per-minute OpenF1 caps.

    The service publishes two limits and a burst is only allowed by the
    per-second cap, so both windows are tracked separately and the longest
    wait wins.
    """

    def __init__(self, burst: int = RATE_LIMIT_PER_SECOND,
                 per_minute: int = RATE_LIMIT_PER_MINUTE) -> None:
        self.burst = burst
        self.per_minute = per_minute
        self._recent = deque(maxlen=burst)
        self._minute = deque(maxlen=per_minute)
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until the next request is allowed under both caps."""
        while True:
            with self._lock:
                now = time.monotonic()
                minute_wait = self._wait_for(self._minute, now, 60.0, self.per_minute)
                burst_wait = self._wait_for(self._recent, now, 1.0, self.burst)
                wait = max(minute_wait, burst_wait)
                if wait <= 0:
                    self._recent.append(now)
                    self._minute.append(now)
                    return
            time.sleep(min(max(wait, 0.05), 15.0))

    @staticmethod
    def _wait_for(stamps: deque, now: float, window: float,
                  cap: int) -> float:
        """Seconds until the oldest stamp in ``stamps`` falls out of ``window``."""
        live = [s for s in stamps if now - s < window]
        stamps.clear()
        stamps.extend(live)
        if len(live) < cap:
            return 0.0
        return window - (now - live[0])


def _shared_limiter() -> RateLimiter:
    """The one rate limiter of the process.

    The module can be imported twice under two names (``openf1_client`` by the
    app, ``src.openf1_client`` by the test suite), which would otherwise give
    the service two independent limiters. The instance is therefore parked in
    ``builtins`` so both imports share it.
    """
    existing = getattr(builtins, _LIMITER_ATTRIBUTE, None)
    if existing is not None:
        return existing
    limiter = RateLimiter()
    setattr(builtins, _LIMITER_ATTRIBUTE, limiter)
    return limiter


LIMITER = _shared_limiter()


def _retry_after(response: requests.Response) -> float:
    """Seconds the service asked us to wait, defaulting to a short pause."""
    try:
        return min(max(float(response.headers.get('Retry-After', 5)), 1.0), 60.0)
    except (TypeError, ValueError):
        return 5.0


class NotFound(RuntimeError):
    """The service answered 404: the query matches nothing.

    It is an empty result, not a failure. Retrying it is pointless and
    expensive: the backoff of four attempts costs ~14 seconds per missing
    resource, which is what a season build pays for every session the
    archive does not have.
    """


def fetch_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = REQUEST_TIMEOUT_S,
    attempts: int = REQUEST_ATTEMPTS,
) -> Any:
    """GET a JSON document under the shared rate limit.

    HTTP 429 is honoured rather than treated as an error: the service tells us
    how long to wait, so a burst that overshoots costs one pause instead of a
    failed ingest. HTTP 404 raises :class:`NotFound` straight away -- it is
    not a transient failure, so it is never retried. Every other error is
    retried with a backoff, and the last one is raised when the attempts run
    out, so a caller never sees a half-built payload.
    """
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        LIMITER.acquire()
        try:
            response = requests.get(url, params=params, timeout=timeout,
                                    headers=REQUEST_HEADERS)
            if response.status_code == 429:
                time.sleep(_retry_after(response))
                continue
            if response.status_code == 404:
                raise NotFound(f'GET {url} -> 404 (no rows)')
            response.raise_for_status()
            return response.json()
        except NotFound:
            raise
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(min(2.0 ** attempt, 10.0))
    raise RuntimeError(f'GET {url} failed after {attempts} attempts: {last_error}')


def openf1(endpoint: str, **params: Any) -> List[Dict[str, Any]]:
    """Fetch one OpenF1 collection endpoint as a list of records.

    OpenF1 answers HTTP 404 for a query that matches nothing -- a driver who
    never set a lap time, a session without a given channel. That is an empty
    result, not a failure, so it becomes ``[]`` and lets the caller finish with
    the data that does exist.
    """
    try:
        payload = fetch_json(f'{OPENF1_BASE}/{endpoint}', params=params)
    except NotFound:
        logger.info('OpenF1 %s returned no rows for %s', endpoint, params)
        return []
    except RuntimeError as exc:
        if '404' in str(exc):
            logger.info('OpenF1 %s returned no rows for %s', endpoint, params)
            return []
        raise
    return payload if isinstance(payload, list) else []


# ---------------------------------------------------------------------------
# On-disk response cache
# ---------------------------------------------------------------------------

def _cache_key(url: str, params: Optional[Dict[str, Any]]) -> str:
    """Stable cache key for a URL plus its query parameters."""
    material = url + '?' + json.dumps(params or {}, sort_keys=True, default=str)
    return hashlib.sha1(material.encode('utf-8')).hexdigest()


def _cache_path(url: str, params: Optional[Dict[str, Any]]) -> Path:
    return CACHE_ROOT / f'{_cache_key(url, params)}.json'


def cached_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    ttl: int = CACHE_TTL_S,
    timeout: int = REQUEST_TIMEOUT_S,
    attempts: int = REQUEST_ATTEMPTS,
) -> Any:
    """Fetch JSON, serving a fresh on-disk copy when one exists.

    The cached file is written atomically (temp file + replace) so two
    processes warming the cache at the same time cannot produce a truncated
    file. A corrupt cache entry is discarded and refetched rather than
    propagated.
    """
    path = _cache_path(url, params)
    try:
        if path.exists() and (time.time() - path.stat().st_mtime) < ttl:
            with path.open('r', encoding='utf-8') as handle:
                return json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning('Discarding unusable cache entry %s: %s', path.name, exc)
        try:
            path.unlink()
        except OSError:
            pass

    payload = fetch_json(url, params=params, timeout=timeout, attempts=attempts)
    try:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        with tmp.open('w', encoding='utf-8') as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning('Could not cache %s: %s', url, exc)
    return payload


def _remember_missing(url: str, params: Optional[Dict[str, Any]]) -> None:
    """Remember that a query has no rows, so it is not asked again straight away.

    The archive does not have every session of every season (a sprint that has
    not been ingested yet, a driver without data). Without this, every season
    build asks for each of them again. The note lives for a few minutes, so a
    session that appears soon is picked up quickly.
    """
    path = _missing_path(url, params)
    try:
        _MISSING_ROOT.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        tmp.write_text('missing', encoding='utf-8')
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug('Could not remember the missing %s: %s', url, exc)


def _missing_path(url: str, params: Optional[Dict[str, Any]]) -> Path:
    return _MISSING_ROOT / f'{_cache_key(url, params)}.missing'


def _is_missing(url: str, params: Optional[Dict[str, Any]]) -> bool:
    """True when this query was answered with "no rows" recently."""
    path = _missing_path(url, params)
    try:
        return path.exists() and (time.time() - path.stat().st_mtime) < MISSING_TTL_S
    except OSError:
        return False


def cached_openf1(
    endpoint: str,
    ttl: int = CACHE_TTL_S,
    **params: Any,
) -> List[Dict[str, Any]]:
    """Cached variant of :func:`openf1` for the collection endpoints.

    A 404 still resolves to ``[]`` so an empty meeting or a driver without
    data behaves exactly like the uncached call.
    """
    url = f'{OPENF1_BASE}/{endpoint}'
    if _is_missing(url, params or None):
        return []
    try:
        payload = cached_json(url, params=params or None, ttl=ttl)
    except NotFound:
        logger.info('OpenF1 %s returned no rows for %s', endpoint, params)
        _remember_missing(url, params or None)
        return []
    except RuntimeError as exc:
        if '404' in str(exc):
            logger.info('OpenF1 %s returned no rows for %s', endpoint, params)
            return []
        raise
    return payload if isinstance(payload, list) else []
