# -*- coding: utf-8 -*-
"""Session weather, and a rain outlook for the session after it.

Two sources answer two different questions:

* :func:`session_weather` summarises what the OpenF1 ``/weather`` feed
  recorded *during* a session -- air and track temperature, humidity,
  rainfall and wind. That is measured data.
* :func:`forecast` asks Open-Meteo, a free public forecast service, for the
  hour of an upcoming session at the circuit's coordinates, and
  :func:`rain_outlook` turns that into a plain-language rain band with an
  honest confidence level.

Nothing here invents a number: an empty feed returns ``None`` (the page shows
an empty state), a network failure returns ``None`` (the page says the
forecast is unavailable), and every forecast figure is labelled
``estimated``.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
import re
import threading
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from openf1_client import cached_openf1

logger = logging.getLogger(__name__)

ESTIMATED = 'estimated'

OPEN_METEO_URL = 'https://api.open-meteo.com/v1/forecast'
OPEN_METEO_HOURLY = 'temperature_2m,precipitation_probability,wind_speed_10m,rain'
FORECAST_TIMEOUT_S = 6
FORECAST_ATTEMPTS = 1
FORECAST_NEAREST_HOURS = 3

# A rain probability is a forecast, so the band is a plan-ahead label, not a
# measurement. The boundaries are the ones the page states next to it.
RAIN_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ('dry', 0.0, 20.0),
    ('possible', 20.0, 60.0),
    ('likely', 60.0, 100.0),
)

# Circuits of the current calendar, keyed by the slug the matcher produces.
# The coordinates are the start/finish area of each circuit (Open-Meteo only
# needs the neighbourhood; the forecast is not sensitive to a few hundred
# metres).
CIRCUIT_COORDS: Dict[str, Tuple[float, float]] = {
    'australia': (-37.8497, 144.9680),
    'china': (31.3389, 121.2196),
    'japan': (34.8431, 136.5407),
    'bahrain': (26.0325, 50.5106),
    'saudi arabia': (21.6319, 39.1044),
    'miami': (25.9581, -80.2389),
    'imola': (44.3439, 11.7167),
    'monaco': (43.7347, 7.4206),
    'barcelona': (41.5700, 2.2611),
    # The 2026 Spanish Grand Prix moves to Madrid; use the city centre, which
    # is precise enough for a rain forecast.
    'madrid': (40.4168, -3.7038),
    'canada': (45.5000, -73.5228),
    'austria': (47.2197, 14.7647),
    'britain': (52.0786, -1.0169),
    'belgium': (50.4372, 5.9714),
    'hungary': (47.5789, 19.2486),
    'netherlands': (52.3888, 4.5409),
    'italy': (45.6156, 9.2811),
    'azerbaijan': (40.3725, 49.8533),
    'singapore': (1.2914, 103.8640),
    'usa': (30.1328, -97.6411),
    'mexico': (19.4042, -99.0907),
    'brazil': (-23.7036, -46.6997),
    'las vegas': (36.2715, -115.0104),
    'qatar': (25.4900, 51.4542),
    'abu dhabi': (24.4672, 54.6031),
}

# The event names in the calendar are "Australian Grand Prix" and "Marina
# Bay", not "Australia": match the adjective, the city and the circuit name
# too, so a name from either the FastF1 event or the OpenF1 meeting resolves.
_ALIASES: Dict[str, str] = {
    'australian': 'australia',
    'melbourne': 'australia',
    'albert park': 'australia',
    'chinese': 'china',
    'shanghai': 'china',
    'japanese': 'japan',
    'suzuka': 'japan',
    'bahraini': 'bahrain',
    'sakhir': 'bahrain',
    'saudi': 'saudi arabia',
    'saudi arabian': 'saudi arabia',
    'jeddah': 'saudi arabia',
    'miami gardens': 'miami',
    'emilia romagna': 'imola',
    'san marino': 'imola',
    'monte carlo': 'monaco',
    'catalunya': 'barcelona',
    'catalonia': 'barcelona',
    'catalan': 'barcelona',
    # No generic 'spanish'/'spain' alias: the Spanish Grand Prix was Barcelona
    # until 2025 and moves to Madrid in 2026, so the event name alone does not
    # identify the circuit. The page passes the schedule's Location instead.
    'madring': 'madrid',
    'ifema': 'madrid',
    'canadian': 'canada',
    'montreal': 'canada',
    'austrian': 'austria',
    'spielberg': 'austria',
    'red bull ring': 'austria',
    'british': 'britain',
    'great britain': 'britain',
    'silverstone': 'britain',
    'english': 'britain',
    'belgian': 'belgium',
    'spa': 'belgium',
    'francorchamps': 'belgium',
    'spa francorchamps': 'belgium',
    'hungarian': 'hungary',
    'budapest': 'hungary',
    'hungaroring': 'hungary',
    'dutch': 'netherlands',
    'zandvoort': 'netherlands',
    'italian': 'italy',
    'monza': 'italy',
    'azerbaijani': 'azerbaijan',
    'baku': 'azerbaijan',
    'marina bay': 'singapore',
    'singaporean': 'singapore',
    'united states': 'usa',
    'american': 'usa',
    'austin': 'usa',
    'circuit of the americas': 'usa',
    'cota': 'usa',
    'mexican': 'mexico',
    'mexico city': 'mexico',
    'brazilian': 'brazil',
    'sao paulo': 'brazil',
    'interlagos': 'brazil',
    'vegas': 'las vegas',
    'qatari': 'qatar',
    'lusail': 'qatar',
    'doha': 'qatar',
    'emirati': 'abu dhabi',
    'yas marina': 'abu dhabi',
}

_IMPLICATIONS: Dict[str, str] = {
    'dry': (
        'No rain is expected around the session hour, so slick tyres are the '
        'likely choice and the weather is unlikely to bring out a safety car.'
    ),
    'possible': (
        'A shower is possible around the session hour, so teams may keep '
        'intermediate and wet tyres ready; a shower would raise the chance of '
        'a safety car.'
    ),
    'likely': (
        'Rain is likely around the session hour, so wet or intermediate tyres '
        'are likely and a safety car or a red flag is a real risk.'
    ),
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _records(rows: Any) -> List[Dict[str, Any]]:
    """The rows as a list of dicts, from a list or a pandas frame."""
    if rows is None:
        return []
    if hasattr(rows, 'to_dict'):
        try:
            records = rows.to_dict('records')
        except (TypeError, ValueError):
            return []
        return [row for row in records if isinstance(row, dict)]
    try:
        return [row for row in rows if isinstance(row, dict)]
    except TypeError:
        return []


def _numbers(rows: Sequence[Dict[str, Any]], key: str,
             drop_zero: bool = False) -> List[float]:
    """The finite numeric values of one key, skipping everything else."""
    values: List[float] = []
    for row in rows:
        raw = row.get(key)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        if drop_zero and value == 0.0:
            continue
        values.append(value)
    return values


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _min(values: Sequence[float]) -> Optional[float]:
    return min(values) if values else None


def _max(values: Sequence[float]) -> Optional[float]:
    return max(values) if values else None


def _utc_datetime(value: Any) -> Optional[_dt.datetime]:
    """A timezone-aware UTC datetime, or None when the value is unusable."""
    if value is None:
        return None
    if hasattr(value, 'to_pydatetime'):        # pandas Timestamp
        try:
            value = value.to_pydatetime()
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith('Z'):
            text = text[:-1] + '+00:00'
        try:
            value = _dt.datetime.fromisoformat(text)
        except ValueError:
            try:
                value = _dt.datetime.strptime(text[:19], '%Y-%m-%dT%H:%M:%S')
            except ValueError:
                return None
    if not isinstance(value, _dt.datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=_dt.timezone.utc)
    return value.astimezone(_dt.timezone.utc)


def _iso_utc(value: Any) -> Optional[str]:
    moment = _utc_datetime(value)
    if moment is None:
        return None
    return moment.strftime('%Y-%m-%dT%H:%M:%SZ')


def _hour_utc(value: Any) -> Optional[_dt.datetime]:
    moment = _utc_datetime(value)
    if moment is None:
        return None
    return moment.replace(minute=0, second=0, microsecond=0)


def _normalise(name: Any) -> str:
    """A name reduced to lowercase words: 'Sao Paulo GP' -> 'sao paulo gp'."""
    text = unicodedata.normalize('NFKD', str(name or ''))
    text = text.encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def _phrase_in(text: str, phrase: str) -> bool:
    """Whether ``phrase`` appears in ``text`` as whole words."""
    pattern = r'(?<![a-z0-9])' + re.escape(phrase) + r'(?![a-z0-9])'
    return re.search(pattern, text) is not None


# ---------------------------------------------------------------------------
# Measured session weather (OpenF1)
# ---------------------------------------------------------------------------

def summarise_weather(rows: Any) -> Optional[Dict[str, Any]]:
    """Summarise the OpenF1 weather rows of one session.

    ``rows`` is the list of dicts from the ``/weather`` endpoint (or the
    equivalent frame). ``None`` is returned for an empty feed: the caller
    shows an empty state instead of an invented number.

    The feed reports a track temperature of exactly 0 degrees when the probe
    is unavailable (one such reading in the 2026 Singapore race). Those
    readings are dropped, counted in ``track_temp_zeros_dropped`` and never
    applied to the air temperature, where 0 degrees is a real reading.
    """
    records = _records(rows)
    if not records:
        return None

    air = _numbers(records, 'air_temperature')
    track_all = _numbers(records, 'track_temperature')
    track = [value for value in track_all if value != 0.0]
    humidity = _numbers(records, 'humidity')
    pressure = _numbers(records, 'pressure')
    wind = _numbers(records, 'wind_speed')
    rain = _numbers(records, 'rainfall')

    stamps = [_utc_datetime(record.get('date')) for record in records]
    stamps = [stamp for stamp in stamps if stamp is not None]

    rain_samples = sum(1 for value in rain if value > 0.0)
    wet = None if not rain else rain_samples > 0
    wind_max = _max(wind)

    def _key(records_key: str) -> Any:
        for record in records:
            value = record.get(records_key)
            if value is not None:
                return value
        return None

    return {
        'session_key': _key('session_key'),
        'meeting_key': _key('meeting_key'),
        'samples': len(records),
        'started_at': _iso_utc(min(stamps)) if stamps else None,
        'ended_at': _iso_utc(max(stamps)) if stamps else None,
        'air_temp_mean': _mean(air),
        'air_temp_min': _min(air),
        'air_temp_max': _max(air),
        'track_temp_mean': _mean(track),
        'track_temp_min': _min(track),
        'track_temp_max': _max(track),
        'track_temp_zeros_dropped': len(track_all) - len(track),
        'humidity_mean': _mean(humidity),
        'humidity_min': _min(humidity),
        'humidity_max': _max(humidity),
        'pressure_mean': _mean(pressure),
        'rainfall_total': sum(rain) if rain else None,
        'rainfall_min': _min(rain),
        'rainfall_max': _max(rain),
        'rain_samples': rain_samples,
        'wet': wet,
        'wind_speed_mean': _mean(wind),
        'wind_speed_min': _min(wind),
        'wind_speed_max': wind_max,
        'wind_speed_max_kmh': wind_max * 3.6 if wind_max is not None else None,
    }


def session_weather(session_key: Any) -> Optional[Dict[str, Any]]:
    """The measured weather of one session, or None when it has none.

    The request goes through the shared OpenF1 client (rate limiter and disk
    cache); a missing session or an unavailable endpoint is answered with
    ``None`` rather than an exception.
    """
    try:
        key = int(session_key)
    except (TypeError, ValueError):
        return None
    try:
        rows = cached_openf1('weather', session_key=key)
    except Exception as exc:                    # noqa: BLE001 - never raise
        logger.warning('OpenF1 weather unavailable for session %s: %s', key, exc)
        return None
    return summarise_weather(rows)


# ---------------------------------------------------------------------------
# Forecast (Open-Meteo)
# ---------------------------------------------------------------------------

def coords_for(track_name: Any) -> Optional[Tuple[float, float]]:
    """The (lat, lon) of a circuit, matched tolerantly, or None.

    The matcher normalises case, punctuation and accents, then looks for the
    longest circuit key or alias as whole words: 'Singapore Grand Prix',
    'marina bay' and 'SINGAPORE' all resolve to the same circuit, while an
    unknown name returns None so the caller can say so.
    """
    text = _normalise(track_name)
    if not text:
        return None
    if text in CIRCUIT_COORDS:
        return CIRCUIT_COORDS[text]
    if text in _ALIASES:
        return CIRCUIT_COORDS.get(_ALIASES[text])

    best: Optional[Tuple[str, str]] = None
    for phrase, key in _ALIASES.items():
        if _phrase_in(text, phrase) and (best is None or len(phrase) > len(best[0])):
            best = (phrase, key)
    for key in CIRCUIT_COORDS:
        if _phrase_in(text, key) and (best is None or len(key) > len(best[0])):
            best = (key, key)
    if best is None:
        return None
    return CIRCUIT_COORDS.get(best[1])


def _day_payload(coords: Tuple[float, float],
                 day: _dt.date) -> Optional[Dict[str, Any]]:
    """One day of Open-Meteo hourly data, cached in-process.

    A failure is cached as None too: a page rerun does not retry a network
    call that just failed, and nothing here raises at the caller.
    """
    key = (round(float(coords[0]), 3), round(float(coords[1]), 3), day.isoformat())
    with _FORECAST_LOCK:
        if key in _FORECAST_CACHE:
            return _FORECAST_CACHE[key]

    try:
        response = requests.get(
            OPEN_METEO_URL,
            params={
                'latitude': key[0],
                'longitude': key[1],
                'hourly': OPEN_METEO_HOURLY,
                'timezone': 'UTC',
                'start_date': key[2],
                'end_date': (day + _dt.timedelta(days=1)).isoformat(),
            },
            timeout=FORECAST_TIMEOUT_S,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get('hourly'), dict):
            payload = None
    except Exception as exc:                    # noqa: BLE001 - never raise
        logger.warning('Open-Meteo forecast unavailable for %s: %s', key[:2], exc)
        payload = None

    with _FORECAST_LOCK:
        _FORECAST_CACHE[key] = payload
    return payload


def clear_forecast_cache() -> None:
    """Forget the in-process forecast cache (used by the tests)."""
    with _FORECAST_LOCK:
        _FORECAST_CACHE.clear()


def parse_forecast(payload: Any, when_utc: Any) -> Optional[Dict[str, Any]]:
    """The forecast for the hour of ``when_utc`` out of an Open-Meteo payload.

    Returns None when the payload has no usable hourly block, and skips a
    value the service left empty instead of substituting a number.
    """
    hour = _hour_utc(when_utc)
    if hour is None or not isinstance(payload, dict):
        return None
    hourly = payload.get('hourly')
    if not isinstance(hourly, dict):
        return None
    times = hourly.get('time')
    if not isinstance(times, (list, tuple)) or not times:
        return None

    target = hour.strftime('%Y-%m-%dT%H:00')
    index: Optional[int] = None
    try:
        index = list(times).index(target)
    except ValueError:
        best: Optional[Tuple[float, int]] = None
        for position, raw in enumerate(times):
            moment = _utc_datetime(raw)
            if moment is None:
                continue
            distance = abs((moment - hour).total_seconds())
            if best is None or distance < best[0]:
                best = (distance, position)
        if best is not None and best[0] <= FORECAST_NEAREST_HOURS * 3600:
            index = best[1]
    if index is None:
        return None

    def value_at(name: str) -> Optional[float]:
        series = hourly.get(name)
        if not isinstance(series, (list, tuple)) or index >= len(series):
            return None
        try:
            value = float(series[index])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return value

    temperature = value_at('temperature_2m')
    probability = value_at('precipitation_probability')
    wind_speed = value_at('wind_speed_10m')
    rain_mm = value_at('rain')
    if temperature is None and probability is None and wind_speed is None:
        return None

    return {
        'time_utc': str(times[index]),
        'temperature_c': temperature,
        'precipitation_probability': (
            int(round(probability)) if probability is not None else None
        ),
        'wind_speed_kmh': wind_speed,
        'rain_mm': rain_mm,
        'latitude': payload.get('latitude'),
        'longitude': payload.get('longitude'),
        'source': 'Open-Meteo',
        'estimated': True,
    }


def forecast(track_name: Any, when_utc: Any) -> Optional[Dict[str, Any]]:
    """The Open-Meteo forecast for the hour of one session, or None.

    None means "not available" -- the circuit is not in the coordinate table,
    the time is unusable, or the service did not answer. It never raises.
    """
    coords = coords_for(track_name)
    hour = _hour_utc(when_utc)
    if coords is None or hour is None:
        return None
    payload = _day_payload(coords, hour.date())
    if payload is None:
        return None
    return parse_forecast(payload, hour)


def rain_band(probability: Any) -> Optional[str]:
    """The rain band of a probability: dry, possible or likely."""
    if probability is None:
        return None
    try:
        value = float(probability)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    value = min(max(value, 0.0), 100.0)
    for label, low, high in RAIN_BANDS:
        if low <= value < high:
            return label
    return 'likely'


def rain_outlook(track_name: Any, when_utc: Any,
                 now: Any = None) -> Optional[Dict[str, Any]]:
    """A rain outlook for one session hour, labelled as an estimate.

    The confidence is about the forecast horizon, not about the circuit: a
    session inside the next three days gets ``moderate``, anything further
    ahead ``low``. A horizon that cannot be computed is ``low`` as well, so
    the label is never optimistic without evidence.
    """
    data = forecast(track_name, when_utc)
    if not data:
        return None
    probability = data.get('precipitation_probability')
    if probability is None:
        return None
    band = rain_band(probability)
    if band is None:
        return None

    hour = _hour_utc(when_utc)
    reference = _utc_datetime(now) if now is not None else _dt.datetime.now(_dt.timezone.utc)
    horizon_hours: Optional[float] = None
    if hour is not None and reference is not None:
        horizon_hours = (hour - reference).total_seconds() / 3600.0
    if horizon_hours is not None and horizon_hours < 72:
        confidence = 'moderate'
    else:
        confidence = 'low'

    return {
        'band': band,
        'probability': float(probability),
        'confidence': confidence,
        'horizon_hours': horizon_hours,
        'implication': _IMPLICATIONS[band],
        'estimated': True,
        'source': 'Open-Meteo forecast',
        'track_name': str(track_name),
        'when_utc': hour.strftime('%Y-%m-%dT%H:%M:%SZ') if hour is not None else None,
        'temperature_c': data.get('temperature_c'),
        'wind_speed_kmh': data.get('wind_speed_kmh'),
    }


# The in-process cache of one day of hourly data per (circuit, day).
_FORECAST_CACHE: Dict[Tuple[float, float, str], Optional[Dict[str, Any]]] = {}
_FORECAST_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Live rain radar (RainViewer)
# ---------------------------------------------------------------------------
#
# RainViewer publishes a free, keyless tile set: the index document lists the
# frames it has, and each frame is a 256 px transparent PNG overlay in Web
# Mercator. Two limits shape this module:
#
#   * the free tier stops at zoom 7 (a request for zoom 8 comes back as a
#     "Zoom Level Not Supported" placeholder, not an error), and
#   * a new frame appears every ~10 minutes, not every 5, so the index is
#     polled on the panel's five-minute cadence and the tiles are downloaded
#     by the browser only when the frame path changes.
#
# Nothing here downloads a tile: the renderer in ``assets/radar_scope.html``
# does that client-side, so a Streamlit rerun costs one 1.5 KB index request at
# most.

RADAR_INDEX_URL = 'https://api.rainviewer.com/public/weather-maps.json'
RADAR_TIMEOUT_S = 8
RADAR_INDEX_TTL_S = 300          # the panel polls the index every five minutes
RADAR_FAILURE_TTL_S = 60         # a failed poll is not retried on every rerun
RADAR_MIN_ZOOM = 4
RADAR_MAX_ZOOM = 7               # the free tier caps here
RADAR_TILE_SIZE = 256
RADAR_COLOR_SCHEME = 4           # RainViewer "Universal Blue"
RADAR_OPTIONS = '1_1'            # smooth (1) + snow (1)
RADAR_FRAME_LIMIT = 13           # the whole published history (2 h)
RADAR_VIEW_WIDTH_PX = 760        # the width the caption's span is quoted for
RADAR_RINGS_KM = (50, 100, 200)
RADAR_ATTRIBUTION = 'Weather data by RainViewer'
RADAR_ATTRIBUTION_URL = 'https://www.rainviewer.com/api.html'
BASEMAP_ATTRIBUTION = 'OpenStreetMap contributors'
BASEMAP_ATTRIBUTION_URL = 'https://www.openstreetmap.org/copyright'

# Earth circumference at the equator divided into 256 px tiles, per degree.
_WORLD_CIRCUMFERENCE_PX = 256.0


def _radar_label(stamp: int) -> str:
    """A frame time as HH:MM UTC, the clock the rest of the page uses."""
    return _dt.datetime.fromtimestamp(stamp, _dt.timezone.utc).strftime('%H:%M')


def parse_radar_index(payload: Any,
                      now: Any = None) -> Optional[Dict[str, Any]]:
    """The usable frames out of a RainViewer index document, or None.

    Only a frame that carries both a path (which must be a site-absolute path,
    so it cannot point at another host) and a timestamp is kept. The frames
    are sorted oldest first and trimmed to the published history, so the
    renderer can treat the last one as the current picture.
    """
    if not isinstance(payload, dict):
        return None
    host = payload.get('host')
    radar = payload.get('radar')
    if not isinstance(host, str) or not host.strip():
        return None
    if not isinstance(radar, dict):
        return None
    past = radar.get('past')
    if not isinstance(past, (list, tuple)) or not past:
        return None

    frames: List[Dict[str, Any]] = []
    for item in past:
        if not isinstance(item, dict):
            continue
        path = item.get('path')
        stamp = item.get('time')
        if not isinstance(path, str) or not path.startswith('/'):
            continue
        try:
            stamp = int(stamp)
        except (TypeError, ValueError):
            continue
        frames.append({'path': path, 'time': stamp, 'label': _radar_label(stamp)})
    if not frames:
        return None

    frames.sort(key=lambda frame: frame['time'])
    frames = frames[-RADAR_FRAME_LIMIT:]
    latest = frames[-1]
    generated = payload.get('generated')
    try:
        generated = int(generated)
    except (TypeError, ValueError):
        generated = latest['time']

    reference = (_utc_datetime(now) if now is not None
                 else _dt.datetime.now(_dt.timezone.utc))
    age_s = None
    if reference is not None:
        age_s = max(0.0, reference.timestamp() - float(latest['time']))
    observed = _dt.datetime.fromtimestamp(latest['time'], _dt.timezone.utc)

    return {
        'host': host.strip().rstrip('/'),
        'frames': frames,
        'count': len(frames),
        'latest': latest['path'],
        'generated': generated,
        'observed_at': _iso_utc(observed),
        'span_min': (latest['time'] - frames[0]['time']) / 60.0,
        'age_s': age_s,
        'age_min': (age_s / 60.0) if age_s is not None else None,
        'min_zoom': RADAR_MIN_ZOOM,
        'max_zoom': RADAR_MAX_ZOOM,
        'tile_size': RADAR_TILE_SIZE,
        'color_scheme': RADAR_COLOR_SCHEME,
        'options': RADAR_OPTIONS,
        'rings_km': list(RADAR_RINGS_KM),
        'attribution': RADAR_ATTRIBUTION,
        'attribution_url': RADAR_ATTRIBUTION_URL,
        'basemap_attribution': BASEMAP_ATTRIBUTION,
        'basemap_attribution_url': BASEMAP_ATTRIBUTION_URL,
        'source': 'RainViewer radar',
        # Radar is an observation, not a forecast, so it is not labelled as an
        # estimate -- but it is a regional picture, which the panel says.
        'measured': True,
    }


def radar_frames(now: Any = None) -> Optional[Dict[str, Any]]:
    """The radar frame index, cached for five minutes.

    A failed poll is cached for a minute, so a service outage costs one
    request per minute rather than one per rerun. Returns None rather than
    raising, like the rest of this module.
    """
    reference = (_utc_datetime(now) if now is not None
                 else _dt.datetime.now(_dt.timezone.utc))
    stamp = reference.timestamp() if reference is not None else None

    with _RADAR_LOCK:
        cached = _RADAR_CACHE.get('index')
        if cached is not None and stamp is not None:
            fetched_at, cached_payload = cached
            keep = (RADAR_INDEX_TTL_S if cached_payload is not None
                    else RADAR_FAILURE_TTL_S)
            if 0.0 <= stamp - fetched_at < keep:
                return cached_payload

    payload: Optional[Dict[str, Any]] = None
    try:
        response = requests.get(
            RADAR_INDEX_URL,
            timeout=RADAR_TIMEOUT_S,
            headers={'User-Agent': 'f1lab/weather-radar'},
        )
        response.raise_for_status()
        payload = parse_radar_index(response.json(), reference)
    except Exception as exc:                    # noqa: BLE001 - never raise
        logger.warning('RainViewer radar index unavailable: %s', exc)
        payload = None

    if stamp is not None:
        with _RADAR_LOCK:
            _RADAR_CACHE['index'] = (stamp, payload)
    return payload


def clear_radar_cache() -> None:
    """Forget the in-process radar index (used by the tests)."""
    with _RADAR_LOCK:
        _RADAR_CACHE.clear()


def metres_per_pixel(zoom: int, latitude: float) -> float:
    """Ground resolution of the tile scheme at a latitude, in metres a pixel."""
    zoom = int(zoom)
    return (156543.03392 * math.cos(math.radians(float(latitude)))
            / (2 ** zoom))


def radar_span_km(zoom: int, latitude: float,
                  width_px: int = RADAR_VIEW_WIDTH_PX) -> float:
    """How many kilometres across a view of that width covers, at that zoom."""
    return metres_per_pixel(zoom, latitude) * float(width_px) / 1000.0


def radar_view(track_name: Any, now: Any = None,
               zoom: int = RADAR_MAX_ZOOM) -> Optional[Dict[str, Any]]:
    """Everything the radar panel needs to draw the circuit's surroundings.

    Coordinates come from the circuit table, frames from the cached index, and
    the tiles are fetched by the browser. Returns None for an unknown circuit
    or an unavailable service, so the page can say so instead of drawing an
    empty map.
    """
    coords = coords_for(track_name)
    if coords is None:
        return None
    index = radar_frames(now)
    if index is None:
        return None

    latitude, longitude = float(coords[0]), float(coords[1])
    view = dict(index)
    view.update({
        'track_name': str(track_name),
        'center': {'lat': latitude, 'lon': longitude},
        'zoom': max(RADAR_MIN_ZOOM, min(int(zoom), RADAR_MAX_ZOOM)),
        'span_km': radar_span_km(zoom, latitude),
    })
    return view


# The in-process cache of the radar index: (fetched_at_epoch, payload or None).
_RADAR_CACHE: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}
_RADAR_LOCK = threading.Lock()
