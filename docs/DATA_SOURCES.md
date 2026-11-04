# F1 LAB — free data sources, priority and provenance

F1 LAB uses **free and open data sources only**: no paid subscriptions,
no API keys that require a paid plan, no fabricated statistics, no fake
live telemetry. This document is the contract every module follows;
`src/sources.py` encodes the same table in code.

## Source priority and fallbacks

| Dataset | Primary source | Fallback |
|---|---|---|
| Race results | FastF1 | Jolpica |
| Lap times | FastF1 | OpenF1 |
| Telemetry | FastF1 | OpenF1 |
| Tyre stints | FastF1 | OpenF1 |
| Pit stops | OpenF1 | Jolpica |
| Standings | Jolpica | Computed from results (labelled *estimated*) |
| Weather history | FastF1 | OpenF1 |
| Weather forecast | Open-Meteo | Unavailable (stated, never fabricated) |
| Rain radar | RainViewer | Unavailable (stated) |
| Race replay | OpenF1 / FastF1 positions | Unavailable when no position data exists |
| Championship points | OpenF1 season results | Jolpica |

Datasets are never merged blindly. Every returned dataset carries
`source_name`, `fetched_at` and a `data_status`:

- **Verified** — measured, cross-checked or official-style source
- **Available** — measured, single source
- **Partial** — some fields or rounds missing
- **Estimated** — derived estimate, always labelled in the UI
- **Unavailable** — no source could provide the data; the page says so

A measured value is never silently replaced by a forecast value.

## The sources

- **FastF1** (`src/shared.py`, session loading) — session results, laps,
  sectors, tyres, pit stops, telemetry, weather, schedules. The
  official timing archive may be closed to anonymous requests; the app
  probes it once per process and falls back to the OpenF1 rebuild,
  labelled with its provenance.
- **OpenF1** (`src/openf1_client.py`, `src/openf1_results.py`,
  `src/openf1_session.py`) — meetings, sessions, drivers, laps,
  position, intervals, pit, weather. Free historical tier from 2023
  onwards (3 requests/second, 30/minute). No real-time data; nothing
  bypasses paid authentication.
- **Jolpica-F1** (`src/jolpica.py`) — the free Ergast successor
  (`https://api.jolpi.ca/ergast/f1/`): seasons, circuits, driver and
  constructor standings, classifications. Historical tables are cached
  on disk (`f1_cache/jolpica/`); the current season refreshes every
  5 minutes. The retired Ergast hostname is never used.
- **Open-Meteo** (`src/wx.py`) — free non-commercial forecasts (air
  temperature, humidity, wind, precipitation probability, weather
  codes) at circuit coordinates. Hourly data; 15-minute series only
  where the provider offers them — no fabricated 15-minute values.
- **RainViewer** (`src/wx.py`, `src/radar.py`) — radar index and frames
  (`https://api.rainviewer.com/public/weather-maps.json`) with
  attribution. Coverage varies by region; absence is stated.

## Caching and rate limits

- FastF1 disk cache in `f1_cache/`; OpenF1 disk cache with a one-day
  TTL for finished sessions and 5 minutes for live ones.
- Jolpica disk cache in `f1_cache/jolpica/` with the same policy.
- HTTP retries with backoff; a missing resource (404) is "no rows",
  never a failure.

## Endpoint mapping (service layer = the API)

The Streamlit pages call the service layer directly; the functions
below are the de-facto endpoints (same inputs/outputs, no HTTP hop):

| Endpoint (brief) | Service function |
|---|---|
| `GET /api/seasons` | `jolpica.seasons()` |
| `GET /api/meetings`, `/api/sessions` | `season_config.get_season_calendar`, `shared.load_fastf1_session` |
| `GET /api/drivers` | `championship.driver_meta`, `openf1_results.drivers_for_session` |
| `GET /api/race/overview` | `race_analysis.race_overview` |
| `GET /api/race/results` | `race_analysis.results_table` |
| `GET /api/race/laps` | `race_analysis.lap_mode_filter` |
| `GET /api/race/telemetry` | `race_analysis.align_telemetry`, `race_analysis.telemetry_summary` |
| `GET /api/race/sectors` | `sector.sector_summary`, `race_analysis.theoretical_best` |
| `GET /api/race/stints` | `race_analysis.stint_timeline`, `race_analysis.degradation_slope` |
| `GET /api/race/pit-stops` | `race_analysis.pit_summary` |
| `GET /api/race/positions` | `race_analysis.position_changes`, `race_analysis.conservative_overtakes` |
| `GET /api/race/pace` | `race_analysis.pace_distribution` |
| `GET /api/race/insights` | `race_analysis.automated_insights` |
| `GET /api/weather/*` | `wx.session_weather`, `wx.forecast`, `wx.radar_view` |
| `GET /api/standings/*` | `jolpica.driver_standings`, `jolpica.constructor_standings` |

## Phase coverage

- Phase 1 (sources + caching): `sources.py`, `jolpica.py`,
  `openf1_client.py`, `wx.py`, `radar.py`.
- Phase 2 (normalisation + validation): `loader.clean_data`,
  `race_analysis.validate_laps` / `data_status`.
- Phase 3 (race analysis): `race_analysis.py` + Analysis Center tabs
  (pace, strategy, battles, insights, sector, setup/telemetry).
- Phase 4 (weekend report + weather): Weekend Report + `wx.py`
  (measured / forecast / radar separated, provenance in every payload).
- Phase 5 (championship): Championship Race tab + `championship.py` +
  Jolpica standings + `permutations.title_clinch_checks`.
- Phase 6 (replay): Race Replay + `telemetry.py` / `replay_payload.py`
  (SQLite cache, sampling and missing-data honesty).
- Phase 7 (validation, tests, export): `tests/` suite, CSV export in
  the Pace tab, downloadable weekend report.

## Architecture note

The brief's `backend/` FastAPI layout was evaluated and deliberately
**not** adopted: the Streamlit app already centralises every source
behind service modules (`shared`, `openf1_*`, `jolpica`, `wx`,
`radar`), so a second server would duplicate the data layer and break
the working app. New sources are added as service modules under
`src/` with provenance, then consumed by pages — the same separation
the brief asks for, without a rewrite.
