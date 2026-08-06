# F1 Lab -- Development Guide

## Project Overview

Real-time Formula 1 analytics dashboard for the **2025 season**.
Built with Streamlit, FastF1 API, and Plotly.

```
streamlit run src/f1.py      # Main multi-page app
```

## Architecture

```
src/
  f1.py                     # Entry point: page config, sidebar, st.navigation()
  shared.py                 # Shared utilities (data loading, cache, helpers)
  season_config.py          # Dynamic calendar via FastF1 API + fallbacks
  config.py                 # Static data: teams, drivers, profiles, MODEL_CONFIG (2025)
  seasons.py                # Season discovery from data/ (discover_available_years)
  loader.py                 # CSV loading + cleaning
  analysis.py               # Stats aggregation
  model.py                  # Race strategy simulator (uses MODEL_CONFIG)
  prediction.py             # Race predictor + fantasy points (weighted model)
  pace.py                   # Normalised pace, long-run, degradation, data_quality()
  whatif.py                 # What-if strategy scenarios + undercut (R2: traffic/rival)
  backtest.py               # Walk-forward backtest vs grid-order baseline (P3)
  briefing.py               # Pre-race briefing / post-race debrief text (P4)
  rival.py                  # Head-to-head, watchlist, circuit insights (R1)
  sector.py                 # Sector/corner insights from lap sector times (R3)
  compare.py                # Session-to-session pace comparison (R3)
  fastf1_extended.py        # FastF1 data extraction (weather, pits, sectors, etc.)
  advanced_viz.py           # Plotly telemetry comparison
  assets/
    style.css               # Global CSS (glassmorphism dark theme)
  pages/
    1_Home.py               # Home dashboard (+ About & Limitations)
    2_HeadToHead.py         # Head-to-head comparisons
    3_Predictor.py          # Race predictor + model accuracy/backtest tab
    4_Report.py             # Pre-race briefing / post-race debrief
    5_Replay.py             # Race replay
    6_Analysis.py           # Race analysis center (Pace, Strategy, Battles, Insights, Sector & Sessions)
    7_Competitor.py         # Rival watchlist, head-to-head, circuit insights
```

## Multi-Season Support

- **Sidebar selector**: `st.session_state.selected_year` (set in `f1.py`)
- **Season discovery**: `seasons.discover_available_years(data_dir)` scans `data/Formula1_{year}Season_RaceResults.csv`; `config.py` derives `FASTF1_CONFIG.default_year` and `max_supported_year` from it (falls back to seed 2025 if `data/` is empty)
- **Calendar**: `season_config.get_completed_races(year)` -- fetches live from FastF1, cached in-memory
- **CSV naming**: `Formula1_{year}Season_RaceResults.csv` -- year injected dynamically
- **FastF1 sessions**: Always pass `st.session_state.selected_year` as first arg
- **Fallback**: `season_config.FALLBACK_CALENDARS` (keyed by year) used when FastF1 API is unavailable

### Adding a new season (e.g. 2026)
1. Place CSV files in `data/Formula1_2026Season_*.csv` -- the app auto-detects the new year
2. Add a fallback calendar entry to `season_config.py` `FALLBACK_CALENDARS` (optional; only used offline)
3. Add teams/drivers to `config.py` (`F1_2026_TEAMS`, `DRIVER_PROFILES`, `DRIVER_DETAILS`)

## Data Honesty & Explainability (primary rule)

Never fabricate, hide, or overstate data. Conventions across all modules:

- Every derived figure is labelled `estimated` (constant `ESTIMATED` per module) and shows
  a **confidence level** from the underlying sample via `pace.data_quality(n)`.
- **Sample thresholds**: `good` ≥ 15 clean laps, `moderate` ≥ 8, `low` ≥ 2, `insufficient`
  < 2. Below a fit threshold, modules return an empty frame / `None` / an
  "insufficient data" message instead of recommending on noise.
- Analysis uses **clean laps only** (`pace.clean_laps_for_analysis`: out/in laps, Safety Car
  windows, inaccurate laps removed).
- Every strategy/prediction recommendation shows expected outcome, risk, confidence,
  assumptions and reasons. Predictions and strategy are validated by
  **walk-forward backtesting** (`src/backtest.py`, trains each race only on prior races —
  no look-ahead) against a grid-order baseline; weaknesses are reported, not hidden.

## Honest Positioning

F1 Lab is **not** a race-critical tool and **not** a substitute for Mercedes/Red Bull
(or any team's) internal data. It is a public-data analytics workbench for post-session
analysis, strategy rehearsal, race briefing and competitor research. No readout here
includes team telemetry, setup, or car information; do not present its outputs as a
team's operational view. The Home page carries this framing in "About & Limitations".

## Testing

- **Test behavior, not implementation.** New modules ship with tests under `tests/`
  using synthetic data (see `tests/test_sector.py`, `test_compare.py`, `test_whatif.py`,
  `test_backtest.py`, `test_briefing.py`, `test_rival.py`).
- Run relevant tests during work and the full suite before pushing:
  `PYTHONPATH=src .venv/Scripts/python -m pytest -q`
- Lint: `.venv/Scripts/python -m ruff check --select F <file>` (F-rules only; the
  `# -*- coding: utf-8 -*-` header and `Optional` typing are the codebase idiom and stay).
- Model/prediction/strategy changes MUST be verified via historical backtesting before
  claiming improvement (compare against the existing baseline, report honestly).

## Performance Optimizations

- `@st.cache_data(ttl=3600)` on data loading functions
- `@st.cache_resource(ttl=3600)` on FastF1 session loader
- In-memory `_schedule_cache` dict for `fastf1.get_event_schedule()` calls
- FastF1 disk cache in `f1_cache/` directory

## Key Conventions

- **Page files** go in `src/pages/` -- each has a single `page()` function
- **Shared utilities** live in `src/shared.py` -- import from there, not from `f1.py`
- **All session data** loaded via `shared.load_race_data(year)` or `fastf1.get_session(year, ...)`
- **Plotly charts** use `shared.show_plotly_chart()` -- hides toolbar, applies dark theme
- **No emojis** in page titles or headers -- professional appearance
- **Dark theme**: CSS in `assets/style.css` -- don't add inline `<style>` blocks

## Dependencies

```
pandas, numpy, streamlit, fastf1, plotly, matplotlib
```
