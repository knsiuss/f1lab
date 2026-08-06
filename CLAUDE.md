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
  config.py                 # Static data: teams, drivers, profiles (2025)
  seasons.py                # Season discovery from data/ (discover_available_years)
  loader.py                 # CSV loading + cleaning
  analysis.py               # Stats aggregation
  model.py                  # Race strategy simulator
  fastf1_extended.py        # FastF1 data extraction (weather, pits, sectors, etc.)
  advanced_viz.py           # Plotly telemetry comparison
  assets/
    style.css               # Global CSS (glassmorphism dark theme)
  pages/
    1_Home.py               # Home dashboard
    6_Analysis.py           # Race analysis center
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
