<h1 align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg" width="60" alt="F1">
  <br>
  F1 Lab
</h1>

<p align="center">
  Real-time Formula 1 analytics dashboard for the 2025 and 2026 seasons.<br>
  Built with Streamlit, the OpenF1 and FastF1 APIs, and Plotly.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.31+-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Data-OpenF1-E10600" alt="OpenF1">
  <img src="https://img.shields.io/badge/Data-FastF1-FF6B00" alt="FastF1">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

##  Features

###  Built to be read
The app is written for people who do not follow Formula 1 closely:

- **One story per screen** — every dashboard opens with a hero band that says what is
  happening in one plain sentence, and every chart or table is followed by a
  *What this means* line
- **Plain-language sections** — every section explains itself in one line under its
  title, and the words of the sport ("stint", "undercut", "rolling average") are
  defined in one shared glossary at the bottom of every page
- **A guide for first-time visitors** — the sidebar carries a *New to F1? Start here*
  button that explains the sport in four steps
- **Two languages for the explanation layer** — the sidebar's *Explainer language*
  selector switches the glossary and the newcomer guide between English and Bahasa
  Indonesia (`src/glossary_id.py`, same 30 terms, keyed alike). The data labels and
  charts stay in English on purpose, so a page is never half translated
- **Honesty in plain words** — anything calculated rather than reported is marked
  *estimated*, with a confidence level that follows the size of the sample, and the
  model publishes its own weakness next to its forecast
- **One visual system** — the same component library (`src/ui.py`), the same chart
  height scale and the same table style on every page
- **A phone layout** — the same build at 390 px: the menu starts collapsed behind the
  hamburger, KPI tiles go two-up, tables scroll, and the tab bar scrolls sideways

###  Season Analytics
- **Championship Standings** — Driver and constructor tables from the OpenF1 results feed
  (race **and** sprint points)
- **Points Progression** — Championship points accumulated round by round
- **Latest Race Result** — The classification of the most recent Grand Prix
- **Season Totals** — Points, races run and the leaders, in one KPI row

###  Race Center
- **Race Analysis** — Lap times, position changes, gap-to-leader, pit stops and tyre stints
- **Strategy Simulator** — What-if tyre plans with expected time, risk, confidence and the
  assumptions behind the number
- **Sector Insights** — Per-sector deficits against the session best, with a clean-lap count
- **Session Comparison** — Pace delta between two sessions (like-for-like compounds only)
- **Setup & Aero** — A setup proxy from public telemetry: trap speed, cornering buckets and a
  shift detector between two sessions

###  Race Replay
- **Broadcast Player** — Scrub a whole race on a track map: timing tower, tyre stints, gaps and race control
- **OpenF1 Telemetry** — 4 Hz speed, throttle, brake, DRS and gear traces per driver
- **Driver Selection** — Click a car in the timing tower to focus on it
- **Cached Ingest** — A race is downloaded once from OpenF1 into a local SQLite cache and replayed from there
- **Honest about what is measured** — Lap times, gaps, stints and race order are measured; the
  position *on track* is an estimate from lap progress, and OpenF1's telemetry coverage is
  partial (a driver without telemetry is labelled, not shown reading zero)

###  Predictions & Strategy
- **Race Forecast** — A transparent heuristic (grid 40%, team pace 35%, form 25%), not a
  black box: the model's own weakness is published next to the numbers
- **Walk-forward Backtest** — Every race is scored with a model trained only on the rounds
  before it, against a naive "the result is the grid" baseline
- **Fantasy Lineup** — A budget-constrained greedy team builder on the predicted points

###  Competitor Research
- **Rival Watchlist** — Who is close, with the quality and confidence of the sample behind it
- **Head-to-Head** — Race record, qualifying record and average finishing position per pair
- **Circuit Insights** — The winners and pole sitters at a circuit, with the race count and
  the confidence of that sample

---

##  Quick Start

```bash
# Clone
git clone https://github.com/maxvyquincy9393/f1lab.git
cd f1lab

# Install
pip install -r requirements.txt

# Run
streamlit run src/f1.py
```

Open **http://localhost:8501** in your browser.

---

##  Project Structure

```
f1lab/
├── src/                    # Source code
│   ├── f1.py               # Main Streamlit application (entry point)
│   ├── shared.py           # Shared utilities (data loading, cache, chart theme)
│   ├── config.py           # Configuration and constants (MODEL_CONFIG, teams, drivers)
│   ├── loader.py           # Frame normalisation (clean_data, race tracks)
│   ├── openf1_client.py    # OpenF1 HTTP client: rate limiter, retries, disk cache
│   ├── openf1_results.py   # Season/sprint results and grids from the OpenF1 API
│   ├── openf1_session.py   # Session fallback: rebuilds a FastF1-shaped session from OpenF1
│   ├── analysis.py         # Statistical calculations
│   ├── model.py            # Race strategy simulator
│   ├── prediction.py       # Race predictor + fantasy points
│   ├── backtest.py         # Walk-forward backtest vs grid baseline
│   ├── whatif.py           # Strategy what-if scenarios + undercut
│   ├── pace.py             # Normalised pace, degradation, data-quality gates
│   ├── pitwindow.py        # Pit-window radar per driver
│   ├── setupfingerprint.py # Setup fingerprint from telemetry (aero/trade-off proxy)
│   ├── briefing.py         # Pre-race briefing / post-race debrief
│   ├── rival.py            # Rival watchlist, head-to-head, circuit insights
│   ├── sector.py           # Sector insights from lap sector times
│   ├── compare.py          # Session-to-session pace comparison
│   ├── fastf1_extended.py  # FastF1 extraction (weather, pits, sectors, telemetry)
│   ├── telemetry.py        # OpenF1 telemetry ingest into SQLite + position estimate
│   ├── replay.py           # Animated position replay figure
│   ├── replay_payload.py   # Session -> compact JSON payload for the player
│   ├── report.py           # Markdown weekend report
│   ├── styles.py           # Plotly design tokens
│   ├── ui.py               # Shared UI components (hero, sections, KPI tiles, tables, glossary)
│   ├── glossary.py         # Plain-language definitions of the sport's vocabulary
│   ├── assets/
│   │   ├── style.css           # Design tokens + component classes (dark theme)
│   │   └── replay_player.html  # Broadcast replay player (iframe component)
│   └── pages/              # Streamlit pages
│       ├── 1_Home.py       # Dashboard (+ About & Limitations)
│       ├── 2_HeadToHead.py # Head-to-head comparisons
│       ├── 3_Predictor.py  # Predictor + fantasy points
│       ├── 4_Report.py     # Pre-race briefing / post-race debrief
│       ├── 5_Replay.py     # Race replay player (OpenF1 telemetry)
│       ├── 6_Analysis.py   # Race analysis center
│       └── 7_Competitor.py # Competitor analysis
├── notebooks/              # Jupyter analysis notebooks
├── tests/                  # Unit tests
├── tools/                  # latency_check.py (timings), shot.py / shots_all.py (screenshots)
├── .streamlit/             # Streamlit configuration
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
└── LICENSE                 # MIT License
```

There are no CSV exports: the season frame is built from the OpenF1 API
(`src/openf1_results.py`), and the replay telemetry is cached in
`f1_cache/telemetry.sqlite`.

---

##  Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit, Plotly, Matplotlib |
| **Data Processing** | Pandas, NumPy |
| **F1 Data** | OpenF1 API, FastF1 API, MultiViewer API |
| **Visualization** | Plotly |
| **Prediction** | Transparent heuristic model (no ML framework) |
| **Deployment** | Docker, GitHub Actions |

---

## 📊 Data Sources

This project uses two public F1 data APIs:

**[FastF1](https://github.com/theOehrly/Fast-F1)** — an unofficial F1 data API that provides:
- Session telemetry and timing data
- Driver and car information
- Lap-by-lap detailed timing
- Tyre compound data
- Weather information

**[OpenF1](https://openf1.org/)** — provides:
- Session telemetry for the replay: 4 Hz car data, laps, tyre stints, intervals, pit stops and race control
- Championship results, sprint results and qualifying grids (the app's season frame)
- Lap times, sector times, tyre compounds, stint boundaries and pit stops for the
  analysis pages — used directly, and as a fallback when FastF1 cannot serve a session

OpenF1's archive starts with the **2023 season**, so earlier seasons have no championship
data here and the pages say so instead of showing an empty table. Circuit geometry for the
replay track map comes from the [MultiViewer API](https://api.multiviewer.app/).

### When FastF1 cannot serve a session

FastF1 reads the official timing archive (`livetiming.formula1.com`), which is now
closed to anonymous requests — a session load fails with *"Cannot load lap times for
first lap from Ergast"*. The app does not show an empty page for that:

- a cheap **archive probe** checks the timing index once per process, and skips FastF1
  entirely when it is closed (no 10-second wait per page load);
- `src/openf1_session.py` then **rebuilds the same session shape from OpenF1** — lap
  times, sector times, tyre compound, stint number, tyre age, in/out laps, pit stops
  and the classification — so the Analysis, Report and Replay pages keep working;
- the gap is disclosed on the page, not hidden: OpenF1 publishes **no race-control
  flags**, so a safety-car lap is not marked and the running order is **estimated**
  from the cumulative lap time (official order is used when the feed covers it);
- the OpenF1 fallback carries **no car telemetry**, so the setup fingerprint in the
  Analysis Center says it is unavailable instead of guessing.

**Team colours come from the API too.** OpenF1's driver feed reports the official livery
colour per driver, so the app paints the current field correctly without a hardcoded
palette - a team that joins the grid is the right colour from its first race.

### Licensing

OpenF1 data is published under **CC BY-NC-SA 4.0**. That licence is **not compatible**
with this repository's MIT licence for commercial use: the code is MIT, but any deployment
that displays OpenF1-derived data must be non-commercial, or it must obtain a separate
licence from OpenF1. FastF1 data is provided under its own open terms.

---

##  Testing

```bash
pytest tests/ -v
```

---

##  Docker

```bash
docker build -t f1lab .
docker run -p 8501:8501 f1lab
```

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) for the amazing F1 data API
- [Streamlit](https://streamlit.io/) for the web framework
- Formula 1® and FIA for the sport we love

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/maxvyquincy9393">Maxvy</a>
</p>
