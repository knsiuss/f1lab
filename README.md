<h1 align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg" width="60" alt="F1">
  <br>
  F1 Lab
</h1>

<p align="center">
  Real-time Formula 1 analytics dashboard for the 2025 season.<br>
  Built with Streamlit, FastF1, and Plotly.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.31+-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Data-FastF1-E10600" alt="FastF1">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

##  Features

###  Season Analytics
- **Championship Standings** — Live driver and constructor leaderboards
- **Points Progression** — Visual race-by-race championship evolution
- **Driver Profiles** — Career stats, biography, social links
- **Team Analysis** — Constructor performance comparisons

###  Race Center
- **Race Analysis** — Lap times, position changes, gap analysis
- **Pit Strategy** — Stop timings, undercuts, tyre strategy visualization
- **Qualifying** — Sector times, lap evolution, session comparisons
- **Official Plots** — FIA-style race summary charts

###  Telemetry
- **Speed Traces** — Throttle, brake, gear data from FastF1
- **Driver Comparison** — Side-by-side telemetry overlays
- **Track Visualization** — Circuit maps with corner annotations
- **Tyre Degradation** — Compound performance over stints

###  Race Replay
- **Animated Visualization** — Watch races unfold on track
- **Browser Replay** — Animated position replay with Play/Pause (FastF1)
- **Live Leaderboard** — Real-time position updates
- **Driver Selection** — Click to focus on any driver

###  Predictions
- **Race Forecasting** — ML-based finishing position predictions
- **Strategy Simulation** — What-if scenario analysis
- **Model Evaluation** — Accuracy metrics and validation

###  Live Timing
- **Session Monitor** — Real-time practice, quali, race data
- **Lap Updates** — Live sector and lap times
- **Track Status** — Flags, safety car, red flag alerts

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
│   ├── loader.py           # CSV loading + cleaning
│   ├── analysis.py         # Statistical calculations
│   ├── model.py            # Race strategy simulator
│   ├── prediction.py       # Race predictor + fantasy points
│   ├── backtest.py         # Walk-forward backtest vs grid baseline
│   ├── whatif.py           # Strategy what-if scenarios + undercut
│   ├── pace.py             # Normalised pace, degradation, data-quality gates
│   ├── pitwindow.py        # Pit-window radar per driver
│   ├── briefing.py         # Pre-race briefing / post-race debrief
│   ├── rival.py            # Rival watchlist, head-to-head, circuit insights
│   ├── sector.py           # Sector insights from lap sector times
│   ├── compare.py          # Session-to-session pace comparison
│   ├── fastf1_extended.py  # FastF1 extraction (weather, pits, sectors, telemetry)
│   ├── replay.py           # Animated position replay
│   ├── report.py           # Markdown weekend report
│   ├── styles.py           # Plotly design tokens
│   └── pages/              # Streamlit pages (Dashboard ... Competitor Analysis)
├── data/                   # Season datasets (Formula1_{year}Season_*.csv)
├── notebooks/              # Jupyter analysis notebooks
├── tests/                  # Unit tests
├── .streamlit/             # Streamlit configuration
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
└── LICENSE                 # MIT License
```

---

##  Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit, Plotly, Matplotlib |
| **Data Processing** | Pandas, NumPy |
| **F1 Data** | FastF1 API |
| **Visualization** | Plotly, Plotly Express |
| **Machine Learning** | Scikit-learn |
| **Deployment** | Docker, GitHub Actions |

---

## 📊 Data Sources

This project uses [FastF1](https://github.com/theOehrly/Fast-F1), an unofficial F1 data API that provides:
- Session telemetry and timing data
- Driver and car information
- Lap-by-lap detailed timing
- Tyre compound data
- Weather information

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
