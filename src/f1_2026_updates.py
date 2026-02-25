# -*- coding: utf-8 -*-
"""F1 2026 research, news, live data monitoring, and design-forward Streamlit hub."""

from __future__ import annotations

import json
import inspect
import re
import xml.etree.ElementTree as ET
from html import unescape
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import fastf1
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from src.loader import clean_data
    from src.model import train_model
    from src.evaluate import evaluate_model
    from src.preseason_testing import (
        DEFAULT_PRESEASON_TEAM_SUMMARY_PATH,
        load_preseason_testing_team_summary,
        load_preseason_testing_team_features,
    )
except ImportError:  # pragma: no cover - fallback for direct execution
    from loader import clean_data
    from model import train_model
    from evaluate import evaluate_model
    from preseason_testing import (
        DEFAULT_PRESEASON_TEAM_SUMMARY_PATH,
        load_preseason_testing_team_summary,
        load_preseason_testing_team_features,
    )


OFFICIAL_2026_CALENDAR: List[Dict[str, Any]] = [
    {"round": 1, "event": "Australian Grand Prix", "country": "Australia", "location": "Melbourne", "race_date": "2026-03-08"},
    {"round": 2, "event": "Chinese Grand Prix", "country": "China", "location": "Shanghai", "race_date": "2026-03-15"},
    {"round": 3, "event": "Japanese Grand Prix", "country": "Japan", "location": "Suzuka", "race_date": "2026-03-29"},
    {"round": 4, "event": "Bahrain Grand Prix", "country": "Bahrain", "location": "Sakhir", "race_date": "2026-04-12"},
    {"round": 5, "event": "Saudi Arabian Grand Prix", "country": "Saudi Arabia", "location": "Jeddah", "race_date": "2026-04-19"},
    {"round": 6, "event": "Miami Grand Prix", "country": "USA", "location": "Miami", "race_date": "2026-05-03"},
    {"round": 7, "event": "Canadian Grand Prix", "country": "Canada", "location": "Montreal", "race_date": "2026-05-24"},
    {"round": 8, "event": "Monaco Grand Prix", "country": "Monaco", "location": "Monte Carlo", "race_date": "2026-06-07"},
    {"round": 9, "event": "Spanish Grand Prix", "country": "Spain", "location": "Barcelona-Catalunya", "race_date": "2026-06-14"},
    {"round": 10, "event": "Austrian Grand Prix", "country": "Austria", "location": "Spielberg", "race_date": "2026-06-28"},
    {"round": 11, "event": "British Grand Prix", "country": "Great Britain", "location": "Silverstone", "race_date": "2026-07-05"},
    {"round": 12, "event": "Belgian Grand Prix", "country": "Belgium", "location": "Spa-Francorchamps", "race_date": "2026-07-19"},
    {"round": 13, "event": "Hungarian Grand Prix", "country": "Hungary", "location": "Budapest", "race_date": "2026-07-26"},
    {"round": 14, "event": "Dutch Grand Prix", "country": "Netherlands", "location": "Zandvoort", "race_date": "2026-08-23"},
    {"round": 15, "event": "Italian Grand Prix", "country": "Italy", "location": "Monza", "race_date": "2026-09-06"},
    {"round": 16, "event": "Madrid Grand Prix", "country": "Spain", "location": "Madrid", "race_date": "2026-09-13"},
    {"round": 17, "event": "Azerbaijan Grand Prix", "country": "Azerbaijan", "location": "Baku", "race_date": "2026-09-26"},
    {"round": 18, "event": "Singapore Grand Prix", "country": "Singapore", "location": "Singapore", "race_date": "2026-10-11"},
    {"round": 19, "event": "United States Grand Prix", "country": "USA", "location": "Austin", "race_date": "2026-10-25"},
    {"round": 20, "event": "Mexico City Grand Prix", "country": "Mexico", "location": "Mexico City", "race_date": "2026-11-01"},
    {"round": 21, "event": "Sao Paulo Grand Prix", "country": "Brazil", "location": "Sao Paulo", "race_date": "2026-11-08"},
    {"round": 22, "event": "Las Vegas Grand Prix", "country": "USA", "location": "Las Vegas", "race_date": "2026-11-21"},
    {"round": 23, "event": "Qatar Grand Prix", "country": "Qatar", "location": "Lusail", "race_date": "2026-11-29"},
    {"round": 24, "event": "Abu Dhabi Grand Prix", "country": "UAE", "location": "Yas Marina", "race_date": "2026-12-06"},
]

SPRINT_2026 = {
    "Chinese Grand Prix",
    "Miami Grand Prix",
    "Canadian Grand Prix",
    "British Grand Prix",
    "Dutch Grand Prix",
    "Singapore Grand Prix",
}

PRESEASON_TESTS_2026 = [
    {"label": "Private Test 1", "venue": "Barcelona-Catalunya", "country": "Spain", "start_date": "2026-01-26", "end_date": "2026-01-30", "session_type": "Pre-season"},
    {"label": "Official Test 2", "venue": "Bahrain", "country": "Bahrain", "start_date": "2026-02-11", "end_date": "2026-02-13", "session_type": "Pre-season"},
    {"label": "Official Test 3", "venue": "Bahrain", "country": "Bahrain", "start_date": "2026-02-18", "end_date": "2026-02-20", "session_type": "Pre-season"},
]

OFFICIAL_2026_UPDATES = [
    {
        "date": "2025-03-07",
        "category": "Teams",
        "title": "Cadillac approved to join F1 from 2026",
        "summary": "FIA and Formula 1 confirmed Cadillac met requirements to join the championship from 2026, expanding the grid to 11 teams.",
        "source": "FIA",
        "url": "https://www.fia.com/news/fia-and-formula-1-can-confirm-cadillac-formula-1-team-has-met-their-requirements-join-2026",
    },
    {
        "date": "2025-06-10",
        "category": "Calendar",
        "title": "24-round 2026 calendar announced with Madrid debut",
        "summary": "Formula 1 and FIA unveiled the 2026 calendar and highlighted logistical sequencing improvements plus Madrid's debut.",
        "source": "Formula1.com / FIA",
        "url": "https://www.formula1.com/en/latest/article.fia-and-formula-1-unveil-calendar-for-2026-season-as-madrid-makes-its-debut.4lEs5Z6ow8NLTHF1nEr4oq",
    },
    {
        "date": "2025-09-16",
        "category": "Sprint",
        "title": "2026 Sprint calendar confirmed",
        "summary": "Corporate Formula 1 confirmed six Sprint events for 2026.",
        "source": "Corporate Formula 1",
        "url": "https://corp.formula1.com/formula-1-confirms-calendar-for-six-sprint-events-across-2026-season/",
    },
    {
        "date": "2025-09-30",
        "category": "Calendar",
        "title": "FIA WMSC updates 2026 calendar and pre-season tests",
        "summary": "FIA World Motor Sport Council ratified calendar updates and confirmed pre-season testing windows in Barcelona and Bahrain.",
        "source": "FIA",
        "url": "https://www.fia.com/news/fia-world-motor-sport-council-updates-2026-formula-one-world-championship-calendar",
    },
    {
        "date": "2025-10-12",
        "category": "Governance",
        "title": "WMSC approves 2026 FIA sporting calendars",
        "summary": "FIA published sporting calendar approvals including Formula One and other FIA series.",
        "source": "FIA",
        "url": "https://www.fia.com/news/fia-world-motor-sport-council-approves-2026-fia-sporting-calendars",
    },
    {
        "date": "2025-12-12",
        "category": "Governance",
        "title": "2026 Concorde Governance Agreement signed",
        "summary": "FIA announced the 2026 Concorde Governance Agreement signed by Formula 1, all 11 teams and FIA.",
        "source": "FIA",
        "url": "https://www.fia.com/news/fia-formula-one-and-all-11-teams-sign-2026-concorde-governance-agreement",
    },
    {
        "date": "2025-12-10",
        "category": "Regulations",
        "title": "Latest 2026 regulations issue documents published",
        "summary": "FIA regulations portal lists issue updates for 2026 technical, sporting, financial and power unit regulations.",
        "source": "FIA",
        "url": "https://www.fia.com/regulation/category/110",
    },
    {
        "date": "2026-02-25",
        "category": "Platform",
        "title": "Formula1.com 2026 season hub live",
        "summary": "Formula1.com 2026 season pages show calendar, teams (including Audi and Cadillac), and regulations hubs.",
        "source": "Formula1.com",
        "url": "https://www.formula1.com/en/racing/2026",
    },
]

NEWS_FALLBACK_2026 = [
    {
        "published_at": "2026-02-24",
        "headline": "Formula 1 confirms 2026 pre-season testing dates and issues calendar update",
        "summary": "Formula1.com update covers 2026 testing dates and a calendar adjustment.",
        "url": "https://www.formula1.com/en/latest/article.formula-1-confirms-2026-pre-season-testing-dates-and-issues-calendar.7sM4qoM8h6EYt4W4aiRcY8",
        "source": "Formula1.com",
        "topic": "Calendar",
        "official": True,
        "feed": "fallback",
    },
    {
        "published_at": "2026-02-18",
        "headline": "Formula 1 confirms official Grand Prix start times for 2026 season",
        "summary": "Formula1.com published confirmed race start times for the full 2026 season schedule.",
        "url": "https://www.formula1.com/en/latest/article.formula-1-confirms-official-grand-prix-start-times-for-2026-season.1jCdE3D8fAwI7Vk4ofeJPf",
        "source": "Formula1.com",
        "topic": "Calendar",
        "official": True,
        "feed": "fallback",
    },
    {
        "published_at": "2026-02-17",
        "headline": "All the key dates for your diary ahead of F1 pre-season in 2026",
        "summary": "Formula1.com pre-season guide summarises launch/testing milestones before the 2026 campaign begins.",
        "url": "https://www.formula1.com/en/latest/article.all-the-key-dates-for-your-diary-ahead-of-f1-pre-season-in-2026.3Iu4llQdb8NLmH8g66FUqr",
        "source": "Formula1.com",
        "topic": "Pre-season",
        "official": True,
        "feed": "fallback",
    },
    {
        "published_at": "2026-02-12",
        "headline": "F1 reveals all 10 car launch dates and key details for 2026 season",
        "summary": "Formula1.com published the launch-date roundup for teams before the 2026 season start.",
        "url": "https://www.formula1.com/en/latest/article.f1-reveals-all-10-car-launch-dates-and-key-details-for-2026-season.5Sp59L4z8l9Mv0zDguUEtM",
        "source": "Formula1.com",
        "topic": "Teams",
        "official": True,
        "feed": "fallback",
    },
    {
        "published_at": "2025-12-12",
        "headline": "FIA, Formula One and all 11 teams sign 2026 Concorde Governance Agreement",
        "summary": "FIA confirms governance agreement completion ahead of the new regulatory era.",
        "url": "https://www.fia.com/news/fia-formula-one-and-all-11-teams-sign-2026-concorde-governance-agreement",
        "source": "FIA",
        "topic": "Governance",
        "official": True,
        "feed": "fallback",
    },
]

FEATURE_READINESS_2026 = [
    {"category": "Season Analytics", "feature": "Championship Standings", "status": "Partial", "priority": "High", "data_dependency": "Jolpica/FastF1 + 2026 race results", "notes": "Can show standings via connected feeds now; advanced stats need race-by-race persistence."},
    {"category": "Season Analytics", "feature": "Points Progression", "status": "Season Start", "priority": "High", "data_dependency": "2026 round results", "notes": "Full progression chart matures over the season."},
    {"category": "Season Analytics", "feature": "Driver Profiles", "status": "Partial", "priority": "Medium", "data_dependency": "Roster/content updates", "notes": "Needs 2026 roster/bio refresh."},
    {"category": "Season Analytics", "feature": "Team Analysis", "status": "Partial", "priority": "High", "data_dependency": "2026 constructors metadata", "notes": "Needs team metadata + color updates for new grid era."},
    {"category": "Race Center", "feature": "Race Analysis", "status": "Ready", "priority": "High", "data_dependency": "FastF1 sessions", "notes": "FastF1-backed once 2026 sessions are available."},
    {"category": "Race Center", "feature": "Pit Strategy", "status": "Ready", "priority": "High", "data_dependency": "FastF1 laps/stints", "notes": "Validate tyre/stint schema in new era."},
    {"category": "Race Center", "feature": "Qualifying", "status": "Ready", "priority": "High", "data_dependency": "FastF1 qualifying sessions", "notes": "Test event naming/session labels."},
    {"category": "Race Center", "feature": "Official Plots", "status": "Ready", "priority": "Medium", "data_dependency": "FastF1/Matplotlib", "notes": "Mostly reusable; check color mappings."},
    {"category": "Race Center", "feature": "Export Data", "status": "Ready", "priority": "Medium", "data_dependency": "FastF1 session objects", "notes": "Needs year selector support in UI."},
    {"category": "Telemetry", "feature": "Speed Traces", "status": "Ready", "priority": "High", "data_dependency": "FastF1 telemetry", "notes": "Track/session availability dependent."},
    {"category": "Telemetry", "feature": "Driver Comparison", "status": "Ready", "priority": "High", "data_dependency": "FastF1 telemetry", "notes": "Validate new driver codes/lineups."},
    {"category": "Telemetry", "feature": "Track Visualization", "status": "Partial", "priority": "Medium", "data_dependency": "Track map + Madrid support", "notes": "Madrid geometry likely needs tuning once available."},
    {"category": "Telemetry", "feature": "Tyre Degradation", "status": "Partial", "priority": "High", "data_dependency": "FastF1 stint/compound data", "notes": "Thresholds need recalibration under 2026 rules."},
    {"category": "Race Replay", "feature": "Animated Visualization", "status": "Ready", "priority": "Medium", "data_dependency": "FastF1 position/telemetry", "notes": "Expected to work after 2026 selectors are added."},
    {"category": "Race Replay", "feature": "Desktop Player", "status": "Ready", "priority": "Low", "data_dependency": "Replay data generator", "notes": "Needs 2026 smoke test."},
    {"category": "Race Replay", "feature": "Live Leaderboard", "status": "Partial", "priority": "Medium", "data_dependency": "FastF1 results/live timing", "notes": "Handle partial/live schemas robustly."},
    {"category": "Race Replay", "feature": "Driver Selection", "status": "Ready", "priority": "Low", "data_dependency": "Driver code map", "notes": "Roster refresh only."},
    {"category": "Predictions", "feature": "Race Forecasting", "status": "Rebuild", "priority": "High", "data_dependency": "New-era training data", "notes": "Regulation reset likely breaks old model assumptions."},
    {"category": "Predictions", "feature": "Strategy Simulation", "status": "Rebuild", "priority": "High", "data_dependency": "2026 tyre/pace priors", "notes": "Needs new calibration and assumptions."},
    {"category": "Predictions", "feature": "Model Evaluation", "status": "Season Start", "priority": "High", "data_dependency": "Observed 2026 race outcomes", "notes": "Start after retraining and first rounds."},
    {"category": "Live Timing", "feature": "Session Monitor", "status": "Ready", "priority": "High", "data_dependency": "FastF1 schedule + live availability", "notes": "Needs year/session selector plumbing."},
    {"category": "Live Timing", "feature": "Lap Updates", "status": "Ready", "priority": "High", "data_dependency": "FastF1 live timing", "notes": "Smoke test on first 2026 weekend."},
    {"category": "Live Timing", "feature": "Track Status", "status": "Ready", "priority": "High", "data_dependency": "FastF1 track status/messages", "notes": "Ready after session plumbing."},
    {"category": "Live Timing", "feature": "Race Control Alerts", "status": "Ready", "priority": "Medium", "data_dependency": "FastF1 messages", "notes": "Existing project modules already support this."},
    {"category": "2026 Hub", "feature": "Official Research", "status": "Ready", "priority": "High", "data_dependency": "FIA / Formula1.com", "notes": "Curated verified timeline + references."},
    {"category": "2026 Hub", "feature": "News Feed", "status": "Partial", "priority": "Medium", "data_dependency": "RSS / search feeds", "notes": "Live RSS + fallback; feed availability varies."},
    {"category": "2026 Hub", "feature": "Data Services Monitor", "status": "Ready", "priority": "High", "data_dependency": "FastF1/Jolpica/OpenF1", "notes": "Latency/rows/status and previews implemented."},
]

DATA_ACCESS_CATALOG = [
    {"source": "Local CSV (f1lab)", "dataset": "2025 race/quali/sprint results", "type": "Historical structured", "telemetry": "No", "interactive_use": "Season stats / standings / model features", "status": "Ready"},
    {"source": "Local CSV (f1lab)", "dataset": "2026 pre-season testing team summary (Bahrain + Spain shakedown)", "type": "Curated structured", "telemetry": "No", "interactive_use": "ML pipeline priors / preseason visuals / diagnostics", "status": "Ready"},
    {"source": "FastF1", "dataset": "Event schedule", "type": "Live Feed", "telemetry": "No", "interactive_use": "Calendar/session selection", "status": "Ready"},
    {"source": "FastF1", "dataset": "Session results/laps/weather", "type": "Live Feed", "telemetry": "Partial", "interactive_use": "Race center / live / export", "status": "Ready"},
    {"source": "FastF1", "dataset": "Telemetry & car/position data", "type": "Live Feed", "telemetry": "Yes", "interactive_use": "Telemetry charts / replay / strategy views", "status": "Ready"},
    {"source": "Jolpica (Ergast-compatible)", "dataset": "Schedule", "type": "Live Feed", "telemetry": "No", "interactive_use": "Lightweight calendar fallback", "status": "Ready"},
    {"source": "Jolpica (Ergast-compatible)", "dataset": "Driver/constructor standings", "type": "Live Feed", "telemetry": "No", "interactive_use": "Standings panels", "status": "Partial"},
    {"source": "OpenF1", "dataset": "Meetings/Sessions", "type": "Live Feed", "telemetry": "No", "interactive_use": "Session discovery and linking", "status": "Ready"},
    {"source": "OpenF1", "dataset": "Drivers/Laps/Weather/Stints", "type": "Live Feed", "telemetry": "Partial", "interactive_use": "Interactive data lab previews", "status": "Partial"},
    {"source": "OpenF1", "dataset": "Car telemetry (car_data)", "type": "Live Feed", "telemetry": "Yes", "interactive_use": "Interactive telemetry timeline in Data Lab", "status": "Partial"},
    {"source": "Google News RSS", "dataset": "News headlines", "type": "Feed", "telemetry": "No", "interactive_use": "2026 news monitoring", "status": "Partial"},
    {"source": "ML Pipeline (scikit-learn)", "dataset": "Race prediction train/eval artifacts", "type": "Local compute", "telemetry": "No", "interactive_use": "Predictions / model diagnostics / feature importances", "status": "Ready"},
]

F1_LIBRARY_COVERAGE = [
    {"library": "FastF1", "category": "Schedule", "capability": "get_event_schedule(year)", "data_kind": "Calendar/events", "interactive": "Yes", "coverage_status": "Integrated", "perf_note": "Cached in app"},
    {"library": "FastF1", "category": "Sessions", "capability": "get_session + load()", "data_kind": "Results/laps/weather", "interactive": "Yes", "coverage_status": "Integrated", "perf_note": "Lazy per tab"},
    {"library": "FastF1", "category": "Telemetry", "capability": "car/position/telemetry via session", "data_kind": "Speed/throttle/brake/gear/position", "interactive": "Yes", "coverage_status": "Integrated", "perf_note": "Heavy; on-demand"},
    {"library": "FastF1", "category": "Live Timing", "capability": "Live session partial loads", "data_kind": "Track status/messages/results", "interactive": "Yes", "coverage_status": "Integrated", "perf_note": "Network-dependent"},
    {"library": "Jolpica (Ergast-compatible)", "category": "Schedule", "capability": "/ergast/f1/{year}[races].json", "data_kind": "Calendar", "interactive": "Yes", "coverage_status": "Integrated", "perf_note": "Lightweight fallback"},
    {"library": "Jolpica (Ergast-compatible)", "category": "Standings", "capability": "driverStandings/constructorStandings", "data_kind": "Points standings", "interactive": "Yes", "coverage_status": "Integrated", "perf_note": "Season start dependent"},
    {"library": "OpenF1", "category": "Meetings", "capability": "/v1/meetings", "data_kind": "Meeting metadata", "interactive": "Yes", "coverage_status": "Integrated", "perf_note": "Lightweight"},
    {"library": "OpenF1", "category": "Sessions", "capability": "/v1/sessions", "data_kind": "Session metadata + keys", "interactive": "Yes", "coverage_status": "Integrated", "perf_note": "Key for explorer"},
    {"library": "OpenF1", "category": "Drivers", "capability": "/v1/drivers", "data_kind": "Roster/session drivers", "interactive": "Yes", "coverage_status": "Integrated", "perf_note": "Per session"},
    {"library": "OpenF1", "category": "Laps", "capability": "/v1/laps", "data_kind": "Lap-level times", "interactive": "Yes", "coverage_status": "Integrated", "perf_note": "Sampled when needed"},
    {"library": "OpenF1", "category": "Weather", "capability": "/v1/weather", "data_kind": "Weather timeline", "interactive": "Yes", "coverage_status": "Integrated", "perf_note": "Per session"},
    {"library": "OpenF1", "category": "Telemetry", "capability": "/v1/car_data", "data_kind": "Car telemetry", "interactive": "Yes", "coverage_status": "Integrated", "perf_note": "On-demand + capped rows"},
    {"library": "OpenF1", "category": "Position", "capability": "/v1/position", "data_kind": "Track position trace", "interactive": "Yes", "coverage_status": "Integrated", "perf_note": "On-demand + capped rows"},
    {"library": "Streamlit", "category": "UI Runtime", "capability": "Tabs/sidebar/state/cache", "data_kind": "Interaction", "interactive": "Yes", "coverage_status": "Integrated", "perf_note": "Fast reruns with cache"},
    {"library": "Plotly", "category": "Visualization", "capability": "Interactive charts", "data_kind": "Charts/telemetry/schedule", "interactive": "Yes", "coverage_status": "Integrated", "perf_note": "Client-side rendering"},
    {"library": "scikit-learn", "category": "ML Pipeline", "capability": "Feature prep + HistGradientBoostingRegressor + evaluation", "data_kind": "Predictions/metrics", "interactive": "Yes", "coverage_status": "Integrated", "perf_note": "Train on-demand and cache diagnostics"},
]

REG_CHANGE = pd.DataFrame(
    [
        {"metric": "Car width (cm)", "2025": 200, "2026": 190},
        {"metric": "Car length (cm)", "2025": 360, "2026": 340},
        {"metric": "Floor width (cm)", "2025": 150, "2026": 140},
        {"metric": "Max wheelbase (mm)", "2025": 3600, "2026": 3400},
        {"metric": "Minimum weight (kg)", "2025": 798, "2026": 768},
    ]
)

PU_SPLIT = pd.DataFrame(
    [
        {"era": "2025 PU", "source": "Electric", "share": 20},
        {"era": "2025 PU", "source": "ICE", "share": 80},
        {"era": "2026 PU", "source": "Electric", "share": 50},
        {"era": "2026 PU", "source": "ICE", "share": 50},
    ]
)

REGION_MAP = {
    "Australia": "APAC",
    "China": "APAC",
    "Japan": "APAC",
    "Singapore": "APAC",
    "Bahrain": "Middle East",
    "Saudi Arabia": "Middle East",
    "Qatar": "Middle East",
    "UAE": "Middle East",
    "United Arab Emirates": "Middle East",
    "Monaco": "Europe",
    "Spain": "Europe",
    "Austria": "Europe",
    "Great Britain": "Europe",
    "Belgium": "Europe",
    "Hungary": "Europe",
    "Netherlands": "Europe",
    "Italy": "Europe",
    "Azerbaijan": "Eurasia",
    "USA": "Americas",
    "United States": "Americas",
    "Canada": "Americas",
    "Mexico": "Americas",
    "Brazil": "Americas",
}

PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False}
UTC_OFFSET_OPTIONS = [
    "-12:00", "-11:00", "-10:00", "-09:00", "-08:00", "-07:00", "-06:00", "-05:00",
    "-04:00", "-03:00", "-02:00", "-01:00", "+00:00", "+01:00", "+02:00", "+03:00",
    "+04:00", "+05:00", "+05:30", "+06:00", "+07:00", "+08:00", "+09:00", "+09:30",
    "+10:00", "+11:00", "+12:00", "+13:00", "+14:00"
]


def _plotly_chart(fig: go.Figure, **kwargs: Any) -> Any:
    """Wrap Streamlit Plotly rendering with a stable per-call-site key."""
    if kwargs.get("key") is None:
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        caller_name = caller.f_code.co_name if caller is not None else "unknown"
        caller_line = caller.f_lineno if caller is not None else 0
        kwargs["key"] = f"f1_2026_plotly_{caller_name}_{caller_line}"
        del frame
        del caller
    return st.plotly_chart(fig, **kwargs)


def _http_text(url: str, timeout: int = 8) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    t0 = perf_counter()
    req = Request(
        url,
        headers={
            "User-Agent": "f1lab-2026-hub",
            "Accept": "application/json,text/xml,application/xml,text/plain,*/*",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return raw.decode("utf-8", errors="replace"), int((perf_counter() - t0) * 1000), None
    except HTTPError as exc:
        return None, None, f"HTTP {exc.code}"
    except URLError as exc:
        return None, None, str(exc.reason)
    except Exception as exc:  # pragma: no cover
        return None, None, str(exc)


def _http_json(url: str, timeout: int = 8) -> Tuple[Optional[Any], Optional[int], Optional[str]]:
    text, latency, err = _http_text(url, timeout=timeout)
    if err or text is None:
        return None, latency, err
    try:
        return json.loads(text), latency, None
    except Exception as exc:  # pragma: no cover
        return None, latency, str(exc)


def _clean_html_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    txt = unescape(str(text))
    txt = re.sub(r"<[^>]+>", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def _safe_datetime(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce", utc=True)


def _get_user_utc_offset() -> str:
    val = str(st.session_state.get("user_utc_offset", "+00:00"))
    return val if val in UTC_OFFSET_OPTIONS else "+00:00"


def _apply_user_utc_offset(ts: Any) -> pd.Timestamp:
    t = pd.to_datetime(ts, errors="coerce", utc=True)
    if pd.isna(t):
        return t
    offset = _get_user_utc_offset()
    sign = -1 if offset.startswith("-") else 1
    hh_mm = offset[1:] if offset and offset[0] in "+-" else offset
    try:
        hh, mm = hh_mm.split(":")
        delta = pd.Timedelta(hours=int(hh), minutes=int(mm)) * sign
        return t + delta
    except Exception:
        return t


def _fmt_user_time(ts: Any, fmt: str = "%Y-%m-%d %H:%M") -> str:
    t = _apply_user_utc_offset(ts)
    if pd.isna(t):
        return "N/A"
    return f"{t.strftime(fmt)} (UTC{_get_user_utc_offset()})"


def _classify_news_topic(headline: str, summary: str = "") -> str:
    blob = f"{headline} {summary}".lower()
    rules = [
        ("calendar", "Calendar"),
        ("sprint", "Sprint"),
        ("testing", "Pre-season"),
        ("pre-season", "Pre-season"),
        ("launch", "Teams"),
        ("concorde", "Governance"),
        ("regulation", "Regulations"),
        ("rule", "Regulations"),
        ("cadillac", "Teams"),
        ("audi", "Teams"),
        ("start time", "Calendar"),
    ]
    for keyword, label in rules:
        if keyword in blob:
            return label
    return "General"


def _decorate_calendar(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["round", "event", "country", "location", "race_date", "month", "month_num", "region", "source", "is_sprint"]
        )
    out = df.copy()
    out["round"] = pd.to_numeric(out.get("round"), errors="coerce")
    out["race_date"] = pd.to_datetime(out.get("race_date"), errors="coerce", utc=True)
    out = out.dropna(subset=["race_date"]).copy()
    out["country"] = out.get("country", "").fillna("").astype(str)
    out["location"] = out.get("location", "").fillna("").astype(str)
    out["event"] = out.get("event", "").fillna("").astype(str)
    out["month"] = out["race_date"].dt.strftime("%b")
    out["month_num"] = out["race_date"].dt.month
    out["region"] = out["country"].map(REGION_MAP).fillna("Unknown")
    out["source"] = source
    out["is_sprint"] = out["event"].isin(SPRINT_2026)
    return out.sort_values(["race_date", "round"], na_position="last").reset_index(drop=True)


def _calendar_fallback() -> pd.DataFrame:
    return _decorate_calendar(pd.DataFrame(OFFICIAL_2026_CALENDAR), "Official fallback (F1/FIA research)")


def _fetch_fastf1_schedule() -> Dict[str, Any]:
    t0 = perf_counter()
    try:
        schedule = fastf1.get_event_schedule(2026)
        df = schedule.copy()
        if "RoundNumber" in df.columns:
            df = df[pd.to_numeric(df["RoundNumber"], errors="coerce").fillna(0) > 0]
        race_date_col = "EventDate" if "EventDate" in df.columns else ("Session5Date" if "Session5Date" in df.columns else None)
        if race_date_col is None:
            raise ValueError("FastF1 schedule missing race date column")
        normalized = pd.DataFrame(
            {
                "round": df.get("RoundNumber"),
                "event": df.get("EventName", df.get("OfficialEventName")),
                "country": df.get("Country"),
                "location": df.get("Location"),
                "race_date": df.get(race_date_col),
            }
        )
        parsed = _decorate_calendar(normalized, "FastF1")
        return {
            "api": "FastF1 schedule",
            "ok": not parsed.empty,
            "rows": len(parsed),
            "latency_ms": int((perf_counter() - t0) * 1000),
            "error": None if not parsed.empty else "No 2026 rows returned",
            "url": "fastf1.get_event_schedule(2026)",
            "data": parsed,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "api": "FastF1 schedule",
            "ok": False,
            "rows": 0,
            "latency_ms": int((perf_counter() - t0) * 1000),
            "error": str(exc),
            "url": "fastf1.get_event_schedule(2026)",
            "data": pd.DataFrame(),
        }


def _fetch_jolpica_schedule() -> Dict[str, Any]:
    candidates = (
        "https://api.jolpi.ca/ergast/f1/2026.json",
        "https://api.jolpi.ca/ergast/f1/2026/races.json",
    )
    last: Dict[str, Any] = {
        "api": "Jolpica schedule",
        "ok": False,
        "rows": 0,
        "latency_ms": None,
        "error": "Unknown error",
        "url": candidates[0],
        "data": pd.DataFrame(),
    }
    for url in candidates:
        payload, latency, err = _http_json(url)
        if err:
            last = {**last, "latency_ms": latency, "error": err, "url": url}
            continue
        races = (((payload or {}).get("MRData") or {}).get("RaceTable") or {}).get("Races", [])
        rows = []
        for race in races if isinstance(races, list) else []:
            circuit = (race or {}).get("Circuit") or {}
            location = circuit.get("Location") or {}
            dt = pd.to_datetime(f"{race.get('date', '')}T{race.get('time', '')}", errors="coerce", utc=True)
            if pd.isna(dt):
                dt = pd.to_datetime(race.get("date"), errors="coerce", utc=True)
            rows.append(
                {
                    "round": race.get("round"),
                    "event": race.get("raceName"),
                    "country": location.get("country"),
                    "location": location.get("locality"),
                    "race_date": dt,
                }
            )
        parsed = _decorate_calendar(pd.DataFrame(rows), "Jolpica")
        if not parsed.empty:
            return {
                "api": "Jolpica schedule",
                "ok": True,
                "rows": len(parsed),
                "latency_ms": latency,
                "error": None,
                "url": url,
                "data": parsed,
            }
        last = {**last, "latency_ms": latency, "error": "No 2026 schedule rows", "url": url}
    return last


def _parse_jolpica_driver_standings(payload: Any) -> pd.DataFrame:
    standings_lists = (((payload or {}).get("MRData") or {}).get("StandingsTable") or {}).get("StandingsLists", [])
    if not isinstance(standings_lists, list) or not standings_lists:
        return pd.DataFrame()
    rows_raw = (standings_lists[0] or {}).get("DriverStandings", [])
    rows: List[Dict[str, Any]] = []
    for item in rows_raw if isinstance(rows_raw, list) else []:
        drv = item.get("Driver") or {}
        ctors = item.get("Constructors") or []
        team = (ctors[0] or {}).get("name", "") if isinstance(ctors, list) and ctors else ""
        name = " ".join([str(drv.get("givenName", "")).strip(), str(drv.get("familyName", "")).strip()]).strip()
        rows.append(
            {
                "rank": pd.to_numeric(item.get("position"), errors="coerce"),
                "driver": name or drv.get("driverId", ""),
                "code": drv.get("code", ""),
                "team": team,
                "points": pd.to_numeric(item.get("points"), errors="coerce"),
                "wins": pd.to_numeric(item.get("wins"), errors="coerce"),
            }
        )
    return pd.DataFrame(rows).sort_values("rank", na_position="last").reset_index(drop=True) if rows else pd.DataFrame()


def _parse_jolpica_constructor_standings(payload: Any) -> pd.DataFrame:
    standings_lists = (((payload or {}).get("MRData") or {}).get("StandingsTable") or {}).get("StandingsLists", [])
    if not isinstance(standings_lists, list) or not standings_lists:
        return pd.DataFrame()
    rows_raw = (standings_lists[0] or {}).get("ConstructorStandings", [])
    rows: List[Dict[str, Any]] = []
    for item in rows_raw if isinstance(rows_raw, list) else []:
        ctor = item.get("Constructor") or {}
        rows.append(
            {
                "rank": pd.to_numeric(item.get("position"), errors="coerce"),
                "team": ctor.get("name", "") or ctor.get("constructorId", ""),
                "points": pd.to_numeric(item.get("points"), errors="coerce"),
                "wins": pd.to_numeric(item.get("wins"), errors="coerce"),
            }
        )
    return pd.DataFrame(rows).sort_values("rank", na_position="last").reset_index(drop=True) if rows else pd.DataFrame()


def _fetch_jolpica_standings(kind: str) -> Dict[str, Any]:
    if kind == "driver":
        urls = (
            "https://api.jolpi.ca/ergast/f1/2026/driverStandings.json",
            "https://api.jolpi.ca/ergast/f1/2026/driverstandings.json",
        )
        parser = _parse_jolpica_driver_standings
        api_name = "Jolpica driver standings"
    else:
        urls = (
            "https://api.jolpi.ca/ergast/f1/2026/constructorStandings.json",
            "https://api.jolpi.ca/ergast/f1/2026/constructorstandings.json",
        )
        parser = _parse_jolpica_constructor_standings
        api_name = "Jolpica constructor standings"

    last = {"api": api_name, "ok": False, "rows": 0, "latency_ms": None, "error": "No rows yet", "url": urls[0], "data": pd.DataFrame()}
    for url in urls:
        payload, latency, err = _http_json(url)
        if err:
            last = {**last, "latency_ms": latency, "error": err, "url": url}
            continue
        parsed = parser(payload)
        if not parsed.empty:
            return {"api": api_name, "ok": True, "rows": len(parsed), "latency_ms": latency, "error": None, "url": url, "data": parsed}
        last = {**last, "latency_ms": latency, "error": "No 2026 standings rows (season may not have started)", "url": url}
    return last


def _fetch_openf1_sessions() -> Dict[str, Any]:
    url = "https://api.openf1.org/v1/sessions?year=2026"
    payload, latency, err = _http_json(url)
    if err:
        return {"api": "OpenF1 sessions", "ok": False, "rows": 0, "latency_ms": latency, "error": err, "url": url, "data": pd.DataFrame()}
    rows = []
    for item in payload if isinstance(payload, list) else []:
        start = _safe_datetime(item.get("date_start"))
        if pd.notna(start) and start.year == 2026:
            rows.append(
                {
                    "session_key": item.get("session_key"),
                    "meeting_key": item.get("meeting_key"),
                    "meeting_name": item.get("meeting_name", ""),
                    "session_name": item.get("session_name", ""),
                    "session_type": item.get("session_type", ""),
                    "country_name": item.get("country_name", ""),
                    "location": item.get("location", ""),
                    "date_start": start,
                }
            )
    df = pd.DataFrame(rows).sort_values("date_start") if rows else pd.DataFrame()
    return {"api": "OpenF1 sessions", "ok": not df.empty, "rows": len(df), "latency_ms": latency, "error": None if not df.empty else "No 2026 session rows", "url": url, "data": df}


def _fetch_openf1_meetings() -> Dict[str, Any]:
    url = "https://api.openf1.org/v1/meetings?year=2026"
    payload, latency, err = _http_json(url)
    if err:
        return {"api": "OpenF1 meetings", "ok": False, "rows": 0, "latency_ms": latency, "error": err, "url": url, "data": pd.DataFrame()}
    rows = []
    for item in payload if isinstance(payload, list) else []:
        start = _safe_datetime(item.get("date_start"))
        if pd.notna(start) and start.year == 2026:
            rows.append(
                {
                    "meeting_key": item.get("meeting_key"),
                    "meeting_name": item.get("meeting_name", ""),
                    "country_name": item.get("country_name", ""),
                    "location": item.get("location", ""),
                    "meeting_official_name": item.get("meeting_official_name", ""),
                    "date_start": start,
                }
            )
    df = pd.DataFrame(rows).sort_values("date_start") if rows else pd.DataFrame()
    return {"api": "OpenF1 meetings", "ok": not df.empty, "rows": len(df), "latency_ms": latency, "error": None if not df.empty else "No 2026 meeting rows", "url": url, "data": df}


@st.cache_data(ttl=1800, show_spinner=False)
def load_2026_api_snapshot(refresh_nonce: int = 0) -> Dict[str, Any]:
    _ = refresh_nonce
    fastf1_res = _fetch_fastf1_schedule()
    jolpica_schedule = _fetch_jolpica_schedule()
    jolpica_driver = _fetch_jolpica_standings("driver")
    jolpica_constructor = _fetch_jolpica_standings("constructor")
    openf1_sessions = _fetch_openf1_sessions()
    openf1_meetings = _fetch_openf1_meetings()
    items = [fastf1_res, jolpica_schedule, jolpica_driver, jolpica_constructor, openf1_sessions, openf1_meetings]
    status = pd.DataFrame([{k: v for k, v in item.items() if k != "data"} for item in items])
    return {
        "status": status,
        "fastf1_schedule": fastf1_res,
        "jolpica_schedule": jolpica_schedule,
        "jolpica_driver": jolpica_driver,
        "jolpica_constructor": jolpica_constructor,
        "openf1_sessions": openf1_sessions,
        "openf1_meetings": openf1_meetings,
    }


def _pick_calendar(snapshot: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    for key in ("fastf1_schedule", "jolpica_schedule"):
        df = snapshot.get(key, {}).get("data")
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df, snapshot[key]["api"]
    return _calendar_fallback(), "Official fallback (F1/FIA research)"


def _parse_rss_items(xml_text: str, source_name: str, source_url: str) -> pd.DataFrame:
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return pd.DataFrame()

    rows = []
    for item in root.findall(".//item"):
        title = _clean_html_text(item.findtext("title"))
        link = (item.findtext("link") or "").strip()
        desc = _clean_html_text(item.findtext("description"))
        pub_raw = (item.findtext("pubDate") or item.findtext("{http://purl.org/dc/elements/1.1/}date") or "").strip()
        source_tag = item.findtext("source")
        pub_dt = _safe_datetime(pub_raw)
        if source_name == "Google News" and title and " - " in title:
            parts = title.rsplit(" - ", 1)
            if len(parts) == 2 and len(parts[1]) < 64:
                title = parts[0].strip()
                if not source_tag:
                    source_tag = parts[1].strip()
        rows.append(
            {
                "published_at": pub_dt,
                "headline": title,
                "summary": desc,
                "url": link,
                "source": _clean_html_text(source_tag) or source_name,
                "topic": _classify_news_topic(title, desc),
                "official": any(x in (source_tag or source_name) for x in ["Formula1", "FIA"]),
                "feed": source_name,
                "feed_url": source_url,
            }
        )
    return pd.DataFrame(rows)


def _fetch_google_news(query: str, limit: int = 20) -> Dict[str, Any]:
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    xml_text, latency, err = _http_text(url, timeout=10)
    if err or xml_text is None:
        return {"api": "Google News RSS", "ok": False, "rows": 0, "latency_ms": latency, "error": err or "No response", "url": url, "data": pd.DataFrame()}
    df = _parse_rss_items(xml_text, "Google News", url)
    if not df.empty:
        mask = (
            df["headline"].str.contains(r"\b2026\b|formula 1|f1|cadillac|concorde|pre-season|testing|grand prix|calendar", case=False, regex=True, na=False)
            | df["summary"].str.contains(r"\b2026\b|formula 1|f1|cadillac|concorde|pre-season|testing|grand prix|calendar", case=False, regex=True, na=False)
        )
        df = df[mask].copy()
        df = df.sort_values("published_at", ascending=False, na_position="last").head(limit).reset_index(drop=True)
    return {"api": "Google News RSS", "ok": not df.empty, "rows": len(df), "latency_ms": latency, "error": None if not df.empty else "No matching news rows", "url": url, "data": df}


@st.cache_data(ttl=900, show_spinner=False)
def load_2026_news_snapshot(refresh_nonce: int = 0) -> Dict[str, Any]:
    _ = refresh_nonce
    queries = ["Formula 1 2026 site:formula1.com", "FIA Formula One 2026 site:fia.com"]
    results = [_fetch_google_news(q, limit=20) for q in queries]
    live_parts = [r["data"] for r in results if isinstance(r.get("data"), pd.DataFrame) and not r["data"].empty]
    live_df = pd.concat(live_parts, ignore_index=True) if live_parts else pd.DataFrame()
    if not live_df.empty:
        live_df = live_df.drop_duplicates(subset=["headline", "url"]).sort_values("published_at", ascending=False, na_position="last").reset_index(drop=True)

    fallback_df = pd.DataFrame(NEWS_FALLBACK_2026).copy()
    if not fallback_df.empty:
        fallback_df["published_at"] = pd.to_datetime(fallback_df["published_at"], errors="coerce", utc=True)

    merged = pd.concat([live_df, fallback_df], ignore_index=True) if not live_df.empty else fallback_df
    if not merged.empty:
        merged["published_at"] = pd.to_datetime(merged["published_at"], errors="coerce", utc=True)
        merged = merged.drop_duplicates(subset=["headline", "url"]).sort_values("published_at", ascending=False, na_position="last").reset_index(drop=True)

    status = pd.DataFrame([{k: v for k, v in r.items() if k != "data"} for r in results])
    return {"status": status, "news": merged}


def run_2026_diagnostics(snapshot: Dict[str, Any], news_snapshot: Dict[str, Any]) -> pd.DataFrame:
    """Professional-style diagnostics for data availability, readiness, and performance."""
    rows: List[Dict[str, Any]] = []

    def add(component: str, check: str, status: str, severity: str, value: Any, threshold: str, details: str) -> None:
        rows.append(
            {
                "component": component,
                "check": check,
                "status": status,
                "severity": severity,
                "value": value,
                "threshold": threshold,
                "details": details,
            }
        )

    # Data service status checks
    status_df = snapshot.get("status", pd.DataFrame())
    if isinstance(status_df, pd.DataFrame) and not status_df.empty:
        api_ok_count = int(status_df["ok"].fillna(False).sum())
        api_total = int(len(status_df))
        add("Data Services", "Availability count", "PASS" if api_ok_count >= max(1, api_total // 2) else "WARN", "High", f"{api_ok_count}/{api_total}", ">= 50% available", "Source availability may vary during pre-season or network interruptions.")
        latency = pd.to_numeric(status_df.get("latency_ms"), errors="coerce").dropna()
        if not latency.empty:
            med = float(latency.median())
            add("Data Services", "Median latency", "PASS" if med < 2500 else "WARN", "Medium", f"{med:.0f} ms", "< 2500 ms", "Measured from cached snapshot fetch run.")
    else:
        add("Data Services", "Status frame exists", "FAIL", "High", "0 rows", "> 0 rows", "No data-service status rows produced.")

    # Calendar integrity
    cal_df, cal_src = _pick_calendar(snapshot)
    cal_rows = int(len(cal_df)) if isinstance(cal_df, pd.DataFrame) else 0
    add("Calendar", "Round count", "PASS" if cal_rows == 24 else ("WARN" if cal_rows > 0 else "FAIL"), "High", cal_rows, "24 rounds", f"Source = {cal_src}")
    if isinstance(cal_df, pd.DataFrame) and not cal_df.empty:
        sprint_count = int(cal_df["is_sprint"].fillna(False).sum()) if "is_sprint" in cal_df.columns else 0
        add("Calendar", "Sprint count", "PASS" if sprint_count == 6 else "WARN", "Medium", sprint_count, "6 sprint weekends", "Uses confirmed 2026 sprint list tagging.")
        if "race_date" in cal_df.columns:
            monotonic = pd.to_datetime(cal_df["race_date"], errors="coerce", utc=True).is_monotonic_increasing
            add("Calendar", "Race dates sorted", "PASS" if monotonic else "WARN", "Low", str(monotonic), "True", "Sorted calendar improves UI consistency and performance.")

    # News checks
    news_df = news_snapshot.get("news", pd.DataFrame())
    news_status = news_snapshot.get("status", pd.DataFrame())
    news_rows = int(len(news_df)) if isinstance(news_df, pd.DataFrame) else 0
    add("News", "Merged article rows", "PASS" if news_rows >= 5 else ("WARN" if news_rows > 0 else "FAIL"), "Medium", news_rows, ">= 5 rows", "RSS + fallback should guarantee baseline coverage.")
    if isinstance(news_status, pd.DataFrame) and not news_status.empty:
        rss_ok = int(news_status["ok"].fillna(False).sum())
        add("News", "RSS query availability", "PASS" if rss_ok >= 1 else "WARN", "Low", f"{rss_ok}/{len(news_status)}", ">= 1 query live", "Fallback still keeps UI usable.")

    # OpenF1 session keys for telemetry explorer
    openf1_sessions = snapshot.get("openf1_sessions", {}).get("data", pd.DataFrame())
    if isinstance(openf1_sessions, pd.DataFrame) and not openf1_sessions.empty:
        has_keys = "session_key" in openf1_sessions.columns and openf1_sessions["session_key"].notna().any()
        add("Data Lab", "OpenF1 session keys", "PASS" if has_keys else "WARN", "High", str(bool(has_keys)), "True", "Needed for telemetry explorer session selection.")
    else:
        add("Data Lab", "OpenF1 sessions rows", "WARN", "Medium", 0, "> 0 rows", "Pre-season feed timing may delay availability.")

    # Local data file checks (performance-sensitive local pipeline)
    data_dir = Path(__file__).parent.parent / "data"
    file_2025 = data_dir / "Formula1_2025Season_RaceResults.csv"
    file_2026 = data_dir / "Formula1_2026Season_RaceResults.csv"
    file_preseason = data_dir / "F1_2026_PreseasonTesting_TeamSummary.csv"
    add("Local Data", "2025 race CSV", "PASS" if file_2025.exists() else "FAIL", "High", str(file_2025.exists()), "True", str(file_2025))
    add("Local Data", "2026 race CSV", "PASS" if file_2026.exists() else "WARN", "Medium", str(file_2026.exists()), "Optional pre-season", str(file_2026))
    add("Local Data", "2026 pre-season testing CSV", "PASS" if file_preseason.exists() else "FAIL", "High", str(file_preseason.exists()), "True", str(file_preseason))
    if file_preseason.exists():
        try:
            pre_df = load_preseason_testing_team_summary(str(file_preseason))
            rows_count = len(pre_df) if isinstance(pre_df, pd.DataFrame) else 0
            venues = int(pre_df["venue"].nunique()) if isinstance(pre_df, pd.DataFrame) and "venue" in pre_df.columns else 0
            add("ML Pipeline", "Pre-season source coverage", "PASS" if venues >= 2 else "WARN", "Medium", f"{rows_count} rows / {venues} venues", ">= 2 venues (Spain + Bahrain)", "Curated team summary feeds 2026 ML priors.")
        except Exception as e:
            add("ML Pipeline", "Pre-season source parse", "WARN", "Medium", "parse_error", "parsable CSV", str(e))
    feature_schema_path = Path(__file__).parent.parent / "models" / "feature_columns.json"
    add("ML Pipeline", "Feature schema artifact", "PASS" if feature_schema_path.exists() else "WARN", "Low", str(feature_schema_path.exists()), "Optional (after training)", str(feature_schema_path))

    # Library coverage completeness (inventory, not runtime)
    coverage_df = pd.DataFrame(F1_LIBRARY_COVERAGE)
    integrated = int((coverage_df["coverage_status"] == "Integrated").sum()) if not coverage_df.empty else 0
    add("Coverage", "Library capability inventory", "PASS" if integrated >= 10 else "WARN", "Low", f"{integrated}/{len(coverage_df)}", ">= 10 integrated rows", "Tracks feature coverage by data source.")

    diag_df = pd.DataFrame(rows)
    status_order = pd.CategoricalDtype(categories=["FAIL", "WARN", "PASS"], ordered=True)
    if not diag_df.empty:
        diag_df["status"] = diag_df["status"].astype(status_order)
        diag_df = diag_df.sort_values(["status", "severity", "component", "check"]).reset_index(drop=True)
    return diag_df


def _fig_diagnostics_summary(diag_df: pd.DataFrame) -> go.Figure:
    if diag_df is None or diag_df.empty:
        return go.Figure()
    g = diag_df.groupby(["status", "severity"], dropna=False).size().reset_index(name="count")
    fig = px.bar(
        g,
        x="status",
        y="count",
        color="severity",
        barmode="stack",
        text="count",
        title="Diagnostics/Test Result Summary",
        color_discrete_map={"High": "#E10600", "Medium": "#F5B942", "Low": "#4D7CFE"},
        category_orders={"status": ["FAIL", "WARN", "PASS"]},
    )
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=55, b=20), xaxis_title="", yaxis_title="Checks")
    return fig


@st.cache_data(ttl=1800, show_spinner=False)
def load_preseason_testing_snapshot() -> Dict[str, Any]:
    summary_df = load_preseason_testing_team_summary()
    features_df = load_preseason_testing_team_features()
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        df = summary_df.copy()
        if "start_date" in df.columns:
            df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce", utc=True)
        if "end_date" in df.columns:
            df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce", utc=True)
        venue_counts = (
            df.groupby(["venue", "test_label"], dropna=False).size().reset_index(name="rows")
            if "venue" in df.columns and "test_label" in df.columns
            else pd.DataFrame()
        )
    else:
        df = pd.DataFrame()
        venue_counts = pd.DataFrame()
    return {"summary": df, "features": features_df, "venue_counts": venue_counts}


def _fig_preseason_bahrain_laps(summary_df: pd.DataFrame) -> go.Figure:
    if summary_df is None or summary_df.empty:
        return go.Figure()
    df = summary_df.copy()
    if "venue" not in df.columns:
        return go.Figure()
    df = df[df["venue"].astype(str).str.contains("bahrain", case=False, na=False)].copy()
    if df.empty or "total_laps" not in df.columns:
        return go.Figure()
    df["total_laps"] = pd.to_numeric(df["total_laps"], errors="coerce")
    df = df.dropna(subset=["total_laps"])
    if df.empty:
        return go.Figure()
    fig = px.bar(
        df,
        x="team",
        y="total_laps",
        color="test_label",
        barmode="group",
        text="total_laps",
        title="Bahrain 2026 Official Tests: Team Mileage (Total Laps)",
        color_discrete_sequence=["#00D2BE", "#E10600"],
    )
    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=55, b=20), xaxis_title="", yaxis_title="Laps")
    return fig


def _fig_preseason_bahrain_pace(summary_df: pd.DataFrame) -> go.Figure:
    if summary_df is None or summary_df.empty:
        return go.Figure()
    df = summary_df.copy()
    if "venue" not in df.columns:
        return go.Figure()
    df = df[df["venue"].astype(str).str.contains("bahrain", case=False, na=False)].copy()
    if df.empty:
        return go.Figure()
    df["fastest_lap_seconds"] = pd.to_numeric(df.get("fastest_lap_seconds"), errors="coerce")
    df = df.dropna(subset=["fastest_lap_seconds"])
    if df.empty:
        return go.Figure()
    fig = px.scatter(
        df,
        x="team",
        y="fastest_lap_seconds",
        color="test_label",
        hover_data=["fastest_driver", "fastest_lap_time", "total_laps"],
        title="Bahrain 2026 Official Tests: Best Team Lap Time (Lower is Better)",
        color_discrete_sequence=["#00D2BE", "#E10600"],
    )
    fig.update_traces(mode="markers+lines")
    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=55, b=20), xaxis_title="", yaxis_title="Best lap (s)")
    return fig


def _fig_spain_shakedown_participation(summary_df: pd.DataFrame) -> go.Figure:
    if summary_df is None or summary_df.empty:
        return go.Figure()
    df = summary_df.copy()
    df = df[df["venue"].astype(str).str.contains("barcelona", case=False, na=False)].copy()
    if df.empty:
        return go.Figure()
    df["participated"] = pd.to_numeric(df.get("participated"), errors="coerce").fillna(0).astype(int)
    df["late_start"] = pd.to_numeric(df.get("late_start"), errors="coerce").fillna(0).astype(int)
    df["status"] = df.apply(
        lambda r: "Late Start" if int(r.get("late_start", 0)) == 1 else ("Participated" if int(r.get("participated", 0)) == 1 else "Absent"),
        axis=1,
    )
    fig = px.bar(
        df.sort_values(["participated", "late_start", "team"], ascending=[False, False, True]),
        x="team",
        y="participated",
        color="status",
        text="status",
        title="Barcelona 2026 Private Shakedown: Participation Status (Officially Reported)",
        color_discrete_map={"Participated": "#00D2BE", "Late Start": "#F5B942", "Absent": "#E10600"},
    )
    fig.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=55, b=20), xaxis_title="", yaxis_title="Participated (1/0)")
    return fig


def _compute_position_accuracy_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    """Derive classification-like accuracy metrics from regression predictions."""
    if not isinstance(results_df, pd.DataFrame) or results_df.empty:
        return {}
    actual = pd.to_numeric(results_df.get("Actual"), errors="coerce")
    pred = pd.to_numeric(results_df.get("Predicted"), errors="coerce")
    valid = actual.notna() & pred.notna()
    if not valid.any():
        return {}

    actual_i = actual[valid].round().astype(int)
    pred_i = pred[valid].round().astype(int)
    abs_pos_err = (actual_i - pred_i).abs()
    n = len(abs_pos_err)
    if n == 0:
        return {}

    return {
        "ACC_EXACT_PCT": round(float((abs_pos_err == 0).mean() * 100.0), 2),
        "ACC_TOL1_PCT": round(float((abs_pos_err <= 1).mean() * 100.0), 2),
        "ACC_TOL2_PCT": round(float((abs_pos_err <= 2).mean() * 100.0), 2),
        "ACC_TOL3_PCT": round(float((abs_pos_err <= 3).mean() * 100.0), 2),
        "MEDIAN_ABS_POS_ERR_ROUND": round(float(abs_pos_err.median()), 2),
    }


def _get_ml_training_file_status(data_dir: Path) -> Tuple[pd.DataFrame, str]:
    rows: List[Dict[str, Any]] = []
    sig_parts: List[str] = []
    for name in ["Formula1_2025Season_RaceResults.csv", "Formula1_2026Season_RaceResults.csv"]:
        path = Path(data_dir) / name
        exists = path.exists()
        size_bytes = int(path.stat().st_size) if exists else 0
        mtime_epoch = int(path.stat().st_mtime) if exists else 0
        rows.append(
            {
                "file": name,
                "exists": exists,
                "size_bytes": size_bytes if exists else None,
                "modified_at_utc": pd.to_datetime(mtime_epoch, unit="s", utc=True) if exists else pd.NaT,
            }
        )
        sig_parts.append(f"{name}:{int(exists)}:{size_bytes}:{mtime_epoch}")
    return pd.DataFrame(rows), "|".join(sig_parts)


@st.cache_data(ttl=900, show_spinner=False)
def run_ml_pipeline_snapshot(refresh_nonce: int = 0) -> Dict[str, Any]:
    _ = refresh_nonce
    data_dir = Path(__file__).parent.parent / "data"
    race_file_status, train_data_signature = _get_ml_training_file_status(data_dir)
    raw_df, cleaned_df, err = _load_ml_training_dataframes(data_dir)
    if err:
        return {
            "ok": False,
            "error": err,
            "metrics": {},
            "results": pd.DataFrame(),
            "feature_importances": pd.DataFrame(),
            "feature_coverage": pd.DataFrame(),
            "dataset_summary": pd.DataFrame(),
            "preseason_summary": pd.DataFrame(),
            "preseason_features": pd.DataFrame(),
            "race_file_status": race_file_status,
            "train_data_signature": train_data_signature,
        }

    model, X_test, y_test = train_model(
        cleaned_df,
        enable_preseason_features=True,
        preseason_csv_path=str(DEFAULT_PRESEASON_TEAM_SUMMARY_PATH),
        persist=False,
    )
    metrics, results_df = evaluate_model(model, X_test, y_test)
    metrics = {**metrics, **_compute_position_accuracy_metrics(results_df)}

    feature_names = list(X_test.columns)
    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != len(feature_names):
        importances = [0.0] * len(feature_names)
    importances_df = pd.DataFrame(
        {"feature": feature_names, "importance": pd.to_numeric(importances, errors="coerce")}
    ).sort_values("importance", ascending=False, na_position="last")

    preseason_cols = [c for c in feature_names if str(c).startswith("preseason_")]
    feature_coverage = pd.DataFrame()
    if preseason_cols:
        cov_rows = []
        for col in preseason_cols:
            series = pd.to_numeric(X_test[col], errors="coerce")
            cov_rows.append(
                {
                    "feature": col,
                    "nonzero_ratio": float((series.fillna(0) != 0).mean()) if len(series) else 0.0,
                    "mean": float(series.fillna(0).mean()) if len(series) else 0.0,
                    "min": float(series.min()) if series.notna().any() else None,
                    "max": float(series.max()) if series.notna().any() else None,
                }
            )
        feature_coverage = pd.DataFrame(cov_rows).sort_values("nonzero_ratio", ascending=False)

    dataset_summary = pd.DataFrame(
        [
            {"metric": "Race rows (raw)", "value": len(raw_df)},
            {"metric": "Race rows (cleaned)", "value": len(cleaned_df)},
            {"metric": "Race rows finished", "value": int(cleaned_df["Finished"].astype(bool).sum()) if "Finished" in cleaned_df.columns else 0},
            {"metric": "Train/Test features", "value": len(feature_names)},
            {"metric": "Pre-season feature columns", "value": len(preseason_cols)},
            {"metric": "X_test rows", "value": len(X_test)},
            {"metric": "Pre-season CSV rows", "value": len(load_preseason_testing_team_summary())},
        ]
    )

    return {
        "ok": True,
        "error": None,
        "metrics": metrics,
        "results": results_df,
        "feature_importances": importances_df,
        "feature_coverage": feature_coverage,
        "dataset_summary": dataset_summary,
        "preseason_summary": load_preseason_testing_team_summary(),
        "preseason_features": load_preseason_testing_team_features(),
        "race_file_status": race_file_status,
        "train_data_signature": train_data_signature,
    }


def _load_ml_training_dataframes(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    data_dir = Path(__file__).parent.parent / "data"
    race_paths = []
    race_2025 = data_dir / "Formula1_2025Season_RaceResults.csv"
    race_2026 = data_dir / "Formula1_2026Season_RaceResults.csv"
    if race_2025.exists():
        race_paths.append((2025, race_2025))
    if race_2026.exists():
        race_paths.append((2026, race_2026))

    raw_frames: List[pd.DataFrame] = []
    for season, path in race_paths:
        try:
            df_raw = pd.read_csv(path)
            if not df_raw.empty:
                df_raw["Season"] = season
                raw_frames.append(df_raw)
        except Exception:
            continue
    raw_df = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    if raw_df.empty:
        return pd.DataFrame(), pd.DataFrame(), "No race CSV available for ML pipeline diagnostics."

    try:
        cleaned_df = clean_data(raw_df)
    except Exception as e:
        return raw_df, pd.DataFrame(), f"clean_data failed: {e}"
    return raw_df, cleaned_df, None


def run_ml_pipeline_persist_training() -> Dict[str, Any]:
    """Train and persist production ML artifacts (model + encoders + feature schema)."""
    data_dir = Path(__file__).parent.parent / "data"
    race_file_status, train_data_signature = _get_ml_training_file_status(data_dir)
    raw_df, cleaned_df, err = _load_ml_training_dataframes(data_dir)
    if err:
        return {
            "ok": False,
            "error": err,
            "metrics": {},
            "artifacts": pd.DataFrame(),
            "results": pd.DataFrame(),
            "race_file_status": race_file_status,
            "train_data_signature": train_data_signature,
        }

    model, X_test, y_test = train_model(
        cleaned_df,
        enable_preseason_features=True,
        preseason_csv_path=str(DEFAULT_PRESEASON_TEAM_SUMMARY_PATH),
        persist=True,
    )
    metrics, results_df = evaluate_model(model, X_test, y_test)
    metrics = {**metrics, **_compute_position_accuracy_metrics(results_df)}

    models_dir = Path(__file__).parent.parent / "models"
    artifact_rows = []
    for name in ["f1_model.pkl", "Driver_encoder.pkl", "Team_encoder.pkl", "Track_encoder.pkl", "feature_columns.json"]:
        path = models_dir / name
        artifact_rows.append(
            {
                "artifact": name,
                "exists": path.exists(),
                "size_bytes": int(path.stat().st_size) if path.exists() else None,
                "modified_at_utc": pd.to_datetime(path.stat().st_mtime, unit="s", utc=True) if path.exists() else pd.NaT,
            }
        )
    artifacts_df = pd.DataFrame(artifact_rows)

    return {
        "ok": True,
        "error": None,
        "metrics": metrics,
        "results": results_df,
        "artifacts": artifacts_df,
        "feature_count": len(X_test.columns),
        "eval_rows": len(X_test),
        "model_has_preseason_features": any(str(c).startswith("preseason_") for c in X_test.columns),
        "race_file_status": race_file_status,
        "train_data_signature": train_data_signature,
    }


def _openf1_list_to_df(payload: Any) -> pd.DataFrame:
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        for key in ("data", "items", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return pd.DataFrame(value)
    return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def load_openf1_session_bundle(session_key: int, driver_number: Optional[int] = None, refresh_nonce: int = 0) -> Dict[str, pd.DataFrame]:
    _ = refresh_nonce
    base = "https://api.openf1.org/v1"
    endpoints = {
        "drivers": f"{base}/drivers?session_key={session_key}",
        "weather": f"{base}/weather?session_key={session_key}",
        "laps": f"{base}/laps?session_key={session_key}",
    }
    if driver_number is not None:
        endpoints["car_data"] = f"{base}/car_data?session_key={session_key}&driver_number={driver_number}"
        endpoints["position"] = f"{base}/position?session_key={session_key}&driver_number={driver_number}"

    out: Dict[str, pd.DataFrame] = {}
    for name, url in endpoints.items():
        payload, _, err = _http_json(url, timeout=12)
        if err:
            out[name] = pd.DataFrame()
            continue
        df = _openf1_list_to_df(payload)
        if df.empty:
            out[name] = df
            continue
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        if "date_start" in df.columns:
            df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce", utc=True)
        if name == "car_data" and len(df) > 3000:
            df = df.sort_values("date" if "date" in df.columns else df.columns[0]).tail(3000).reset_index(drop=True)
        if name in {"weather", "position"} and len(df) > 1000:
            sort_col = "date" if "date" in df.columns else ("date_start" if "date_start" in df.columns else df.columns[0])
            df = df.sort_values(sort_col).tail(1000).reset_index(drop=True)
        if name == "laps" and len(df) > 500:
            # Keep all laps for selected driver if filtered; otherwise sample latest rows.
            sort_col = "lap_number" if "lap_number" in df.columns else df.columns[0]
            df = df.sort_values(sort_col).tail(500).reset_index(drop=True)
        out[name] = df
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def load_fastf1_schedule_preview(year: int, refresh_nonce: int = 0) -> pd.DataFrame:
    _ = refresh_nonce
    try:
        sched = fastf1.get_event_schedule(year)
        if sched is None or sched.empty:
            return pd.DataFrame()
        return sched.copy()
    except Exception:
        return pd.DataFrame()


def _inject_2026_hub_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --hub-bg-1: #090b10;
          --hub-bg-2: #121722;
          --hub-card: rgba(255,255,255,0.03);
          --hub-border: rgba(255,255,255,0.08);
          --hub-red: #E10600;
          --hub-teal: #00D2BE;
          --hub-amber: #F5B942;
          --hub-text: #E9EEF7;
          --hub-muted: #AAB5C8;
        }
        .f1-2026-hero {
          background: radial-gradient(1200px 300px at 10% -10%, rgba(225,6,0,0.35), transparent 60%),
                      radial-gradient(800px 280px at 95% 0%, rgba(0,210,190,0.18), transparent 60%),
                      linear-gradient(135deg, var(--hub-bg-1), var(--hub-bg-2));
          border: 1px solid var(--hub-border);
          border-radius: 18px;
          padding: 18px 18px 14px 18px;
          margin: 8px 0 14px 0;
        }
        .f1-2026-eyebrow { color: var(--hub-teal); font-size: 12px; text-transform: uppercase; letter-spacing: 1.4px; font-weight: 700; margin-bottom: 6px; }
        .f1-2026-title { color: var(--hub-text); font-weight: 800; font-size: 28px; line-height: 1.1; margin: 0 0 8px 0; }
        .f1-2026-sub { color: var(--hub-muted); font-size: 14px; margin-bottom: 12px; max-width: 900px; }
        .f1-2026-pill { display: inline-block; border-radius: 999px; border: 1px solid var(--hub-border); padding: 4px 10px; color: var(--hub-text); font-size: 12px; margin: 2px 6px 2px 0; background: rgba(255,255,255,0.02); }
        .f1-2026-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
        .f1-2026-card { background: var(--hub-card); border: 1px solid var(--hub-border); border-radius: 14px; padding: 10px 12px; }
        .f1-2026-card-label { color: var(--hub-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
        .f1-2026-card-value { color: var(--hub-text); font-size: 20px; font-weight: 800; }
        .f1-2026-card-note { color: var(--hub-muted); font-size: 11px; margin-top: 2px; }
        .f1-2026-news-card { border: 1px solid var(--hub-border); background: rgba(255,255,255,0.02); border-radius: 12px; padding: 10px 12px; margin: 8px 0; }
        .f1-2026-news-meta { color: var(--hub-muted); font-size: 12px; margin-bottom: 4px; }
        .f1-2026-news-headline { color: var(--hub-text); font-weight: 700; font-size: 14px; line-height: 1.25; margin-bottom: 5px; }
        .f1-2026-news-summary { color: var(--hub-muted); font-size: 12px; line-height: 1.3; }
        @media (max-width: 900px) { .f1-2026-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .f1-2026-title { font-size: 22px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _fig_api_status(status: pd.DataFrame) -> go.Figure:
    if status is None or status.empty:
        return go.Figure()
    d = status.copy()
    d["latency_ms"] = pd.to_numeric(d.get("latency_ms"), errors="coerce")
    d["rows"] = pd.to_numeric(d.get("rows"), errors="coerce").fillna(0)
    d["latency_plot"] = d["latency_ms"].fillna(0)
    d["status_label"] = d["ok"].map({True: "OK", False: "Unavailable"}).fillna("Unknown")
    fig = px.bar(
        d,
        x="latency_plot",
        y="api",
        color="status_label",
        orientation="h",
        text="rows",
        title="Live Data Feed Status (Latency + Row Count)",
        color_discrete_map={"OK": "#00D2BE", "Unavailable": "#E10600", "Unknown": "#8A8A8A"},
        hover_data={"url": True, "error": True, "latency_ms": True, "rows": True},
    )
    fig.update_layout(template="plotly_dark", height=370, margin=dict(l=20, r=20, t=55, b=20), xaxis_title="Latency (ms)")
    return fig


def _fig_calendar_timeline(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return go.Figure()
    fig = px.scatter(
        df,
        x="race_date",
        y="round",
        color="region",
        symbol="is_sprint",
        hover_data={"event": True, "location": True, "country": True, "race_date": "|%d %b %Y"},
        title="F1 2026 Calendar Timeline",
        color_discrete_sequence=["#E10600", "#00D2BE", "#1E41FF", "#F4C430", "#888888"],
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(template="plotly_dark", height=470, margin=dict(l=20, r=20, t=55, b=20))
    fig.update_yaxes(dtick=1)
    return fig


def _fig_calendar_months(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return go.Figure()
    g = (
        df.groupby(["month_num", "month", "region"], dropna=False)
        .size()
        .reset_index(name="races")
        .sort_values(["month_num", "region"])
    )
    fig = px.bar(
        g,
        x="month",
        y="races",
        color="region",
        barmode="stack",
        text="races",
        title="Calendar Density by Month / Region",
        color_discrete_sequence=["#E10600", "#00D2BE", "#1E41FF", "#F4C430", "#888888"],
    )
    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def _fig_reg_changes() -> go.Figure:
    d = REG_CHANGE.melt(id_vars="metric", var_name="era", value_name="value")
    fig = px.bar(
        d,
        x="metric",
        y="value",
        color="era",
        barmode="group",
        text="value",
        title="Selected 2025 vs 2026 Regulation Metrics",
        color_discrete_map={"2025": "#777777", "2026": "#E10600"},
    )
    fig.update_layout(template="plotly_dark", height=430, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def _fig_pu_split() -> go.Figure:
    fig = px.bar(
        PU_SPLIT,
        x="era",
        y="share",
        color="source",
        barmode="stack",
        text="share",
        title="Power Unit Split (Current vs 2026 Target)",
        color_discrete_map={"Electric": "#00D2BE", "ICE": "#E10600"},
    )
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="inside")
    fig.update_layout(template="plotly_dark", yaxis=dict(range=[0, 100]), height=350, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def _fig_update_timeline() -> go.Figure:
    d = pd.DataFrame(OFFICIAL_2026_UPDATES).copy()
    d["date"] = pd.to_datetime(d["date"], utc=True)
    fig = px.scatter(
        d.sort_values("date"),
        x="date",
        y="category",
        color="category",
        hover_name="title",
        hover_data={"source": True, "summary": True, "date": "|%d %b %Y"},
        title="Verified 2026 Update Timeline (Official Sources)",
        color_discrete_sequence=["#E10600", "#00D2BE", "#1E41FF", "#F4C430", "#999999"],
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=20, r=20, t=55, b=20), showlegend=False)
    return fig


def _fig_preseason_tests() -> go.Figure:
    d = pd.DataFrame(PRESEASON_TESTS_2026).copy()
    d["start_date"] = pd.to_datetime(d["start_date"], utc=True)
    d["end_date"] = pd.to_datetime(d["end_date"], utc=True)
    fig = px.timeline(
        d,
        x_start="start_date",
        x_end="end_date",
        y="label",
        color="venue",
        hover_data={"country": True, "venue": True},
        title="2026 Pre-season Testing Windows (FIA WMSC Update)",
        color_discrete_sequence=["#E10600", "#00D2BE", "#1E41FF"],
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=55, b=20), xaxis_title="Date")
    return fig


def _fig_feature_readiness_counts(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    g = df.groupby(["category", "status"], dropna=False).size().reset_index(name="count")
    fig = px.bar(
        g,
        x="category",
        y="count",
        color="status",
        barmode="stack",
        text="count",
        title="2026 Readiness Across Existing App Features",
        color_discrete_map={"Ready": "#00D2BE", "Partial": "#F5B942", "Season Start": "#4D7CFE", "Rebuild": "#E10600"},
    )
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=55, b=20), xaxis_title="", yaxis_title="Feature count")
    return fig


def _fig_feature_priority(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    g = df.groupby(["priority", "status"], dropna=False).size().reset_index(name="count")
    fig = px.bar(
        g,
        x="priority",
        y="count",
        color="status",
        barmode="group",
        text="count",
        title="Backlog Priority vs Status",
        color_discrete_map={"Ready": "#00D2BE", "Partial": "#F5B942", "Season Start": "#4D7CFE", "Rebuild": "#E10600"},
    )
    fig.update_layout(template="plotly_dark", height=340, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def _fig_news_timeline(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return go.Figure()
    d = df.copy()
    d["published_at"] = pd.to_datetime(d["published_at"], errors="coerce", utc=True)
    d = d.dropna(subset=["published_at"]).sort_values("published_at")
    if d.empty:
        return go.Figure()
    fig = px.scatter(
        d,
        x="published_at",
        y="source",
        color="topic",
        hover_name="headline",
        hover_data={"published_at": "|%Y-%m-%d %H:%M UTC", "feed": True},
        title="2026 News Timeline (Live RSS + Curated Fallback)",
        color_discrete_sequence=["#E10600", "#00D2BE", "#1E41FF", "#F5B942", "#9B59B6", "#95A5A6"],
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def _fig_news_topics(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return go.Figure()
    g = df.groupby(["topic", "source"], dropna=False).size().reset_index(name="count")
    fig = px.bar(g, x="topic", y="count", color="source", barmode="stack", text="count", title="News Coverage by Topic and Source")
    fig.update_layout(template="plotly_dark", height=340, margin=dict(l=20, r=20, t=55, b=20), xaxis_title="", yaxis_title="Articles")
    return fig


def _fig_openf1_car_data(car_df: pd.DataFrame) -> go.Figure:
    if car_df is None or car_df.empty:
        return go.Figure()
    d = car_df.copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True)
        d = d.dropna(subset=["date"])
    x_col = "date" if "date" in d.columns else d.columns[0]

    fig = go.Figure()
    if "speed" in d.columns:
        fig.add_trace(go.Scatter(x=d[x_col], y=pd.to_numeric(d["speed"], errors="coerce"), mode="lines", name="Speed", line=dict(color="#E10600", width=2)))
    if "throttle" in d.columns:
        fig.add_trace(go.Scatter(x=d[x_col], y=pd.to_numeric(d["throttle"], errors="coerce"), mode="lines", name="Throttle", yaxis="y2", line=dict(color="#00D2BE", width=1.6)))
    if "brake" in d.columns:
        fig.add_trace(go.Scatter(x=d[x_col], y=pd.to_numeric(d["brake"], errors="coerce") * 100, mode="lines", name="Brake x100", yaxis="y2", line=dict(color="#F5B942", width=1.4)))
    if "drs" in d.columns:
        fig.add_trace(go.Scatter(x=d[x_col], y=pd.to_numeric(d["drs"], errors="coerce") * 10, mode="lines", name="DRS x10", yaxis="y2", line=dict(color="#9B59B6", width=1.2)))

    fig.update_layout(
        template="plotly_dark",
        title="OpenF1 Car Telemetry (Sample)",
        height=360,
        margin=dict(l=20, r=20, t=55, b=20),
        yaxis=dict(title="Speed"),
        yaxis2=dict(title="Control signals", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def _fig_openf1_laps(laps_df: pd.DataFrame) -> go.Figure:
    if laps_df is None or laps_df.empty:
        return go.Figure()
    d = laps_df.copy()
    if "lap_number" not in d.columns or "lap_duration" not in d.columns:
        return go.Figure()
    d["lap_number"] = pd.to_numeric(d["lap_number"], errors="coerce")
    d["lap_duration"] = pd.to_numeric(d["lap_duration"], errors="coerce")
    d = d.dropna(subset=["lap_number", "lap_duration"])
    if d.empty:
        return go.Figure()
    color_col = "driver_number" if "driver_number" in d.columns else None
    fig = px.line(d.sort_values("lap_number"), x="lap_number", y="lap_duration", color=color_col, markers=True, title="OpenF1 Lap Duration by Lap")
    fig.update_layout(template="plotly_dark", height=340, margin=dict(l=20, r=20, t=55, b=20), xaxis_title="Lap", yaxis_title="Lap duration (s)")
    return fig


def _fig_openf1_weather(weather_df: pd.DataFrame) -> go.Figure:
    if weather_df is None or weather_df.empty:
        return go.Figure()
    d = weather_df.copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True)
        d = d.dropna(subset=["date"])
    x_col = "date" if "date" in d.columns else d.columns[0]
    fig = go.Figure()
    for col, color in [
        ("air_temperature", "#00D2BE"),
        ("track_temperature", "#E10600"),
        ("humidity", "#4D7CFE"),
        ("wind_speed", "#F5B942"),
    ]:
        if col in d.columns:
            fig.add_trace(go.Scatter(x=d[x_col], y=pd.to_numeric(d[col], errors="coerce"), mode="lines", name=col))
    fig.update_layout(template="plotly_dark", title="OpenF1 Weather Trend", height=340, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def _fig_openf1_position(position_df: pd.DataFrame) -> go.Figure:
    if position_df is None or position_df.empty:
        return go.Figure()
    d = position_df.copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True)
        d = d.dropna(subset=["date"])
    pos_col = "position" if "position" in d.columns else ("track_position" if "track_position" in d.columns else None)
    if pos_col is None:
        return go.Figure()
    fig = px.line(d, x=("date" if "date" in d.columns else d.columns[0]), y=pos_col, title="OpenF1 Position Trace")
    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=55, b=20), yaxis_autorange="reversed")
    return fig


def _render_hero(snapshot: Dict[str, Any], news_snapshot: Dict[str, Any]) -> None:
    calendar_df, calendar_source = _pick_calendar(snapshot)
    status = snapshot.get("status", pd.DataFrame())
    news_df = news_snapshot.get("news", pd.DataFrame())

    api_ok = int(status["ok"].fillna(False).sum()) if not status.empty else 0
    api_total = int(len(status)) if status is not None else 0
    races = int(len(calendar_df)) if isinstance(calendar_df, pd.DataFrame) else 0
    sprints = int(calendar_df["is_sprint"].fillna(False).sum()) if isinstance(calendar_df, pd.DataFrame) and not calendar_df.empty else 0
    latest_news = "N/A"
    if isinstance(news_df, pd.DataFrame) and not news_df.empty:
        ts = pd.to_datetime(news_df["published_at"], errors="coerce", utc=True).dropna()
        if not ts.empty:
            latest_news = ts.max().strftime("%Y-%m-%d")

    st.markdown(
        f"""
        <div class="f1-2026-hero">
          <div class="f1-2026-eyebrow">Research + Live Data + News</div>
          <div class="f1-2026-title">F1 2026 Mission Control</div>
          <div class="f1-2026-sub">
            Hub transisi dari dashboard musim 2025 ke era regulasi 2026. Menggabungkan riset resmi FIA/Formula1.com,
            monitor beberapa sumber data publik, berita terbaru, dan matriks kesiapan semua fitur utama app.
          </div>
          <span class="f1-2026-pill">Regulation reset</span>
          <span class="f1-2026-pill">11 teams (Cadillac)</span>
          <span class="f1-2026-pill">24 rounds</span>
          <span class="f1-2026-pill">6 sprints</span>
          <span class="f1-2026-pill">Source: {calendar_source}</span>
          <div class="f1-2026-grid">
            <div class="f1-2026-card">
              <div class="f1-2026-card-label">Calendar Rounds</div>
              <div class="f1-2026-card-value">{races}</div>
              <div class="f1-2026-card-note">official fallback + live sync</div>
            </div>
            <div class="f1-2026-card">
              <div class="f1-2026-card-label">Sprint Weekends</div>
              <div class="f1-2026-card-value">{sprints}</div>
              <div class="f1-2026-card-note">confirmed 2026 sprint venues</div>
            </div>
            <div class="f1-2026-card">
              <div class="f1-2026-card-label">Data Feed Health</div>
              <div class="f1-2026-card-value">{api_ok}/{api_total}</div>
              <div class="f1-2026-card-note">FastF1 + Jolpica + OpenF1 monitored</div>
            </div>
            <div class="f1-2026-card">
              <div class="f1-2026-card-label">Latest News in Feed</div>
              <div class="f1-2026-card-value">{latest_news}</div>
              <div class="f1-2026-card-note">RSS + curated official fallback</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_overview_tab(snapshot: Dict[str, Any], news_snapshot: Dict[str, Any]) -> None:
    st.caption(f"Time display uses user offset: UTC{_get_user_utc_offset()}")
    _plotly_chart(_fig_update_timeline(), use_container_width=True, config=PLOTLY_CONFIG)

    col1, col2 = st.columns([3, 2])
    with col1:
        cal_df, cal_src = _pick_calendar(snapshot)
        st.markdown(f"### 2026 Calendar Snapshot ({cal_src})")
        show = cal_df[["round", "race_date", "event", "location", "country", "is_sprint"]].copy()
        if not show.empty:
            show["race_date"] = show["race_date"].apply(lambda x: _fmt_user_time(x, "%Y-%m-%d %H:%M"))
            show = show.rename(columns={"race_date": f"race_date (UTC{_get_user_utc_offset()})"})
        st.dataframe(show, hide_index=True, use_container_width=True)
    with col2:
        st.markdown("### Pre-season Testing")
        _plotly_chart(_fig_preseason_tests(), use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("### 2026 Research Coverage")
        st.markdown("- Calendar + Sprint + Testing")
        st.markdown("- Regulations + Governance")
        st.markdown("- News feed + official fallback")
        st.markdown("- Feature readiness matrix")
        st.markdown("- Live data health & previews")

    news_df = news_snapshot.get("news", pd.DataFrame())
    st.markdown("### Top News Headlines")
    if isinstance(news_df, pd.DataFrame) and not news_df.empty:
        for row in news_df.head(5).to_dict("records"):
            pub = pd.to_datetime(row.get("published_at"), errors="coerce", utc=True)
            pub_s = _fmt_user_time(pub, "%Y-%m-%d %H:%M") if pd.notna(pub) else "N/A"
            st.markdown(
                f"""
                <div class="f1-2026-news-card">
                  <div class="f1-2026-news-meta">{pub_s} | {row.get('source', 'Unknown')} | {row.get('topic', 'General')}</div>
                  <div class="f1-2026-news-headline">{row.get('headline', '')}</div>
                  <div class="f1-2026-news-summary">{row.get('summary', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Belum ada item berita dari feed saat ini. Fallback resmi akan digunakan jika tersedia.")


def _render_research_tab() -> None:
    updates_df = pd.DataFrame(OFFICIAL_2026_UPDATES).copy()
    updates_df["date"] = pd.to_datetime(updates_df["date"])
    updates_df = updates_df.sort_values("date", ascending=False).reset_index(drop=True)

    st.markdown("### Official Research Digest")
    st.markdown(
        "Riset ini merangkum update 2026 dari FIA, Formula1.com, dan Corporate Formula 1 yang berdampak langsung ke fitur dashboard."
    )
    _plotly_chart(_fig_update_timeline(), use_container_width=True, config=PLOTLY_CONFIG)
    st.dataframe(updates_df[["date", "category", "title", "source", "url"]], hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 2026 Testing Windows")
        test_df = pd.DataFrame(PRESEASON_TESTS_2026).copy()
        test_df["start_date"] = test_df["start_date"].apply(lambda x: _fmt_user_time(pd.to_datetime(x, utc=True), "%Y-%m-%d %H:%M"))
        test_df["end_date"] = test_df["end_date"].apply(lambda x: _fmt_user_time(pd.to_datetime(x, utc=True), "%Y-%m-%d %H:%M"))
        st.dataframe(test_df, hide_index=True, use_container_width=True)
    with c2:
        st.markdown("### Sprint Weekends (Confirmed)")
        st.dataframe(pd.DataFrame(sorted(SPRINT_2026), columns=["event"]), hide_index=True, use_container_width=True)

    st.markdown("### Regulation Highlights")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Car Width", "190 cm", "-10 cm")
    m2.metric("Car Length", "340 cm", "-20 cm")
    m3.metric("Min Weight", "768 kg", "-30 kg")
    m4.metric("PU Split", "50% electric", "MGU-H removed")
    st.caption("Regulation figures are based on Formula1.com 2026 rules explainer and tracked against FIA regulation issue updates.")


def _render_feature_readiness_tab() -> None:
    df = pd.DataFrame(FEATURE_READINESS_2026)
    st.markdown("### Feature Readiness Matrix (Semua Fitur Utama)")
    st.markdown("Ini memetakan semua fitur yang sudah ada di app ke kesiapan operasional untuk musim/regulasi 2026.")

    a, b, c, d = st.columns(4)
    a.metric("Tracked", len(df))
    b.metric("Ready", int((df["status"] == "Ready").sum()))
    c.metric("Partial/Season Start", int(df["status"].isin(["Partial", "Season Start"]).sum()))
    d.metric("Rebuild", int((df["status"] == "Rebuild").sum()))

    _plotly_chart(_fig_feature_readiness_counts(df), use_container_width=True, config=PLOTLY_CONFIG)
    _plotly_chart(_fig_feature_priority(df), use_container_width=True, config=PLOTLY_CONFIG)

    priorities = st.multiselect("Priority filter", ["High", "Medium", "Low"], ["High", "Medium", "Low"], key="f1_2026_prio_filter")
    statuses = st.multiselect("Status filter", ["Ready", "Partial", "Season Start", "Rebuild"], ["Ready", "Partial", "Season Start", "Rebuild"], key="f1_2026_status_filter")
    filtered = df[df["priority"].isin(priorities) & df["status"].isin(statuses)].copy()
    st.dataframe(filtered, hide_index=True, use_container_width=True)


def _render_api_monitor_tab(snapshot: Dict[str, Any]) -> None:
    status = snapshot.get("status", pd.DataFrame())
    st.markdown("### Data Services")
    st.markdown("Memantau FastF1, Jolpica (schedule + standings), dan OpenF1 (meetings + sessions) untuk kesiapan data 2026.")
    st.caption(f"Schedule/session timestamps displayed in UTC{_get_user_utc_offset()}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sources checked", len(status))
    c2.metric("Available", int(status["ok"].fillna(False).sum()) if not status.empty else 0)
    c3.metric("Rows", int(pd.to_numeric(status["rows"], errors="coerce").fillna(0).sum()) if not status.empty else 0)
    lat = pd.to_numeric(status["latency_ms"], errors="coerce").dropna() if not status.empty else pd.Series(dtype=float)
    c4.metric("Median latency", f"{int(lat.median())} ms" if not lat.empty else "N/A")

    _plotly_chart(_fig_api_status(status), use_container_width=True, config=PLOTLY_CONFIG)
    st.dataframe(status, hide_index=True, use_container_width=True)

    cal_df, cal_src = _pick_calendar(snapshot)
    st.markdown(f"### Schedule Preview ({cal_src})")
    cal_show = cal_df[["round", "race_date", "event", "location", "country", "is_sprint"]].copy()
    if not cal_show.empty:
        cal_show["race_date"] = cal_show["race_date"].apply(lambda x: _fmt_user_time(x, "%Y-%m-%d %H:%M"))
        cal_show = cal_show.rename(columns={"race_date": f"race_date (UTC{_get_user_utc_offset()})"})
    st.dataframe(cal_show, hide_index=True, use_container_width=True)

    x1, x2 = st.columns(2)
    with x1:
        st.markdown("### Jolpica Driver Standings")
        drv_df = snapshot.get("jolpica_driver", {}).get("data", pd.DataFrame())
        if isinstance(drv_df, pd.DataFrame) and not drv_df.empty:
            st.dataframe(drv_df, hide_index=True, use_container_width=True)
        else:
            st.info(snapshot.get("jolpica_driver", {}).get("error", "Standings will appear once season results are published."))
    with x2:
        st.markdown("### Jolpica Constructor Standings")
        ctor_df = snapshot.get("jolpica_constructor", {}).get("data", pd.DataFrame())
        if isinstance(ctor_df, pd.DataFrame) and not ctor_df.empty:
            st.dataframe(ctor_df, hide_index=True, use_container_width=True)
        else:
            st.info(snapshot.get("jolpica_constructor", {}).get("error", "Standings will appear once season results are published."))

    x3, x4 = st.columns(2)
    with x3:
        st.markdown("### OpenF1 Meetings (sample)")
        meet_df = snapshot.get("openf1_meetings", {}).get("data", pd.DataFrame())
        if isinstance(meet_df, pd.DataFrame) and not meet_df.empty:
            out = meet_df.copy()
            out["date_start"] = out["date_start"].apply(lambda x: _fmt_user_time(x, "%Y-%m-%d %H:%M"))
            st.dataframe(out.tail(10), hide_index=True, use_container_width=True)
        else:
            st.info(snapshot.get("openf1_meetings", {}).get("error", "No meeting rows returned in this refresh."))
    with x4:
        st.markdown("### OpenF1 Sessions (sample)")
        sess_df = snapshot.get("openf1_sessions", {}).get("data", pd.DataFrame())
        if isinstance(sess_df, pd.DataFrame) and not sess_df.empty:
            out = sess_df.copy()
            out["date_start"] = out["date_start"].apply(lambda x: _fmt_user_time(x, "%Y-%m-%d %H:%M"))
            st.dataframe(out.tail(10), hide_index=True, use_container_width=True)
        else:
            st.info(snapshot.get("openf1_sessions", {}).get("error", "No session rows returned in this refresh."))


def _render_visuals_tab(snapshot: Dict[str, Any], news_snapshot: Dict[str, Any]) -> None:
    cal_df, cal_src = _pick_calendar(snapshot)
    st.caption(f"Calendar visuals source: {cal_src}")
    _plotly_chart(_fig_calendar_timeline(cal_df), use_container_width=True, config=PLOTLY_CONFIG)

    c1, c2 = st.columns(2)
    with c1:
        _plotly_chart(_fig_calendar_months(cal_df), use_container_width=True, config=PLOTLY_CONFIG)
    with c2:
        _plotly_chart(_fig_pu_split(), use_container_width=True, config=PLOTLY_CONFIG)

    _plotly_chart(_fig_reg_changes(), use_container_width=True, config=PLOTLY_CONFIG)

    news_df = news_snapshot.get("news", pd.DataFrame())
    if isinstance(news_df, pd.DataFrame) and not news_df.empty:
        n1, n2 = st.columns(2)
        with n1:
            _plotly_chart(_fig_news_timeline(news_df), use_container_width=True, config=PLOTLY_CONFIG)
        with n2:
            _plotly_chart(_fig_news_topics(news_df), use_container_width=True, config=PLOTLY_CONFIG)


def _render_news_tab(news_snapshot: Dict[str, Any]) -> None:
    st.markdown("### 2026 News Feed")
    st.markdown("Feed memakai Google News RSS query (focus Formula1.com/FIA) + curated fallback resmi supaya tetap ada berita saat feed gagal.")
    st.caption(f"News timestamps are displayed in UTC{_get_user_utc_offset()}.")

    status = news_snapshot.get("status", pd.DataFrame())
    news_df = news_snapshot.get("news", pd.DataFrame())

    c1, c2, c3 = st.columns(3)
    c1.metric("RSS queries", len(status))
    c2.metric("RSS available", int(status["ok"].fillna(False).sum()) if not status.empty else 0)
    c3.metric("Articles", len(news_df) if isinstance(news_df, pd.DataFrame) else 0)

    with st.expander("News feed status"):
        st.dataframe(status, hide_index=True, use_container_width=True)

    if not isinstance(news_df, pd.DataFrame) or news_df.empty:
        st.info("Belum ada berita untuk filter yang dipilih. Coba ubah filter atau refresh feed.")
        return

    source_opts = sorted([x for x in news_df["source"].dropna().astype(str).unique().tolist() if x])
    topic_opts = sorted([x for x in news_df["topic"].dropna().astype(str).unique().tolist() if x])
    selected_sources = st.multiselect("Source filter", source_opts, default=source_opts[: min(len(source_opts), 8)], key="f1_2026_news_sources")
    selected_topics = st.multiselect("Topic filter", topic_opts, default=topic_opts, key="f1_2026_news_topics")
    keyword = st.text_input("Keyword filter", value="", key="f1_2026_news_keyword")

    df = news_df.copy()
    if selected_sources:
        df = df[df["source"].isin(selected_sources)]
    if selected_topics:
        df = df[df["topic"].isin(selected_topics)]
    if keyword.strip():
        df = df[df["headline"].str.contains(keyword, case=False, na=False) | df["summary"].str.contains(keyword, case=False, na=False)]

    _plotly_chart(_fig_news_timeline(df), use_container_width=True, config=PLOTLY_CONFIG)

    show_count = st.slider("Jumlah berita ditampilkan", 5, 40, 12, 1, key="f1_2026_news_count")
    for row in df.head(show_count).to_dict("records"):
        pub = pd.to_datetime(row.get("published_at"), errors="coerce", utc=True)
        pub_s = _fmt_user_time(pub, "%Y-%m-%d %H:%M") if pd.notna(pub) else "Unknown date"
        st.markdown(
            f"""
            <div class="f1-2026-news-card">
              <div class="f1-2026-news-meta">{pub_s} | {row.get('source','')} | {row.get('topic','General')} | {row.get('feed','')}</div>
              <div class="f1-2026-news-headline"><a href="{row.get('url','')}" target="_blank" style="color:#E9EEF7; text-decoration:none;">{row.get('headline','')}</a></div>
              <div class="f1-2026-news-summary">{row.get('summary','')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Raw news table"):
        out = df.copy()
        out["published_at"] = pd.to_datetime(out["published_at"], errors="coerce", utc=True).apply(lambda x: _fmt_user_time(x, "%Y-%m-%d %H:%M"))
        st.dataframe(out[["published_at", "source", "topic", "headline", "url", "feed"]], hide_index=True, use_container_width=True)


def _render_data_lab_tab(snapshot: Dict[str, Any]) -> None:
    st.markdown("### Data Lab: Akses Data Lintas Source (termasuk telemetry)")
    st.markdown(
        "Tab ini mengumpulkan akses data dari source/library yang dipakai app (local CSV, FastF1, Jolpica, OpenF1) "
        "dan menambahkan explorer telemetry interaktif via OpenF1. FastF1 telemetry penuh tetap tersedia di `Race Center > Telemetry`."
    )

    catalog_df = pd.DataFrame(DATA_ACCESS_CATALOG)
    c1, c2 = st.columns(2)
    with c1:
        status_filter = st.multiselect(
            "Catalog status",
            sorted(catalog_df["status"].unique().tolist()),
            default=sorted(catalog_df["status"].unique().tolist()),
            key="f1_2026_catalog_status",
        )
    with c2:
        source_filter = st.multiselect(
            "Catalog source",
            sorted(catalog_df["source"].unique().tolist()),
            default=sorted(catalog_df["source"].unique().tolist()),
            key="f1_2026_catalog_source",
        )
    filtered_catalog = catalog_df[catalog_df["status"].isin(status_filter) & catalog_df["source"].isin(source_filter)].copy()
    st.dataframe(filtered_catalog, hide_index=True, use_container_width=True)
    st.caption(f"Timestamps in Data Lab are shown in UTC{_get_user_utc_offset()} when available.")

    st.divider()
    st.markdown("### OpenF1 Telemetry Explorer (Interactive)")
    sessions_df = snapshot.get("openf1_sessions", {}).get("data", pd.DataFrame())
    if not isinstance(sessions_df, pd.DataFrame) or sessions_df.empty or "session_key" not in sessions_df.columns:
        st.info("OpenF1 session list akan muncul otomatis saat feed sesi tersedia. Coba refresh data.")
    else:
        openf1_sessions = sessions_df.copy()
        openf1_sessions["date_start"] = pd.to_datetime(openf1_sessions["date_start"], errors="coerce", utc=True)
        openf1_sessions["session_key"] = pd.to_numeric(openf1_sessions["session_key"], errors="coerce")
        openf1_sessions = openf1_sessions.dropna(subset=["session_key"]).copy()
        openf1_sessions["session_key"] = openf1_sessions["session_key"].astype(int)
        if openf1_sessions.empty:
            st.info("OpenF1 sessions belum menampilkan `session_key` valid. Coba refresh data beberapa saat lagi.")
        else:
            openf1_sessions["label"] = openf1_sessions.apply(
                lambda r: f"{r.get('meeting_name','')} | {r.get('session_name','')} | {_fmt_user_time(r.get('date_start'), '%Y-%m-%d %H:%M')} | key={r.get('session_key')}",
                axis=1,
            )
            selected_label = st.selectbox("Select OpenF1 session", openf1_sessions["label"].tolist(), key="f1_2026_openf1_session_label")
            selected_row = openf1_sessions.loc[openf1_sessions["label"] == selected_label].iloc[0]
            selected_session_key = int(selected_row["session_key"])

            with st.spinner("Loading OpenF1 session bundle (drivers/laps/weather)..."):
                lite_bundle = load_openf1_session_bundle(selected_session_key, None, st.session_state.get("f1_2026_refresh_nonce", 0))

            drivers_df = lite_bundle.get("drivers", pd.DataFrame())
            laps_df = lite_bundle.get("laps", pd.DataFrame())
            weather_df = lite_bundle.get("weather", pd.DataFrame())

            dcol1, dcol2, dcol3 = st.columns(3)
            dcol1.metric("Drivers rows", len(drivers_df) if isinstance(drivers_df, pd.DataFrame) else 0)
            dcol2.metric("Laps rows", len(laps_df) if isinstance(laps_df, pd.DataFrame) else 0)
            dcol3.metric("Weather rows", len(weather_df) if isinstance(weather_df, pd.DataFrame) else 0)

            if isinstance(drivers_df, pd.DataFrame) and not drivers_df.empty:
                display_cols = [c for c in ["driver_number", "full_name", "name_acronym", "team_name", "broadcast_name"] if c in drivers_df.columns]
                st.dataframe(drivers_df[display_cols] if display_cols else drivers_df, hide_index=True, use_container_width=True)
                if "driver_number" in drivers_df.columns:
                    driver_options = (
                        drivers_df.assign(driver_number_num=pd.to_numeric(drivers_df["driver_number"], errors="coerce"))
                        .dropna(subset=["driver_number_num"])
                        .assign(driver_number_num=lambda x: x["driver_number_num"].astype(int))
                    )
                    driver_options["driver_label"] = driver_options.apply(
                        lambda r: f"#{r['driver_number_num']} - {r.get('full_name') or r.get('broadcast_name') or r.get('name_acronym') or 'Unknown'} ({r.get('team_name','')})",
                        axis=1,
                    )
                    if not driver_options.empty:
                        driver_label = st.selectbox("Select driver for telemetry", driver_options["driver_label"].tolist(), key="f1_2026_openf1_driver_label")
                        driver_row = driver_options.loc[driver_options["driver_label"] == driver_label].iloc[0]
                        driver_number = int(driver_row["driver_number_num"])

                        if st.button("Load Telemetry Sample", key="f1_2026_openf1_load_telemetry"):
                            with st.spinner("Loading OpenF1 telemetry (car_data + position) ..."):
                                full_bundle = load_openf1_session_bundle(selected_session_key, driver_number, st.session_state.get("f1_2026_refresh_nonce", 0))

                            car_df = full_bundle.get("car_data", pd.DataFrame())
                            pos_df = full_bundle.get("position", pd.DataFrame())

                            t1, t2 = st.columns(2)
                            with t1:
                                _plotly_chart(_fig_openf1_car_data(car_df), use_container_width=True, config=PLOTLY_CONFIG)
                            with t2:
                                _plotly_chart(_fig_openf1_position(pos_df), use_container_width=True, config=PLOTLY_CONFIG)

                            t3, t4 = st.columns(2)
                            with t3:
                                _plotly_chart(_fig_openf1_laps(laps_df[laps_df["driver_number"].astype(str) == str(driver_number)] if isinstance(laps_df, pd.DataFrame) and "driver_number" in laps_df.columns else laps_df), use_container_width=True, config=PLOTLY_CONFIG)
                            with t4:
                                _plotly_chart(_fig_openf1_weather(weather_df), use_container_width=True, config=PLOTLY_CONFIG)

                            with st.expander("OpenF1 raw telemetry tables"):
                                if isinstance(car_df, pd.DataFrame) and not car_df.empty:
                                    cshow = car_df.copy()
                                    if "date" in cshow.columns:
                                        cshow["date"] = pd.to_datetime(cshow["date"], errors="coerce", utc=True).apply(lambda x: _fmt_user_time(x, "%Y-%m-%d %H:%M:%S"))
                                    st.markdown("**car_data**")
                                    st.dataframe(cshow.head(200), hide_index=True, use_container_width=True)
                                if isinstance(pos_df, pd.DataFrame) and not pos_df.empty:
                                    pshow = pos_df.copy()
                                    if "date" in pshow.columns:
                                        pshow["date"] = pd.to_datetime(pshow["date"], errors="coerce", utc=True).apply(lambda x: _fmt_user_time(x, "%Y-%m-%d %H:%M:%S"))
                                    st.markdown("**position**")
                                    st.dataframe(pshow.head(200), hide_index=True, use_container_width=True)
                else:
                    st.info("Kolom `driver_number` tidak ditemukan di response OpenF1 drivers.")
            else:
                st.info("Driver list untuk session ini akan muncul saat feed OpenF1 mengirimkan roster sesi.")

            l1, l2 = st.columns(2)
            with l1:
                _plotly_chart(_fig_openf1_laps(laps_df), use_container_width=True, config=PLOTLY_CONFIG)
            with l2:
                _plotly_chart(_fig_openf1_weather(weather_df), use_container_width=True, config=PLOTLY_CONFIG)

    st.divider()
    st.markdown("### FastF1 Session Access (Schedule / Session Discovery)")
    st.caption(
        "FastF1 sudah dipakai penuh di app untuk telemetry/replay/live. Data Lab ini menambahkan preview schedule agar navigasi 2025/2026 lebih mudah; "
        "telemetry FastF1 tetap paling lengkap di tab `Race Center > Telemetry` dan `Race Replay`."
    )
    fcol1, fcol2 = st.columns([1, 3])
    with fcol1:
        fastf1_year = st.selectbox("FastF1 year", [2026, 2025], key="f1_2026_fastf1_year")
    with fcol2:
        st.caption("Pilih tahun untuk preview event schedule dari FastF1 (membantu transisi 2025 -> 2026).")
    fastf1_sched = load_fastf1_schedule_preview(int(fastf1_year), st.session_state.get("f1_2026_refresh_nonce", 0))
    if isinstance(fastf1_sched, pd.DataFrame) and not fastf1_sched.empty:
        preview_cols = [c for c in ["RoundNumber", "EventName", "Country", "Location", "EventDate", "Session1", "Session2", "Session3", "Session4", "Session5"] if c in fastf1_sched.columns]
        display = fastf1_sched[preview_cols].copy()
        if "EventDate" in display.columns:
            display["EventDate"] = pd.to_datetime(display["EventDate"], errors="coerce", utc=True).apply(lambda x: _fmt_user_time(x, "%Y-%m-%d %H:%M"))
        st.dataframe(display, hide_index=True, use_container_width=True)
    else:
        st.info("FastF1 schedule preview tidak tersedia untuk tahun ini (atau koneksi gagal).")


def _render_library_coverage_tab() -> None:
    st.markdown("### First-Principles: Library + Data Source F1 yang dipakai dan capability-nya")
    st.markdown(
        "Tabel ini adalah inventory capability dari semua library dan data source yang dipakai app untuk F1. "
        "Tujuannya supaya integrasi fitur 2026 jelas, terukur, dan tidak ada area data yang terlewat."
    )

    coverage_df = pd.DataFrame(F1_LIBRARY_COVERAGE)
    if coverage_df.empty:
        st.info("Coverage matrix sedang disiapkan.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Libraries/Sources", coverage_df["library"].nunique())
    c2.metric("Capability rows", len(coverage_df))
    c3.metric("Interactive = Yes", int((coverage_df["interactive"] == "Yes").sum()))
    c4.metric("Integrated", int((coverage_df["coverage_status"] == "Integrated").sum()))

    lib_filter = st.multiselect(
        "Filter library/source",
        sorted(coverage_df["library"].unique().tolist()),
        default=sorted(coverage_df["library"].unique().tolist()),
        key="f1_2026_cov_library_filter",
    )
    cat_filter = st.multiselect(
        "Filter category",
        sorted(coverage_df["category"].unique().tolist()),
        default=sorted(coverage_df["category"].unique().tolist()),
        key="f1_2026_cov_category_filter",
    )

    filtered = coverage_df[coverage_df["library"].isin(lib_filter) & coverage_df["category"].isin(cat_filter)].copy()
    st.dataframe(filtered, hide_index=True, use_container_width=True)

    g = filtered.groupby(["library", "category"], dropna=False).size().reset_index(name="count")
    fig = px.bar(g, x="library", y="count", color="category", barmode="stack", text="count", title="Capability Coverage by Library")
    fig.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=55, b=20), xaxis_title="", yaxis_title="Capability rows")
    _plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("### Ringkasan implementasi saat ini")
    st.markdown("- `FastF1` dipakai untuk schedule, session load, telemetry, replay, live timing, plot integrations.")
    st.markdown("- `Jolpica` dipakai untuk schedule + driver/constructor standings ringan (fallback-friendly).")
    st.markdown("- `OpenF1` dipakai untuk meetings/sessions discovery dan Data Lab telemetry explorer (`drivers/laps/weather/car_data/position`).")
    st.markdown("- `Streamlit + Plotly` dipakai untuk interaktivitas, caching, dan visualisasi performa tinggi.")


def _render_diagnostics_tab(snapshot: Dict[str, Any], news_snapshot: Dict[str, Any]) -> None:
    st.markdown("### Diagnostics / Test Results (Professional Summary)")
    st.markdown(
        "Panel ini menampilkan hasil health check dan readiness checks untuk data services, kalender, news feed, Data Lab, dan local data. "
        "Bukan unit test `pytest`, tetapi operational diagnostics yang relevan untuk performa dan reliability di runtime."
    )

    diag_df = run_2026_diagnostics(snapshot, news_snapshot)
    if diag_df.empty:
        st.warning("No diagnostics generated.")
        return

    fail_count = int((diag_df["status"] == "FAIL").sum())
    warn_count = int((diag_df["status"] == "WARN").sum())
    pass_count = int((diag_df["status"] == "PASS").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Checks", len(diag_df))
    c2.metric("PASS", pass_count)
    c3.metric("WARN", warn_count)
    c4.metric("FAIL", fail_count)

    _plotly_chart(_fig_diagnostics_summary(diag_df), use_container_width=True, config=PLOTLY_CONFIG)

    severity_filter = st.multiselect(
        "Severity filter",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
        key="f1_2026_diag_severity_filter",
    )
    status_filter = st.multiselect(
        "Status filter",
        ["FAIL", "WARN", "PASS"],
        default=["FAIL", "WARN", "PASS"],
        key="f1_2026_diag_status_filter",
    )
    filtered = diag_df[diag_df["severity"].isin(severity_filter) & diag_df["status"].astype(str).isin(status_filter)].copy()

    st.dataframe(filtered, hide_index=True, use_container_width=True)

    with st.expander("Test methodology"):
        st.markdown("- Data feed availability/latency checks diambil dari snapshot fetch yang sama dengan UI.")
        st.markdown("- Calendar integrity memeriksa jumlah round, sprint tagging, dan urutan tanggal.")
        st.markdown("- News checks menguji feed live dan fallback merge.")
        st.markdown("- Data Lab checks menguji ketersediaan OpenF1 session keys (syarat telemetry explorer).")
        st.markdown("- Local file checks memeriksa keberadaan CSV 2025/2026 tanpa melakukan load berat tambahan.")


def _render_ml_pipeline_tab() -> None:
    st.markdown("### ML Pipeline (2026-ready, pre-season enriched)")
    st.markdown(
        "Panel ini menggabungkan hasil testing **Bahrain** (official timed tests) dan **Spain/Barcelona shakedown** (private, no official timings) "
        "ke pipeline ML sebagai fitur prior 2026. Evaluasi model ditampilkan dengan format profesional (metrics + diagnostics)."
    )
    if "f1_2026_ml_refresh_nonce" not in st.session_state:
        st.session_state["f1_2026_ml_refresh_nonce"] = 0
    if "f1_2026_ml_persist_result" not in st.session_state:
        st.session_state["f1_2026_ml_persist_result"] = None
    if "f1_2026_ml_auto_retrain_last_sig" not in st.session_state:
        st.session_state["f1_2026_ml_auto_retrain_last_sig"] = None

    r1, r2, r3 = st.columns([1, 1.3, 3])
    with r1:
        if st.button("Refresh ML", key="f1_2026_refresh_ml"):
            st.session_state["f1_2026_ml_refresh_nonce"] += 1
            st.rerun()
    with r2:
        if st.button("Retrain & Persist 2026 Model", key="f1_2026_retrain_persist_model", type="primary"):
            with st.spinner("Training and persisting ML model artifacts..."):
                st.session_state["f1_2026_ml_persist_result"] = run_ml_pipeline_persist_training()
            st.session_state["f1_2026_ml_refresh_nonce"] += 1
            st.rerun()
    with r3:
        st.caption(
            "ML diagnostics menggunakan cache dan berjalan lokal (`persist=False`) agar cepat dan tidak otomatis overwrite model produksi. "
            "Timestamps ditampilkan dalam UTC user saat tersedia."
        )

    g1, g2, g3, g4 = st.columns([1.1, 1.4, 1.1, 3])
    with g1:
        auto_retrain_enabled = st.checkbox(
            "Auto-retrain",
            value=False,
            key="f1_2026_ml_auto_retrain_enabled",
            help="Auto persist model jika akurasi (metric terpilih) turun di bawah threshold.",
        )
    acc_metric_labels = ["Accuracy ±2 pos", "Accuracy ±1 pos", "Exact accuracy (rounded pos)"]
    acc_metric_map = {
        "Accuracy ±2 pos": "ACC_TOL2_PCT",
        "Accuracy ±1 pos": "ACC_TOL1_PCT",
        "Exact accuracy (rounded pos)": "ACC_EXACT_PCT",
    }
    with g2:
        auto_metric_label = st.selectbox(
            "Trigger metric",
            acc_metric_labels,
            index=0,
            key="f1_2026_ml_auto_retrain_metric",
        )
    with g3:
        auto_threshold_pct = st.slider(
            "Min accuracy %",
            min_value=50,
            max_value=100,
            value=85,
            step=1,
            key="f1_2026_ml_auto_retrain_threshold",
        )
    with g4:
        st.caption(
            "Auto-retrain menggunakan race CSV lokal terbaru (2025/2026) yang sudah ada di folder `data/`. "
            "Begitu file hasil race di-update, fingerprint data berubah dan guard bisa trigger lagi."
        )

    persist_result = st.session_state.get("f1_2026_ml_persist_result")
    if isinstance(persist_result, dict):
        if persist_result.get("ok"):
            pr_metrics = persist_result.get("metrics", {}) or {}
            st.success(
                f"Model persisted successfully | MAE={pr_metrics.get('MAE','N/A')} | RMSE={pr_metrics.get('RMSE','N/A')} | "
                f"R²={pr_metrics.get('R2','N/A')} | Features={persist_result.get('feature_count','N/A')}"
            )
            art_df = persist_result.get("artifacts", pd.DataFrame())
            if isinstance(art_df, pd.DataFrame) and not art_df.empty:
                art_show = art_df.copy()
                if "modified_at_utc" in art_show.columns:
                    art_show["modified_at_utc"] = pd.to_datetime(art_show["modified_at_utc"], errors="coerce", utc=True).apply(
                        lambda x: _fmt_user_time(x, "%Y-%m-%d %H:%M:%S") if pd.notna(x) else ""
                    )
                st.dataframe(art_show, hide_index=True, use_container_width=True)
        else:
            st.error(f"Persist training failed: {persist_result.get('error', 'Unknown error')}")

    preseason_snapshot = load_preseason_testing_snapshot()
    summary_df = preseason_snapshot.get("summary", pd.DataFrame())
    features_df = preseason_snapshot.get("features", pd.DataFrame())

    st.markdown("#### Pre-season Testing Source Data (Bahrain + Spain)")
    if not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
        st.warning("Pre-season testing source CSV belum terisi atau masih kosong.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Source rows", len(summary_df))
        c2.metric("Teams", int(summary_df["team"].nunique()) if "team" in summary_df.columns else 0)
        c3.metric("Tests tracked", int(summary_df["test_label"].nunique()) if "test_label" in summary_df.columns else 0)
        c4.metric("Venues", int(summary_df["venue"].nunique()) if "venue" in summary_df.columns else 0)

        v1, v2 = st.columns(2)
        with v1:
            _plotly_chart(_fig_preseason_bahrain_laps(summary_df), use_container_width=True, config=PLOTLY_CONFIG)
        with v2:
            _plotly_chart(_fig_preseason_bahrain_pace(summary_df), use_container_width=True, config=PLOTLY_CONFIG)

        _plotly_chart(_fig_spain_shakedown_participation(summary_df), use_container_width=True, config=PLOTLY_CONFIG)

        bahrain_df = summary_df[summary_df["venue"].astype(str).str.contains("bahrain", case=False, na=False)].copy()
        spain_df = summary_df[summary_df["venue"].astype(str).str.contains("barcelona", case=False, na=False)].copy()
        for frame in (bahrain_df, spain_df):
            for dt_col in ("start_date", "end_date"):
                if dt_col in frame.columns:
                    frame[dt_col] = pd.to_datetime(frame[dt_col], errors="coerce", utc=True).apply(lambda x: _fmt_user_time(x, "%Y-%m-%d %H:%M"))

        b1, b2 = st.columns(2)
        with b1:
            st.markdown("##### Bahrain Official Test Results (Team Summary)")
            cols = [c for c in ["test_label", "team", "total_laps", "fastest_driver", "fastest_lap_time", "start_date", "end_date"] if c in bahrain_df.columns]
            st.dataframe(bahrain_df[cols] if cols else bahrain_df, hide_index=True, use_container_width=True)
        with b2:
            st.markdown("##### Spain / Barcelona Private Shakedown (Official Metadata)")
            cols = [c for c in ["test_label", "team", "participated", "late_start", "official_timing_available", "source_note"] if c in spain_df.columns]
            st.dataframe(spain_df[cols] if cols else spain_df, hide_index=True, use_container_width=True)
            st.caption("Spain shakedown bersifat private. Formula1.com melaporkan partisipasi/status tim, tetapi tidak mempublikasikan timing resmi.")

    st.markdown("#### Pre-season Feature Engineering Output (for ML)")
    if isinstance(features_df, pd.DataFrame) and not features_df.empty:
        f1, f2, f3 = st.columns(3)
        f1.metric("Teams with feature rows", len(features_df))
        f2.metric("Numeric feature cols", int(sum(1 for c in features_df.columns if str(c).startswith("preseason_"))))
        f3.metric("Bahrain pace rows", int(pd.to_numeric(features_df.get("preseason_bahrain_best_lap_sec"), errors="coerce").notna().sum()) if "preseason_bahrain_best_lap_sec" in features_df.columns else 0)

        show_cols = [c for c in features_df.columns if c == "team_norm" or str(c).startswith("preseason_")]
        st.dataframe(features_df[show_cols], hide_index=True, use_container_width=True)
    else:
        st.info("Aggregated pre-season feature table akan muncul setelah source pre-season ter-load.")

    st.divider()
    st.markdown("#### Model Training & Evaluation Snapshot (Local)")
    st.caption(
        "Menggunakan race results CSV lokal + fitur pre-season testing 2026. Tujuan panel ini adalah validasi pipeline dan transparansi kualitas sebelum season berjalan penuh."
    )
    with st.spinner("Running ML pipeline diagnostics..."):
        ml_snapshot = run_ml_pipeline_snapshot(st.session_state["f1_2026_ml_refresh_nonce"])

    if not ml_snapshot.get("ok"):
        st.error(ml_snapshot.get("error", "ML diagnostics failed"))
        return

    metrics = ml_snapshot.get("metrics", {}) or {}
    results_df = ml_snapshot.get("results", pd.DataFrame())
    importances_df = ml_snapshot.get("feature_importances", pd.DataFrame())
    coverage_df = ml_snapshot.get("feature_coverage", pd.DataFrame())
    dataset_summary = ml_snapshot.get("dataset_summary", pd.DataFrame())
    race_file_status = ml_snapshot.get("race_file_status", pd.DataFrame())

    results_eval = results_df.copy() if isinstance(results_df, pd.DataFrame) else pd.DataFrame()
    if isinstance(results_eval, pd.DataFrame) and not results_eval.empty:
        results_eval["Actual_Rounded"] = pd.to_numeric(results_eval.get("Actual"), errors="coerce").round()
        results_eval["Predicted_Rounded"] = pd.to_numeric(results_eval.get("Predicted"), errors="coerce").round()
        results_eval["Abs_Pos_Error_Rounded"] = (
            pd.to_numeric(results_eval["Actual_Rounded"], errors="coerce")
            - pd.to_numeric(results_eval["Predicted_Rounded"], errors="coerce")
        ).abs()

    selected_acc_key = acc_metric_map.get(auto_metric_label, "ACC_TOL2_PCT")
    selected_acc_val = pd.to_numeric(pd.Series([metrics.get(selected_acc_key)]), errors="coerce").iloc[0]
    selected_acc_label = auto_metric_label
    train_data_signature = str(ml_snapshot.get("train_data_signature", "no_sig"))
    auto_guard_sig = f"{train_data_signature}|{selected_acc_key}|{int(auto_threshold_pct)}"

    if auto_retrain_enabled and pd.notna(selected_acc_val):
        if float(selected_acc_val) < float(auto_threshold_pct):
            if st.session_state.get("f1_2026_ml_auto_retrain_last_sig") != auto_guard_sig:
                with st.spinner(
                    f"Auto-retrain triggered: {selected_acc_label}={float(selected_acc_val):.1f}% < {auto_threshold_pct}% (persisting model)..."
                ):
                    st.session_state["f1_2026_ml_persist_result"] = run_ml_pipeline_persist_training()
                st.session_state["f1_2026_ml_auto_retrain_last_sig"] = auto_guard_sig
                st.session_state["f1_2026_ml_refresh_nonce"] += 1
                st.rerun()
            else:
                st.warning(
                    f"Auto-retrain guard aktif, tapi sudah pernah trigger untuk dataset/threshold ini "
                    f"({selected_acc_label}={float(selected_acc_val):.1f}% < {auto_threshold_pct}%)."
                )
        else:
            st.success(
                f"Auto-retrain guard aktif: {selected_acc_label}={float(selected_acc_val):.1f}% "
                f"(threshold {auto_threshold_pct}%) -> no retrain."
            )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{metrics.get('MAE', 'N/A')}")
    m2.metric("RMSE", f"{metrics.get('RMSE', 'N/A')}")
    m3.metric("R²", f"{metrics.get('R2', 'N/A')}")
    m4.metric("Eval rows", len(results_df) if isinstance(results_df, pd.DataFrame) else 0)

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Exact Acc (round)", f"{metrics.get('ACC_EXACT_PCT', 'N/A')}%")
    a2.metric("Acc +/-1 pos", f"{metrics.get('ACC_TOL1_PCT', 'N/A')}%")
    a3.metric("Acc +/-2 pos", f"{metrics.get('ACC_TOL2_PCT', 'N/A')}%")
    a4.metric("Median |err| (round)", f"{metrics.get('MEDIAN_ABS_POS_ERR_ROUND', 'N/A')}")

    d1, d2 = st.columns([2, 3])
    with d1:
        st.markdown("##### Dataset Summary")
        st.dataframe(dataset_summary, hide_index=True, use_container_width=True)
        if isinstance(race_file_status, pd.DataFrame) and not race_file_status.empty:
            files_show = race_file_status.copy()
            if "modified_at_utc" in files_show.columns:
                files_show["modified_at_utc"] = pd.to_datetime(files_show["modified_at_utc"], errors="coerce", utc=True).apply(
                    lambda x: _fmt_user_time(x, "%Y-%m-%d %H:%M:%S") if pd.notna(x) else ""
                )
            st.markdown("##### Race CSV Freshness (Local)")
            st.dataframe(files_show, hide_index=True, use_container_width=True)
        if isinstance(coverage_df, pd.DataFrame) and not coverage_df.empty:
            st.markdown("##### Pre-season Feature Coverage in X_test")
            cov_plot = coverage_df.copy()
            cov_plot["pct_nonzero"] = pd.to_numeric(cov_plot["nonzero_ratio"], errors="coerce").fillna(0) * 100
            fig_cov = px.bar(
                cov_plot.sort_values("pct_nonzero", ascending=False),
                x="pct_nonzero",
                y="feature",
                orientation="h",
                text="pct_nonzero",
                title="Pre-season Feature Coverage (Non-zero in Evaluation Split)",
                color="pct_nonzero",
                color_continuous_scale="Tealgrn",
            )
            fig_cov.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=55, b=20), xaxis_title="% non-zero rows", yaxis_title="")
            _plotly_chart(fig_cov, use_container_width=True, config=PLOTLY_CONFIG)
    with d2:
        if isinstance(importances_df, pd.DataFrame) and not importances_df.empty:
            top_imp = importances_df.head(15).copy()
            fig_imp = px.bar(
                top_imp.sort_values("importance", ascending=True),
                x="importance",
                y="feature",
                orientation="h",
                text="importance",
                title="Top Model Features (Importance)",
                color="importance",
                color_continuous_scale="Turbo",
            )
            fig_imp.update_layout(template="plotly_dark", height=520, margin=dict(l=20, r=20, t=55, b=20), xaxis_title="Importance", yaxis_title="")
            _plotly_chart(fig_imp, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("Feature importances tidak tersedia pada snapshot ini.")

    if isinstance(results_df, pd.DataFrame) and not results_df.empty:
        p1, p2 = st.columns(2)
        with p1:
            fig_scatter = px.scatter(
                results_df,
                x="Actual",
                y="Predicted",
                color="Abs_Error" if "Abs_Error" in results_df.columns else None,
                title="Actual vs Predicted Positions",
                trendline=None,
            )
            fig_scatter.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=55, b=20))
            _plotly_chart(fig_scatter, use_container_width=True, config=PLOTLY_CONFIG)
        with p2:
            fig_err = px.histogram(
                results_df,
                x="Error",
                nbins=20,
                title="Prediction Error Distribution (Actual - Predicted)",
                color_discrete_sequence=["#E10600"],
            )
            fig_err.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=55, b=20), xaxis_title="Error", yaxis_title="Count")
            _plotly_chart(fig_err, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("##### Engineering Diagnostics (Accuracy / Residual Behavior)")
        e1, e2 = st.columns(2)
        with e1:
            tol_df = pd.DataFrame(
                [
                    {"metric": "Exact", "accuracy_pct": pd.to_numeric(metrics.get("ACC_EXACT_PCT"), errors="coerce")},
                    {"metric": "+/-1 pos", "accuracy_pct": pd.to_numeric(metrics.get("ACC_TOL1_PCT"), errors="coerce")},
                    {"metric": "+/-2 pos", "accuracy_pct": pd.to_numeric(metrics.get("ACC_TOL2_PCT"), errors="coerce")},
                    {"metric": "+/-3 pos", "accuracy_pct": pd.to_numeric(metrics.get("ACC_TOL3_PCT"), errors="coerce")},
                ]
            )
            fig_tol = px.bar(
                tol_df,
                x="metric",
                y="accuracy_pct",
                text="accuracy_pct",
                title="Tolerance Accuracy Ladder (Rounded Positions)",
                color="accuracy_pct",
                color_continuous_scale="Viridis",
            )
            fig_tol.update_layout(
                template="plotly_dark",
                height=340,
                margin=dict(l=20, r=20, t=55, b=20),
                xaxis_title="",
                yaxis_title="Accuracy (%)",
                yaxis_range=[0, 100],
            )
            fig_tol.add_hline(y=float(auto_threshold_pct), line_dash="dash", line_color="#F5B942")
            _plotly_chart(fig_tol, use_container_width=True, config=PLOTLY_CONFIG)
        with e2:
            if "Abs_Pos_Error_Rounded" in results_eval.columns:
                cdf_df = results_eval[["Abs_Pos_Error_Rounded"]].copy()
                cdf_df["Abs_Pos_Error_Rounded"] = pd.to_numeric(cdf_df["Abs_Pos_Error_Rounded"], errors="coerce")
                cdf_df = cdf_df.dropna().sort_values("Abs_Pos_Error_Rounded").reset_index(drop=True)
                if not cdf_df.empty:
                    cdf_df["sample_pct"] = ((cdf_df.index + 1) / len(cdf_df)) * 100.0
                    fig_cdf = px.line(
                        cdf_df,
                        x="Abs_Pos_Error_Rounded",
                        y="sample_pct",
                        markers=True,
                        title="Cumulative Distribution of Rounded Absolute Error",
                    )
                    fig_cdf.update_layout(
                        template="plotly_dark",
                        height=340,
                        margin=dict(l=20, r=20, t=55, b=20),
                        xaxis_title="Rounded absolute position error",
                        yaxis_title="% evaluation rows",
                        yaxis_range=[0, 100],
                    )
                    _plotly_chart(fig_cdf, use_container_width=True, config=PLOTLY_CONFIG)

        if {"Actual_Rounded", "Error"}.issubset(results_eval.columns):
            resid_df = results_eval[["Actual_Rounded", "Error"]].copy()
            resid_df["Actual_Rounded"] = pd.to_numeric(resid_df["Actual_Rounded"], errors="coerce")
            resid_df["Error"] = pd.to_numeric(resid_df["Error"], errors="coerce")
            resid_df = resid_df.dropna()
            if not resid_df.empty:
                resid_df["Actual_Rounded"] = resid_df["Actual_Rounded"].astype(int).astype(str)
                fig_resid = px.box(
                    resid_df,
                    x="Actual_Rounded",
                    y="Error",
                    points="outliers",
                    title="Residual Spread by Actual Finishing Position",
                    color_discrete_sequence=["#00D2BE"],
                )
                fig_resid.update_layout(
                    template="plotly_dark",
                    height=360,
                    margin=dict(l=20, r=20, t=55, b=20),
                    xaxis_title="Actual position (rounded)",
                    yaxis_title="Residual (Actual - Predicted)",
                )
                fig_resid.add_hline(y=0, line_dash="dash", line_color="#FFFFFF")
                _plotly_chart(fig_resid, use_container_width=True, config=PLOTLY_CONFIG)

        with st.expander("Raw ML evaluation tables"):
            if isinstance(importances_df, pd.DataFrame) and not importances_df.empty:
                st.markdown("**Feature importances**")
                st.dataframe(importances_df, hide_index=True, use_container_width=True)
            st.markdown("**Evaluation rows**")
            st.dataframe(results_eval.head(200), hide_index=True, use_container_width=True)

    with st.expander("ML pipeline notes"):
        st.markdown("- Training model: `HistGradientBoostingRegressor` (existing project model), evaluated on local train/test split.")
        st.markdown("- Feature set now supports optional `preseason_*` features derived from curated 2026 testing dataset.")
        st.markdown("- Bahrain features use official timed tests (Feb 11-13 and Feb 18-20, 2026).")
        st.markdown("- Spain/Barcelona shakedown is private; official metadata is used for participation/availability flags, not timing.")


def _render_sources_tab() -> None:
    st.markdown("### Official references used in this hub")
    refs = pd.DataFrame(OFFICIAL_2026_UPDATES)[["date", "source", "title", "url"]].sort_values("date", ascending=False)
    st.dataframe(refs, hide_index=True, use_container_width=True)
    st.markdown("- FIA regulations portal (2026 docs): https://www.fia.com/regulation/category/110")
    st.markdown("- Formula1.com 2026 season hub: https://www.formula1.com/en/racing/2026")
    st.markdown("- Formula1.com 2026 regulations explainer: https://www.formula1.com/en/latest/article/your-guide-to-the-all-new-2026-rules-package-in-f1.37sUYGAYovZZWwOKwK32t6")
    st.markdown("- Formula1.com/FIA 2026 calendar announcement (2025-06-10): https://www.formula1.com/en/latest/article.fia-and-formula-1-unveil-calendar-for-2026-season-as-madrid-makes-its-debut.4lEs5Z6ow8NLTHF1nEr4oq")
    st.markdown("- Corporate Formula 1 sprint calendar (2025-09-16): https://corp.formula1.com/formula-1-confirms-calendar-for-six-sprint-events-across-2026-season/")
    st.markdown("- FIA WMSC calendar + testing update (2025-09-30): https://www.fia.com/news/fia-world-motor-sport-council-updates-2026-formula-one-world-championship-calendar")
    st.markdown("- FIA sporting calendars approval (2025-10-12): https://www.fia.com/news/fia-world-motor-sport-council-approves-2026-fia-sporting-calendars")
    st.markdown("- FIA Cadillac approval (2025-03-07): https://www.fia.com/news/fia-and-formula-1-can-confirm-cadillac-formula-1-team-has-met-their-requirements-join-2026")
    st.markdown("- FIA Concorde Governance Agreement (2025-12-12): https://www.fia.com/news/fia-formula-one-and-all-11-teams-sign-2026-concorde-governance-agreement")
    st.markdown("- Formula1.com (Bahrain Test 1 team summary, IN NUMBERS): https://www.formula1.com/en/latest/article/in-numbers-who-was-the-fastest-and-who-completed-the-most-laps-at-the-first.1BhUw1jriKD4k6di4cZd72")
    st.markdown("- Formula1.com (Bahrain Test 2 team summary, IN NUMBERS): https://www.formula1.com/en/latest/article/in-numbers-who-was-the-fastest-and-who-recorded-the-most-laps-at-the-second.7x0s5SNQ78lh04oUqtYpX8")
    st.markdown("- Formula1.com (Barcelona shakedown learnings): https://www.formula1.com/en/latest/article/what-we-learned-from-the-f1-2026-barcelona-shakedown.4fXULvjc2tPjDqIZgNj6kZ")
    st.markdown("- Formula1.com (Williams absent from Barcelona shakedown): https://www.formula1.com/en/latest/article/heres-why-williams-didnt-take-part-in-barcelona-shakedown.6KjRV6JY3tAqW7dWcHDpqR")


def render_f1_2026_updates_tab() -> None:
    _inject_2026_hub_styles()
    st.header("F1 2026 Update Hub")
    st.info(
        "Hub ini menambahkan riset resmi, berita, readiness matrix semua fitur, monitoring data services, dan visualisasi 2026. "
        "Tab analytics lain di app tetap fokus ke data musim 2025."
    )

    if "f1_2026_refresh_nonce" not in st.session_state:
        st.session_state["f1_2026_refresh_nonce"] = 0
    if "f1_2026_news_refresh_nonce" not in st.session_state:
        st.session_state["f1_2026_news_refresh_nonce"] = 0

    r1, r2, r3 = st.columns([1, 1, 3])
    with r1:
        if st.button("Refresh Data", key="f1_2026_refresh_apis"):
            st.session_state["f1_2026_refresh_nonce"] += 1
            st.rerun()
    with r2:
        if st.button("Refresh News", key="f1_2026_refresh_news"):
            st.session_state["f1_2026_news_refresh_nonce"] += 1
            st.rerun()
    with r3:
        st.caption(
            "Sources: FastF1, Jolpica, OpenF1, Google News RSS (query-based) + curated official fallback (FIA/Formula1.com). "
            "Jika feed data gagal, hub tetap berjalan dengan fallback."
        )

    with st.spinner("Loading 2026 data feeds..."):
        snapshot = load_2026_api_snapshot(st.session_state["f1_2026_refresh_nonce"])
    with st.spinner("Loading 2026 news feed..."):
        news_snapshot = load_2026_news_snapshot(st.session_state["f1_2026_news_refresh_nonce"])

    _render_hero(snapshot, news_snapshot)
    view_mode = st.radio(
        "2026 Hub View",
        ["Analyst", "Engineer"],
        horizontal=True,
        index=0,
        key="f1_2026_hub_view_mode",
    )
    if view_mode == "Analyst":
        st.caption("Analyst view menyembunyikan tab developer-heavy (Library Coverage, Diagnostics, ML Pipeline, Data Lab).")
    else:
        st.caption("Engineer view menampilkan semua tab termasuk diagnostics, ML pipeline, dan data lab.")

    tab_specs: List[Tuple[str, Any, Tuple[Any, ...]]] = [
        ("Overview", _render_overview_tab, (snapshot, news_snapshot)),
        ("Research", _render_research_tab, ()),
        ("Library Coverage", _render_library_coverage_tab, ()),
        ("Feature Readiness", _render_feature_readiness_tab, ()),
        ("Data Services", _render_api_monitor_tab, (snapshot,)),
        ("Diagnostics", _render_diagnostics_tab, (snapshot, news_snapshot)),
        ("ML Pipeline", _render_ml_pipeline_tab, ()),
        ("Data Lab", _render_data_lab_tab, (snapshot,)),
        ("Visuals", _render_visuals_tab, (snapshot, news_snapshot)),
        ("News", _render_news_tab, (news_snapshot,)),
        ("Sources", _render_sources_tab, ()),
    ]
    if view_mode == "Analyst":
        analyst_order = ["Overview", "Visuals", "News", "Research", "Feature Readiness", "Data Services", "Sources"]
        tab_specs = [spec for label in analyst_order for spec in tab_specs if spec[0] == label]

    tabs = st.tabs([label for label, _, _ in tab_specs])
    for tab, (_, renderer, args) in zip(tabs, tab_specs):
        with tab:
            renderer(*args)
