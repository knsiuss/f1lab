# -*- coding: utf-8 -*-
"""
app.py
~~~~~~
Streamlit dashboard entry point.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fastf1
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import joblib
import warnings
import time

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules
from config import TEAM_COLORS, DRIVER_PROFILES, F1_2025_COMPLETED_RACES, F1_2025_CALENDAR, F1_2025_RACE_NAMES, STREAMLIT_CONFIG, SOCIAL_MEDIA_CONFIG
from loader import load_data as load_csv_data, clean_data, load_combined_data
from analysis import calculate_driver_stats, calculate_team_stats, calculate_combined_constructor_standings, calculate_teammate_comparison
from config import DATA_FILES
from fastf1_extended import (
    get_session_info, get_weather_summary, get_tyre_stints,
    get_pit_stops, get_sector_times, get_speed_data,
    get_track_status, get_car_data, get_position_data,
    get_position_changes, get_flag_events,
    get_race_results, get_session_schedule, get_gaps_to_leader,
    get_tyre_degradation, get_best_sectors, get_top_speeds,
    export_session_to_csv, get_circuit_layout_info,
    get_race_control_messages, get_detailed_pit_analysis
)
from model import load_trained_model, RaceStrategySimulator
from features import prepare_features
from fastf1_plotting import (
    plot_circuit_with_corners, plot_team_pace_comparison, 
    plot_tyre_strategy_summary, plot_gear_shift_on_track,
    plot_speed_on_track
)
from advanced_viz import plot_telemetry_comparison, plot_track_3d, plot_gear_shift_trace, plot_corner_performance, plot_tyre_shape, plot_circuit_context
from qualifying_viz import plot_qualifying_evolution, plot_qualifying_gap, plot_sector_dominance
from home import render_home_tab
from f1_2026_updates import (
    render_f1_2026_updates_tab,
    OFFICIAL_2026_CALENDAR,
    SPRINT_2026,
    PRESEASON_TESTS_2026,
    OFFICIAL_2026_UPDATES,
    load_2026_api_snapshot,
    load_2026_news_snapshot,
)
from race_replay_data import get_race_replay_frames, get_circuit_rotation, format_race_time
from race_replay_viz import create_replay_animation, create_static_replay_frame, create_leaderboard_table, create_telemetry_gauges
try:
    from preseason_testing import load_preseason_testing_team_features, normalize_team_name
except ImportError:
    from src.preseason_testing import load_preseason_testing_team_features, normalize_team_name
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title=STREAMLIT_CONFIG.page_title,
    page_icon=STREAMLIT_CONFIG.page_icon,
    layout=STREAMLIT_CONFIG.layout,
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': None,
        'Report a bug': None,
        'About': None # You can put custom text here or None to hide
    }
)

# Initialize session state for navigation persistence
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 0
if 'driver_profile_selection' not in st.session_state:
    st.session_state.driver_profile_selection = None
if 'team_profile_selection' not in st.session_state:
    st.session_state.team_profile_selection = None
if 'race_detail_selection' not in st.session_state:
    st.session_state.race_detail_selection = None

# Custom CSS
st.markdown("""
<style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Aggressive Height Reduction for Header */
    header {visibility: hidden !important;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    
    /* Hide specific deploy button */
    /* Hide specific deploy button and badges */
    .stDeployButton {display: none !important;}
    [data-testid="stAppDeployButton"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    div[class*="stAppDeployButton"] {display: none !important;}
    
    /* Hide Fork button and GitHub link */
    .viewerBadge_container__1QSob {display: none;}
    .viewerBadge_link__1S137 {display: none;}
    button[title="View source on GitHub"] {display: none;}
    a[href*="github.com"] {display: none !important;}
    
    /* Hide Streamlit logo/branding bottom right */
    .stDeployButton {display: none;}
    div[data-testid="stDecoration"] {display: none;}
    button[kind="header"] {display: none;}
    .css-18ni7ap {display: none;}
    .css-1dp5vir {display: none;}
    
    /* Hide floating action buttons - ALL VARIANTS */
    .st-emotion-cache-1gulkj5 {display: none !important;}
    .st-emotion-cache-1wmy9hl {display: none !important;}
    [data-testid="stChatActionButtonIcon"] {display: none !important;}
    
    /* Hide bottom right floating buttons */
    div[data-testid="stBottomBlockContainer"] {display: none !important;}
    .stActionButton {display: none !important;}
    button[data-testid="stActionButton"] {display: none !important;}
    
    /* Hide ALL bottom right corner elements */
    div[class*="fixed"] {display: none !important;}
    div[style*="position: fixed"][style*="bottom"] {display: none !important;}
    div[style*="position: fixed"][style*="right"] {display: none !important;}
    
    /* Target the specific buttons in screenshot */
    button[aria-label*="Location"] {display: none !important;}
    button[kind="primaryFormSubmit"] {display: none !important;}
    div.stChatFloatingButtonContainer {display: none !important;}
    
    /* Hide streamlit branding */
    ._profileContainer_gzau3_53 {display: none !important;}
    ._profilePreview_gzau3_63 {display: none !important;}
    
    /* Custom styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E10600;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .driver-card {
        background: #15151e;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #333;
    }
    .driver-photo {
        border-radius: 50%;
        border: 3px solid #E10600;
    }
    .stat-box {
        background: #1f1f2e;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        text-align: center;
    }
    .weather-box {
        background: linear-gradient(135deg, #0f3460 0%, #16537e 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
    }
    .pit-stop-row {
        background: #1a1a2e;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .gauge-container {
        text-align: center;
        padding: 10px;
    }
    .telemetry-panel {
        background: #0a0a0f;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)


def setup_fastf1_cache():
    """Setup FastF1 cache directory."""
    cache_dir = Path("./f1_cache")
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    return cache_dir


def show_plotly_chart(fig, use_container_width=True, **kwargs):
    """Display Plotly chart with hidden toolbar and logo."""
    config = {
        'displayModeBar': False,  # Hide the entire toolbar
        'displaylogo': False,     # Hide Plotly logo
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'zoomIn2d', 'zoomOut2d', 
                                   'autoScale2d', 'resetScale2d', 'toImage'],
        'staticPlot': False       # Keep it interactive
    }
    # Streamlit can generate duplicate auto-IDs when identical charts are rendered in multiple tabs.
    # Assign a unique key unless the caller already provided one.
    if "key" not in kwargs:
        plot_counter = int(st.session_state.get("_plotly_auto_key_counter", 0)) + 1
        st.session_state["_plotly_auto_key_counter"] = plot_counter
        kwargs["key"] = f"plotly_auto_{plot_counter}"
    st.plotly_chart(fig, use_container_width=use_container_width, config=config, **kwargs)


def maybe_run_countdown_autorefresh() -> None:
    """Global 1s rerun for live countdown panels."""
    # Small sleep at end-of-render to keep countdowns ticking without aggressive CPU usage.
    time.sleep(1)
    st.rerun()


def get_active_season_year() -> int:
    """Resolve active season from sidebar mode without requiring explicit prop threading."""
    mode = str(st.session_state.get("sidebar_season_mode", "2026"))
    return 2026 if mode.startswith("2026") else 2025


@st.cache_data(ttl=1800)
def get_fastf1_schedule_cached(year: int) -> pd.DataFrame:
    """Cache FastF1 schedule to reduce repeated remote lookups across tabs."""
    try:
        schedule = fastf1.get_event_schedule(year)
        return schedule.copy() if schedule is not None else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Could not load FastF1 schedule for {year}: {e}")
        if year == 2026 and OFFICIAL_2026_CALENDAR:
            # Local fallback keeps countdown/calendar UI working when network access is blocked.
            rows = []
            for item in OFFICIAL_2026_CALENDAR:
                rows.append(
                    {
                        "RoundNumber": item.get("round"),
                        "EventName": item.get("event"),
                        "Location": item.get("location"),
                        "Country": item.get("country"),
                        "EventDate": pd.to_datetime(item.get("race_date"), errors="coerce", utc=True),
                        "Session1": None,
                        "Session1Date": pd.NaT,
                        "Session2": None,
                        "Session2Date": pd.NaT,
                        "Session3": None,
                        "Session3Date": pd.NaT,
                        "Session4": None,
                        "Session4Date": pd.NaT,
                        "Session5": "Race",
                        "Session5Date": pd.to_datetime(item.get("race_date"), errors="coerce", utc=True),
                    }
                )
            return pd.DataFrame(rows)
        return pd.DataFrame()


def get_season_race_choices(year: int | None = None) -> list:
    """Return race names for selected season with API-backed 2026 support and fallback."""
    year = year or get_active_season_year()

    if year == 2025:
        return F1_2025_COMPLETED_RACES

    schedule = get_fastf1_schedule_cached(year)
    if isinstance(schedule, pd.DataFrame) and not schedule.empty:
        name_col = "EventName" if "EventName" in schedule.columns else ("OfficialEventName" if "OfficialEventName" in schedule.columns else None)
        if name_col:
            df = schedule.copy()
            if "RoundNumber" in df.columns:
                df = df[pd.to_numeric(df["RoundNumber"], errors="coerce").fillna(0) > 0]
            names = [str(x) for x in df[name_col].dropna().tolist() if str(x).strip()]
            if names:
                return names

    if year == 2026:
        return [r["event"] for r in OFFICIAL_2026_CALENDAR]

    return []


def get_active_season_label() -> str:
    return str(get_active_season_year())


UTC_OFFSET_OPTIONS = [
    "-12:00", "-11:00", "-10:00", "-09:00", "-08:00", "-07:00", "-06:00", "-05:00",
    "-04:00", "-03:00", "-02:00", "-01:00", "+00:00", "+01:00", "+02:00", "+03:00",
    "+04:00", "+05:00", "+05:30", "+06:00", "+07:00", "+08:00", "+09:00", "+09:30",
    "+10:00", "+11:00", "+12:00", "+13:00", "+14:00"
]


def get_user_utc_offset() -> str:
    offset = str(st.session_state.get("user_utc_offset", "-05:00"))
    return offset if offset in UTC_OFFSET_OPTIONS else "-05:00"


def apply_user_utc_offset(ts) -> pd.Timestamp:
    t = pd.to_datetime(ts, errors="coerce", utc=True)
    if pd.isna(t):
        return t
    offset = get_user_utc_offset()
    sign = -1 if offset.startswith("-") else 1
    hh_mm = offset[1:] if offset and offset[0] in "+-" else offset
    try:
        hh, mm = hh_mm.split(":")
        delta = pd.Timedelta(hours=int(hh), minutes=int(mm)) * sign
        return t + delta
    except Exception:
        return t


def format_user_time(ts, fmt: str = "%Y-%m-%d %H:%M") -> str:
    t = apply_user_utc_offset(ts)
    if pd.isna(t):
        return "N/A"
    return f"{t.strftime(fmt)} (UTC{get_user_utc_offset()})"


@st.cache_data(ttl=300)
def get_next_race_countdown_summary(year: int) -> dict:
    """Small cached next-race summary for sidebar/home-level countdown widgets."""
    now = pd.Timestamp.now(tz="UTC")
    schedule = get_fastf1_schedule_cached(year)
    if isinstance(schedule, pd.DataFrame) and not schedule.empty and "EventDate" in schedule.columns:
        df = schedule.copy()
        df["EventDate"] = pd.to_datetime(df["EventDate"], errors="coerce", utc=True)
        if "RoundNumber" in df.columns:
            df = df[pd.to_numeric(df["RoundNumber"], errors="coerce").fillna(0) > 0]
        upcoming = df[df["EventDate"] >= now].sort_values("EventDate")
        if not upcoming.empty:
            row = upcoming.iloc[0]
            delta = row["EventDate"] - now
            secs = max(0, int(delta.total_seconds()))
            return {
                "ok": True,
                "source": "Live Schedule",
                "event": str(row.get("EventName", "Next Race")),
                "location": str(row.get("Location", "TBA")),
                "race_time": row["EventDate"],
                "days": secs // 86400,
                "hours": (secs % 86400) // 3600,
                "minutes": (secs % 3600) // 60,
                "seconds": secs % 60,
                "is_race_week": secs <= 7 * 86400,
            }

    if year == 2026 and OFFICIAL_2026_CALENDAR:
        rows = pd.DataFrame(OFFICIAL_2026_CALENDAR).copy()
        rows["race_date"] = pd.to_datetime(rows["race_date"], errors="coerce", utc=True)
        upcoming = rows[rows["race_date"] >= now].sort_values("race_date")
        if not upcoming.empty:
            row = upcoming.iloc[0]
            secs = max(0, int((row["race_date"] - now).total_seconds()))
            return {
                "ok": True,
                "source": "Official Fallback",
                "event": str(row.get("event", "Next Race")),
                "location": str(row.get("location", "TBA")),
                "race_time": row["race_date"],
                "days": secs // 86400,
                "hours": (secs % 86400) // 3600,
                "minutes": (secs % 3600) // 60,
                "seconds": secs % 60,
                "is_race_week": secs <= 7 * 86400,
            }

    return {"ok": False}


@st.cache_data(ttl=3600)
def load_race_data(year: int = 2025):
    """Load and cache race data (local CSV) for a season if available."""
    try:
        data_dir = Path(DATA_FILES.race_results).parent
        race_file = data_dir / f"Formula1_{year}Season_RaceResults.csv"
        sprint_file = data_dir / f"Formula1_{year}Season_SprintResults.csv"

        if year == 2025:
            # Keep existing combined loader path for the current local dataset
            try:
                df = load_combined_data(str(data_dir))
                if df is not None:
                    return clean_data(df)
            except Exception as e:
                logger.error(f"Error loading combined local {year} data: {e}")

        if not race_file.exists():
            # Return empty df for unsupported local seasons (e.g., 2026 pre-season) to keep UI responsive.
            return pd.DataFrame()

        df_race = load_csv_data(str(race_file))
        if df_race is None or df_race.empty:
            return pd.DataFrame()
        df_race["SessionType"] = "Race"
        frames = [df_race]

        if sprint_file.exists():
            df_sprint = load_csv_data(str(sprint_file))
            if df_sprint is not None and not df_sprint.empty:
                df_sprint["SessionType"] = "Sprint"
                frames.append(df_sprint)

        df = pd.concat(frames, ignore_index=True)
        return clean_data(df) if df is not None and not df.empty else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading race data for {year}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_sprint_data(year: int = 2025):
    """Load and cache sprint data for a season if available."""
    try:
        sprint_file = Path(DATA_FILES.race_results).parent / f'Formula1_{year}Season_SprintResults.csv'
        if sprint_file.exists():
            df = load_csv_data(str(sprint_file))
            if df is not None:
                df = clean_data(df)
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading sprint data for {year}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_preseason_prediction_priors() -> pd.DataFrame:
    """Load and score aggregated pre-season team priors for early-season prediction confidence."""
    try:
        priors = load_preseason_testing_team_features()
        if priors is None or priors.empty:
            return pd.DataFrame()
        out = priors.copy()

        def _rank_to_score(series: pd.Series) -> pd.Series:
            s = pd.to_numeric(series, errors="coerce")
            valid = s.dropna()
            if valid.empty:
                return pd.Series([np.nan] * len(s), index=s.index)
            min_v, max_v = valid.min(), valid.max()
            if min_v == max_v:
                return pd.Series([1.0 if pd.notna(x) else np.nan for x in s], index=s.index, dtype="float64")
            return s.apply(lambda x: (max_v - x) / (max_v - min_v) if pd.notna(x) else np.nan)

        out["pre_pace_score"] = _rank_to_score(out.get("preseason_bahrain_pace_rank", pd.Series(dtype=float)))
        out["pre_mileage_score"] = _rank_to_score(out.get("preseason_bahrain_mileage_rank", pd.Series(dtype=float)))
        out["pre_spain_participation_score"] = pd.to_numeric(
            out.get("preseason_spain_shakedown_participated"), errors="coerce"
        ).fillna(0.0)
        out["pre_spain_participation_score"] = (
            out["pre_spain_participation_score"]
            - 0.25 * pd.to_numeric(out.get("preseason_spain_shakedown_late_start"), errors="coerce").fillna(0.0)
        ).clip(lower=0.0, upper=1.0)

        out["preseason_prior_score"] = (
            out["pre_pace_score"].fillna(0) * 0.5
            + out["pre_mileage_score"].fillna(0) * 0.3
            + out["pre_spain_participation_score"].fillna(0) * 0.2
        )
        out["preseason_prior_coverage"] = (
            out[["pre_pace_score", "pre_mileage_score", "pre_spain_participation_score"]]
            .notna()
            .sum(axis=1)
            / 3.0
        )
        keep_cols = [
            c for c in [
                "team_norm",
                "preseason_prior_score",
                "preseason_prior_coverage",
                "pre_pace_score",
                "pre_mileage_score",
                "pre_spain_participation_score",
                "preseason_bahrain_pace_rank",
                "preseason_bahrain_mileage_rank",
                "preseason_bahrain_best_lap_sec",
                "preseason_bahrain_total_laps",
            ] if c in out.columns
        ]
        return out[keep_cols].copy()
    except Exception as e:
        logger.warning(f"Could not load pre-season prediction priors: {e}")
        return pd.DataFrame()


def _prediction_confidence_label(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    if score >= 75:
        return "High"
    if score >= 55:
        return "Medium"
    return "Low"


def _augment_predictions_with_preseason_signals(
    pred_df: pd.DataFrame,
    using_live_grid: bool,
    preseason_weight: float,
    use_weighted_sort: bool,
) -> pd.DataFrame:
    """Add preseason prior scores, scenario ranking, and confidence labels."""
    if pred_df is None or pred_df.empty:
        return pred_df

    out = pred_df.copy()
    priors_df = load_preseason_prediction_priors()
    if isinstance(priors_df, pd.DataFrame) and not priors_df.empty and "Team" in out.columns:
        out["team_norm"] = out["Team"].apply(normalize_team_name)
        out = out.merge(priors_df, on="team_norm", how="left")
    else:
        out["preseason_prior_score"] = np.nan
        out["preseason_prior_coverage"] = np.nan

    # Normalize model prediction into score space (higher = stronger)
    pred_num = pd.to_numeric(out.get("Predicted_Position"), errors="coerce")
    valid_pred = pred_num.dropna()
    if not valid_pred.empty:
        min_pred, max_pred = valid_pred.min(), valid_pred.max()
        if min_pred == max_pred:
            out["model_score_norm"] = 1.0
        else:
            out["model_score_norm"] = pred_num.apply(lambda x: (max_pred - x) / (max_pred - min_pred) if pd.notna(x) else np.nan)
    else:
        out["model_score_norm"] = np.nan

    out["preseason_prior_score"] = pd.to_numeric(out.get("preseason_prior_score"), errors="coerce")
    out["preseason_prior_coverage"] = pd.to_numeric(out.get("preseason_prior_coverage"), errors="coerce")
    w = float(np.clip(preseason_weight, 0.0, 0.5))
    prior_component = out["preseason_prior_score"].fillna(0.0)
    if "preseason_prior_coverage" in out.columns:
        prior_component = prior_component * out["preseason_prior_coverage"].fillna(0.0)

    out["weighted_prerace_score"] = (1.0 - w) * out["model_score_norm"].fillna(0.0) + w * prior_component
    out["model_component_weighted"] = ((1.0 - w) * out["model_score_norm"].fillna(0.0) * 100.0).round(2)
    out["prior_component_weighted"] = (w * prior_component * 100.0).round(2)
    out["Scenario_Rank"] = (
        out["weighted_prerace_score"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    out["Rank_Shift_vs_Model"] = (
        pd.to_numeric(out.get("Rank"), errors="coerce")
        - pd.to_numeric(out.get("Scenario_Rank"), errors="coerce")
    )

    # Confidence score (0-100): model-grid quality + preseason coverage + alignment
    out["grid_quality_score"] = 85.0 if using_live_grid else 55.0
    out["prior_coverage_score"] = out["preseason_prior_coverage"].fillna(0.0) * 100.0
    alignment = 1.0 - (out["model_score_norm"].fillna(0.5) - out["preseason_prior_score"].fillna(out["model_score_norm"].fillna(0.5))).abs()
    out["prior_alignment_score"] = alignment.clip(lower=0.0, upper=1.0) * 100.0
    out["Prediction_Confidence_Score"] = (
        out["grid_quality_score"] * 0.5
        + out["prior_coverage_score"] * 0.3
        + out["prior_alignment_score"] * 0.2
    ).round(1)
    out["Prediction_Confidence"] = out["Prediction_Confidence_Score"].apply(_prediction_confidence_label)

    def _why_changed_badge(row: pd.Series) -> str:
        shift = pd.to_numeric(row.get("Rank_Shift_vs_Model"), errors="coerce")
        coverage = pd.to_numeric(row.get("preseason_prior_coverage"), errors="coerce")
        pace_rank = pd.to_numeric(row.get("preseason_bahrain_pace_rank"), errors="coerce")
        mileage_rank = pd.to_numeric(row.get("preseason_bahrain_mileage_rank"), errors="coerce")
        reasons = []

        if pd.notna(coverage):
            if coverage >= 0.66:
                reasons.append("good coverage")
            elif coverage <= 0.0:
                reasons.append("no priors")
            else:
                reasons.append("partial coverage")
        if pd.notna(pace_rank):
            if pace_rank <= 4:
                reasons.append("strong Bahrain pace")
            elif pace_rank >= 8:
                reasons.append("weaker Bahrain pace")
        if pd.notna(mileage_rank):
            if mileage_rank <= 4:
                reasons.append("strong mileage")
            elif mileage_rank >= 8:
                reasons.append("low mileage")

        reason_text = ", ".join(reasons[:2]) if reasons else "model-driven"
        if pd.isna(shift):
            return f"No shift | {reason_text}"
        shift_i = int(shift)
        if shift_i > 0:
            return f"+{shift_i} | {reason_text}"
        if shift_i < 0:
            return f"{shift_i} | {reason_text}"
        return f"0 | {reason_text}"

    out["Why_Changed"] = out.apply(_why_changed_badge, axis=1)

    if use_weighted_sort:
        out = out.sort_values(["Scenario_Rank", "Predicted_Position"], ascending=[True, True]).reset_index(drop=True)
        out["Display_Rank"] = range(1, len(out) + 1)
    else:
        out = out.sort_values("Predicted_Position").reset_index(drop=True)
        out["Display_Rank"] = range(1, len(out) + 1)

    return out.drop(columns=[c for c in ["team_norm"] if c in out.columns])


def _render_prediction_results_panel(
    pred_df: pd.DataFrame,
    using_live: bool,
    preseason_weight: float,
    use_weighted_sort: bool,
    race_name: str,
) -> None:
    """Render prediction table, rank comparison, and per-driver confidence explanation."""
    if pred_df is None or pred_df.empty:
        st.info("No prediction results to display.")
        return

    coverage_ratio = float(
        pd.to_numeric(pred_df.get("preseason_prior_coverage"), errors="coerce").fillna(0).gt(0).mean()
    ) if "preseason_prior_coverage" in pred_df.columns else 0.0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Drivers", len(pred_df))
    m2.metric("Live Grid", "Yes" if using_live else "Estimated")
    m3.metric("Pre-season Coverage", f"{coverage_ratio*100:.0f}%")
    m4.metric("Scenario Weight", f"{preseason_weight:.2f}")

    if "Team" in pred_df.columns:
        team_summary = (
            pred_df.groupby("Team", dropna=False)
            .agg(
                Drivers=("Driver", "nunique"),
                Avg_Model_Rank=("Rank", "mean"),
                Avg_Weighted_Rank=("Scenario_Rank", "mean"),
                Net_Shift=("Rank_Shift_vs_Model", "sum"),
                Avg_Confidence=("Prediction_Confidence_Score", "mean"),
                Avg_Prior=("preseason_prior_score", "mean"),
                Prior_Coverage=("preseason_prior_coverage", "mean"),
            )
            .reset_index()
        )
        for col in ["Avg_Model_Rank", "Avg_Weighted_Rank", "Net_Shift", "Avg_Confidence", "Avg_Prior", "Prior_Coverage"]:
            if col in team_summary.columns:
                team_summary[col] = pd.to_numeric(team_summary[col], errors="coerce")
        if "Prior_Coverage" in team_summary.columns:
            team_summary["Prior_Coverage_Pct"] = (team_summary["Prior_Coverage"].fillna(0) * 100).round(0)
        if "Net_Shift" in team_summary.columns:
            team_summary["Shift_Badge"] = team_summary["Net_Shift"].apply(
                lambda x: "Up vs model" if pd.notna(x) and x > 0 else ("Down vs model" if pd.notna(x) and x < 0 else "Neutral")
            )

        st.markdown("#### Team Confidence Summary")
        t1, t2 = st.columns([3, 2])
        with t1:
            team_show = team_summary.copy()
            rename_team_cols = {
                "Avg_Model_Rank": "Avg_Model_Rank",
                "Avg_Weighted_Rank": "Avg_Weighted_Rank",
                "Net_Shift": "Net_Shift",
                "Avg_Confidence": "Avg_Conf",
                "Avg_Prior": "Avg_Prior",
                "Prior_Coverage_Pct": "Prior_Cov_%",
            }
            keep_team_cols = [c for c in ["Team", "Drivers", "Avg_Model_Rank", "Avg_Weighted_Rank", "Net_Shift", "Shift_Badge", "Avg_Confidence", "Prior_Coverage_Pct"] if c in team_show.columns]
            team_show = team_show[keep_team_cols].rename(columns=rename_team_cols)
            for c in [c for c in team_show.columns if c.startswith("Avg_") or c.endswith("_%") or c == "Net_Shift"]:
                team_show[c] = pd.to_numeric(team_show[c], errors="coerce").round(2)
            st.dataframe(team_show.sort_values(["Avg_Weighted_Rank", "Avg_Model_Rank"], na_position="last"), use_container_width=True, hide_index=True)
        with t2:
            if {"Team", "Avg_Confidence"}.issubset(team_summary.columns):
                fig_team_conf = px.bar(
                    team_summary.sort_values("Avg_Confidence", ascending=True),
                    x="Avg_Confidence",
                    y="Team",
                    orientation="h",
                    color="Shift_Badge" if "Shift_Badge" in team_summary.columns else None,
                    text="Avg_Confidence",
                    title="Avg Confidence by Team",
                    color_discrete_map={"Up vs model": "#00D2BE", "Down vs model": "#E10600", "Neutral": "#F5B942"},
                )
                fig_team_conf.update_layout(
                    xaxis_title="Confidence score",
                    yaxis_title="",
                    margin=dict(l=20, r=20, t=45, b=20),
                    height=330,
                    showlegend=False,
                )
                show_plotly_chart(fig_team_conf, use_container_width=True)

    display_cols = [
        c for c in [
            'Display_Rank', 'Rank', 'Scenario_Rank',
            'Driver', 'Team', 'Starting Grid', 'Predicted_Position',
            'Prediction_Confidence', 'Prediction_Confidence_Score',
            'Rank_Shift_vs_Model', 'Why_Changed',
            'preseason_prior_score', 'preseason_bahrain_pace_rank',
            'preseason_bahrain_mileage_rank'
        ] if c in pred_df.columns
    ]
    display_df = pred_df[display_cols].copy()
    rename_map = {
        "Display_Rank": "Table_Rank",
        "Rank": "Model_Rank",
        "Scenario_Rank": "Weighted_Rank",
        "Predicted_Position": "Pred_Pos",
        "Prediction_Confidence": "Confidence",
        "Prediction_Confidence_Score": "Conf_Score",
        "Rank_Shift_vs_Model": "Shift_vs_Model",
        "Why_Changed": "Why_Changed",
        "preseason_prior_score": "Preseason_Prior",
        "preseason_bahrain_pace_rank": "BHR_Pace_Rank",
        "preseason_bahrain_mileage_rank": "BHR_Mileage_Rank",
    }
    display_df = display_df.rename(columns=rename_map)
    if "Shift_vs_Model" in display_df.columns:
        display_df["Shift_vs_Model"] = pd.to_numeric(display_df["Shift_vs_Model"], errors="coerce").round(0).astype("Int64")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if {'Driver', 'Rank', 'Scenario_Rank'}.issubset(pred_df.columns):
        rank_chart_df = pred_df[['Driver', 'Rank', 'Scenario_Rank']].copy().melt(
            id_vars='Driver', value_vars=['Rank', 'Scenario_Rank'],
            var_name='RankType', value_name='RankValue'
        )
        fig_rank = px.line(
            rank_chart_df,
            x='Driver',
            y='RankValue',
            color='RankType',
            markers=True,
            title='Model Rank vs Weighted Scenario Rank',
            color_discrete_map={'Rank': '#00D2BE', 'Scenario_Rank': '#E10600'}
        )
        fig_rank.update_layout(
            yaxis_title='Rank (lower is better)',
            yaxis_autorange='reversed',
            xaxis_title='',
            xaxis_tickangle=-45,
            margin=dict(l=20, r=20, t=50, b=80)
        )
        show_plotly_chart(fig_rank, use_container_width=True)

    with st.expander("Confidence Explanation per Driver", expanded=False):
        driver_options = pred_df.sort_values("Display_Rank")[["Driver", "Team"]].drop_duplicates()
        labels = driver_options.apply(lambda r: f"{r['Driver']} ({r['Team']})", axis=1).tolist()
        if not labels:
            st.info("No driver rows available.")
        else:
            key_suffix = f"{get_active_season_year()}_{race_name}".replace(" ", "_")
            selected_label = st.selectbox(
                "Select driver",
                labels,
                key=f"pred_conf_explain_driver_{key_suffix}",
            )
            selected_driver = selected_label.split(" (")[0]
            row = pred_df[pred_df["Driver"] == selected_driver].iloc[0]

            rank_shift = pd.to_numeric(row.get("Rank_Shift_vs_Model"), errors="coerce")
            rank_shift_txt = (
                f"{int(rank_shift):+d} vs model rank"
                if pd.notna(rank_shift) else "N/A"
            )

            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Model Rank", int(row["Rank"]) if pd.notna(row.get("Rank")) else "N/A")
            x2.metric("Weighted Rank", int(row["Scenario_Rank"]) if pd.notna(row.get("Scenario_Rank")) else "N/A", rank_shift_txt)
            x3.metric("Confidence", str(row.get("Prediction_Confidence", "Unknown")))
            x4.metric("Conf Score", f"{float(pd.to_numeric(row.get('Prediction_Confidence_Score'), errors='coerce')):.1f}" if pd.notna(pd.to_numeric(row.get("Prediction_Confidence_Score"), errors="coerce")) else "N/A")

            def _num(v, default: float = 0.0) -> float:
                parsed = pd.to_numeric(v, errors="coerce")
                return float(parsed) if pd.notna(parsed) else float(default)

            grid_quality = _num(row.get("grid_quality_score"))
            prior_coverage = _num(row.get("prior_coverage_score"))
            prior_alignment = _num(row.get("prior_alignment_score"))

            contrib_df = pd.DataFrame([
                {
                    "Component": "Grid Data Quality (50%)",
                    "RawScore": grid_quality,
                    "Weight": 0.50,
                    "Contribution": grid_quality * 0.50,
                },
                {
                    "Component": "Pre-season Coverage (30%)",
                    "RawScore": prior_coverage,
                    "Weight": 0.30,
                    "Contribution": prior_coverage * 0.30,
                },
                {
                    "Component": "Model/Prior Alignment (20%)",
                    "RawScore": prior_alignment,
                    "Weight": 0.20,
                    "Contribution": prior_alignment * 0.20,
                },
            ])

            blend_df = pd.DataFrame([
                {
                    "Part": "Model score component",
                    "WeightedScorePts": float(pd.to_numeric(row.get("model_component_weighted"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("model_component_weighted"), errors="coerce")) else 0.0,
                },
                {
                    "Part": "Pre-season prior component",
                    "WeightedScorePts": float(pd.to_numeric(row.get("prior_component_weighted"), errors="coerce")) if pd.notna(pd.to_numeric(row.get("prior_component_weighted"), errors="coerce")) else 0.0,
                },
            ])

            c_left, c_right = st.columns(2)
            with c_left:
                fig_conf = px.bar(
                    contrib_df.sort_values("Contribution", ascending=True),
                    x="Contribution",
                    y="Component",
                    orientation="h",
                    text="Contribution",
                    title="Confidence Score Breakdown",
                    color="Contribution",
                    color_continuous_scale="Blues",
                )
                fig_conf.update_layout(
                    xaxis_title="Contribution points (0-100 scale)",
                    yaxis_title="",
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=320,
                )
                show_plotly_chart(fig_conf, use_container_width=True)
                st.dataframe(contrib_df.round(2), use_container_width=True, hide_index=True)
            with c_right:
                fig_blend = px.bar(
                    blend_df,
                    x="Part",
                    y="WeightedScorePts",
                    text="WeightedScorePts",
                    title="Scenario Rank Blend (Model vs Pre-season Prior)",
                    color="Part",
                    color_discrete_map={
                        "Model score component": "#00D2BE",
                        "Pre-season prior component": "#E10600",
                    },
                )
                fig_blend.update_layout(
                    xaxis_title="",
                    yaxis_title="Weighted score points",
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=320,
                    showlegend=False,
                )
                show_plotly_chart(fig_blend, use_container_width=True)

                st.markdown("**Explanation**")
                grid_source_txt = "live qualifying grid" if using_live else "estimated historical grid"
                prior_cov = pd.to_numeric(row.get("preseason_prior_coverage"), errors="coerce")
                prior_cov_pct = float(prior_cov * 100.0) if pd.notna(prior_cov) else 0.0
                pace_rank = pd.to_numeric(row.get("preseason_bahrain_pace_rank"), errors="coerce")
                mileage_rank = pd.to_numeric(row.get("preseason_bahrain_mileage_rank"), errors="coerce")
                st.markdown(f"- Grid source: `{grid_source_txt}`")
                st.markdown(f"- Pre-season coverage: `{prior_cov_pct:.0f}%`")
                if pd.notna(pace_rank):
                    st.markdown(f"- Bahrain pace rank: `{int(pace_rank)}` (lebih kecil = lebih cepat)")
                if pd.notna(mileage_rank):
                    st.markdown(f"- Bahrain mileage rank: `{int(mileage_rank)}` (lebih kecil = lebih banyak lap)")
                if pd.notna(rank_shift):
                    if rank_shift > 0:
                        st.markdown(f"- Overlay pre-season menaikkan posisi skenario sebanyak `{int(rank_shift)}`.")
                    elif rank_shift < 0:
                        st.markdown(f"- Overlay pre-season menurunkan posisi skenario sebanyak `{abs(int(rank_shift))}`.")
                    else:
                        st.markdown("- Overlay pre-season tidak mengubah ranking driver ini.")
                st.markdown(f"- Baseline model rank tetap tersedia sebagai `Model_Rank`; overlay hanya memengaruhi `Weighted_Rank`.")

            raw_cols = [
                c for c in [
                    "Driver", "Team", "Starting Grid", "Predicted_Position", "Rank", "Scenario_Rank",
                    "weighted_prerace_score", "model_score_norm", "preseason_prior_score", "preseason_prior_coverage",
                    "grid_quality_score", "prior_coverage_score", "prior_alignment_score",
                    "model_component_weighted", "prior_component_weighted",
                    "Prediction_Confidence", "Prediction_Confidence_Score",
                    "preseason_bahrain_best_lap_sec", "preseason_bahrain_total_laps",
                    "preseason_bahrain_pace_rank", "preseason_bahrain_mileage_rank"
                ] if c in pred_df.columns
            ]
            with st.expander("Raw driver explanation fields"):
                st.dataframe(pd.DataFrame([row[raw_cols]]), use_container_width=True, hide_index=True)

    winner = pred_df.iloc[0]
    rank_label = "Weighted scenario winner" if (use_weighted_sort and preseason_weight > 0) else "Predicted Winner"
    conf = winner.get('Prediction_Confidence', 'Unknown')
    st.success(f"{rank_label}: {winner['Driver']} ({winner['Team']}) | Confidence: {conf}")


def load_fastf1_session(year: int, race: str, session_type: str, load_telemetry: bool = False):
    """Load FastF1 session with caching and optional telemetry."""
    try:
        session = fastf1.get_session(year, race, session_type)
        if load_telemetry:
            session.load()
        else:
            session.load(telemetry=False, laps=True, weather=True)
        return session
    except Exception as e:
        logger.error(f"Error loading FastF1 session: {e}")
        return None


def get_fastf1_session_state_cached(
    *,
    year: int,
    race: str,
    session_type: str,
    load_telemetry: bool = False,
    cache_namespace: str = "default",
    force_reload: bool = False,
):
    """Reuse loaded FastF1 session objects across reruns for the same selection."""
    sig = f"{year}|{race}|{session_type}|{int(load_telemetry)}"
    sig_key = f"{cache_namespace}_sig"
    obj_key = f"{cache_namespace}_session_obj"
    err_key = f"{cache_namespace}_last_error"

    if not force_reload and st.session_state.get(sig_key) == sig and obj_key in st.session_state:
        return st.session_state.get(obj_key)

    setup_fastf1_cache()
    session = load_fastf1_session(year, race, session_type, load_telemetry=load_telemetry)
    if session is not None:
        st.session_state[sig_key] = sig
        st.session_state[obj_key] = session
        st.session_state.pop(err_key, None)
    else:
        st.session_state[err_key] = f"{race} - {session_type}"
    return session


@st.cache_data(ttl=3600)
def get_real_grid_positions(year: int, race: str) -> dict:
    """
    Fetch actual Qualifying grid positions from FastF1.
    """
    try:
        # Try Loading Qualifying
        session = fastf1.get_session(year, race, 'Q')
        # Light load (no telemetry)
        session.load(telemetry=False, laps=False, weather=False)
        
        if session.results is None or session.results.empty:
            return {}
            
        # Create mapping: Driver -> GridPosition
        # Ensure GridPosition is valid (not 0.0 unless pole?)
        # 0.0 usually means pit lane or DQ? No, 0.0 is unclassified?
        # Pole is 1.0. FastF1 uses 0.0 for missing?
        grid_map = {}
        for drv in session.results['Abbreviation']:
            res = session.results.loc[session.results['Abbreviation'] == drv].iloc[0]
            grid = res['GridPosition']
            if grid <= 0:
                # Fallback to Position if Grid is 0 (e.g. penalities applied later, or just use finish position of Q)
                grid = res['Position']
            grid_map[drv] = grid
            
        return grid_map
    except Exception as e:
        logger.warning(f"Could not fetch real grid for {race}: {e}")
        return {}


def create_gauge(value, max_value, title, color="#E10600"):
    """Create a gauge chart for telemetry display."""
    # Safely convert value to float
    try:
        if value is None:
            value = 0.0
        elif isinstance(value, (bool, np.bool_)):
            value = 100.0 if value else 0.0
        elif hasattr(value, 'item'):
            value = float(value.item())
        elif isinstance(value, (np.integer, np.floating)):
            value = float(value)
        else:
            value = float(value)
    except (TypeError, ValueError):
        value = 0.0
    
    # Ensure max_value is valid
    try:
        max_value = float(max_value) if max_value and max_value > 0 else 100.0
    except:
        max_value = 100.0
    
    # Clamp value to valid range
    value = max(0.0, min(float(value), float(max_value)))
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, max_value], 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "gray",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, max_value * 0.6], 'color': '#1a1a2e'},
                {'range': [max_value * 0.6, max_value * 0.85], 'color': '#2a2a4e'},
                {'range': [max_value * 0.85, max_value], 'color': '#3a3a6e'}
            ],
        },
        number={'font': {'color': 'white'}}
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def format_f1_time(td: pd.Timedelta) -> str:
    """Formats a pandas Timedelta object into an F1-style lap time string (M:SS.mmm or H:MM:SS.mmm)."""
    if pd.isna(td):
        return "N/A"
    
    total_seconds = td.total_seconds()
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:06.3f}"
    else:
        return f"{minutes:d}:{seconds:06.3f}"


def render_race_replay_tab():
    """Race Replay visualization tab with animated track and leaderboard."""
    st.subheader("Race Replay")
    st.markdown("Watch the race unfold with animated driver positions on track.")
    
    # Initialize session state for replay
    if 'replay_data' not in st.session_state:
        st.session_state.replay_data = None
    if 'replay_frame_idx' not in st.session_state:
        st.session_state.replay_frame_idx = 0
    if 'selected_replay_driver' not in st.session_state:
        st.session_state.selected_replay_driver = None
    
    # Race selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_race = st.selectbox(
            "Select Race",
            get_season_race_choices(),
            key="replay_race_select"
        )
    
    with col2:
        session_type = st.selectbox(
            "Session",
            ["Race", "Sprint"],
            key="replay_session_type"
        )
    
    with col3:
        frame_skip = st.slider(
            "Quality",
            min_value=1,
            max_value=20,
            value=5,
            help="Lower = more frames, smoother animation but slower"
        )
    
    # Load buttons - side by side
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        if st.button("Load Web Replay", type="primary", key="load_replay_btn"):
            with st.spinner("Loading telemetry data... This may take 1-2 minutes for first load."):
                try:
                    session_code = 'S' if session_type == "Sprint" else 'R'
                    session = get_fastf1_session_state_cached(
                        year=get_active_season_year(),
                        race=selected_race,
                        session_type=session_code,
                        load_telemetry=True,
                        cache_namespace="replay_fastf1",
                        force_reload=True,
                    )
                    
                    if session is not None:
                        replay_data = get_race_replay_frames(session, session_type=session_code)
                        st.session_state.replay_data = replay_data
                        st.session_state.replay_frame_idx = 0
                        st.success(f"Loaded {len(replay_data['frames'])} frames for {selected_race}")
                    else:
                        st.error("Could not load session data. Make sure the race has been completed.")
                except Exception as e:
                    st.error(f"Error loading replay: {e}")
                    logger.exception("Replay load error")
    
    with btn_col2:
        if st.button("Launch Desktop Replay", type="secondary", key="launch_desktop_btn"):
            import subprocess
            import sys
            
            session_code = 'S' if session_type == "Sprint" else 'R'
            script_path = Path(__file__).parent / "arcade_replay_window.py"
            
            try:
                # Spawn desktop replay window as separate process
                subprocess.Popen([
                    sys.executable,
                    str(script_path),
                    "--year", str(get_active_season_year()),
                    "--race", selected_race,
                    "--session", session_code
                ], cwd=str(Path(__file__).parent.parent))
                
                st.success("Desktop replay window launching... (Check your taskbar)")
                st.info("Controls: SPACE=Play/Pause, LEFT/RIGHT=Seek, UP/DOWN=Speed, 1-4=Quick Speed, R=Reset, ESC=Close")
            except Exception as e:
                st.error(f"Could not launch desktop replay: {e}")
    
    # Display replay if data is loaded
    if st.session_state.replay_data is not None:
        data = st.session_state.replay_data
        frames = data['frames']
        track = data['track']
        colors = data['driver_colors']
        total_laps = data['total_laps']
        
        st.divider()
        
        # Mode selection
        replay_mode = st.radio(
            "Replay Mode",
            ["Animated", "Manual Step"],
            horizontal=True,
            key="replay_mode"
        )
        
        if replay_mode == "Animated":
            # Animated replay with Plotly
            st.markdown("**Use the Play/Pause buttons and slider below the track to control playback.**")
            
            try:
                rotation_session = get_fastf1_session_state_cached(
                    year=get_active_season_year(),
                    race=selected_race,
                    session_type='R',
                    load_telemetry=False,
                    cache_namespace="replay_rotation_fastf1",
                    force_reload=False,
                )
                rotation = get_circuit_rotation(rotation_session) if track and rotation_session is not None else 0
            except:
                rotation = 0
            
            fig = create_replay_animation(
                frames=frames,
                track_coords=track,
                driver_colors=colors,
                total_laps=total_laps,
                rotation=rotation,
                frame_skip=frame_skip
            )
            
            if fig:
                config = {
                    'displayModeBar': False,
                    'displaylogo': False,
                }
                st.plotly_chart(fig, use_container_width=True, config=config)
            else:
                st.warning("Could not create animation. Track data may be unavailable.")
        
        else:
            # Manual stepping mode
            col_track, col_info = st.columns([2, 1])
            
            with col_track:
                # Frame slider
                frame_idx = st.slider(
                    "Race Progress",
                    min_value=0,
                    max_value=len(frames) - 1,
                    value=st.session_state.replay_frame_idx,
                    key="frame_slider"
                )
                st.session_state.replay_frame_idx = frame_idx
                
                current_frame = frames[frame_idx]
                
                try:
                    rotation_session = get_fastf1_session_state_cached(
                        year=get_active_season_year(),
                        race=selected_race,
                        session_type='R',
                        load_telemetry=False,
                        cache_namespace="replay_rotation_fastf1",
                        force_reload=False,
                    )
                    rotation = get_circuit_rotation(rotation_session) if track and rotation_session is not None else 0
                except:
                    rotation = 0
                
                fig = create_static_replay_frame(
                    frame=current_frame,
                    track_coords=track,
                    driver_colors=colors,
                    total_laps=total_laps,
                    rotation=rotation
                )
                
                if fig:
                    config = {'displayModeBar': False, 'displaylogo': False}
                    st.plotly_chart(fig, use_container_width=True, config=config)
            
            with col_info:
                # Leaderboard
                st.markdown("**Live Standings**")
                drivers_data = current_frame.get('drivers', {})
                
                if drivers_data:
                    leaderboard_fig = create_leaderboard_table(drivers_data, colors)
                    if leaderboard_fig:
                        st.plotly_chart(leaderboard_fig, use_container_width=True, config={'displayModeBar': False})
                    
                    # Driver selection for telemetry
                    st.markdown("**Driver Telemetry**")
                    driver_codes = sorted(drivers_data.keys(), key=lambda d: drivers_data[d]['position'])
                    selected_driver = st.selectbox(
                        "Select Driver",
                        driver_codes,
                        key="telemetry_driver_select"
                    )
                    
                    if selected_driver and selected_driver in drivers_data:
                        driver_data = drivers_data[selected_driver]
                        color = colors.get(selected_driver, '#E10600')
                        
                        telemetry_fig = create_telemetry_gauges(driver_data, color)
                        if telemetry_fig:
                            st.plotly_chart(telemetry_fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Select a race and click `Load Web Replay` to start.")


def render_header():
    """Render dashboard header."""
    st.markdown(f'<h1 class="main-header">F1 {get_active_season_label()} Season Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analytics | Telemetry | Predictions</p>', unsafe_allow_html=True)


F1_2026_GRID_TECH_PREVIEW = [
    {
        "team_display": "McLaren",
        "color_key": "McLaren",
        "base": "Woking, United Kingdom",
        "team_principal": "Andrea Stella",
        "car_name_2026": "TBA (2026 launch)",
        "pu_supplier_2026": "Mercedes",
        "pu_programme": "Mercedes customer",
        "drivers_preview": ["Lando Norris", "Oscar Piastri"],
        "lineup_status": "Tracked",
        "car_image": "https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/mclaren.png",
        "notes": "Reference car image uses latest available team asset until 2026 launch renders are published.",
    },
    {
        "team_display": "Ferrari",
        "color_key": "Ferrari",
        "base": "Maranello, Italy",
        "team_principal": "Frederic Vasseur",
        "car_name_2026": "TBA (2026 launch)",
        "pu_supplier_2026": "Ferrari",
        "pu_programme": "Works PU",
        "drivers_preview": ["Charles Leclerc", "Lewis Hamilton"],
        "lineup_status": "Tracked",
        "car_image": "https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/ferrari.png",
        "notes": "Team + driver gallery syncs to local profile library for portraits and career highlights.",
    },
    {
        "team_display": "Mercedes",
        "color_key": "Mercedes",
        "base": "Brackley, United Kingdom",
        "team_principal": "Toto Wolff",
        "car_name_2026": "TBA (2026 launch)",
        "pu_supplier_2026": "Mercedes",
        "pu_programme": "Works PU",
        "drivers_preview": ["George Russell", "Andrea Kimi Antonelli"],
        "lineup_status": "Tracked",
        "car_image": "https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/mercedes.png",
        "notes": "2026 technical package preview focuses on PU supplier and regulation impact readiness.",
    },
    {
        "team_display": "Red Bull Racing",
        "color_key": "Red Bull Racing Honda RBPT",
        "base": "Milton Keynes, United Kingdom",
        "team_principal": "Christian Horner",
        "car_name_2026": "TBA (2026 launch)",
        "pu_supplier_2026": "Red Bull Ford Powertrains",
        "pu_programme": "Works PU (RBPT/Ford)",
        "drivers_preview": ["Max Verstappen", "TBA"],
        "lineup_status": "Tracker + TBA",
        "car_image": "https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/red-bull-racing.png",
        "notes": "Driver slot tracker updates as official entries are published.",
    },
    {
        "team_display": "Aston Martin",
        "color_key": "Aston Martin Aramco Mercedes",
        "base": "Silverstone, United Kingdom",
        "team_principal": "Mike Krack",
        "car_name_2026": "TBA (2026 launch)",
        "pu_supplier_2026": "Honda",
        "pu_programme": "Works / strategic PU partner",
        "drivers_preview": ["Fernando Alonso", "Lance Stroll"],
        "lineup_status": "Tracked",
        "car_image": "https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/aston-martin.png",
        "notes": "2026 PU supplier transition highlighted in the engine matrix.",
    },
    {
        "team_display": "Alpine",
        "color_key": "Alpine Renault",
        "base": "Enstone, United Kingdom",
        "team_principal": "Oliver Oakes",
        "car_name_2026": "TBA (2026 launch)",
        "pu_supplier_2026": "Mercedes",
        "pu_programme": "Customer PU",
        "drivers_preview": ["Pierre Gasly", "Jack Doohan"],
        "lineup_status": "Tracked",
        "car_image": "https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/alpine.png",
        "notes": "Power unit programme shown as pre-season preview tracker.",
    },
    {
        "team_display": "Williams",
        "color_key": "Williams Mercedes",
        "base": "Grove, United Kingdom",
        "team_principal": "James Vowles",
        "car_name_2026": "TBA (2026 launch)",
        "pu_supplier_2026": "Mercedes",
        "pu_programme": "Customer PU",
        "drivers_preview": ["Alexander Albon", "Carlos Sainz"],
        "lineup_status": "Tracked",
        "car_image": "https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/williams.png",
        "notes": "Barcelona shakedown participation signals are visible in ML/2026 Hub diagnostics.",
    },
    {
        "team_display": "Racing Bulls",
        "color_key": "Racing Bulls Honda RBPT",
        "base": "Faenza, Italy",
        "team_principal": "Laurent Mekies",
        "car_name_2026": "TBA (2026 launch)",
        "pu_supplier_2026": "Red Bull Ford Powertrains",
        "pu_programme": "Customer / family PU",
        "drivers_preview": ["Yuki Tsunoda", "TBA"],
        "lineup_status": "Tracker + TBA",
        "car_image": "https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/rb.png",
        "notes": "Second seat shown as tracker slot until final confirmation in source feeds.",
    },
    {
        "team_display": "Audi",
        "color_key": "Kick Sauber Ferrari",
        "base": "Hinwil, Switzerland",
        "team_principal": "Mattia Binotto",
        "car_name_2026": "TBA (Audi 2026 launch)",
        "pu_supplier_2026": "Audi",
        "pu_programme": "Works PU",
        "drivers_preview": ["Nico Hulkenberg", "Gabriel Bortoleto"],
        "lineup_status": "Tracked",
        "car_image": "https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/kick-sauber.png",
        "notes": "Car image uses latest available Sauber reference until Audi launch assets are published.",
    },
    {
        "team_display": "Haas",
        "color_key": "Haas Ferrari",
        "base": "Kannapolis, USA",
        "team_principal": "Ayao Komatsu",
        "car_name_2026": "TBA (2026 launch)",
        "pu_supplier_2026": "Ferrari",
        "pu_programme": "Customer PU",
        "drivers_preview": ["Esteban Ocon", "Oliver Bearman"],
        "lineup_status": "Tracked",
        "car_image": "https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/haas.png",
        "notes": "Constructor preview keeps car image and engine family visible before season results arrive.",
    },
    {
        "team_display": "Cadillac",
        "color_key": "Cadillac",
        "base": "United States / Silverstone (entry operations)",
        "team_principal": "TBA",
        "car_name_2026": "TBA (entry debut package)",
        "pu_supplier_2026": "Customer PU (TBC at launch)",
        "pu_programme": "Entry team (GM/Cadillac programme evolving)",
        "drivers_preview": ["TBA", "TBA"],
        "lineup_status": "New entry",
        "car_image": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/512px-F1.svg.png",
        "notes": "Cadillac joins as the 11th team in 2026; lineup and launch assets update as official announcements arrive.",
    },
]

F1_2026_REGULATION_METRICS = pd.DataFrame(
    [
        {"Metric": "Car Width", "2025": 200, "2026": 190, "Unit": "cm", "Direction": "Smaller"},
        {"Metric": "Car Length", "2025": 360, "2026": 340, "Unit": "cm", "Direction": "Shorter"},
        {"Metric": "Minimum Weight", "2025": 798, "2026": 768, "Unit": "kg", "Direction": "Lighter"},
        {"Metric": "PU Electric Share", "2025": 20, "2026": 50, "Unit": "%", "Direction": "Higher"},
    ]
)

F1_2026_REGULATION_FEATURES = [
    {"feature": "Active Aero", "status": "New", "impact": "Straight-line efficiency + cornering balance modes"},
    {"feature": "MGU-H", "status": "Removed", "impact": "Simpler PU package and new energy deployment strategies"},
    {"feature": "Sustainable Fuel", "status": "100%", "impact": "Fuel development becomes a performance lever"},
    {"feature": "Chassis Package", "status": "Smaller/Lighter", "impact": "Agility, braking, and tyre load patterns change"},
    {"feature": "Energy Deployment", "status": "Higher Electric Share", "impact": "Race strategy and overtaking energy management shift"},
]


def _countdown_parts_precise(target_ts, now: pd.Timestamp | None = None) -> dict:
    now_ts = pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now, errors="coerce", utc=True)
    target = pd.to_datetime(target_ts, errors="coerce", utc=True)
    if pd.isna(now_ts) or pd.isna(target):
        return {"ok": False, "total_seconds": None, "days": None, "hours": None, "minutes": None, "seconds": None}
    total_seconds = max(0, int((target - now_ts).total_seconds()))
    return {
        "ok": True,
        "total_seconds": total_seconds,
        "days": total_seconds // 86400,
        "hours": (total_seconds % 86400) // 3600,
        "minutes": (total_seconds % 3600) // 60,
        "seconds": total_seconds % 60,
    }


def _countdown_text_precise(target_ts, now: pd.Timestamp | None = None) -> str:
    p = _countdown_parts_precise(target_ts, now)
    if not p.get("ok"):
        return "N/A"
    return f"{p['days']}d {p['hours']}h {p['minutes']}m {p['seconds']}s"


@st.cache_data(ttl=3600)
def get_2026_calendar_preview_df() -> pd.DataFrame:
    if not OFFICIAL_2026_CALENDAR:
        return pd.DataFrame()
    d = pd.DataFrame(OFFICIAL_2026_CALENDAR).copy()
    if d.empty:
        return d
    d["race_date"] = pd.to_datetime(d["race_date"], errors="coerce", utc=True)
    d["month"] = d["race_date"].dt.strftime("%b")
    d["month_num"] = d["race_date"].dt.month
    if "event" in d.columns:
        d["is_sprint"] = d["event"].isin(SPRINT_2026)
    d = d.sort_values("race_date").reset_index(drop=True)
    d["round_idx"] = np.arange(1, len(d) + 1)
    return d


@st.cache_data(ttl=3600)
def get_2026_grid_tech_preview_df() -> pd.DataFrame:
    d = pd.DataFrame(F1_2026_GRID_TECH_PREVIEW).copy()
    if d.empty:
        return d
    d["team_color"] = d["color_key"].apply(lambda x: TEAM_COLORS.get(x, "#888888"))
    d["drivers_label"] = d["drivers_preview"].apply(lambda x: " / ".join([str(v) for v in (x or [])]))
    return d


@st.cache_data(ttl=3600)
def get_2026_driver_gallery_df() -> pd.DataFrame:
    tech_df = get_2026_grid_tech_preview_df()
    rows = []
    for _, row in tech_df.iterrows():
        for idx, driver in enumerate(row.get("drivers_preview", []) or [], start=1):
            prof = DRIVER_PROFILES.get(driver, {})
            rows.append(
                {
                    "Driver": driver,
                    "Team": row.get("team_display"),
                    "Slot": idx,
                    "Lineup_Status": row.get("lineup_status"),
                    "PU_Supplier_2026": row.get("pu_supplier_2026"),
                    "Team_Color": row.get("team_color"),
                    "Image": prof.get("image_url"),
                    "Number": prof.get("number"),
                    "Country": prof.get("country"),
                    "Debut": prof.get("debut"),
                    "Titles": prof.get("titles"),
                    "Wins": prof.get("wins"),
                    "Podiums": prof.get("podiums"),
                    "Bio": prof.get("bio"),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["Has_Image"] = out["Image"].fillna("").astype(str).str.len() > 0
    return out


def _find_schedule_event_row(year: int, race_name: str):
    sched = get_fastf1_schedule_cached(year)
    if not isinstance(sched, pd.DataFrame) or sched.empty or "EventName" not in sched.columns:
        return None
    df = sched.copy()
    m = df[df["EventName"].astype(str).str.lower() == str(race_name).lower()]
    if m.empty:
        m = df[df["EventName"].astype(str).str.contains(str(race_name), case=False, na=False)]
    if m.empty:
        return None
    row = m.iloc[0].copy()
    for c in [x for x in row.index if str(x).endswith("Date")]:
        row[c] = pd.to_datetime(row[c], errors="coerce", utc=True)
    return row


def _render_data_availability_badges(items: list[tuple[str, bool, str]]) -> None:
    chips = []
    for label, ok, _note in items:
        bg = "rgba(0,210,190,0.15)" if ok else "rgba(245,185,66,0.14)"
        fg = "#bff8f2" if ok else "#ffe5a8"
        border = "rgba(0,210,190,0.35)" if ok else "rgba(245,185,66,0.35)"
        chips.append(
            f"<span style='display:inline-block; margin:4px 6px 0 0; padding:4px 10px; border-radius:999px; "
            f"border:1px solid {border}; background:{bg}; color:{fg}; font-size:12px; font-weight:700;'>{label}: {'Ready' if ok else 'Pending'}</span>"
        )
    if chips:
        st.markdown("".join(chips), unsafe_allow_html=True)
    with st.expander("Availability details"):
        st.dataframe(
            pd.DataFrame([{"Data": label, "Status": "Ready" if ok else "Pending", "Notes": note} for label, ok, note in items]),
            hide_index=True,
            use_container_width=True,
        )

def _render_2026_season_overview_fallback() -> None:
    st.info("Race results are not loaded yet for 2026. Showing an interactive pre-season dashboard (calendar, grid, tech, regulations, and live data services).")

    cal_df = get_2026_calendar_preview_df()
    tech_df = get_2026_grid_tech_preview_df()
    priors_df = load_preseason_prediction_priors()
    next_race = get_next_race_countdown_summary(2026)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Calendar Rounds", len(cal_df))
    m2.metric("Sprint Weekends", int(cal_df["is_sprint"].fillna(False).sum()) if isinstance(cal_df, pd.DataFrame) and not cal_df.empty and "is_sprint" in cal_df.columns else 0)
    m3.metric("Teams Tracked", len(tech_df))
    m4.metric("PU Suppliers", tech_df["pu_supplier_2026"].nunique() if isinstance(tech_df, pd.DataFrame) and not tech_df.empty else 0)
    countdown_value = (
        f"{int(next_race.get('days', 0))}d {int(next_race.get('hours', 0))}h {int(next_race.get('minutes', 0))}m {int(next_race.get('seconds', 0))}s"
        if next_race.get("ok") else "N/A"
    )
    m5.metric("Next Race Countdown", countdown_value)

    if next_race.get("ok"):
        badge_text = "RACE WEEK" if next_race.get("is_race_week") else "COUNTDOWN ACTIVE"
        st.markdown(
            f"""
            <div style="border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:12px 14px; margin:8px 0 12px 0;
                        background:linear-gradient(135deg, rgba(225,6,0,0.10), rgba(0,210,190,0.08));">
                <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
                    <div>
                        <div style="font-size:12px; color:#b8c2d4; text-transform:uppercase; letter-spacing:1px;">Next Round</div>
                        <div style="font-size:18px; font-weight:800; color:#f2f5fb;">{next_race.get('event')}</div>
                        <div style="font-size:13px; color:#c8d2e2;">{next_race.get('location')} | {format_user_time(next_race.get('race_time'), '%d %b %Y %H:%M:%S')}</div>
                    </div>
                    <div style="padding:4px 10px; border:1px solid rgba(0,210,190,0.35); border-radius:999px; background:rgba(0,210,190,0.14); color:#bff8f2; font-size:12px; font-weight:700;">{badge_text}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### 2026 Calendar & Championship Roadmap")
        if isinstance(cal_df, pd.DataFrame) and not cal_df.empty:
            plot_df = cal_df.copy()
            plot_df["date_local"] = plot_df["race_date"].apply(lambda x: apply_user_utc_offset(x))
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=plot_df["date_local"],
                    y=plot_df["round_idx"],
                    mode="lines+markers+text",
                    text=plot_df["round"],
                    textposition="top center",
                    line=dict(color="#E10600", width=2),
                    marker=dict(size=9, color=np.where(plot_df["is_sprint"].fillna(False), "#00D2BE", "#E10600")),
                    customdata=plot_df[["event", "location", "country", "is_sprint"]],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>Round %{text}<br>%{customdata[1]}, %{customdata[2]}"
                        "<br>Date: %{x|%Y-%m-%d %H:%M}<br>Sprint: %{customdata[3]}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                height=360,
                xaxis_title=f"Race Date (UTC{get_user_utc_offset()})",
                yaxis_title="Round",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(l=10, r=10, t=10, b=40),
            )
            show_plotly_chart(fig, use_container_width=True)
        else:
            st.info("Calendar preview is loading.")
    with right:
        st.markdown("### Power Unit Supplier Mix (2026)")
        if isinstance(tech_df, pd.DataFrame) and not tech_df.empty:
            pu_counts = tech_df.groupby("pu_supplier_2026").size().reset_index(name="Teams")
            fig = px.pie(pu_counts, names="pu_supplier_2026", values="Teams", hole=0.45, title="Teams per supplier")
            fig.update_layout(
                height=360,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(l=10, r=10, t=55, b=10),
            )
            show_plotly_chart(fig, use_container_width=True)
        else:
            st.info("Grid tech preview is loading.")

    st.markdown("### Championship Standings (Pre-season Mode)")
    a1, a2 = st.columns([1, 3])
    with a1:
        if st.button("Refresh Data Services", key="season_overview_2026_refresh_services"):
            st.session_state["season_overview_2026_refresh_nonce"] = st.session_state.get("season_overview_2026_refresh_nonce", 0) + 1
            st.rerun()
    with a2:
        st.caption("When official standings are unavailable, the dashboard shows pre-season readiness and rollout indicators.")

    nonce = int(st.session_state.get("season_overview_2026_refresh_nonce", 0))
    try:
        snapshot = load_2026_api_snapshot(nonce)
    except Exception as e:
        snapshot = {}
        st.warning(f"Could not refresh data services snapshot: {e}")

    drv_df = snapshot.get("jolpica_driver", {}).get("data", pd.DataFrame()) if isinstance(snapshot, dict) else pd.DataFrame()
    ctor_df = snapshot.get("jolpica_constructor", {}).get("data", pd.DataFrame()) if isinstance(snapshot, dict) else pd.DataFrame()

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Drivers Championship**")
        if isinstance(drv_df, pd.DataFrame) and not drv_df.empty:
            st.dataframe(drv_df.head(20), hide_index=True, use_container_width=True)
        elif isinstance(priors_df, pd.DataFrame) and not priors_df.empty:
            pri = priors_df.copy().sort_values("preseason_prior_score", ascending=False)
            fig = px.bar(
                pri.head(10),
                x="preseason_prior_score",
                y="team_norm",
                orientation="h",
                color="preseason_prior_coverage",
                title="Pre-season Readiness Signal (Teams)",
                color_continuous_scale="Tealgrn",
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=360,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(l=10, r=10, t=55, b=20),
            )
            show_plotly_chart(fig, use_container_width=True)
        else:
            st.info("Official standings will populate after race results are published.")
    with b2:
        st.markdown("**Constructors Championship**")
        if isinstance(ctor_df, pd.DataFrame) and not ctor_df.empty:
            st.dataframe(ctor_df.head(11), hide_index=True, use_container_width=True)
        elif isinstance(tech_df, pd.DataFrame) and not tech_df.empty:
            supplier_count = tech_df.groupby("pu_supplier_2026").size().reset_index(name="teams")
            fig = px.bar(supplier_count, x="pu_supplier_2026", y="teams", color="pu_supplier_2026", title="PU Supplier Coverage (Grid Preview)")
            fig.update_layout(
                height=360,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                showlegend=False,
                margin=dict(l=10, r=10, t=55, b=30),
                xaxis_title="",
                yaxis_title="Teams",
            )
            show_plotly_chart(fig, use_container_width=True)
        else:
            st.info("Constructor preview is loading.")

    st.markdown("### Championship Progression Dashboard (Interactive)")
    p1, p2 = st.columns([2, 2])
    with p1:
        if isinstance(cal_df, pd.DataFrame) and not cal_df.empty:
            now = pd.Timestamp.now(tz="UTC")
            table_df = cal_df[["round", "race_date", "event", "location", "country", "is_sprint"]].copy()
            table_df["Countdown"] = table_df["race_date"].apply(lambda x: _countdown_text_precise(x, now))
            table_df["race_date"] = table_df["race_date"].apply(lambda x: format_user_time(x, "%Y-%m-%d %H:%M:%S"))
            table_df = table_df.rename(columns={"race_date": f"Race Date (UTC{get_user_utc_offset()})"})
            st.dataframe(table_df.head(12), hide_index=True, use_container_width=True)
        else:
            st.info("Calendar table is loading.")
    with p2:
        reg_df = F1_2026_REGULATION_METRICS.copy()
        fig = go.Figure()
        for yr, color in [("2025", "#6b7280"), ("2026", "#00D2BE")]:
            fig.add_trace(
                go.Bar(
                    x=reg_df["Metric"],
                    y=reg_df[yr],
                    name=yr,
                    marker_color=color,
                    text=[f"{v}{u}" for v, u in zip(reg_df[yr], reg_df["Unit"])],
                    textposition="outside",
                )
            )
        fig.update_layout(
            barmode="group",
            height=380,
            title="2025 vs 2026 Regulations (Key Metrics)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(l=10, r=10, t=55, b=40),
            xaxis_title="",
        )
        show_plotly_chart(fig, use_container_width=True)


def _render_2026_driver_gallery_fallback() -> None:
    st.info("Driver Profiles is using the 2026 grid tracker until race-result data is available.")
    drivers_df = get_2026_driver_gallery_df()
    if drivers_df.empty:
        st.warning("Driver gallery preview is not available.")
        return

    c1, c2, c3 = st.columns([1.2, 1.2, 2.2])
    with c1:
        teams = sorted([t for t in drivers_df["Team"].dropna().astype(str).unique().tolist() if t])
        team_filter = st.selectbox("Team Filter", ["All Teams"] + teams, key="drivers_2026_team_filter")
    with c2:
        portrait_filter = st.selectbox("Portrait Filter", ["All", "Has Image", "Missing Image"], key="drivers_2026_portrait_filter")
    with c3:
        search = st.text_input("Search Driver", value="", key="drivers_2026_search")

    view_df = drivers_df.copy()
    if team_filter != "All Teams":
        view_df = view_df[view_df["Team"] == team_filter]
    if portrait_filter == "Has Image":
        view_df = view_df[view_df["Has_Image"] == True]
    elif portrait_filter == "Missing Image":
        view_df = view_df[view_df["Has_Image"] == False]
    if search.strip():
        view_df = view_df[view_df["Driver"].astype(str).str.contains(search.strip(), case=False, na=False)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Drivers Shown", len(view_df))
    m2.metric("Teams", view_df["Team"].nunique())
    m3.metric("Portraits", int(view_df["Has_Image"].sum()) if "Has_Image" in view_df.columns else 0)
    m4.metric("PU Suppliers", view_df["PU_Supplier_2026"].nunique() if "PU_Supplier_2026" in view_df.columns else 0)

    selectable_df = view_df.copy()
    if not selectable_df.empty:
        selectable_df["DriverLabel"] = selectable_df.apply(
            lambda r: f"{r['Driver']} ({r['Team']})" if str(r.get("Driver", "")).strip().upper() == "TBA" else str(r.get("Driver", "")),
            axis=1,
        )
    selectable = selectable_df["DriverLabel"].tolist() if not selectable_df.empty else []
    if selectable:
        selected_label = st.selectbox("Select Driver Card", selectable, key="drivers_2026_select_card")
        row = selectable_df[selectable_df["DriverLabel"] == selected_label].iloc[0]
        selected_driver = str(row.get("Driver", "TBA"))
        team_color = row.get("Team_Color") or "#E10600"
        prof = DRIVER_PROFILES.get(selected_driver, {})
        st.markdown(
            f"""
            <div style="padding:14px; border-radius:14px; margin:10px 0 14px 0; border:1px solid {team_color}55;
                        background:linear-gradient(135deg, {team_color}22, rgba(14,17,23,0.95));">
                <div style="font-size:12px; color:#b8c2d4; text-transform:uppercase; letter-spacing:1px;">2026 Driver Tracker</div>
                <div style="font-size:22px; font-weight:800; color:#f2f5fb;">{selected_driver}</div>
                <div style="font-size:13px; color:#c8d2e2;">{row.get('Team', 'Unknown')} | PU: {row.get('PU_Supplier_2026', 'TBA')} | Status: {row.get('Lineup_Status', 'Tracked')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        d1, d2 = st.columns([1, 2])
        with d1:
            if row.get("Image"):
                st.image(row["Image"], use_container_width=True)
            else:
                st.warning("Portrait not available in local profile library.")
        with d2:
            i1, i2, i3, i4 = st.columns(4)
            i1.metric("Number", prof.get("number", row.get("Number") or "TBA"))
            i2.metric("Titles", int(prof.get("titles", 0)) if pd.notna(prof.get("titles", 0)) else 0)
            i3.metric("Wins", int(prof.get("wins", 0)) if pd.notna(prof.get("wins", 0)) else 0)
            i4.metric("Podiums", int(prof.get("podiums", 0)) if pd.notna(prof.get("podiums", 0)) else 0)
            st.markdown(f"**Country:** {row.get('Country') or prof.get('country', 'N/A')}")
            st.markdown(f"**Debut:** {prof.get('debut', row.get('Debut') or 'N/A')}")
            if prof.get("bio"):
                st.caption(prof["bio"])

    if not view_df.empty:
        country_counts = view_df["Country"].fillna("Unknown").value_counts().head(12).reset_index()
        country_counts.columns = ["Country", "Drivers"]
        fig = px.bar(country_counts, x="Drivers", y="Country", orientation="h", title="Driver Pool by Country")
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(l=10, r=10, t=55, b=20),
        )
        show_plotly_chart(fig, use_container_width=True)
    else:
        st.info("No drivers match the current filters.")

    show_cols = [c for c in ["Driver", "Team", "Slot", "Lineup_Status", "PU_Supplier_2026", "Country", "Number", "Has_Image"] if c in view_df.columns]
    st.dataframe(view_df[show_cols], hide_index=True, use_container_width=True)


def _render_2026_constructor_tech_fallback(key_prefix: str = "constructors_2026") -> None:
    st.info("Constructor Analysis is showing the 2026 Grid & Tech preview until race-result data becomes available.")
    tech_df = get_2026_grid_tech_preview_df()
    if tech_df.empty:
        st.warning("Constructor tech preview is not available.")
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        selected_team = st.selectbox(
            "Select Team",
            tech_df["team_display"].tolist(),
            key=f"{key_prefix}_team_selector",
        )
    with c2:
        pu_opts = sorted(tech_df["pu_supplier_2026"].dropna().unique().tolist())
        pu_filter = st.multiselect(
            "PU Supplier Filter",
            pu_opts,
            default=pu_opts,
            key=f"{key_prefix}_pu_filter",
        )
    filtered = tech_df[tech_df["pu_supplier_2026"].isin(pu_filter)] if pu_filter else tech_df.copy()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Teams", len(filtered))
    m2.metric("Suppliers", filtered["pu_supplier_2026"].nunique())
    m3.metric("Tracked Lineups", int(filtered["lineup_status"].astype(str).str.contains("Tracked", case=False, na=False).sum()))
    m4.metric("New Entries", int(filtered["lineup_status"].astype(str).str.contains("New", case=False, na=False).sum()))

    g1, g2 = st.columns(2)
    with g1:
        pu_count = filtered.groupby("pu_supplier_2026").size().reset_index(name="Teams")
        fig = px.bar(pu_count, x="pu_supplier_2026", y="Teams", color="pu_supplier_2026", title="Engine / PU Supplier Matrix")
        fig.update_layout(
            height=330,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
            margin=dict(l=10, r=10, t=55, b=30),
            xaxis_title="",
        )
        show_plotly_chart(fig, use_container_width=True)
    with g2:
        base_count = filtered["base"].fillna("Unknown").astype(str).str.split(",").str[-1].str.strip().value_counts().reset_index()
        base_count.columns = ["Country", "Teams"]
        fig = px.pie(base_count, names="Country", values="Teams", hole=0.45, title="Team Base Countries")
        fig.update_layout(
            height=330,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(l=10, r=10, t=55, b=10),
        )
        show_plotly_chart(fig, use_container_width=True)

    row = tech_df[tech_df["team_display"] == selected_team].iloc[0]
    team_color = row.get("team_color") or "#E10600"
    st.markdown(
        f"""
        <div style="padding:14px; border-radius:14px; border:1px solid {team_color}55; margin:6px 0 12px 0;
                    background:linear-gradient(135deg, {team_color}20, rgba(14,17,23,0.96));">
            <div style="font-size:12px; color:#b8c2d4; text-transform:uppercase; letter-spacing:1px;">2026 Grid & Tech</div>
            <div style="font-size:22px; font-weight:800; color:#f2f5fb;">{row.get('team_display')}</div>
            <div style="font-size:13px; color:#c8d2e2;">{row.get('base')} | {row.get('pu_programme')} | Line-up status: {row.get('lineup_status')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    v1, v2 = st.columns([1, 2])
    with v1:
        if row.get("car_image"):
            st.image(row["car_image"], use_container_width=True)
            st.caption("Latest available car image reference (2026 launch render updates later).")
    with v2:
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Car", row.get("car_name_2026", "TBA"))
        x2.metric("PU Supplier", row.get("pu_supplier_2026", "TBA"))
        x3.metric("Team Principal", row.get("team_principal", "TBA"))
        x4.metric("Driver Slots", len(row.get("drivers_preview", []) or []))
        st.markdown(f"**Driver Tracker:** {row.get('drivers_label', 'TBA')}")
        st.caption(row.get("notes", ""))

    driver_cols = st.columns(max(1, min(3, len(row.get("drivers_preview", []) or []))))
    for idx, driver_name in enumerate(row.get("drivers_preview", []) or []):
        with driver_cols[idx % len(driver_cols)]:
            prof = DRIVER_PROFILES.get(driver_name, {})
            st.markdown(f"**{driver_name}**")
            if prof.get("image_url"):
                st.image(prof["image_url"], use_container_width=True)
            else:
                st.caption("Portrait pending")
            st.caption(f"No. {prof.get('number', 'TBA')} | {prof.get('country', 'N/A')}")

    show_cols = ["team_display", "drivers_label", "pu_supplier_2026", "pu_programme", "car_name_2026", "team_principal", "base", "lineup_status"]
    st.dataframe(filtered[show_cols], hide_index=True, use_container_width=True)


def render_2026_grid_tech_tab() -> None:
    st.header("2026 Grid & Tech Center")
    _render_2026_constructor_tech_fallback(key_prefix="gridtech_2026")


def render_2026_regulations_tab() -> None:
    st.header("2026 Regulations Center")
    st.markdown("Interactive dashboard for the 2026 technical and sporting reset, including PU changes, chassis metrics, and rollout timeline.")

    reg_df = F1_2026_REGULATION_METRICS.copy()
    feat_df = pd.DataFrame(F1_2026_REGULATION_FEATURES)
    updates_df = pd.DataFrame(OFFICIAL_2026_UPDATES).copy()
    if not updates_df.empty:
        updates_df["date"] = pd.to_datetime(updates_df["date"], errors="coerce")
        updates_df = updates_df.sort_values("date", ascending=False)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Key Metrics", len(reg_df))
    m2.metric("Feature Changes", len(feat_df))
    m3.metric("Official Updates", len(updates_df))
    m4.metric("PU Electric Share", "50%")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=reg_df["Metric"], y=reg_df["2025"], name="2025", marker_color="#6b7280"))
        fig.add_trace(go.Bar(x=reg_df["Metric"], y=reg_df["2026"], name="2026", marker_color="#00D2BE"))
        fig.update_layout(
            barmode="group",
            height=360,
            title="Core Package Metrics: 2025 vs 2026",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(l=10, r=10, t=55, b=30),
        )
        show_plotly_chart(fig, use_container_width=True)
    with c2:
        pu_split = pd.DataFrame({"Component": ["Electric", "ICE"], "Share": [50, 50]})
        fig = px.pie(pu_split, names="Component", values="Share", hole=0.5, title="2026 Power Unit Energy Split Target")
        fig.update_layout(
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(l=10, r=10, t=55, b=10),
        )
        show_plotly_chart(fig, use_container_width=True)

    st.markdown("### Regulation Impact Board")
    status_filter = st.multiselect(
        "Filter by Status",
        sorted(feat_df["status"].unique().tolist()),
        default=sorted(feat_df["status"].unique().tolist()),
        key="regs_2026_status_filter",
    )
    filtered_feat = feat_df[feat_df["status"].isin(status_filter)] if status_filter else feat_df.copy()
    st.dataframe(filtered_feat, hide_index=True, use_container_width=True)

    st.markdown("### Official Rollout Timeline")
    if not updates_df.empty:
        cat_opts = sorted([c for c in updates_df["category"].dropna().astype(str).unique().tolist() if c])
        cat_sel = st.multiselect("Category Filter", cat_opts, default=cat_opts, key="regs_2026_cat_filter")
        u = updates_df[updates_df["category"].isin(cat_sel)] if cat_sel else updates_df.copy()
        fig = px.scatter(u, x="date", y="category", color="source", hover_data=["title"], title="2026 Update Timeline (Official Sources)")
        fig.update_layout(
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(l=10, r=10, t=55, b=20),
            xaxis_title="Date",
            yaxis_title="Category",
        )
        show_plotly_chart(fig, use_container_width=True)
        st.dataframe(u[["date", "category", "title", "source", "url"]], hide_index=True, use_container_width=True)
    else:
        st.info("Official update timeline is loading.")


def render_overview_tab(df, total_points_combined=None):
    """Season Overview tab content."""
    st.header(f"{get_active_season_label()} Season Overview")
    
    if df is None or df.empty:
        if get_active_season_year() == 2026:
            _render_2026_season_overview_fallback()
            return
        st.error("No data available")
        return
    
    # Calculate total laps from race data
    total_laps = df['Laps'].sum() if 'Laps' in df.columns else 0
    total_all_points = sum(total_points_combined.values()) if total_points_combined else df['Points'].sum()
    
    # Season metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_races = df['Track'].nunique()
    total_drivers = df['Driver'].nunique()
    total_teams = df['Team'].nunique()
    
    with col1:
        st.metric("Races Completed", total_races)
    with col2:
        st.metric("Drivers", total_drivers)
    with col3:
        st.metric("Teams", total_teams)
    with col4:
        st.metric("Total Race Laps", f"{total_laps:,}")
    with col5:
        st.metric("Total Points", f"{int(total_all_points):,}")
    
    st.divider()
    
    # Championship standings with custom display options
    st.subheader("Championship Standings")
    
    # Display options
    col_opt1, col_opt2, col_opt3 = st.columns([1, 1, 2])
    with col_opt1:
        num_drivers_display = st.slider("Drivers to show", min_value=5, max_value=20, value=10, key="num_drivers")
    with col_opt2:
        num_teams_display = st.slider("Teams to show", min_value=5, max_value=10, value=10, key="num_teams")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Drivers Championship**")
        driver_stats = calculate_driver_stats(df)
        if driver_stats is not None and not driver_stats.empty:
            # Reset index to get Driver as column
            driver_stats_display = driver_stats.reset_index()
            top_drivers = driver_stats_display.nlargest(num_drivers_display, 'Total_Points')
            
            fig = go.Figure()
            # Get team for each driver from original df
            driver_teams = df.groupby('Driver')['Team'].first().to_dict()
            colors = [TEAM_COLORS.get(driver_teams.get(drv, ''), '#666666') for drv in top_drivers['Driver']]
            
            fig.add_trace(go.Bar(
                x=top_drivers['Total_Points'],
                y=top_drivers['Driver'],
                orientation='h',
                marker_color=colors,
                text=top_drivers['Total_Points'].astype(int),
                textposition='outside'
            ))
            fig.update_layout(
                height=max(350, num_drivers_display * 35),
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Points",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=10, r=120, t=10, b=40)
            )
            show_plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Constructors Championship**")
        # Load sprint data and calculate combined standings
        df_sprint = load_sprint_data(get_active_season_year())
        if df_sprint is not None:
            constructor_standings = calculate_combined_constructor_standings(df, df_sprint)
        else:
            # Fallback to race-only stats
            constructor_standings = calculate_team_stats(df)
        
        if constructor_standings is not None and not constructor_standings.empty:
            # Reset index to get Team as column
            team_stats_display = constructor_standings.reset_index()
            team_stats_display = team_stats_display.sort_values('Total_Points', ascending=False).head(num_teams_display)
            
            fig = go.Figure()
            colors = [TEAM_COLORS.get(team, '#666666') for team in team_stats_display['Team']]
            
            fig.add_trace(go.Bar(
                x=team_stats_display['Total_Points'],
                y=team_stats_display['Team'],
                orientation='h',
                marker_color=colors,
                text=team_stats_display['Total_Points'].astype(int),
                textposition='outside'
            ))
            fig.update_layout(
                height=max(350, num_teams_display * 40),
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Points",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=10, r=120, t=10, b=40)
            )
            show_plotly_chart(fig, use_container_width=True)
    
    # Points progression
    st.subheader("Championship Points Progression")
    
    races = df['Track'].unique()
    # Use driver_stats with reset index
    top_5_drivers = driver_stats_display.nlargest(5, 'Total_Points')['Driver'].tolist() if driver_stats is not None else []
    
    if top_5_drivers:
        fig = go.Figure()
        
        for driver in top_5_drivers:
            driver_data = df[df['Driver'] == driver].copy()
            driver_data = driver_data.sort_values('Track')
            cumsum_points = driver_data['Points'].cumsum()
            team = driver_data['Team'].iloc[0] if not driver_data.empty else ''
            color = TEAM_COLORS.get(team, '#666666')
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(cumsum_points) + 1)),
                y=cumsum_points,
                name=driver,
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            height=400,
            xaxis_title="Race Number",
            yaxis_title="Cumulative Points",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        show_plotly_chart(fig, use_container_width=True)


def render_drivers_tab(df, total_points_combined=None):
    """Driver Profiles tab content."""
    st.header("Driver Profiles")
    
    if df is None or df.empty:
        if get_active_season_year() == 2026:
            _render_2026_driver_gallery_fallback()
            return
        st.error("No data available")
        return
    
    active_year = get_active_season_year()
    drivers_available = sorted([d for d in df['Driver'].unique().tolist() if d in DRIVER_PROFILES])
    
    col_sel, _ = st.columns([1, 3])
    with col_sel:
        selected_driver = st.selectbox("Select Driver", drivers_available, key="driver_profile_selector")
    
    if selected_driver:
        profile = DRIVER_PROFILES.get(selected_driver, {})
        driver_df = df[df['Driver'] == selected_driver]
        team = driver_df['Team'].iloc[0] if not driver_df.empty else "Unknown"
        team_color = TEAM_COLORS.get(team, '#666666')
        
        # --- Hero Section ---
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, {team_color} 0%, #0E1117 100%); padding: 2px; border-radius: 10px; margin-bottom: 20px;">
            <div style="background: #0E1117; border-radius: 8px; padding: 20px;">
                <h1 style="color: white; margin: 0; font-size: 3rem; text-transform: uppercase; letter-spacing: 2px;">
                    <span style="color: {team_color};">{profile.get('number', '')}</span> {selected_driver}
                </h1>
                <h3 style="color: #888; margin: 0; font-weight: 300;">{team}</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # --- Profile Content ---
        col_profile, col_stats = st.columns([1, 2])
        
        with col_profile:
            if profile.get('image_url'):
                st.image(profile['image_url'], use_container_width=True)
            else:
                st.warning("Driver image not available")
                
            # Biographical Data Table
            st.markdown("### Biography")
            st.markdown(f"""
            **Nationality:** {profile.get('country', 'N/A')}  
            **Debut:** {profile.get('debut', 'N/A')}  
            **Seasons:** {active_year - profile.get('debut', active_year) if isinstance(profile.get('debut'), int) else 'N/A'}
            """)
            st.info(profile.get('bio', ''))

            # Social Media
            st.markdown("### Connect")
            cols_social = st.columns(len(SOCIAL_MEDIA_CONFIG))
            for idx, (platform, conf) in enumerate(SOCIAL_MEDIA_CONFIG.items()):
                handle = profile.get(platform)
                if handle:
                    url = f"{conf['url_prefix']}{handle.replace('@', '')}"
                    with cols_social[idx]:
                         st.markdown(f"[{platform.capitalize()}]({url})")
            
            # Career Stats
            if 'titles' in profile:
                st.markdown("### Career Highlights")
                c_data = {
                    "Metric": ["World Titles", "Grand Prix Wins", "Pole Positions", "Podiums"],
                    "Value": [
                        profile.get('titles', 0),
                        profile.get('wins', 0),
                        profile.get('poles', 0),
                        profile.get('podiums', 0)
                    ]
                }
                st.dataframe(pd.DataFrame(c_data), hide_index=True, use_container_width=True)

        with col_stats:
            # Active season performance
            st.subheader(f"{active_year} Season Performance")
            
            # Stats Calculation
            race_points = driver_df['Points'].sum()
            total_points = total_points_combined.get(selected_driver, race_points) if total_points_combined else race_points
            wins = len(driver_df[driver_df['Position'] == 1])
            podiums = len(driver_df[driver_df['Position'] <= 3])
            avg_pos = driver_df['Position'].mean()
            best_finish = driver_df['Position'].min() if not driver_df.empty else 0
            
            # Custom Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"<h2 style='text-align:center; color:{team_color}'>{int(total_points)}</h2><p style='text-align:center'>POINTS</p>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<h2 style='text-align:center; color:white'>{wins}</h2><p style='text-align:center'>WINS</p>", unsafe_allow_html=True)
            with m3:
                st.markdown(f"<h2 style='text-align:center; color:white'>{podiums}</h2><p style='text-align:center'>PODIUMS</p>", unsafe_allow_html=True)
            with m4:
                st.markdown(f"<h2 style='text-align:center; color:white'>P{int(best_finish) if not pd.isna(best_finish) else '-'}</h2><p style='text-align:center'>BEST</p>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Recent Form Chart
            st.subheader("Recent Form")
            
            fig = go.Figure()
            
            # Position Trend
            fig.add_trace(go.Scatter(
                x=driver_df['Track'],
                y=driver_df['Position'],
                mode='lines+markers',
                name='Finish Position',
                line=dict(color=team_color, width=3),
                marker=dict(size=10, color='white', line=dict(width=2, color=team_color))
            ))
            
            # Add Qualifying comparison if available (simplified)
            if 'Starting Grid' in driver_df.columns:
                fig.add_trace(go.Scatter(
                    x=driver_df['Track'],
                    y=driver_df['Starting Grid'],
                    mode='markers',
                    name='Grid Position',
                    marker=dict(size=8, symbol='diamond', color='#888888')
                ))
            
            fig.update_layout(
                yaxis=dict(autorange='reversed', title='Position', gridcolor='#333'),
                xaxis=dict(showgrid=False),
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(orientation='h', y=1.1),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            show_plotly_chart(fig, use_container_width=True)
            
            # Results Table
            with st.expander(f"Full {active_year} Results"):
                res_cols = ['Track', 'Starting Grid', 'Position', 'Points', 'Laps']
                valid_cols = [c for c in res_cols if c in driver_df.columns]
                st.dataframe(driver_df[valid_cols], hide_index=True, use_container_width=True)


def render_teams_tab(df):
    """Constructor Analysis tab content."""
    st.header("Constructor Analysis")
    
    if df is None or df.empty:
        if get_active_season_year() == 2026:
            _render_2026_constructor_tech_fallback(key_prefix="constructors_fallback_2026")
            return
        st.error("No data available")
        return
    
    # 2025 Team Data with car images and specs
    # Keys match CSV team names exactly for proper matching
    TEAM_DATA = {
        'McLaren': {
            'full_name': 'McLaren Formula 1 Team',
            'base': 'Woking, United Kingdom',
            'team_principal': 'Andrea Stella',
            'technical_director': 'Peter Prodromou',
            'chassis': 'MCL39',
            'power_unit': 'Mercedes M15 E Performance',
            'first_entry': 1966,
            'championships': 8,
            'constructor_titles': 8,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/mclaren.png',
        },
        'McLaren Mercedes': {
            'full_name': 'McLaren Formula 1 Team',
            'base': 'Woking, United Kingdom',
            'team_principal': 'Andrea Stella',
            'technical_director': 'Peter Prodromou',
            'chassis': 'MCL39',
            'power_unit': 'Mercedes M15 E Performance',
            'first_entry': 1966,
            'championships': 8,
            'constructor_titles': 8,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/mclaren.png',
        },
        'Red Bull Racing Honda RBPT': {
            'full_name': 'Oracle Red Bull Racing',
            'base': 'Milton Keynes, United Kingdom',
            'team_principal': 'Christian Horner',
            'technical_director': 'Pierre Wache',
            'chassis': 'RB21',
            'power_unit': 'Honda RBPT',
            'first_entry': 2005,
            'championships': 7,
            'constructor_titles': 6,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/red-bull-racing.png',
        },
        'Red Bull Racing Honda EBPT': {
            'full_name': 'Oracle Red Bull Racing',
            'base': 'Milton Keynes, United Kingdom',
            'team_principal': 'Christian Horner',
            'technical_director': 'Pierre Wache',
            'chassis': 'RB21',
            'power_unit': 'Honda RBPT',
            'first_entry': 2005,
            'championships': 7,
            'constructor_titles': 6,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/red-bull-racing.png',
        },
        'Ferrari': {
            'full_name': 'Scuderia Ferrari HP',
            'base': 'Maranello, Italy',
            'team_principal': 'Frederic Vasseur',
            'technical_director': 'Loic Serra',
            'chassis': 'SF-25',
            'power_unit': 'Ferrari 067',
            'first_entry': 1950,
            'championships': 15,
            'constructor_titles': 16,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/ferrari.png',
        },
        'Mercedes': {
            'full_name': 'Mercedes-AMG Petronas F1 Team',
            'base': 'Brackley, United Kingdom',
            'team_principal': 'Toto Wolff',
            'technical_director': 'James Allison',
            'chassis': 'W16',
            'power_unit': 'Mercedes M15 E Performance',
            'first_entry': 2010,
            'championships': 7,
            'constructor_titles': 8,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/mercedes.png',
        },
        'Aston Martin Aramco Mercedes': {
            'full_name': 'Aston Martin Aramco F1 Team',
            'base': 'Silverstone, United Kingdom',
            'team_principal': 'Mike Krack',
            'technical_director': 'Dan Fallows',
            'chassis': 'AMR25',
            'power_unit': 'Mercedes M15 E Performance',
            'first_entry': 2021,
            'championships': 0,
            'constructor_titles': 0,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/aston-martin.png',
        },
        'Alpine Renault': {
            'full_name': 'BWT Alpine F1 Team',
            'base': 'Enstone, United Kingdom',
            'team_principal': 'Oliver Oakes',
            'technical_director': 'David Sanchez',
            'chassis': 'A525',
            'power_unit': 'Renault E-Tech RE25',
            'first_entry': 2021,
            'championships': 0,
            'constructor_titles': 2,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/alpine.png',
        },
        'Williams Mercedes': {
            'full_name': 'Williams Racing',
            'base': 'Grove, United Kingdom',
            'team_principal': 'James Vowles',
            'technical_director': 'Pat Fry',
            'chassis': 'FW47',
            'power_unit': 'Mercedes M15 E Performance',
            'first_entry': 1978,
            'championships': 7,
            'constructor_titles': 9,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/williams.png',
        },
        'Racing Bulls Honda RBPT': {
            'full_name': 'Visa Cash App Racing Bulls',
            'base': 'Faenza, Italy',
            'team_principal': 'Laurent Mekies',
            'technical_director': 'Jody Egginton',
            'chassis': 'VCARB 02',
            'power_unit': 'Honda RBPT',
            'first_entry': 2006,
            'championships': 0,
            'constructor_titles': 0,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/rb.png',
        },
        'Kick Sauber Ferrari': {
            'full_name': 'Stake F1 Team Kick Sauber',
            'base': 'Hinwil, Switzerland',
            'team_principal': 'Mattia Binotto',
            'technical_director': 'James Key',
            'chassis': 'C45',
            'power_unit': 'Ferrari 067',
            'first_entry': 1993,
            'championships': 0,
            'constructor_titles': 0,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/kick-sauber.png',
        },
        'Haas Ferrari': {
            'full_name': 'MoneyGram Haas F1 Team',
            'base': 'Kannapolis, USA',
            'team_principal': 'Ayao Komatsu',
            'technical_director': 'Andrea De Zordo',
            'chassis': 'VF-25',
            'power_unit': 'Ferrari 067',
            'first_entry': 2016,
            'championships': 0,
            'constructor_titles': 0,
            'car_image': 'https://media.formula1.com/d_team_car_fallback_image.png/content/dam/fom-website/teams/2024/haas.png',
        },
    }
    
    # Team selector with session state
    teams = df['Team'].unique().tolist()
    selected_team = st.selectbox(
        "Select Team", 
        sorted(teams),
        key="team_profile_selector"
    )
    
    if selected_team:
        team_df = df[df['Team'] == selected_team]
        team_color = TEAM_COLORS.get(selected_team, '#666666')
        
        # Find matching team data - prioritize exact matches
        team_info = None
        
        # First try exact match
        for key, data in TEAM_DATA.items():
            if key.lower() == selected_team.lower():
                team_info = data
                break
        
        # If no exact match, try to find best partial match
        # But avoid "Ferrari" matching "Haas Ferrari" - use startswith instead
        if team_info is None:
            for key, data in TEAM_DATA.items():
                # Check if selected_team starts with key or key starts with selected_team
                if selected_team.lower().startswith(key.lower()) or key.lower().startswith(selected_team.lower()):
                    team_info = data
                    break
        
        # Final fallback - word-based match
        if team_info is None:
            selected_words = set(selected_team.lower().split())
            for key, data in TEAM_DATA.items():
                key_words = set(key.lower().split())
                # Match if the key's main word is in selected_team
                if key_words & selected_words:
                    team_info = data
                    break
        
        if team_info is None:
            team_info = {}
        
        st.divider()
        
        # Team header with car image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if team_info.get('car_image'):
                st.image(team_info['car_image'], use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <div style="padding: 20px; background: linear-gradient(135deg, {team_color}33 0%, {team_color}11 100%); border-radius: 15px; border: 2px solid {team_color};">
                <h2 style="color: {team_color}; margin: 0;">{team_info.get('full_name', selected_team)}</h2>
                <p style="color: #888; margin-top: 5px;">{team_info.get('base', 'Unknown')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Team Specifications
        st.subheader(f"{get_active_season_label()} Technical Specifications")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            **Chassis**  
            {team_info.get('chassis', 'N/A')}
            """)
        with col2:
            st.markdown(f"""
            **Power Unit**  
            {team_info.get('power_unit', 'N/A')}
            """)
        with col3:
            st.markdown(f"""
            **Team Principal**  
            {team_info.get('team_principal', 'N/A')}
            """)
        with col4:
            st.markdown(f"""
            **Technical Director**  
            {team_info.get('technical_director', 'N/A')}
            """)
        
        # Team history
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("First Entry", team_info.get('first_entry', 'N/A'))
        with col2:
            st.metric("Driver Titles", team_info.get('championships', 0))
        with col3:
            st.metric("Constructor Titles", team_info.get('constructor_titles', 0))
        with col4:
            active_year = get_active_season_year()
            st.metric("Years in F1", active_year - team_info.get('first_entry', active_year) if team_info.get('first_entry') else 'N/A')
        
        st.divider()
        
        # Active season stats
        st.subheader(f"{get_active_season_label()} Season Performance")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_points = team_df['Points'].sum()
        wins = len(team_df[team_df['Position'] == 1])
        podiums = len(team_df[team_df['Position'] <= 3])
        avg_pos = team_df['Position'].mean()
        drivers = team_df['Driver'].unique()
        
        with col1:
            st.metric("Total Points", int(total_points))
        with col2:
            st.metric("Wins", wins)
        with col3:
            st.metric("Podiums", podiums)
        with col4:
            st.metric("Avg Position", f"{avg_pos:.1f}")
        with col5:
            st.metric("Drivers", len(drivers))
        
        st.divider()
        
        # Driver comparison
        st.subheader("Driver Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Points comparison - fix color issue
            driver_points = team_df.groupby('Driver')['Points'].sum().reset_index()
            
            # Create lighter version of team color for second driver
            def lighten_color(hex_color, factor=0.5):
                hex_color = hex_color.lstrip('#')
                r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
                r = int(r + (255 - r) * factor)
                g = int(g + (255 - g) * factor)
                b = int(b + (255 - b) * factor)
                return f'#{r:02x}{g:02x}{b:02x}'
            
            colors = [team_color, lighten_color(team_color)]
            
            fig = go.Figure(data=[go.Pie(
                labels=driver_points['Driver'],
                values=driver_points['Points'],
                marker_colors=colors[:len(driver_points)],
                hole=0.4
            )])
            fig.update_layout(
                title="Points Distribution",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            show_plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average position comparison - fix color issue
            driver_avg = team_df.groupby('Driver')['Position'].mean().reset_index()
            colors = [team_color, lighten_color(team_color)]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=driver_avg['Driver'],
                y=driver_avg['Position'],
                marker_color=colors[:len(driver_avg)],
                text=[f"{p:.1f}" for p in driver_avg['Position']],
                textposition='outside'
            ))
            fig.update_layout(
                title="Average Race Position",
                yaxis=dict(autorange='reversed', title='Position'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            show_plotly_chart(fig, use_container_width=True)
        
        # Points progression
        st.subheader("Team Points Progression")
        
        fig = go.Figure()
        colors = [team_color, lighten_color(team_color)]
        
        for i, driver in enumerate(drivers):
            driver_data = team_df[team_df['Driver'] == driver].copy()
            driver_data = driver_data.sort_values('Track')
            cumsum = driver_data['Points'].cumsum()
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(cumsum) + 1)),
                y=cumsum,
                name=driver,
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            xaxis_title="Race Number",
            yaxis_title="Cumulative Points",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        show_plotly_chart(fig, use_container_width=True)


def render_weekend_dashboard_tab() -> None:
    st.header("Weekend Dashboard")
    st.markdown("Interactive session intelligence dashboard using FastF1 (weather, tyre strategy, speeds, sectors, and control messages when available).")

    races = get_season_race_choices()
    if not races:
        st.info("Race list is not available yet.")
        return

    s1, s2, s3 = st.columns([2, 1, 1])
    with s1:
        gp = st.selectbox("Select Grand Prix", races, key="weekend_dash_gp")
    with s2:
        sess = st.selectbox("Session", ["Race", "Qualifying", "Sprint", "Practice 1", "Practice 2", "Practice 3"], key="weekend_dash_session")
    with s3:
        load_clicked = st.button("Load Weekend Data", key="weekend_dash_reload", type="primary")

    active_year = get_active_season_year()
    load_sig = f"{active_year}|{gp}|{sess}"
    if load_clicked:
        st.session_state["weekend_dash_loaded_sig"] = load_sig

    event_row = _find_schedule_event_row(active_year, gp)
    if event_row is not None:
        st.caption(f"Session clock displays user time in UTC{get_user_utc_offset()} with seconds.")
        now = pd.Timestamp.now(tz="UTC")
        sched_rows = []
        for key in ["Session1", "Session2", "Session3", "Session4", "Session5"]:
            date_key = f"{key}Date"
            if key in event_row.index and date_key in event_row.index and pd.notna(event_row[date_key]):
                dt_val = pd.to_datetime(event_row[date_key], errors="coerce", utc=True)
                if pd.isna(dt_val):
                    continue
                status = "Upcoming"
                if dt_val <= now <= dt_val + timedelta(hours=2.5):
                    status = "Live"
                elif now > dt_val + timedelta(hours=2.5):
                    status = "Completed"
                sched_rows.append(
                    {
                        "Session": str(event_row.get(key, key)),
                        f"Start (UTC{get_user_utc_offset()})": format_user_time(dt_val, "%d %b %Y %H:%M:%S"),
                        "Countdown": _countdown_text_precise(dt_val, now),
                        "Status": status,
                    }
                )
        if sched_rows:
            st.dataframe(pd.DataFrame(sched_rows), hide_index=True, use_container_width=True)

    if st.session_state.get("weekend_dash_loaded_sig") != load_sig:
        st.info("Select a race/session and click `Load Weekend Data` to fetch the weekend dashboard.")
        return

    with st.spinner("Loading FastF1 session data..."):
        session = get_fastf1_session_state_cached(
            year=active_year,
            race=gp,
            session_type=sess,
            load_telemetry=False,
            cache_namespace="weekend_dash_fastf1",
            force_reload=bool(load_clicked),
        )

    if session is None:
        st.warning("Selected session data is not available yet. Calendar and countdown remain active.")
        return

    info = get_session_info(session)
    laps_df = session.laps.copy() if hasattr(session, "laps") and session.laps is not None else pd.DataFrame()
    results_df = get_race_results(session)
    weather_df = session.weather_data.copy() if hasattr(session, "weather_data") and session.weather_data is not None else pd.DataFrame()
    stints_df = get_tyre_stints(session)
    top_speeds_df = get_top_speeds(session)
    best_sectors_df = get_best_sectors(session)
    pit_df = get_pit_stops(session)
    track_status_df = get_track_status(session)
    race_control_df = get_race_control_messages(session)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Event", str(info.get("event_name", gp)))
    m2.metric("Session", str(info.get("session_name", sess)))
    m3.metric("Drivers", int(info.get("num_drivers", len(getattr(session, "drivers", []) or []))))
    m4.metric("Laps Rows", len(laps_df) if isinstance(laps_df, pd.DataFrame) else 0)
    m5.metric("Weather Rows", len(weather_df) if isinstance(weather_df, pd.DataFrame) else 0)

    _render_data_availability_badges([
        ("Results", isinstance(results_df, pd.DataFrame) and not results_df.empty, "FastF1 session results"),
        ("Weather", isinstance(weather_df, pd.DataFrame) and not weather_df.empty, "session.weather_data"),
        ("Tyre Stints", isinstance(stints_df, pd.DataFrame) and not stints_df.empty, "Computed from laps"),
        ("Pit Stops", isinstance(pit_df, pd.DataFrame) and not pit_df.empty, "Pit in/out detection from laps"),
        ("Top Speeds", isinstance(top_speeds_df, pd.DataFrame) and not top_speeds_df.empty, "Speed traps in lap data"),
        ("Best Sectors", isinstance(best_sectors_df, pd.DataFrame) and not best_sectors_df.empty, "Best sector extraction"),
        ("Track Status", isinstance(track_status_df, pd.DataFrame) and not track_status_df.empty, "Flags / SC / VSC events"),
        ("Race Control", isinstance(race_control_df, pd.DataFrame) and not race_control_df.empty, "Race control messages feed"),
    ])

    dash_tabs = st.tabs(["Conditions", "Strategy", "Speed & Sectors", "Control", "Raw Tables"])

    with dash_tabs[0]:
        st.subheader("Weather & Session Conditions")
        weather_summary = get_weather_summary(session)
        w1, w2, w3, w4, w5 = st.columns(5)
        w1.metric("Conditions", weather_summary.get("conditions", "N/A") if weather_summary else "N/A")
        w2.metric("Air Temp", f"{weather_summary.get('air_temp_avg', 'N/A')}°C" if weather_summary.get("available") else "N/A")
        w3.metric("Track Temp", f"{weather_summary.get('track_temp_avg', 'N/A')}°C" if weather_summary.get("available") else "N/A")
        w4.metric("Humidity", f"{weather_summary.get('humidity_avg', 'N/A')}%" if weather_summary.get("available") else "N/A")
        w5.metric("Rain", "Yes" if weather_summary.get("rainfall") else "No")

        if isinstance(weather_df, pd.DataFrame) and not weather_df.empty:
            wf = weather_df.copy()
            wf["TimeLabel"] = wf["Time"].astype(str) if "Time" in wf.columns else np.arange(len(wf)).astype(str)
            metric_options = [c for c in ["AirTemp", "TrackTemp", "Humidity", "WindSpeed"] if c in wf.columns]
            selected_metrics = st.multiselect(
                "Weather Metrics",
                metric_options,
                default=metric_options[: min(2, len(metric_options))],
                key="weekend_dash_weather_metrics",
            )
            if selected_metrics:
                fig = go.Figure()
                for metric in selected_metrics:
                    fig.add_trace(
                        go.Scatter(
                            x=wf["TimeLabel"],
                            y=pd.to_numeric(wf[metric], errors="coerce"),
                            mode="lines",
                            name=metric,
                        )
                    )
                fig.update_layout(
                    height=360,
                    xaxis_title="Session Time",
                    yaxis_title="Value",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    margin=dict(l=10, r=10, t=20, b=40),
                )
                show_plotly_chart(fig, use_container_width=True)
        else:
            st.info("Weather feed is not available for this session yet.")

    with dash_tabs[1]:
        st.subheader("Tyre & Pit Strategy")
        if isinstance(stints_df, pd.DataFrame) and not stints_df.empty and "Laps" in stints_df.columns:
            fig = px.bar(
                stints_df,
                x="Laps",
                y="Driver",
                color="Compound" if "Compound" in stints_df.columns else None,
                orientation="h",
                hover_data=[c for c in ["Stint", "StartLap", "EndLap"] if c in stints_df.columns],
                title="Tyre Stints by Driver",
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=max(360, 24 * int(stints_df["Driver"].nunique())),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(l=10, r=10, t=55, b=20),
            )
            show_plotly_chart(fig, use_container_width=True)
            st.dataframe(stints_df.head(100), hide_index=True, use_container_width=True)
        else:
            st.info("Tyre stint data is not available for this session.")

        st.markdown("#### Pit Stop Analysis")
        if isinstance(pit_df, pd.DataFrame) and not pit_df.empty:
            p = pit_df.copy()
            duration_col = next((c for c in ["PitLaneTime", "PitStopTime", "Duration", "StopTime"] if c in p.columns), None)
            if duration_col:
                p[duration_col] = pd.to_numeric(p[duration_col], errors="coerce")
                p = p.dropna(subset=[duration_col])
                if not p.empty:
                    fig = px.scatter(
                        p,
                        x="Lap" if "Lap" in p.columns else p.index,
                        y=duration_col,
                        color="Driver" if "Driver" in p.columns else None,
                        title=f"Pit Stop Timing ({duration_col})",
                    )
                    fig.update_layout(
                        height=320,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                        margin=dict(l=10, r=10, t=55, b=20),
                    )
                    show_plotly_chart(fig, use_container_width=True)
            st.dataframe(pit_df.head(50), hide_index=True, use_container_width=True)
        else:
            st.info("No pit stop events detected (or not available for this session type).")

    with dash_tabs[2]:
        st.subheader("Speed & Sectors")
        left, right = st.columns(2)
        with left:
            if isinstance(top_speeds_df, pd.DataFrame) and not top_speeds_df.empty:
                speed_cols = [c for c in top_speeds_df.columns if c.startswith("Max_")]
                metric_col = st.selectbox("Speed Metric", speed_cols, key="weekend_dash_speed_metric") if speed_cols else None
                if metric_col:
                    fig = px.bar(
                        top_speeds_df.sort_values(metric_col, ascending=False),
                        x="Driver",
                        y=metric_col,
                        color=metric_col,
                        title=f"Top Speeds ({metric_col})",
                    )
                    fig.update_layout(
                        height=340,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                        margin=dict(l=10, r=10, t=55, b=30),
                        xaxis_title="",
                    )
                    show_plotly_chart(fig, use_container_width=True)
                st.dataframe(top_speeds_df, hide_index=True, use_container_width=True)
            else:
                st.info("Speed trap data is not available.")
        with right:
            if isinstance(best_sectors_df, pd.DataFrame) and not best_sectors_df.empty:
                heat_cols = [c for c in ["Best_Sector1", "Best_Sector2", "Best_Sector3", "FastestLap", "TheoreticalBest"] if c in best_sectors_df.columns]
                if heat_cols:
                    heat_df = best_sectors_df[["Driver"] + heat_cols].copy().set_index("Driver")
                    fig = px.imshow(heat_df, aspect="auto", color_continuous_scale="Turbo", title="Best Sector / Lap Heatmap (s)")
                    fig.update_layout(
                        height=340,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                        margin=dict(l=10, r=10, t=55, b=20),
                    )
                    show_plotly_chart(fig, use_container_width=True)
                st.dataframe(best_sectors_df, hide_index=True, use_container_width=True)
            else:
                st.info("Sector timing detail is not available.")

    with dash_tabs[3]:
        st.subheader("Track Status & Race Control")
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**Track Status Events**")
            if isinstance(track_status_df, pd.DataFrame) and not track_status_df.empty:
                st.dataframe(track_status_df.tail(50), hide_index=True, use_container_width=True)
            else:
                st.info("No track status events in current feed.")
        with c_right:
            st.markdown("**Race Control Messages**")
            if isinstance(race_control_df, pd.DataFrame) and not race_control_df.empty:
                st.dataframe(race_control_df.tail(50), hide_index=True, use_container_width=True)
            else:
                st.info("No race control messages in current feed.")

    with dash_tabs[4]:
        st.subheader("Raw Session Tables")
        choice = st.selectbox(
            "Table",
            ["Results", "Laps", "Weather", "Tyre Stints", "Pit Stops", "Top Speeds", "Best Sectors"],
            key="weekend_dash_raw_choice",
        )
        mapping = {
            "Results": results_df,
            "Laps": laps_df,
            "Weather": weather_df,
            "Tyre Stints": stints_df,
            "Pit Stops": pit_df,
            "Top Speeds": top_speeds_df,
            "Best Sectors": best_sectors_df,
        }
        out = mapping.get(choice, pd.DataFrame())
        if isinstance(out, pd.DataFrame) and not out.empty:
            st.dataframe(out.head(300), hide_index=True, use_container_width=True)
        else:
            st.info(f"{choice} table is not available for this session.")


def render_race_detail_tab(df):
    """Race Weekend Details tab content."""
    st.header("Race Weekend Details")
    
    # Race selector - 2025 only
    available_races = get_season_race_choices()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_race = st.selectbox(
            "Select Grand Prix", 
            available_races,
            key="race_detail_gp_selector"
        )
    with col2:
        session_type = st.selectbox(
            "Session", 
            ["Race", "Qualifying", "Sprint", "Practice 1", "Practice 2", "Practice 3"],
            key="race_detail_session_selector"
        )
    with col3:
        load_details_clicked = st.button("Load Session Details", key="race_detail_load_btn", type="primary")
    
    if selected_race:
        st.divider()
        active_year = get_active_season_year()
        detail_sig = f"{active_year}|{selected_race}|{session_type}"
        if load_details_clicked:
            st.session_state["race_detail_loaded_sig"] = detail_sig

        if st.session_state.get("race_detail_loaded_sig") != detail_sig:
            event_row = _find_schedule_event_row(active_year, selected_race)
            if event_row is not None:
                now = pd.Timestamp.now(tz="UTC")
                sched_rows = []
                for key in ["Session1", "Session2", "Session3", "Session4", "Session5"]:
                    date_key = f"{key}Date"
                    if key in event_row.index and date_key in event_row.index and pd.notna(event_row[date_key]):
                        dt_val = pd.to_datetime(event_row[date_key], errors="coerce", utc=True)
                        if pd.isna(dt_val):
                            continue
                        sched_rows.append({
                            "Session": str(event_row.get(key, key)),
                            f"Start (UTC{get_user_utc_offset()})": format_user_time(dt_val, "%d %b %Y %H:%M:%S"),
                            "Countdown": _countdown_text_precise(dt_val, now),
                        })
                if sched_rows:
                    with st.expander("Weekend Session Clock", expanded=False):
                        st.dataframe(pd.DataFrame(sched_rows), hide_index=True, use_container_width=True)
            st.info("Select a race/session and click `Load Session Details` to fetch weekend data.")
            return
        
        # Load FastF1 data
        with st.spinner("Loading session data..."):
            session = get_fastf1_session_state_cached(
                year=active_year,
                race=selected_race,
                session_type=session_type,
                load_telemetry=False,
                cache_namespace="race_detail_fastf1",
                force_reload=bool(load_details_clicked),
            )
        
        if session is None:
            st.error(f"Could not load session data for {selected_race} - {session_type}")
            return
        
        st.divider()
        
        # Weather Panel
        st.subheader("Weather Conditions")
        
        weather = get_weather_summary(session)
        
        if weather and weather.get('available', False):
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                air_temp = weather.get('air_temp_avg')
                st.metric("Air Temp", f"{air_temp:.1f}°C" if air_temp is not None else "N/A")
            with col2:
                track_temp = weather.get('track_temp_avg')
                st.metric("Track Temp", f"{track_temp:.1f}°C" if track_temp is not None else "N/A")
            with col3:
                humidity = weather.get('humidity_avg')
                st.metric("Humidity", f"{humidity:.0f}%" if humidity is not None else "N/A")
            with col4:
                wind = weather.get('wind_speed_avg')
                st.metric("Wind Speed", f"{wind:.1f} km/h" if wind is not None else "N/A")
            with col5:
                rain_status = "Yes" if weather.get('rainfall', False) else "No" # Removed emojis
                st.metric("Rainfall", rain_status)
        else:
            st.info("Weather data not available for this session")
        
        st.divider()
        
        # Tabs for different data views - expanded with more FastF1 features
        race_tabs = st.tabs([
            "Results", 
            "Pit Stops", 
            "Tyre Strategy", 
            "Track Status", 
            "Sector Times",
            "Speed Traps",
            "Position Changes",
            "Lap Times"
        ])
        
        with race_tabs[0]:
            # Race Results
            st.subheader("Session Results")
            try:
                results = session.results
                if results is not None and not results.empty:
                    preferred_cols = ['Position', 'Abbreviation', 'FullName', 'TeamName', 'Time', 'Status']
                    available_cols = [c for c in preferred_cols if c in results.columns]
                    display_results = results[available_cols].copy()
                    
                    # Apply global formatter
                    if 'Time' in display_results.columns:
                        display_results['Time'] = display_results['Time'].apply(format_f1_time)
                    
                    display_results = display_results.rename(columns={
                        'Position': 'Pos',
                        'Abbreviation': 'Code',
                        'FullName': 'Driver',
                        'TeamName': 'Team'
                    })
                    st.dataframe(display_results, use_container_width=True, hide_index=True)
                else:
                    st.info("Results are not available for this session yet.")
            except Exception as e:
                st.error(f"Could not load results: {e}")
        
        with race_tabs[1]:
            # Pit Stops
            st.subheader("Pit Stop Analysis")
            pit_stops = get_pit_stops(session)
            
            if pit_stops is not None and not pit_stops.empty:
                pit_df = pit_stops.copy()
                duration_col = next((c for c in ['PitTime', 'PitLaneTime', 'PitStopTime', 'Duration', 'StopTime'] if c in pit_df.columns), None)
                lap_col = 'Lap' if 'Lap' in pit_df.columns else None
                stop_col = 'Stop' if 'Stop' in pit_df.columns else ('StopNumber' if 'StopNumber' in pit_df.columns else None)

                if duration_col is None:
                    st.info("Pit stop events found, but timing duration is not available for this session.")
                    st.dataframe(pit_df, use_container_width=True, hide_index=True)
                else:
                    pit_df[duration_col] = pd.to_numeric(pit_df[duration_col], errors='coerce')
                    pit_df = pit_df.dropna(subset=[duration_col])
                    if pit_df.empty:
                        st.info("Pit stop duration values are not available.")
                    else:
                        # Summary metrics.
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Pit Stops", len(pit_df))
                        with col2:
                            avg_time = pit_df[duration_col].mean()
                            st.metric("Average Pit Time", f"{avg_time:.1f}s")
                        with col3:
                            fastest = pit_df[duration_col].min()
                            fastest_driver = pit_df[pit_df[duration_col] == fastest]['Driver'].iloc[0] if 'Driver' in pit_df.columns else 'N/A'
                            st.metric("Fastest Stop", f"{fastest:.1f}s ({fastest_driver})")
                        with col4:
                            slowest = pit_df[duration_col].max()
                            st.metric("Slowest Stop", f"{slowest:.1f}s")
                        
                        st.divider()
                        
                        # Pit stop chart - horizontal bar chart grouped by driver.
                        st.markdown("**Pit Stop Times by Driver**")
                        
                        # Get unique drivers and sort by first stop lap.
                        if 'Driver' in pit_df.columns:
                            if lap_col:
                                driver_order = pit_df.groupby('Driver')[lap_col].min().sort_values().index.tolist()
                            else:
                                driver_order = pit_df['Driver'].dropna().astype(str).unique().tolist()
                        else:
                            driver_order = []
                        
                        fig = go.Figure()
                        
                        for driver in driver_order[:15]:  # Top 15 drivers.
                            driver_stops = pit_df[pit_df['Driver'] == driver].copy()
                            if stop_col and stop_col in driver_stops.columns:
                                driver_stops = driver_stops.sort_values(stop_col)
                            
                            for _, stop in driver_stops.iterrows():
                                stop_no = int(stop[stop_col]) if stop_col and pd.notna(stop.get(stop_col)) else 1
                                lap_no = int(stop[lap_col]) if lap_col and pd.notna(stop.get(lap_col)) else 0
                                pit_time = float(stop.get(duration_col))
                                fig.add_trace(go.Bar(
                                    y=[driver],
                                    x=[pit_time],
                                    orientation='h',
                                    name=f"Stop {stop_no}",
                                    text=f"L{lap_no}: {pit_time:.1f}s" if lap_no else f"{pit_time:.1f}s",
                                    textposition='auto',
                                    marker_color=['#E10600', '#FFD700', '#00FF00', '#3333FF'][(stop_no-1) % 4],
                                    showlegend=False,
                                    hovertemplate=f"{driver} - Stop {stop_no}<br>Lap: {lap_no if lap_no else 'N/A'}<br>Time: {pit_time:.1f}s<extra></extra>"
                                ))
                        
                        fig.update_layout(
                            title="Pit Stop Duration (seconds)",
                            xaxis_title="Pit Time (seconds)",
                            yaxis_title="Driver",
                            height=max(400, len(driver_order) * 30),
                            barmode='group',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            xaxis=dict(range=[0, max(pit_df[duration_col].max() * 1.1, 30)])
                        )
                        show_plotly_chart(fig, use_container_width=True)
                        
                        # Detailed pit stop table.
                        st.markdown("**Detailed Pit Stop Data**")
                        display_pit = pit_df.copy()
                        display_pit[duration_col] = display_pit[duration_col].apply(lambda x: f"{x:.1f}s")
                        display_pit = display_pit.rename(columns={'Stop': 'Stop #', 'StopNumber': 'Stop #', duration_col: 'Duration'})
                        st.dataframe(display_pit, use_container_width=True, hide_index=True)
            else:
                st.info("Pit stop data not available")
        
        with race_tabs[2]:
            # Tyre Strategy
            st.subheader("Tyre Strategy")
            tyre_data = get_tyre_stints(session)
            
            # Check if this is a sprint weekend
            sprint_tracks = ['China', 'Miami', 'Belgium', 'United States', 'Brazil', 'Qatar']
            is_sprint_weekend = selected_race in sprint_tracks
            
            if is_sprint_weekend:
                st.info(f"This is a Sprint Weekend - {selected_race}")
            
            if tyre_data is not None and not tyre_data.empty:
                # Compound Legend with visual - Removed emojis
                st.markdown("---")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.markdown("**SOFT** (Red)")
                with col2:
                    st.markdown("**MEDIUM** (Yellow)")
                with col3:
                    st.markdown("**HARD** (White)")
                with col4:
                    st.markdown("**INTERMEDIATE** (Green)")
                with col5:
                    st.markdown("**WET** (Blue)")
                
                # Stint details table
                with st.expander("View Detailed Stint Data"):
                    display_tyres = tyre_data[['Driver', 'Stint', 'Compound', 'StartLap', 'EndLap', 'Laps']].copy()
                    display_tyres['Stint'] = display_tyres['Stint'] + 1
                    display_tyres = display_tyres.rename(columns={'Stint': 'Stint #', 'StartLap': 'Start', 'EndLap': 'End'})
                    st.dataframe(display_tyres, use_container_width=True, hide_index=True)
            else:
                st.info("Tyre strategy data not available")
        
        with race_tabs[3]:
            # Track Status
            st.subheader("Track Status & Race Control")
            
            track_status = get_track_status(session)
            flag_events = get_flag_events(session)
            
            if track_status is not None and not track_status.empty:
                status_colors = {
                    '1': '#00FF00',  # Green
                    '2': '#FFFF00',  # Yellow
                    '4': '#FF0000',  # Red
                    '5': '#FF6600',  # SC
                    '6': '#FF00FF',  # VSC
                    '7': '#0000FF'   # VSC Ending
                }
                
                fig = go.Figure()
                
                for _, status in track_status.iterrows():
                    status_code = str(status.get('Status', '1'))
                    time_val = status.get('Time', 0)
                    color = status_colors.get(status_code, '#888888')
                    
                    fig.add_trace(go.Scatter(
                        x=[time_val],
                        y=[1],
                        mode='markers',
                        marker=dict(size=15, color=color),
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title="Track Status Timeline",
                    xaxis_title="Session Time",
                    height=200,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                show_plotly_chart(fig, use_container_width=True)
            
            if flag_events is not None and len(flag_events) > 0:
                st.subheader("Flag Events")
                flag_df = pd.DataFrame(flag_events)
                st.dataframe(flag_df, use_container_width=True, hide_index=True)
            else:
                st.info("Flag event data not available")
        
        with race_tabs[4]:
            # Sector Times
            st.subheader("Sector Times Analysis")
            
            # Field Best Sectors
            with st.expander("View Field Best Sectors"):
                best_sectors_df = get_best_sectors(session)
                if not best_sectors_df.empty:
                    st.dataframe(best_sectors_df, use_container_width=True, hide_index=True)
            
            sector_times = get_sector_times(session)
            
            if sector_times is not None and not sector_times.empty and 'Sector1' in sector_times.columns:
                # Select driver for sector analysis
                drivers_list = sector_times['Driver'].unique().tolist()
                selected_driver = st.selectbox("Select Driver for Sector Analysis", drivers_list, key="sector_driver")
                
                driver_sectors = sector_times[sector_times['Driver'] == selected_driver]
                driver_sectors = driver_sectors.dropna(subset=['Sector1', 'Sector2', 'Sector3'], how='all')
                
                if not driver_sectors.empty and len(driver_sectors) > 0:
                    # Check if we have valid data
                    has_s1 = 'Sector1' in driver_sectors.columns and driver_sectors['Sector1'].notna().any()
                    has_s2 = 'Sector2' in driver_sectors.columns and driver_sectors['Sector2'].notna().any()
                    has_s3 = 'Sector3' in driver_sectors.columns and driver_sectors['Sector3'].notna().any()
                    
                    if has_s1 or has_s2 or has_s3:
                        fig = make_subplots(rows=1, cols=3, subplot_titles=('Sector 1', 'Sector 2', 'Sector 3'))
                        
                        colors = ['#FF3333', '#00FF00', '#3333FF']
                        
                        for i, sector in enumerate(['Sector1', 'Sector2', 'Sector3'], 1):
                            if sector in driver_sectors.columns and driver_sectors[sector].notna().any():
                                valid_data = driver_sectors[driver_sectors[sector].notna()]
                                fig.add_trace(
                                    go.Scatter(
                                        x=valid_data['Lap'],
                                        y=valid_data[sector],
                                        mode='lines+markers',
                                        name=f"S{i}",
                                        line=dict(color=colors[i-1], width=2),
                                        marker=dict(size=4)
                                    ),
                                    row=1, col=i
                                )
                        
                        fig.update_layout(
                            height=350,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            showlegend=False
                        )
                        fig.update_yaxes(title_text="Time (s)")
                        fig.update_xaxes(title_text="Lap")
                        show_plotly_chart(fig, use_container_width=True)
                        
                        # Best sectors table
                        st.markdown("**Best Sector Times:**")
                        best_s1 = driver_sectors['Sector1'].min() if 'Sector1' in driver_sectors.columns else None
                        best_s2 = driver_sectors['Sector2'].min() if 'Sector2' in driver_sectors.columns else None
                        best_s3 = driver_sectors['Sector3'].min() if 'Sector3' in driver_sectors.columns else None
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Best S1", f"{best_s1:.3f}s" if best_s1 else "N/A")
                        with col2:
                            st.metric("Best S2", f"{best_s2:.3f}s" if best_s2 else "N/A")
                        with col3:
                            st.metric("Best S3", f"{best_s3:.3f}s" if best_s3 else "N/A")
                        with col4:
                            theoretical = (best_s1 or 0) + (best_s2 or 0) + (best_s3 or 0)
                            st.metric("Theoretical Best", f"{theoretical:.3f}s" if theoretical > 0 else "N/A")
                    else:
                        st.info("No valid sector time data for this driver")
                else:
                    st.info("No sector time data available for this driver")
            else:
                st.info("Sector time data not available for this session")
        
        with race_tabs[5]:
            # Speed Traps
            st.subheader("Speed Trap Analysis")
            top_speeds = get_top_speeds(session)
            
            if top_speeds is not None and not top_speeds.empty:
                st.dataframe(top_speeds, use_container_width=True, hide_index=True)
                
                if 'Max_SpeedST' in top_speeds.columns:
                    fig = go.Figure()
                    # Sort by speed
                    plot_data = top_speeds.sort_values('Max_SpeedST', ascending=True)
                    
                    fig.add_trace(go.Bar(
                        x=plot_data['Max_SpeedST'],
                        y=plot_data['Driver'],
                        orientation='h',
                        marker_color='#E10600',
                        text=plot_data['Max_SpeedST'].apply(lambda x: f"{x:.1f}"),
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        title="Top Speeds (Speed Trap)",
                        xaxis_title="Speed (km/h)",
                        height=max(400, len(plot_data) * 25),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        margin=dict(r=100)
                    )
                    show_plotly_chart(fig, use_container_width=True)
            else:
                st.info("Speed trap data not available")

        with race_tabs[6]:
            # Position Changes - Improved visualization
            st.subheader("Position Changes Throughout Race")
            
            position_changes = get_position_changes(session)
            
            if position_changes is not None and not position_changes.empty:
                # Overview metrics
                gainers = position_changes[position_changes['PositionsGained'] > 0]
                losers = position_changes[position_changes['PositionsGained'] < 0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if not gainers.empty:
                        best = gainers.nlargest(1, 'PositionsGained').iloc[0]
                        st.metric(
                            "Best Mover", 
                            best['Driver'],
                            f"+{int(best['PositionsGained'])} positions"
                        )
                    else:
                        st.metric("Best Mover", "N/A", "0")
                
                with col2:
                    if not losers.empty:
                        worst = losers.nsmallest(1, 'PositionsGained').iloc[0]
                        st.metric(
                            "Biggest Drop", 
                            worst['Driver'],
                            f"{int(worst['PositionsGained'])} positions"
                        )
                    else:
                        st.metric("Biggest Drop", "N/A", "0")
                
                with col3:
                    no_change = len(position_changes[position_changes['PositionsGained'] == 0])
                    st.metric("Held Position", f"{no_change} drivers")
                
                st.divider()
                
                # Detailed chart with dual view
                col1, col2 = st.columns(2)
                
                with col1:
                    # Positions Gained/Lost Bar Chart
                    pos_changes_sorted = position_changes.sort_values('PositionsGained', ascending=True)
                    
                    colors = ['rgba(0, 255, 0, 0.8)' if x > 0 else 'rgba(255, 0, 0, 0.8)' if x < 0 else 'rgba(136, 136, 136, 0.8)' 
                              for x in pos_changes_sorted['PositionsGained']]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=pos_changes_sorted['PositionsGained'],
                        y=pos_changes_sorted['Driver'],
                        orientation='h',
                        marker_color=colors,
                        text=[f"+{int(x)}" if x > 0 else str(int(x)) for x in pos_changes_sorted['PositionsGained']],
                        textposition='outside',
                        textfont=dict(color='white', size=12)
                    ))
                    
                    fig.update_layout(
                        title="Positions Gained/Lost",
                        xaxis_title="Positions Change",
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        margin=dict(l=10, r=120, t=40, b=40),
                        xaxis=dict(zeroline=True, zerolinecolor='white', zerolinewidth=1)
                    )
                    show_plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Grid vs Finish Position comparison
                    pos_sorted = position_changes.sort_values('FinalPosition')
                    
                    fig2 = go.Figure()
                    
                    # Grid position
                    fig2.add_trace(go.Scatter(
                        x=pos_sorted['Driver'],
                        y=pos_sorted['GridPosition'],
                        mode='markers+lines',
                        name='Grid',
                        marker=dict(size=12, color='#FFD700', symbol='diamond'),
                        line=dict(color='#FFD700', dash='dash')
                    ))
                    
                    # Finish position
                    fig2.add_trace(go.Scatter(
                        x=pos_sorted['Driver'],
                        y=pos_sorted['FinalPosition'],
                        mode='markers+lines',
                        name='Finish',
                        marker=dict(size=12, color='#00FF00', symbol='circle'),
                        line=dict(color='#00FF00')
                    ))
                    
                    fig2.update_layout(
                        title="Grid vs Finish Position",
                        yaxis_title="Position",
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        yaxis=dict(autorange='reversed'),  # P1 at top
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                        xaxis=dict(tickangle=45)
                    )
                    show_plotly_chart(fig2, use_container_width=True)
                
                st.divider()
                
                # Position changes detailed table
                st.markdown("**Complete Position Data:**")
                
                # Format table
                display_df = position_changes.copy()
                
                # Add status column
                def get_status(row):
                    change = row['PositionsGained']
                    if change > 0:
                        return f"UP {int(change)}"
                    elif change < 0:
                        return f"DOWN {int(abs(change))}"
                    else:
                        return "SAME"
                
                display_df['Status'] = display_df.apply(get_status, axis=1)
                
                # Reorder columns
                cols_order = ['Driver', 'GridPosition', 'FinalPosition', 'PositionsGained', 'Status']
                available_cols = [c for c in cols_order if c in display_df.columns]
                display_df = display_df[available_cols].sort_values('FinalPosition')
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("Position change data not available")
        
        with race_tabs[7]:
            # Lap Times Analysis
            st.subheader("Lap Time Analysis") # Removed emoji
            
            try:
                laps = session.laps
                
                if laps is not None and not laps.empty:
                    # Driver selector for lap times
                    drivers_list = laps['Driver'].unique().tolist()
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        selected_drivers = st.multiselect(
                            "Select Drivers (max 5)", 
                            drivers_list, 
                            default=drivers_list[:3] if len(drivers_list) >= 3 else drivers_list,
                            max_selections=5,
                            key="lap_drivers"
                        )
                    
                    if selected_drivers:
                        fig = go.Figure()
                        
                        for driver in selected_drivers:
                            driver_laps = laps[laps['Driver'] == driver]
                            lap_times = driver_laps['LapTime'].dt.total_seconds()
                            
                            # Get team color
                            team_name = driver_laps['Team'].iloc[0] if 'Team' in driver_laps.columns and len(driver_laps) > 0 else ''
                            color = TEAM_COLORS.get(team_name, '#888888')
                            
                            fig.add_trace(go.Scatter(
                                x=driver_laps['LapNumber'],
                                y=lap_times,
                                mode='lines+markers',
                                name=driver,
                                line=dict(color=color, width=2),
                                marker=dict(size=4)
                            ))
                        
                        fig.update_layout(
                            title="Lap Time Comparison",
                            xaxis_title="Lap Number",
                            yaxis_title="Lap Time (seconds)",
                            height=450,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02)
                        )
                        show_plotly_chart(fig, use_container_width=True)
                        
                        # Fastest laps table
                        st.markdown("**Fastest Laps:**")
                        fastest_laps = laps.groupby('Driver').apply(
                            lambda x: x.nsmallest(1, 'LapTime')
                        ).reset_index(drop=True)
                        
                        if not fastest_laps.empty:
                            display_cols = ['Driver', 'LapNumber', 'LapTime', 'Compound']
                            available_cols = [c for c in display_cols if c in fastest_laps.columns]
                            fastest_display = fastest_laps[available_cols].sort_values('LapTime') # Show all drivers
                            fastest_display['LapTime'] = fastest_display['LapTime'].apply(format_f1_time) # Apply formatter
                            st.dataframe(fastest_display, use_container_width=True, hide_index=True)
                else:
                    st.info("Lap time data not available")
            except Exception as e:
                st.error(f"Error loading lap times: {e}")


def render_race_analysis_tab(df):
    """Race Analysis tab content."""
    st.header("Race Analysis Center")
    st.markdown("Lap times, Pace comparison, Strategy analysis, Track Visualization") # Removed emojis
    
    if df is None or df.empty:
        st.caption("Local season table is empty for this mode. Session analysis can still run from FastF1 when available.")
    
    # Race selector
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        # Use active season races (API-backed for 2026)
        all_races = get_season_race_choices()
        selected_race = st.selectbox("Select Grand Prix", all_races, key="analysis_race")
    with col2:
        session_type = st.selectbox("Session", ["Race", "Qualifying", "Sprint"], key="analysis_session")
    with col3:
        load_analysis_clicked = st.button("Load Analysis Session", key="analysis_load_btn", type="primary")
    
    # Check if race has happened or is ongoing
    try:
        schedule = get_fastf1_schedule_cached(get_active_season_year())
        if schedule['EventDate'].dt.tz is None:
             schedule['EventDate'] = schedule['EventDate'].dt.tz_localize('UTC')
        
        # Find event
        race_event = schedule[schedule['EventName'] == selected_race]
        
        if not race_event.empty:
            event = race_event.iloc[0]
            now = pd.Timestamp.now(tz='UTC')
            
            # Check session specific time
            session_key = 'Session1' # Default
            if session_type == "Race": session_key = 'Session5'
            elif session_type == "Qualifying": session_key = 'Session4'
            elif session_type == "Sprint": session_key = 'Session3'
            
            # Map session names loosely if exact key match fails or for standard logic
            # FastF1 schedule keys: Session1...Session5. We need to find which one corresponds to selected type.
            # Actually, easier to just check if date is future.
            
            # Simple check: is event in future?
            if event['EventDate'] > (now + timedelta(hours=48)): # More than 2 days in future
                st.info(f"Analysis not available yet. {selected_race} has not started.")
                return
            
            # More detailed check if we had session times mapping, but simple catch-all exception on load works too.
    except Exception as e:
        pass # Ignore schedule check errors, let load fail naturally if needed

    active_year = get_active_season_year()
    analysis_sig = f"{active_year}|{selected_race}|{session_type}"
    if load_analysis_clicked:
        st.session_state["race_analysis_loaded_sig"] = analysis_sig
    if st.session_state.get("race_analysis_loaded_sig") != analysis_sig:
        st.info("Select a race/session and click `Load Analysis Session` to fetch analysis data.")
        return

    # Analysis sub-tabs
    analysis_tabs = st.tabs([
        "Lap Analysis", "Pace Comparison", "Stint Analysis", 
        "Gap Chart", "Track Visualization", "Position Chart", "Battle Analysis",
        "Strategy Tools", "Driver Scores", "Telemetry Compare",
        "Race Control", "Engineering"
    ])
    
    # Load session
    with st.spinner("Loading session data..."):
        race_lookup = selected_race
        session = get_fastf1_session_state_cached(
            year=active_year,
            race=race_lookup,
            session_type=session_type,
            load_telemetry=False,
            cache_namespace="race_analysis_fastf1",
            force_reload=bool(load_analysis_clicked),
        )
    
    if session is None:
        st.warning(f"Data not available for {selected_race} - {session_type}. The session might not have started yet.")
        return
    
    # TAB 1: LAP ANALYSIS
    with analysis_tabs[0]:
        st.subheader("Lap Time Analysis")
        try:
            laps = session.laps
            if laps is not None and not laps.empty:
                drivers_list = sorted(laps['Driver'].unique().tolist())
                selected_drivers = st.multiselect("Select Drivers (max 6)", drivers_list, 
                    default=drivers_list[:3] if len(drivers_list) >= 3 else drivers_list, max_selections=6, key="lap_drv")
                
                if selected_drivers:
                    fig = go.Figure()
                    for driver in selected_drivers:
                        drv_laps = laps[laps['Driver'] == driver]
                        drv_laps = drv_laps[drv_laps['LapTime'].notna()]
                        if not drv_laps.empty:
                            times = drv_laps['LapTime'].dt.total_seconds()
                            team = drv_laps['Team'].iloc[0] if 'Team' in drv_laps.columns else ''
                            color = TEAM_COLORS.get(team, '#888888')
                            fig.add_trace(go.Scatter(x=drv_laps['LapNumber'], y=times, mode='lines+markers',
                                name=driver, line=dict(color=color, width=2), marker=dict(size=4)))
                    
                    fig.update_layout(title="Lap Time Evolution", xaxis_title="Lap", yaxis_title="Time (s)",
                        height=450, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    show_plotly_chart(fig, use_container_width=True)
                    
                    # Fastest laps table
                    st.markdown("**Fastest Laps:**")
                    fastest_laps = laps.groupby('Driver').apply(
                        lambda x: x.nsmallest(1, 'LapTime')
                    ).reset_index(drop=True)
                    
                    if not fastest_laps.empty:
                        display_cols = ['Driver', 'LapNumber', 'LapTime', 'Compound']
                        available_cols = [c for c in display_cols if c in fastest_laps.columns]
                        fastest_display = fastest_laps[available_cols].sort_values('LapTime')
                        fastest_display['LapTime'] = fastest_display['LapTime'].apply(format_f1_time)
                        st.dataframe(fastest_display, use_container_width=True, hide_index=True)
            else:
                st.info("No lap data")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # TAB 2: PACE COMPARISON
    with analysis_tabs[1]:
        st.subheader("Pace Comparison")
        try:
            laps = session.laps
            if laps is not None and not laps.empty:
                drivers_list = sorted(laps['Driver'].unique().tolist())
                c1, c2 = st.columns(2)
                with c1:
                    drv_a = st.selectbox("Driver A", drivers_list, key="pace_a")
                with c2:
                    drv_b = st.selectbox("Driver B", [d for d in drivers_list if d != drv_a], key="pace_b")
                
                if st.button("Compare", type="primary", key="compare_btn"):
                    laps_a = laps[(laps['Driver'] == drv_a) & (laps['LapTime'].notna())].sort_values('LapNumber')
                    laps_b = laps[(laps['Driver'] == drv_b) & (laps['LapTime'].notna())].sort_values('LapNumber')
                    
                    if not laps_a.empty and not laps_b.empty:
                        times_a = laps_a['LapTime'].dt.total_seconds()
                        times_b = laps_b['LapTime'].dt.total_seconds()
                        team_a = laps_a['Team'].iloc[0] if 'Team' in laps_a.columns else ''
                        team_b = laps_b['Team'].iloc[0] if 'Team' in laps_b.columns else ''
                        
                        avg_diff = times_a.mean() - times_b.mean()
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric(f"{drv_a} Avg", f"{times_a.mean():.3f}s", f"{avg_diff:+.3f}s")
                        with c2:
                            st.metric(f"{drv_b} Avg", f"{times_b.mean():.3f}s", f"{-avg_diff:+.3f}s")
                        with c3:
                            faster = drv_a if avg_diff < 0 else drv_b
                            st.metric("Faster", faster, f"{abs(avg_diff):.3f}s")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=laps_a['LapNumber'], y=times_a, name=drv_a,
                            line=dict(color=TEAM_COLORS.get(team_a, '#FF0000'), width=2)))
                        fig.add_trace(go.Scatter(x=laps_b['LapNumber'], y=times_b, name=drv_b,
                            line=dict(color=TEAM_COLORS.get(team_b, '#00FF00'), width=2)))
                        fig.update_layout(title=f"{drv_a} vs {drv_b}", xaxis_title="Lap", yaxis_title="Time (s)",
                            height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        show_plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
    
    # TAB 3: STINT ANALYSIS
    with analysis_tabs[2]:
        st.subheader("Tyre Stint Analysis")
        try:
            tyre_data = get_tyre_stints(session)
            if tyre_data is not None and not tyre_data.empty:
                compound_colors = {'SOFT': '#FF3333', 'MEDIUM': '#FFD700', 'HARD': '#FFFFFF', 'INTERMEDIATE': '#43B02A', 'WET': '#0067AD'}
                
                try:
                    results = session.results
                    driver_order = results.sort_values('Position')['Abbreviation'].tolist() if results is not None else tyre_data['Driver'].unique().tolist()
                except:
                    driver_order = tyre_data['Driver'].unique().tolist()
                
                drivers = [d for d in driver_order if d in tyre_data['Driver'].values][:20]
                fig = go.Figure()
                
                for driver in drivers:
                    driver_tyres = tyre_data[tyre_data['Driver'] == driver].sort_values('Stint')
                    
                    for _, stint in driver_tyres.iterrows():
                        compound = str(stint.get('Compound', 'MEDIUM')).upper()
                        start = int(stint.get('StartLap', 1))
                        end = int(stint.get('EndLap', start + 10))
                        laps = end - start + 1
                        color = compound_colors.get(compound, '#888888')
                        
                        fig.add_trace(go.Bar(x=[laps], y=[driver], orientation='h', base=start-1,
                            marker_color=color, marker_line_color='#333', marker_line_width=1,
                            showlegend=False, text=compound[0] if compound != 'UNKNOWN' else '?',
                            textposition='inside', textfont=dict(color='black' if compound in ['MEDIUM','HARD'] else 'white', size=10)))
                
                total_laps = tyre_data['EndLap'].max() if 'EndLap' in tyre_data.columns else 50
                fig.update_layout(title="Tyre Strategy", xaxis_title="Lap", xaxis=dict(range=[0, total_laps+2]),
                    height=max(450, len(drivers)*25), barmode='overlay', paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                    yaxis=dict(categoryorder='array', categoryarray=drivers[::-1]))
                show_plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.markdown("**SOFT** (Red)")
                with col2:
                    st.markdown("**MEDIUM** (Yellow)")
                with col3:
                    st.markdown("**HARD** (White)")
                with col4:
                    st.markdown("**INTERMEDIATE** (Green)")
                with col5:
                    st.markdown("**WET** (Blue)")
                
                # Stint details table
                with st.expander("View Detailed Stint Data"):
                    display_tyres = tyre_data[['Driver', 'Stint', 'Compound', 'StartLap', 'EndLap', 'Laps']].copy()
                    display_tyres['Stint'] = display_tyres['Stint'] + 1
                    display_tyres = display_tyres.rename(columns={'Stint': 'Stint #', 'StartLap': 'Start', 'EndLap': 'End'})
                    st.dataframe(display_tyres, use_container_width=True, hide_index=True)
            else:
                st.info("Tyre strategy data not available")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # TAB 4: GAP CHART
    with analysis_tabs[3]:
        st.subheader("Gap to Leader")
        try:
            laps = session.laps
            if laps is not None and not laps.empty:
                drivers_list = sorted(laps['Driver'].unique().tolist())
                gap_drivers = st.multiselect("Select Drivers (max 8)", drivers_list,
                    default=drivers_list[:5] if len(drivers_list) >= 5 else drivers_list, max_selections=8, key="gap_drv")
                
                if gap_drivers:
                    # Find leader
                    leader_data = None
                    min_time = float('inf')
                    for driver in gap_drivers:
                        drv_laps = laps[(laps['Driver'] == driver) & (laps['LapTime'].notna())].sort_values('LapNumber')
                        if not drv_laps.empty:
                            cum = drv_laps['LapTime'].dt.total_seconds().cumsum()
                            total = cum.iloc[-1]
                            if total < min_time:
                                min_time = total
                                leader_data = (driver, drv_laps, cum)
                    
                    if leader_data:
                        leader_name, leader_laps, leader_cum = leader_data
                        leader_dict = dict(zip(leader_laps['LapNumber'], leader_cum))
                        
                        fig = go.Figure()
                        for driver in gap_drivers:
                            drv_laps = laps[(laps['Driver'] == driver) & (laps['LapTime'].notna())].sort_values('LapNumber')
                            if not drv_laps.empty:
                                team = drv_laps['Team'].iloc[0] if 'Team' in drv_laps.columns else ''
                                cum = drv_laps['LapTime'].dt.total_seconds().cumsum()
                                gaps = [c - leader_dict.get(l, c) for l, c in zip(drv_laps['LapNumber'], cum)]
                                fig.add_trace(go.Scatter(x=drv_laps['LapNumber'], y=gaps, mode='lines',
                                    name=driver, line=dict(color=TEAM_COLORS.get(team, '#888888'), width=2)))
                        
                        fig.update_layout(title=f"Gap to Leader ({leader_name})", xaxis_title="Lap", yaxis_title="Gap (s)",
                            height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        show_plotly_chart(fig, use_container_width=True)
            else:
                st.info("No lap data")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # TAB 5: TRACK VISUALIZATION
    with analysis_tabs[4]:
        st.subheader("Track Visualization")
        
        with st.expander("3D Track Map Analysis", expanded=True):
            st.markdown("Interactive 3D map colored by speed. Drag to rotate, scroll to zoom.")
            try:
                # Get driver list for specific driver highlight
                if hasattr(session, 'drivers'):
                    drivers = session.drivers
                    drivers_list = [session.get_driver(d)["Abbreviation"] for d in drivers]
                    sel_driver = st.selectbox("Highlight Driver Line", ["Fastest Lap"] + sorted(drivers_list), key="3d_drv_sel")
                    driver_arg = None if sel_driver == "Fastest Lap" else sel_driver
                    
                    if st.button("Generate 3D Map", key="gen_3d_btn"):
                        with st.spinner("Generating 3D Map..."):
                            fig_3d = plot_track_3d(session, driver=driver_arg)
                            if fig_3d:
                                show_plotly_chart(fig_3d, use_container_width=True)
                            else:
                                st.error("Could not generate map (missing telemetry?)")
            except Exception as e:
                st.error(f"Error in 3D Map: {e}")

        st.divider()
        st.subheader("2D Live Race Replay")
        
        try:
            # Speed control
            col_speed, col_lap = st.columns([1, 2])
            with col_speed:
                frame_duration = st.slider("Animation Speed (ms)", 10, 300, 50, 10, 
                    help="Lower = faster animation", key="anim_speed")
            with col_lap:
                selected_lap_anim = st.number_input("Select Lap", min_value=1, value=1, key="anim_lap_select")
            
            if st.button("Generate Animation", type="primary", key="gen_anim_btn"):
                with st.spinner("Loading telemetry data..."):
                    # Get all drivers in session
                    laps = session.laps
                    if laps is None or laps.empty:
                        st.error("No lap data available")
                    else:
                        drivers_in_session = laps['Driver'].unique().tolist()
                        
                        # Collect telemetry for selected lap from all drivers
                        all_tel_data = []
                        driver_colors = {}
                        
                        for driver in drivers_in_session:
                            try:
                                drv_laps = laps.pick_driver(driver)
                                if drv_laps.empty:
                                    continue
                                
                                # Get team color
                                team = drv_laps['Team'].iloc[0] if 'Team' in drv_laps.columns else ''
                                driver_colors[driver] = TEAM_COLORS.get(team, '#888888')
                                
                                # Get lap telemetry
                                lap_data = drv_laps[drv_laps['LapNumber'] == selected_lap_anim]
                                if lap_data.empty:
                                    lap_data = drv_laps.iloc[[0]]
                                
                                tel = lap_data.iloc[0].get_telemetry()
                                if tel is not None and not tel.empty and 'X' in tel.columns and 'Y' in tel.columns:
                                    tel = tel[['X', 'Y']].copy()
                                    tel['Driver'] = driver
                                    tel['Time_idx'] = range(len(tel))
                                    all_tel_data.append(tel)
                            except:
                                continue
                        
                        if not all_tel_data:
                            st.error("No telemetry data available for this lap")
                        else:
                            # Get track outline from driver with most data points
                            track_tel = max(all_tel_data, key=len)
                            track_x = track_tel['X'].values
                            track_y = track_tel['Y'].values
                            
                            # Get track boundaries
                            x_min, x_max = track_x.min(), track_x.max()
                            y_min, y_max = track_y.min(), track_y.max()
                            x_margin = (x_max - x_min) * 0.08
                            y_margin = (y_max - y_min) * 0.08
                            
                            # Number of animation frames
                            num_points = 150
                            
                            # Interpolate all drivers to same number of time points
                            interpolated_data = {}
                            for tel_df in all_tel_data:
                                driver = tel_df['Driver'].iloc[0]
                                orig_x = tel_df['X'].values
                                orig_y = tel_df['Y'].values
                                
                                if len(orig_x) < 2:
                                    continue
                                
                                # Linear interpolation
                                orig_indices = np.linspace(0, 1, len(orig_x))
                                new_indices = np.linspace(0, 1, num_points)
                                
                                interp_x = np.interp(new_indices, orig_indices, orig_x)
                                interp_y = np.interp(new_indices, orig_indices, orig_y)
                                
                                interpolated_data[driver] = {'X': interp_x, 'Y': interp_y}
                            
                            if not interpolated_data:
                                st.error("Could not interpolate driver data")
                            else:
                                # Create frames - each frame includes track + all car positions
                                frames = []
                                for frame_idx in range(num_points):
                                    frame_data = [
                                        # Track background (must be in every frame)
                                        go.Scatter(
                                            x=track_x, y=track_y,
                                            mode='lines',
                                            line=dict(color='#333333', width=20),
                                            showlegend=False,
                                            hoverinfo='skip'
                                        ),
                                        go.Scatter(
                                            x=track_x, y=track_y,
                                            mode='lines',
                                            line=dict(color='#555555', width=1),
                                            showlegend=False,
                                            hoverinfo='skip'
                                        )
                                    ]
                                    
                                    # Add all drivers at current frame position
                                    for driver, data in interpolated_data.items():
                                        frame_data.append(go.Scatter(
                                            x=[data['X'][frame_idx]],
                                            y=[data['Y'][frame_idx]],
                                            mode='markers+text',
                                            marker=dict(size=14, color=driver_colors.get(driver, '#888888'), 
                                                       line=dict(width=2, color='white')),
                                            text=[driver],
                                            textposition='top center',
                                            textfont=dict(size=9, color='white'),
                                            name=driver,
                                            showlegend=False
                                        ))
                                    
                                    frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
                                
                                # Initial figure data (frame 0)
                                initial_data = [
                                    go.Scatter(
                                        x=track_x, y=track_y,
                                        mode='lines',
                                        line=dict(color='#333333', width=20),
                                        name='Track',
                                        showlegend=False,
                                        hoverinfo='skip'
                                    ),
                                    go.Scatter(
                                        x=track_x, y=track_y,
                                        mode='lines',
                                        line=dict(color='#555555', width=1),
                                        showlegend=False,
                                        hoverinfo='skip'
                                    )
                                ]
                                
                                for driver, data in interpolated_data.items():
                                    initial_data.append(go.Scatter(
                                        x=[data['X'][0]],
                                        y=[data['Y'][0]],
                                        mode='markers+text',
                                        marker=dict(size=14, color=driver_colors.get(driver, '#888888'),
                                                   line=dict(width=2, color='white')),
                                        text=[driver],
                                        textposition='top center',
                                        textfont=dict(size=9, color='white'),
                                        name=driver
                                    ))
                                
                                # Build animated figure
                                fig = go.Figure(
                                    data=initial_data,
                                    frames=frames,
                                    layout=go.Layout(
                                        title=dict(text=f"Lap {selected_lap_anim} - {selected_race}", 
                                                  font=dict(color='white', size=16)),
                                        xaxis=dict(range=[x_min - x_margin, x_max + x_margin], 
                                                  visible=False, scaleanchor='y'),
                                        yaxis=dict(range=[y_min - y_margin, y_max + y_margin], visible=False),
                                        height=600,
                                        paper_bgcolor='#0e1117',
                                        plot_bgcolor='#0e1117',
                                        font=dict(color='white'),
                                        showlegend=True,
                                        legend=dict(x=1.02, y=1, bgcolor='rgba(0,0,0,0.5)', font=dict(size=10)),
                                        updatemenus=[dict(
                                            type='buttons',
                                            showactive=False,
                                            y=0,
                                            x=0.1,
                                            xanchor='right',
                                            yanchor='top',
                                            buttons=[
                                                dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(
                                                        frame=dict(duration=frame_duration, redraw=True),
                                                        fromcurrent=True,
                                                        transition=dict(duration=0)
                                                    )]),
                                                dict(label='Pause',
                                                    method='animate',
                                                    args=[[None], dict(
                                                        frame=dict(duration=0, redraw=False),
                                                        mode='immediate',
                                                        transition=dict(duration=0)
                                                    )])
                                            ]
                                        )],
                                        sliders=[dict(
                                            active=0,
                                            yanchor='top',
                                            xanchor='left',
                                            currentvalue=dict(prefix='Progress: ', visible=True, 
                                                             xanchor='right', font=dict(color='white', size=12)),
                                            transition=dict(duration=0),
                                            pad=dict(b=10, t=50),
                                            len=0.9,
                                            x=0.1,
                                            y=0,
                                            steps=[dict(args=[[f.name], dict(
                                                frame=dict(duration=0, redraw=True),
                                                mode='immediate',
                                                transition=dict(duration=0)
                                            )], label='', method='animate') 
                                                for f in frames]
                                        )]
                                    )
                                )
                                
                                show_plotly_chart(fig, use_container_width=True)
                                st.caption(f"Animation: {len(interpolated_data)} drivers, {num_points} frames. Click Play to start.")
            
            # Show Track Dominance Map
            st.markdown("---")
            st.subheader("ðŸ—ºï¸ Circuit Dominance Map")
            st.markdown("Color-coded track map showing which team/driver is fastest in each mini-sector.")
            
            if st.button("Generate Dominance Map", type="primary", key="dom_btn"):
                with st.spinner("Calculating mini-sector dominance..."):
                    try:
                        laps = session.laps
                        if laps is not None and not laps.empty:
                            # 1. Get telemetry for all fastest laps per driver
                            drivers = laps['Driver'].unique()
                            telemetry_data = []
                            
                            for d in drivers:
                                dl = laps.pick_driver(d).pick_fastest()
                                if dl is not None:
                                    t = dl.get_telemetry()
                                    if t is not None:
                                        t['Driver'] = d
                                        t['Team'] = dl['Team']
                                        telemetry_data.append(t)
                            
                            if telemetry_data:
                                # 2. Merge all telemetry
                                all_tel = pd.concat(telemetry_data)
                                
                                # 3. Create mini-sectors relative to distance
                                all_tel['DistanceInt'] = (all_tel['Distance'] // 50).astype(int) # 50m chunks
                                
                                # 4. Find fastest driver per chunk (Highest Speed)
                                # Simplified: Using Speed as proxy for 'fastest through sector'
                                sector_dominance = all_tel.loc[all_tel.groupby('DistanceInt')['Speed'].idxmax()]
                                
                                # 5. Plot
                                fig_dom = go.Figure()
                                for team in sector_dominance['Team'].unique():
                                    team_segments = sector_dominance[sector_dominance['Team'] == team]
                                    color = TEAM_COLORS.get(team, '#888')
                                    
                                    fig_dom.add_trace(go.Scatter(
                                        x=team_segments['X'], y=team_segments['Y'],
                                        mode='markers',
                                        marker=dict(size=4, color=color),
                                        name=team
                                    ))
                                
                                fig_dom.update_layout(
                                    height=500,
                                    paper_bgcolor='#0e1117',
                                    plot_bgcolor='#0e1117',
                                    xaxis=dict(visible=False, scaleanchor='y'),
                                    yaxis=dict(visible=False),
                                    title="Top Speed Dominance by Team",
                                    font=dict(color='white')
                                )
                                show_plotly_chart(fig_dom, use_container_width=True)
                            else:
                                st.warning("Not enough telemetry data")
                    except Exception as ex:
                        st.error(f"Dominance Map Error: {ex}")
            
            # Static Preview (Fallback)

            try:
                laps = session.laps
                if laps is not None and not laps.empty:
                    sample_driver = laps['Driver'].iloc[0]
                    drv_laps = laps.pick_driver(sample_driver)
                    if not drv_laps.empty:
                        fastest = drv_laps.pick_fastest()
                        if fastest is not None:
                            tel = fastest.get_telemetry()
                            if tel is not None and 'X' in tel.columns:
                                fig_preview = go.Figure()
                                fig_preview.add_trace(go.Scatter(
                                    x=tel['X'], y=tel['Y'],
                                    mode='lines',
                                    line=dict(color='#E10600', width=3),
                                    showlegend=False
                                ))
                                fig_preview.update_layout(
                                    height=250,
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    xaxis=dict(visible=False, scaleanchor='y'),
                                    yaxis=dict(visible=False),
                                    margin=dict(l=0, r=0, t=0, b=0)
                                )
                                show_plotly_chart(fig_preview, use_container_width=True)
            except:
                pass

            # NEW: Corner Analysis Matrix
            st.divider()
            st.subheader("ðŸ‘‘ Corner Mastery Matrix")
            st.markdown("Average speed in **Low (<120)**, **Medium (120-230)**, and **High (>230)** speed zones.")
            
            if st.button("Generate Performance Matrix", key="corn_btn"):
                with st.spinner("Analyzing corner speeds..."):
                    fig_corn = plot_corner_performance(session)
                    if fig_corn:
                        show_plotly_chart(fig_corn, use_container_width=True)
                    else:
                        st.info("Could not generate matrix")
                
        except Exception as e:
            st.error(f"Error: {e}")
    
    # TAB 6: POSITION CHART
    with analysis_tabs[5]:
        st.subheader("Position Chart")
        try:
            laps = session.laps
            if laps is not None and not laps.empty:
                pos_data = []
                # Fix: Ensure correct types
                laps = laps.copy()
                laps['Position'] = pd.to_numeric(laps['Position'], errors='coerce')
                
                # Check column existence
                if 'Position' in laps.columns:
                    for lap_num in sorted(laps['LapNumber'].unique()):
                        lap = laps[laps['LapNumber'] == lap_num]
                        for _, row in lap.iterrows():
                            if pd.notna(row['Position']):
                                pos_data.append({'Lap': int(lap_num), 'Driver': row['Driver'], 
                                    'Position': int(row['Position']), 'Team': row.get('Team', '')})
                
                if pos_data:
                    pos_df = pd.DataFrame(pos_data)
                    total_laps = pos_df['Lap'].max()
                    
                    color_map = {d: TEAM_COLORS.get(pos_df[pos_df['Driver']==d]['Team'].iloc[0], '#888888') 
                                 for d in pos_df['Driver'].unique()}
                    
                    selected_lap = st.slider("Lap", 1, int(total_laps), 1, key="pos_lap")
                    lap_pos = pos_df[pos_df['Lap'] == selected_lap].sort_values('Position')
                    
                    if not lap_pos.empty:
                        fig = go.Figure()
                        for _, row in lap_pos.iterrows():
                            fig.add_trace(go.Bar(x=[100 - (row['Position']-1)*4], y=[row['Driver']], orientation='h',
                                marker_color=color_map.get(row['Driver'], '#888'), text=f"P{int(row['Position'])}",
                                textposition='inside', textfont=dict(color='white', size=12), showlegend=False))
                        
                        fig.update_layout(title=f"Lap {selected_lap}/{int(total_laps)}", xaxis=dict(visible=False, range=[0,110]),
                            yaxis=dict(categoryorder='array', categoryarray=lap_pos.sort_values('Position', ascending=False)['Driver'].tolist()),
                            height=max(400, len(lap_pos)*25), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        show_plotly_chart(fig, use_container_width=True)
                    
                    # Position evolution
                    st.markdown("**Position Evolution**")
                    fig2 = go.Figure()
                    for driver in pos_df['Driver'].unique()[:15]:
                        drv_pos = pos_df[pos_df['Driver'] == driver].sort_values('Lap')
                        fig2.add_trace(go.Scatter(x=drv_pos['Lap'], y=drv_pos['Position'], mode='lines',
                            name=driver, line=dict(color=color_map.get(driver, '#888888'), width=2)))
                    fig2.update_layout(title="Position Changes", xaxis_title="Lap", yaxis_title="Position",
                        yaxis=dict(autorange='reversed', dtick=1), height=500, paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    show_plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No lap data")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # TAB 7: BATTLE ANALYSIS
    with analysis_tabs[6]:
        st.subheader("Battle Analysis")
        try:
            laps = session.laps
            if laps is not None and not laps.empty:
                drivers_list = sorted(laps['Driver'].unique().tolist())
                c1, c2 = st.columns(2)
                with c1:
                    battle_a = st.selectbox("Driver A", drivers_list, key="battle_a")
                with c2:
                    battle_b = st.selectbox("Driver B", [d for d in drivers_list if d != battle_a], key="battle_b")
                
                # Context Map
                with st.expander("Circuit Context", expanded=False):
                    ctx_fig = plot_circuit_context(session)
                    if ctx_fig: show_plotly_chart(ctx_fig, use_container_width=False)

                if st.button("Analyze Battle", type="primary", key="battle_btn"):
                    laps_a = laps[(laps['Driver'] == battle_a) & (laps['LapTime'].notna())].sort_values('LapNumber')
                    laps_b = laps[(laps['Driver'] == battle_b) & (laps['LapTime'].notna())].sort_values('LapNumber')
                    
                    if not laps_a.empty and not laps_b.empty:
                        team_a = laps_a['Team'].iloc[0] if 'Team' in laps_a.columns else ''
                        team_b = laps_b['Team'].iloc[0] if 'Team' in laps_b.columns else ''
                        color_a = TEAM_COLORS.get(team_a, '#FF4444')
                        color_b = TEAM_COLORS.get(team_b, '#44FF44')
                        
                        common_laps = set(laps_a['LapNumber']) & set(laps_b['LapNumber'])
                        faster_a, faster_b = 0, 0
                        
                        for lap in common_laps:
                            t_a = laps_a[laps_a['LapNumber'] == lap]['LapTime'].dt.total_seconds().iloc[0]
                            t_b = laps_b[laps_b['LapNumber'] == lap]['LapTime'].dt.total_seconds().iloc[0]
                            if t_a < t_b: faster_a += 1
                            else: faster_b += 1
                        
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown(f"""<div style="text-align:center;padding:20px;background:linear-gradient(135deg,{color_a}55 0%,{color_a}22 100%);border-radius:10px;border:2px solid {color_a};">
                                <h3 style="color:white;margin:0;">{battle_a}</h3><h1 style="color:{color_a};margin:10px 0;">{faster_a}</h1><p style="color:#888;">Faster Laps</p></div>""", unsafe_allow_html=True)
                        with c2:
                            st.markdown(f"""<div style="text-align:center;padding:20px;background:#1a1a2e;border-radius:10px;border:2px solid #444;">
                                <h3 style="color:white;margin:0;">VS</h3><h2 style="color:#FFD700;margin:10px 0;">{len(common_laps)}</h2><p style="color:#888;">Laps</p></div>""", unsafe_allow_html=True)
                        with c3:
                            st.markdown(f"""<div style="text-align:center;padding:20px;background:linear-gradient(135deg,{color_b}55 0%,{color_b}22 100%);border-radius:10px;border:2px solid {color_b};">
                                <h3 style="color:white;margin:0;">{battle_b}</h3><h1 style="color:{color_b};margin:10px 0;">{faster_b}</h1><p style="color:#888;">Faster Laps</p></div>""", unsafe_allow_html=True)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=laps_a['LapNumber'], y=laps_a['LapTime'].dt.total_seconds(), name=battle_a,
                            line=dict(color=color_a, width=2)))
                        fig.add_trace(go.Scatter(x=laps_b['LapNumber'], y=laps_b['LapTime'].dt.total_seconds(), name=battle_b,
                            line=dict(color=color_b, width=2)))
                        fig.update_layout(title=f"{battle_a} vs {battle_b}", xaxis_title="Lap", yaxis_title="Time (s)",
                            height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        show_plotly_chart(fig, use_container_width=True)
            else:
                st.info("No lap data")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # TAB 8: STRATEGY TOOLS
    with analysis_tabs[7]:
        st.subheader("Strategy Analysis")
        try:
            # 1. Tyre Visuals (New)
            st.markdown("### Tyre Stint Analysis")
            s_laps = session.laps
            driver_sel = st.selectbox("Select Driver for Stint View", sorted(s_laps['Driver'].unique()), key="stint_viz_drv")
            
            c1, c2 = st.columns([1, 2])
            with c1:
                # Plot Donut for Current Stint / All Stints ?
                # Let's plot the Start Tyres
                t_fig = plot_tyre_shape(session, driver_sel)
                if t_fig: show_plotly_chart(t_fig, use_container_width=True)
            
            with c2:
                # Existing Pit Detail
                pass 

            pit_stops = get_pit_stops(session)
            if pit_stops is not None and not pit_stops.empty:
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Total Stops", len(pit_stops))
                with c2: st.metric("Avg Time", f"{pit_stops['PitTime'].mean():.1f}s")
                with c3: st.metric("Fastest", f"{pit_stops['PitTime'].min():.1f}s")
                with c4: st.metric("Popular Lap", f"L{int(pit_stops['Lap'].mode().iloc[0])}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pit_stops['Lap'], y=pit_stops['PitTime'], mode='markers',
                    marker=dict(size=12, color=pit_stops['PitTime'], colorscale='RdYlGn_r', showscale=True),
                    text=pit_stops['Driver'], hovertemplate="%{text}<br>Lap: %{x}<br>Time: %{y:.1f}s<extra></extra>"))
                fig.update_layout(title="Pit Stop Distribution", xaxis_title="Lap", yaxis_title="Time (s)",
                    height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                show_plotly_chart(fig, use_container_width=True)
                
                pit_laps = pit_stops.groupby('Lap').size().reset_index(name='Stops')
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=pit_laps['Lap'], y=pit_laps['Stops'], marker_color='#E10600'))
                fig2.update_layout(title="Pit Windows", xaxis_title="Lap", yaxis_title="Stops",
                    height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            else:
                st.info("No pit stop data")
        except Exception as e:
            st.error(f"Error: {e}")

        # --- GOD MODE SIMULATION INTEGRATION ---
        st.divider()
        st.markdown("### âš¡ Dynamic Strategy Simulation")
        
        col1, col2 = st.columns(2)
        with col1:
            base_lap = st.number_input("Base Lap Time (s)", 80.0, 120.0, 90.0, step=0.1, key="strat_base")
            total_laps_sim = st.number_input("Total Laps", 10, 80, 52, key="strat_tot")
        
        sim = RaceStrategySimulator(base_lap, total_laps_sim)
        
        sc1, sc2, sc3 = st.columns(3)
        with sc1: driver_sim = st.text_input("Driver", "Verstappen", key="strat_drv")
        with sc2: start_tire = st.selectbox("Start Compound", ["SOFT", "MEDIUM", "HARD"], key="strat_tire")
        with sc3: current_lap_sim = st.slider("Current Lap", 0, int(total_laps_sim), 0, key="strat_lap")
        
        if st.button("Run Strategy Simulation", type="primary"):
            strategy = sim.predict_strategy(driver_sim, start_tire, current_lap_sim)
            st.success(f"Recommended: **{strategy['recommended']}**")
            m1, m2, m3 = st.columns(3)
            m1.metric("1-Stop", f"{strategy['1_stop_time']:.1f}s")
            m2.metric("2-Stop", f"{strategy['2_stop_time']:.1f}s")
            m3.metric("Delta", f"{strategy['delta']:.1f}s")
            
            # Catch up Logic
            st.subheader("Catch-Up Prediction")
            cc1, cc2 = st.columns(2)
            with cc1: gap = st.number_input("Gap (s)", 0.0, 60.0, 5.0, key="strat_gap")
            with cc2: 
                chaser_tyre = st.selectbox("Chaser", ["SOFT", "MEDIUM"], key="strat_chaser")
                leader_tyre = st.selectbox("Leader", ["MEDIUM", "HARD"], index=1, key="strat_leader")
            
            laps_catch = sim.catch_up_prediction(gap, chaser_tyre, leader_tyre, total_laps_sim - current_lap_sim)
            if laps_catch != -1:
                st.info(f"ðŸš€ Overtake in **{laps_catch} laps**")
            else:
                st.warning("âš ï¸ Overtake unlikely")
    
    # TAB 9: DRIVER SCORES
    with analysis_tabs[8]:
        st.subheader("Driver Performance Scores")
        try:
            laps = session.laps
            if laps is not None and not laps.empty:
                scores = []
                session_best = laps['LapTime'].dt.total_seconds().min()
                field_median = laps['LapTime'].dt.total_seconds().median()
                
                for driver in laps['Driver'].unique():
                    drv_laps = laps[(laps['Driver'] == driver) & (laps['LapTime'].notna())]
                    if len(drv_laps) >= 3:
                        times = drv_laps['LapTime'].dt.total_seconds()
                        pace = max(0, 100 - (times.min() - session_best) * 10)
                        consistency = max(0, 100 - times.std() * 20)
                        race_pace = max(0, 100 - (times.mean() - field_median) * 5)
                        overall = pace * 0.4 + consistency * 0.35 + race_pace * 0.25
                        team = drv_laps['Team'].iloc[0] if 'Team' in drv_laps.columns else ''
                        scores.append({'Driver': driver, 'Team': team, 'Pace': round(pace,1), 
                            'Consistency': round(consistency,1), 'Race Pace': round(race_pace,1), 
                            'Overall': round(overall,1), 'Best': f"{times.min():.3f}"})
                
                if scores:
                    scores_df = pd.DataFrame(scores).sort_values('Overall', ascending=False)
                    scores_df['Rank'] = range(1, len(scores_df)+1)
                    
                    st.divider()
                    
                    # --- NEW: RADAR CHART ---
                    st.markdown("### ðŸ•¸ï¸ Driver Capability Radar")
                    col_radar, col_table = st.columns([1, 1])
                    
                    with col_radar:
                        radar_drivers = st.multiselect("Compare Drivers", scores_df['Driver'].unique(), 
                                                     default=scores_df['Driver'].head(3).tolist() if len(scores_df)>=3 else scores_df['Driver'].tolist())
                        
                        if radar_drivers:
                            fig_radar = go.Figure()
                            for d in radar_drivers:
                                d_row = scores_df[scores_df['Driver'] == d].iloc[0]
                                d_team = d_row['Team']
                                d_color = TEAM_COLORS.get(d_team, '#888')
                                
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=[d_row['Pace'], d_row['Consistency'], d_row['Race Pace'], d_row['Overall']],
                                    theta=['Quali Pace', 'Consistency', 'Race Pace', 'Overall Rating'],
                                    fill='toself',
                                    name=d,
                                    line_color=d_color
                                ))
                            
                            fig_radar.update_layout(
                                polar=dict(
                                    radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, linecolor='#444'),
                                    bgcolor='rgba(0,0,0,0)'
                                ),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'),
                                margin=dict(l=40, r=40, t=20, b=20),
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                            )
                            show_plotly_chart(fig_radar, use_container_width=True)
                    
                    with col_table:
                        st.markdown("### Leaderboard")
                        st.dataframe(
                            scores_df[['Rank','Driver','Team','Overall']].style.background_gradient(subset=['Overall'], cmap='plasma'), 
                            use_container_width=True, hide_index=True
                        )
                    
                    # Radar chart
                    fig = go.Figure()
                    for _, row in scores_df.head(5).iterrows():
                        fig.add_trace(go.Scatterpolar(r=[row['Pace'], row['Consistency'], row['Race Pace']],
                            theta=['Pace', 'Consistency', 'Race Pace'], fill='toself', name=row['Driver'],
                            line=dict(color=TEAM_COLORS.get(row['Team'], '#888888'))))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100]), bgcolor='rgba(0,0,0,0)'),
                        height=450, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    show_plotly_chart(fig, use_container_width=True)
            else:
                st.info("No lap data")
        except Exception as e:
            st.error(f"Error: {e}")


    # TAB 10: TELEMETRY COMPARE
    with analysis_tabs[9]:
        st.subheader("Driver Telemetry Comparison")
        try:
            laps = session.laps
            if laps is not None and not laps.empty:
                drivers_list = sorted(laps['Driver'].unique().tolist())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    drv1 = st.selectbox("Driver 1", drivers_list, key="tel_d1")
                with col2:
                    drv2 = st.selectbox("Driver 2", [d for d in drivers_list if d != drv1], key="tel_d2")
                with col3:
                    lap_mode = st.radio("Lap Selection", ["Fastest Lap", "Specific Lap"], horizontal=True, key="tel_mode")
                
                selected_lap = None
                if lap_mode == "Specific Lap":
                    max_laps = int(laps['LapNumber'].max())
                    selected_lap = st.slider("Select Lap", 1, max_laps, 1, key="tel_lap_sel")
                
                if st.button("Compare Telemetry", type="primary", key="tel_comp_btn"):
                    with st.spinner("Generating detailed telemetry..."):
                        fig = plot_telemetry_comparison(session, drv1, drv2, selected_lap)
                        if fig:
                            show_plotly_chart(fig, use_container_width=True)
                            
                            # Also show gear shift map
                            st.subheader("Gear Shift Analysis")
                            c1, c2 = st.columns(2)
                            with c1:
                                fig_g1 = plot_gear_shift_trace(session, drv1)
                                if fig_g1: show_plotly_chart(fig_g1, use_container_width=True)
                            with c2:
                                fig_g2 = plot_gear_shift_trace(session, drv2)
                                if fig_g2: show_plotly_chart(fig_g2, use_container_width=True)
                        else:
                            st.error("Could not generate telemetry plot")
            else:
                st.info("No session data")
        except Exception as e:
            st.error(f"Error: {e}")

    # TAB 10: RACE CONTROL
    with analysis_tabs[10]:
        st.subheader("Race Control Events")
        try:
             rc_msgs = get_race_control_messages(session)
             if not rc_msgs.empty:
                 # Color code flags
                 def highlight_flag(val):
                     color = ''
                     val_str = str(val).upper()
                     if 'RED' in val_str: color = 'background-color: #ff4b4b; color: white'
                     elif 'YELLOW' in val_str: color = 'background-color: #fca130; color: black'
                     elif 'GREEN' in val_str: color = 'background-color: #09ab3b; color: white'
                     elif 'BLACK' in val_str: color = 'background-color: black; color: white'
                     elif 'BLUE' in val_str: color = 'background-color: #0068c9; color: white'
                     return color

                 st.dataframe(rc_msgs.style.map(highlight_flag, subset=['Flag']), 
                     use_container_width=True, hide_index=True)
             else:
                 st.info("No race control messages available.")
        except Exception as e:
            st.error(f"Error loading race control: {e}")

    # TAB 11: ENGINEERING
    with analysis_tabs[11]:
        st.subheader("Engineering & Reliability")
        eng_tabs = st.tabs(["Pit Stop Details", "Tyre Degradation"])
        
        with eng_tabs[0]:
            st.markdown("#### Pit Stop Analysis")
            try:
                pit_detailed = get_detailed_pit_analysis(session)
                if not pit_detailed.empty:
                    st.dataframe(pit_detailed, use_container_width=True, hide_index=True)
                    
                    # Pit Duration Distribution
                    fig_pit = px.histogram(pit_detailed, x="Duration", nbins=20, 
                        title="Pit Stop Duration Distribution", color_discrete_sequence=['#00D2BE'])
                    fig_pit.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    show_plotly_chart(fig_pit, use_container_width=True)
                else:
                    st.info("No pit detail data.")
            except Exception as e:
                st.error(f"Error in pit analysis: {e}")

        with eng_tabs[1]:
             st.markdown("#### Tyre Life Estimation")
             try:
                 laps = session.laps
                 if laps is not None and not laps.empty:
                    drivers = st.multiselect("Select Drivers", sorted(laps['Driver'].unique()), default=[laps['Driver'].iloc[0]], key="eng_tyre_drv")
                    if drivers:
                        fig_deg = go.Figure()
                        for d in drivers:
                            d_laps = laps.pick_driver(d).pick_quicklaps()
                            if not d_laps.empty:
                                fig_deg.add_trace(go.Scatter(x=d_laps['TyreLife'], y=d_laps['LapTime'].dt.total_seconds(),
                                    mode='markers', name=d))
                        fig_deg.update_layout(title="Lap Time vs Tyre Life", xaxis_title="Tyre Life (Laps)", 
                            yaxis_title="Time (s)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        show_plotly_chart(fig_deg, use_container_width=True)
             except Exception as e:
                 st.error(f"Error in tyre degradation: {e}")


def render_prediction_tab(df, total_points_combined=None):
    """AI Race Predictions tab content."""
    st.header("AI Race Predictor")
    
    if df is None or df.empty:
        st.error("No data available")
        return
    
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.warning("Model not trained. Please train the model first (see documentation).")
        return

    # Get next race details from schedule
    try:
        schedule = get_fastf1_schedule_cached(get_active_season_year())
        if schedule['EventDate'].dt.tz is None:
             schedule['EventDate'] = schedule['EventDate'].dt.tz_localize('UTC')
        
        now = pd.Timestamp.now(tz='UTC')
        upcoming = schedule[schedule['EventDate'] >= (now - timedelta(days=3))].head(1)
        
        if not upcoming.empty:
            next_event = upcoming.iloc[0]
            race_name = next_event['EventName']
            st.info(f"Predicting for: {race_name}")
        else:
            race_name = "Upcoming Race"
            st.info("Predicting for next available race.")
            
    except Exception:
        race_name = "Next Race"

    # Prediction tabs
    pred_tabs = st.tabs(["Full Grid Prediction", "Model Insights"])
    
    with pred_tabs[0]:
        st.subheader(f"Predicted Results - {race_name}")
        priors_preview = load_preseason_prediction_priors()
        has_preseason_priors = isinstance(priors_preview, pd.DataFrame) and not priors_preview.empty
        is_2026_mode = get_active_season_year() == 2026
        pred_state_key = f"prediction_results_df_{get_active_season_year()}"
        pred_meta_key = f"prediction_results_meta_{get_active_season_year()}"
        rendered_prediction_results = False

        c1, c2 = st.columns([2, 2])
        with c1:
            preseason_weight = st.slider(
                "Pre-season prior weight (scenario overlay)",
                min_value=0.0,
                max_value=0.35,
                value=0.12 if (is_2026_mode and has_preseason_priors) else 0.0,
                step=0.01,
                key="pred_preseason_weight",
                help="Mengatur bobot overlay skenario (Bahrain pace/mileage + Spain shakedown status). Tidak mengubah prediksi model dasar.",
            )
        with c2:
            use_weighted_sort = st.checkbox(
                "Sort by weighted scenario rank",
                value=bool(is_2026_mode and has_preseason_priors and preseason_weight > 0),
                key="pred_use_weighted_sort",
                help="Jika aktif, tabel diurutkan berdasarkan skor gabungan model + prior pre-season (experimental).",
            )
        if has_preseason_priors:
            st.caption(
                "Pre-season priors tersedia dari Bahrain official tests (pace + mileage) dan Spain/Barcelona shakedown participation metadata. "
                "Baseline model ranking tetap ditampilkan sebagai `Rank`; skenario overlay muncul sebagai `Scenario_Rank`."
            )
        else:
            st.caption("Pre-season prior signal belum masuk untuk sesi ini. Prediksi tetap berjalan dengan baseline model.")
        
        if st.button("Generate Predictions", type="primary", key="gen_pred_btn"):
            with st.spinner("Running Gradient Boosting model..."):
                # Prepare input data
                drivers = sorted(df['Driver'].unique().tolist())
                driver_stats = calculate_driver_stats(df)
                
                predictions = []
                
                # Get encoders if available (would need to load them, but assuming prepare_features handles it or we recreate basic encoding for display)
                # Ideally we use the pipeline from model.py. 
                # For now, we'll construct the feature DF and assume encoders are handled or we pass raw if pipeline supports it.
                # Actually, prepare_features(train_mode=False) loads encoders.
                
                # Live Data Injection
                real_grid = get_real_grid_positions(get_active_season_year(), race_name)
                using_live = False
                
                if real_grid:
                    st.success(f"Using Live Qualifying Grid for {race_name}")
                    using_live = True
                else:
                    st.warning("Qualifying data unavailable. Using historical averages (accuracy lower).")

                # Create a dataframe for all drivers for the next race
                pred_rows = []
                for driver in drivers:
                    driver_data = df[df['Driver'] == driver]
                    if driver_data.empty:
                        continue
                        
                    team = driver_data['Team'].iloc[0]
                    
                    # Estimate grid position from season average
                    if driver_stats is not None and driver in driver_stats.index:
                        # avg_grid not in driver_stats by default, let's calculate it on the fly
                        avg_grid = driver_data['Starting Grid'].mean() if 'Starting Grid' in driver_data.columns else 10
                    else:
                        avg_grid = 10
                    
                    # INJECT LIVE GRID
                    if using_live:
                         start_pos = real_grid.get(driver, avg_grid)
                    else:
                         start_pos = round(avg_grid)

                    pred_rows.append({
                        'Driver': driver,
                        'Team': team,
                        'Starting Grid': start_pos,
                        'Track': race_name,
                        'Finished': True # Dummy for feature prep
                    })
                
                pred_df = pd.DataFrame(pred_rows)
                
                try:
                    include_preseason_for_inference = False
                    schema_path = Path("models") / "feature_columns.json"
                    if schema_path.exists():
                        try:
                            feature_schema = json.loads(schema_path.read_text(encoding="utf-8"))
                            include_preseason_for_inference = any(
                                str(col).startswith("preseason_") for col in (feature_schema or [])
                            )
                        except Exception:
                            include_preseason_for_inference = False

                    # Transform features
                    X_pred, _, _ = prepare_features(
                        pred_df,
                        train_mode=False,
                        include_preseason_features=include_preseason_for_inference,
                    )
                    
                    # Predict
                    y_pred = model.predict(X_pred)
                    
                    pred_df['Predicted_Position'] = y_pred
                    pred_df = pred_df.sort_values('Predicted_Position').reset_index(drop=True)
                    pred_df['Rank'] = range(1, len(pred_df) + 1)
                    pred_df = _augment_predictions_with_preseason_signals(
                        pred_df,
                        using_live_grid=using_live,
                        preseason_weight=preseason_weight,
                        use_weighted_sort=use_weighted_sort and preseason_weight > 0,
                    )
                    st.session_state[pred_state_key] = pred_df.copy()
                    st.session_state[pred_meta_key] = {
                        "race_name": race_name,
                        "using_live": bool(using_live),
                        "preseason_weight": float(preseason_weight),
                        "use_weighted_sort": bool(use_weighted_sort and preseason_weight > 0),
                    }
                    _render_prediction_results_panel(
                        pred_df,
                        using_live=using_live,
                        preseason_weight=float(preseason_weight),
                        use_weighted_sort=bool(use_weighted_sort and preseason_weight > 0),
                        race_name=race_name,
                    )
                    rendered_prediction_results = True
                    
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.caption("Ensure categorical encoders (Driver, Team, Track) are generated via training.")

        if not rendered_prediction_results:
            cached_pred_df = st.session_state.get(pred_state_key)
            cached_meta = st.session_state.get(pred_meta_key, {})
            if isinstance(cached_pred_df, pd.DataFrame) and not cached_pred_df.empty:
                cached_race_name = str(cached_meta.get("race_name", race_name))
                st.caption(
                    f"Showing last generated predictions for `{cached_race_name}`. "
                    "Generate again to refresh with current settings/live grid."
                )
                _render_prediction_results_panel(
                    cached_pred_df.copy(),
                    using_live=bool(cached_meta.get("using_live", False)),
                    preseason_weight=float(cached_meta.get("preseason_weight", preseason_weight)),
                    use_weighted_sort=bool(cached_meta.get("use_weighted_sort", False)),
                    race_name=cached_race_name,
                )

    with pred_tabs[1]:
        st.subheader("Prediction Factors")
        
        if hasattr(model, 'feature_importances_'):
            feature_names = None
            if hasattr(model, 'feature_names_in_'):
                try:
                    feature_names = [str(x) for x in list(model.feature_names_in_)]
                except Exception:
                    feature_names = None
            if not feature_names:
                schema_path = Path("models") / "feature_columns.json"
                if schema_path.exists():
                    try:
                        feature_names = json.loads(schema_path.read_text(encoding="utf-8"))
                    except Exception:
                        feature_names = None
            if not feature_names:
                feature_names = [f"Feature {i+1}" for i in range(len(model.feature_importances_))]
            
            if len(model.feature_importances_) == len(feature_names):
                importances = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importances, x='Importance', y='Feature', orientation='h',
                             title="Model Feature Importance")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                show_plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Key Factors:**
                * **Starting Grid:** Historical data shows qualifying performance is the strongest predictor.
                * **Driver/Team:** Adjusts for car performance relative to the field.
                * **Track:** Accounts for circuit-specific performance characteristics.
                * **Pre-season Testing (2026 mode):** Bahrain test pace/mileage and Spain shakedown participation can be used as early-season priors.
                """)
            else:
                st.info("Feature importance details unavailable.")


def render_telemetry_tab(df):
    """Live Telemetry Monitor tab content."""
    st.header("Live Telemetry Monitor")
    st.markdown("Engineer-style telemetry display with detailed car data")
    
    # Race and driver selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        available_races = get_season_race_choices()
        if not available_races:
            st.warning("No race schedule available for the selected season yet.")
            return
        selected_race = st.selectbox("Select Grand Prix", available_races, key="telem_race")
    
    with col2:
        session_type = st.selectbox("Session", ["Race", "Qualifying", "Sprint"], key="telem_session")
    
    with col3:
        if isinstance(df, pd.DataFrame) and not df.empty and 'Driver' in df.columns:
            drivers = sorted(df['Driver'].dropna().astype(str).unique().tolist())
        else:
            drivers = sorted([d for d in DRIVER_PROFILES.keys() if isinstance(d, str) and d.strip()])
        if not drivers:
            st.warning("No driver list available yet for this season.")
            return
        selected_driver = st.selectbox("Select Driver", drivers, key="telem_driver")

    active_year = get_active_season_year()
    telem_sig = f"{active_year}|{selected_race}|{session_type}|{selected_driver}"
    telem_load_clicked = st.button("Load Telemetry", type="primary", key="telem_load_btn")
    if telem_load_clicked:
        st.session_state["telemetry_loaded_sig"] = telem_sig

    if st.session_state.get("telemetry_loaded_sig") != telem_sig:
        st.info("Select race/session/driver and click `Load Telemetry` to fetch telemetry traces.")
        return

    st.divider()
    
    with st.spinner("Loading telemetry data..."):
        session = get_fastf1_session_state_cached(
            year=active_year,
            race=selected_race,
            session_type=session_type,
            load_telemetry=True,
            cache_namespace="telemetry_fastf1",
            force_reload=bool(telem_load_clicked),
        )
        
        if session is None:
            st.error(f"Could not load session: {selected_race}")
            return
        
        # Get driver lap data
        try:
            # Get driver abbreviation
            driver_abbr = None
            for drv in session.drivers:
                drv_info = session.get_driver(drv)
                if selected_driver in str(drv_info.get('FullName', '')):
                    driver_abbr = drv_info.get('Abbreviation')
                    break
            
            if driver_abbr is None:
                # Try to match by last name
                for drv in session.drivers:
                    drv_info = session.get_driver(drv)
                    if selected_driver.split()[-1].upper() in str(drv_info.get('FullName', '')).upper():
                        driver_abbr = drv_info.get('Abbreviation')
                        break
            
            if driver_abbr is None:
                st.error(f"Could not find driver {selected_driver} in session")
                return
            
            # Get laps and car data
            driver_laps = session.laps.pick_driver(driver_abbr)
            
            if driver_laps.empty:
                st.error("No lap data available for this driver")
                return
            
            # Get fastest lap
            fastest_lap = driver_laps.pick_fastest()
            
            if fastest_lap is None or fastest_lap.empty:
                st.warning("No valid laps found, using first lap")
                fastest_lap = driver_laps.iloc[0] if len(driver_laps) > 0 else None
            
            if fastest_lap is None:
                st.error("Could not get lap data")
                return
            
            # Get telemetry
            telemetry = fastest_lap.get_telemetry()
            
            if telemetry is None or telemetry.empty:
                st.error("No telemetry data available")
                return
            
            # Get team color
            team = df[df['Driver'] == selected_driver]['Team'].iloc[0] if selected_driver in df['Driver'].values else "Unknown"
            team_color = TEAM_COLORS.get(team, '#E10600')
            
            # Header with lap info
            lap_time = format_f1_time(fastest_lap.get('LapTime', pd.NaT))
            lap_number = fastest_lap.get('LapNumber', 'N/A')
            compound = fastest_lap.get('Compound', 'N/A')
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {team_color}33 0%, #1a1a2e 100%); 
                        padding: 20px; border-radius: 15px; border: 2px solid {team_color}; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-around; text-align: center;">
                    <div>
                        <h4 style="color: #888; margin: 0;">Driver</h4>
                        <h2 style="color: white; margin: 5px 0;">{selected_driver}</h2>
                    </div>
                    <div>
                        <h4 style="color: #888; margin: 0;">Lap Time</h4>
                        <h2 style="color: {team_color}; margin: 5px 0;">{lap_time}</h2>
                    </div>
                    <div>
                        <h4 style="color: #888; margin: 0;">Lap</h4>
                        <h2 style="color: white; margin: 5px 0;">{lap_number}</h2>
                    </div>
                    <div>
                        <h4 style="color: #888; margin: 0;">Compound</h4>
                        <h2 style="color: white; margin: 5px 0;">{compound}</h2>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Gauge row
            st.subheader("Real-time Gauges")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Get latest values
            latest_speed = telemetry['Speed'].iloc[-1] if 'Speed' in telemetry.columns else 0
            latest_throttle = telemetry['Throttle'].iloc[-1] if 'Throttle' in telemetry.columns else 0
            latest_brake = telemetry['Brake'].iloc[-1] if 'Brake' in telemetry.columns else 0
            latest_rpm = telemetry['RPM'].iloc[-1] if 'RPM' in telemetry.columns else 0
            
            max_speed = telemetry['Speed'].max() if 'Speed' in telemetry.columns else 350
            max_rpm = 15000
            
            with col1:
                fig = create_gauge(latest_speed, max_speed, "SPEED (km/h)", team_color)
                show_plotly_chart(fig, use_container_width=True, key="speed_gauge")
            
            with col2:
                fig = create_gauge(latest_throttle, 100, "THROTTLE (%)", "#00FF00")
                show_plotly_chart(fig, use_container_width=True, key="throttle_gauge")
            
            with col3:
                fig = create_gauge(latest_brake, 100, "BRAKE (%)", "#FF0000")
                show_plotly_chart(fig, use_container_width=True, key="brake_gauge")
            
            with col4:
                fig = create_gauge(latest_rpm, max_rpm, "RPM", "#FFD700")
                show_plotly_chart(fig, use_container_width=True, key="rpm_gauge")
            
            st.divider()
            
            # Speed trace
            st.subheader("Speed Trace")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=telemetry['Distance'] if 'Distance' in telemetry.columns else telemetry.index,
                y=telemetry['Speed'] if 'Speed' in telemetry.columns else [],
                mode='lines',
                name='Speed',
                line=dict(color=team_color, width=2),
                fill='tozeroy',
                fillcolor=f'rgba{tuple(list(int(team_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}'
            ))
            
            fig.update_layout(
                height=300,
                xaxis_title="Distance (m)",
                yaxis_title="Speed (km/h)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            show_plotly_chart(fig, use_container_width=True)
            
            # Throttle/Brake trace
            st.subheader("Throttle & Brake Trace")
            
            fig = go.Figure()
            
            x_axis = telemetry['Distance'] if 'Distance' in telemetry.columns else telemetry.index
            
            if 'Throttle' in telemetry.columns:
                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=telemetry['Throttle'],
                    mode='lines',
                    name='Throttle',
                    line=dict(color='#00FF00', width=2)
                ))
            
            if 'Brake' in telemetry.columns:
                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=telemetry['Brake'] * 100 if telemetry['Brake'].max() <= 1 else telemetry['Brake'],
                    mode='lines',
                    name='Brake',
                    line=dict(color='#FF0000', width=2)
                ))
            
            fig.update_layout(
                height=250,
                xaxis_title="Distance (m)",
                yaxis_title="Input (%)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            show_plotly_chart(fig, use_container_width=True)
            
            # Gear and DRS
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Gear Usage")
                if 'nGear' in telemetry.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_axis,
                        y=telemetry['nGear'],
                        mode='lines',
                        name='Gear',
                        line=dict(color='#FFD700', width=2)
                    ))
                    fig.update_layout(
                        height=200,
                        xaxis_title="Distance (m)",
                        yaxis_title="Gear",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    show_plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("DRS Status")
                if 'DRS' in telemetry.columns:
                    fig = go.Figure()
                    drs_values = telemetry['DRS'].apply(lambda x: 1 if x > 0 else 0)
                    fig.add_trace(go.Scatter(
                        x=x_axis,
                        y=drs_values,
                        mode='lines',
                        name='DRS',
                        line=dict(color='#00BFFF', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0,191,255,0.3)'
                    ))
                    fig.update_layout(
                        height=200,
                        xaxis_title="Distance (m)",
                        yaxis_title="DRS Active",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    show_plotly_chart(fig, use_container_width=True)
            
            # Track Map with Speed
            st.subheader("Track Map - Speed Visualization")
            
            if 'X' in telemetry.columns and 'Y' in telemetry.columns:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=telemetry['X'],
                    y=telemetry['Y'],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=telemetry['Speed'] if 'Speed' in telemetry.columns else 'white',
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title='Speed (km/h)')
                    ),
                    hovertemplate='Speed: %{marker.color:.0f} km/h<extra></extra>'
                ))
                
                fig.update_layout(
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False, scaleanchor='x'),
                    showlegend=False
                )
                show_plotly_chart(fig, use_container_width=True)
            else:
                st.info("Track position data not available for this lap")
            
        except Exception as e:
            st.error(f"Error loading telemetry: {e}")
            logger.error(f"Telemetry error: {e}", exc_info=True)


def render_analysis_performance_lab_tab(df) -> None:
    st.header("Performance Lab")
    st.markdown("Interactive analysis dashboard for pace, finishing efficiency, and pre-season intelligence.")

    priors_df = load_preseason_prediction_priors()
    if df is None or df.empty:
        st.info("Season race results are not loaded yet. Showing pre-season intelligence and 2026 rollout signals.")
        c1, c2 = st.columns(2)
        with c1:
            if isinstance(priors_df, pd.DataFrame) and not priors_df.empty:
                pri = priors_df.copy().sort_values("preseason_prior_score", ascending=False)
                fig = px.bar(
                    pri,
                    x="preseason_prior_score",
                    y="team_norm",
                    orientation="h",
                    color="preseason_prior_coverage",
                    title="Pre-season Team Signal (ML priors)",
                    color_continuous_scale="Tealgrn",
                )
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    height=360,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    margin=dict(l=10, r=10, t=55, b=20),
                )
                show_plotly_chart(fig, use_container_width=True)
            else:
                st.info("Pre-season ML priors are not available.")
        with c2:
            updates_df = pd.DataFrame(OFFICIAL_2026_UPDATES)
            if not updates_df.empty and "category" in updates_df.columns:
                cat = updates_df["category"].value_counts().reset_index()
                cat.columns = ["Category", "Updates"]
                fig = px.pie(cat, names="Category", values="Updates", hole=0.45, title="2026 Official Update Categories")
                fig.update_layout(
                    height=360,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    margin=dict(l=10, r=10, t=55, b=10),
                )
                show_plotly_chart(fig, use_container_width=True)
            else:
                st.info("Official updates list is loading.")
        return

    work_df = df.copy()
    if "Position" not in work_df.columns:
        st.warning("Position column is missing from the loaded season dataset.")
        return

    f1, f2, f3 = st.columns([1.2, 1.2, 2.2])
    with f1:
        teams = sorted([t for t in work_df["Team"].dropna().unique().tolist() if t]) if "Team" in work_df.columns else []
        team_sel = st.selectbox("Team Filter", ["All Teams"] + teams, key="analysis_lab_team_filter")
    with f2:
        metric_mode = st.selectbox("Chart Focus", ["Grid vs Finish", "Points by Team", "Position Consistency"], key="analysis_lab_metric_mode")
    with f3:
        track_opts = [str(t) for t in work_df["Track"].dropna().unique().tolist()] if "Track" in work_df.columns else []
        selected_tracks = st.multiselect("Track Filter", track_opts, default=track_opts, key="analysis_lab_track_filter")

    if team_sel != "All Teams" and "Team" in work_df.columns:
        work_df = work_df[work_df["Team"] == team_sel]
    if selected_tracks and "Track" in work_df.columns:
        work_df = work_df[work_df["Track"].astype(str).isin(selected_tracks)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", len(work_df))
    m2.metric("Drivers", work_df["Driver"].nunique() if "Driver" in work_df.columns else 0)
    m3.metric("Teams", work_df["Team"].nunique() if "Team" in work_df.columns else 0)
    avg_finish = pd.to_numeric(work_df["Position"], errors="coerce").mean() if len(work_df) else np.nan
    m4.metric("Avg Finish", f"{avg_finish:.2f}" if pd.notna(avg_finish) else "N/A")

    if metric_mode == "Grid vs Finish" and "Starting Grid" in work_df.columns:
        fig = px.scatter(
            work_df,
            x="Starting Grid",
            y="Position",
            color="Team" if "Team" in work_df.columns else None,
            hover_data=[c for c in ["Driver", "Track", "Points"] if c in work_df.columns],
            title="Grid Position vs Finish Position",
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(l=10, r=10, t=55, b=30),
        )
        show_plotly_chart(fig, use_container_width=True)
    elif metric_mode == "Points by Team" and "Team" in work_df.columns and "Points" in work_df.columns:
        pts = work_df.groupby("Team", dropna=False)["Points"].sum().reset_index().sort_values("Points", ascending=False)
        fig = px.bar(pts, x="Team", y="Points", color="Team", title="Points by Team")
        fig.update_layout(
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
            margin=dict(l=10, r=10, t=55, b=40),
            xaxis_title="",
        )
        show_plotly_chart(fig, use_container_width=True)
    elif metric_mode == "Position Consistency" and "Driver" in work_df.columns:
        fig = px.box(work_df, x="Driver", y="Position", color="Team" if "Team" in work_df.columns else None, title="Finishing Position Consistency")
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(l=10, r=10, t=55, b=40),
            xaxis_title="",
        )
        show_plotly_chart(fig, use_container_width=True)
    else:
        st.info("The selected analysis view requires data columns that are not available for the current dataset.")

    with st.expander("Filtered Analysis Table"):
        st.dataframe(work_df.head(300), hide_index=True, use_container_width=True)


def render_live_control_tab() -> None:
    st.header("Live Control")
    st.markdown("Countdown, session clock, and signal monitor in one control-room panel.")

    active_year = get_active_season_year()
    now_utc = pd.Timestamp.now(tz="UTC")
    next_race = get_next_race_countdown_summary(active_year)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("System UTC", now_utc.strftime("%H:%M:%S"))
    c2.metric("User Offset", f"UTC{get_user_utc_offset()}")
    c3.metric("Active Season", str(active_year))
    c4.metric("Next Race", _countdown_text_precise(next_race.get("race_time"), now_utc) if next_race.get("ok") else "N/A")

    if next_race.get("ok"):
        st.markdown(
            f"""
            <div style="border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:12px 14px; margin:6px 0 12px 0;
                        background:linear-gradient(135deg, rgba(20,24,34,0.95), rgba(32,38,50,0.92));">
                <div style="font-size:12px; color:#b8c2d4; text-transform:uppercase; letter-spacing:1px;">Live Countdown</div>
                <div style="font-size:18px; font-weight:800; color:#f2f5fb;">{next_race.get('event')}</div>
                <div style="font-size:13px; color:#c8d2e2;">{next_race.get('location')} | {format_user_time(next_race.get('race_time'), '%d %b %Y %H:%M:%S')}</div>
                <div style="margin-top:8px; font-size:15px; color:#bff8f2; font-weight:700;">{_countdown_text_precise(next_race.get('race_time'), now_utc)} until lights out</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    next_event_row = _find_schedule_event_row(active_year, next_race.get("event")) if next_race.get("ok") else None
    if next_event_row is not None:
        rows = []
        for key in ["Session1", "Session2", "Session3", "Session4", "Session5"]:
            date_key = f"{key}Date"
            if key in next_event_row.index and date_key in next_event_row.index and pd.notna(next_event_row[date_key]):
                dtv = pd.to_datetime(next_event_row[date_key], errors="coerce", utc=True)
                if pd.isna(dtv):
                    continue
                status = "Upcoming"
                if dtv <= now_utc <= dtv + timedelta(hours=2.5):
                    status = "Live"
                elif now_utc > dtv + timedelta(hours=2.5):
                    status = "Completed"
                rows.append(
                    {
                        "Session": str(next_event_row.get(key, key)),
                        f"Start (UTC{get_user_utc_offset()})": format_user_time(dtv, "%d %b %Y %H:%M:%S"),
                        "Countdown": _countdown_text_precise(dtv, now_utc),
                        "Status": status,
                    }
                )
        if rows:
            st.markdown("### Session Clock")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("### Signal Monitor")
    sm1, sm2 = st.columns([1, 3])
    with sm1:
        if st.button("Refresh Signals", key="live_control_refresh_signals"):
            st.session_state["live_control_refresh_nonce"] = st.session_state.get("live_control_refresh_nonce", 0) + 1
            st.rerun()
    with sm2:
        st.caption("Monitors data services and news feeds used by the dashboard (best coverage in 2026 mode).")

    nonce = int(st.session_state.get("live_control_refresh_nonce", 0))
    status_df = pd.DataFrame()
    news_df = pd.DataFrame()
    if active_year == 2026:
        try:
            api_snap = load_2026_api_snapshot(nonce)
            status_df = api_snap.get("status", pd.DataFrame()) if isinstance(api_snap, dict) else pd.DataFrame()
        except Exception as e:
            st.warning(f"Signal monitor (data services) unavailable: {e}")
        try:
            news_snap = load_2026_news_snapshot(nonce)
            news_df = news_snap.get("news", pd.DataFrame()) if isinstance(news_snap, dict) else pd.DataFrame()
        except Exception as e:
            st.warning(f"Signal monitor (news) unavailable: {e}")

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Sources Checked", len(status_df) if isinstance(status_df, pd.DataFrame) else 0)
    n2.metric("Sources Ready", int(status_df["ok"].fillna(False).sum()) if isinstance(status_df, pd.DataFrame) and not status_df.empty else 0)
    n3.metric("News Items", len(news_df) if isinstance(news_df, pd.DataFrame) else 0)
    n4.metric("Refresh Counter", nonce)

    if isinstance(status_df, pd.DataFrame) and not status_df.empty:
        plot_df = status_df.copy()
        if "source" not in plot_df.columns:
            if "api" in plot_df.columns:
                plot_df["source"] = plot_df["api"]
            else:
                plot_df["source"] = plot_df.index.astype(str)
        plot_df["latency_ms_num"] = pd.to_numeric(plot_df.get("latency_ms"), errors="coerce")
        plot_df["Status"] = np.where(plot_df["ok"].fillna(False), "Ready", "Pending")
        fig = px.bar(
            plot_df,
            x="source",
            y="latency_ms_num",
            color="Status",
            hover_data=[c for c in ["rows", "error"] if c in plot_df.columns],
            title="Data Service Latency",
            barmode="group",
        )
        fig.update_layout(
            height=330,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(l=10, r=10, t=55, b=30),
            xaxis_title="",
            yaxis_title="Latency (ms)",
        )
        show_plotly_chart(fig, use_container_width=True)
        with st.expander("Signal Monitor Table"):
            st.dataframe(status_df, hide_index=True, use_container_width=True)
    else:
        st.info("Signal monitor will populate when 2026 data services are enabled and reachable.")


def render_live_timing_tab():
    """Live Timing Session tab content."""
    st.header("Live Timing Session") # Removed emoji
    
    # Current time (UTC)
    now = pd.Timestamp.now(tz='UTC')
    st.markdown(f"**Current System Time (UTC):** {now.strftime('%Y-%m-%d %H:%M:%S')}")
    active_year = get_active_season_year()

    # Completed seasons should show archive/final mode instead of waiting for live sessions.
    if active_year < now.year:
        st.info(f"{active_year} season is completed. Live mode is switched to archive summary.")
        final_df = load_race_data(active_year)
        if final_df is None or final_df.empty:
            st.warning("Final season archive data is not available locally.")
            return
        race_only_df = final_df.copy()
        if "SessionType" in race_only_df.columns:
            race_only_df = race_only_df[race_only_df["SessionType"].astype(str).str.lower() != "sprint"].copy()
        if race_only_df.empty:
            race_only_df = final_df.copy()

        total_points = race_only_df.groupby("Driver")["Points"].sum().sort_values(ascending=False).reset_index()
        total_points.columns = ["Driver", "RacePoints"]
        sprint_df = load_sprint_data(active_year)
        if sprint_df is not None and not sprint_df.empty and {"Driver", "Points"}.issubset(sprint_df.columns):
            sprint_pts = sprint_df.groupby("Driver")["Points"].sum().reset_index()
            sprint_pts.columns = ["Driver", "SprintPoints"]
            total_points = total_points.merge(sprint_pts, on="Driver", how="left")
        else:
            total_points["SprintPoints"] = 0
        total_points["TotalPoints"] = total_points["RacePoints"].fillna(0) + total_points["SprintPoints"].fillna(0)
        total_points = total_points.sort_values("TotalPoints", ascending=False).reset_index(drop=True)

        team_map = race_only_df.groupby("Driver")["Team"].first().to_dict() if {"Driver", "Team"}.issubset(race_only_df.columns) else {}
        total_points["Team"] = total_points["Driver"].map(team_map).fillna("")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Champion", total_points.iloc[0]["Driver"] if not total_points.empty else "N/A")
        c2.metric("Points", int(total_points.iloc[0]["TotalPoints"]) if not total_points.empty else 0)
        c3.metric("Races", int(race_only_df["Track"].nunique()) if "Track" in race_only_df.columns else 0)
        c4.metric("Archive Mode", "Enabled")

        tab_a, tab_b, tab_c = st.tabs(["Final Standings", "Winners Timeline", "Archive Notes"])
        with tab_a:
            max_drivers_display = max(5, min(20, len(total_points)))
            default_drivers_display = min(10, max_drivers_display)
            top_n = st.slider("Drivers to display", 5, max_drivers_display, default_drivers_display, key="live_archive_top_n")
            show = total_points.head(top_n).copy()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=show["TotalPoints"],
                y=show["Driver"],
                orientation="h",
                marker_color=[TEAM_COLORS.get(team_map.get(d, ""), "#666666") for d in show["Driver"]],
                text=show["TotalPoints"].astype(int),
                textposition="outside"
            ))
            fig.update_layout(
                height=max(350, top_n * 32),
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Points",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=10, r=80, t=20, b=30)
            )
            show_plotly_chart(fig, use_container_width=True)
            st.dataframe(show[["Driver", "Team", "RacePoints", "SprintPoints", "TotalPoints"]], hide_index=True, use_container_width=True)
        with tab_b:
            if {"Track", "Driver", "Team", "Position"}.issubset(race_only_df.columns):
                winners = race_only_df[pd.to_numeric(race_only_df["Position"], errors="coerce") == 1].copy()
                if not winners.empty:
                    race_order = race_only_df["Track"].dropna().astype(str).drop_duplicates().tolist()
                    winners["Track"] = pd.Categorical(winners["Track"].astype(str), categories=race_order, ordered=True)
                    winners = winners.sort_values("Track").reset_index(drop=True)
                    winners["Round"] = winners.index + 1
                    fig = px.scatter(
                        winners,
                        x="Round",
                        y="Driver",
                        color="Team",
                        hover_data=["Track", "Points"],
                        title=f"{active_year} Grand Prix Winners Timeline"
                    )
                    fig.update_layout(
                        height=360,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        margin=dict(l=10, r=10, t=55, b=20)
                    )
                    show_plotly_chart(fig, use_container_width=True)
                    st.dataframe(winners[["Round", "Track", "Driver", "Team", "Points"]], hide_index=True, use_container_width=True)
                else:
                    st.info("Race winner archive is not available.")
            else:
                st.info("Required columns for winners timeline are missing.")
        with tab_c:
            st.markdown("- Use `Race Center` to load archived sessions (Race / Qualifying / Practice) from FastF1.")
            st.markdown("- Use `Analysis` for predictions and performance comparisons on final season results.")
            st.markdown("- Home and Season Stats now show final-season interactive summaries for completed years.")
        return
    
    # Get schedule for 2025
    try:
        schedule = get_fastf1_schedule_cached(active_year)
        
        # Ensure schedule dates are timezone-aware UTC for comparison
        if schedule['EventDate'].dt.tz is None:
             schedule['EventDate'] = schedule['EventDate'].dt.tz_localize('UTC')

        # Find next or current event (filter for events happening today or in future)
        # We look for events where the last session is in the future or today
        upcoming = schedule[schedule['EventDate'] >= (now - timedelta(days=3))].head(1)
        
        if not upcoming.empty:
            next_event = upcoming.iloc[0]
            st.subheader(f"Target Event: {next_event['EventName']}")
            st.write(f"Location: {next_event['Location']}")
            st.caption(f"Session times shown in UTC{get_user_utc_offset()}")
            
            # Display sessions
            sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
            
            schedule_data = []
            active_session_name = None
            
            for s in sessions:
                s_date_col = f'{s}Date'
                s_name_col = s
                
                if s_date_col in next_event and s_name_col in next_event:
                    s_date = next_event[s_date_col]
                    s_name = next_event[s_name_col]
                    
                    if pd.notna(s_date):
                        # Ensure s_date is timezone aware for comparison
                        if isinstance(s_date, pd.Timestamp) and s_date.tz is None:
                            s_date = s_date.tz_localize('UTC')

                        status = "Upcoming"
                        # Check if currently active (assuming 2h duration)
                        if isinstance(s_date, pd.Timestamp):
                            s_end = s_date + timedelta(hours=2)
                            if s_date <= now <= s_end:
                                status = "LIVE" # Removed emoji
                            elif now > s_end:
                                status = "Completed"
                        
                        schedule_data.append({
                            'Session': s_name, 
                            'Time': format_user_time(s_date, '%Y-%m-%d %H:%M') if pd.notna(s_date) else 'TBD',
                            'Status': status
                        })
            
            st.table(pd.DataFrame(schedule_data))
            
            # Race Week Notification - Removed emojis
            event_date = next_event['EventDate']
            if isinstance(event_date, pd.Timestamp):
                if event_date.tz is None:
                    event_date = event_date.tz_localize('UTC')
                
                days_diff = (event_date - now).days
                if -1 <= days_diff <= 7:
                    st.markdown(f"""
                    <div style="background-color: #E10600; color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px; overflow: hidden; white-space: nowrap;">
                        <div style="display: inline-block; animation: marquee 15s linear infinite; font-weight: bold; font-size: 1.2em;">
                            IT'S RACE WEEK! {next_event['EventName'].upper()} IS COMING UP! GET READY FOR THE ACTION!
                        </div>
                    </div>
                    <style>
                    @keyframes marquee {{
                        0% {{ transform: translateX(100%); }}
                        100% {{ transform: translateX(-100%); }}
                    }}
                    </style>
                    """, unsafe_allow_html=True)
            
            # Connection controls
            st.divider()
            st.markdown("### Live Data Connection")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                default_idx = 0
                if active_session_name:
                    try:
                        default_idx = [s['Session'] for s in schedule_data].index(active_session_name)
                    except:
                        pass
                session_sel = st.selectbox("Select Session to Monitor", [s['Session'] for s in schedule_data], index=default_idx)
            with col2:
                auto_refresh = st.checkbox("Auto-refresh (Simulated)", value=False)
                if auto_refresh:
                    st.caption("Page will refresh interaction to update data")
            with col3:
                connect_clicked = st.button("Connect / Refresh", type="primary", key="live_timing_connect_btn")

            conn_sig = f"{active_year}|{next_event['EventName']}|{session_sel}"
            if connect_clicked:
                st.session_state["live_timing_connection_sig"] = conn_sig
            elif auto_refresh and st.session_state.get("live_timing_connection_sig") == conn_sig:
                # keep current connection active during reruns
                pass

            if st.session_state.get("live_timing_connection_sig") != conn_sig:
                st.info("Select a session and click `Connect / Refresh` to start the live feed.")
                return

            if session_sel:
                with st.spinner(f"Connecting to {next_event['EventName']} - {session_sel}..."):
                    try:
                        # Attempt to load
                        session = get_fastf1_session_state_cached(
                            year=active_year,
                            race=next_event['EventName'],
                            session_type=session_sel,
                            load_telemetry=False,
                            cache_namespace="live_timing_fastf1",
                            force_reload=bool(auto_refresh or connect_clicked),
                        )
                        if session is None:
                            st.error("Could not connect to session feed.")
                            return
                        try:
                            session.load(telemetry=False, laps=True, weather=True, messages=True)
                        except Exception as e:
                            st.warning(f"Full live refresh unavailable (expected for some sessions): {e}")
                        
                        st.success(f"Connected to {session.name}")
                        
                        # Dashboard
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("Session Status", "Connected")
                        with c2:
                            if hasattr(session, 'weather_data') and session.weather_data is not None and not session.weather_data.empty:
                                temp = session.weather_data['TrackTemp'].iloc[-1]
                                st.metric("Track Temp", f"{temp:.1f}°C")
                            else:
                                st.metric("Track Temp", "N/A")
                        with c3:
                            if hasattr(session, 'drivers'):
                                st.metric("Drivers", len(session.drivers))
                            else:
                                st.metric("Drivers", "0")
                        with c4:
                            st.metric("Last Update", now.strftime('%H:%M:%S'))
                            
                        st.divider()
                        
                        # Leaderboard
                        st.subheader("Live Leaderboard")
                        if hasattr(session, 'results') and session.results is not None and not session.results.empty:
                            cols = ['Position', 'Abbreviation', 'Time', 'Status', 'Points']
                            valid_cols = [c for c in cols if c in session.results.columns]
                            st.dataframe(session.results[valid_cols], hide_index=True, use_container_width=True)
                        else:
                            st.info("Waiting for timing feed. Results will appear as soon as session data is published.")
                        
                        # Latest Messages
                        st.subheader("Race Control Messages")
                        messages = get_track_status(session)
                        if not messages.empty:
                            st.dataframe(messages.tail(10).sort_values('Time', ascending=False), hide_index=True, use_container_width=True)
                        else:
                            st.info("No messages received.")
                            
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
                        st.info("Note: FastF1 requires the session to be started to fetch data.")
        else:
            st.info(f"No upcoming events found for {get_active_season_label()}. Calendar sync may still be updating.")
            
    except Exception as e:
        st.error(f"Error loading schedule: {e}")


def render_qualifying_tab():
    """Qualifying Analysis tab content."""
    st.header("Qualifying Deep Dive")
    st.markdown("Analyze qualifying performance, lap evolution, and sector dominance.")
    
    col1, col2 = st.columns(2)
    with col1:
        q_races = get_season_race_choices()
        if not q_races:
            st.warning("No race schedule available for the selected season yet.")
            return
        q_race = st.selectbox("Select Race for Qualifying Analysis", q_races, key="q_race")

    q_year = get_active_season_year()
    quali_sig = f"{q_year}|{q_race}|Qualifying"
    analyze_quali_clicked = st.button("Analyze Qualifying", type="primary", key="analyze_quali_btn")
    if analyze_quali_clicked:
        st.session_state["qualifying_loaded_sig"] = quali_sig

    if st.session_state.get("qualifying_loaded_sig") != quali_sig:
        st.info("Select a race and click `Analyze Qualifying` to load qualifying visuals.")
        return

    with st.spinner(f"Loading {q_race} Qualifying Data..."):
            session = get_fastf1_session_state_cached(
                year=q_year,
                race=q_race,
                session_type="Qualifying",
                load_telemetry=False,
                cache_namespace="qualifying_fastf1",
                force_reload=bool(analyze_quali_clicked),
            )
            
            if session:
                st.divider()
                
                # 1. Evolution Plot
                st.subheader("Lap Time Evolution")
                fig_evo = plot_qualifying_evolution(session)
                if fig_evo: show_plotly_chart(fig_evo, use_container_width=True)
                else: st.info("Evolution data unavailable")
                
                col1, col2 = st.columns(2)
                
                # 2. Gap to Pole
                with col1:
                    st.subheader("Gap to Pole")
                    fig_gap = plot_qualifying_gap(session)
                    if fig_gap: show_plotly_chart(fig_gap, use_container_width=True)
                    
                # 3. Sector Dominance
                with col2:
                    st.subheader("Sector Dominance")
                    fig_sec = plot_sector_dominance(session)
                    if fig_sec: show_plotly_chart(fig_sec, use_container_width=True)
            else:
                st.error("Could not load session")


def render_teammate_battle_tab(df):
    """Teammate Battle tab content."""
    st.header("Teammate Head-to-Head Wars")
    st.markdown("Comparing performance between teammates across the season.")
    
    if df is None or df.empty:
        st.error("No season data available")
        return
        
    comparison_df = calculate_teammate_comparison(df)
    
    if comparison_df.empty:
        st.warning("Not enough data to generate comparisons.")
        return
        
    # Team Selector
    teams = sorted(comparison_df['Team'].unique())
    selected_team = st.selectbox("Select Team to Compare", teams, key="tm_comp_team")
    
    if selected_team:
        team_data_row = comparison_df[comparison_df['Team'] == selected_team].iloc[0] # Use a different var name
        
        d1 = team_data_row['Driver 1']
        d2 = team_data_row['Driver 2']
        
        st.divider()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.markdown(f"<h2 style='text-align: center; color: {TEAM_COLORS.get(selected_team, 'white')}'>{d1}</h2>", unsafe_allow_html=True)
            prof1 = DRIVER_PROFILES.get(d1, {})
            if prof1.get('image_url'):
                st.image(prof1['image_url'], use_container_width=True)
                
        with col2:
            st.markdown("<h1 style='text-align: center;'>VS</h1>", unsafe_allow_html=True)
            
            comp_metrics = [
                ("Races Together", team_data_row['Races Together'], team_data_row['Races Together']),
                ("Race Head-to-Head", team_data_row['D1 Race Wins'], team_data_row['D2 Race Wins']),
                ("Quali Head-to-Head", team_data_row['Quali H2H'].split(' - ')[0], team_data_row['Quali H2H'].split(' - ')[1]),
                ("Total Points", int(team_data_row['Pts 1']), int(team_data_row['Pts 2']))
            ]
            
            for label, v1, v2 in comp_metrics:
                c_a, c_b, c_c = st.columns([1, 2, 1])
                with c_a: st.markdown(f"<h3 style='text-align: right;'>{v1}</h3>", unsafe_allow_html=True)
                with c_b: st.markdown(f"<p style='text-align: center; color: #888;'>{label}</p>", unsafe_allow_html=True)
                with c_c: st.markdown(f"<h3 style='text-align: left;'>{v2}</h3>", unsafe_allow_html=True)
                st.divider()

        with col3:
            st.markdown(f"<h2 style='text-align: center; color: {TEAM_COLORS.get(selected_team, 'white')}'>{d2}</h2>", unsafe_allow_html=True)
            prof2 = DRIVER_PROFILES.get(d2, {})
            if prof2.get('image_url'):
                st.image(prof2['image_url'], use_container_width=True)
                
        with st.expander("View All Teammate Comparisons"):
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)


def render_official_plots_tab():
    """Official F1 Style Plots tab content."""
    st.header("Official F1 Style Plots")
    st.markdown("High-quality static visualizations using FastF1's matplotlib integration.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_races = get_season_race_choices()
        if not plot_races:
            st.warning("No race schedule available for the selected season yet.")
            return
        plot_race = st.selectbox("Select Race", plot_races, key="plot_race")
    with col2:
        plot_session = st.selectbox("Session Type", ["Race", "Qualifying"], key="plot_session")
    with col3:
        st.markdown("Driver specific plots:")
        plot_driver = st.text_input("Driver Code (e.g., VER, HAM)", value="VER", key="plot_driver")
        
    plot_driver = plot_driver.strip().upper() or "VER"
    plot_year = get_active_season_year()
    plot_sig = f"{plot_year}|{plot_race}|{plot_session}|{plot_driver}"
    generate_plots_clicked = st.button("Generate Plots", type="primary", key="generate_official_plots_btn")
    if generate_plots_clicked:
        st.session_state["official_plots_loaded_sig"] = plot_sig

    if st.session_state.get("official_plots_loaded_sig") != plot_sig:
        st.info("Select race/session/driver and click `Generate Plots` to render this dashboard.")
        return

    with st.spinner("Generating plots..."):
            session = get_fastf1_session_state_cached(
                year=plot_year,
                race=plot_race,
                session_type=plot_session,
                load_telemetry=False,
                cache_namespace="official_plots_fastf1",
                force_reload=bool(generate_plots_clicked),
            )
            if session:
                st.subheader("Circuit with Corners")
                fig1 = plot_circuit_with_corners(session)
                if fig1: st.pyplot(fig1)
                
                st.subheader("Team Pace Comparison")
                fig2 = plot_team_pace_comparison(session)
                if fig2: st.pyplot(fig2)
                
                st.subheader("Tyre Strategy Summary")
                fig3 = plot_tyre_strategy_summary(session)
                if fig3: st.pyplot(fig3)
                
                st.subheader(f"Speed Visualization ({plot_driver})")
                fig4 = plot_speed_on_track(session, plot_driver)
                if fig4: st.pyplot(fig4)
                
                st.subheader(f"Gear Shift Visualization ({plot_driver})")
                fig5 = plot_gear_shift_on_track(session, plot_driver)
                if fig5: st.pyplot(fig5)
            else:
                st.error("Could not load session.")


def main():
    """Main application entry point."""
    if "sidebar_season_mode" not in st.session_state:
        st.session_state["sidebar_season_mode"] = "2026"

    render_header()
    
    active_year = get_active_season_year()

    # Load data
    df = load_race_data(active_year)
    
    # Calculate total points properly (race + sprint)
    total_points_combined = {}
    total_laps_all = 0
    total_all_points = 0
    if df is not None and not df.empty:
        data_dir = Path(__file__).parent.parent / 'data'
        try:
            race_file = data_dir / f'Formula1_{active_year}Season_RaceResults.csv'
            sprint_file = data_dir / f'Formula1_{active_year}Season_SprintResults.csv'
            race_df = pd.read_csv(race_file)
            sprint_df = pd.read_csv(sprint_file) if sprint_file.exists() else pd.DataFrame(columns=['Driver', 'Points'])
            
            race_points = race_df.groupby('Driver')['Points'].sum()
            sprint_points = sprint_df.groupby('Driver')['Points'].sum() if not sprint_df.empty else pd.Series(dtype=float)
            total_points_combined = race_points.add(sprint_points, fill_value=0).to_dict()
            total_laps_all = race_df['Laps'].sum()
            total_all_points = sum(total_points_combined.values())
        except:
            total_points_combined = df.groupby('Driver')['Points'].sum().to_dict()
            total_laps_all = df['Laps'].sum() if 'Laps' in df.columns else 0
            total_all_points = sum(total_points_combined.values())
    
    # Sidebar
    with st.sidebar:
        active_year_sidebar = get_active_season_year()
        badge_label = "2026 Live" if active_year_sidebar == 2026 else "2025"
        badge_bg = "rgba(0,210,190,0.16)" if active_year_sidebar == 2026 else "rgba(245,185,66,0.16)"
        badge_fg = "#bff8f2" if active_year_sidebar == 2026 else "#ffe5a8"
        badge_border = "rgba(0,210,190,0.40)" if active_year_sidebar == 2026 else "rgba(245,185,66,0.40)"
        # Removed emoji from image source
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/512px-F1.svg.png", width=80) 
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-top:4px;">
                <div style="font-size:1.35rem; font-weight:700;">F1 Lab</div>
                <div style="padding:4px 10px; border:1px solid {badge_border}; background:{badge_bg}; border-radius:999px; font-size:0.75rem; font-weight:700; color:{badge_fg}; white-space:nowrap;">
                    {badge_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")

        st.markdown("**Season Menu**")
        _season_mode = st.radio(
            "Season",
            ["2025", "2026"],
            key="sidebar_season_mode"
        )

        default_utc = st.session_state.get("user_utc_offset", "-05:00")
        if default_utc not in UTC_OFFSET_OPTIONS:
            default_utc = "-05:00"
        st.selectbox(
            "User UTC Offset",
            UTC_OFFSET_OPTIONS,
            index=UTC_OFFSET_OPTIONS.index(default_utc),
            key="user_utc_offset",
            help="Used to convert race, schedule, and news timestamps across the dashboard."
        )

        next_race_sidebar = get_next_race_countdown_summary(active_year_sidebar)
        if next_race_sidebar.get("ok"):
            race_week_badge = (
                '<span style="padding:3px 8px; border-radius:999px; background:rgba(225,6,0,0.16); color:#ffd1ce; border:1px solid rgba(225,6,0,0.35); font-size:11px; font-weight:700;">RACE WEEK</span>'
                if next_race_sidebar.get("is_race_week")
                else '<span style="padding:3px 8px; border-radius:999px; background:rgba(0,210,190,0.16); color:#bff8f2; border:1px solid rgba(0,210,190,0.35); font-size:11px; font-weight:700;">COUNTDOWN</span>'
            )
            st.markdown(
                f"""
                <div style="margin:8px 0 10px 0; border:1px solid rgba(255,255,255,0.10); border-radius:12px; padding:10px 12px;
                            background:linear-gradient(135deg, rgba(20,24,34,0.92), rgba(28,34,48,0.92));">
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:6px;">
                        <div style="font-size:12px; color:#b8c2d4; text-transform:uppercase; letter-spacing:1px;">Next Race Countdown</div>
                        <div>{race_week_badge}</div>
                    </div>
                    <div style="font-weight:700; font-size:15px; color:#f2f5fb;">{next_race_sidebar.get('event','Next Race')}</div>
                    <div style="font-size:12px; color:#b8c2d4; margin:2px 0 8px 0;">{next_race_sidebar.get('location','TBA')} | {format_user_time(next_race_sidebar.get('race_time'), '%d %b %Y %H:%M')}</div>
                    <div style="display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px;">
                        <div style="border:1px solid rgba(255,255,255,0.07); border-radius:8px; padding:6px 8px; text-align:center;">
                            <div style="font-size:11px; color:#aeb8c8;">Days</div>
                            <div style="font-size:16px; font-weight:800; color:#fff;">{int(next_race_sidebar.get('days', 0))}</div>
                        </div>
                        <div style="border:1px solid rgba(255,255,255,0.07); border-radius:8px; padding:6px 8px; text-align:center;">
                            <div style="font-size:11px; color:#aeb8c8;">Hours</div>
                            <div style="font-size:16px; font-weight:800; color:#fff;">{int(next_race_sidebar.get('hours', 0))}</div>
                        </div>
                        <div style="border:1px solid rgba(255,255,255,0.07); border-radius:8px; padding:6px 8px; text-align:center;">
                            <div style="font-size:11px; color:#aeb8c8;">Minutes</div>
                            <div style="font-size:16px; font-weight:800; color:#fff;">{int(next_race_sidebar.get('minutes', 0))}</div>
                        </div>
                        <div style="border:1px solid rgba(255,255,255,0.07); border-radius:8px; padding:6px 8px; text-align:center;">
                            <div style="font-size:11px; color:#aeb8c8;">Seconds</div>
                            <div style="font-size:16px; font-weight:800; color:#fff;">{int(next_race_sidebar.get('seconds', 0))}</div>
                        </div>
                    </div>
                    <div style="margin-top:8px; font-size:11px; color:#aeb8c8;">Source: {next_race_sidebar.get('source', 'Calendar')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown(f"**Season:** {active_year_sidebar}")
        st.markdown(f"**Races:** {len(get_season_race_choices(active_year_sidebar))}/24")
        if df is not None and not df.empty:
            st.markdown(f"**Total Points:** {int(total_all_points):,}")
        else:
            st.markdown("**Total Points:** 0")
        st.markdown("---")

    # Main Navigation (Grouped Tabs) - Removed emojis
    main_tabs = st.tabs([
        "Home",
        "Season Stats", 
        "Race Center",
        "Analysis",
        "Live",
        "2026 Hub"
    ])
    
    # 1. Home
    with main_tabs[0]:
        render_home_tab()
    
    # 2. Season Stats
    with main_tabs[1]:
        st.subheader("Season Statistics")
        season_subtabs = st.tabs(["Overview", "Drivers", "Constructors", "Grid & Tech", "Regulations"])
        
        with season_subtabs[0]:
            render_overview_tab(df, total_points_combined)
        with season_subtabs[1]:
            render_drivers_tab(df, total_points_combined)
        with season_subtabs[2]:
            render_teams_tab(df)
        with season_subtabs[3]:
            render_2026_grid_tech_tab()
        with season_subtabs[4]:
            render_2026_regulations_tab()
            
    # 3. Race Center
    with main_tabs[2]:
        st.subheader("Race Weekend Center")
        race_subtabs = st.tabs([
            "Weekend Dashboard",
            "Weekend Details", 
            "Race Analysis", 
            "Qualifying", 
            "Telemetry",
            "Race Replay",
            "Official Plots",
            "Export Data"
        ])
        
        with race_subtabs[0]:
            render_weekend_dashboard_tab()
        with race_subtabs[1]:
            render_race_detail_tab(df)
        with race_subtabs[2]:
            render_race_analysis_tab(df)
        with race_subtabs[3]:
            render_qualifying_tab()
        with race_subtabs[4]:
            render_telemetry_tab(df)
        with race_subtabs[5]:
            render_race_replay_tab()
        with race_subtabs[6]:
            render_official_plots_tab()
        with race_subtabs[7]:
            # Data Export Logic Inline
            st.subheader("Data Export")
            st.markdown("Export session data to CSV for external analysis.")
            c1, c2 = st.columns(2)
            with c1:
                ex_race = st.selectbox("Select Race", get_season_race_choices(), key="export_race")
            with c2:
                ex_session = st.selectbox("Session Type", ["Race", "Qualifying", "Sprint"], key="export_session")
                
            export_year = get_active_season_year()
            export_sig = f"{export_year}|{ex_race}|{ex_session}"
            prepare_export_clicked = st.button("Prepare Export", type="primary", key="prepare_export_btn")
            if prepare_export_clicked:
                st.session_state["export_ready_sig"] = export_sig

            if st.session_state.get("export_ready_sig") != export_sig:
                st.info("Select race/session and click `Prepare Export` to generate downloadable files.")
            else:
                export_cache = st.session_state.setdefault("export_payload_cache", {})
                exports = export_cache.get(export_sig)

                if exports is None or prepare_export_clicked:
                    with st.spinner("Processing data..."):
                        session = get_fastf1_session_state_cached(
                            year=export_year,
                            race=ex_race,
                            session_type=ex_session,
                            load_telemetry=False,
                            cache_namespace="export_fastf1",
                            force_reload=bool(prepare_export_clicked),
                        )
                        if session:
                            exports = export_session_to_csv(session)
                            export_cache[export_sig] = exports
                        else:
                            export_cache.pop(export_sig, None)
                            exports = None
                
                if exports:
                    cols = st.columns(3)
                    if 'laps' in exports:
                        cols[0].download_button("Download Laps", exports['laps'], f"{ex_race}_laps.csv", "text/csv")
                    if 'results' in exports:
                        cols[1].download_button("Download Results", exports['results'], f"{ex_race}_results.csv", "text/csv")
                    if 'weather' in exports:
                        cols[2].download_button("Download Weather", exports['weather'], f"{ex_race}_weather.csv", "text/csv")
                    st.success("Data ready for download!")
                else:
                    st.error("Could not load session.")

    # 4. Analysis
    with main_tabs[3]:
        st.subheader("Advanced Analysis")
        # removed God Mode separate tab link, moved content to Strategy Tools
        analysis_subtabs = st.tabs(["Teammate Battle", "Race Predictions", "Performance Lab"])
        
        with analysis_subtabs[0]:
            render_teammate_battle_tab(df)
        with analysis_subtabs[1]:
            render_prediction_tab(df, total_points_combined)
        with analysis_subtabs[2]:
            render_analysis_performance_lab_tab(df)
            
    # 5. Live
    with main_tabs[4]:
        live_subtabs = st.tabs(["Live Timing", "Live Control"])
        with live_subtabs[0]:
            render_live_timing_tab()
        with live_subtabs[1]:
            render_live_control_tab()

    # 6. 2026 Hub
    with main_tabs[5]:
        render_f1_2026_updates_tab()

    maybe_run_countdown_autorefresh()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        st.error(f"Critical Error: {e}")
        st.code(traceback.format_exc())

