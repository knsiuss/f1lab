import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fastf1
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Local imports
from config import TEAM_COLORS, DRIVER_PROFILES, STREAMLIT_CONFIG, SOCIAL_MEDIA_CONFIG, DATA_FILES, DATA_DIR
from season_config import get_season_calendar, get_completed_races, get_race_names
from loader import load_data as load_csv_data, clean_data, load_combined_data

logger = logging.getLogger(__name__)


def render_header():
    """Render dashboard header."""
    year = st.session_state.get('selected_year', 2025)
    st.markdown(f'<h1 class="main-header">F1 {year} Season Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analytics | Telemetry | Predictions</p>', unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_race_data(year):
    try:
        data_dir = str(Path(DATA_FILES.race_results).parent)
        race_path = str(Path(data_dir) / f'Formula1_{year}Season_RaceResults.csv')
        sprint_path = str(Path(data_dir) / f'Formula1_{year}Season_SprintResults.csv')

        df_race = load_csv_data(race_path)
        if df_race is None: return None
        df_race['SessionType'] = 'Race'

        try:
            df_sprint = load_csv_data(sprint_path)
            if df_sprint is not None and not df_sprint.empty:
                df_sprint['SessionType'] = 'Sprint'
                df = pd.concat([df_race, df_sprint], ignore_index=True)
            else:
                df = df_race
        except:
            df = df_race

        if df is not None:
            df = clean_data(df)
        return df
    except Exception as e:
        logger.error(f"Error loading combined data: {e}")
        try:
            fallback_path = str(DATA_DIR / f'Formula1_{year}Season_RaceResults.csv')
            df = load_csv_data(fallback_path)
            if df is not None:
                df = clean_data(df)
            return df
        except Exception as e2:
            logger.error(f"Error loading race data: {e2}")
            return None


@st.cache_data(ttl=3600)
def load_sprint_data(year):
    try:
        data_dir = str(Path(DATA_FILES.race_results).parent)
        sprint_file = str(Path(data_dir) / f'Formula1_{year}Season_SprintResults.csv')
        df = load_csv_data(sprint_file)
        if df is not None:
            df = clean_data(df)
        return df
    except Exception as e:
        return None


@st.cache_data(ttl=3600)
def get_total_points_combined(year):
    """Calculate total points (race + sprint) for all drivers."""
    total_points_combined = {}
    total_laps_all = 0
    total_all_points = 0
    try:
        race_path = DATA_DIR / f'Formula1_{year}Season_RaceResults.csv'
        sprint_path = DATA_DIR / f'Formula1_{year}Season_SprintResults.csv'

        df_race = load_csv_data(str(race_path))
        if df_race is None or df_race.empty:
            return total_points_combined, total_laps_all, total_all_points

        race_points = df_race.groupby('Driver')['Points'].sum()
        total_laps_all = int(df_race['Laps'].sum()) if 'Laps' in df_race.columns else 0

        if sprint_path.exists():
            df_sprint = load_csv_data(str(sprint_path))
            if df_sprint is not None and not df_sprint.empty:
                sprint_points = df_sprint.groupby('Driver')['Points'].sum()
                total_points_combined = race_points.add(sprint_points, fill_value=0).to_dict()
            else:
                total_points_combined = race_points.to_dict()
        else:
            total_points_combined = race_points.to_dict()

        total_all_points = sum(total_points_combined.values())
    except Exception:
        pass
    return total_points_combined, total_laps_all, total_all_points


@st.cache_resource(ttl=3600)
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


@st.cache_data(ttl=3600)
def get_real_grid_positions(year: int, race: str) -> dict:
    """Fetch actual Qualifying grid positions from FastF1."""
    try:
        session = fastf1.get_session(year, race, 'Q')
        session.load(telemetry=False, laps=False, weather=False)

        if session.results is None or session.results.empty:
            return {}

        grid_map = {}
        for drv in session.results['Abbreviation']:
            res = session.results.loc[session.results['Abbreviation'] == drv].iloc[0]
            grid = res['GridPosition']
            if grid <= 0:
                grid = res['Position']
            grid_map[drv] = grid

        return grid_map
    except Exception as e:
        logger.warning(f"Could not fetch real grid for {race}: {e}")
        return {}


def create_gauge(value, max_value, title, color="#E10600"):
    """Create a gauge chart for telemetry display."""
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

    try:
        max_value = float(max_value) if max_value and max_value > 0 else 100.0
    except:
        max_value = 100.0

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
    """Formats a pandas Timedelta object into an F1-style lap time string."""
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


def setup_fastf1_cache():
    cache_dir = Path("./f1_cache")
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    return cache_dir


def show_plotly_chart(fig, use_container_width=True, apply_theme=True, **kwargs):
    """Display Plotly chart with custom theme and hidden toolbar."""
    if apply_theme:
        fig.update_layout(
            font=dict(family="Outfit, Inter, sans-serif", color="white"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="rgba(20,20,30,0.85)",
                font_size=13,
                font_family="Inter, sans-serif",
                bordercolor="rgba(255,255,255,0.2)"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=12, color="white")
            ),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        # Apply gridlines to all axes unless they were explicitly hidden
        fig.update_xaxes(
            showgrid=True, 
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.2)'
        )
        fig.update_yaxes(
            showgrid=True, 
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.2)'
        )

    config = {
        'displayModeBar': False,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'zoomIn2d', 'zoomOut2d',
                                   'autoScale2d', 'resetScale2d', 'toImage'],
        'staticPlot': False
    }
    st.plotly_chart(fig, use_container_width=use_container_width, config=config, **kwargs)
