import streamlit as st
import pandas as pd
import fastf1
import logging

# Local imports
from config import STREAMLIT_CONFIG, DATA_DIR, FASTF1_CONFIG
from loader import load_data as load_csv_data, load_season_data

logger = logging.getLogger(__name__)



@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl)
def load_race_data(year):
    """Cached season loader; the pure logic lives in loader.load_season_data."""
    try:
        return load_season_data(str(DATA_DIR), year)
    except Exception as e:
        logger.error(f"Error loading season data for {year}: {e}")
        return None



@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl)
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
    except Exception as e:
        logger.warning(f"Could not compute total points for {year}: {e}")
    return total_points_combined, total_laps_all, total_all_points


@st.cache_resource(ttl=STREAMLIT_CONFIG.cache_ttl)
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
    cache_dir = FASTF1_CONFIG.cache_dir
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    return cache_dir


def show_plotly_chart(fig, width='stretch', apply_theme=True, **kwargs):
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
    st.plotly_chart(fig, width=width, config=config, **kwargs)


def empty_state(title: str, hint: str = '') -> None:
    """Consistent, calm empty state panel (season not started / no data yet)."""
    hint_html = (f'<div style="color:#8a8a9a;font-size:.88rem;margin-top:.35rem;">{hint}</div>'
                 if hint else '')
    st.markdown(f"""
    <div style="text-align:center;padding:2.2rem 1rem;border:1px dashed rgba(255,255,255,.14);
                border-radius:14px;background:rgba(255,255,255,.02);margin:.4rem 0;">
      <div style="font-weight:700;color:#e0e0e0;letter-spacing:.3px;">{title}</div>
      {hint_html}
    </div>""", unsafe_allow_html=True)

