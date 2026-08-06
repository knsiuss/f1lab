# -*- coding: utf-8 -*-
"""
f1.py
~~~~~
Streamlit dashboard entry point.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import streamlit as st
import logging
from pathlib import Path
import sys

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent))

from config import STREAMLIT_CONFIG, FASTF1_CONFIG, _ensure_dirs
from shared import (
    load_race_data, get_total_points_combined,
    setup_fastf1_cache
)
from season_config import get_completed_races, get_season_calendar

logger = logging.getLogger(__name__)

# Ensure directories exist, then setup cache
_ensure_dirs()
setup_fastf1_cache()

# Page config
st.set_page_config(
    page_title=STREAMLIT_CONFIG.page_title,
    page_icon=STREAMLIT_CONFIG.page_icon,
    layout=STREAMLIT_CONFIG.layout,
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': None,
        'Report a bug': None,
        'About': None,
    }
)

# Initialize session state
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = FASTF1_CONFIG.default_year
if 'driver_profile_selection' not in st.session_state:
    st.session_state.driver_profile_selection = None
if 'team_profile_selection' not in st.session_state:
    st.session_state.team_profile_selection = None
if 'race_detail_selection' not in st.session_state:
    st.session_state.race_detail_selection = None

# Custom CSS
css_file = Path(__file__).parent / 'assets' / 'style.css'
if css_file.exists():
    with open(css_file, 'r') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    # Load data
    yr = st.session_state.selected_year
    df = load_race_data(yr)
    _, _, total_all_points = get_total_points_combined(yr)

    # Sidebar
    with st.sidebar:
        st.markdown(f"<h1 style='color:#E10600; margin:0; font-size:1.8rem;'>F1 {st.session_state.selected_year}</h1>", unsafe_allow_html=True)
        st.markdown("---")

        # Season selector -- FastF1-supported historical seasons.
        # Pages degrade gracefully (show "No data available") for years
        # without local CSV results.
        supported_years = FASTF1_CONFIG.get_supported_years()
        current_year = st.session_state.selected_year
        default_index = supported_years.index(current_year) if current_year in supported_years else -1
        st.session_state.selected_year = st.selectbox(
            "Season", supported_years, index=default_index,
        )
        st.markdown("---")

        if df is not None:
            comp_races = get_completed_races(st.session_state.selected_year)
            total_races = len(get_season_calendar(st.session_state.selected_year))
            st.markdown(f"**Races:** {len(comp_races)}/{total_races}")
            st.markdown(f"**Total Points:** {int(total_all_points):,}")

        st.markdown("---")
        st.caption("Created by **Maxvy**")

    # Navigation
    pg = st.navigation({
        "F1 Lab": [
            st.Page("pages/1_Home.py", title="Dashboard", default=True),
            st.Page("pages/2_HeadToHead.py", title="Head-to-Head & Season"),
            st.Page("pages/3_Predictor.py", title="Predictor & Fantasy"),
            st.Page("pages/4_Report.py", title="Weekend Report"),
            st.Page("pages/5_Replay.py", title="Live & Replay"),
            st.Page("pages/6_Analysis.py", title="Analysis Center"),
            st.Page("pages/7_Competitor.py", title="Competitor Analysis"),
        ]
    })
    pg.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        st.error(f"Critical Error: {e}")
        st.code(traceback.format_exc())
