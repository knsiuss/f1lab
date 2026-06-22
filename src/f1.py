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

from config import STREAMLIT_CONFIG, TEAM_COLORS, _ensure_dirs
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
    st.session_state.selected_year = 2025
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

        st.session_state.selected_year = 2025
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
            st.Page("pages/6_Analysis.py", title="Analysis Center"),
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
