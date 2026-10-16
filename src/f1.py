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
from shared import get_total_points_combined, setup_fastf1_cache
from season_config import get_completed_races, get_season_calendar

logger = logging.getLogger(__name__)

PAGES = {
    "Dashboard": ("pages/1_Home.py", ":material/home:"),
    "Head-to-Head & Season": ("pages/2_HeadToHead.py", ":material/compare_arrows:"),
    "Predictor & Fantasy": ("pages/3_Predictor.py", ":material/insights:"),
    "Weekend Report": ("pages/4_Report.py", ":material/description:"),
    "Live & Replay": ("pages/5_Replay.py", ":material/sensors:"),
    "Analysis Center": ("pages/6_Analysis.py", ":material/monitoring:"),
    "Competitor Analysis": ("pages/7_Competitor.py", ":material/groups:"),
}


def _nav_pages():
    """St.Page objects built once.

    ``st.navigation`` registers them and the sidebar's links reuse the same
    objects, so ``st.page_link`` takes its object branch instead of resolving
    a path against the pages registry, which is incomplete on a deep-link
    entry and raises ``KeyError: url_pathname`` (Streamlit 1.61).
    """
    return [
        st.Page(path, title=title, icon=icon, default=index == 0)
        for index, (title, (path, icon)) in enumerate(PAGES.items())
    ]
def _season_snapshot(yr, total_all_points):
    """Races done / calendar size / points for the sidebar stat block."""
    races_done = len(get_completed_races(yr))
    try:
        races_total = len(get_season_calendar(yr))
    except Exception as e:
        logger.warning(f"Calendar unavailable for {yr}: {e}")
        races_total = races_done
    return {"races_done": races_done, "races_total": races_total,
            "total_points": int(total_all_points or 0)}


def _render_sidebar(snapshot, pages):
    """Brand -> menu -> season -> stats -> credit, in the reference order."""
    st.markdown("""
    <div class="brand">
        <div class="brand-mark">F1</div>
        <div class="brand-text">
            <div class="brand-name">F1 LAB</div>
            <div class="brand-sub">Data · Strategy · Insights</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-label">Menu</div>', unsafe_allow_html=True)
    for page in pages:
        st.page_link(page)

    st.markdown('<div class="side-label">Season</div>', unsafe_allow_html=True)
    supported_years = FASTF1_CONFIG.get_supported_years()
    current_year = st.session_state.selected_year
    default_index = supported_years.index(current_year) if current_year in supported_years else -1
    st.session_state.selected_year = st.selectbox(
        "Season", supported_years, index=default_index, label_visibility="collapsed",
    )

    st.markdown('<div class="side-label">Season Snapshot</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="side-stat">
        <span class="side-stat-label">Races</span>
        <span class="side-stat-value">{snapshot['races_done']} / {snapshot['races_total']}</span>
    </div>
    <div class="side-stat">
        <span class="side-stat-label">Total Points</span>
        <span class="side-stat-value">{snapshot['total_points']:,}</span>
    </div>
    <div class="side-footer">F1 Lab · Public data analytics</div>
    """, unsafe_allow_html=True)

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

# Custom CSS
css_file = Path(__file__).parent / 'assets' / 'style.css'
if css_file.exists():
    with open(css_file, 'r') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    yr = st.session_state.selected_year
    _, _, total_all_points = get_total_points_combined(yr)
    snapshot = _season_snapshot(yr, total_all_points)

    # Register the pages before the sidebar renders its links. Navigation
    # stays hidden: the sidebar draws the menu from `pages` itself.
    pages = _nav_pages()
    pg = st.navigation(pages, position="hidden")

    with st.sidebar:
        _render_sidebar(snapshot, pages)
    pg.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        st.error(f"Critical Error: {e}")
        st.code(traceback.format_exc())
