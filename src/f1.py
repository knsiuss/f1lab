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
    load_race_data, get_total_points_combined, season_progress,
    setup_fastf1_cache
)
from season_config import calendar_is_live
import ui
import glossary

logger = logging.getLogger(__name__)

# The pages of the app, in navigation order. One list feeds both the sidebar
# menu (opened by the hamburger button) and st.navigation, so the two can
# never drift apart.
PAGES = [
    ("Dashboard", "pages/1_Home.py", ":material/speed:"),
    ("Head-to-Head & Season", "pages/2_HeadToHead.py", ":material/compare_arrows:"),
    ("Predictor & Fantasy", "pages/3_Predictor.py", ":material/query_stats:"),
    ("Weekend Report", "pages/4_Report.py", ":material/article:"),
    ("Analysis Center", "pages/6_Analysis.py", ":material/monitoring:"),
    ("Competitor Analysis", "pages/7_Competitor.py", ":material/person_search:"),
    ("Race Replay", "pages/5_Replay.py", ":material/play_circle:"),
]

# Ensure directories exist, then setup cache
_ensure_dirs()
setup_fastf1_cache()

# Page config
st.set_page_config(
    page_title=STREAMLIT_CONFIG.page_title,
    page_icon=STREAMLIT_CONFIG.page_icon,
    layout=STREAMLIT_CONFIG.layout,
    # "auto" keeps the menu open on a desktop window and collapsed on a phone,
    # where an expanded sidebar would cover the whole 390 px screen before the
    # reader sees any data. The hamburger in the brand row shows it again.
    initial_sidebar_state="auto",
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
    # The pages are registered FIRST. st.page_link can only resolve a page that
    # the navigation already knows about, so the sidebar menu below links to the
    # same st.Page objects the navigation is built from -- one list, no drift,
    # and no "page not found" when the menu renders before the navigation.
    page_objects = [
        st.Page(path, title=title, icon=icon, default=(index == 0))
        for index, (title, path, icon) in enumerate(PAGES)
    ]
    pg = st.navigation({"F1 Lab": page_objects})

    # Sidebar
    with st.sidebar:
        # --- brand + hamburger -------------------------------------------------
        btn_col, brand_col = st.columns([1, 8], gap="small")
        with btn_col:
            if st.button('Menu', key='nav_toggle',
                         help='Show or hide the page menu'):
                st.session_state.nav_open = not st.session_state.get('nav_open', True)
        with brand_col:
            st.markdown("""
            <div style="padding:.1rem 0 .7rem;">
              <div style="font-family:'Outfit',sans-serif;font-size:1.25rem;font-weight:700;
                          color:#E8EAF0;letter-spacing:.06em;">F1 LAB</div>
              <div style="font-size:.68rem;color:#6B7280;letter-spacing:.14em;
                          text-transform:uppercase;margin-top:.1rem;">
                Public data analytics</div>
            </div>
            """, unsafe_allow_html=True)

        # --- navigation (the menu the hamburger opens) -------------------------
        if st.session_state.get('nav_open', True):
            for (title, _path, _icon), page_object in zip(PAGES, page_objects):
                st.page_link(page_object, label=title, icon=_icon)

        st.markdown('<div class="f1-sidebar-rule"></div>', unsafe_allow_html=True)

        # Season selector -- FastF1-supported historical seasons.
        # Pages degrade gracefully for years outside the OpenF1 archive.
        supported_years = FASTF1_CONFIG.get_supported_years()
        current_year = st.session_state.selected_year
        default_index = supported_years.index(current_year) if current_year in supported_years else -1
        st.session_state.selected_year = st.selectbox(
            "Season", supported_years, index=default_index,
        )

        # The data is read after the season picker, so the numbers below always
        # belong to the season that is actually selected.
        df = load_race_data(st.session_state.selected_year)
        _, _, total_all_points = get_total_points_combined(st.session_state.selected_year)

        if df is not None:
            # The same numbers the dashboard shows: races with results over the
            # races in the calendar.
            races_done, total_races = season_progress(st.session_state.selected_year)
            if calendar_is_live(st.session_state.selected_year):
                races_text = f"{races_done}/{total_races}"
            else:
                # The calendar is unreachable: the number of completed races is
                # unknown, not zero.
                races_text = f"unknown/{total_races}"
            st.markdown(f"""
            <div style="margin:.4rem 0 1rem;padding:.75rem .85rem;border:1px solid rgba(255,255,255,.08);
                        border-radius:10px;background:rgba(255,255,255,.02);">
              <div style="display:flex;justify-content:space-between;font-size:.8rem;
                          color:#9AA1AE;margin-bottom:.35rem;">
                <span>Races</span><span style="color:#E8EAF0;font-weight:600;">{races_text}</span></div>
              <div style="display:flex;justify-content:space-between;font-size:.8rem;color:#9AA1AE;">
                <span>Points</span>
                <span style="color:#E8EAF0;font-weight:600;">{int(total_all_points):,}</span></div>
            </div>
            """, unsafe_allow_html=True)

        # --- the language of the explanation layer ---------------------------
        # Only the glossary and the newcomer guide are translated. The data
        # labels, charts and sections stay in English, so a page can never end up
        # half translated.
        _lang_codes = list(glossary.LANGUAGES)
        if st.session_state.get('explainer_lang') not in _lang_codes:
            st.session_state['explainer_lang'] = glossary.DEFAULT_LANGUAGE
        st.selectbox(
            'Explainer language',
            options=_lang_codes,
            format_func=lambda code: glossary.LANGUAGES[code],
            key='explainer_lang',
            help="Language of the glossary and the 'New to F1' guide below.",
        )
        _reader_lang = ui.explainer_language()

        # --- a short guide for people who do not follow F1 --------------------
        # The app is written in plain words, but the sport itself is not common
        # knowledge. This button is the first thing a newcomer can open.
        with st.popover("New to F1? Start here", use_container_width=True):
            if _reader_lang == 'id':
                _guide_steps = [
                    "Setiap angka berasal dari hasil resmi publik. Tidak ada yang "
                    "berasal dari alat ukur milik tim.",
                    "Dashboard menunjukkan posisi musim ini. Predictor menebak "
                    "balapan berikutnya dan diuji dengan balapan yang sudah lewat.",
                    "Angka yang dihitung, bukan dilaporkan, ditandai 'perkiraan', "
                    "dengan catatan seberapa banyak datanya.",
                    "Setiap halaman ditutup glosarium, jadi tidak ada istilah yang "
                    "dibiarkan membingungkan.",
                ]
                _guide_title = "Cara memakai aplikasi ini"
            else:
                _guide_steps = [
                    "Every number comes from the public results feed. Nothing here "
                    "comes from a team's own instruments.",
                    "The Dashboard says where the season stands. The Predictor "
                    "guesses the next race and is checked against past races.",
                    "Anything calculated rather than reported is marked "
                    "'estimated', with a note on how much data it is based on.",
                    "Every page ends with a glossary, so no word has to stay a "
                    "puzzle.",
                ]
                _guide_title = "How to use this app"
            ui.steps(_guide_steps, title=_guide_title)
            for term in ("race", "points", "qualifying"):
                _entry = glossary.terms_for([term], lang=_reader_lang)
                _label = _entry[0]['term'] if _entry else term.title()
                ui.explain(_label, glossary.long(term, lang=_reader_lang))

        st.markdown("""
        <div style="font-size:.72rem;color:#6B7280;line-height:1.5;border-top:1px solid rgba(255,255,255,.07);
                    padding-top:.8rem;margin-top:.3rem;">
          Results, lap times, sectors, tyres and pit stops from
          <a href="https://openf1.org/" target="_blank" rel="noopener">OpenF1</a>
          (CC BY-NC-SA 4.0). Session data from
          <a href="https://docs.fastf1.dev/" target="_blank" rel="noopener">FastF1</a>
          where the official timing archive is reachable, otherwise from OpenF1.
        </div>
        """, unsafe_allow_html=True)
        st.caption("Created by **Maxvy**")

    # Run the page that the navigation resolved for this URL.
    pg.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        st.error(f"Critical Error: {e}")
        st.code(traceback.format_exc())
