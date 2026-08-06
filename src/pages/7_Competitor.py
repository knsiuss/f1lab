import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from shared import load_race_data
from season_config import get_race_names
from rival import (
    build_watchlist,
    circuit_insights,
    head_to_head,
    rival_attribution_md,
)


def page():
    year = st.session_state.get('selected_year', 2025)
    df = load_race_data(year)
    if df is None or df.empty:
        st.error("No data available")
        return

    race_df = df[df['SessionType'] == 'Race'].copy() if 'SessionType' in df.columns else df.copy()

    st.markdown(f"""
    <div style="text-align:center;padding:1.2rem 0 0.8rem 0;">
        <h1 style="font-size:2.2rem;font-weight:800;color:#E10600;margin:0;
                   text-transform:uppercase;letter-spacing:3px;">
            Competitor Analysis
        </h1>
        <p style="font-size:0.9rem;color:#666;margin-top:0.3rem;letter-spacing:0.5px;">
            {year} Season &mdash; Rival Watchlist &middot; Head-to-Head &middot; Circuit Insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Competitor analysis is built from public season results. It is a "
        "decision-support view for post-session study &mdash; not a race-critical "
        "tool and not a substitute for a team's internal data. Figures derived "
        "from the model are labelled &ldquo;estimated&rdquo; with a confidence level."
    )

    drivers = sorted(race_df['Driver'].dropna().unique().tolist())
    if not drivers:
        st.warning("No drivers found in season data.")
        return

    # -- Rival Watchlist --------------------------------------------------
    st.markdown("### Rival Watchlist")
    c1, c2 = st.columns([2, 3])
    with c1:
        focus = st.selectbox("Focus driver", drivers, key="rival_focus")
    with c2:
        top_n = st.slider("Watchlist size", 3, 10, 5, key="rival_top")

    watch = build_watchlist(race_df, focus=focus, top=top_n)
    if watch.empty:
        st.info("Insufficient data to build a watchlist.")
    else:
        shown = watch[['Rank', 'Driver', 'Team', 'Points', 'GapToFocus',
                       'Level', 'Confidence']].copy()
        shown = shown.rename(columns={
            'Points': 'Points', 'GapToFocus': 'Gap to focus',
            'Level': 'Quality', 'Confidence': 'Confidence',
        })
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.caption(
            f"Gap to focus is points ahead (+) or behind (-) of **{focus}**. "
            "Points are real results; rival status and confidence reflect the "
            "number of completed races behind them (estimated)."
        )

    # -- Head-to-Head -----------------------------------------------------
    st.markdown("### Head-to-Head")
    h1, h2 = st.columns(2)
    with h1:
        driver_a = st.selectbox("Driver A", drivers, index=0, key="rival_a")
    with h2:
        driver_b = st.selectbox("Driver B", [d for d in drivers if d != driver_a],
                                index=0, key="rival_b")

    h2h = head_to_head(race_df, driver_a, driver_b)
    st.markdown(rival_attribution_md(h2h))
    if h2h.get('level') != 'insufficient':
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Races compared", h2h.get('races'))
        with m2: st.metric(f"{driver_a} ahead", h2h.get('a_wins'))
        with m3: st.metric(f"{driver_b} ahead", h2h.get('b_wins'))
        with m4:
            pts = h2h.get('points_diff')
            st.metric("Points diff (A - B)", f"{pts:+d}" if pts is not None else "n/a")
        st.caption(
            f"Avg finishing position &mdash; {driver_a}: P{h2h.get('a_avg_pos')} vs "
            f"{driver_b}: P{h2h.get('b_avg_pos')} (estimated, "
            f"confidence {h2h.get('confidence')})."
        )

    # -- Circuit Insights -------------------------------------------------
    st.markdown("### Circuit Insights")
    race_names = get_race_names(year)
    all_races = list(race_names.keys()) if race_names else []
    selected_track = st.selectbox("Circuit", all_races, key="rival_track") if all_races else None
    if selected_track:
        ins = circuit_insights(race_df, selected_track)
        if ins.get('level') == 'insufficient':
            st.info(f"Insufficient data for {selected_track}.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Winner", ins.get('winner', 'n/a'))
            with m2: st.metric("Pole", ins.get('pole', 'n/a'))
            with m3: st.metric("Pole to win", "Yes" if ins.get('pole_to_win') else "No")
            with m4: st.metric("Avg places gained", f"{ins.get('avg_places_gained', 0):+.2f}")
            st.caption(
                f"Fastest lap: {ins.get('fastest_lap', 'n/a')} &mdash; "
                f"confidence {ins.get('confidence')} (estimated). "
                + " &middot; ".join(ins.get('assumptions', []))
            )


page()
