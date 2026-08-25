import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from datetime import timedelta

from shared import load_race_data, load_fastf1_session, show_plotly_chart, format_f1_time
from config import FASTF1_CONFIG, SESSION_GUARD_HOURS
from season_config import get_race_names, get_event_schedule_cached
from replay import positions_by_lap, build_position_replay


def _event_state(year, race):
    """Return ('future' | 'live' | 'past', EventDate or None) for a race weekend."""
    try:
        schedule = get_event_schedule_cached(year)
        if schedule is None or schedule.empty:
            return 'past', None
        if schedule['EventDate'].dt.tz is None:
            schedule['EventDate'] = schedule['EventDate'].dt.tz_localize('UTC')
        ev = schedule[schedule['EventName'] == race]
        if ev.empty:
            return 'past', None
        date = ev.iloc[0]['EventDate']
        now = pd.Timestamp.now(tz='UTC')
        if date > now + timedelta(hours=SESSION_GUARD_HOURS):
            return 'future', date
        if date >= now - timedelta(hours=SESSION_GUARD_HOURS):
            return 'live', date
        return 'past', date
    except Exception:
        return 'past', None


def page():
    year = st.session_state.get('selected_year', FASTF1_CONFIG.default_year)
    df = load_race_data(year)
    if df is None or df.empty:
        st.error("No data available")
        return

    race_df = df[df['SessionType'] == 'Race'].copy() if 'SessionType' in df.columns else df.copy()

    st.markdown(f"""
    <div style="text-align:center;padding:1.2rem 0 0.8rem 0;">
        <h1 style="font-size:2.2rem;font-weight:800;color:#E10600;margin:0;
                   text-transform:uppercase;letter-spacing:3px;">
            Live &amp; Replay
        </h1>
        <p style="font-size:0.9rem;color:#666;margin-top:0.3rem;letter-spacing:0.5px;">
            {year} Season &mdash; Animated Position Replay &middot; Live Best-Effort
        </p>
    </div>
    """, unsafe_allow_html=True)

    race_names = get_race_names(year)
    all_races = list(race_names.keys()) if race_names else []
    if not all_races:
        st.warning("No races available.")
        return

    c1, c2 = st.columns([3, 1])
    with c1:
        selected_race = st.selectbox("Grand Prix", all_races, key="replay_race",
                                     label_visibility="collapsed")
    with c2:
        session_type = st.selectbox("Session", ["Race", "Sprint", "Qualifying"],
                                    key="replay_session")

    state, _ = _event_state(year, selected_race)
    if state == 'future':
        st.info(f"{selected_race} has not started yet.")
        return

    with st.spinner("Loading session data..."):
        session = load_fastf1_session(year, selected_race, session_type)

    if session is None:
        st.warning(f"Data not available for {selected_race} ({session_type}).")
        return

    laps = session.laps
    if laps is None or laps.empty:
        st.warning("No lap data for this session.")
        return

    if state == 'live':
        st.markdown(
            '<span style="color:#E10600;font-weight:700;letter-spacing:1px;">LIVE '
            '&mdash; best-effort timing</span>', unsafe_allow_html=True)

    # Driver -> team map from the offline season data
    team_map = race_df.groupby('Driver')['Team'].first().to_dict() if 'Driver' in race_df.columns else {}

    drivers = sorted(laps['Driver'].dropna().unique())
    selected = st.multiselect("Drivers", drivers, default=drivers,
                              max_selections=20, key="replay_drv")
    if not selected:
        return

    pivot, order = positions_by_lap(laps, drivers=selected)
    if pivot.empty:
        st.info(f"Position evolution is not available for a {session_type} session.")
        return

    fig = build_position_replay(
        pivot, team_map,
        title=f"{selected_race} — {session_type}: Position Replay",
    )
    show_plotly_chart(fig)
    st.caption("Use the Play button to run the race. Driver markers move down the "
               "inverted position axis; a marker reaching P1 is the leader.")

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Total Laps", int(order[-1]))
    with k2:
        fastest = laps['LapTime'].dropna().min()
        st.metric("Fastest Lap", format_f1_time(fastest) if not pd.isna(fastest) else '-')
    with k3:
        st.metric("Drivers Shown", len(selected))


page()