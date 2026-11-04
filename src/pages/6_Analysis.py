# -*- coding: utf-8 -*-
"""Race Analysis Center: how fast the cars went, the tyres they ran, who
passed whom, where the time was lost and how the car was set up.

The page is composed from the shared UI components (:mod:`ui`) instead of
hand-written markup: one page header, a section per block, KPI tiles instead
of ad-hoc cards, and charts that carry no title of their own (the section
header names them).

Every derived figure keeps its honesty labels: "estimated", the confidence
level from ``pace.data_quality`` and the sample size behind it.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared import (
    load_fastf1_session, setup_fastf1_cache,
    show_plotly_chart, format_f1_time, team_colour,
)
from config import FASTF1_CONFIG, MODEL_CONFIG, SESSION_GUARD_HOURS, STREAMLIT_CONFIG
from season_config import (
    get_race_names, get_event_schedule_cached, get_completed_races,
    latest_race_index,
)
from fastf1_extended import (
    get_tyre_stints, get_pit_stops, get_race_control_messages,
    pit_laps_by_driver, ordered_drivers,
)
from model import RaceStrategySimulator
from pace import normalised_pace_by_stint, long_run_quality, clean_laps_for_analysis
from whatif import Scenario, compare_scenarios, undercut_delta
from sector import sector_summary, sector_deficits
from compare import compare_sessions
from setupfingerprint import fingerprint, normalise_fingerprint, fingerprint_shift
from pitwindow import pit_window
from analysis import (
    compare_driver_pace, driver_performance_scores,
    gap_to_leader,
)
import race_analysis as ra
from datetime import timedelta

from styles import (
    chart_height,
    COMPOUND_COLORS, FONT, GRID_EMPH, LABEL_FONT, PLOT_PALETTE,
    fig_layout as _fig_layout,
    scaled_height,
    TEXT_PRIMARY, TEXT_SECONDARY, ZERO_LINE,
    time_axis_layout as _time_axis_layout,
)
import glossary as glossary_terms
import intervals as live_feed
import ui
from ui import (
    caption, confidence_tag, data_table, empty, estimate_note, kpis, legend, note, page_header,
    takeaways,
)

# How many drivers the analysis views show by default.
DEFAULT_DRIVERS = 3


def _quickest_drivers(laps, count=DEFAULT_DRIVERS):
    """The quickest drivers of the session, used as the default selection.

    Taking the first names alphabetically would put three backmarkers on
    screen the first time the page is opened, which is not what the page is
    about: the quickest drivers are.
    """
    names = []
    try:
        best = laps.groupby('Driver')['LapTime'].min().sort_values()
        names = [str(name) for name in best.index[:count]]
    except Exception:
        names = []
    if not names:
        try:
            names = sorted(laps['Driver'].unique())[:count]
        except Exception:
            names = []
    return names


# The wording of the data-quality labels, shared by every table on the page.
QUALITY_WORDS = {
    'good': 'Good', 'moderate': 'Medium', 'low': 'Low',
    'insufficient': 'Not enough data',
}
QUALITY_TONES = {
    'good': 'good', 'moderate': 'accent', 'low': 'dim', 'insufficient': 'dim',
}
CONFIDENCE_TONES = {
    'good': 'good', 'high': 'good', 'moderate': 'accent', 'medium': 'accent',
    'low': 'dim', 'insufficient': 'dim',
}


def _num(value, digits=2, suffix=''):
    """A number for a table cell: formatted, never guessed."""
    try:
        if value is None or pd.isna(value):
            return '-'
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return '-'


def _quality_cell(level):
    """The 'how much data' label of an estimate, as a table cell."""
    key = str(level or '').strip().lower()
    return {
        'value': QUALITY_WORDS.get(key, str(level) if level else 'n/a'),
        'tone': QUALITY_TONES.get(key, 'dim'),
    }


def _confidence_cell(confidence):
    """The confidence word of an estimate, as a table cell."""
    key = str(confidence or '').strip().lower()
    return {
        'value': str(confidence) if confidence else 'n/a',
        'tone': CONFIDENCE_TONES.get(key, 'dim'),
    }


def _flag_cell(flag):
    """A race-control flag, coloured by what it means."""
    text = str(flag or '').strip()
    key = text.upper()
    tone = 'dim'
    if 'RED' in key:
        tone = 'bad'
    elif 'YELLOW' in key:
        tone = 'accent'
    elif 'GREEN' in key:
        tone = 'good'
    elif 'BLUE' in key:
        tone = 'accent'
    return {'value': text or '-', 'tone': tone}


def _weakest_level(levels):
    """The least trustworthy data-quality level of a list of them."""
    order = {'good': 0, 'moderate': 1, 'low': 2, 'insufficient': 3}
    values = [str(level).strip().lower() for level in levels if level]
    if not values:
        return 'insufficient'
    return max(values, key=lambda level: order.get(level, 3))


def _pace_verdict(pace, drv_a, drv_b):
    """What the head-to-head numbers add up to, in one plain sentence."""
    try:
        diff = float(pace['diff'])
    except (TypeError, ValueError):
        return "There is not enough data to compare these two drivers."
    faster = drv_a if diff < 0 else drv_b
    slower = drv_b if diff < 0 else drv_a
    return (f"On the clean laps of this session {faster} was "
            f"{abs(diff):.3f}s per lap quicker than {slower}, worked out from "
            f"{pace['n_a']} and {pace['n_b']} clean laps "
            f"({pace['confidence']} confidence).")


def _scenario_verdict(cmp_df):
    """What the plan comparison adds up to, in one plain sentence."""
    try:
        best = cmp_df.iloc[0]
        worst = cmp_df.iloc[-1]
        gap = float(worst['GapToBest'])
    except Exception:
        return "There is not enough data to compare the plans."
    return (f"Under these assumptions the {best['Name']} plan is the quickest "
            f"of the three, {gap:.1f}s ahead of the slowest. It is a "
            f"projection, not a recommendation.")


def _scores_verdict(scores_df):
    """What the performance scores add up to, in one plain sentence."""
    try:
        top = scores_df.iloc[0]
        last = scores_df.iloc[-1]
        spread = float(top['Overall']) - float(last['Overall'])
    except Exception:
        return "There is not enough data to score the drivers in this session."
    return (f"{top['Driver']} has the highest overall score "
            f"({float(top['Overall']):.1f}), {spread:.1f} points clear of "
            f"{last['Driver']}. The score blends how fast, how steady and how "
            f"the race went, so it summarises the session rather than "
            f"measuring a car.")


def _sector_rows(summary):
    """One table row per driver for the sector summary."""
    rows = []
    for _, r in summary.iterrows():
        row = [
            str(r.get('Driver', '')),
            str(int(r['N'])) if pd.notna(r.get('N')) else '-',
            _quality_cell(r.get('Level')),
            _confidence_cell(r.get('Confidence')),
        ]
        for part in ('1', '2', '3'):
            for stat in ('Best', 'Mean', 'Std'):
                row.append(_num(r.get(f"Sector{part}{stat}"), 3))
        rows.append(row)
    return rows


def _deficit_rows(deficits):
    """One table row per driver for the sector time differences."""
    rows = []
    for _, r in deficits.iterrows():
        rows.append([
            str(r.get('Driver', '')),
            str(int(r['N'])) if pd.notna(r.get('N')) else '-',
            _quality_cell(r.get('Level')),
            _confidence_cell(r.get('Confidence')),
            _num(r.get('Sector1Def'), 3),
            _num(r.get('Sector2Def'), 3),
            _num(r.get('Sector3Def'), 3),
        ])
    return rows


def _sector_verdict(summary, deficits):
    """What the sector numbers add up to, in one plain sentence."""
    try:
        best = summary.iloc[0]
        level = str(best.get('Level', '')).strip().lower()
    except Exception:
        return "There is not enough data to say where the time was lost."
    if level == 'insufficient':
        return "There is not enough clean-lap data to say where the time was " \
               "lost in this session."
    text = f"{best.get('Driver')} was the quickest in the first third of the " \
           f"circuit. "
    try:
        quick = str(deficits.iloc[0].get('Driver', ''))
        text += (f"{quick} lost the least time against the best in that part "
                 f"of the lap.")
    except Exception:
        pass
    return text


def _session_verdict(out, session_type, ref):
    """What the session comparison adds up to, in one plain sentence."""
    try:
        driver = str(out.iloc[0].get('Driver', ''))
        delta = float(out.iloc[0].get('Delta'))
    except Exception:
        return "There is not enough data to compare the two sessions."
    faster = (f"{driver} was {abs(delta):.3f}s per lap quicker in the "
              f"{session_type} than in the {ref}")
    slower = (f"{driver} was {delta:.3f}s per lap slower in the {session_type} "
              f"than in the {ref}")
    return ((faster if delta < 0 else slower)
            + ", with the wear of the tyre taken out. The comparison is an "
              "estimate from the clean laps of both sessions.")


def _shift_verdict(shift):
    """What the shift between two races adds up to, in one plain sentence."""
    try:
        top = shift.iloc[0]
        name = str(top.get('Driver', ''))
        size = float(top.get('AbsShift'))
    except Exception:
        return "There is not enough data to compare the two races."
    return (f"{name} shows the largest change between the two races, a total "
            f"shift of {size:.2f} across the readings. Small shifts are noise, "
            f"so this is a pointer rather than a conclusion.")

# Compounds that get a legend entry on the strategy maps.
LEGEND_COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']


def _compound_legend(compounds):
    """A colour legend for the tyre compounds used in a session."""
    used = [c for c in LEGEND_COMPOUNDS if c in set(compounds)]
    if not used:
        return
    legend([(COMPOUND_COLORS.get(c, '#888'), c) for c in used])


@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl, show_spinner=False)
def _cached_fingerprint(year, race, session_type='Race'):
    """Session fingerprint, cached: telemetry extraction is too heavy per rerun.

    The session is loaded **with telemetry**: the cornering block of the
    fingerprint needs the car data, and without it every driver row came back
    with empty apex speeds.
    """
    setup_fastf1_cache()
    s = load_fastf1_session(year, race, session_type, load_telemetry=True)
    if s is None:
        return pd.DataFrame()
    try:
        if s.laps is None or s.laps.empty:
            return pd.DataFrame()
        if not hasattr(s, 'car_data'):
            # The OpenF1 fallback has no car telemetry: the fingerprint reads
            # the car's own data, so it cannot be built from the lap feed.
            note(f"The rough setup reading needs the car's own telemetry, "
                 f"which the OpenF1 feed does not publish, so it is not "
                 f"available for {race} ({session_type}).", kind='warn')
            return pd.DataFrame()
        return fingerprint(s.laps, s)
    except Exception as e:
        # A session without car data (or a partial load) must not take the
        # whole tab down: the fingerprint is simply unavailable.
        note(f"The rough setup reading is not available for {race} "
             f"({session_type}): {e}", kind='warn')
        return pd.DataFrame()


@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl, show_spinner=False)
def _cached_shift(year, race_a, race_b, session_type='Race'):
    """Fingerprint diff between two cached session fingerprints."""
    fa = _cached_fingerprint(year, race_a, session_type)
    fb = _cached_fingerprint(year, race_b, session_type)
    return fingerprint_shift(fa, fb)


class SessionContext:
    """The session data and the slow derivations every tab needs.

    The lap table is read by all six tabs, and the pit-stop table, the clean
    lap list and the pace fit are derived from it again and again. They only
    change when the session changes, so they are built once and shared.
    """

    def __init__(self, session):
        self.session = session
        self.laps = session.laps
        try:
            self.pit_stops = get_pit_stops(session)
        except Exception:
            self.pit_stops = pd.DataFrame()
        try:
            self.clean_laps = clean_laps_for_analysis(self.laps)
        except Exception:
            self.clean_laps = pd.DataFrame()
        try:
            self.pace = normalised_pace_by_stint(self.laps)
        except Exception:
            self.pace = pd.DataFrame()


@st.cache_resource(ttl=STREAMLIT_CONFIG.cache_ttl, show_spinner=False)
def _session_context(year, race, session_type):
    """The session context for one session, built once and reused."""
    session = load_fastf1_session(year, race, session_type)
    if session is None or getattr(session, 'laps', None) is None:
        return None
    return SessionContext(session)


def _header_actions(year, selected_race, session_type):
    """Export Report + session status chips (data-source indicator).

    The export is built from the loaded laps only, so the report can never
    claim a number the page does not show. Status is honest: FastF1 is the
    official timing archive, OpenF1 is the fallback feed.
    """
    cols = st.columns([4, 1])
    with cols[1]:
        st.download_button(
            "Export Report",
            data=_report_csv(year, selected_race, session_type),
            file_name=f"F1_{year}_{selected_race.replace(' ', '_')}_{session_type}.csv",
            mime="text/csv",
            key="analysis_export",
        )
    return cols


def _report_csv(year, race, session_type):
    """Lap-level CSV for the selected session (empty when not loaded)."""
    try:
        context = _session_context(year, race, session_type)
        if context is None or context.laps is None or context.laps.empty:
            return "no lap data available\n"
        laps = context.laps.copy()
        keep = [c for c in ('Driver', 'Team', 'LapNumber', 'LapTime', 'Position',
                            'Compound', 'TyreLife', 'TrackStatus', 'PitInTime',
                            'PitOutTime') if c in laps.columns]
        frame = laps[keep].sort_values(['Driver', 'LapNumber'])
        frame.insert(0, 'Session', f"{year} {race} {session_type}")
        return frame.to_csv(index=False)
    except Exception:
        return "no lap data available\n"


def _body():
    year = st.session_state.get('selected_year', FASTF1_CONFIG.default_year)

    race_names = get_race_names(year)
    all_races = list(race_names.keys()) if race_names else []
    if not all_races:
        empty("No races found for this season")
        return

    c1, c2 = st.columns([3, 1])
    with c1:
        # Default to the most recent completed race: the season opener has no
        # session data worth analysing, the latest one does.
        completed = get_completed_races(year)
        selected_race = st.selectbox(
            "Grand Prix", all_races,
            index=latest_race_index(year, all_races, completed),
            key="analysis_race")
    with c2:
        session_type = st.selectbox("Session", ["Race", "Qualifying", "Sprint"],
                                    key="analysis_session")

    page_header(
        "Race Analysis",
        "Dive deep into race data with advanced performance insights.",
        chips=[('season', str(year)), ('', selected_race), ('', session_type)],
    )
    _header_actions(year, selected_race, session_type)

    # -- Session guard --
    try:
        schedule = get_event_schedule_cached(year)
        if schedule is not None and not schedule.empty:
            if schedule['EventDate'].dt.tz is None:
                schedule['EventDate'] = schedule['EventDate'].dt.tz_localize('UTC')
            race_event = schedule[schedule['EventName'] == selected_race]
            if not race_event.empty:
                event = race_event.iloc[0]
                now = pd.Timestamp.now(tz='UTC')
                if event['EventDate'] > (now + timedelta(hours=SESSION_GUARD_HOURS)):
                    empty(f"{selected_race} has not started yet",
                          "Live session data appears once the session is running.")
                    return
    except Exception:
        pass

    with st.spinner(f"Loading the {session_type} session data for "
                     f"{selected_race} (the first load downloads and parses "
                     f"the session, later ones are instant)..."):
        setup_fastf1_cache()
        context = _session_context(year, selected_race, session_type)

    if context is None:
        empty(f"No data available for {selected_race} ({session_type})",
              "The session may not have run yet, or its data is not published.")
        return

    session = context.session
    laps = context.laps
    if laps is None or laps.empty:
        empty(f"No lap data for {selected_race} ({session_type})")
        return

    # -- Where the numbers come from --
    # The official timing archive is not always reachable, so the session can
    # be rebuilt from the OpenF1 feed.  That feed has no safety-car flags and
    # no official running order, and the reader is told so.
    source = str(getattr(session, 'source', 'FastF1') or 'FastF1')
    if source != 'FastF1':
        note(
            "Lap times, sectors, tyres and pit stops for this session come "
            "from the OpenF1 feed (the official timing archive was not "
            "answering). Two things are missing from that feed: the "
            "race-control flags, so a lap under a safety car is not marked "
            "and may be part of a pace average, and the official running "
            "order, so the order shown in 'Position Evolution' is estimated from "
            "the lap times and the pit stops.",
            kind='info',
        )

    # -- KPI row --
    _render_kpi_row(context)

    # Every panel is built on every rerun: lazy tabs (on_change="rerun" +
    # tab.open) left the panel blank for a few seconds while the second run
    # arrived, which reads as a broken page. The costly part is the data
    # preparation, and that is cached in SessionContext instead.
    tabs = st.tabs(["Position Evolution", "Driver Pace",
                    "Sector Analysis", "Telemetry",
                    "Race Context", "Tyres & Stops",
                    "Setup", "Gaps & Radio"])

    with tabs[0]:
        _tab_battles(context, year)

    with tabs[1]:
        _tab_pace(context, year)

    with tabs[2]:
        _tab_sector_compare(laps, selected_race, year, session_type)

    with tabs[3]:
        _tab_telemetry(context, year, selected_race, session_type)

    with tabs[4]:
        _tab_insights(context, year)

    with tabs[5]:
        _tab_strategy(context, year)

    with tabs[6]:
        _tab_setup_aero(year, selected_race, session_type)

    with tabs[7]:
        _tab_feed(context)


# KPI ROW
def _render_kpi_row(context):
    """The six session headline numbers. All measured, none estimated."""
    laps = context.laps
    total_laps = int(laps['LapNumber'].nunique())
    leader_laps_total = 0
    try:
        leader_laps_total = int(
            laps.groupby('Driver')['LapNumber'].max().max())
    except Exception:
        leader_laps_total = total_laps

    fastest_row = laps.loc[laps['LapTime'].idxmin()]
    fastest_drv = fastest_row.get('Driver', 'N/A')
    fastest_time = format_f1_time(fastest_row['LapTime'])
    fastest_lap_no = fastest_row.get('LapNumber', '')
    fastest_team = str(fastest_row.get('Team', '') or '')

    leader_drv = '-'
    leader_team = ''
    if 'Position' in laps.columns:
        pos = pd.to_numeric(laps['Position'], errors='coerce')
        leader_laps = laps[pos == 1]
        if not leader_laps.empty:
            leader_drv = leader_laps['Driver'].mode().iloc[0]
            leader_team = str(leader_laps['Team'].iloc[0]) \
                if 'Team' in leader_laps.columns else ''

    pit_stops = context.pit_stops
    pit_count = len(pit_stops) if pit_stops is not None and not pit_stops.empty else 0
    pit_drivers = 0
    if pit_stops is not None and not pit_stops.empty and 'Driver' in pit_stops.columns:
        pit_drivers = int(pit_stops['Driver'].nunique())

    position_changes = 0
    classified = 0
    total_steps = 0
    if 'Position' in laps.columns:
        for _driver, group in laps.groupby('Driver'):
            group = group.sort_values('LapNumber')
            positions = pd.to_numeric(group['Position'], errors='coerce')
            steps = positions.diff().fillna(0) != 0
            position_changes += int(steps.sum())
            total_steps += int(positions.notna().sum())
            if positions.notna().any():
                classified += 1
    starters = int(laps['Driver'].nunique()) if 'Driver' in laps.columns else classified
    pct_changes = (position_changes / total_steps * 100) if total_steps else 0.0

    kpis([
        {'label': 'Total laps', 'value': str(total_laps),
         'sub': f"leader completed {leader_laps_total}", 'tone': '',
         'icon': 'flag'},
        {'label': 'Fastest lap', 'value': fastest_time,
         'sub': f"{fastest_drv} · Lap {fastest_lap_no}"
                + (f" · {fastest_team}" if fastest_team else ''),
         'tone': 'accent', 'icon': 'activity'},
        {'label': 'Race leader', 'value': str(leader_drv),
         'sub': leader_team or 'most laps led', 'tone': 'good',
         'icon': 'target'},
        {'label': 'Position changes', 'value': str(position_changes),
         'sub': f"{pct_changes:.0f}% of recorded steps", 'tone': '',
         'icon': 'chart'},
        {'label': 'Pit stops', 'value': str(pit_count),
         'sub': f"{pit_drivers} drivers stopped", 'tone': '',
         'icon': 'trend-up'},
        {'label': 'Classified', 'value': f"{classified}/{starters}",
         'sub': 'classified / starters', 'tone': 'info',
         'icon': 'trend-down'},
    ])


# TAB 1: PACE & LAPS
@st.fragment
def _tab_pace(context, year):
    laps = context.laps
    note(
        "This tab answers 'how fast were the cars going?'. It shows the lap "
        "times as they happened, the quickest lap of every driver, and how "
        "two drivers compared once the pit laps and the worn tyres were "
        "taken out."
    )

    ui.section("Lap time evolution", "measured — every lap, pit laps included",
            hint="Every lap time of the drivers you pick, as it happened. "
                 "The lap with a pit stop in it is much slower, so it is shown "
                 "here and left out of the comparisons below.")

    drivers_list = sorted(laps['Driver'].unique())
    selected = st.multiselect("Drivers", drivers_list,
                              default=_quickest_drivers(laps, DEFAULT_DRIVERS),
                              max_selections=8, key="pace_drv")

    if not selected:
        empty("No drivers selected")
        return

    clean_count = int(len(context.clean_laps))
    kpis([
        {'label': 'Drivers shown', 'value': str(len(selected)),
         'sub': 'of the session field', 'tone': ''},
        {'label': 'Clean laps', 'value': str(clean_count),
         'sub': 'available for estimates', 'tone': 'info'},
        {'label': 'Session fastest',
         'value': str(laps.loc[laps['LapTime'].idxmin()].get('Driver', 'N/A')),
         'sub': format_f1_time(laps.loc[laps['LapTime'].idxmin()]['LapTime']),
         'tone': 'accent'},
    ])

    # Main lap-time chart with compound colors and pit markers
    fig = go.Figure()
    all_times = []
    pit_laps_map = {}

    try:
        pit_stops = context.pit_stops
        # The pit-stop frame names its lap column 'Lap'. Reading 'LapNumber'
        # instead put every stop on lap 0, so the pit markers never appeared.
        pit_laps_map = pit_laps_by_driver(pit_stops)
    except Exception:
        pit_laps_map = {}

    overall_fastest = laps.loc[laps['LapTime'].idxmin()] if not laps.empty else None

    for index, driver in enumerate(selected):
        drv = laps[(laps['Driver'] == driver) & laps['LapTime'].notna()].sort_values('LapNumber').copy()
        if drv.empty:
            continue
        color = PLOT_PALETTE[index % len(PLOT_PALETTE)]
        times = drv['LapTime'].dt.total_seconds()
        all_times.extend(times.tolist())

        fig.add_trace(go.Scatter(
            x=drv['LapNumber'], y=times,
            mode='lines+markers', name=driver,
            line=dict(color=color, width=2.5, shape='spline', smoothing=0.3),
            marker=dict(size=6, color=color, opacity=0.8,
                        line=dict(width=1, color='rgba(255,255,255,0.4)')),
            hovertemplate=(
                "<b>%{customdata[2]}</b><br>"
                "Lap: %{x}<br>"
                "Time: %{customdata[0]}<br>"
                "Compound: %{customdata[1]}<extra></extra>"
            ),
            customdata=list(zip(
                [format_f1_time(pd.Timedelta(seconds=t)) for t in times],
                drv['Compound'].fillna('').tolist() if 'Compound' in drv.columns else [''] * len(times),
                [driver] * len(times),
            )),
        ))

        # Fastest lap star marker
        best_idx = times.idxmin()
        best_row = drv.loc[best_idx]
        fig.add_trace(go.Scatter(
            x=[best_row['LapNumber']], y=[best_row['LapTime'].total_seconds()],
            mode='markers+text', name=driver + " FL",
            marker=dict(size=14, color=color, symbol='star',
                        line=dict(width=2, color='white')),
            text='FL', textposition='top center',
            textfont=dict(size=10, color=TEXT_PRIMARY, family=FONT),
            showlegend=False, hoverinfo='skip',
        ))

        # Pit stop X markers
        for pl in pit_laps_map.get(driver, []):
            pit_row = laps[(laps['Driver'] == driver) & (laps['LapNumber'] == pl)]
            if not pit_row.empty and pit_row['LapTime'].notna().iloc[0]:
                fig.add_trace(go.Scatter(
                    x=[pl], y=[pit_row['LapTime'].iloc[0].total_seconds()],
                    mode='markers', showlegend=False,
                    marker=dict(size=12, color='white', symbol='x',
                                line=dict(width=2.5, color=color)),
                ))

    if overall_fastest is not None:
        fl_lap = overall_fastest.get('LapNumber', 0)
        fig.add_vline(x=fl_lap, line_dash="dash", line_color="#FFD166",
                      line_width=1.5, opacity=0.5)

    if all_times:
        y_min, y_max = min(all_times) - 1, max(all_times) + 1
        yaxis = _time_axis_layout("Lap Time", (y_min, y_max))
    else:
        yaxis = dict(title=dict(text="Lap Time", font=LABEL_FONT))

    fig.update_layout(**_fig_layout(height=chart_height('tall'),
        xaxis=dict(),
        yaxis=yaxis,
        hovermode='x unified',
    ))
    show_plotly_chart(fig)
    caption(
        "Measured lap times. The gold marker is each driver's fastest lap; "
        "the crossed markers are pit stops. These are raw lap times, so a "
        "pit lap is included — the comparisons below use clean laps only."
    )

    # -- Export + pace distribution --
    export_laps = laps[laps['Driver'].isin(selected)].copy()
    if not export_laps.empty:
        st.download_button(
            "Export lap times (CSV)", export_laps.to_csv(index=False),
            file_name="lap_times.csv", mime="text/csv", key="pace_csv")

    ui.section("Race pace distribution", "estimated — clean laps only",
            hint="How steady each driver's pace was: the box spans the "
                 "middle half of their clean laps, the line is the median.")
    dist = ra.pace_distribution(laps)
    if not dist.empty:
        clean = clean_laps_for_analysis(laps)
        fig_box = go.Figure()
        for index, driver in enumerate(selected):
            times = clean[clean['Driver'] == driver]['LapTime_s'].dropna()
            if times.empty:
                continue
            fig_box.add_trace(go.Box(
                y=times, name=driver, boxpoints='outliers',
                marker_color=PLOT_PALETTE[index % len(PLOT_PALETTE)],
                line_color=PLOT_PALETTE[index % len(PLOT_PALETTE)],
            ))
        fig_box.update_layout(**_fig_layout(height=chart_height('block'),
            yaxis_title=dict(text="Clean lap time (s)", font=LABEL_FONT),
        ))
        show_plotly_chart(fig_box)
        rows = [[row['Driver'], _num(row['Median_s'], 3),
                 _num(row['Mean_s'], 3), _num(row['Std_s'], 3),
                 str(int(row['Outliers'])), str(int(row['Laps']))]
                for _, row in dist.iterrows() if row['Driver'] in selected]
        data_table(['Driver', 'Median (s)', 'Mean (s)', 'Std dev (s)',
                    'Outliers', 'Clean laps'], rows)

    # -- Fastest laps table --
    ui.section("Fastest laps", "measured",
            hint="The quickest lap each driver set in this session, and the "
                 "tyre it was set on.")
    fastest = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()].reset_index(drop=True)
    if not fastest.empty:
        cols_show = ['Driver', 'LapNumber', 'LapTime', 'Compound']
        cols_show = [c for c in cols_show if c in fastest.columns]
        table = fastest[cols_show].sort_values('LapTime')
        rows = []
        for _, row in table.iterrows():
            lap_no = row.get('LapNumber')
            rows.append([
                str(row.get('Driver', '')),
                f"L{int(lap_no)}" if pd.notna(lap_no) else '-',
                format_f1_time(row['LapTime']),
                str(row.get('Compound', '') or '-'),
            ])
        data_table(
            ['Driver', 'Lap', 'Lap time', 'Tyre'],
            rows,
        )
    else:
        empty("No lap times recorded")

    # -- Head-to-head pace --
    ui.section("How fast the two cars were going", "estimated — clean laps only",
            hint="The average lap time of two drivers, worked out from their "
                 "clean laps. A clean lap is one that can be compared with "
                 "another lap: the car was on track, the track was green and "
                 "the timing was reliable.")

    estimate_note(
        "How fast the cars went is estimated from clean laps only (the lap "
        "into and out of the pit, the laps under a safety car and the laps "
        "with a doubtful time are removed). The confidence tag follows the "
        "smaller of the two samples."
    )

    c1, c2 = st.columns(2)
    with c1:
        drv_a = st.selectbox("Driver A", drivers_list, key="pace_h2h_a")
    with c2:
        drv_b = st.selectbox("Driver B", [d for d in drivers_list if d != drv_a],
                             key="pace_h2h_b")

    pace = compare_driver_pace(laps, drv_a, drv_b)
    if pace['avg_a'] is None or pace['avg_b'] is None:
        empty("Not enough clean lap data for both drivers to compare")
        ui.verdict("There is not enough clean lap data for both drivers in this "
                "session, so they cannot be compared here.")
        return

    la = pace['laps_a']
    lb = pace['laps_b']
    ta = la['LapTime_s']
    tb = lb['LapTime_s']
    team_a = la['Team'].iloc[0] if 'Team' in la.columns else ''
    team_b = lb['Team'].iloc[0] if 'Team' in lb.columns else ''
    diff = pace['diff']
    faster = drv_a if diff < 0 else drv_b

    kpis([
        {'label': f'{drv_a} average lap', 'value': format_f1_time(pd.Timedelta(seconds=pace['avg_a'])),
         'sub': f"{pace['n_a']} clean laps", 'tone': ''},
        {'label': f'{drv_b} average lap', 'value': format_f1_time(pd.Timedelta(seconds=pace['avg_b'])),
         'sub': f"{pace['n_b']} clean laps", 'tone': ''},
        {'label': 'Quicker car', 'value': str(faster), 'sub': f"{abs(diff):.3f}s per lap",
         'tone': 'accent'},
        {'label': 'How much data', 'value': str(pace['confidence']),
         'sub': f"{pace['level']} confidence", 'tone': 'info'},
    ])
    confidence_tag(pace['level'])

    color_a = team_colour(team_a, year)
    color_b = team_colour(team_b, year)

    fig_battle = make_subplots(rows=2, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.08,
                                subplot_titles=("Lap time", "Cumulative advantage"))

    # Row 1: Lap times
    fig_battle.add_trace(go.Scatter(
        x=la['LapNumber'], y=ta, mode='lines+markers',
        name=drv_a, line=dict(color=color_a, width=2.5),
        marker=dict(size=6, color=color_a),
        hovertemplate="{}: %{{customdata}}<extra></extra>".format(drv_a),
        customdata=[format_f1_time(pd.Timedelta(seconds=t)) for t in ta],
    ), row=1, col=1)
    fig_battle.add_trace(go.Scatter(
        x=lb['LapNumber'], y=tb, mode='lines+markers',
        name=drv_b, line=dict(color=color_b, width=2.5),
        marker=dict(size=6, color=color_b),
        hovertemplate="{}: %{{customdata}}<extra></extra>".format(drv_b),
        customdata=[format_f1_time(pd.Timedelta(seconds=t)) for t in tb],
    ), row=1, col=1)

    # Row 2: Cumulative advantage
    la_s = la[['LapNumber']].copy().sort_values('LapNumber')
    la_s['ta'] = la.sort_values('LapNumber')['LapTime_s'].to_numpy()
    lb_s = lb[['LapNumber']].copy().sort_values('LapNumber')
    lb_s['tb'] = lb.sort_values('LapNumber')['LapTime_s'].to_numpy()
    merged = pd.merge(la_s, lb_s, on='LapNumber', how='inner')
    if not merged.empty:
        merged['cum_adv'] = (merged['ta'] - merged['tb']).cumsum()
        adv_drv = drv_a if diff < 0 else drv_b
        fig_battle.add_trace(go.Scatter(
            x=merged['LapNumber'], y=merged['cum_adv'],
            mode='lines+markers', name="{} ahead".format(adv_drv),
            line=dict(color='#FFD166', width=2.5),
            fill='tozeroy', fillcolor='rgba(255,209,102,0.10)',
            hovertemplate="Lap %{x}<br>Adv: %{y:.3f}s<extra></extra>",
        ), row=2, col=1)
    fig_battle.add_hline(y=0, line_dash="dot", line_color='rgba(255,255,255,0.3)', row=2, col=1)

    fig_battle.update_layout(**_fig_layout(height=chart_height('tall'),
        hovermode='x unified',
        showlegend=True,
    ))
    fig_battle.update_xaxes(title_text="Lap", row=2, col=1)
    fig_battle.update_xaxes(row=1, col=1)
    y_min2 = min(ta.min(), tb.min()) * 0.995
    y_max2 = max(ta.max(), tb.max()) * 1.005
    fig_battle.update_yaxes(title_text="Lap Time", row=1, col=1,
                            **_time_axis_layout("", (y_min2, y_max2)))
    fig_battle.update_yaxes(title_text="Advantage (s)", row=2, col=1)
    show_plotly_chart(fig_battle)
    ui.verdict(_pace_verdict(pace, drv_a, drv_b))

    # Normalised pace & data quality (Data Honesty: estimated + sample size)
    ui.section("How fast the car was going on fresh tyres",
            "estimated — tyre age removed",
            hint=glossary_terms.short('pace') + " The wear of the tyre is "
                 "taken out first, so two different runs of laps can be "
                 "compared.")

    estimate_note(
        "These figures are estimated from a straight-line fit of lap time "
        "against the number of laps the tyre had already done. Every figure "
        "is flagged estimated with a confidence level derived from the "
        "number of clean laps behind it; short samples are marked low rather "
        "than presented as fact."
    )

    npace = context.pace
    if npace.empty:
        empty("Not enough clean lap data to work out how fast the cars went")
    else:
        rows = []
        for _, r in npace.iterrows():
            stint = r.get('Stint')
            rows.append([
                str(r.get('Driver', '')),
                f"L{int(stint)}" if pd.notna(stint) else '-',
                str(r.get('Compound', '') or '-'),
                str(int(r['N'])) if pd.notna(r.get('N')) else '-',
                _num(r.get('MeanPace'), 3),
                _num(r.get('BasePace'), 3),
                _num(r.get('DegPerLap'), 3),
                _quality_cell(r.get('Level')),
                _confidence_cell(r.get('Confidence')),
            ])
        data_table(
            ['Driver', 'Stint', 'Tyre', 'Clean laps', 'Average lap time (s)',
             'Lap time on fresh tyres (s, est)', 'Slower each lap (s, est)',
             'How much data', 'Confidence'],
            rows,
        )

    lr = long_run_quality(laps)
    if not lr.empty:
        ui.section("Long runs on one set of tyres", "estimated",
                hint=glossary_terms.short('stint') + " Only runs of at least "
                     "ten laps are shown, because a shorter run says nothing "
                     "about how a tyre lasts.")
        rows = []
        for _, r in lr.iterrows():
            stint = r.get('Stint')
            rows.append([
                str(r.get('Driver', '')),
                f"L{int(stint)}" if pd.notna(stint) else '-',
                str(r.get('Compound', '') or '-'),
                str(int(r['Length'])) if pd.notna(r.get('Length')) else '-',
                _num(r.get('MeanPace'), 3),
                _num(r.get('DegPerLap'), 3),
                _quality_cell(r.get('Level')),
                _confidence_cell(r.get('Confidence')),
            ])
        data_table(
            ['Driver', 'Stint', 'Tyre', 'Laps on this tyre',
             'Average lap time (s)', 'Slower each lap (s, est)',
             'How much data', 'Confidence'],
            rows,
        )

    # -- Pace consistency: std of clean laps (lower = more consistent) --
    ui.section("Pace consistency", "estimated — clean laps only",
            hint="Standard deviation of each driver's clean lap times. "
                 "Lower means steadier, not faster.")
    try:
        clean = context.clean_laps
        if clean is not None and not clean.empty:
            stats = clean.copy()
            stats['LapTime_s'] = pd.to_timedelta(
                stats['LapTime']).dt.total_seconds()
            stats = stats.dropna(subset=['LapTime_s'])
            rows_stats = []
            for drv, grp in stats.groupby('Driver'):
                vals = grp['LapTime_s']
                rows_stats.append({
                    'Driver': drv, 'N': int(len(vals)),
                    'Std': float(vals.std()) if len(vals) > 1 else 0.0,
                    'Mean': float(vals.mean()),
                    'Median': float(vals.median()),
                })
            if rows_stats:
                cdf = pd.DataFrame(rows_stats).sort_values('Std')
                fig_c = go.Figure(go.Bar(
                    x=cdf['Driver'], y=cdf['Std'],
                    marker_color=PLOT_PALETTE[0], marker_line_width=0,
                    hovertemplate='<b>%{x}</b><br>Std: %{y:.3f}s'
                                  '<extra></extra>'))
                fig_c.update_layout(**_fig_layout(
                    height=chart_height('row'),
                    yaxis_title=dict(text='Std dev of clean laps (s)',
                                     font=LABEL_FONT)))
                show_plotly_chart(fig_c)
                rows = [[r['Driver'], str(r['N']),
                         _num(r['Std'], 3), _num(r['Mean'], 3),
                         _num(r['Median'], 3)]
                        for _, r in cdf.iterrows()]
                data_table(['Driver (most consistent first)', 'Clean laps',
                            'Std dev (s)', 'Mean (s)', 'Median (s)'], rows)
                caption("Consistency is steadiness of lap times, not outright "
                        "speed: the fastest driver can be the least consistent.")
    except Exception:
        empty("Pace consistency unavailable")


# TAB 2: STRATEGY & STINTS
@st.fragment
def _tab_strategy(context, year):
    laps = context.laps
    session = context.session
    note(
        "This tab answers 'which tyres did they run, and what would a "
        "different plan have cost?'. It shows the tyre plan of every driver, "
        "the pit stops, and a projection of a few alternative strategies."
    )

    ui.section("Tyre strategy",
            "measured — the tyre runs of this session",
            hint=glossary_terms.short('stint') + " One bar per run of laps "
                 "on one tyre, coloured by the tyre.")

    tyre_data = get_tyre_stints(session)

    if tyre_data is None or tyre_data.empty:
        empty("Tyre strategy data not available")
    else:
        drivers = ordered_drivers(session, tyre_data)
        # "UNKNOWN" and "NAN" are what an absent compound reads as; they are
        # not a tyre, so they are left out of the list rather than shown to the
        # reader as if somebody had run them.
        compounds_used = sorted({
            str(c).upper() for c in tyre_data['Compound'].dropna()
            if str(c).upper() not in ('UNKNOWN', 'NAN', 'NONE', '')
        })
        kpis([
            {'label': 'Drivers with a tyre plan', 'value': str(len(drivers)),
             'sub': 'in this session', 'tone': ''},
            {'label': 'Runs on one tyre', 'value': str(len(tyre_data)),
             'sub': 'across the drivers', 'tone': 'info'},
            {'label': 'Tyres used', 'value': ', '.join(compounds_used) or 'not published',
             'sub': 'across the field', 'tone': ''},
        ])
        fig = go.Figure()

        for driver in drivers:
            dt = tyre_data[tyre_data['Driver'] == driver].sort_values('Stint')
            for _, stint in dt.iterrows():
                compound = str(stint.get('Compound', 'MEDIUM')).upper()
                start = int(stint.get('StartLap', 1))
                end = int(stint.get('EndLap', start + 10))
                width = end - start + 1
                color = COMPOUND_COLORS.get(compound, '#888')
                fig.add_trace(go.Bar(
                    x=[width], y=[driver], orientation='h', base=start - 1,
                    marker_color=color, marker_line_color='rgba(0,0,0,0.4)',
                    marker_line_width=0.6, showlegend=False,
                    text=compound[0], textposition='inside',
                    textfont=dict(
                        color='white' if compound in ['SOFT', 'WET'] else '#111',
                        size=10, family=FONT
                    ),
                    hovertemplate=(
                        f"<b>{driver}</b><br>"
                        f"Tyre: {compound}<br>"
                        f"Lap {start}-{end}<br>"
                        f"Laps on this tyre: {width}"
                        "<extra></extra>"
                    ),
                ))

        total_laps = int(tyre_data['EndLap'].max()) if 'EndLap' in tyre_data.columns else 50
        fig.update_layout(**_fig_layout(height=scaled_height('tall', len(drivers)),
            xaxis=dict(range=[0, total_laps + 2],
                       showgrid=False), xaxis_title=dict(text="Lap", font=LABEL_FONT),
            yaxis=dict(categoryorder='array', categoryarray=drivers[::-1], showgrid=True),
            barmode='overlay',
        ))
        show_plotly_chart(fig)
        _compound_legend(tyre_data['Compound'])
        caption("One bar per run of laps on one tyre, coloured by the tyre.")

    # -- Pit stops --
    ui.section("Pit stops", "measured where telemetry allows, modelled otherwise",
            hint="When each driver stopped and how long the stop took in the "
                 "pit lane.")

    pit_stops = context.pit_stops
    if pit_stops is None or pit_stops.empty:
        empty("No pit stop data available")
    else:
        # Driver filter for pit stops. The default follows the quickest cars of
        # the session (an alphabetical default put backmarkers on screen).
        pit_all_drivers = ordered_drivers(session, pit_stops)
        pit_drv_filter = st.multiselect(
            "Filter Drivers", pit_all_drivers,
            default=pit_all_drivers[:min(DEFAULT_DRIVERS, len(pit_all_drivers))],
            max_selections=10, key="pit_drv",
        )

        # The filter applies to the KPIs as well as to the chart: a metric
        # that ignores the filter right above it is a different number from
        # the one the reader asked for.
        if pit_drv_filter:
            pit_filtered = pit_stops[pit_stops['Driver'].isin(pit_drv_filter)]
        else:
            pit_filtered = pit_stops

        avg_time = pit_filtered['PitTime'].mean() if not pit_filtered.empty else float('nan')
        fastest = pit_filtered['PitTime'].min() if not pit_filtered.empty else float('nan')
        mode = pit_filtered['Lap'].mode() if not pit_filtered.empty else pd.Series(dtype=float)
        popular = mode.iloc[0] if not mode.empty else None

        kpis([
            {'label': 'Total stops', 'value': str(len(pit_filtered)),
             'sub': 'in the selection', 'tone': ''},
            {'label': 'Average stop', 'value': f"{avg_time:.1f}s" if avg_time == avg_time else 'n/a',
             'sub': 'pit lane time', 'tone': ''},
            {'label': 'Fastest stop', 'value': f"{fastest:.1f}s" if fastest == fastest else 'n/a',
             'sub': 'best of the selection', 'tone': 'good'},
            {'label': 'Popular lap', 'value': f"L{int(popular)}" if popular is not None else 'n/a',
             'sub': 'most common stop lap', 'tone': ''},
        ])

        n_measured = int(pit_filtered['Measured'].sum()) if 'Measured' in pit_filtered.columns \
            else len(pit_filtered)
        if n_measured < len(pit_filtered):
            estimate_note(
                f"{n_measured}/{len(pit_filtered)} stop times measured from telemetry; "
                f"the rest use the model default "
                f"({MODEL_CONFIG['pit_loss_sec']:.0f}s, estimated)."
            )

        fig_pit = go.Figure()
        fig_pit.add_trace(go.Scatter(
            x=pit_filtered['Lap'], y=pit_filtered['PitTime'],
            mode='markers',
            marker=dict(
                size=13, color=pit_filtered['PitTime'],
                colorscale=[[0, '#3ECF8E'], [0.5, '#FFB020'], [1, '#E10600']],
                showscale=True, colorbar=dict(title="Seconds", thickness=10, len=0.5),
                line=dict(width=1, color='rgba(255,255,255,0.3)'),
            ),
            text=pit_filtered['Driver'],
            hovertemplate="%{text}<br>Lap %{x}<br>%{y:.1f}s<extra></extra>",
        ))
        fig_pit.update_layout(**_fig_layout(height=chart_height('block'),
            xaxis_title=dict(text="Lap", font=LABEL_FONT),
            yaxis_title=dict(text="Pit Time (s)", font=LABEL_FONT),
        ))
        show_plotly_chart(fig_pit)

        pit_table = ra.pit_summary(pit_filtered)
        if pit_table.get('status') == 'Available':
            caption(
                f"Pit summary: {pit_table['total']} stops, average "
                f"{pit_table['average_s']:.1f}s, fastest "
                f"{pit_table['fastest_s']:.1f}s"
                + (f", window laps {pit_table['window'][0]}-"
                   f"{pit_table['window'][1]}." if pit_table.get('window')
                   else "."))

    ui.section("Tyre degradation per stint", "estimated — clean laps",
            hint="The fitted lap-time slope of each stint (seconds per "
                 "lap). Traffic, fuel load and safety cars are not "
                 "controlled for, so read this as a wear proxy, not a "
                 "pure tyre measurement.")
    slopes = ra.degradation_slope(laps)
    if not slopes.empty:
        rows = [[row['Driver'], str(int(row['Stint'])),
                 str(int(row['Laps'])), _num(row['Slope_s_per_lap'], 3)]
                for _, row in slopes.sort_values(
                    'Slope_s_per_lap', ascending=False).iterrows()]
        data_table(['Driver', 'Stint', 'Laps', 'Slope (s/lap)'], rows)

    # -- Strategy simulator --
    ui.section("What a different plan would have cost",
            "estimated — a projection, not a recommendation",
            hint="A projection of the total race time of a one-stop and a "
                 "two-stop plan, built on the model's own assumptions about "
                 "tyres, fuel and the time lost in the pits.")

    estimate_note(
        "A projection of the race time of a one-stop and a two-stop plan, "
        "built on the model's own assumptions about how fast the tyres wear "
        "out, the fuel burn and the time lost in the pits. It is estimated, "
        "it is not a recommendation, and the confidence label follows the "
        "number of clean laps behind the lap time on fresh tyres."
    )

    # The clean-lap sample of this session is what the base lap time (and the
    # degradation rates) rest on: it drives the confidence label.
    clean_sample = int(len(context.clean_laps))

    sc1, sc2 = st.columns(2)
    with sc1:
        base_lap = st.number_input(
            "Lap time on fresh tyres (s)", 80.0, 120.0,
            float(MODEL_CONFIG['default_base_lap_time']), step=0.1, key="strat_base")
        total_laps_sim = st.number_input(
            "Race length (laps)", 10, 80, int(MODEL_CONFIG['default_total_laps']),
            key="strat_tot")
    with sc2:
        driver_sim = st.text_input(
            "Driver (a label only — the model has no data per driver)", "VER",
            key="strat_drv")
        start_tire = st.selectbox("Starting tyre", ["SOFT", "MEDIUM", "HARD"],
                                  key="strat_tire")

    current_lap_sim = st.slider("Current Lap", 0, int(total_laps_sim), 0,
                                 key="strat_lap", label_visibility="collapsed")

    if st.button("Project this strategy", type="primary", key="run_sim"):
        sim = RaceStrategySimulator(base_lap, total_laps_sim,
                                    n_sample=clean_sample)
        strategy = sim.predict_strategy(driver_sim, start_tire, current_lap_sim)

        level = strategy.get('level', 'insufficient')
        confidence = strategy.get('confidence', 'low')

        kpis([
            {'label': 'One-stop total', 'value': f"{strategy['1_stop_time']:.1f}s",
             'sub': 'estimated', 'tone': ''},
            {'label': 'Two-stop total', 'value': f"{strategy['2_stop_time']:.1f}s",
             'sub': 'estimated', 'tone': ''},
            {'label': 'Difference', 'value': f"{strategy['delta']:.1f}s",
             'sub': 'estimated', 'tone': 'info'},
            {'label': 'How much data', 'value': str(confidence),
             'sub': f"{level} confidence, {strategy.get('n_sample', 0)} clean laps",
             'tone': 'good' if level in ('good', 'moderate') else 'warn'},
        ])
        confidence_tag(level)

        caption(
            f"One-stop plan: {' → '.join(strategy.get('compounds_1_stop', []))} at "
            f"{strategy.get('stop_laps_1_stop')}; two-stop plan: "
            f"{' → '.join(strategy.get('compounds_2_stop', []))} at "
            f"{strategy.get('stop_laps_2_stop')}."
        )
        if strategy.get('reason'):
            note(f"Insufficient data: {strategy['reason']}.", kind='warn')
        elif level == 'insufficient':
            note(
                "The projection rests on the model's constants only (no clean "
                "laps behind the base lap time), so it is labelled insufficient "
                "rather than presented as a recommendation.",
                kind='warn',
            )
        else:
            note(
                f"Under these assumptions the faster plan is "
                f"{strategy['recommended']}. It is a projection, not a "
                f"recommendation, and it says nothing about traffic, weather "
                f"or the race situation."
            )
        with st.expander("Assumptions behind this projection"):
            for assumption in strategy.get('assumptions', []):
                st.markdown(f"- {assumption}")

    ui.section("Comparing three plans", "estimated — ranked race-time projection",
            hint="Three candidate tyre plans, ranked by the race time each "
                 "one is projected to take.")

    estimate_note(
        "Ranked race-time projection across candidate strategies. Times are "
        "estimates built on explicit assumptions (how fast the tyres wear "
        "out, the fuel burn and the time lost in the pits); the fastest is "
        "only the fastest under those assumptions, not a recommendation."
    )

    mid = int(total_laps_sim) // 2
    scenarios = [
        Scenario(name=f"{start_tire}->HARD @ L{mid}",
                 compounds=[start_tire, "HARD"], stop_laps=[mid]),
        Scenario(name=f"{start_tire}->HARD @ L{mid - 3}",
                 compounds=[start_tire, "HARD"], stop_laps=[max(1, mid - 3)]),
        Scenario(name=f"{start_tire}->MEDIUM->HARD",
                 compounds=[start_tire, "MEDIUM", "HARD"],
                 stop_laps=[mid // 2, mid]),
    ]
    prov1, prov2 = st.columns(2)
    with prov1:
        traffic_pen = st.number_input("Traffic penalty (s/lap)", 0.0, 2.0, 0.0,
                                      step=0.1, key="strat_traffic")
    with prov2:
        rival_resp = st.number_input("Rival response (s, one-off)", 0.0, 30.0, 0.0,
                                     step=0.5, key="strat_rival")

    cmp_df = compare_scenarios(
        scenarios, total_laps=int(total_laps_sim), base_lap_time=float(base_lap),
        traffic_penalty=float(traffic_pen), rival_response=float(rival_resp)
    )
    if not cmp_df.empty:
        rows = []
        for _, r in cmp_df.iterrows():
            rows.append([
                str(int(r['Rank'])) if pd.notna(r.get('Rank')) else '-',
                str(r.get('Name', '')),
                _num(r.get('TotalTime'), 1),
                _num(r.get('GapToBest'), 1),
                str(int(r['Stops'])) if pd.notna(r.get('Stops')) else '-',
                str(r.get('Compounds', '') or '-'),
                str(r.get('Risk', '') or '-'),
                _confidence_cell(r.get('Confidence')),
            ])
        data_table(
            ['Rank', 'Plan', 'Projected race time (s, est)',
             'Behind the best plan (s, est)', 'Stops', 'Tyres', 'Risk',
             'Confidence'],
            rows,
        )

        und = undercut_delta(total_laps=int(total_laps_sim),
                             compound_new="HARD", compound_old=start_tire,
                             base_lap_time=float(base_lap), earlier_by=1,
                             rival_response=float(rival_resp))
        direction = "faster" if und['delta_sec'] < 0 else "slower"
        caption(
            f"Stopping earlier than the rival to get fresher tyres (an "
            f"undercut): stopping 1 lap earlier projects "
            f"{abs(und['delta_sec']):.1f}s {direction} (estimated)."
        )
        ui.verdict(_scenario_verdict(cmp_df))
    else:
        empty("Not enough laps to compare scenarios")
        ui.verdict("There are not enough laps in this session to compare the "
                "plans, so no strategy is recommended here.")

    # -- Pit Window Radar --------------------------------------------------
    ui.section("The best lap to stop on",
            "estimated — fitted to each driver's own clean laps",
            hint=glossary_terms.short('pit window') + " The window is the "
                 "range of laps within a second of the best answer.")

    estimate_note(
        "Per-driver best stop lap, searched with the same strategy engine as "
        "above but fitted to each driver's own clean laps (how fast the car "
        "went on a fresh tyre, and how fast the tyre wears out). The window "
        "is the range of stop laps within a second of that best answer. "
        "Confidence follows the clean-lap sample: good at 15+ laps, moderate "
        "at 8+, low at 2+."
    )

    pw_c1, pw_c2 = st.columns([1, 2])
    with pw_c1:
        n_stops_sel = st.radio("How many stops to plan for", [1, 2], index=0,
                               horizontal=True, key="strat_pitstops")
    with pw_c2:
        caption(
            "Drivers whose tyre wear cannot be fitted reliably show a blank "
            "window rather than an invented early stop."
        )

    pw = pit_window(laps, n_stops=int(n_stops_sel), tolerance=1.0)
    if pw.empty:
        empty("Not enough clean-lap data to compute a pit window for this session")
    else:
        pw_shown = pw.copy()
        pw_shown['Window'] = pw_shown.apply(
            lambda r: f"L{int(r['WindowStart'])}-L{int(r['WindowEnd'])}"
            if pd.notna(r['WindowStart']) and pd.notna(r['WindowEnd']) else "-",
            axis=1,
        )
        pw_shown['Optimal stop'] = pw_shown['OptimalLap'].apply(
            lambda v: f"L{int(v)}" if pd.notna(v) else "-"
        )
        # A two-stop plan has a second stop lap: without it the reader only
        # sees the first half of the strategy they asked for.
        if 'OptimalLap2' in pw_shown.columns:
            pw_shown['Second stop'] = pw_shown['OptimalLap2'].apply(
                lambda v: f"L{int(v)}" if pd.notna(v) else "-"
            )
            pw_shown['Window 2'] = pw_shown.apply(
                lambda r: f"L{int(r['WindowStart2'])}-L{int(r['WindowEnd2'])}"
                if pd.notna(r.get('WindowStart2')) and pd.notna(r.get('WindowEnd2'))
                else "-",
                axis=1,
            )
        two_stops = 'Second stop' in pw_shown.columns
        rows = []
        for _, r in pw_shown.iterrows():
            rows.append([
                str(r.get('Driver', '')),
                str(r.get('Optimal stop', '-')),
                str(r.get('Second stop', '-')) if two_stops else '-',
                str(r.get('Window', '-')),
                str(r.get('Window 2', '-')) if two_stops else '-',
                _num(r.get('BasePace'), 3),
                _num(r.get('DegPerLap'), 3),
                _quality_cell(r.get('Level')),
                _confidence_cell(r.get('Confidence')),
            ])
        data_table(
            ['Driver', 'Best lap to stop on', 'Second stop (two-stop plans)',
             'Laps within a second of the best', 'Second window',
             'Lap time on fresh tyres (s, est)', 'Slower each lap (s, est)',
             'How much data', 'Confidence'],
            rows,
        )
        if 'NStops' in pw.columns and int(pw['NStops'].max()) > 1:
            caption(
                f"Two-stop plans: the first stop is in the “Optimal stop” "
                f"column, the second in “Second stop” "
                f"(searched over {int(pw['Candidates'].max())} candidate "
                f"plans, stride {int(pw['SearchStride'].max())})."
            )
        caption(
            "All figures estimated from this session's clean laps. The best "
            "lap to stop on is a modelling result under stated assumptions, "
            "not a team call."
        )
        ui.verdict(
            "The best lap to stop on is a modelling result for this session's "
            "own clean laps; the windows show how much room there is around "
            "it. It is not a recommendation."
        )


# TAB 3: RACE BATTLES
@st.fragment
def _tab_battles(context, year):
    laps = context.laps
    note(
        "This tab answers 'who was where, and who was quicker?'. It shows the "
        "running order after every lap, how far behind the leader each car "
        "was, and a lap-by-lap comparison of two drivers."
    )

    ui.section("Position evolution", "measured — position at the end of each lap",
            hint="Where each car stood at the end of every lap. A line that "
                 "moves up means the car gained places.")

    pos_laps = laps.copy()
    pos_laps['Position'] = pd.to_numeric(pos_laps['Position'], errors='coerce')

    if 'Position' not in pos_laps.columns:
        empty("Position data not available")
        return

    pos_data = []
    for lap_num in sorted(pos_laps['LapNumber'].unique()):
        lap = pos_laps[pos_laps['LapNumber'] == lap_num]
        for _, row in lap.iterrows():
            if pd.notna(row['Position']):
                pos_data.append({
                    'Lap': int(lap_num), 'Driver': row['Driver'],
                    'Position': int(row['Position']), 'Team': row.get('Team', '')
                })

    if not pos_data:
        empty("No position data")
        return

    pos_df = pd.DataFrame(pos_data)
    color_map = {}
    for d in pos_df['Driver'].unique():
        team = pos_df[pos_df['Driver'] == d]['Team'].iloc[0]
        color_map[d] = team_colour(team, year)

    all_drivers = sorted(pos_df['Driver'].unique())
    view = st.radio("Show", ["All drivers", "Top 3", "Top 5", "Custom"],
                    horizontal=True, key="pos_view")
    if view == "All drivers":
        pos_drivers = all_drivers
        caption(f"Showing all {len(all_drivers)} drivers.")
    elif view == "Top 3":
        pos_drivers = _quickest_drivers(laps, 3)
    elif view == "Top 5":
        pos_drivers = _quickest_drivers(laps, 5)
    else:
        # A saved selection from an earlier session (or a wider field) can
        # exceed max_selections and crash the widget on creation, so the
        # saved state is clamped before the widget exists.
        _saved = st.session_state.get("pos_drv")
        if _saved is not None and len(_saved) > 10:
            st.session_state["pos_drv"] = list(_saved)[:10]
        pos_drivers = st.multiselect(
            "Drivers", all_drivers,
            default=_quickest_drivers(laps, DEFAULT_DRIVERS),
            max_selections=10, key="pos_drv",
        )

    if not pos_drivers:
        empty("No drivers selected")
        return

    gains = []
    for driver, group in pos_df.groupby('Driver'):
        group = group.sort_values('Lap')
        if len(group) >= 2:
            gains.append((driver, int(group['Position'].iloc[0] - group['Position'].iloc[-1])))
    biggest_gain = max(gains, key=lambda item: item[1]) if gains else None

    kpis([
        {'label': 'Drivers shown', 'value': str(len(pos_drivers)),
         'sub': 'position traces', 'tone': ''},
        {'label': 'Laps covered', 'value': str(int(pos_df['Lap'].nunique())),
         'sub': 'in this session', 'tone': 'info'},
        {'label': 'Leaders', 'value': str(int(pos_df['Position'].min())),
         'sub': 'best position held', 'tone': 'accent'},
        {'label': 'Biggest gain', 'value': biggest_gain[0] if biggest_gain else '—',
         'sub': (f"+{biggest_gain[1]} places, start to finish"
                 if biggest_gain and biggest_gain[1] > 0 else 'no net gain'),
         'tone': 'good'},
    ])

    fig_pos = go.Figure()
    pit_map_pos = {}
    try:
        pit_map_pos = pit_laps_by_driver(context.pit_stops)
    except Exception:
        pit_map_pos = {}

    for index, d in enumerate(pos_drivers):
        dp = pos_df[pos_df['Driver'] == d].sort_values('Lap')
        color = PLOT_PALETTE[index % len(PLOT_PALETTE)]
        fig_pos.add_trace(go.Scatter(
            x=dp['Lap'], y=dp['Position'],
            mode='lines+markers', name=d,
            line=dict(color=color, width=3, shape='spline', smoothing=0.4),
            marker=dict(size=7, color=color, opacity=0.85, line=dict(width=1.5, color='white')),
            hovertemplate="<b>{}</b><br>Lap: %{{x}}<br>Position: P%{{y}}<extra></extra>".format(d)))
        for pl in pit_map_pos.get(d, []):
            prow = pos_df[(pos_df['Driver'] == d) & (pos_df['Lap'] == pl)]
            if not prow.empty:
                fig_pos.add_trace(go.Scatter(
                    x=[pl], y=[prow['Position'].iloc[0]],
                    mode='markers', showlegend=False,
                    marker=dict(size=14, color='white', symbol='x', line=dict(width=2.5, color=color))))

    fig_pos.update_layout(**_fig_layout(height=chart_height('tall'),
        xaxis=dict(),
        yaxis=dict(autorange='reversed',
                   tickmode='linear', tick0=1, dtick=1,  gridwidth=2,
                   zeroline=False), yaxis_title=dict(text="Position", font=LABEL_FONT),
        hovermode='x unified'))
    # Safety Car / VSC / Red Flag annotations — only from real race-control
    # messages; the OpenF1 fallback has none, so nothing is invented.
    try:
        rc = get_race_control_messages(context.session)
        if rc is not None and not rc.empty:
            shapes = []
            for _, r in rc.iterrows():
                try:
                    lap_no = int(r.get('Lap'))
                except (TypeError, ValueError):
                    continue
                msg = str(r.get('Message', '') or '')
                flag = str(r.get('Flag', '') or '')
                if 'RED' in flag.upper() or 'red flag' in msg.lower():
                    colour, label = 'rgba(225,6,0,0.18)', 'Red Flag'
                elif 'virtual' in msg.lower():
                    colour, label = 'rgba(255,215,0,0.15)', 'VSC'
                elif 'safety car' in msg.lower() or 'YELLOW' in flag.upper():
                    colour, label = 'rgba(255,135,0,0.15)', 'Safety Car'
                else:
                    continue
                shapes.append(dict(type='rect', x0=lap_no, x1=lap_no + 1,
                                   y0=0, y1=22, fillcolor=colour,
                                   line_width=0, layer='below'))
                fig_pos.add_annotation(
                    x=lap_no, y=1.2, text=label, showarrow=False,
                    font=dict(size=9, color='#F4F5F7'))
            if shapes:
                fig_pos.update_layout(shapes=shapes)
    except Exception:
        pass
    show_plotly_chart(fig_pos)
    caption("Crossed markers are pit stops. Tinted bands are Safety Car / "
            "VSC / Red Flag periods from race control (only when published).")

    ui.section("Position gain and loss", "measured — start vs final",
            hint="Places gained (positive) or lost (negative) between "
                 "the first and last recorded position of each driver. "
                 "A gain alone does not prove an overtake: pit stops "
                 "also move positions.")
    changes = ra.position_changes(laps)
    if not changes.empty:
        rows = [[row['Driver'], str(int(row['Start'])),
                 str(int(row['Final'])),
                 {'value': f"{int(row['Gain']):+d}",
                  'tone': 'good' if row['Gain'] > 0
                  else 'bad' if row['Gain'] < 0 else ''}]
                for _, row in changes.sort_values('Gain',
                                                  ascending=False).iterrows()]
        data_table(['Driver', 'Start', 'Final', 'Gain'], rows)

    ui.section("Gap to leader", "measured — cumulative lap time",
            hint="How far behind the reference driver each car was, in "
                 "seconds, lap by lap. The reference is the driver who "
                 "completed the most laps.")

    drivers_list = sorted(laps['Driver'].unique())
    # Default: skip the drivers already shown in Position Evolution and show
    # the next quickest ones.
    gap_default = _quickest_drivers(laps, DEFAULT_DRIVERS + 4)[DEFAULT_DRIVERS:]
    _saved_gap = st.session_state.get("gap_drv")
    if _saved_gap is not None and len(_saved_gap) > 10:
        st.session_state["gap_drv"] = list(_saved_gap)[:10]
    gap_drivers = st.multiselect("Drivers", drivers_list,
                                 default=gap_default,
                                 max_selections=10, key="gap_drv")

    if not gap_drivers:
        empty("Select at least one driver")
    else:
        # The reference is the driver who completed the most laps (ties broken
        # by the best position), not the one with the lowest cumulative lap
        # time: a driver who crashed on lap 5 has the lowest cumulative time
        # of the field and would otherwise become the "leader" of the chart.
        leader_name, gap_frame = gap_to_leader(laps, gap_drivers)
        if leader_name is None or gap_frame.empty:
            empty("No valid lap times for the selected drivers")
        else:
            fig_gap = go.Figure()
            for index, driver in enumerate(gap_drivers):
                rows = gap_frame[gap_frame['Driver'] == driver].sort_values('LapNumber')
                if rows.empty:
                    continue
                dlaps = laps[(laps['Driver'] == driver) & laps['LapTime'].notna()].sort_values('LapNumber')
                team = dlaps['Team'].iloc[0] if 'Team' in dlaps.columns and not dlaps.empty else ''
                driver_color = PLOT_PALETTE[index % len(PLOT_PALETTE)]
                compounds = dlaps.set_index('LapNumber')['Compound'].to_dict() \
                    if 'Compound' in dlaps.columns else {}
                compound_colors = [COMPOUND_COLORS.get(compounds.get(lap, ''), driver_color)
                                   for lap in rows['LapNumber']]
                fig_gap.add_trace(go.Scatter(
                    x=rows['LapNumber'], y=rows['Gap'], mode='lines+markers', name=driver,
                    line=dict(color=driver_color, width=2.5, shape='spline', smoothing=0.3),
                    marker=dict(size=6, color=compound_colors, opacity=0.8, line=dict(width=1, color=driver_color)),
                    hovertemplate="<b>{}</b><br>Lap: %{{x}}<br>Gap: %{{y:.2f}}s<extra></extra>".format(driver)))
            fig_gap.update_layout(**_fig_layout(height=chart_height('feature'),
                xaxis=dict(),
                yaxis=dict(zeroline=True,
                           zerolinecolor='rgba(255,255,255,0.3)', zerolinewidth=2), yaxis_title=dict(text="Gap (s)", font=LABEL_FONT),
                hovermode='x unified'))
            show_plotly_chart(fig_gap)
            caption(
                f"Gap to {leader_name} (the driver who completed the most laps). "
                f"The gap is only plotted for laps the reference driver actually "
                f"completed."
            )
            gap_table = ra.gap_stats(gap_frame)
            if not gap_table.empty:
                # Lapped status: laps completed vs the reference driver.
                try:
                    ref_laps = int(
                        laps[laps['Driver'] == leader_name]['LapNumber']
                        .max())
                except Exception:
                    ref_laps = 0
                rows = []
                for _, row in gap_table.iterrows():
                    drv = row['Driver']
                    try:
                        drv_laps = int(
                            laps[laps['Driver'] == drv]['LapNumber'].max())
                        laps_down = ref_laps - drv_laps
                        status = (f"Lapped +{laps_down}" if laps_down > 0
                                  else "Same lap")
                    except Exception:
                        status = "Unknown"
                    rows.append([drv, _num(row['LargestGap_s'], 2),
                                 _num(row['SmallestGap_s'], 2),
                                 _num(row['FinalGap_s'], 2),
                                 _num(row['ClosingRate_s_per_lap'], 3),
                                 status])
                data_table(['Driver', 'Largest gap (s)', 'Smallest gap (s)',
                            'Final gap (s)', 'Closing rate (s/lap)',
                            'Lapped status'], rows)

    ui.section("Which of the two was quicker", "estimated — clean laps only",
            hint="A lap-by-lap comparison of two drivers on their clean "
                 "laps, with the time difference on each lap.")

    c1, c2 = st.columns(2)
    with c1:
        battle_a = st.selectbox("Driver A", drivers_list, key="battle_a")
    with c2:
        battle_b = st.selectbox("Driver B", [d for d in drivers_list if d != battle_a],
                                key="battle_b")

    # Clean laps only: the pit-entry, pit-exit and Safety Car laps are not
    # comparable to a racing lap, and counting them as "faster laps" would
    # mis-state the battle.
    battle_pace = compare_driver_pace(laps, battle_a, battle_b)
    la = battle_pace['laps_a']
    lb = battle_pace['laps_b']

    if battle_pace['avg_a'] is None or battle_pace['avg_b'] is None:
        empty("Not enough clean lap data to compare these two drivers")
        return

    team_a = la['Team'].iloc[0] if 'Team' in la.columns else ''
    team_b = lb['Team'].iloc[0] if 'Team' in lb.columns else ''
    color_a = team_colour(team_a, year)
    color_b = team_colour(team_b, year)

    la_times = la.set_index('LapNumber')['LapTime_s'].to_dict()
    lb_times = lb.set_index('LapNumber')['LapTime_s'].to_dict()
    common = sorted(set(la_times) & set(lb_times))
    faster_a = faster_b = 0
    diffs = []
    for lap_n in common:
        ta = la_times[lap_n]
        tb = lb_times[lap_n]
        if ta < tb:
            faster_a += 1
        else:
            faster_b += 1
        diffs.append(ta - tb)

    avg_diff = np.mean(diffs) if diffs else 0
    pct_a = faster_a / len(common) * 100 if common else 0
    pct_b = faster_b / len(common) * 100 if common else 0

    kpis([
        {'label': f'{battle_a} quicker laps', 'value': str(faster_a),
         'sub': f'{pct_a:.0f}% of common laps', 'tone': 'accent'},
        {'label': f'{battle_b} quicker laps', 'value': str(faster_b),
         'sub': f'{pct_b:.0f}% of common laps', 'tone': 'info'},
        {'label': 'Common clean laps', 'value': str(len(common)),
         'sub': f'avg {abs(avg_diff):.3f}s gap', 'tone': ''},
        {'label': 'How much data', 'value': str(battle_pace['confidence']),
         'sub': f"{battle_pace['level']} confidence", 'tone': ''},
    ])
    confidence_tag(battle_pace['level'])

    estimate_note(
        f"Estimated from clean laps only: {battle_pace['n_a']} clean laps for "
        f"{battle_a}, {battle_pace['n_b']} for {battle_b}."
    )

    # Lap-by-lap delta bar chart
    fig_b = go.Figure()
    lap_nums = common
    delta_vals = diffs

    colors = [color_a if d < 0 else color_b for d in delta_vals]
    fig_b.add_trace(go.Bar(
        x=lap_nums, y=delta_vals, marker_color=colors,
        name=f"{battle_a} vs {battle_b}",
        marker_line=dict(width=0),
        hovertemplate=(
            f"<b>Lap %{{x}}</b><br>"
            f"{battle_a} - {battle_b}: %{{y:+.3f}}s"
            "<extra></extra>"
        ),
    ))
    fig_b.update_layout(**_fig_layout(height=chart_height('block'),
        xaxis_title=dict(text="Lap", font=LABEL_FONT),
        yaxis_title=dict(text=f"Time difference ({battle_a} - {battle_b}) in seconds", font=LABEL_FONT),
        yaxis=dict(zeroline=True, zerolinecolor=ZERO_LINE, zerolinewidth=2),
        bargap=0.15,
    ))
    show_plotly_chart(fig_b)
    caption("A negative number means the first driver was quicker on that "
            "lap.")
    if common:
        ui.verdict(
            f"On the {len(common)} laps both drivers ran cleanly, {battle_a} "
            f"was quicker on {faster_a} of them and {battle_b} on {faster_b}, "
            f"an average of {abs(avg_diff):.3f}s per lap "
            f"({battle_pace['confidence']} confidence)."
        )
    else:
        ui.verdict("The two drivers have no lap in common in this session, so "
                "there is not enough data to compare them.")


# TAB 4: PERFORMANCE INSIGHTS
@st.fragment
def _tab_insights(context, year):
    laps = context.laps
    session = context.session
    note(
        "This tab answers 'how did the session go?'. It turns the session "
        "into one score per driver, then looks at how fast the tyres wore "
        "out and at what the officials said during the session."
    )

    ui.section("Automated key insights", "deterministic — rule based",
            hint="Findings computed from this session's laps, each with "
                 "the numbers behind it. Nothing is narrated by hand.")
    found = ra.automated_insights(laps, context.pit_stops)
    if found:
        takeaways([(item['title'],
                    f"{item['text']} (Evidence: {item['evidence']}.)")
                   for item in found])
    else:
        empty("No insights could be derived from this session.",
              "The session needs timed laps before insights can be computed.")

    ui.section("Driver performance scores", "estimated — clean laps only",
            hint="One number per driver, built from how fast they went, how "
                 "steady their lap times were and how the race went for them.")

    estimate_note(
        "The score blends three parts -- how fast the driver went, how steady "
        "their lap times were and how the race went for them -- computed from "
        "clean laps only (the lap into and out of the pit, the laps under a "
        "safety car and the laps with a doubtful time are removed). Every row "
        "shows the number of clean laps behind it and the confidence that "
        "follows from it; a driver with too few clean laps gets no score at "
        "all."
    )

    scores_df = driver_performance_scores(laps)
    if scores_df.empty:
        empty("Not enough clean lap data to compute scores")
        return

    scores_df = scores_df.reset_index(drop=True)
    scores_df.index += 1
    scores_df.index.name = 'Rank'

    kpis([
        {'label': 'Drivers scored', 'value': str(len(scores_df)),
         'sub': 'with enough clean laps', 'tone': ''},
        {'label': 'Top score', 'value': f"{scores_df['Overall'].iloc[0]:.1f}",
         'sub': str(scores_df['Driver'].iloc[0]), 'tone': 'accent'},
        {'label': 'Spread', 'value': f"{scores_df['Overall'].iloc[0] - scores_df['Overall'].iloc[-1]:.1f}",
         'sub': 'points between first and last', 'tone': 'info'},
    ])

    # -- Radar + leaderboard side-by-side --
    rc1, rc2 = st.columns([1, 1])

    with rc1:
        radar_drivers = st.multiselect(
            "Compare",
            scores_df['Driver'].tolist(),
            default=scores_df['Driver'].head(3).tolist(),
            key="radar_drv",
        )
        if radar_drivers:
            fig_r = go.Figure()
            for index, d in enumerate(radar_drivers):
                row = scores_df[scores_df['Driver'] == d].iloc[0]
                fig_r.add_trace(go.Scatterpolar(
                    r=[row['Pace'], row['Consistency'], row['Race Pace'], row['Overall'], row['Pace']],
                    theta=['How fast', 'How steady', 'The race', 'Overall', 'How fast'],
                    fill='toself', name=d,
                    line_color=team_colour(row['Team'], year),
                    opacity=0.65,
                ))
            fig_r.update_layout(**_fig_layout(height=chart_height('feature'),
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100],
                                    showticklabels=False, showgrid=False),
                    bgcolor='rgba(0,0,0,0)',
                    angularaxis=dict(tickfont=dict(size=12, color=TEXT_SECONDARY)),
                ),
                showlegend=True,
            ))
            show_plotly_chart(fig_r)
        else:
            empty("No drivers selected for the radar")

    with rc2:
        ui.section("Leaderboard",
                hint="The same scores, ranked, with the number of clean laps "
                     "behind each row.")
        rows = []
        for _, r in scores_df.iterrows():
            rows.append([
                str(r.get('Driver', '')),
                str(r.get('Team', '') or '-'),
                _num(r.get('Pace'), 1),
                _num(r.get('Consistency'), 1),
                _num(r.get('Race Pace'), 1),
                {'value': _num(r.get('Overall'), 1), 'tone': 'strong'},
                str(int(r['N'])) if pd.notna(r.get('N')) else '-',
                _quality_cell(r.get('Level')),
                _confidence_cell(r.get('Confidence')),
            ])
        data_table(
            ['Driver', 'Team', 'How fast they went', 'How steady they were',
             'How the race went', 'Overall score', 'Clean laps',
             'How much data', 'Confidence'],
            rows,
        )

    confidence_tag(str(scores_df['Level'].iloc[0])
                   if 'Level' in scores_df.columns else 'insufficient')
    caption("The confidence tag belongs to the top-scored driver's sample; "
            "every row of the table carries its own.")
    ui.verdict(_scores_verdict(scores_df))

    # Tyre Degradation Analysis
    ui.section("How fast the tyres wore out",
            "estimated — fitted per driver and compound",
            hint=glossary_terms.short('degradation') + " The rate is fitted "
                 "from the laps of each driver on each tyre.")

    drivers_list = sorted(laps['Driver'].unique())
    deg_drvs = st.multiselect("Drivers", drivers_list,
                              default=_quickest_drivers(laps, DEFAULT_DRIVERS),
                              max_selections=6, key="deg_drv")

    if not deg_drvs:
        empty("No drivers selected")
        return

    # 1. Lap time against the laps on the tyre
    ui.section("Lap time against laps on this tyre",
            hint="Each dot is one lap, placed against the number of laps the "
                 "tyre had already been used. The dashed line is the fitted "
                 "trend.")

    fig_deg = go.Figure()
    all_tyrelife_y = []
    deg_data = []

    for d in deg_drvs:
        dl = laps[
            (laps['Driver'] == d) &
            laps['LapTime'].notna() &
            laps['TyreLife'].notna()
        ].copy()
        if dl.empty:
            continue
        dl = dl.sort_values('LapNumber')
        dl['LapTime_s'] = dl['LapTime'].dt.total_seconds()
        all_tyrelife_y.extend(dl['LapTime_s'].tolist())

        for compound, stint_dl in dl.groupby('Compound') if 'Compound' in dl.columns else [('ALL', dl)]:
            if stint_dl.empty or len(stint_dl) < 2:
                continue
            stint_dl = stint_dl.sort_values('TyreLife')
            x = stint_dl['TyreLife'].values.astype(float)
            y = stint_dl['LapTime_s'].values
            comp_color = COMPOUND_COLORS.get(compound, '#888')

            fig_deg.add_trace(go.Scatter(
                x=x, y=y, mode='markers',
                name=f"{d} {compound}",
                legendgroup=f"{d}_{compound}",
                marker=dict(size=7, color=comp_color, opacity=0.7,
                            line=dict(width=1, color='rgba(255,255,255,0.35)')),
                hovertemplate=(
                    f"<b>{d}</b> ({compound})<br>"
                    "Laps on this tyre: %{x:.0f}<br>"
                    "Lap time: %{customdata}"
                    "<extra></extra>"
                ),
                customdata=[format_f1_time(pd.Timedelta(seconds=t)) for t in y],
            ))

            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            x_line = np.linspace(x.min(), x.max(), 30)
            y_line = np.polyval(coeffs, x_line)
            fig_deg.add_trace(go.Scatter(
                x=x_line, y=y_line, mode='lines',
                name=f"{d} trend ({compound})",
                legendgroup=f"{d}_{compound}",
                line=dict(color=comp_color, width=2.5, dash='dash'),
                showlegend=False,
                hovertemplate=(
                    f"<b>{d}</b> ({compound})<br>"
                    "Degradation: %{customdata:.4f}s/lap"
                    "<extra></extra>"
                ),
                customdata=np.full_like(x_line, slope),
            ))

            total_deg = slope * (x.max() - x.min())
            deg_data.append({
                'Driver': d, 'Compound': compound,
                'Deg (s/lap)': round(slope, 4),
                'Total Loss (s)': round(total_deg, 2),
                'Stint (laps)': "L{}".format(int(stint_dl['LapNumber'].min())) + "-" + "L{}".format(int(stint_dl['LapNumber'].max())),
                'Tyre Age (laps)': "{}-{}".format(int(x.min()), int(x.max())),
                # The fastest lap of the stint, not the fitted fresh-tyre pace:
                # the fit's intercept is a different number.
                'Fastest Lap (s)': round(y.min(), 3),
            })

    if all_tyrelife_y:
        y_min_d, y_max_d = min(all_tyrelife_y) - 0.5, max(all_tyrelife_y) + 0.5
        time_y = _time_axis_layout("Lap Time", (y_min_d, y_max_d))
    else:
        time_y = dict(title=dict(text="Lap Time", font=LABEL_FONT))

    fig_deg.update_layout(**_fig_layout(height=chart_height('feature'),
        xaxis=dict(),
        yaxis=time_y,
        showlegend=True,
    ))
    show_plotly_chart(fig_deg)
    caption("Dots are laps, dashed lines the fitted trend of how much "
            "slower each lap got.")

    # 2. Degradation Rate Comparison
    ui.section("How much slower each lap got",
            hint="The extra time each lap cost, for every driver and tyre "
                 "type in the selection.")

    if deg_data:
        deg_df = pd.DataFrame(deg_data)
        fig_rate = go.Figure()
        sorted_deg = deg_df.sort_values('Deg (s/lap)', ascending=False)
        for _, row in sorted_deg.iterrows():
            comp_color = COMPOUND_COLORS.get(row['Compound'], '#888')
            fig_rate.add_trace(go.Bar(
                x=[row['Deg (s/lap)']],
                y=["{} {}".format(row['Driver'], row['Compound'])],
                orientation='h',
                marker=dict(color=comp_color, line=dict(color='rgba(255,255,255,0.3)', width=1)),
                hovertemplate=(
                    "<b>{}</b> {}<br>".format(row['Driver'], row['Compound']) +
                    "Deg: {:.4f}s/lap<br>".format(row['Deg (s/lap)']) +
                    "Total Loss: {}s<br>".format(row['Total Loss (s)']) +
                    "Stint: {}<extra></extra>".format(row['Stint (laps)'])
                ),
            ))

        fig_rate.update_layout(**_fig_layout(height=scaled_height('tile', len(sorted_deg)),
            xaxis=dict(),
            yaxis=dict(showgrid=False),
            barmode='group', showlegend=False,
        ))
        show_plotly_chart(fig_rate)
    else:
        empty("No degradation fit available for the selected drivers")

    # 3. Tyre Strategy Overview
    ui.section("Tyre plan over the whole session", "measured — stints by lap",
            hint=glossary_terms.short('stint') + " Each bar is one run of "
                 "laps on one tyre.")

    stint_drvs = [d for d in deg_drvs if d in laps['Driver'].values]
    if stint_drvs and 'TyreLife' in laps.columns and 'Compound' in laps.columns:
        all_stints = []
        for d in stint_drvs:
            dl = laps[(laps['Driver'] == d) & laps['LapNumber'].notna()].copy()
            if dl.empty:
                continue
            dl = dl.sort_values('LapNumber')
            dl['stint_id'] = ((dl['Compound'] != dl['Compound'].shift()) |
                              (dl['TyreLife'] <= dl['TyreLife'].shift())).cumsum()
            for _, stint in dl.groupby('stint_id'):
                comp = stint['Compound'].iloc[0]
                all_stints.append({
                    'Driver': d, 'Compound': comp,
                    'LapStart': int(stint['LapNumber'].min()),
                    'LapEnd': int(stint['LapNumber'].max()),
                })

        if all_stints:
            stint_df = pd.DataFrame(all_stints)
            max_lap = int(laps['LapNumber'].max())
            n_drivers = len(stint_drvs)

            fig_strat = go.Figure()
            for i, d in enumerate(stint_drvs):
                drv_stints = stint_df[stint_df['Driver'] == d]
                if drv_stints.empty:
                    continue
                y_pos = n_drivers - 1 - i
                for _, s in drv_stints.iterrows():
                    comp = s['Compound']
                    comp_color = COMPOUND_COLORS.get(comp, '#888')
                    x0 = s['LapStart']
                    x1 = s['LapEnd']
                    width = x1 - x0 + 1
                    fig_strat.add_trace(go.Bar(
                        x=[width], y=[y_pos], base=[x0],
                        orientation='h', width=0.6,
                        marker=dict(color=comp_color, opacity=0.85,
                                    line=dict(width=1.5, color='rgba(255,255,255,0.3)')),
                        showlegend=False,
                        hovertemplate="<b>{}</b><br>Compound: {}<br>Laps {}-{} ({})<extra></extra>".format(d, comp, x0, x1, width),
                    ))
                    mid = (x0 + x1) / 2
                    fig_strat.add_annotation(
                        x=mid, y=y_pos, text="<b>{}</b>".format(comp),
                        showarrow=False, font=dict(size=10, color='#111', family=FONT),
                        xanchor='center', yanchor='middle',
                    )

            fig_strat.update_layout(**_fig_layout(height=scaled_height('tile', n_drivers),
                xaxis=dict(range=[0.5, max_lap + 1]), xaxis_title=dict(text="Lap", font=LABEL_FONT),
                yaxis=dict(tickmode='array', tickvals=list(range(n_drivers)),
                           ticktext=list(reversed(stint_drvs)),
                           tickfont=dict(size=12, color=TEXT_PRIMARY)),
                barmode='overlay', showlegend=False,
            ))
            show_plotly_chart(fig_strat)
            _compound_legend(stint_df['Compound'])
    else:
        empty("No stint data for the selected drivers")

    # 4. Degradation Summary Table
    ui.section("Degradation summary", "estimated — fitted per driver and compound",
            hint="The same numbers as a table. A tyre that wears out fast "
                 "costs a driver more stops.")

    if deg_data:
        sum_df = pd.DataFrame(deg_data)

        rows = []
        for _, r in sum_df.iterrows():
            try:
                deg = float(r['Deg (s/lap)'])
            except (TypeError, ValueError):
                deg = 0.0
            tone = 'bad' if deg > 0.15 else ('accent' if deg > 0.08 else 'good')
            rows.append([
                str(r.get('Driver', '')),
                str(r.get('Compound', '') or '-'),
                {'value': _num(deg, 4), 'tone': tone},
                _num(r.get('Total Loss (s)'), 2),
                str(r.get('Stint (laps)', '-')),
                str(r.get('Tyre Age (laps)', '-')),
                _num(r.get('Fastest Lap (s)'), 3),
            ])
        data_table(
            ['Driver', 'Tyre', 'Slower each lap (s, est)',
             'Time lost over the run (s, est)', 'Laps of the run',
             'Laps on this tyre', 'Fastest lap (s)'],
            rows,
        )

        parsed = []
        for _, r in sum_df.iterrows():
            try:
                parsed.append({
                    'Driver': r['Driver'], 'Compound': r['Compound'],
                    'Deg': float(r['Deg (s/lap)'])
                })
            except Exception:
                pass

        if parsed:
            worst = max(parsed, key=lambda x: x['Deg'])
            best = min(parsed, key=lambda x: x['Deg'])
            avg_deg = sum(x['Deg'] for x in parsed) / len(parsed)
            kpis([
                {'label': 'Highest degradation',
                 'value': "{} ({})".format(worst['Driver'], worst['Compound']),
                 'sub': "{:+.4f}s/lap".format(worst['Deg']), 'tone': 'warn'},
                {'label': 'Lowest degradation',
                 'value': "{} ({})".format(best['Driver'], best['Compound']),
                 'sub': "{:+.4f}s/lap".format(best['Deg']), 'tone': 'good'},
                {'label': 'Field average', 'value': "{:+.4f}".format(avg_deg),
                 'sub': 's/lap', 'tone': 'info'},
            ])
    else:
        empty("No degradation summary available")

    ui.section("What the officials said", "measured",
            hint="The messages the race control sent during the session: "
                 "warnings, penalties, safety cars and the flags on track.")

    try:
        rc = get_race_control_messages(session)
        if rc is not None and not rc.empty:
            rows = []
            for _, r in rc.iterrows():
                lap = r.get('Lap')
                rows.append([
                    str(r.get('Time', '')),
                    str(r.get('Category', '') or '-'),
                    str(r.get('Message', '') or '-'),
                    _flag_cell(r.get('Flag', '')),
                    str(r.get('Scope', '') or '-'),
                    str(r.get('Sector', '') or '-'),
                    f"L{int(lap)}" if pd.notna(lap) else '-',
                ])
            data_table(
                ['Time', 'Category', 'What was said', 'Flag', 'Where',
                 'Part of the lap', 'Lap'],
                rows,
            )
        else:
            empty("No race control messages")
    except Exception as e:
        empty("Race control data unavailable", str(e))

    ui.section("Event timeline", "measured — race control + pit stops",
            hint="Safety Car, VSC, Red Flag, pit stops and retirements "
                 "in lap order. No radio text is invented.")
    try:
        events = []
        try:
            rc_tl = get_race_control_messages(session)
            if rc_tl is not None and not rc_tl.empty:
                for _, r in rc_tl.iterrows():
                    try:
                        lap_no = int(r.get('Lap'))
                    except (TypeError, ValueError):
                        continue
                    msg = str(r.get('Message', '') or '')
                    if any(k in msg.lower() for k in
                           ('safety car', 'virtual', 'red flag', 'restart',
                            'retir', 'dnf', 'dsq')):
                        events.append((lap_no, str(r.get('Time', '')),
                                       'Race control', msg))
        except Exception:
            pass
        try:
            ps = context.pit_stops
            if ps is not None and not ps.empty and 'Lap' in ps.columns:
                for _, r in ps.iterrows():
                    try:
                        lap_no = int(r.get('Lap'))
                    except (TypeError, ValueError):
                        continue
                    events.append((lap_no, '', 'Pit stop',
                                   str(r.get('Driver', ''))))
        except Exception:
            pass
        if events:
            events.sort(key=lambda e: e[0])
            data_table(['Lap', 'Time', 'Type', 'Event'],
                       [[f"L{e[0]}", e[1] or '-', e[2], e[3]]
                        for e in events])
        else:
            empty("No timeline events published for this session")
    except Exception as e:
        empty("Event timeline unavailable", str(e))


# TAB 5: SECTOR & SESSIONS
@st.fragment
def _tab_sector_compare(laps, race_name, year, session_type):
    """Sector / corner insights and session-to-session pace comparison."""
    note(
        "This tab answers 'where was the time lost?'. It splits the lap into "
        "thirds to show which part of the circuit each driver was quick in, "
        "and compares this session with the other session of the same "
        "weekend."
    )

    _render_sector_insights(laps)
    _render_session_compare(laps, race_name, year, session_type)


def _render_sector_insights(laps):
    ui.section("Where each driver was quick or slow",
            "estimated — clean laps only",
            hint=glossary_terms.short('sector') + " The third of the circuit "
                 "where a driver is quick is where the time is won.")

    estimate_note(
        "Where each driver was quick on the lap: the best, the average and "
        "the spread of their times in every third of the circuit, plus the "
        "time difference to the best time anyone set in that third. Figures "
        "are estimates computed from clean laps only (the lap into and out of "
        "the pit and the laps under a safety car are removed)."
    )

    if not any(s in laps.columns for s in ("Sector1", "Sector2", "Sector3")):
        empty("No sector timing available in this session's lap table")
        return

    try:
        summary = sector_summary(laps)
        deficits = sector_deficits(laps)
    except Exception as e:
        empty("Sector analysis unavailable", str(e))
        return

    if summary is None or summary.empty:
        empty("Not enough clean laps to estimate sector pace — analysis skipped")
        return

    sector_columns = [s for s in ("Sector1", "Sector2", "Sector3") if s in laps.columns]
    # Three sector spotlight cards: fastest driver per sector (clean laps).
    sector_cards = []
    for idx, col in enumerate(sector_columns, start=1):
        key = f"Sector{idx}Best"
        if key in summary.columns and 'Driver' in summary.columns:
            best_row = summary.sort_values(key).iloc[0]
            sector_cards.append({
                'label': f'Sector {idx}',
                'value': str(best_row['Driver']),
                'sub': f"{float(best_row[key]):.3f}s best · "
                       f"mean {float(best_row.get(f'Sector{idx}Mean', 0)):.3f}s",
                'tone': 'accent' if idx == 1 else '',
                'icon': 'chart',
            })
    if sector_cards:
        kpis(sector_cards)
    kpis([
        {'label': 'Drivers with sectors', 'value': str(len(summary)),
         'sub': 'clean laps only', 'tone': ''},
        {'label': 'Sectors per lap', 'value': str(len(sector_columns)),
         'sub': 'timed in this session', 'tone': 'info'},
    ])

    c_left, c_right = st.columns(2)
    with c_left:
        ui.section("Per-driver sector stats",
                hint="The best, the average and the spread of each driver's "
                     "times in every third of the lap.")
        data_table(
            ['Driver', 'Clean laps', 'How much data', 'Confidence',
             'Best 1st third (s)', 'Average 1st third (s)',
             'Spread 1st third (s)', 'Best 2nd third (s)',
             'Average 2nd third (s)', 'Spread 2nd third (s)',
             'Best 3rd third (s)', 'Average 3rd third (s)',
             'Spread 3rd third (s)'],
            _sector_rows(summary),
        )
    with c_right:
        ui.section("Time difference to the best in that part of the circuit",
                hint="How much slower than the fastest driver in the session "
                     "each driver was in each third of the lap. Zero means "
                     "the driver set the best time of the session.")
        data_table(
            ['Driver', 'Clean laps', 'How much data', 'Confidence',
             '1st third (s)', '2nd third (s)', '3rd third (s)'],
            _deficit_rows(deficits),
        )

    confidence_tag(_weakest_level(summary['Level'].tolist()
                                  if 'Level' in summary.columns else []))
    caption(
        "How much data follows the number of clean laps behind each row "
        "(good at 15 or more, medium at 8 or more, low at 2 or more). A row "
        "marked low or not enough data is a small sample — treat the "
        "estimate with caution."
    )
    ui.verdict(_sector_verdict(summary, deficits))

    # Multi-line sector comparison: best sector time per driver (selectable).
    ui.section("Sector time comparison",
            "estimated — clean laps only",
            hint="Best sector time per driver, one line per selected driver. "
                 "Lower is quicker in that third.")
    sector_drivers = st.multiselect(
        "Drivers", summary['Driver'].tolist(),
        default=summary['Driver'].head(3).tolist(),
        max_selections=6, key="sector_drv")
    if sector_drivers:
        fig_s = go.Figure()
        for index, drv in enumerate(sector_drivers):
            row = summary[summary['Driver'] == drv].iloc[0]
            xs, ys = [], []
            for idx in range(1, len(sector_columns) + 1):
                key = f"Sector{idx}Best"
                if key in summary.columns and pd.notna(row.get(key)):
                    xs.append(f"Sector {idx}")
                    ys.append(float(row[key]))
            fig_s.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines+markers', name=drv,
                line=dict(color=PLOT_PALETTE[index % len(PLOT_PALETTE)],
                          width=2.5),
                marker=dict(size=8)))
        fig_s.update_layout(**_fig_layout(height=chart_height('row'),
            yaxis_title=dict(text='Best sector time (s)', font=LABEL_FONT)))
        show_plotly_chart(fig_s)


def _render_session_compare(laps, race_name, year, session_type):
    ui.section("This session against the other one",
            "estimated — normalised base pace",
            hint="The same stint measured in two sessions of the same "
                 "weekend, with the wear of the tyre taken out, so the two "
                 "can be compared.")

    estimate_note(
        "Compares how fast each driver went on fresh tyres, per driver and "
        "per stint, across two sessions. Taking the tyre age out first means "
        "the time difference is a fair comparison rather than raw lap time."
    )

    # Stint keys only align meaningfully between race-format sessions (Race ↔
    # Sprint). Qualifying laps have stint numbers that don't correspond to race
    # stints, so pairing them here would manufacture a misleading delta.
    partners = {"Race": ["Sprint"], "Sprint": ["Race"]}.get(session_type, [])
    if not partners:
        empty(
            "Stint-keyed pace comparison is offered between race-format "
            "sessions (Race ↔ Sprint). Qualifying lap stint numbers don't "
            "correspond to race stints, so it is excluded rather than showing "
            "a misleading delta."
        )
        return

    ref = st.selectbox(
        "Compare against", partners, key="compare_ref_session",
        help="The session to compare with. The time difference is (this "
             "session - the other one); a negative number means this "
             "session was faster.",
    )
    caption(
        f"Comparing {session_type} (this session) against {ref} (the other "
        f"one). A negative time difference means this session was faster with "
        f"the same tyre age."
    )

    with st.spinner(f"Loading {ref} session..."):
        try:
            setup_fastf1_cache()
            ref_session = load_fastf1_session(year, race_name, ref)
        except Exception as e:
            empty("Reference session unavailable", str(e))
            return

    if ref_session is None or ref_session.laps is None or ref_session.laps.empty:
        empty(f"No lap data available for {race_name} ({ref})")
        return

    out = compare_sessions(laps, ref_session.laps)
    if out is None or out.empty:
        empty(
            "No matching drivers/stints with enough clean laps in both "
            "sessions — comparison is skipped rather than reported on noise."
        )
        return

    rows = []
    for _, r in out.iterrows():
        stint = r.get('Stint')
        rows.append([
            str(r.get('Driver', '')),
            f"L{int(stint)}" if pd.notna(stint) else '-',
            str(r.get('Compound', '') or '-'),
            _num(r.get('BasePace_A'), 3),
            _num(r.get('BasePace_B'), 3),
            _num(r.get('Delta'), 3),
            str(int(r['N_A'])) if pd.notna(r.get('N_A')) else '-',
            str(int(r['N_B'])) if pd.notna(r.get('N_B')) else '-',
            _quality_cell(r.get('Level')),
            _confidence_cell(r.get('Confidence')),
        ])
    data_table(
        ['Driver', 'Stint', 'Tyre',
         f'Lap time on fresh tyres, {session_type} (s, est)',
         f'Lap time on fresh tyres, {ref} (s, est)',
         'Time difference (s, est)', f'Clean laps, {session_type}',
         f'Clean laps, {ref}', 'How much data', 'Confidence'],
        rows,
    )
    confidence_tag(_weakest_level(out['Level'].tolist()
                                  if 'Level' in out.columns else []))
    caption(
        "All figures are estimates with a confidence level from each driver's "
        "clean-lap sample. Rows without enough clean laps in either session "
        "are excluded."
    )
    ui.verdict(_session_verdict(out, session_type, ref))


# TAB 6: SETUP & AERO
@st.fragment
def _tab_setup_aero(year, selected_race, session_type='Race'):
    note(
        "This tab answers 'what was the car set up like?'. It is a rough "
        "guess at the setup, read from how the car turns in, and it needs the "
        "car's own telemetry — so it is only loaded when you ask for it."
    )

    ui.section("A rough guess at the car setup",
            f"estimated — {session_type} session",
            hint=glossary_terms.short('setup fingerprint') + " Everything here "
                 "is estimated from the car's own telemetry, not measured by "
                 "the team.")

    estimate_note(
        "A rough guess at the setup, worked out from public telemetry only. "
        "Top speed is the median of the best clean-lap trap readings; the "
        "cornering numbers are the mean speeds through the slow, medium and "
        "fast corners detected on each driver's fastest clean lap, plus the "
        "mean peak braking. Everything is estimated; confidence follows the "
        "clean-lap sample. Trap speeds include slipstream and DRS effects "
        "that cannot be fully removed from public data."
    )

    # The fingerprint needs the car's telemetry, which is by far the largest
    # download in the app. It is opt-in: opening this view must not hold the
    # whole page hostage to it. The result is cached, so it is instant
    # afterwards.
    ready = st.session_state.get('setup_fingerprint_loaded', False)
    if not ready:
        note(
            "This reading needs the car's own telemetry for the session. It "
            "is the largest download in the app, so it is never loaded "
            "automatically: press the button and it is cached for the rest of "
            "the session.",
            kind='info',
        )
        if st.button('Load the setup reading', key='load_fingerprint',
                     type='primary'):
            st.session_state.setup_fingerprint_loaded = True
            st.rerun()
        return

    with st.spinner('Reading the car telemetry and building the fingerprint '
                     '(the first load takes a minute)...'):
        fp = _cached_fingerprint(year, selected_race, session_type)
    if fp.empty:
        empty("Not enough clean-lap data to build a fingerprint for this session")
        return

    corners = pd.to_numeric(fp.get('CornersDetected'), errors='coerce')
    kpis([
        {'label': 'Drivers read', 'value': str(len(fp)),
         'sub': session_type, 'tone': ''},
        {'label': 'Corners read',
         'value': str(int(corners.max())) if not corners.empty else 'n/a',
         'sub': 'on the fastest clean lap', 'tone': 'info'},
        {'label': 'How much data', 'value': str(fp['Confidence'].iloc[0]) if 'Confidence' in fp.columns else 'n/a',
         'sub': 'from the clean-lap sample', 'tone': 'accent'},
    ])
    confidence_tag(str(fp['Level'].iloc[0])
                   if 'Level' in fp.columns else 'insufficient')

    norm = normalise_fingerprint(fp)

    rows = []
    for _, r in fp.iterrows():
        corners_read = r.get('CornersDetected')
        rows.append([
            str(r.get('Driver', '')),
            str(r.get('Team', '') or '-'),
            _num(r.get('TopSpeed'), 1),
            _num(r.get('SlowApex'), 1),
            _num(r.get('MediumApex'), 1),
            _num(r.get('FastApex'), 1),
            _num(r.get('BrakeDecelG'), 2),
            str(int(corners_read)) if pd.notna(corners_read) else '-',
            _confidence_cell(r.get('Confidence')),
        ])
    data_table(
        ['Driver', 'Team', 'Top speed (km/h)',
         'Speed through a slow corner (km/h)',
         'Speed through a medium corner (km/h)',
         'Speed through a fast corner (km/h)', 'Braking (g)', 'Corners',
         'Confidence'],
        rows,
    )
    ui.verdict(
        "The table is a rough reading of how each car turned in, taken from "
        "one clean lap per driver: the numbers are close together, so small "
        "differences are noise."
    )

    # Setup trade-off map: straight-line index vs cornering index.
    ui.section("Straight-line speed against cornering speed",
            "estimated — session best = 1.000",
            hint="Each dot is a driver. A car near the top right was quick "
                 "both in a straight line and through the corners.")

    idx_cols = ['TopSpeedIdx', 'SlowApexIdx', 'MediumApexIdx', 'FastApexIdx']
    if all(c in norm.columns for c in idx_cols):
        plot_df = norm.copy()
        corner_idx = plot_df[idx_cols[1:]].mean(axis=1, skipna=True)
        plot_df['CorneringIdx'] = corner_idx.round(3)
        plot_df = plot_df.dropna(subset=['TopSpeedIdx', 'CorneringIdx'])
        if not plot_df.empty:
            fig = go.Figure()
            for index, r in plot_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[r['TopSpeedIdx']], y=[r['CorneringIdx']], mode='markers+text',
                    text=[r['Driver']], textposition='top center',
                    textfont=dict(size=9, color=TEXT_SECONDARY),
                    marker=dict(size=11,
                                color=team_colour(r.get('Team'), year),
                                line=dict(width=1, color='rgba(255,255,255,0.4)')),
                    name=r['Driver'],
                    hovertemplate=(
                        f"<b>{r['Driver']}</b><br>Top speed idx: %{{x:.3f}}"
                        f"<br>Cornering idx: %{{y:.3f}}<extra></extra>"
                    ),
                ))
            fig.add_hline(y=1.0, line=dict(color=GRID_EMPH, dash='dash'))
            fig.add_vline(x=1.0, line=dict(color=GRID_EMPH, dash='dash'))
            fig.update_layout(**_fig_layout(height=chart_height('feature'),
                xaxis_title=dict(text="Straight-line index", font=LABEL_FONT),
                yaxis_title=dict(text="Cornering index", font=LABEL_FONT),
                showlegend=False,
            ))
            show_plotly_chart(fig)
            caption(
                "A car near the bottom right is quick in a straight line and "
                "less quick through the corners; the top left is the other way "
                "round. The numbers are relative to the best car in this "
                "session (estimated)."
            )

    # Upgrade detector: diff against another race.
    ui.section("How this race compares with another",
            "estimated — per-driver metric difference",
            hint="The difference in each number between two races. A change "
                 "that every driver of one team shares can point at a setup "
                 "change; a small difference is just noise.")

    estimate_note(
        "Per-driver metric difference between two races (positive = higher in "
        "the selected race). A consistent shift across drivers of one team can "
        "indicate a setup change or an upgrade; small shifts are noise, and "
        "the character of the circuit moves every team together."
    )

    race_names = get_race_names(year)
    all_races = list(race_names.keys()) if race_names else []
    if len(all_races) < 2:
        empty("Need at least two races to compute a shift")
        return

    try:
        completed = get_completed_races(year)
        prior = [r for r in all_races if r in completed and r != selected_race]
        default_race = prior[-1] if prior else all_races[0]
    except Exception:
        default_race = all_races[0]

    compare_race = st.selectbox("Compare against", [r for r in all_races if r != selected_race],
                                index=max(0, all_races.index(default_race) - 1
                                          if default_race in all_races else 0),
                                key="aero_shift_race")

    shift = _cached_shift(year, compare_race, selected_race, session_type)
    if shift.empty:
        empty("No comparable drivers between the two sessions")
        return

    rows = []
    for _, r in shift.iterrows():
        rows.append([
            str(r.get('Driver', '')),
            _num(r.get('TopSpeed_delta'), 1),
            _num(r.get('SlowApex_delta'), 1),
            _num(r.get('MediumApex_delta'), 1),
            _num(r.get('FastApex_delta'), 1),
            _num(r.get('BrakeDecelG_delta'), 2),
            _num(r.get('AbsShift'), 2),
        ])
    data_table(
        ['Driver', 'Top speed difference (km/h)',
         'Slow corner difference (km/h)', 'Medium corner difference (km/h)',
         'Fast corner difference (km/h)', 'Braking difference (g)',
         'Total size of the shift'],
        rows,
    )
    ui.verdict(_shift_verdict(shift))


def _clock(value):
    """A timestamp from the live feeds, written as a plain clock time."""
    if value is None or pd.isna(value):
        return '-'
    try:
        return value.strftime('%H:%M:%S') + ' UTC'
    except (AttributeError, ValueError):
        return '-'


@st.cache_resource(ttl=STREAMLIT_CONFIG.session_cache_ttl,
                  show_spinner=False)
def _cached_telemetry_session(year, race, session_type='Race'):
    """Session loaded **with** car telemetry (cached).

    Telemetry is the largest download in the app, so the tab only
    reaches this function after the reader presses its load button.
    Returns ``None`` when the session cannot be loaded.
    """
    setup_fastf1_cache()
    return load_fastf1_session(year, race, session_type,
                               load_telemetry=True)


def _car_data_for_lap(session, laps, driver, lap_number=None):
    """Car telemetry frame for one driver's lap (or ``None``).

    ``lap_number=None`` picks the driver's fastest timed lap. The
    frame carries ``Distance`` (added when FastF1 did not provide it)
    so traces can be aligned by circuit distance downstream.
    """
    if session is None or laps is None or laps.empty:
        return None
    sub = laps[(laps['Driver'] == driver) & laps['LapTime'].notna()]
    if lap_number is not None and 'LapNumber' in sub.columns:
        sub = sub[sub['LapNumber'] == lap_number]
    if sub.empty:
        return None
    lap = sub.loc[sub['LapTime'].idxmin()]
    try:
        car = lap.get_car_data()
    except Exception:
        return None
    if car is None or car.empty:
        return None
    if 'Distance' not in car.columns:
        try:
            car = car.add_distance()
        except Exception:
            return None
    return car


@st.fragment
def _tab_telemetry(context, year, selected_race, session_type='Race'):
    """Telemetry comparison: drivers aligned by circuit distance."""
    laps = context.laps
    note(
        "This tab answers 'how did they drive the lap?'. It compares "
        "the car's own telemetry - speed, pedals, gear, revs and DRS - "
        "of two or more drivers on the same lap, aligned by distance "
        "around the circuit."
    )
    ui.section("Telemetry comparison",
            "measured — car telemetry, opt-in download",
            hint="One chart per variable; every driver is resampled "
                 "onto the same distance grid, so the lines line up "
                 "corner by corner.")

    ready = st.session_state.get('telemetry_loaded', False)
    if not ready:
        note(
            "Car telemetry is the largest download in the app, so it is "
            "never loaded automatically: press the button and the "
            "session is cached for the rest of the visit.",
            kind='info')
        if st.button('Load the telemetry', key='load_telemetry',
                     type='primary'):
            st.session_state.telemetry_loaded = True
            st.rerun()
        return

    with st.spinner('Loading the car telemetry (the first load takes a '
                    'minute)...'):
        session = _cached_telemetry_session(year, selected_race,
                                            session_type)
    if session is None or not hasattr(session, 'car_data'):
        empty("Car telemetry not available for this session.",
              "The OpenF1 fallback feed does not publish car telemetry, "
              "so this comparison needs the official timing archive.")
        return

    drivers = sorted(laps['Driver'].dropna().unique())
    selected = st.multiselect(
        "Drivers", drivers,
        default=_quickest_drivers(laps, DEFAULT_DRIVERS),
        max_selections=6, key="tel_drv")
    if not selected:
        empty("Pick at least one driver.",
              "The telemetry chart needs one or more drivers to compare.")
        return

    max_lap = int(pd.to_numeric(laps.get('LapNumber'), errors='coerce')
                  .max() or 0)
    lap_choice = st.selectbox(
        "Lap", ["Fastest lap"] + list(range(1, max_lap + 1)),
        key="tel_lap")
    lap_number = None if lap_choice == "Fastest lap" else int(lap_choice)

    frames = {}
    for driver in selected:
        car = _car_data_for_lap(session, laps, driver, lap_number)
        if car is not None:
            frames[driver] = car
    if not frames:
        empty("No car telemetry for the selected drivers and lap.",
              "Try another lap - some laps have no telemetry samples.")
        return
    missing = [d for d in selected if d not in frames]
    if missing:
        note(f"No telemetry for: {', '.join(missing)}.", kind='warn')

    aligned = ra.align_telemetry(frames)
    variable = st.radio(
        "Variable", ["Speed", "Throttle", "Brake", "Gear", "RPM", "DRS"],
        horizontal=True, key="tel_var")
    units = {"Speed": "km/h", "Throttle": "%", "Brake": "%",
             "Gear": "gear", "RPM": "rpm", "DRS": "state"}
    fig = go.Figure()
    for index, driver in enumerate(frames):
        trace = aligned[driver]
        fig.add_trace(go.Scatter(
            x=trace['Distance'], y=trace[variable], name=driver,
            mode='lines',
            line=dict(color=PLOT_PALETTE[index % len(PLOT_PALETTE)],
                      width=2),
            hovertemplate=(f"<b>{driver}</b><br>%{{x:.0f}} m<br>"
                           f"%{{y:.1f}} {units[variable]}<extra></extra>"),
        ))
    fig.update_layout(**_fig_layout(
        height=chart_height('feature'),
        xaxis_title=dict(text="Distance around the lap (m)",
                         font=LABEL_FONT),
        yaxis_title=dict(text=f"{variable} ({units[variable]})",
                         font=LABEL_FONT),
    ))
    show_plotly_chart(fig)
    caption("Aligned by circuit distance, not by time: each line is the "
            "same metres of track.")

    ui.section("Telemetry summary",
            "measured — the aligned lap",
            hint="Top and average speed plus how many braking points "
                 "the lap held, per driver.")
    rows = []
    for driver, trace in aligned.items():
        summary = ra.telemetry_summary(trace)
        drs_open = '-'
        try:
            drs = trace.get('DRS')
            if drs is not None and len(drs):
                import numpy as _np
                drs_open = f"{float(_np.mean(_np.asarray(drs) >= 10)) * 100:.0f}%"
        except Exception:
            drs_open = '-'
        rows.append([driver, _num(summary.get('max_speed'), 1),
                     _num(summary.get('avg_speed'), 1),
                     str(summary.get('braking_points', '-')),
                     drs_open])
    data_table(['Driver', 'Max speed (km/h)', 'Average speed (km/h)',
                'Braking points', 'DRS open'], rows)
    caption("Braking points = where the brake goes off→on. DRS open = share "
            "of samples with DRS state ≥10 (10–12 = open in FastF1). "
            "No circuit map is drawn: FastF1 publishes no track outline "
            "in the lap table, so it is not invented.")
    with st.expander('How to read this chart'):
        st.write(
            "Braking points are counted where the brake goes from off "
            "to on. DRS reads 0-12 (0 closed, 10-12 open) in FastF1 "
            "telemetry. A gap in a line means that driver has no "
            "telemetry samples there - it is not interpolated.")


@st.fragment
def _tab_feed(context):
    """The live feeds: the gaps between the cars, the radio, the stewards."""
    session = context.session
    session_key = getattr(session, 'session_key', None)

    ui.section(
        "The gap radar and the radio",
        "OpenF1",
        hint="How far behind the leader each car was while the session ran, "
             "the radio messages that were published and the notes the "
             "stewards sent. OpenF1 only publishes these while a session is "
             "running or shortly after it, so a race from an earlier weekend "
             "can have none of them.",
    )

    status = live_feed.feed_status(session_key)
    if not status['available']:
        empty(status['reason'])
        return

    # -- the gap radar --
    pivot = live_feed.gap_series(session_key)
    if not pivot.empty:
        summary = live_feed.gap_summary(session_key)
        fig = go.Figure()
        drivers = [column for column in pivot.columns if column != 'Date']
        for index, driver in enumerate(drivers):
            fig.add_trace(go.Scatter(
                x=pivot['Date'],
                y=pivot[driver],
                mode='lines',
                name=driver,
                line=dict(width=2,
                          color=PLOT_PALETTE[index % len(PLOT_PALETTE)]),
            ))
        fig.update_layout(**_fig_layout(
            height=chart_height('feature'),
            hovermode='x unified',
            xaxis=dict(),
            yaxis=dict(),
            xaxis_title=dict(text="Time in the session", font=LABEL_FONT),
            yaxis_title=dict(text="Gap to the leader (s)", font=LABEL_FONT),
        ))
        show_plotly_chart(fig, key='analysis_gap_radar')
        caption(
            f"{summary['rows']:,} gap readings for {summary['drivers']} cars "
            f"over {summary['minutes']:.0f} minutes of the session; the last "
            f"reading is from {_clock(summary['last_update'])} and the car "
            f"ahead was {summary['leader']}. Only the eight closest cars are "
            f"drawn, so the lines stay apart."
        )

    # -- where the field was at the last reading --
    latest = live_feed.latest_gaps(session_key)
    if not latest.empty:
        ui.section(
            "Where the field was at the last reading",
            hint="The gap to the leader when the feed was last updated.",
        )
        rows = []
        for position, (_, row) in enumerate(latest.head(10).iterrows(), start=1):
            gap = row.get('GapToLeader')
            if gap is None or pd.isna(gap):
                gap_text = 'leader'
            elif float(gap) <= 0:
                gap_text = 'leader'
            else:
                gap_text = f"+{float(gap):.1f} s"
            rows.append([f"P{position}", str(row.get('Driver', '')), gap_text])
        data_table(['Position', 'Driver', 'Gap to the leader'], rows)

    # -- team radio --
    radio = live_feed.team_radio(session_key)
    if not radio.empty:
        ui.section(
            "What the teams said on the radio",
            hint="The radio messages OpenF1 published, newest first.",
        )
        rows = []
        for _, row in radio.tail(10).iloc[::-1].iterrows():
            rows.append([
                _clock(row.get('Date')),
                str(row.get('Driver', '')),
                'published' if row.get('RecordingUrl') else 'not published',
            ])
        data_table(['Time', 'Driver', 'Recording'], rows)
        newest = radio.iloc[-1]
        if newest.get('RecordingUrl'):
            st.audio(str(newest['RecordingUrl']))
            caption(
                f"The newest clip: {newest.get('Driver', 'a driver')} at "
                f"{_clock(newest.get('Date'))}."
            )

    # -- race control --
    control = live_feed.race_control(session_key)
    if not control.empty:
        ui.section(
            "What the stewards said",
            hint="Flags, investigations and penalties, newest first.",
        )
        rows = []
        for _, row in control.tail(12).iloc[::-1].iterrows():
            lap = row.get('Lap')
            lap_text = str(int(lap)) if pd.notna(lap) else '-'
            rows.append([
                _clock(row.get('Date')),
                lap_text,
                str(row.get('Message', '')),
            ])
        data_table(['Time', 'Lap', 'Message'], rows)

    note(
        "The live feeds are the gaps, the radio and the stewards' messages as "
        "OpenF1 published them. They are not the official timing loops of the "
        "circuit, and a message is only here if the feed carried it.",
        kind='info',
    )


def page():
    """The page: the analysis, then the reading guide."""
    _body()

    ui.steps([
        "Pick the season, Grand Prix and session at the top. The eight tabs "
        "are the analytical views: Position Evolution (with Gap to Leader), "
        "Driver Pace, Sector Analysis, Telemetry, Race Context, Tyres & "
        "Stops, Setup and Gaps & Radio.",
        "A clean lap is a lap that can be compared with another lap: the car "
        "was on track, the track was green and the timing was reliable. "
        "Everything that is estimated uses only clean laps.",
        "Anything marked 'estimated' is worked out from the data, not "
        "measured directly. The chip next to it says how much data the "
        "estimate rests on: good, medium, low or not enough data.",
        "The tables hold the same numbers as the charts above them, so you "
        "can read the exact value behind any point of a chart.",
        "The strategy projection is a modelling result under stated "
        "assumptions. It is not a recommendation, and it knows nothing about "
        "traffic, weather or the race situation.",
    ], title="How to read this page")

    ui.glossary(['pace', 'stint', 'degradation', 'sector', 'undercut',
              'pit window', 'delta', 'estimated', 'confidence',
              'setup fingerprint'], title="What the words mean")


page()
