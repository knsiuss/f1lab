# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shared import (
    load_race_data, load_fastf1_session, setup_fastf1_cache,
    show_plotly_chart, format_f1_time
)
from config import TEAM_COLORS, FASTF1_CONFIG, MODEL_CONFIG, SESSION_GUARD_HOURS, STREAMLIT_CONFIG
from season_config import get_race_names, get_event_schedule_cached, get_completed_races
from fastf1_extended import (
    get_tyre_stints, get_pit_stops, get_race_control_messages
)
from model import RaceStrategySimulator
from pace import normalised_pace_by_stint, long_run_quality
from whatif import Scenario, compare_scenarios, undercut_delta
from sector import sector_summary, sector_deficits
from compare import compare_sessions
from setupfingerprint import fingerprint, normalise_fingerprint, fingerprint_shift
from pitwindow import pit_window
from datetime import timedelta


from styles import (
    COMPOUND_COLORS, FONT, GRID, GRID_EMPH, LABEL_FONT, TEXT_PRIMARY,
    TEXT_SECONDARY, TITLE_FONT, ZERO_LINE, fig_layout as _fig_layout,
    time_axis_layout as _time_axis_layout,
)

DEFAULT_DRIVERS = 3  # Show fewer drivers by default for clarity

@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl, show_spinner=False)
def _cached_fingerprint(year, race):
    """Session fingerprint, cached: telemetry extraction is too heavy per rerun."""
    setup_fastf1_cache()
    s = load_fastf1_session(year, race, 'Race')
    if s is None or s.laps is None or s.laps.empty:
        return pd.DataFrame()
    return fingerprint(s.laps, s)


def _tab_setup_aero(year, selected_race):
    st.subheader("Setup Fingerprint (estimated)")
    st.caption(
        "First-principles setup proxy from public telemetry only. Top speed "
        "is the median of the best clean-lap trap readings; cornering metrics "
        "are mean apex speeds in slow/medium/fast buckets detected on each "
        "driver's fastest clean lap, plus mean peak braking g. Everything is "
        "estimated; confidence follows the clean-lap sample. Trap speeds "
        "include slipstream and DRS effects that cannot be fully removed "
        "from public data."
    )

    fp = _cached_fingerprint(year, selected_race)
    if fp.empty:
        st.info("Not enough clean-lap data to build a fingerprint for this session.")
        return

    norm = normalise_fingerprint(fp)

    shown = fp[['Driver', 'Team', 'TopSpeed', 'SlowApex', 'MediumApex',
                'FastApex', 'BrakeDecelG', 'CornersDetected', 'Confidence']].copy()
    shown.columns = ['Driver', 'Team', 'Top speed (km/h)', 'Slow apex (km/h)',
                     'Medium apex (km/h)', 'Fast apex (km/h)', 'Braking (g)',
                     'Corners', 'Confidence']
    st.dataframe(shown, use_container_width=True, hide_index=True)

    # Setup trade-off map: straight-line index vs cornering index.
    idx_cols = ['TopSpeedIdx', 'SlowApexIdx', 'MediumApexIdx', 'FastApexIdx']
    if all(c in norm.columns for c in idx_cols):
        plot_df = norm.copy()
        corner_idx = plot_df[idx_cols[1:]].mean(axis=1, skipna=True)
        plot_df['CorneringIdx'] = corner_idx.round(3)
        plot_df = plot_df.dropna(subset=['TopSpeedIdx', 'CorneringIdx'])
        if not plot_df.empty:
            fig = go.Figure()
            for _, r in plot_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[r['TopSpeedIdx']], y=[r['CorneringIdx']], mode='markers+text',
                    text=[r['Driver']], textposition='top center',
                    textfont=dict(size=9, color=TEXT_SECONDARY),
                    marker=dict(size=11,
                                color=TEAM_COLORS.get(r.get('Team'), '#888'),
                                line=dict(width=1, color='rgba(255,255,255,0.4)')),
                    name=r['Driver'],
                    hovertemplate=(
                        f"<b>{r['Driver']}</b><br>Top speed idx: %{{x:.3f}}"
                        f"<br>Cornering idx: %{{y:.3f}}<extra></extra>"
                    ),
                ))
            fig.add_hline(y=1.0, line=dict(color=GRID_EMPH, dash='dash'))
            fig.add_vline(x=1.0, line=dict(color=GRID_EMPH, dash='dash'))
            fig.update_layout(**_fig_layout(height=460,
                title=dict(text="Setup trade-off map (session best = 1.000)",
                           font=TITLE_FONT, x=0.01),
                xaxis=dict(title=dict(text="Straight-line index", font=LABEL_FONT),
                           gridcolor=GRID),
                yaxis=dict(title=dict(text="Cornering index", font=LABEL_FONT),
                           gridcolor=GRID),
                showlegend=False,
            ))
            show_plotly_chart(fig)
            st.caption(
                "Bottom-right leans low-drag / top-speed; top-left leans "
                "high-downforce / cornering. Indices are relative to this "
                "session's best (estimated)."
            )

    # Upgrade detector: diff against another race.
    st.markdown("---")
    st.subheader("Shift vs Another Race")
    st.caption(
        "Per-driver metric difference between two races (positive = higher "
        "in the selected race). A consistent shift across drivers of one team "
        "can indicate a setup change or upgrade; small shifts are noise, and "
        "track character moves every team together."
    )

    race_names = get_race_names(year)
    all_races = list(race_names.keys()) if race_names else []
    if len(all_races) < 2:
        st.info("Need at least two races to compute a shift.")
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

    shift = _cached_shift(year, compare_race, selected_race)
    if shift.empty:
        st.info("No comparable drivers between the two sessions.")
        return

    disp = shift.rename(columns={
        'TopSpeed_delta': 'Top speed delta',
        'SlowApex_delta': 'Slow apex delta',
        'MediumApex_delta': 'Medium apex delta',
        'FastApex_delta': 'Fast apex delta',
        'BrakeDecelG_delta': 'Braking delta (g)',
        'AbsShift': 'Total |shift|',
    })
    keep = [c for c in ['Driver', 'Top speed delta', 'Slow apex delta',
                        'Medium apex delta', 'Fast apex delta',
                        'Braking delta (g)', 'Total |shift|'] if c in disp.columns]
    st.dataframe(disp[keep], use_container_width=True, hide_index=True)


@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl, show_spinner=False)
def _cached_shift(year, race_a, race_b):
    """Fingerprint diff between two cached session fingerprints."""
    fa = _cached_fingerprint(year, race_a)
    fb = _cached_fingerprint(year, race_b)
    return fingerprint_shift(fa, fb)


def page():
    year = st.session_state.get('selected_year', FASTF1_CONFIG.default_year)
    df = load_race_data(year)

    if df is None or df.empty:
        st.error("No data available")
        return

    # -- Header --
    st.markdown(f"""
    <div style="text-align:center;padding:1.2rem 0 0.8rem 0;">
        <h1 style="font-size:2.2rem;font-weight:800;color:#E10600;margin:0;
                   text-transform:uppercase;letter-spacing:3px;">
            Race Analysis
        </h1>
        <p style="font-size:0.9rem;color:#666;margin-top:0.3rem;letter-spacing:0.5px;">
            {year} Season &mdash; Pace &middot; Strategy &middot; Battles &middot; Metrics
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -- Race selector --
    race_names = get_race_names(year)
    all_races = list(race_names.keys()) if race_names else []
    if not all_races:
        st.warning("No races found.")
        return

    c1, c2 = st.columns([3, 1])
    with c1:
        selected_race = st.selectbox("Grand Prix", all_races, key="analysis_race",
                                     label_visibility="collapsed")
    with c2:
        session_type = st.selectbox("Session", ["Race", "Qualifying", "Sprint"],
                                    key="analysis_session")

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
                    st.info(f"{selected_race} has not started yet.")
                    return
    except Exception:
        pass

    with st.spinner("Loading session data..."):
        setup_fastf1_cache()
        session = load_fastf1_session(year, selected_race, session_type)

    if session is None:
        st.warning(f"Data not available for {selected_race} ({session_type}).")
        return

    laps = session.laps
    if laps is None or laps.empty:
        st.warning("No lap data for this session.")
        return

    # -- KPI row --
    _render_kpi_row(laps, session)

    # -- Tabs --
    tabs = st.tabs(["Pace & Laps", "Strategy & Stints", "Race Battles",
                    "Performance Insights", "Sector & Sessions", "Setup & Aero"])

    with tabs[0]:
        _tab_pace(laps, session, selected_race)

    with tabs[1]:
        _tab_strategy(laps, session, selected_race)

    with tabs[2]:
        _tab_battles(laps, session)

    with tabs[3]:
        _tab_insights(laps, session)

    with tabs[4]:
        _tab_sector_compare(laps, selected_race, year, session_type)

    with tabs[5]:
        _tab_setup_aero(year, selected_race)


# KPI ROW
def _render_kpi_row(laps, session):
    col_a, col_b, col_c, col_d, col_e = st.columns(5)

    total_laps = int(laps['LapNumber'].nunique())

    fastest_row = laps.loc[laps['LapTime'].idxmin()]
    fastest_drv = fastest_row.get('Driver', 'N/A')
    fastest_time = format_f1_time(fastest_row['LapTime'])

    if 'Position' in laps.columns:
        pos = pd.to_numeric(laps['Position'], errors='coerce')
        leader_laps = laps[pos == 1]
        if not leader_laps.empty:
            leader_drv = leader_laps['Driver'].mode().iloc[0]
        else:
            leader_drv = '-'
    else:
        leader_drv = '-'

    if 'Position' in laps.columns:
        pos = pd.to_numeric(laps['Position'], errors='coerce')
        leader_series = laps[pos == 1]['Driver']
        lead_changes = max(0, int((leader_series != leader_series.shift()).sum()) - 1)
    else:
        lead_changes = 0

    pit_stops = get_pit_stops(session)
    pit_count = len(pit_stops) if pit_stops is not None and not pit_stops.empty else 0

    kpis = [
        ("Total Laps", str(total_laps), None),
        ("Fastest Lap", fastest_time, fastest_drv),
        ("Race Leader", leader_drv, None),
        ("Lead Changes", str(lead_changes), None),
        ("Pit Stops", str(pit_count), None),
    ]

    for col, (label, value, sub) in zip([col_a, col_b, col_c, col_d, col_e], kpis):
        with col:
            sub_html = f'<div style="font-size:0.72rem;color:#777;margin-top:2px;letter-spacing:0.3px;">{sub}</div>' if sub else ''
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
                {sub_html}
            </div>
            """, unsafe_allow_html=True)


# TAB 1: PACE & LAPS
def _tab_pace(laps, session, race_name):
    st.subheader("Lap Time Evolution")

    drivers_list = sorted(laps['Driver'].unique())
    selected = st.multiselect("Drivers", drivers_list,
                              default=drivers_list[:DEFAULT_DRIVERS],
                              max_selections=8, key="pace_drv")

    if not selected:
        return


    # Main lap-time chart with compound colors and pit markers
    fig = go.Figure()
    all_times = []
    pit_laps_map = {}

    try:
        from fastf1_extended import get_pit_stops
        pit_stops = get_pit_stops(session)
        if pit_stops is not None and not pit_stops.empty:
            for _, p in pit_stops.iterrows():
                drv = p.get('Driver', '')
                lap = int(p.get('LapNumber', 0))
                if drv not in pit_laps_map:
                    pit_laps_map[drv] = []
                pit_laps_map[drv].append(lap)
    except Exception:
        pass

    overall_fastest = laps.loc[laps['LapTime'].idxmin()] if not laps.empty else None

    for driver in selected:
        drv = laps[(laps['Driver'] == driver) & laps['LapTime'].notna()].sort_values('LapNumber').copy()
        if drv.empty:
            continue
        team = drv['Team'].iloc[0] if 'Team' in drv.columns else ''
        color = TEAM_COLORS.get(team, '#888')
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
            textfont=dict(size=10, color='white', family='Inter, sans-serif'),
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
        fl_drv = overall_fastest.get('Driver', '')
        fl_lap = overall_fastest.get('LapNumber', 0)
        fl_time = format_f1_time(overall_fastest['LapTime'])
        fig.add_vline(x=fl_lap, line_dash="dash", line_color="#FFD700",
                      line_width=1.5, opacity=0.5,
                      annotation_text="FL: {} {} ({})".format(fl_drv, int(fl_lap), fl_time),
                      annotation_font=dict(size=11, color='#FFD700', family='Inter, sans-serif'))

    if all_times:
        y_min, y_max = min(all_times) - 1, max(all_times) + 1
        yaxis = _time_axis_layout("Lap Time", (y_min, y_max))
    else:
        yaxis = dict(title=dict(text="Lap Time", font=LABEL_FONT))

    fig.update_layout(**_fig_layout(height=500,
        title=dict(text="Lap Time Evolution -- {}".format(race_name), font=TITLE_FONT, x=0.01),
        xaxis=dict(title=dict(text="Lap", font=LABEL_FONT), gridcolor=GRID,
                   tickfont=dict(size=11, color=TEXT_SECONDARY)),
        yaxis=yaxis,
        hovermode='x unified',
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
    ))
    show_plotly_chart(fig)


    # -- Fastest laps table --
    st.markdown("#### Fastest Laps")
    fastest = laps.groupby('Driver').apply(
        lambda g: g.nsmallest(1, 'LapTime')).reset_index(drop=True)
    if not fastest.empty:
        cols_show = ['Driver', 'LapNumber', 'LapTime', 'Compound']
        cols_show = [c for c in cols_show if c in fastest.columns]
        table = fastest[cols_show].sort_values('LapTime')
        table['LapTime'] = table['LapTime'].apply(format_f1_time)
        table = table.rename(columns={'LapNumber': 'Lap', 'Compound': 'Tyre'})
        st.dataframe(table, use_container_width=True, hide_index=True)

    # -- Head-to-head pace --
    st.markdown("---")
    st.subheader("Head-to-Head Pace")

    c1, c2 = st.columns(2)
    with c1:
        drv_a = st.selectbox("Driver A", drivers_list, key="pace_h2h_a")
    with c2:
        drv_b = st.selectbox("Driver B", [d for d in drivers_list if d != drv_a],
                             key="pace_h2h_b")

    la = laps[(laps['Driver'] == drv_a) & laps['LapTime'].notna()].sort_values('LapNumber')
    lb = laps[(laps['Driver'] == drv_b) & laps['LapTime'].notna()].sort_values('LapNumber')

    if la.empty or lb.empty:
        st.info("Not enough lap data for both drivers to compare.")
        return

    ta = la['LapTime'].dt.total_seconds()
    tb = lb['LapTime'].dt.total_seconds()
    team_a = la['Team'].iloc[0] if 'Team' in la.columns else ''
    team_b = lb['Team'].iloc[0] if 'Team' in lb.columns else ''
    diff = ta.mean() - tb.mean()

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric(f"{drv_a} Avg", format_f1_time(pd.Timedelta(seconds=ta.mean())))
    with mc2:
        st.metric(f"{drv_b} Avg", format_f1_time(pd.Timedelta(seconds=tb.mean())))
    with mc3:
        faster = drv_a if diff < 0 else drv_b
        st.metric("Faster", faster, f"{abs(diff):.3f}s")

    color_a = TEAM_COLORS.get(team_a, '#E10600')
    color_b = TEAM_COLORS.get(team_b, '#00D2BE')

    fig_battle = make_subplots(rows=2, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.08,
                                subplot_titles=("Lap Time Comparison", "Cumulative Advantage"))

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
    la_s = la[['LapNumber']].copy()
    la_s['ta'] = la['LapTime'].dt.total_seconds()
    lb_s = lb[['LapNumber']].copy()
    lb_s['tb'] = lb['LapTime'].dt.total_seconds()
    merged = pd.merge(la_s, lb_s, on='LapNumber', how='inner')
    if not merged.empty:
        merged['cum_adv'] = (merged['ta'] - merged['tb']).cumsum()
        adv_drv = drv_a if diff < 0 else drv_b
        fig_battle.add_trace(go.Scatter(
            x=merged['LapNumber'], y=merged['cum_adv'],
            mode='lines+markers', name="{} adv".format(adv_drv),
            line=dict(color='#FFD700', width=2.5),
            fill='tozeroy', fillcolor='rgba(255,215,0,0.1)',
            hovertemplate="Lap %{x}<br>Adv: %{y:.3f}s<extra></extra>",
        ), row=2, col=1)
    fig_battle.add_hline(y=0, line_dash="dot", line_color='rgba(255,255,255,0.3)', row=2, col=1)

    fig_battle.update_layout(**_fig_layout(height=600,
        title=dict(text="{} vs {} - Battle".format(drv_a, drv_b), font=TITLE_FONT, x=0.01),
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", y=-0.02, font=dict(size=11)),
    ))
    fig_battle.update_xaxes(title_text="Lap", row=2, col=1, gridcolor=GRID)
    fig_battle.update_xaxes(gridcolor=GRID, row=1, col=1)
    y_min2 = min(ta.min(), tb.min()) * 0.995
    y_max2 = max(ta.max(), tb.max()) * 1.005
    fig_battle.update_yaxes(title_text="Lap Time", row=1, col=1,
                            **_time_axis_layout("", (y_min2, y_max2)))
    fig_battle.update_yaxes(title_text="Advantage (s)", row=2, col=1, gridcolor=GRID,
                            tickfont=dict(size=11, color=TEXT_SECONDARY))
    show_plotly_chart(fig_battle)

    # Normalised pace & data quality (Data Honesty: estimated + sample size)
    st.markdown("---")
    st.subheader("Normalised Pace & Data Quality (estimated)")
    st.caption(
        "Pace is normalised for tyre age (linear fit of lap time vs tyre life). "
        "Every figure is flagged estimated with a confidence level derived from "
        "the number of clean laps behind it; short samples are marked low rather "
        "than presented as fact."
    )
    npace = normalised_pace_by_stint(laps)
    if npace.empty:
        st.info("Not enough clean lap data to normalise pace.")
    else:
        ncols = [c for c in ['Driver', 'Stint', 'Compound', 'N', 'MeanPace',
                             'BasePace', 'DegPerLap', 'Level', 'Confidence']
                 if c in npace.columns]
        nshow = npace[ncols].copy()
        nshow = nshow.rename(columns={
            'N': 'Laps', 'MeanPace': 'Mean pace (s)',
            'BasePace': 'Base pace (s, est)', 'DegPerLap': 'Deg/lap (s, est)',
            'Level': 'Quality', 'Confidence': 'Confidence',
        })
        st.dataframe(nshow, use_container_width=True, hide_index=True)

    lr = long_run_quality(laps)
    if not lr.empty:
        st.markdown("##### Representative Long Runs")
        lcols = [c for c in ['Driver', 'Stint', 'Compound', 'Length', 'MeanPace',
                             'DegPerLap', 'Level', 'Confidence'] if c in lr.columns]
        lshow = lr[lcols].copy()
        lshow = lshow.rename(columns={
            'Length': 'Laps', 'MeanPace': 'Mean pace (s)',
            'DegPerLap': 'Deg/lap (s, est)', 'Level': 'Quality',
            'Confidence': 'Confidence',
        })
        st.dataframe(lshow, use_container_width=True, hide_index=True)


# TAB 2: STRATEGY & STINTS
def _tab_strategy(laps, session, race_name):
    st.subheader("Tyre Strategy Map")

    tyre_data = get_tyre_stints(session)

    if tyre_data is not None and not tyre_data.empty:
        try:
            results = session.results
            driver_order = results.sort_values('Position')['Abbreviation'].tolist() if results is not None else tyre_data['Driver'].unique().tolist()
        except Exception:
            driver_order = tyre_data['Driver'].unique().tolist()

        drivers = [d for d in driver_order if d in tyre_data['Driver'].values][:20]
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
                        size=10, family="Inter"
                    ),
                    hovertemplate=(
                        f"<b>{driver}</b><br>"
                        f"Compound: {compound}<br>"
                        f"Lap {start}-{end}<br>"
                        f"Stint: {width} laps"
                        "<extra></extra>"
                    ),
                ))

        total_laps = int(tyre_data['EndLap'].max()) if 'EndLap' in tyre_data.columns else 50
        fig.update_layout(**_fig_layout(height=max(480, len(drivers) * 28),
            title=dict(text=f"Tyre Strategy - {race_name}", font=TITLE_FONT, x=0.01),
            xaxis=dict(title=dict(text="Lap", font=LABEL_FONT), range=[0, total_laps + 2],
                       showgrid=False, tickfont=dict(size=11, color=TEXT_SECONDARY)),
            yaxis=dict(categoryorder='array', categoryarray=drivers[::-1], showgrid=True,
                       gridcolor=GRID, tickfont=dict(size=11, color=TEXT_SECONDARY)),
            barmode='overlay',
        ))
        show_plotly_chart(fig)

        # Legend strip
        legend_html = "&nbsp;&nbsp;".join([
            f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px;">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:3px;'
            f'background:{c};box-shadow:0 0 4px {c}44;"></span>'
            f'<span style="color:#aaa;font-size:0.82rem;font-weight:500;">{n}</span></span>'
            for n, c in COMPOUND_COLORS.items() if n in ['SOFT', 'MEDIUM', 'HARD']
        ])
        st.markdown(f'<div style="text-align:center;margin-top:-4px;margin-bottom:4px;">{legend_html}</div>',
                     unsafe_allow_html=True)
    else:
        st.info("Tyre strategy data not available")

    # -- Pit stops --
    st.markdown("---")
    st.subheader("Pit Stop Analysis")

    pit_stops = get_pit_stops(session)
    if pit_stops is not None and not pit_stops.empty:
        # Driver filter for pit stops
        pit_all_drivers = sorted(pit_stops['Driver'].unique())
        pit_drv_filter = st.multiselect(
            "Filter Drivers", pit_all_drivers,
            default=pit_all_drivers[:min(DEFAULT_DRIVERS, len(pit_all_drivers))],
            max_selections=10, key="pit_drv",
        )

        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1: st.metric("Total Stops", len(pit_stops))
        with pc2: st.metric("Avg Time", f"{pit_stops['PitTime'].mean():.1f}s")
        with pc3: st.metric("Fastest", f"{pit_stops['PitTime'].min():.1f}s")
        with pc4:
            popular = pit_stops['Lap'].mode().iloc[0] if not pit_stops['Lap'].mode().empty else '-'
            st.metric("Popular Lap", f"L{int(popular)}")

        n_measured = int(pit_stops['Measured'].sum()) if 'Measured' in pit_stops.columns else len(pit_stops)
        if n_measured < len(pit_stops):
            st.caption(
                f"{n_measured}/{len(pit_stops)} stop times measured from telemetry; "
                f"the rest use the model default ({MODEL_CONFIG['pit_loss_sec']:.0f}s, estimated)."
            )

        if pit_drv_filter:
            pit_filtered = pit_stops[pit_stops['Driver'].isin(pit_drv_filter)]
        else:
            pit_filtered = pit_stops

        fig_pit = go.Figure()
        fig_pit.add_trace(go.Scatter(
            x=pit_filtered['Lap'], y=pit_filtered['PitTime'],
            mode='markers',
            marker=dict(
                size=13, color=pit_filtered['PitTime'],
                colorscale=[[0, '#00D2BE'], [0.5, '#FFD700'], [1, '#E10600']],
                showscale=True, colorbar=dict(title="s", thickness=10, len=0.5,
                                              tickfont=dict(size=10, color='#888')),
                line=dict(width=1, color='rgba(255,255,255,0.3)'),
            ),
            text=pit_filtered['Driver'],
            hovertemplate="%{text}<br>Lap %{x}<br>%{y:.1f}s<extra></extra>",
        ))
        fig_pit.update_layout(**_fig_layout(height=400,
            title=dict(text="Pit Stop Distribution", font=TITLE_FONT, x=0.01),
            xaxis_title=dict(text="Lap", font=LABEL_FONT),
            yaxis_title=dict(text="Pit Time (s)", font=LABEL_FONT),
        ))
        show_plotly_chart(fig_pit)
    else:
        st.info("No pit stop data available")

    # -- Strategy simulator --
    st.markdown("---")
    st.subheader("Strategy Simulator")

    sc1, sc2 = st.columns(2)
    with sc1:
        base_lap = st.number_input("Base Lap Time (s)", 80.0, 120.0, 90.0, step=0.1,
                                   key="strat_base")
        total_laps_sim = st.number_input("Total Laps", 10, 80, 52, key="strat_tot")
    with sc2:
        driver_sim = st.text_input("Driver", "VER", key="strat_drv")
        start_tire = st.selectbox("Start Compound", ["SOFT", "MEDIUM", "HARD"],
                                  key="strat_tire")

    current_lap_sim = st.slider("Current Lap", 0, int(total_laps_sim), 0,
                                 key="strat_lap", label_visibility="collapsed")

    if st.button("Run Simulation", type="primary", key="run_sim"):
        sim = RaceStrategySimulator(base_lap, total_laps_sim)
        strategy = sim.predict_strategy(driver_sim, start_tire, current_lap_sim)

        s1, s2, s3 = st.columns(3)
        with s1: st.metric("1-Stop Total", f"{strategy['1_stop_time']:.1f}s")
        with s2: st.metric("2-Stop Total", f"{strategy['2_stop_time']:.1f}s")
        with s3: st.metric("Delta", f"{strategy['delta']:.1f}s")
        st.success(f"Recommended strategy: **{strategy['recommended']}**")

    st.markdown("---")
    st.subheader("Scenario Comparison (estimated)")
    st.caption(
        "Ranked race-time projection across candidate strategies. Times are "
        "estimates built on explicit assumptions (tyre degradation, fuel burn, "
        "pit loss); the fastest is only the fastest under those assumptions, "
        "not a recommendation."
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
    prov1, prov2, prov3 = st.columns(3)
    with prov1:
        traffic_pen = st.number_input("Traffic penalty (s/lap)", 0.0, 2.0, 0.0,
                                      step=0.1, key="strat_traffic")
    with prov2:
        rival_resp = st.number_input("Rival response (s, one-off)", 0.0, 30.0, 0.0,
                                     step=0.5, key="strat_rival")
    with prov3:
        st.caption("")

    cmp_df = compare_scenarios(
        scenarios, total_laps=int(total_laps_sim), base_lap_time=float(base_lap),
        traffic_penalty=float(traffic_pen), rival_response=float(rival_resp)
    )
    if not cmp_df.empty:
        shown = cmp_df[['Rank', 'Name', 'TotalTime', 'GapToBest', 'Stops',
                        'Risk', 'Confidence']].copy()
        shown.columns = ['Rank', 'Scenario', 'Time (s, est)', 'Gap (s, est)',
                         'Stops', 'Risk', 'Confidence']
        st.dataframe(shown, use_container_width=True, hide_index=True)

        und = undercut_delta(total_laps=int(total_laps_sim),
                             compound_new="HARD", compound_old=start_tire,
                             base_lap_time=float(base_lap), earlier_by=1,
                             rival_response=float(rival_resp))
        direction = "faster" if und['delta_sec'] < 0 else "slower"
        st.caption(
            f"Undercut estimate (incl. rival response): pitting 1 lap earlier "
            f"projects **{abs(und['delta_sec']):.1f}s {direction}** (estimated)"
        )
    else:
        st.info("Not enough laps to compare scenarios.")

    # -- Pit Window Radar --------------------------------------------------
    st.markdown("---")
    st.subheader("Pit Window Radar (estimated)")
    st.caption(
        "Per-driver optimal stop lap, grid-searched with the same strategy "
        "engine as above but fitted to each driver's own clean laps (base "
        "pace + degradation). The window is the range of stop laps within "
        "1s of that optimum. Confidence follows the clean-lap sample: good "
        "at 15+ laps, moderate at 8+, low at 2+."
    )

    pw_c1, pw_c2 = st.columns([1, 2])
    with pw_c1:
        n_stops_sel = st.radio("Stops modelled", [1, 2], index=0,
                               horizontal=True, key="strat_pitstops")
    with pw_c2:
        st.caption(
            "Drivers without a meaningful fitted degradation show a blank "
            "window rather than an invented early stop."
        )

    pw = pit_window(laps, n_stops=int(n_stops_sel), tolerance=1.0)
    if pw.empty:
        st.info("Not enough clean-lap data to compute a pit window for this session.")
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
        disp = pw_shown[['Driver', 'Optimal stop', 'Window', 'BasePace',
                         'DegPerLap', 'Level', 'Confidence']].copy()
        disp.columns = ['Driver', 'Optimal stop', 'Window', 'Base pace (s, est)',
                        'Deg (s/lap, est)', 'Quality', 'Confidence']
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.caption(
            "All figures estimated from this session's clean laps. The optimal "
            "lap is a modelling result under stated assumptions, not a team call."
        )


# TAB 3: RACE BATTLES
def _tab_battles(laps, session):
    st.subheader("Position Evolution")

    pos_laps = laps.copy()
    pos_laps['Position'] = pd.to_numeric(pos_laps['Position'], errors='coerce')

    if 'Position' not in pos_laps.columns:
        st.info("Position data not available")
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
        st.info("No position data")
        return

    pos_df = pd.DataFrame(pos_data)
    color_map = {}
    for d in pos_df['Driver'].unique():
        team = pos_df[pos_df['Driver'] == d]['Team'].iloc[0]
        color_map[d] = TEAM_COLORS.get(team, '#888')

    all_drivers = sorted(pos_df['Driver'].unique())
    pos_drivers = st.multiselect(
        "Drivers", all_drivers,
        default=all_drivers[:DEFAULT_DRIVERS],
        max_selections=10, key="pos_drv",
    )

    if not pos_drivers:
        return


    fig_pos = go.Figure()
    pit_map_pos = {}
    try:
        ps = get_pit_stops(session)
        if ps is not None and not ps.empty:
            for _, p in ps.iterrows():
                d = p.get('Driver', ''); lap = int(p.get('Lap', 0))
                if d not in pit_map_pos: pit_map_pos[d] = []
                pit_map_pos[d].append(lap)
    except Exception: pass

    for d in pos_drivers:
        dp = pos_df[pos_df['Driver'] == d].sort_values('Lap')
        color = color_map.get(d, '#888')
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

    fig_pos.update_layout(**_fig_layout(height=500,
        title=dict(text="Position Evolution", font=TITLE_FONT, x=0.01),
        xaxis=dict(title=dict(text="Lap", font=LABEL_FONT), gridcolor=GRID, tickfont=dict(size=11, color=TEXT_SECONDARY)),
        yaxis=dict(title=dict(text="Position", font=LABEL_FONT), autorange='reversed',
                   tickmode='linear', tick0=1, dtick=1, gridcolor=GRID, gridwidth=2,
                   zeroline=False, tickfont=dict(size=11, color=TEXT_PRIMARY)),
        hovermode='x unified', legend=dict(orientation="h", y=-0.18, font=dict(size=11))))
    show_plotly_chart(fig_pos)

    st.subheader("Gap to Leader")

    drivers_list = sorted(laps['Driver'].unique())
    # Default: skip first DEFAULT_DRIVERS (already shown in Position Evolution)
    # and show a different set
    gap_default = drivers_list[DEFAULT_DRIVERS:DEFAULT_DRIVERS + 4]
    gap_drivers = st.multiselect("Drivers", drivers_list,
                                 default=gap_default,
                                 max_selections=10, key="gap_drv")

    leader_data = None
    min_time = float('inf')
    for driver in gap_drivers:
        dlaps = laps[(laps['Driver'] == driver) & laps['LapTime'].notna()].sort_values('LapNumber')
        if not dlaps.empty:
            cum = dlaps['LapTime'].dt.total_seconds().cumsum()
            if cum.iloc[-1] < min_time:
                min_time = cum.iloc[-1]
                leader_data = (driver, dlaps, cum)

    if not gap_drivers:
        st.info("Select at least one driver.")
    elif leader_data is None:
        st.info("No valid lap times for the selected drivers.")
    else:
        leader_name, leader_laps, leader_cum = leader_data
        leader_dict = dict(zip(leader_laps['LapNumber'], leader_cum))

        fig_gap = go.Figure()
        for driver in gap_drivers:
            dlaps = laps[(laps['Driver'] == driver) & laps['LapTime'].notna()].sort_values('LapNumber').copy()
            if not dlaps.empty:
                team = dlaps['Team'].iloc[0] if 'Team' in dlaps.columns else ''
                team_color = TEAM_COLORS.get(team, '#888')
                compound_colors = [COMPOUND_COLORS.get(r.get('Compound', ''), team_color) for _, r in dlaps.iterrows()]
                cum = dlaps['LapTime'].dt.total_seconds().cumsum()
                gaps = [c - leader_dict.get(l, c) for l, c in zip(dlaps['LapNumber'], cum)]
                fig_gap.add_trace(go.Scatter(
                    x=dlaps['LapNumber'], y=gaps, mode='lines+markers', name=driver,
                    line=dict(color=team_color, width=2.5, shape='spline', smoothing=0.3),
                    marker=dict(size=6, color=compound_colors, opacity=0.8, line=dict(width=1, color=team_color)),
                    hovertemplate="<b>{}</b><br>Lap: %{{x}}<br>Gap: +%{{y:.2f}}s<extra></extra>".format(driver)))
        fig_gap.update_layout(**_fig_layout(height=460,
            title=dict(text="Gap to Leader ({})".format(leader_name), font=TITLE_FONT, x=0.01),
            xaxis=dict(title=dict(text="Lap", font=LABEL_FONT), gridcolor=GRID, tickfont=dict(size=11, color=TEXT_SECONDARY)),
            yaxis=dict(title=dict(text="Gap (s)", font=LABEL_FONT), zeroline=True,
                       zerolinecolor='rgba(255,255,255,0.3)', zerolinewidth=2,
                       gridcolor=GRID, tickfont=dict(size=11, color=TEXT_SECONDARY)),
            hovermode='x unified', legend=dict(orientation="h", y=-0.18, font=dict(size=11))))
        show_plotly_chart(fig_gap)

    st.subheader("Battle Analysis")

    c1, c2 = st.columns(2)
    with c1:
        battle_a = st.selectbox("Driver A", drivers_list, key="battle_a")
    with c2:
        battle_b = st.selectbox("Driver B", [d for d in drivers_list if d != battle_a],
                                key="battle_b")

    la = laps[(laps['Driver'] == battle_a) & laps['LapTime'].notna()].sort_values('LapNumber')
    lb = laps[(laps['Driver'] == battle_b) & laps['LapTime'].notna()].sort_values('LapNumber')

    if not la.empty and not lb.empty:
        team_a = la['Team'].iloc[0] if 'Team' in la.columns else ''
        team_b = lb['Team'].iloc[0] if 'Team' in lb.columns else ''
        color_a = TEAM_COLORS.get(team_a, '#E10600')
        color_b = TEAM_COLORS.get(team_b, '#00D2BE')

        common = set(la['LapNumber']) & set(lb['LapNumber'])
        faster_a = faster_b = 0
        diffs = []
        for lap_n in sorted(common):
            ta = la[la['LapNumber'] == lap_n]['LapTime'].dt.total_seconds().iloc[0]
            tb = lb[lb['LapNumber'] == lap_n]['LapTime'].dt.total_seconds().iloc[0]
            if ta < tb:
                faster_a += 1
            else:
                faster_b += 1
            diffs.append(ta - tb)

        avg_diff = np.mean(diffs) if diffs else 0
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            pct_a = faster_a / len(common) * 100 if common else 0
            st.markdown(f"""
            <div class="battle-card" style="border-color:{color_a};">
                <div style="font-size:1.8rem;font-weight:800;color:{color_a};">{faster_a}</div>
                <div style="color:#aaa;font-size:0.82rem;font-weight:500;">{battle_a} faster laps</div>
                <div style="font-size:0.72rem;color:#666;margin-top:4px;">{pct_a:.0f}% of common laps</div>
            </div>""", unsafe_allow_html=True)
        with bc2:
            st.markdown(f"""
            <div class="battle-card" style="border-color:#FFD700;">
                <div style="font-size:1.5rem;font-weight:800;color:#FFD700;">{len(common)}</div>
                <div style="color:#aaa;font-size:0.82rem;font-weight:500;">common laps</div>
                <div style="font-size:0.72rem;color:#666;margin-top:4px;">avg {abs(avg_diff):.3f}s gap</div>
            </div>""", unsafe_allow_html=True)
        with bc3:
            pct_b = faster_b / len(common) * 100 if common else 0
            st.markdown(f"""
            <div class="battle-card" style="border-color:{color_b};">
                <div style="font-size:1.8rem;font-weight:800;color:{color_b};">{faster_b}</div>
                <div style="color:#aaa;font-size:0.82rem;font-weight:500;">{battle_b} faster laps</div>
                <div style="font-size:0.72rem;color:#666;margin-top:4px;">{pct_b:.0f}% of common laps</div>
            </div>""", unsafe_allow_html=True)

        # Lap-by-lap delta bar chart
        fig_b = go.Figure()
        lap_nums = sorted(common)
        delta_vals = []
        for lap_n in lap_nums:
            ta = la[la['LapNumber'] == lap_n]['LapTime'].dt.total_seconds().iloc[0]
            tb = lb[lb['LapNumber'] == lap_n]['LapTime'].dt.total_seconds().iloc[0]
            delta_vals.append(ta - tb)

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
        fig_b.update_layout(**_fig_layout(height=400,
            title=dict(text=f"{battle_a} vs {battle_b} Lap Delta", font=TITLE_FONT, x=0.01),
            xaxis_title=dict(text="Lap", font=LABEL_FONT),
            yaxis_title=dict(text=f"Delta ({battle_a} - {battle_b}) in seconds", font=LABEL_FONT),
            yaxis=dict(zeroline=True, zerolinecolor=ZERO_LINE, zerolinewidth=2,
                       tickfont=dict(size=11, color=TEXT_SECONDARY)),
            bargap=0.15,
        ))
        show_plotly_chart(fig_b)


# TAB 4: PERFORMANCE INSIGHTS
def _tab_sector_compare(laps, race_name, year, session_type):
    """Sector / corner insights and session-to-session pace comparison."""
    _render_sector_insights(laps)
    st.markdown("---")
    _render_session_compare(laps, race_name, year, session_type)


def _render_sector_insights(laps):
    st.subheader("Sector Insights (estimated)")
    st.caption(
        "Where each driver's pace lives on the lap: per-sector best, average and "
        "consistency, plus the deficit to the session-best sector time. Figures are "
        "estimates computed from clean laps only (out/in and Safety Car laps removed)."
    )
    if not any(s in laps.columns for s in ("Sector1", "Sector2", "Sector3")):
        st.info("No sector timing available in this session's lap table.")
        return

    try:
        summary = sector_summary(laps)
        deficits = sector_deficits(laps)
    except Exception as e:
        st.warning(f"Sector analysis unavailable: {e}")
        return

    if summary is None or summary.empty:
        st.info("Not enough clean laps to estimate sector pace — analysis skipped.")
        return

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("##### Per-driver sector stats")
        st.dataframe(summary, use_container_width=True, hide_index=True)
    with c_right:
        st.markdown("##### Deficit to session-best sector (s)")
        st.dataframe(deficits, use_container_width=True, hide_index=True)

    st.caption(
        "Level reflects the number of clean laps behind each row (good ≥ 15, moderate ≥ 8, "
        "low ≥ 2, insufficient < 2). A row marked low/insufficient is a small sample — treat "
        "the estimate with caution."
    )


def _render_session_compare(laps, race_name, year, session_type):
    st.subheader("Session-to-Session Comparison (estimated)")
    st.caption(
        "Compares normalised base pace (fresh-tyre lap time) per driver/stint across two "
        "sessions. Base pace removes the effect of tyre age, so the delta is a fair pace "
        "read rather than raw lap time."
    )
    # Stint keys only align meaningfully between race-format sessions (Race ↔
    # Sprint). Qualifying laps have stint numbers that don't correspond to race
    # stints, so pairing them here would manufacture a misleading delta.
    partners = {"Race": ["Sprint"], "Sprint": ["Race"]}.get(session_type, [])
    if not partners:
        st.info(
            "Stint-keyed pace comparison is offered between race-format sessions "
            "(**Race ↔ Sprint**). Qualifying lap stint numbers don't correspond to race "
            "stints, so it is excluded rather than showing a misleading delta."
        )
        return
    ref = st.selectbox(
        "Compare against", partners, key="compare_ref_session",
        help="Reference session. Delta is (current − reference); negative = current faster.",
    )
    st.caption(
        f"Comparing **{session_type}** (current, primary) against **{ref}** (reference). "
        "Negative Delta = the current session was faster at equal tyre age."
    )

    with st.spinner(f"Loading {ref} session..."):
        try:
            setup_fastf1_cache()
            ref_session = load_fastf1_session(year, race_name, ref)
        except Exception as e:
            st.warning(f"Reference session unavailable: {e}")
            return

    if ref_session is None or ref_session.laps is None or ref_session.laps.empty:
        st.info(f"No lap data available for {race_name} ({ref}).")
        return

    out = compare_sessions(laps, ref_session.laps)
    if out is None or out.empty:
        st.info(
            "No matching drivers/stints with enough clean laps in both sessions — comparison "
            "is skipped rather than reported on noise."
        )
        return

    display = out.rename(columns={
        'Delta': 'Delta (s)',
        'BasePace_A': f'BasePace {session_type} (s)',
        'BasePace_B': f'BasePace {ref} (s)',
    })
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(
        "All figures are estimates with a confidence level from each driver's clean-lap "
        "sample. Rows without enough clean laps in either session are excluded."
    )


def _tab_insights(laps, session):
    st.subheader("Driver Performance Scores")

    session_best = laps['LapTime'].dropna().min().total_seconds()
    field_median = laps['LapTime'].dropna().median().total_seconds()

    scores = []
    for driver in laps['Driver'].unique():
        drv_laps = laps[(laps['Driver'] == driver) & laps['LapTime'].notna()]
        if len(drv_laps) < 3:
            continue
        times = drv_laps['LapTime'].dt.total_seconds()
        pace = max(0, 100 - (times.min() - session_best) * 10)
        consistency = max(0, 100 - times.std() * 20)
        race_pace = max(0, 100 - (times.mean() - field_median) * 5)
        overall = pace * 0.4 + consistency * 0.35 + race_pace * 0.25
        team = drv_laps['Team'].iloc[0] if 'Team' in drv_laps.columns else ''
        scores.append({
            'Driver': driver, 'Team': team,
            'Pace': round(pace, 1), 'Consistency': round(consistency, 1),
            'Race Pace': round(race_pace, 1), 'Overall': round(overall, 1),
        })

    if not scores:
        st.info("Not enough data to compute scores")
        return

    scores_df = pd.DataFrame(scores).sort_values('Overall', ascending=False).reset_index(drop=True)
    scores_df.index += 1
    scores_df.index.name = 'Rank'

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
            for d in radar_drivers:
                row = scores_df[scores_df['Driver'] == d].iloc[0]
                fig_r.add_trace(go.Scatterpolar(
                    r=[row['Pace'], row['Consistency'], row['Race Pace'], row['Overall'], row['Pace']],
                    theta=['Pace', 'Consistency', 'Race Pace', 'Overall', 'Pace'],
                    fill='toself', name=d,
                    line_color=TEAM_COLORS.get(row['Team'], '#888'),
                    opacity=0.65,
                ))
            fig_r.update_layout(**_fig_layout(height=440,
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], showticklabels=False,
                                    linecolor='#333', gridcolor='#333', gridwidth=1),
                    bgcolor='rgba(0,0,0,0)',
                    angularaxis=dict(tickfont=dict(size=12, color='#aaa')),
                ),
                showlegend=True,
                legend=dict(orientation="h", y=-0.12, font=dict(size=12)),
            ))
            show_plotly_chart(fig_r)

    with rc2:
        st.markdown("#### Leaderboard")
        display = scores_df[['Driver', 'Team', 'Pace', 'Consistency', 'Race Pace', 'Overall']]
        st.dataframe(
            display.style.background_gradient(subset=['Overall'], cmap='RdYlGn'),
            use_container_width=True, height=400,
        )

    # Tyre Degradation Analysis
    st.markdown("---")
    st.subheader("Tyre Degradation Analysis")

    drivers_list = sorted(laps['Driver'].unique())
    deg_drvs = st.multiselect("Drivers", drivers_list,
                              default=drivers_list[:DEFAULT_DRIVERS],
                              max_selections=6, key="deg_drv")

    if not deg_drvs:
        return

    # 1. Lap Time vs Tyre Life
    st.markdown("##### Lap Time vs Tyre Life")

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
        team = dl['Team'].iloc[0] if 'Team' in dl.columns else ''
        drv_color = TEAM_COLORS.get(team, '#888')

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
                            line=dict(width=1, color=drv_color)),
                hovertemplate=(
                    f"<b>{d}</b> ({compound})<br>"
                    "Tyre Life: %{x:.0f} laps<br>"
                    "Lap Time: %{customdata}"
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
                'Base Pace (s)': round(y.min(), 3),
            })

    if all_tyrelife_y:
        y_min_d, y_max_d = min(all_tyrelife_y) - 0.5, max(all_tyrelife_y) + 0.5
        time_y = _time_axis_layout("Lap Time", (y_min_d, y_max_d))
    else:
        time_y = dict(title=dict(text="Lap Time", font=LABEL_FONT))

    fig_deg.update_layout(**_fig_layout(height=440,
        title=dict(text="Lap Time vs Tyre Life (Degradation per Compound)", font=TITLE_FONT, x=0.01),
        xaxis=dict(title=dict(text="Tyre Life (laps)", font=LABEL_FONT), gridcolor=GRID,
                   tickfont=dict(size=11, color=TEXT_SECONDARY)),
        yaxis=time_y,
        showlegend=True,
        legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
    ))
    show_plotly_chart(fig_deg)

    st.markdown("---")

    # 2. Degradation Rate Comparison
    st.markdown("##### Degradation Rate Comparison")

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

        fig_rate.update_layout(**_fig_layout(height=max(200, len(sorted_deg) * 40 + 120),
            title=dict(text="Degradation Rate by Driver & Compound", font=TITLE_FONT, x=0.01),
            xaxis=dict(title=dict(text="Degradation (s/lap)", font=LABEL_FONT),
                       gridcolor=GRID, tickfont=dict(size=11, color=TEXT_SECONDARY)),
            yaxis=dict(tickfont=dict(size=11, color=TEXT_PRIMARY)),
            barmode='group', showlegend=False,
            margin=dict(l=150, r=30, t=55, b=50),
        ))
        show_plotly_chart(fig_rate)

    st.markdown("---")

    # 3. Tyre Strategy Overview
    st.markdown("##### Tyre Strategy Overview")

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
                        showarrow=False, font=dict(size=10, color='black', family=FONT),
                        xanchor='center', yanchor='middle',
                    )

            fig_strat.update_layout(**_fig_layout(height=max(180, n_drivers * 45 + 60),
                title=dict(text="Race Strategy - Tyre Stints", font=TITLE_FONT, x=0.01),
                xaxis=dict(title=dict(text="Lap", font=LABEL_FONT), range=[0.5, max_lap + 1],
                           gridcolor=GRID, tickfont=dict(size=11, color=TEXT_SECONDARY)),
                yaxis=dict(tickmode='array', tickvals=list(range(n_drivers)),
                           ticktext=list(reversed(stint_drvs)),
                           gridcolor='rgba(255,255,255,0.03)',
                           tickfont=dict(size=12, color=TEXT_PRIMARY)),
                barmode='overlay', showlegend=False,
            ))
            show_plotly_chart(fig_strat)

            used_compounds = stint_df['Compound'].unique()
            parts = []
            for c in ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']:
                if c in used_compounds:
                    color = COMPOUND_COLORS.get(c, "#888")
                    parts.append(
                        '<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px;font-size:0.82rem;">'
                        '<span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:{};"></span>'
                        '<span style="color:#aaa;">{}</span></span>'.format(color, c)
                    )
            st.markdown(
                '<div style="text-align:center;margin-top:-6px;margin-bottom:4px;">{}</div>'.format(
                    "&nbsp;&nbsp;".join(parts)
                ),
                unsafe_allow_html=True
            )

    st.markdown("---")

    # 4. Degradation Summary Table
    st.markdown("##### Degradation Summary")

    if deg_data:
        sum_df = pd.DataFrame(deg_data)

        def _color_deg(val):
            try:
                v = float(val)
                if v > 0.15:
                    return 'color:#FF6B6B; font-weight:600;'
                elif v > 0.08:
                    return 'color:#FFD93D; font-weight:500;'
                elif v > 0:
                    return 'color:#6BCB77;'
                else:
                    return 'color:#4FC3F7;'
            except Exception:
                return ''

        styled = sum_df.style.map(_color_deg, subset=['Deg (s/lap)'])
        st.dataframe(styled, use_container_width=True, hide_index=True)

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
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Highest Degradation", "{} ({})".format(worst['Driver'], worst['Compound']), "{:+.4f}s/lap".format(worst['Deg']))
            with k2:
                st.metric("Lowest Degradation", "{} ({})".format(best['Driver'], best['Compound']), "{:+.4f}s/lap".format(best['Deg']))
            with k3:
                st.metric("Field Average Degradation", "{:+.4f}".format(avg_deg), "s/lap")

    st.subheader("Race Control Messages")

    try:
        rc = get_race_control_messages(session)
        if rc is not None and not rc.empty:
            def _flag_color(val):
                v = str(val).upper()
                if 'RED' in v: return 'background-color:#e74c3c;color:white'
                if 'YELLOW' in v: return 'background-color:#f39c12;color:black'
                if 'GREEN' in v: return 'background-color:#27ae60;color:white'
                if 'BLACK' in v: return 'background-color:#222;color:white'
                if 'BLUE' in v: return 'background-color:#2980b9;color:white'
                return ''
            st.dataframe(rc.style.map(_flag_color, subset=['Flag']),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No race control messages")
    except Exception as e:
        st.info(f"Race control data unavailable: {e}")


page()

