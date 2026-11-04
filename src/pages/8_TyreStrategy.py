# -*- coding: utf-8 -*-
"""Tyre Strategy & Pit Stop Analysis: tyre usage, pit performance, tyre
degradation and what-if strategy simulation for one session.

Every figure comes from the session lap table (FastF1, or the OpenF1
rebuild when the timing archive does not answer) via the extractors in
``fastf1_extended`` and the calculations in ``strategy_analytics``.
Measured pit-stop times and modelled estimates are never mixed: the
pit-stop frame carries a ``Measured`` flag and the page surfaces the
split next to every average.

All maths lives in ``strategy_analytics`` / ``whatif`` / ``pitwindow``;
this file only decides what is shown and in which order.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from shared import load_fastf1_session, setup_fastf1_cache, show_plotly_chart
from config import FASTF1_CONFIG, MODEL_CONFIG, STREAMLIT_CONFIG
from styles import (
    chart_height, fig_layout as _fig_layout, LABEL_FONT, TEXT_SECONDARY,
    scaled_height, COMPOUND_COLORS,
)
from fastf1_extended import get_tyre_stints, get_pit_stops, ordered_drivers
from pace import clean_laps_for_analysis
from season_config import get_race_names, get_completed_races, latest_race_index
import strategy_analytics as sa

import ui
from ui import caption, data_table, empty, estimate_note, hero, kpis, note


# ---------------------------------------------------------------------------
# Session context (same shape as the Analysis Center's)
# ---------------------------------------------------------------------------

class _Context:
    """Session data plus the derivations every tab shares, built once."""

    def __init__(self, session):
        self.session = session
        self.laps = session.laps
        try:
            self.pit_stops = get_pit_stops(session)
        except Exception:
            self.pit_stops = pd.DataFrame()
        try:
            self.stints = sa.calculate_stint_length(get_tyre_stints(session))
        except Exception:
            self.stints = pd.DataFrame()
        try:
            self.clean = clean_laps_for_analysis(self.laps)
        except Exception:
            self.clean = pd.DataFrame()


@st.cache_resource(ttl=STREAMLIT_CONFIG.cache_ttl, show_spinner=False)
def _context(year, race, session_type):
    session = load_fastf1_session(year, race, session_type)
    if session is None or getattr(session, "laps", None) is None:
        return None
    return _Context(session)


def _compound_colour(compound):
    return COMPOUND_COLORS.get(str(compound or "").upper(), "#888888")


def _download(label, frame, filename):
    if frame is None or frame.empty:
        return
    st.download_button(
        label, frame.to_csv(index=False).encode(),
        file_name=filename, mime="text/csv", key=f"dl_{filename}",
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def page():
    year = st.session_state.get('selected_year', FASTF1_CONFIG.default_year)

    race_names = get_race_names(year)
    all_races = list(race_names.keys()) if race_names else []
    if not all_races:
        empty("No races found for this season")
        return

    c1, c2 = st.columns([3, 1])
    with c1:
        completed = get_completed_races(year)
        selected_race = st.selectbox(
            "Grand Prix", all_races,
            index=latest_race_index(year, all_races, completed),
            key="tyre_race")
    with c2:
        session_type = st.selectbox("Session", ["Race", "Qualifying", "Sprint"],
                                    key="tyre_session")

    hero(
        f"{year} season — {selected_race} · {session_type}",
        "Tyre Strategy & Pit Stop Analysis",
        headline="Tyre usage, pit stop performance and strategy projections "
                 "from the session data itself.",
        subline="Stints, compounds, degradation and pit windows — measured "
                "where telemetry allows, labelled estimated where it does not.",
    )
    ui.steps([
        'Pick the Grand Prix and session at the top; the page loads '
        'only that session.',
        'Read the tabs in order: tyre strategy first, then pit stops, '
        'then degradation.',
        'The comparison and simulator tabs are projections - they say '
        'what a plan would cost, not what will happen.',
        'Every figure is labelled measured or estimated, with the '
        'sample behind it.',
    ], title='How to read this page')

    with st.spinner(f"Loading the {session_type} session for {selected_race} "
                    f"(the first load downloads and parses the session)..."):
        setup_fastf1_cache()
        context = _context(year, selected_race, session_type)

    if context is None or context.laps is None or context.laps.empty:
        empty(f"No data available for {selected_race} ({session_type})",
              "The session may not have run yet, or its data is not published.")
        return

    source = str(getattr(context.session, 'source', 'FastF1') or 'FastF1')
    if source != 'FastF1':
        note("Lap times, tyres and pit stops for this session come from the "
             "OpenF1 feed (the official timing archive was not answering). "
             "Pit-stop durations may be modelled rather than measured; the "
             "Measured column in the pit table says which.", kind='info')

    _metric_cards(context)

    tabs = st.tabs(["Tyre Strategy", "Pit Stop Analysis", "Tyre Degradation",
                    "Strategy Comparison", "What-If Simulator"])
    with tabs[0]:
        _tab_strategy(context, year)
    with tabs[1]:
        _tab_pits(context)
    with tabs[2]:
        _tab_degradation(context)
    with tabs[3]:
        _tab_compare(context)
    with tabs[4]:
        _tab_simulator(context)

    _insights(context)
    try:
        lap_count = int(context.laps['LapNumber'].nunique())
    except (KeyError, TypeError, ValueError):
        lap_count = 0
    ui.verdict(
        f"This page read {lap_count} laps of {selected_race} "
        f"({session_type}): the tyre, pit and degradation numbers above "
        f"are the whole story of how the race was run on tyres.",
        label="What this means")
    ui.glossary(['stint', 'degradation', 'pit window', 'estimated', 'pace'])


def _metric_cards(context):
    """The four headline numbers, all counted from the session frames."""
    totals = sa.calculate_total_pit_stops(context.pit_stops)
    durations = sa.calculate_average_pit_duration(context.pit_stops)
    fastest = sa.calculate_fastest_pit_stop(context.pit_stops, context.laps)
    usage = sa.calculate_compound_usage(context.laps)

    if usage["usage"]:
        compound, info = max(usage["usage"].items(), key=lambda kv: kv[1]["laps"])
        n_drivers = len({d for d, per in usage["by_driver"].items()
                         if compound in per})
        compound_sub = (f"{info['pct']}% of {usage['total_laps']} laps · "
                        f"{n_drivers} drivers")
    else:
        compound, compound_sub = "Not published", "no tyre laps recorded"

    kpis([
        {'label': 'Total pit stops', 'value': str(totals["total"]),
         'sub': f"{totals['measured']} measured · {totals['estimated']} estimated",
         'icon': 'flag', 'tone': 'accent'},
        {'label': 'Average pit stop', 'value': (
            f"{durations['average']:.2f}s" if durations["average"] is not None
            else "n/a"),
         'sub': (f"fastest {durations['fastest']:.2f}s · "
                 f"slowest {durations['slowest']:.2f}s"
                 if durations["fastest"] is not None else "no valid times"),
         'icon': 'layers'},
        {'label': 'Fastest pit stop', 'value': (
            f"{fastest['duration']:.2f}s" if fastest["duration"] is not None
            else "n/a"),
         'sub': (f"{fastest['driver']}"
                 + (f" ({fastest['team']})" if fastest["team"] else "")
                 + (f", lap {fastest['lap']}" if fastest["lap"] else "")),
         'icon': 'trophy', 'tone': 'good'},
        {'label': 'Most common compound', 'value': compound,
         'sub': compound_sub, 'icon': 'crown'},
    ])


# ---------------------------------------------------------------------------
# Tab 1: Tyre strategy overview + compound analytics
# ---------------------------------------------------------------------------

@st.fragment
def _tab_strategy(context, year):
    laps, stints = context.laps, context.stints
    ui.section("Tyre strategy overview",
               note_text="One bar per stint",
               hint="Stint strategy and tyre usage throughout the race.")

    if stints is None or stints.empty:
        empty("Tyre strategy data not available for this session.")
    else:
        drivers = ordered_drivers(context.session, stints)
        scope_cols = st.columns([1, 1, 1, 3])
        scope = scope_cols[0].button("All drivers", key="scope_all")
        top3 = scope_cols[1].button("Top 3", key="scope_top3")
        top5 = scope_cols[2].button("Top 5", key="scope_top5")
        if scope:
            st.session_state['tyre_scope'] = drivers
        elif top3:
            st.session_state['tyre_scope'] = drivers[:3]
        elif top5:
            st.session_state['tyre_scope'] = drivers[:5]
        view = st.session_state.get('tyre_scope') or drivers
        view = [d for d in view if d in drivers]

        fig = go.Figure()
        for driver in view:
            dt = stints[stints['Driver'] == driver].sort_values('StartLap')
            for _, stint in dt.iterrows():
                compound = str(stint.get('Compound', 'UNKNOWN'))
                start, end = int(stint['StartLap']), int(stint['EndLap'])
                width = end - start + 1
                colour = _compound_colour(compound)
                fig.add_trace(go.Bar(
                    x=[width], y=[driver], orientation='h', base=start - 1,
                    marker_color=colour, marker_line_color='rgba(0,0,0,0.4)',
                    marker_line_width=0.6, showlegend=False,
                    text=compound[0], textposition='inside',
                    textfont=dict(
                        color='white' if compound.upper() in ('SOFT', 'WET') else '#111',
                        size=10),
                    hovertemplate=(
                        f"<b>{driver}</b><br>Tyre: {compound}<br>"
                        f"Lap {start}–{end}<br>Laps on this tyre: {width}"
                        "<extra></extra>"),
                ))
        total_laps = int(stints['EndLap'].max())
        fig.update_layout(**_fig_layout(
            height=scaled_height('tall', len(view)),
            xaxis=dict(range=[0, total_laps + 2], showgrid=False),
            xaxis_title=dict(text="Lap", font=LABEL_FONT),
            yaxis=dict(categoryorder='array', categoryarray=view[::-1],
                       showgrid=True),
            barmode='overlay',
        ))
        show_plotly_chart(fig)
        ui.legend([(colour, name) for name, colour in COMPOUND_COLORS.items()])
        _download("Export stints (CSV)", stints, "tyre_stints.csv")

    # Compound usage donut + pace by compound, side by side.
    usage_col, pace_col = st.columns(2)
    with usage_col:
        ui.section("Tyre compound usage",
                   note_text="Share of laps",
                   hint="Share of recorded laps per compound.")
        usage = sa.calculate_compound_usage(laps)
        if not usage["usage"]:
            empty("No tyre laps recorded for this session.")
        else:
            items = sorted(usage["usage"].items(), key=lambda kv: -kv[1]["laps"])
            fig_donut = go.Figure(go.Pie(
                labels=[f"{k} — {v['pct']}%" for k, v in items],
                values=[v["laps"] for _, v in items],
                hole=0.62,
                marker=dict(colors=[_compound_colour(k) for k, _ in items],
                            line=dict(width=2, color='#12151C')),
                textinfo='none',
                hovertemplate="<b>%{label}</b><br>%{value} laps<extra></extra>",
            ))
            fig_donut.update_layout(**_fig_layout(
                height=chart_height('row'), showlegend=True,
                legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center'),
            ))
            fig_donut.add_annotation(
                text=f"<b>{usage['total_laps']}</b><br><sub>total laps</sub>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=15, color='#F3F5F8'), align='center')
            show_plotly_chart(fig_donut)
            _download("Export compound usage (CSV)",
                      pd.DataFrame([{"Compound": k, **v}
                                    for k, v in items]),
                      "compound_usage.csv")
    with pace_col:
        ui.section("Pace by compound",
                   note_text="Average lap time",
                   hint="Raw averages mix race phases and fuel loads — "
                        "compare with the sample size, not just the bar.")
        mode = st.radio("Pace mode", ["Filtered race pace", "Raw average pace"],
                        key="pace_mode", horizontal=True)
        pace = sa.calculate_average_compound_pace(
            laps, mode="filtered" if mode.startswith("Filtered") else "raw")
        if pace is None or pace.empty:
            empty("No timed laps by compound.")
        else:
            fig_pace = go.Figure(go.Bar(
                x=pace["Compound"], y=pace["Mean (s)"],
                marker_color=[_compound_colour(c) for c in pace["Compound"]],
                marker_line_width=0,
                text=[f"{v:.1f}" for v in pace["Mean (s)"]],
                textposition='outside',
                textfont=dict(size=11, color='#E8EAF0'),
                hovertemplate=("<b>%{x}</b><br>Mean %{y:.2f}s"
                               "<extra></extra>"),
            ))
            fig_pace.update_layout(**_fig_layout(
                height=chart_height('row'),
                yaxis_title=dict(text="Lap time (s)", font=LABEL_FONT),
                showlegend=False,
            ))
            fig_pace.update_yaxes(showgrid=False)
            show_plotly_chart(fig_pace)
            caption(f"Samples: " + " · ".join(
                f"{r['Compound']} n={int(r['Laps'])} ({int(r['Drivers'])} drivers)"
                for _, r in pace.iterrows())
                + ". The fastest average is not proof the compound itself "
                  "is fastest.")


# ---------------------------------------------------------------------------
# Tab 2: Pit stop analysis + timeline
# ---------------------------------------------------------------------------

@st.fragment
def _tab_pits(context):
    ui.section("Pit stop analysis",
               note_text="Pit stop timing and duration",
               hint="Stops sorted by average duration. Measured times come "
                    "from the pit-lane crossing; estimated ones use the "
                    "model default and are labelled as such.")
    table = sa.calculate_pit_stop_table(context.pit_stops, context.laps)
    if table.empty:
        empty("No pit stop data available for this session.")
    else:
        if table["Measured"].sum() < len(table):
            estimate_note(
                f"{int(table['Measured'].sum())}/{len(table)} drivers have at "
                "least one measured stop; the rest use the model default "
                f"({MODEL_CONFIG['pit_loss_sec']:.0f}s, estimated).")
        data_table(
            ['Driver', 'Team', 'Pit Stops', 'Avg Duration (s)',
             'Fastest Stop (s)', 'Slowest Stop (s)', 'Pit Window',
             'Avg Pit Lap', 'Measured'],
            [[r['Driver'], r['Team'], int(r['Pit Stops']),
              r['Avg Duration (s)'], r['Fastest Stop (s)'],
              r['Slowest Stop (s)'], r['Pit Window'], r['Avg Pit Lap'],
              f"{int(r['Measured'])}/{int(r['Pit Stops'])}"]
             for _, r in table.iterrows()])
        _download("Export pit stops (CSV)", table, "pit_stops.csv")

    ui.section("Pit stop timeline",
               note_text="Lap vs duration",
               hint="One dot per stop. Hover for driver, lap and duration.")
    pit = context.pit_stops
    if pit is None or pit.empty:
        empty("No pit stop records to plot.")
        return
    drivers = ordered_drivers(context.session, pit)
    selected = st.multiselect("Highlight drivers", drivers,
                              default=drivers[:5], max_selections=10,
                              key="pit_highlight")
    fig = go.Figure()
    for driver in (selected or drivers):
        group = pit[pit['Driver'] == driver]
        if group.empty:
            continue
        colour = _compound_colour(group['Compound'].iloc[0]) \
            if 'Compound' in group.columns else '#9AA1AE'
        fig.add_trace(go.Scatter(
            x=group['Lap'], y=group['PitTime'], mode='markers+text',
            name=driver, marker=dict(size=11, color=colour,
                                     line=dict(width=1.2, color='white')),
            text=group['Lap'], textposition='top center',
            textfont=dict(size=9, color='#E8EAF0'),
            hovertemplate=(f"<b>{driver}</b><br>Lap %{{x}}<br>"
                           f"%{{y:.1f}}s<extra></extra>"),
        ))
    fig.update_layout(**_fig_layout(
        height=chart_height('block'),
        xaxis_title=dict(text="Pit lap", font=LABEL_FONT),
        yaxis_title=dict(text="Pit stop duration (s)", font=LABEL_FONT),
        legend=dict(orientation='h', y=1.02, x=1, xanchor='right'),
    ))
    show_plotly_chart(fig)
    caption("Dots at the lap the car pitted; the number is the stop lap. "
            "Duration is pit-lane time, not stationary service time.")


# ---------------------------------------------------------------------------
# Tab 3: Tyre degradation + pit window
# ---------------------------------------------------------------------------

@st.fragment
def _tab_degradation(context):
    ui.section("Tyre degradation",
               note_text="Estimated pace trend",
               hint="Median clean-lap time per tyre age, per compound. This "
                    "is a pace trend — traffic, fuel load and track "
                    "evolution are not separated out, so it is not pure "
                    "tyre degradation.")
    trend = sa.degradation_trend(context.laps)
    if trend.empty:
        empty("Not enough clean laps to draw a degradation trend.")
    else:
        fig = go.Figure()
        for compound, group in trend.groupby("Compound"):
            fig.add_trace(go.Scatter(
                x=group["TyreAge"], y=group["MedianLapTime_s"],
                mode='lines+markers', name=compound,
                line=dict(color=_compound_colour(compound), width=2.5),
                marker=dict(size=6, color=_compound_colour(compound),
                            line=dict(width=1, color='white')),
                hovertemplate=(f"<b>{compound}</b><br>Tyre age %{{x:.0f}}"
                               f"<br>Median %{{y:.2f}}s<extra></extra>"),
            ))
        fig.update_layout(**_fig_layout(
            height=chart_height('feature'),
            xaxis_title=dict(text="Tyre age (laps on this set)", font=LABEL_FONT),
            yaxis_title=dict(text="Lap time (s)", font=LABEL_FONT),
            hovermode='x unified',
            legend=dict(orientation='h', y=1.02, x=1, xanchor='right'),
        ))
        show_plotly_chart(fig)

    deg = sa.calculate_tyre_degradation(context.laps)
    if deg is not None and not deg.empty:
        fitted = deg.dropna(subset=["DegPerLap"])
        # Same small-sample guard as the insights: slopes from fewer than
        # 8 clean laps are noise and stay out of the table.
        if "N" in fitted.columns:
            fitted = fitted[fitted["N"] >= 8]
        fitted = fitted.sort_values("DegPerLap", ascending=False)
        if not fitted.empty:
            ui.section("Degradation per stint",
                       note_text="Estimated, seconds per lap",
                       hint="Slope of the pace fit on clean laps "
                            "(stints with fewer than 8 clean laps excluded).")
            data_table(
                ['Driver', 'Stint', 'Compound', 'Clean laps',
                 'Mean pace (s)', 'Fresh-tyre pace (s, est)',
                 'Slower each lap (s, est)'],
                [[r['Driver'], int(r['Stint']), r['Compound'], int(r['N']),
                  r['MeanPace'], r['BasePace'], r['DegPerLap']]
                 for _, r in fitted.head(15).iterrows()])

    ui.section("Best lap to pit stop",
               note_text="Estimated pit window",
               hint="Grid-searched stop lap per driver under the shared "
                    "strategy model. Insufficient data is shown as such.")
    n_stops = st.selectbox("Stops modelled", [1, 2], key="pw_stops")
    windows = sa.optimize_pit_lap(context.laps, n_stops=int(n_stops))
    if windows.empty:
        note("Not enough clean-lap data to optimise a pit window for this "
             "session.", kind='warn')
    else:
        show = windows.copy()
        cols = ['Driver', 'OptimalLap', 'WindowStart', 'WindowEnd']
        if 'OptimalLap2' in show.columns:
            cols += ['OptimalLap2', 'WindowStart2', 'WindowEnd2']
        cols += ['BasePace', 'DegPerLap', 'Level']
        cols = [c for c in cols if c in show.columns]
        data_table(cols, [[r[c] for c in cols] for _, r in show.iterrows()])


# ---------------------------------------------------------------------------
# Tab 4: Strategy comparison (three plans)
# ---------------------------------------------------------------------------

@st.fragment
def _tab_compare(context):
    ui.section("Strategy comparison",
               note_text="Estimated — ranked projection",
               hint="Three customisable plans over the race distance of this "
                    "session. The fastest row is fastest under the stated "
                    "assumptions, not a recommendation.")
    total_laps = int(context.laps['LapNumber'].max()) \
        if 'LapNumber' in context.laps.columns else 50
    base_lap = st.number_input("Fresh-tyre base lap time (s)", 60.0, 150.0,
                               float(MODEL_CONFIG['default_base_lap_time']),
                               step=0.1, key="cmp_base")
    plans = []
    for i in range(3):
        with st.expander(f"Plan {i + 1}", expanded=(i == 0)):
            compounds = st.multiselect(
                "Compounds in order", ["SOFT", "MEDIUM", "HARD"],
                default=[["SOFT", "HARD"], ["MEDIUM", "HARD"],
                         ["SOFT", "MEDIUM", "HARD"]][i], key=f"cmp_c{i}")
            stops_text = st.text_input(
                "Stop laps (comma separated)", ["15", "15", "10, 20"][i],
                key=f"cmp_s{i}")
            stop_laps = [int(s) for s in stops_text.replace(",", " ").split()
                         if s.strip().lstrip('-').isdigit()]
            plans.append({
                "name": f"Plan {i + 1}",
                "compounds": compounds or ["MEDIUM"],
                "stop_laps": stop_laps,
            })
    table = sa.compare_strategies(plans, total_laps,
                                  base_lap_time=float(base_lap))
    if table.empty:
        empty("No valid plans to compare.")
        return
    data_table(
        ['Name', 'TotalTime (est)', 'GapToBest (est)', 'Stops', 'Compounds',
         'Risk', 'Confidence', 'Estimated'],
        [[r['Name'], r['TotalTime'], r['GapToBest'], int(r['Stops']),
          " → ".join(r['Compounds']), r['Risk'], r['Confidence'],
          r['Estimated']] for _, r in table.iterrows()])
    fig = go.Figure(go.Bar(
        x=table['Name'], y=table['TotalTime'],
        marker_color=['#00D59A'] + ['#4388E8'] * (len(table) - 1),
        marker_line_width=0,
        text=[f"{v:.0f}s" for v in table['TotalTime']],
        textposition='outside', textfont=dict(size=11, color='#E8EAF0'),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}s<extra></extra>",
    ))
    fig.update_layout(**_fig_layout(
        height=chart_height('row'),
        yaxis_title=dict(text="Projected race time (s, est)", font=LABEL_FONT),
        showlegend=False,
    ))
    fig.update_yaxes(showgrid=False)
    show_plotly_chart(fig)
    _download("Export strategy comparison (CSV)", table,
              "strategy_comparison.csv")


# ---------------------------------------------------------------------------
# Tab 5: What-if simulator
# ---------------------------------------------------------------------------

@st.fragment
def _tab_simulator(context):
    ui.section("What-if strategy simulator",
               note_text="Estimated — not a prediction",
               hint="Change the plan, the pace assumptions and the penalties; "
                    "the projected race time updates. Real historical pace "
                    "data sets the baseline, the rest are your assumptions.")
    total_laps = int(context.laps['LapNumber'].max()) \
        if 'LapNumber' in context.laps.columns else 50
    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.selectbox("Starting compound", ["SOFT", "MEDIUM", "HARD"],
                             key="sim_start")
        second = st.selectbox("Second compound", ["SOFT", "MEDIUM", "HARD"],
                              key="sim_second")
        third = st.selectbox("Third compound", ["SOFT", "MEDIUM", "HARD"],
                             key="sim_third")
    with c2:
        stop1 = st.number_input("First stop lap", 1, max(2, total_laps - 1),
                                15, key="sim_stop1")
        stop2 = st.number_input("Second stop lap (0 = one-stop)", 0,
                                max(1, total_laps - 1), 0, key="sim_stop2")
        race_laps = st.number_input("Race length (laps)", 5, 100,
                                    total_laps, key="sim_laps")
    with c3:
        base_lap = st.number_input("Fresh-tyre base lap (s)", 60.0, 150.0,
                                   float(MODEL_CONFIG['default_base_lap_time']),
                                   step=0.5, key="sim_base")
        pit_loss = st.number_input("Pit loss per stop (s)", 0.0, 60.0,
                                   float(MODEL_CONFIG['pit_loss_sec']),
                                   step=0.5, key="sim_loss")
        traffic = st.number_input("Traffic penalty (s/lap)", 0.0, 5.0, 0.0,
                                  step=0.05, key="sim_traffic")

    compounds = [start, second] + ([third] if stop2 else [])
    stop_laps = [int(stop1)] + ([int(stop2)] if stop2 else [])
    result = sa.simulate_race_strategy(
        compounds, stop_laps, int(race_laps),
        base_lap_time=float(base_lap), pit_loss=float(pit_loss),
        traffic_penalty=float(traffic))

    if result.get('level') == 'insufficient':
        note(f"Cannot simulate this plan: {result.get('reason')}.", kind='warn')
        return

    pit_loss_total = result['n_stops'] * float(pit_loss)
    kpis([
        {'label': 'Projected race time', 'value': f"{result['total_time']:.0f}s",
         'sub': 'estimated', 'icon': 'trophy', 'tone': 'good'},
        {'label': 'Pit stop loss', 'value': f"{pit_loss_total:.0f}s",
         'sub': f"{result['n_stops']} stop(s) × {float(pit_loss):.0f}s",
         'icon': 'flag'},
        {'label': 'Compound sequence', 'value': " → ".join(result['compounds']),
         'sub': f"stops at laps {', '.join(str(s) for s in result['stops_laps'])}",
         'icon': 'layers'},
        {'label': 'Model confidence', 'value': str(result.get('confidence', '—')),
         'sub': 'under the stated assumptions', 'icon': 'crown'},
    ])
    note("This is a what-if simulation, not a prediction of actual race "
         "results. It uses the model's degradation and fuel constants; "
         "traffic, weather and race situation only enter through the "
         "penalties you set above.", kind='estimate')
    with st.expander("Assumptions behind this simulation"):
        for assumption in result.get('assumptions', []):
            st.markdown(f"- {assumption}")


# ---------------------------------------------------------------------------
# Insights (page footer)
# ---------------------------------------------------------------------------

def _insights(context):
    insights = sa.generate_strategy_insights(
        context.laps, context.stints, context.pit_stops)
    if not insights:
        return
    ui.section("Key insights",
               note_text="Rule-based — every line is number-backed",
               hint="Derived from this session only. Nothing is generated "
                    "without supporting metrics.")
    ui.takeaways([(item['title'], f"{item['value']} — {item['detail']}")
                  for item in insights])


page()
