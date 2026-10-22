# -*- coding: utf-8 -*-
"""Weekend report: what happened, what it means, and what to watch next.

The page is written as a briefing for a reader who does not follow Formula 1
closely. It opens with the facts of the race, then says what those facts
mean for the championship, then points at what to keep an eye on, and
finally carries the full report that can be downloaded.
"""

import datetime as _dt
import html
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go

from shared import (
    load_race_data, load_fastf1_session, load_race_grid, team_colour,
    show_plotly_chart,
)
from config import FASTF1_CONFIG, STREAMLIT_CONFIG
from season_config import (
    get_event_schedule_cached, get_race_names, latest_race_index,
    resolve_csv_track,
)
from report import build_report
from prediction import predict_race, rows_before_race, compute_driver_form
from briefing import pre_race_briefing, post_race_debrief
from pace import data_quality
from styles import (
    chart_height,
    fig_layout as _fig_layout,
    LABEL_FONT,
)
import ui
import wx
import radar
from ui import CONFIDENCE_LABELS


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _ordinal(number) -> str:
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 11 -> '11th'."""
    value = _int_or_none(number)
    if value is None:
        return '—'
    if 10 <= value % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(value % 10, 'th')
    return f'{value}{suffix}'


def _int_or_none(value):
    """A whole number from a frame cell, or None when it is missing."""
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _number_or_none(value):
    """A plain number from a frame cell, or None when it is missing."""
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _track_order(df):
    """The Grand Prix names of a frame in calendar order."""
    if df is None or df.empty or 'Track' not in df.columns:
        return []
    if 'Round' in df.columns:
        rounds = pd.to_numeric(df['Round'], errors='coerce')
        if rounds.notna().any():
            ordered = (df.assign(_round=rounds)
                         .dropna(subset=['_round'])
                         .sort_values('_round', kind='stable'))
            tracks = list(dict.fromkeys(ordered['Track'].dropna().astype(str)))
            if tracks:
                return tracks
    return list(dict.fromkeys(df['Track'].dropna().astype(str)))


def _team_map(df):
    """The team each driver drives for."""
    teams = {}
    if df is None or df.empty or 'Driver' not in df.columns:
        return teams
    if 'Team' not in df.columns:
        return teams
    for driver, group in df.groupby(df['Driver'].astype(str)):
        values = group['Team'].dropna()
        if not values.empty:
            teams[str(driver)] = str(values.iloc[-1])
    return teams


def _points_by_driver(df):
    """Championship points per driver (races and sprints both count)."""
    if df is None or df.empty or 'Driver' not in df.columns:
        return {}
    if 'Points' not in df.columns:
        return {}
    points = pd.to_numeric(df['Points'], errors='coerce').fillna(0)
    return {str(driver): float(value)
            for driver, value in points.groupby(df['Driver'].astype(str)).sum().items()}


def _standings_before_after(df, track):
    """Championship points before a race and through it."""
    order = _track_order(df)
    if track in order:
        index = order.index(track)
        before_rows = df[df['Track'].isin(order[:index])]
        through_rows = df[df['Track'].isin(order[:index + 1])]
    else:
        before_rows = df.iloc[0:0]
        through_rows = df
    return _points_by_driver(before_rows), _points_by_driver(through_rows)


def _race_rows(race_df, track):
    """The rows of one Grand Prix, race session only."""
    if race_df is None or race_df.empty or 'Track' not in race_df.columns:
        return race_df
    rows = race_df[race_df['Track'] == track]
    if rows.empty:
        wanted = str(track).lower()
        rows = race_df[race_df['Track'].astype(str).str.lower() == wanted]
    return rows


def _stint_rows(stints, names):
    """One entry per run of laps on one tyre, numbered per driver."""
    rows = []
    if stints is None or stints.empty:
        return rows
    for code, group in stints.groupby('Driver'):
        if 'Stint' in group.columns:
            group = group.sort_values('Stint')
        for index, (_, row) in enumerate(group.iterrows(), start=1):
            driver = str(code)
            rows.append({
                'driver': names.get(driver, driver),
                'run': f"Run {index}",
                'compound': str(row.get('Compound', '')),
                'laps': str(row.get('Laps', '')),
            })
    return rows


# ---------------------------------------------------------------------------
# Session extras (pit stops and tyre runs)
# ---------------------------------------------------------------------------

def _session_extras(year, race):
    """Best-effort extra stats + report sections from a FastF1 session."""
    extras = {}
    sections = []
    stints = None
    names = {}
    try:
        # Only the lap table is needed here (pit stops, tyre runs), so the
        # weather and race-control feeds are not fetched: every extra feed is
        # another HTTP request in front of the report.
        session = load_fastf1_session(year, race, 'Race', weather=False,
                                      messages=False)
        if session is None:
            return extras, sections, stints, names

        # FastF1 names the drivers by their three-letter code; the rest of the
        # app uses the full name, so the codes are translated here.
        try:
            names = {str(code): str(full) for code, full
                     in dict(getattr(session, 'drivers', {}) or {}).items()}
        except Exception:
            names = {}

        try:
            from fastf1_extended import get_pit_stops
            pits = get_pit_stops(session)
            extras['pit_stops'] = int(len(pits)) if pits is not None and not pits.empty else 0
        except Exception:
            pass

        try:
            from fastf1_extended import get_tyre_stints
            stints = get_tyre_stints(session)
            if stints is not None and not stints.empty:
                lines = ["### Tyre strategy", "",
                         "Driver | Run of laps on one tyre | Tyre | Laps |",
                         "---|---|---|---:"]
                for row in _stint_rows(stints, names)[:16]:
                    lines.append(
                        f"{row['driver']} | {row['run']} | {row['compound']} | "
                        f"{row['laps']}"
                    )
                sections.append("\n".join(lines))
        except Exception:
            pass
    except Exception:
        pass
    return extras, sections, stints, names


# ---------------------------------------------------------------------------
# Charts and the exported document
# ---------------------------------------------------------------------------

def _standings_figure(points, teams, year, limit=10):
    """Bar chart of the championship standings, coloured by team."""
    ordered = sorted(points.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
    ordered = ordered[:limit]
    drivers = [driver for driver, _ in ordered]
    values = [float(value) for _, value in ordered]
    colours = [team_colour(teams.get(driver, ''), year) for driver in drivers]
    fig = go.Figure(go.Bar(
        x=values, y=drivers, orientation='h',
        marker_color=colours,
        marker_line=dict(width=1, color='rgba(255,255,255,0.3)'),
    ))
    # The chart carries no title of its own: the section heading names it.
    fig.update_layout(**_fig_layout(
        height=chart_height('feature'),
        xaxis_title=dict(text="Championship points", font=LABEL_FONT),
        showlegend=False, bargap=0.35,
    ))
    fig.update_xaxes(rangemode='tozero')
    fig.update_yaxes(autorange='reversed', showgrid=False, automargin=True)
    return fig


def _chart_html(fig):
    """Dark standings bar chart embedded as self-contained HTML."""
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl, show_spinner=False)
def _cached_report(year, track, season_df, extra, extra_sections):
    """The markdown report: pure text from the season frame and the extras."""
    return build_report(year, track, season_df, extra=extra,
                        extra_sections=extra_sections)


@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl, show_spinner=False)
def _cached_debrief(race_df, track, forecast):
    """The post-race debrief: written once per result, not once per rerun."""
    return post_race_debrief(race_df, track, forecast=forecast,
                             forecast_includes_race=False)


@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl, show_spinner=False)
def _cached_preview(season_df, grid, race):
    """The pre-race briefing, written from the rounds before the race."""
    return pre_race_briefing(season_df, grid, race=race)


@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl, show_spinner=False)
def _cached_html(report_md, chart_html):
    """The downloadable HTML page (the chart HTML is embedded, not rebuilt)."""
    return _html_report(report_md, chart_html)


def _html_report(report_md, chart_html):
    # The markdown is escaped: a team or driver name containing "&" or "<"
    # would otherwise break the exported document.
    escaped = html.escape(report_md)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>F1 Lab — Weekend Report</title>
<style>
  body {{ background:#0B0D12; color:#E8EAF0; font-family:'Outfit', Inter, Segoe UI, sans-serif;
         margin:0; padding:2.5rem max(1rem, calc(50vw - 30rem)); line-height:1.6; }}
  h1, h2 {{ color:#E10600; }}
  h2 {{ font-size:1.05rem; margin:1.5rem 0 .5rem; }}
  pre {{ white-space:pre-wrap; font-family:ui-monospace, Consolas, monospace;
         background:#12151C; border:1px solid rgba(255,255,255,.08); border-radius:12px;
         padding:1.2rem; }}
</style>
</head>
<body>
<h2>Season standings</h2>
{chart_html}
<pre>{escaped}</pre>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

_CONFIDENCE_WORDS = {'moderate': 'medium', 'low': 'low'}


def _reading(value, unit='', digits=1):
    """A number for a table cell, or an em dash when it is missing."""
    if value is None:
        return '—'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return '—'
    return f"{number:.{digits}f} {unit}".strip()


def _wind_kmh(value, digits=0):
    """A wind speed in km/h from the feed's metres per second."""
    if value is None:
        return '—'
    try:
        return f"{float(value) * 3.6:.{digits}f} km/h"
    except (TypeError, ValueError):
        return '—'


def _horizon_text(hours):
    """'about 12 hours ahead' from a forecast horizon in hours."""
    if hours is None:
        return 'an unknown lead time'
    if hours < 0:
        return 'a session that has already started'
    if hours < 1:
        return 'under an hour ahead'
    if hours < 48:
        return f"about {hours:.0f} hours ahead"
    return f"about {hours / 24:.0f} days ahead"


def _next_session(year):
    """The next calendar session that has not started yet, or None.

    The FastF1 schedule is already loaded for the calendar, so this adds no
    request. A session without a usable UTC start time is skipped rather than
    guessed.
    """
    try:
        schedule = get_event_schedule_cached(year)
    except Exception:
        return None
    if schedule is None or getattr(schedule, 'empty', True):
        return None
    now = _dt.datetime.now(_dt.timezone.utc)
    upcoming = []
    for _, event in schedule.iterrows():
        for number in range(1, 6):
            name = event.get(f'Session{number}')
            start = event.get(f'Session{number}DateUtc')
            if name is None or pd.isna(name):
                continue
            if start is None or pd.isna(start):
                continue
            moment = (start.to_pydatetime() if hasattr(start, 'to_pydatetime')
                      else start)
            if not isinstance(moment, _dt.datetime):
                continue
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=_dt.timezone.utc)
            else:
                moment = moment.astimezone(_dt.timezone.utc)
            if moment <= now:
                continue
            upcoming.append({
                'event': str(event.get('EventName') or ''),
                'location': str(event.get('Location') or ''),
                'session': str(name),
                'start': moment,
            })
    if not upcoming:
        return None
    return min(upcoming, key=lambda item: item['start'])


@st.fragment
def _weather_tab(year, selected_race, track, race_df):
    """The measured weather of one session, and the next session's outlook."""
    ui.section(
        "Weather",
        f"the conditions at the {selected_race}, and the rain risk for the "
        "next session",
        hint="The first block is measured weather: the conditions the "
             "official timing feed recorded during the session. The second is "
             "a forecast for a session that has not run yet, so it is an "
             "estimate and is labelled as one.",
    )

    rows = _race_rows(race_df, track) if race_df is not None else None
    session_key = None
    if rows is not None and not rows.empty and 'SessionKey' in rows.columns:
        keys = pd.to_numeric(rows['SessionKey'], errors='coerce').dropna()
        if not keys.empty:
            session_key = int(keys.iloc[0])
    conditions = wx.session_weather(session_key) if session_key is not None else None

    upcoming = _next_session(year)
    forecast_place = None
    if upcoming is not None:
        for candidate in (upcoming.get('location'), upcoming.get('event')):
            if candidate and wx.coords_for(candidate):
                forecast_place = candidate
                break
    outlook = (wx.rain_outlook(forecast_place, upcoming['start'])
               if forecast_place else None)

    # --- the headline tiles -----------------------------------------------
    tiles = [
        {'label': 'Air temperature',
         'value': _reading((conditions or {}).get('air_temp_mean'), 'C'),
         'sub': 'session average' if conditions else 'weather not published',
         'tone': 'info'},
        {'label': 'Track temperature',
         'value': _reading((conditions or {}).get('track_temp_mean'), 'C'),
         'sub': 'session average' if conditions else 'weather not published',
         'tone': 'accent'},
    ]
    if outlook:
        tiles.append({
            'label': 'Rain probability',
            'value': f"{outlook['probability']:.0f}%",
            'sub': f"{outlook['band']} for the next session",
            'tone': 'good' if outlook['band'] == 'dry' else 'warn',
        })
    else:
        tiles.append({'label': 'Rain probability', 'value': '—',
                      'sub': 'no forecast available'})
    ui.kpis(tiles)

    if conditions is None:
        if rows is None or rows.empty:
            ui.empty(
                "Weather not published for this session",
                "This session has not run yet, so the official feed has no "
                "weather to report. The page does not fill the gap with an "
                "estimate.",
            )
        else:
            ui.empty(
                "Weather not published for this session",
                "The official feed has no weather readings for this session, "
                "so there is nothing to show. The page does not fill the gap "
                "with an estimate.",
            )
    else:
        samples = int(conditions['samples'])
        rain_total = conditions.get('rainfall_total')
        rain_share = (rain_total / samples
                      if (rain_total is not None and samples) else None)
        ui.data_table(
            ['Reading', 'Average', 'Lowest', 'Highest', 'Unit'],
            [
                ['Air temperature',
                 _reading(conditions.get('air_temp_mean')),
                 _reading(conditions.get('air_temp_min')),
                 _reading(conditions.get('air_temp_max')), 'C'],
                ['Track temperature',
                 _reading(conditions.get('track_temp_mean')),
                 _reading(conditions.get('track_temp_min')),
                 _reading(conditions.get('track_temp_max')), 'C'],
                ['Humidity',
                 _reading(conditions.get('humidity_mean')),
                 _reading(conditions.get('humidity_min')),
                 _reading(conditions.get('humidity_max')), '%'],
                ['Wind speed',
                 _wind_kmh(conditions.get('wind_speed_mean')),
                 _wind_kmh(conditions.get('wind_speed_min')),
                 _wind_kmh(conditions.get('wind_speed_max')), 'km/h'],
                ['Air pressure',
                 _reading(conditions.get('pressure_mean')),
                 '—', '—', 'hPa'],
                ['Rainfall flag',
                 _reading(rain_share, '', 2),
                 _reading(conditions.get('rainfall_min')),
                 _reading(conditions.get('rainfall_max')), 'flag'],
            ],
        )
        notes = [f"Measured by the official timing feed during the session "
                 f"({samples} readings).",
                 "Rainfall is the feed's 0 or 1 flag, so the average is the "
                 "share of readings that reported rain; 0 means a dry "
                 "session."]
        dropped = conditions.get('track_temp_zeros_dropped') or 0
        if dropped:
            notes.append(
                f"{dropped} track-temperature reading"
                f"{'s' if dropped != 1 else ''} of 0 C was dropped as a "
                "sensor dropout, so the lowest value above is the lowest "
                "usable reading.")
        ui.caption(' '.join(notes))

    # --- the forecast -----------------------------------------------------
    ui.section(
        "Rain outlook",
        "the forecast for the next session that has not run",
        hint="A forecast for one hour at the circuit, from the public "
             "Open-Meteo service. It says how likely rain is, not how much "
             "will fall, so it is an estimate rather than a measurement.",
    )

    if upcoming is None:
        ui.empty(
            "No upcoming session to forecast",
            "Every session in this season has already run, or the calendar is "
            "not available offline. The measured weather above does not need "
            "a forecast.",
        )
    elif forecast_place is None:
        ui.empty(
            "No circuit coordinates for the next session",
            f"The coordinate table does not know the next circuit "
            f"({upcoming.get('event') or upcoming.get('location') or 'unknown'}"
            f"), so the page cannot ask for a forecast.",
        )
    elif outlook is None:
        ui.empty(
            "Forecast not available",
            "The public forecast service did not answer, or has no data for "
            "that hour. The measured session weather above is unaffected.",
        )
    else:
        ui.data_table(
            ['Forecast reading', 'Value'],
            [
                ['Session', f"{upcoming['session']} at {upcoming['event']}"],
                ['Start (UTC)', upcoming['start'].strftime('%Y-%m-%d %H:%M')],
                ['Circuit', upcoming.get('location') or upcoming.get('event')],
                ['Rain probability', f"{outlook['probability']:.0f}%"],
                ['Rain band', outlook['band']],
                ['Temperature', _reading(outlook.get('temperature_c'), 'C')],
                ['Wind speed', _wind_kmh(outlook.get('wind_speed_kmh'))],
            ],
        )
        ui.confidence_tag(outlook['confidence'])
        ui.caption(
            f"{CONFIDENCE_LABELS[outlook['confidence']][0]}: the session is "
            f"{_horizon_text(outlook['horizon_hours'])} and the forecast comes "
            f"from the public Open-Meteo service."
        )
        ui.estimate_note(
            "Estimated, not measured: the rain probability is what the "
            "service expects for that hour. It can change before the session "
            "and a shower can miss the circuit entirely."
        )

    # --- the live radar ---------------------------------------------------
    ui.section(
        "Rain radar",
        "where the rain is around the circuit, right now",
        hint="Radar sees rain that is already falling; it is not a forecast. "
             "Read it as 'the rain is that way, that far away', not as 'it "
             "will be raining on the track'.",
    )

    radar_place = forecast_place or track
    radar_role = ("the circuit of the next session" if forecast_place
                  else "the circuit of this race")
    radar_data = wx.radar_view(radar_place) if radar_place else None
    if radar_data is None:
        ui.empty(
            "Radar not available",
            "The circuit's coordinates are unknown, or the radar service did "
            "not answer. The measured weather and the outlook above do not "
            "depend on this panel.",
        )
    else:
        components.html(radar.radar_html(radar_data), height=radar.HEIGHT,
                        scrolling=False)
        facts = [f"Live radar of {radar_role}."]
        frames = int(radar_data.get('count') or 0)
        span_min = radar_data.get('span_min')
        age_min = radar_data.get('age_min')
        span_km = radar_data.get('span_km')
        if frames:
            facts.append(f"{frames} frames covering "
                         f"{span_min:.0f} minutes of the recent past."
                         if span_min else f"{frames} frames of the recent past.")
        if age_min is not None:
            when = ("under a minute ago" if age_min < 1
                    else f"{age_min:.0f} minutes ago")
            facts.append(f"The newest frame was observed {when}.")
        if span_km:
            facts.append("The free radar tiles stop at zoom 7, so this is a "
                         f"regional picture about {span_km:.0f} km across: "
                         "enough to see a band of rain approaching, not "
                         "enough to see the track.")
        ui.caption(' '.join(facts))
        ui.caption(
            "Radar imagery by RainViewer, map tiles by OpenStreetMap. The "
            "frames are measurements of rain that has already fallen, not a "
            "forecast, and the free feed carries no nowcast, so the panel "
            "does not draw one."
        )

    # --- the words --------------------------------------------------------
    ui.explain(
        "Air temperature",
        "How warm the air around the circuit was, in degrees Celsius. It "
        "matters less for grip than the track temperature.",
    )
    ui.explain(
        "Track temperature",
        "How warm the asphalt was. A hot track gives more grip but wears the "
        "tyres faster; a cool one can make the tyres hard to switch on.",
    )
    ui.explain(
        "Rain probability",
        "The chance the forecast gives of rain falling in that hour. It is an "
        "estimate from a public forecast, never a measurement.",
    )
    ui.explain(
        "Radar",
        "A radar network detects the rain that is already falling and maps "
        "how heavy it is. The picture above is a measurement, but a regional "
        "one: it shows where the rain is coming from, not whether it will "
        "reach the track.",
    )

    # --- the takeaway -----------------------------------------------------
    air = conditions.get('air_temp_mean') if conditions else None
    track_temp = conditions.get('track_temp_mean') if conditions else None
    detail = (f", with the air at {air:.1f} C and the track at "
              f"{track_temp:.1f} C on average"
              if air is not None and track_temp is not None else "")
    verdict = []
    if conditions is None:
        verdict.append(f"No weather was published for the {selected_race}.")
    elif conditions.get('wet') is True:
        verdict.append(
            f"The {selected_race} ran in the wet: rain was recorded in "
            f"{conditions['rain_samples']} of {conditions['samples']} "
            f"readings{detail}.")
    elif conditions.get('wet') is False:
        verdict.append(
            f"The {selected_race} ran dry: no rain in {conditions['samples']} "
            f"readings{detail}.")
    else:
        verdict.append(
            f"The {selected_race} has weather readings but no rainfall "
            f"values, so the page cannot say whether it rained{detail}.")
    if outlook is not None and upcoming is not None:
        verdict.append(
            f"For the next session ({upcoming['session']} at "
            f"{upcoming['event']}) the forecast gives a "
            f"{outlook['probability']:.0f}% chance of rain, a "
            f"{outlook['band']} band with "
            f"{_CONFIDENCE_WORDS.get(outlook['confidence'], outlook['confidence'])} "
            f"confidence.")
    elif upcoming is not None:
        verdict.append("No forecast is available for the next session.")
    else:
        verdict.append("There is no upcoming session left to forecast.")
    if radar_data is not None:
        span_min = radar_data.get('span_min') or 0
        window = (f"the last {span_min:.0f} minutes of rain around the circuit"
                  if span_min >= 1 else "the most recent radar frame")
        verdict.append(
            f"The radar above is {window}: a measurement of where the rain is, "
            f"and the one reading on this page that is neither an estimate nor "
            f"a forecast.")
    ui.verdict(' '.join(verdict))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def page():
    year = st.session_state.get('selected_year', FASTF1_CONFIG.default_year)
    df = load_race_data(year)
    if df is None or df.empty:
        ui.empty(f"No {year} report yet",
              "Results appear here once races are completed. Live session "
              "data is available in the Analysis Center.")
        return

    race_df = df[df['SessionType'] == 'Race'].copy() if 'SessionType' in df.columns else df.copy()

    race_names = get_race_names(year)
    all_races = list(race_names.keys()) if race_names else []
    if not all_races:
        ui.empty("No races available yet.",
              "The calendar for this season is not published yet.")
        return

    ui.page_header(
        "Weekend report",
        "What happened at the Grand Prix you pick, what it means for the "
        "championship, and what to watch next.",
        chips=[('season', str(year)), ('', 'OpenF1 results')],
    )

    # The frame's Track column holds the FastF1 event name, so the dropdown
    # value matches it directly; the resolution keeps the page working with
    # either spelling.
    csv_tracks = (list(dict.fromkeys(race_df['Track'].astype(str)))
                  if 'Track' in race_df.columns else [])

    # Default to the most recent race with a result, so the report is about
    # the race that just happened rather than the season opener.
    selected_race = st.selectbox("Which Grand Prix?", all_races,
                                 index=latest_race_index(year, all_races, csv_tracks),
                                 key="report_race")
    csv_race = resolve_csv_track(year, selected_race, csv_tracks)
    track = csv_race or selected_race

    with st.spinner("Building the report..."):
        extras, sections, stints, names = _session_extras(year, selected_race)
        # The full frame: the season standings are championship standings and
        # therefore count the sprint points too (the race result inside the
        # report is filtered to the race session).
        report_md = _cached_report(year, track, df, extras, sections)

    rows = _race_rows(race_df, track)
    teams = _team_map(df)
    before, through = _standings_before_after(df, track)
    ordered = sorted(through.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))

    report_tab, weather_tab = st.tabs(["Weekend report", "Weather"])

    with report_tab:
        _section_what_happened(rows, selected_race, extras)
        _section_championship(ordered, before, teams, year, selected_race)
        _section_watch_next(df, track, stints, names, ordered)

        # --- The briefing -----------------------------------------------------
        ui.section(
            "The briefing",
            "what was said before the race and what the race showed",
            hint="The first briefing was written from the rounds before the "
                 "race, so it never saw the result. The second one compares that "
                 "briefing with what actually happened on the day.",
        )
        grid = load_race_grid(year, csv_race) if csv_race else {}
        # The forecast is built from the rounds BEFORE the target race, so the
        # debrief compares the race against a forecast that never saw it.
        history = rows_before_race(df, track) if csv_race else None
        forecast = predict_race(history, grid) if (grid and history is not None
                                                   and not history.empty) else None

        with st.expander("Before the race"):
            if grid and csv_race:
                st.markdown(_cached_preview(df, grid, csv_race))
            else:
                ui.empty("No starting order available to build the briefing.")

        with st.expander("After the race"):
            st.markdown(_cached_debrief(race_df, track, forecast))
        ui.caption("The podium check is retrospective: it compares a forecast made "
                "before the race with the result that followed. The forecast is "
                "built from the rounds before this race, so it never saw the "
                "outcome.")

        # --- The full report --------------------------------------------------
        ui.section(
            "The full report",
            "the same text as a document you can keep or share",
            hint="This is the written report: the result, the championship "
                 "standings and the notes behind them. It is generated from the "
                 "public results, so every figure inside it that is worked out "
                 "rather than reported is labelled estimated.",
        )
        with st.expander("Open the report"):
            st.markdown(report_md)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download as text (.md)", data=report_md,
                file_name=f"F1_{year}_{selected_race.replace(' ', '_')}_report.md",
                mime="text/markdown",
            )
        with c2:
            document = _cached_html(report_md, _chart_html(
                _standings_figure(_points_by_driver(df), teams, year)))
            st.download_button(
                "Download as a page (.html)", data=document,
                file_name=f"F1_{year}_{selected_race.replace(' ', '_')}_report.html",
                mime="text/html",
            )
        ui.estimate_note("The report is generated from public season results. "
                      "Derived figures inside it are labelled estimated.")

        ui.steps([
            "Pick the Grand Prix you want to read about from the box at the top. "
            "The page then rebuilds itself around that race.",
            "Start with what happened: the finishing order, how many places each "
            "driver gained or lost, and how many points each of them scored.",
            "Then read what it means: the championship table shows the points "
            "before the weekend, the points after it, and who took the most out "
            "of it.",
            "What to watch next lists the tyre runs of the race and the drivers "
            "with the strongest recent form.",
            "The briefing at the bottom was written before the race, so it is a "
            "fair test of what the data said in advance.",
        ], title="How to read this page")

        ui.glossary(["race", "sprint", "qualifying", "grid", "points", "podium",
                  "stint", "pit stop", "tyre", "degradation", "estimated",
                  "confidence"])

    with weather_tab:
        _weather_tab(year, selected_race, track, race_df)


def _section_what_happened(rows, selected_race, extras):
    """The facts of the race: the finishing order and the points."""
    ui.section(
        "What happened",
        selected_race,
        hint="The result of the race: who finished where, how many places "
             "each driver gained or lost from the place they started, and how "
             "many points they scored for the championship.",
    )

    if rows is None or rows.empty:
        ui.empty(f"No result recorded for {selected_race}",
              "The report fills in once the race has been run and the "
              "results are published.")
        return

    ordered = rows.copy()
    ordered['_pos'] = pd.to_numeric(ordered['Position'], errors='coerce')
    ordered = ordered.dropna(subset=['_pos']).sort_values('_pos')

    table_rows = []
    for _, row in ordered.iterrows():
        position = _int_or_none(row.get('Position'))
        grid_position = _int_or_none(row.get('Starting Grid'))
        laps = _int_or_none(row.get('Laps'))
        points = _number_or_none(row.get('Points'))
        if grid_position is not None and position is not None:
            change = grid_position - position
            if change > 0:
                change_cell = {
                    'value': f"gained {change} place{'s' if change != 1 else ''}",
                    'tone': 'good',
                }
            elif change < 0:
                change_cell = {
                    'value': f"lost {abs(change)} place{'s' if abs(change) != 1 else ''}",
                    'tone': 'bad',
                }
            else:
                change_cell = {'value': 'finished where they started',
                               'tone': 'dim'}
        else:
            change_cell = {'value': 'starting place not recorded', 'tone': 'dim'}
        table_rows.append([
            f"P{position}" if position is not None else '—',
            row.get('Driver', ''),
            row.get('Team', ''),
            f"P{grid_position}" if grid_position is not None else '—',
            change_cell,
            str(laps) if laps is not None else '—',
            f"{points:.0f}" if points is not None else '0',
        ])
    ui.data_table(
        ['Finishing place', 'Driver', 'Team', 'Starting place',
         'Places gained or lost', 'Laps completed', 'Points scored'],
        table_rows,
    )

    # The headline numbers of the weekend.
    names = [str(row.get('Driver', '')) for _, row in ordered.head(3).iterrows()]
    winner_row = ordered.iloc[0] if len(ordered) else None
    winner = str(winner_row.get('Driver', '')) if winner_row is not None else ''
    winner_grid = _int_or_none(winner_row.get('Starting Grid')) if winner_row is not None else None

    biggest = None
    for _, row in ordered.iterrows():
        position = _int_or_none(row.get('Position'))
        grid_position = _int_or_none(row.get('Starting Grid'))
        if position is None or grid_position is None:
            continue
        gain = grid_position - position
        if gain > 0 and (biggest is None or gain > biggest[0]):
            biggest = (gain, str(row.get('Driver', '')), grid_position, position)

    tiles = [
        {'label': 'Winner', 'value': winner or '—',
         'sub': (f"started {_ordinal(winner_grid)}" if winner_grid else
                 "the race is decided"),
         'tone': 'accent'},
        {'label': 'On the podium', 'value': ', '.join(names) if names else '—',
         'sub': 'the first three finishers', 'tone': 'info'},
    ]
    if biggest:
        tiles.append({
            'label': 'Biggest climb of the race', 'value': biggest[1],
            'sub': f"gained {biggest[0]} place{'s' if biggest[0] != 1 else ''}, "
                   f"from {_ordinal(biggest[2])} to {_ordinal(biggest[3])}",
            'tone': 'good',
        })
    pit_stops = extras.get('pit_stops')
    if pit_stops:
        tiles.append({'label': 'Pit stops', 'value': str(pit_stops),
                      'sub': 'stops to change tyres across the field',
                      'tone': ''})
    ui.kpis(tiles)

    if len(names) >= 2 and biggest:
        ui.verdict(
            f"{winner} won the {selected_race} from {_ordinal(winner_grid)} "
            f"on the grid, with {names[1]} second and {names[2]} third; "
            f"{biggest[1]} gained the most places of the afternoon, "
            f"{biggest[0]} of them from {_ordinal(biggest[2])} to "
            f"{_ordinal(biggest[3])}."
        )
    elif winner:
        ui.verdict(
            f"{winner} won the {selected_race}"
            + (f" from {_ordinal(winner_grid)} on the grid" if winner_grid else "")
            + (f", with {', '.join(names[1:])} behind." if len(names) > 1 else ".")
        )
    else:
        ui.verdict(f"The result of the {selected_race} is shown above.")


def _section_championship(ordered, before, teams, year, selected_race):
    """What the race did to the championship."""
    ui.section(
        "What it means for the championship",
        "the points before the weekend and the points after it",
        hint="The championship is the total of every race and every sprint "
             "in the season, so a weekend can be worth more than the race "
             "alone when there was a sprint. The table shows the points "
             "before this weekend, the points after it, and what this "
             "weekend added.",
    )

    if not ordered:
        ui.empty("No championship table yet",
              "The standings appear once the first race of the season has "
              "been scored.")
        return

    table_rows = []
    for index, (driver, points) in enumerate(ordered[:10], start=1):
        gained = float(points) - float(before.get(driver, 0.0))
        table_rows.append([
            f"P{index}",
            driver,
            teams.get(driver, ''),
            f"{float(before.get(driver, 0.0)):.0f}",
            f"{float(points):.0f}",
            {'value': f"+{gained:.0f}" if gained > 0 else f"{gained:.0f}",
             'tone': 'good' if gained > 0 else 'dim'},
        ])
    ui.data_table(
        ['Championship position', 'Driver', 'Team', 'Points before this weekend',
         'Points after this weekend', 'Points gained this weekend'],
        table_rows,
    )

    show_plotly_chart(_standings_figure(dict(ordered), teams, year))

    leader, leader_points = ordered[0]
    runner_up, runner_points = (ordered[1] if len(ordered) > 1 else ('', 0.0))
    gap = float(leader_points) - float(runner_points)
    gained = {driver: float(points) - float(before.get(driver, 0.0))
              for driver, points in ordered}
    best_driver = max(gained, key=lambda name: gained[name]) if gained else ''
    best_gain = gained.get(best_driver, 0.0)

    if runner_up:
        ui.verdict(
            f"{leader} leads the championship on {float(leader_points):.0f} "
            f"points, {gap:.0f} clear of {runner_up}, and {best_driver} took "
            f"the most points out of this weekend ({best_gain:.0f} points)."
        )
    else:
        ui.verdict(
            f"{leader} leads the championship on {float(leader_points):.0f} "
            f"points after the {selected_race}."
        )


def _section_watch_next(df, track, stints, names, ordered):
    """What to keep an eye on after this weekend."""
    ui.section(
        "What to watch next",
        "the tyre runs of this race and the drivers with the strongest recent form",
        hint="Each run of laps on one tyre (a stint) is listed below: most "
             "drivers do three, the first on the starting tyres, a long "
             "middle one and a short one at the end. The drivers with the "
             "strongest recent form are the ones worth watching next, because "
             "the forecast counts a driver's last three results as well as "
             "the starting order and the team's pace.",
    )

    stint_list = _stint_rows(stints, names)
    if stint_list:
        ui.data_table(
            ['Driver', 'Run of laps on one tyre', 'Tyre', 'Laps on that tyre'],
            [[row['driver'], row['run'], row['compound'], row['laps']]
             for row in stint_list[:24]],
        )
        ui.caption("The runs are counted per driver, in the order they were "
                "driven, and the tyre names are the compounds used.")
    else:
        ui.empty("No tyre runs available for this race",
              "The lap-by-lap timing for this race could not be loaded, so "
              "the tyre strategy is not shown.")

    # Form: the average finishing place over the last three races, up to and
    # including this one. It is an estimate, not a fact about the future.
    order = _track_order(df)
    if track in order:
        index = order.index(track)
        through_rows = df[df['Track'].isin(order[:index + 1])]
    else:
        through_rows = df
    form = compute_driver_form(through_rows)
    leaders = [driver for driver, _ in
               sorted(form.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))[:3]]

    # How much history the form reading rests on. The thresholds are the ones
    # the app uses for every derived figure, so the level means the same
    # thing here as everywhere else.
    n_races = 0
    if 'Round' in through_rows.columns:
        n_races = int(pd.to_numeric(through_rows['Round'],
                                    errors='coerce').nunique())
    quality = data_quality(n_races)
    level = str(quality.get('level', 'insufficient'))
    level_label = CONFIDENCE_LABELS.get(level, (str(level), 'dim'))[0]

    if leaders:
        ui.confidence_tag(level)
        ui.note(
            "The drivers with the strongest recent form are "
            + ", ".join(leaders)
            + f" (estimated, from their last three results; {level_label}, "
            + f"because it rests on {n_races} races so far). Form is one of "
              "the three things the forecast is built on, so a run of good "
              "results matters for the next race.",
            kind='estimate',
        )

    if ordered:
        leader, leader_points = ordered[0]
        runner_up = ordered[1][0] if len(ordered) > 1 else ''
        gap = float(leader_points) - float(ordered[1][1]) if len(ordered) > 1 else 0.0
        gap_text = (f"the championship is {gap:.0f} points wide between "
                    f"{leader} and {runner_up}" if runner_up else
                    f"{leader} leads the championship on "
                    f"{float(leader_points):.0f} points")
    else:
        gap_text = "the championship is still open"

    if leaders:
        ui.verdict(
            f"{gap_text}, and the drivers in form are {', '.join(leaders)} "
            f"({level_label}, estimated); the tyre runs above show how the "
            f"race was actually decided."
        )
    else:
        ui.verdict(f"{gap_text}.")


page()
