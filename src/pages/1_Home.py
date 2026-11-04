# -*- coding: utf-8 -*-
"""
1_Home.py
~~~~~~~~~
The season dashboard.

The page answers one question before anything else -- *where does the season
stand right now?* -- and then backs it up with the numbers, in this order:

1. the hero band: the story in one sentence, the championship points
   chart, plus the four numbers that matter;
2. the tiles: the same four numbers in a row;
3. "Where the title stands": the championship tables for drivers and teams;
4. "Last weekend": the finishing order of the most recent Grand Prix;
5. "The title fight": what the gap to the leader is worth;
6. "How the season has gone": the running points total and the race wins;
7. "What to watch": who has scored the most lately;
8. "How to read this dashboard": the steps and the words.

Every number and every sentence is computed from the season results at run
time: the headline, the gap, the form line -- none of them is written into
the source. What is measured (the results) is never dressed up as a
forecast, and the one derived summary -- the recent form -- is labelled and
given a confidence level.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from ui import esc
from analysis import latest_race_rows, points_progression
from config import FASTF1_CONFIG
from glossary import short as glossary_short
from shared import load_race_data, season_progress, show_plotly_chart, team_colour
from styles import (
    ACCENT,
    LABEL_FONT,
    TEXT_SECONDARY,
    chart_height,
    fig_layout as _fig_layout,
    scaled_height,
)
import permutations as title_math
import schedule as race_schedule

logger = logging.getLogger(__name__)

# The points a Grand Prix win is worth. It comes from the points system, so
# the page can say what one race is worth when it explains a gap.
WIN_POINTS = 25

# How many rows the championship table shows before the rest is summarised.
DRIVERS_SHOWN = 10

# How many recent rounds the "form" summary looks at, and how many rows the
# form table shows.
FORM_ROUNDS = 3
FORM_SHOWN = 6


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _driver_team_map(df):
    """Each driver's team in their **most recent** race.

    A mid-season team change must not keep the old team's colour for the rest
    of the season.
    """
    if df is None or df.empty or 'Driver' not in df.columns or 'Team' not in df.columns:
        return {}
    teams = df['Team'].replace('', pd.NA)
    return teams.groupby(df['Driver']).last().dropna().to_dict()


def _fmt(value) -> str:
    """A points total as a whole number with a thousands separator."""
    try:
        return f'{int(round(float(value))):,}'
    except (TypeError, ValueError):
        return '0'


def _plural(count: int, singular: str, plural: str = '') -> str:
    """``1 race`` / ``6 races`` -- the number and its noun."""
    word = singular if abs(count) == 1 else (plural or f'{singular}s')
    return f'{count} {word}'


def _headline(leader, leader_pts, second, gap, races_completed, races_left,
              calendar_known) -> str:
    """The one sentence that says where the title stands.

    It is built from the championship table itself, so it can never drift
    away from the numbers on the page.
    """
    if not leader or leader == '-':
        return 'No championship points have been scored yet.'
    if second is None:
        return f'{leader} leads the championship with {_fmt(leader_pts)} points.'
    if gap == 0:
        tail = f' after {races_completed} races.' if races_completed else '.'
        return f'{leader} and {second} are level on {_fmt(leader_pts)} points{tail}'
    if calendar_known and races_left:
        tail = f' with {_plural(races_left, "race")} left'
    else:
        tail = f' after {races_completed} races'
    return f'{leader} leads the championship by {gap} points{tail}.'


def _title_fight(leader, gap, second, races_left, calendar_known) -> str:
    """What the gap to the leader means, in plain words."""
    if second is None:
        return f'{leader} is the only driver on the points table so far.'
    if gap == 0:
        return (f'{leader} and {second} are level on points. A single race win is '
                f'worth {WIN_POINTS} points, so one race could decide the title.')
    if gap < WIN_POINTS:
        text = (f'{leader} is {gap} points ahead of {second} - less than one '
                f'race win ({WIN_POINTS} points).')
    else:
        text = (f'{leader} is {gap} points ahead of {second} - about '
                f'{gap // WIN_POINTS} race wins.')
    if calendar_known and races_left:
        text += f' There are {_plural(races_left, "race")} left in the calendar.'
    return text


def _form_confidence(rounds_used: int) -> str:
    """How much to trust the recent-form summary.

    The confidence vocabulary is the app's own -- the glossary explains it:
    the bigger the sample, the steadier the read. Here the sample is the
    number of rounds that were counted, so the shared thresholds apply.
    """
    if rounds_used >= 15:
        return 'good'
    if rounds_used >= 8:
        return 'moderate'
    if rounds_used >= 2:
        return 'low'
    return 'insufficient'


def _margin_text(value) -> str:
    """The gap between the winner and the car behind, as a sentence."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 'not reported by the timing feed'
    if isinstance(value, (int, float)):
        return f'{float(value):.3f} seconds'
    return str(value)


# ---------------------------------------------------------------------------
# The championship tables
# ---------------------------------------------------------------------------

def _drivers_table(drv_points, wins, podiums, driver_teams):
    """The drivers' championship as rows for ``ui.data_table``."""
    rows = []
    for position, (driver, points) in enumerate(
            drv_points.head(DRIVERS_SHOWN).items(), start=1):
        leader = position == 1
        rows.append([
            str(position),
            {'value': driver, 'tone': 'strong'} if leader else driver,
            driver_teams.get(driver, ''),
            {'value': _fmt(points), 'tone': 'strong'} if leader else _fmt(points),
            str(int(wins.get(driver, 0))),
            str(int(podiums.get(driver, 0))),
        ])
    return rows


# ---------------------------------------------------------------------------
# The charts
# ---------------------------------------------------------------------------

def _progression_chart(df, drivers, driver_teams, year):
    """The running championship total of the leading drivers.

    The curve is stepped per round (a weekend counts once, race and sprint
    together), so the x axis is the round of the calendar and the tick label
    is the Grand Prix that was run.
    """
    progression = points_progression(df, drivers=list(drivers))
    if progression.empty:
        return None
    x_col = 'Round' if 'Round' in progression.columns else 'RaceNum'

    # The points each driver scored in each round (race and sprint together),
    # for the hover label.
    round_points = {}
    if 'Round' in df.columns:
        scored = df.groupby(['Driver', 'Round'])['Points'].sum()
        round_points = {key: int(value) for key, value in scored.items()}

    # Which Grand Prix belongs to which round.
    race_rows = (df[df['SessionType'] == 'Race'] if 'SessionType' in df.columns
                 else df)
    round_track = {}
    if 'Round' in race_rows.columns and 'Track' in race_rows.columns:
        for rnd, track in race_rows.groupby('Round')['Track'].first().items():
            round_track[int(rnd)] = str(track)

    fig = go.Figure()
    for driver in drivers:
        rows = progression[progression['Driver'] == driver].sort_values(x_col)
        if rows.empty:
            continue
        colour = team_colour(driver_teams.get(driver, ''), year)
        custom = []
        for _, row in rows.iterrows():
            rnd = row.get('Round') if 'Round' in rows.columns else None
            custom.append([str(row.get('Track', '')),
                           round_points.get((driver, rnd), 0)])
        fig.add_trace(go.Scatter(
            x=rows[x_col], y=rows['Points'], mode='lines+markers',
            name=driver, line=dict(color=colour, width=2.5),
            marker=dict(size=7, color=colour,
                        line=dict(width=1, color='rgba(255,255,255,0.6)')),
            hovertemplate=("<b>%{customdata[0]}</b><br>"
                           "%{customdata[1]} points that weekend<br>"
                           "Total %{y} points<extra></extra>"),
            customdata=custom,
        ))
    if not fig.data:
        return None

    if x_col == 'Round' and round_track:
        tick_vals = sorted(round_track)
        tick_text = [round_track[rnd][:16] for rnd in tick_vals]
    else:
        max_races = int(progression.groupby('Driver').size().max())
        tick_vals = list(range(1, max_races + 1))
        tick_text = [round_track.get(rnd, str(rnd))[:16]
                     for rnd in tick_vals]

    fig.update_layout(**_fig_layout(
        height=chart_height('feature'),
        xaxis=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text,
                   tickangle=-30),
        xaxis_title=dict(text='Round', font=LABEL_FONT),
        yaxis=dict(rangemode='tozero'),
        yaxis_title=dict(text='Points', font=LABEL_FONT),
        hovermode='x unified',
        legend=dict(orientation='h', x=0, y=-0.18),
    ))
    return fig


def _wins_chart(wins, driver_teams, year):
    """How many races each driver has won."""
    if wins is None or wins.empty:
        return None
    fig = go.Figure()
    for driver, count in wins.items():
        fig.add_trace(go.Bar(
            x=[int(count)], y=[driver], orientation='h',
            marker=dict(color=team_colour(driver_teams.get(driver, ''), year),
                        line=dict(width=0)),
            text=str(int(count)), textposition='outside',
            textfont=dict(size=11, color=TEXT_SECONDARY),
            # The driver's name goes into the hover label as HTML, so it is
            # escaped like any other dynamic value.
            hovertemplate=(f'<b>{esc(driver)}</b><br>Race wins: '
                           f'{int(count)}<extra></extra>'),
        ))
    fig.update_layout(**_fig_layout(
        height=scaled_height('bars', len(wins)),
        xaxis=dict(rangemode='tozero'),
        xaxis_title=dict(text='Race wins', font=LABEL_FONT),
        yaxis=dict(autorange='reversed', showgrid=False),
        showlegend=False, bargap=0.35,
    ))
    return fig


def _points_chart(drv_points, driver_teams, year):
    """Championship points of the leading drivers, as horizontal bars.

    The hero band points at this chart: the story is the shape of the
    standings, not a paragraph, so the page opens with the bars and one
    sentence rather than a block of text. Team colours come from the
    results feed via ``team_colour`` like every other chart.
    """
    if drv_points is None or drv_points.empty:
        return None
    top = drv_points.head(8)
    fig = go.Figure()
    for driver, points in top.items():
        fig.add_trace(go.Bar(
            x=[int(points)], y=[driver], orientation='h',
            marker=dict(color=team_colour(driver_teams.get(driver, ''), year),
                        line=dict(width=0)),
            text=_fmt(points), textposition='outside',
            textfont=dict(size=11, color=TEXT_SECONDARY),
            # The driver's name goes into the hover label as HTML, so it is
            # escaped like any other dynamic value.
            hovertemplate=(f'<b>{esc(driver)}</b><br>Points: '
                           f'{_fmt(points)}<extra></extra>'),
        ))
    fig.update_layout(**_fig_layout(
        height=scaled_height('bars', len(top)),
        xaxis=dict(rangemode='tozero'),
        xaxis_title=dict(text='Points', font=LABEL_FONT),
        yaxis=dict(autorange='reversed', showgrid=False),
        showlegend=False, bargap=0.35,
    ))
    return fig


def _bars_chart(values, colours, x_title, hover_unit='points',
                ref_line=None, ref_label=''):
    """Horizontal bars for a name -> number mapping, in team colours.

    Shared by the small gap/form charts on this page: the shape of the
    numbers is the story, so they are drawn, not written out in a table.
    """
    if values is None or len(values) == 0:
        return None
    fig = go.Figure()
    for name, value in values.items():
        fig.add_trace(go.Bar(
            x=[value], y=[name], orientation='h',
            marker=dict(color=colours.get(name, TEXT_SECONDARY),
                        line=dict(width=0)),
            text=_fmt(value), textposition='outside',
            textfont=dict(size=11, color=TEXT_SECONDARY),
            hovertemplate=(f'<b>{esc(name)}</b><br>{hover_unit}: '
                           f'{_fmt(value)}<extra></extra>'),
        ))
    layout = dict(
        height=scaled_height('bars', len(values)),
        xaxis=dict(rangemode='tozero'),
        xaxis_title=dict(text=x_title, font=LABEL_FONT),
        yaxis=dict(autorange='reversed', showgrid=False),
        showlegend=False, bargap=0.35,
    )
    if ref_line is not None:
        layout['shapes'] = [dict(
            type='line', x0=ref_line, x1=ref_line, yref='paper', y0=0, y1=1,
            line=dict(color=ACCENT, width=1.5, dash='dash'),
        )]
        layout['annotations'] = [dict(
            x=ref_line, yref='paper', y=1.02, text=ref_label,
            showarrow=False, font=dict(size=10, color=TEXT_SECONDARY),
        )]
    fig.update_layout(**_fig_layout(**layout))
    return fig


def _session_timeline(weekend):
    """The weekend as a timeline: one marker per session at its start time.

    A marker (not a bar) because session durations are not published --
    the start time is the one honest position on the axis. The exact
    WIB/UTC start rides in the hover label, and the axis carries the
    day/hour so the shape of the weekend reads at a glance instead of a
    column of clock times.
    """
    sessions = (weekend or {}).get('sessions') or []
    if not sessions:
        return None
    now = datetime.now(timezone.utc)
    fig = go.Figure()
    for item in sessions:
        start = item.get('start')
        if start is None:
            continue
        times = race_schedule.session_times(start)
        is_race = str(item.get('type') or '').lower() == 'race'
        start_utc = start if start.tzinfo is not None else start.replace(
            tzinfo=timezone.utc)
        done = start_utc <= now
        # Race red, upcoming blue, already-run sessions muted.
        colour = ACCENT if is_race else ('#4388E8' if not done else '#626D7D')
        label = str(item.get('type') or '')
        fig.add_trace(go.Scatter(
            x=[start], y=[label], mode='markers+text',
            marker=dict(size=13, color=colour,
                        line=dict(width=1.5, color='rgba(255,255,255,0.6)')),
            text=[times['wib']], textposition='top center',
            textfont=dict(size=10,
                          color=TEXT_SECONDARY if done else '#F4F5F7'),
            hovertemplate=(f'<b>{esc(label)}</b><br>'
                           f'{esc(times["wib"])} WIB<br>'
                           f'{esc(times["utc"])}<extra></extra>'),
        ))
    if not fig.data:
        return None
    fig.update_layout(**_fig_layout(
        height=scaled_height('bars', len(sessions)),
        xaxis=dict(tickformat='%a %H:%M', dtick=6 * 60 * 60 * 1000),
        yaxis=dict(autorange='reversed', showgrid=False),
        showlegend=False,
    ))
    return fig


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

def _next_race_block(year):
    """Countdown and session times for the next race, or the live weekend.

    Everything comes from the calendar; when the calendar is unknown the block
    is simply not drawn rather than showing a guessed date.
    """
    weekend = race_schedule.next_weekend(year)
    if weekend is None:
        return

    now = datetime.now(timezone.utc)
    session = race_schedule.next_session(year, now)
    live = race_schedule.live_weekend(year, now)

    ui.section(
        'Next race',
        note_text=weekend['name'],
        hint=('WIB is the clock most readers of this dashboard use; '
              'UTC is shown too.'))

    if live is not None:
        ui.verdict(
            f'The {weekend["name"]} weekend is under way - results follow '
            f'after each session.',
            label='Live weekend')

    country = str(weekend.get('country') or weekend.get('location') or '')
    ui.card(
        f'<div class="f1-racecard">{ui.flag_html(country)}'
        f'<div><div class="f1-racecard-name">{esc(weekend["name"])}</div>'
        f'<div class="f1-racecard-meta">'
        f'{esc(country)} · Round {esc(weekend.get("round") or "—")}</div>'
        f'</div></div>'
    )

    if session is None:
        ui.caption('Every session on the calendar has already started, so there '
                   'is no countdown left this season.')
        return

    left = race_schedule.countdown(session['start'], now)
    times = race_schedule.session_times(session['start'])
    ui.kpis([
        {'label': 'Next session', 'icon': 'flag', 'value': session['type'],
         'sub': weekend['name'], 'tone': 'accent'},
        {'label': 'Starts in', 'value': left['label'] if left else '-',
         'sub': f'{times["wib"]} WIB', 'tone': 'info'},
        {'label': 'Round', 'value': str(weekend['round'] or '-'),
         'sub': (weekend['country'] or weekend['location'] or ''),
         'tone': 'info'},
    ])

    fig_timeline = _session_timeline(weekend)
    if fig_timeline is not None:
        show_plotly_chart(fig_timeline)
        ui.caption('Each dot is a session start, in WIB; hover for the exact '
                   'WIB and UTC time.')


def _title_permutations(drv_points, races_left, driver_teams=None, year=None):
    """Who can still win the title, and what they need - plain arithmetic."""
    rows = title_math.title_clinch_checks(drv_points, races_left)
    if not rows:
        return

    max_left = rows[0]['max_points_left']
    ui.section(
        'Can they still win the title?',
        note_text='Mathematics, not a prediction',
        hint=('The most points still available to one driver, and what each '
              'rival has to do to catch the leader.'))

    gaps = {row['driver']: row['gap_to_leader'] for row in rows[:6]}
    teams = driver_teams or {}
    gap_colours = {
        row['driver']: (team_colour(teams.get(row['driver'], ''), year)
                        if row.get('can_win') else '#6B7280')
        for row in rows[:6]
    }
    fig_can = _bars_chart(
        gaps, gap_colours, 'Points behind the leader',
        ref_line=max_left, ref_label=f'max still available: {max_left}')
    if fig_can is not None:
        show_plotly_chart(fig_can)
        ui.caption('Dashed line: the most points still available to one driver.')

    leader_row = rows[0]
    second_row = rows[1] if len(rows) > 1 else None
    if second_row is not None:
        ui.verdict(
            title_math.explain_clinch(second_row, leader_row['driver'],
                                      races_left),
            label='The closest rival')

    ui.caption(
        f'Max still available with {races_left} race'
        f'{"s" if races_left != 1 else ""} left: {max_left} points. '
        f'{title_math.ESTIMATED.capitalize()} - arithmetic, not a forecast.')


def page():
    year = st.session_state.get('selected_year', FASTF1_CONFIG.default_year)
    df = load_race_data(year)

    if df is None or df.empty:
        ui.hero(
            eyebrow=f'{year} season',
            title='Where the season stands',
            headline=f'There are no results for the {year} season yet.',
            subline='The results feed (OpenF1) publishes every race and sprint '
                    'from the 2023 season onwards. Pick another season in the '
                    'sidebar, or come back once the first race has been run.',
        )
        ui.empty(f'No {year} results yet',
                 'Championship results appear here once races are completed. '
                 'Live session data is available in the Analysis Center.')
        return

    # The same numbers the sidebar shows: races with results over the races in
    # the calendar (one helper, so the two cannot drift apart).
    races_completed, n_races = season_progress(year)
    calendar_known = n_races > 0
    races_left = max(0, n_races - races_completed)

    # The championship counts race AND sprint points, so the standings and the
    # season total come from the full frame. Race-only views (wins, the
    # latest race result) use the race slice.
    race_df = (df[df['SessionType'] == 'Race'].copy()
               if 'SessionType' in df.columns else df.copy())
    sprint_df = (df[df['SessionType'] == 'Sprint'].copy()
                 if 'SessionType' in df.columns else pd.DataFrame())

    driver_teams = _driver_team_map(df)

    drv_points = df.groupby('Driver')['Points'].sum().sort_values(ascending=False)
    team_points = df.groupby('Team')['Points'].sum().sort_values(ascending=False)
    total_points = int(drv_points.sum()) if not drv_points.empty else 0

    wins = (race_df[race_df['Position'] == 1]['Driver'].value_counts()
            if 'Position' in race_df.columns else pd.Series(dtype=int))
    podiums = (race_df[race_df['Position'].isin([1, 2, 3])]['Driver'].value_counts()
               if 'Position' in race_df.columns else pd.Series(dtype=int))
    team_wins = (race_df[race_df['Position'] == 1]['Team'].value_counts()
                 if 'Position' in race_df.columns else pd.Series(dtype=int))

    leader = drv_points.index[0] if not drv_points.empty else '-'
    leader_pts = int(drv_points.iloc[0]) if not drv_points.empty else 0
    second = drv_points.index[1] if len(drv_points) > 1 else None
    gap = leader_pts - (int(drv_points.iloc[1]) if second is not None else 0)

    team_leader = team_points.index[0] if not team_points.empty else '-'
    team_leader_pts = int(team_points.iloc[0]) if not team_points.empty else 0
    team_second = team_points.index[1] if len(team_points) > 1 else None
    team_gap = team_leader_pts - (int(team_points.iloc[1])
                                  if team_second is not None else 0)

    # --- 1. the hero band --------------------------------------------------
    # The band is one sentence; the chart underneath carries the story, so
    # no paragraph repeats the numbers the bars already show.
    headline = _headline(leader, leader_pts, second, gap, races_completed,
                         races_left, calendar_known)

    ui.hero(
        eyebrow=(f'{year} season - {races_completed} of {n_races} races run'
                 if calendar_known else f'{year} season - {races_completed} races run'),
        title='Where the season stands',
        headline=headline,
        stats=[
            {'label': 'Leader', 'value': leader,
             'sub': f'{_fmt(leader_pts)} pts'},
            {'label': 'Gap to P2', 'value': f'{gap:,} pts' if second else '-',
             'sub': second or ''},
            {'label': 'Races left',
             'value': str(races_left) if calendar_known else '-',
             'sub': f'of {n_races}' if calendar_known else ''},
        ],
    )
    # The four numbers live in the tile row below, so they are not repeated
    # inside the band.
    ui.chips([('', 'Results from OpenF1')])

    fig_points = _points_chart(drv_points, driver_teams, year)
    if fig_points is not None:
        show_plotly_chart(fig_points)
        ui.caption('Drivers\' championship points - race and sprint together.')

    # --- 2. the next race --------------------------------------------------
    _next_race_block(year)

    # --- 2. the tiles ------------------------------------------------------
    ui.kpis([
        {'label': 'Races completed', 'icon': 'flag',
         'value': f'{races_completed}/{n_races}' if calendar_known
                  else str(races_completed),
         'sub': 'Grand Prix run so far', 'tone': 'info'},
        {'label': 'Season points', 'icon': 'layers',
         'value': f'{total_points:,}', 'sub': 'race + sprint', 'tone': 'accent'},
        {'label': 'Championship leader', 'icon': 'crown', 'value': leader,
         'sub': f'{_fmt(leader_pts)} pts', 'tone': 'good'},
        {'label': 'Constructors leader', 'icon': 'trophy', 'value': team_leader,
         'sub': f'{_fmt(team_leader_pts)} pts', 'tone': 'good'},
    ])

    # --- 3. where the title stands -----------------------------------------
    ui.section('Where the title stands',
               note_text='Points include race and sprint results',
               hint='Each driver and team, ranked by total points.')

    left, right = st.columns([1.15, 1])

    with left:
        ui.section('Drivers\' championship',
                   note_text=(f'Top {min(DRIVERS_SHOWN, len(drv_points))} '
                              f'of {len(drv_points)} drivers'),
                   hint=('Where every driver stands after the races and sprints '
                         'run so far.'))
        if drv_points.empty:
            ui.empty('No driver results yet')
        else:
            ui.data_table(['#', 'Driver', 'Team', 'Points', 'Wins', 'Podiums'],
                          _drivers_table(drv_points, wins, podiums, driver_teams))
            ui.caption('Wins and podiums count Grand Prix races only; the points '
                       'column counts every race and every sprint.')
            # The team livery colours of the teams in the table, so the reader
            # can match a colour with the team it belongs to.
            shown_teams = []
            for driver in drv_points.head(DRIVERS_SHOWN).index:
                team = driver_teams.get(driver, '')
                if team and team not in shown_teams:
                    shown_teams.append(team)
            if shown_teams:
                ui.legend([(team_colour(team, year), team)
                           for team in shown_teams])
                ui.caption('Team colours are the liveries from the results feed. '
                           'The same colours are used in the charts below.')

    with right:
        ui.section('Teams\' championship', note_text=f'{len(team_points)} teams',
                   hint=('Both drivers score for their team, so the team can win '
                         'the title even when one driver scores less than a rival.'))
        if team_points.empty:
            ui.empty('No constructor results yet')
        else:
            ui.result_rows([
                {
                    'position': position,
                    'name': team,
                    'value': f'{_fmt(points)} pts',
                    'sub': (f'{int(team_wins.get(team, 0))} race wins'
                            if int(team_wins.get(team, 0)) else 'no race win yet'),
                    'colour': team_colour(team, year),
                }
                for position, (team, points) in enumerate(team_points.items(),
                                                           start=1)
            ])
            ui.caption('A team collects the points of its drivers. The dot beside '
                       'each name is the team livery colour.')

    # --- 4. last weekend ---------------------------------------------------
    latest = latest_race_rows(race_df)
    latest_track = ''
    latest_round = None
    if not latest.empty:
        if 'Track' in latest.columns:
            tracks = latest['Track'].dropna().astype(str).unique()
            latest_track = tracks[0] if len(tracks) else ''
        if 'Round' in latest.columns:
            rounds = pd.to_numeric(latest['Round'], errors='coerce').dropna()
            latest_round = int(rounds.max()) if not rounds.empty else None

    ui.section('Last weekend',
               note_text=latest_track,
               hint='The finishing order of the most recent Grand Prix. The first '
                    'three rows are the top three finishers (the podium).')

    if latest.empty:
        ui.empty('No race result to show yet',
                 'The page shows the finishing order as soon as the first race '
                 'result has been published.')
    else:
        left, right = st.columns([1.25, 1])

        with left:
            classified = latest[latest['Position'].notna()].sort_values('Position')
            rows = []
            for _, row in classified.head(10).iterrows():
                team = row.get('Team', '')
                status = row.get('Status')
                status = '' if status is None or pd.isna(status) else str(status)
                points = row.get('Points')
                rows.append({
                    'position': int(row['Position']),
                    'name': row.get('Driver'),
                    'team': team,
                    'value': f'{int(points)} pts' if pd.notna(points) else '0 pts',
                    'sub': '' if status.lower() == 'finished' else status,
                    'colour': team_colour(team, year),
                })
            ui.result_rows(rows)
            ui.caption(f'The top {min(10, len(classified))} of '
                       f'{len(classified)} classified finishers.')

        with right:
            ui.section('The race in numbers')
            tiles = []
            laps = pd.to_numeric(latest.get('Laps'), errors='coerce').max()
            winner = classified.iloc[0] if not classified.empty else None
            if winner is not None:
                tiles.append({'label': 'Winner', 'icon': 'trophy',
                              'value': str(winner.get('Driver') or '-'),
                              'sub': str(winner.get('Team', '') or ''),
                              'tone': 'accent'})
                grid = winner.get('Starting Grid')
                if pd.notna(grid):
                    tiles.append({'label': 'Started from',
                                  'value': ('pole' if int(grid) == 1
                                            else f'P{int(grid)}'),
                                  'sub': 'on the grid', 'tone': 'info'})
            runner_up = classified.iloc[1] if len(classified) > 1 else None
            if runner_up is not None:
                tiles.append({'label': 'Winning margin',
                              'value': _margin_text(
                                  runner_up.get('GapToLeader')),
                              'sub': 'to runner-up', 'tone': 'info'})
            if pd.notna(laps):
                tiles.append({'label': 'Distance',
                              'value': f'{int(laps)} laps',
                              'sub': f'round {latest_round}' if latest_round
                                     else '', 'tone': 'info'})
            if tiles:
                ui.kpis(tiles)
            sprint_rows = (sprint_df[sprint_df['Round'] == latest_round]
                           if latest_round is not None else sprint_df)
            if not sprint_rows.empty:
                sprint_winner = sprint_rows.sort_values('Position').iloc[0]
                ui.chips([('Saturday sprint',
                           f'{sprint_winner.get("Driver")} won it')])
                ui.explain('Sprint', glossary_short('sprint'))

    # --- 5. the title fight ------------------------------------------------
    ui.section('The title fight',
               hint='What the gap to the leader means in race wins.')
    if second is not None:
        gap_values = {name: int(points) for name, points in
                      list(drv_points.items())[:6]}
        leader_pts_val = int(drv_points.iloc[0])
        gap_values = {name: leader_pts_val - pts
                      for name, pts in gap_values.items()}
        gap_colours = {name: team_colour(driver_teams.get(name, ''), year)
                       for name in gap_values}
        fig_gap = _bars_chart(
            gap_values, gap_colours, 'Points behind the leader',
            ref_line=WIN_POINTS, ref_label=f'one race win: {WIN_POINTS} pts')
        if fig_gap is not None:
            show_plotly_chart(fig_gap)
    ui.verdict(_title_fight(leader, gap, second, races_left, calendar_known),
               label='The drivers\' title')
    if team_second is not None:
        ui.verdict(f'{team_leader} lead {team_second} by {team_gap} points '
                   f'({team_gap // WIN_POINTS} race wins).',
                   label='The teams\' title')

    # --- 5b. who can still win it ------------------------------------------
    _title_permutations(drv_points, races_left, driver_teams, year)

    # --- 6. how the season has gone ----------------------------------------
    ui.section('How the season has gone',
               hint='How the points have piled up round by round, and who has won '
                    'the most races.')

    known_drivers = set(df['Driver'].values) if 'Driver' in df.columns else set()
    top_drivers = [driver for driver in drv_points.head(5).index
                   if driver in known_drivers]
    if not top_drivers:
        ui.empty('Insufficient data for the progression chart')
    else:
        try:
            fig_progression = _progression_chart(df, top_drivers, driver_teams, year)
        except Exception as exc:  # pragma: no cover - a broken chart must not
            logger.warning(f'Progression unavailable: {exc}')  # take the page down
            fig_progression = None
        if fig_progression is None:
            ui.empty('Could not build the progression chart')
        else:
            show_plotly_chart(fig_progression)
            ui.caption('Each step adds that weekend\'s points: the Grand Prix '
                       'and, when there was one, the sprint.')

    if wins.empty:
        ui.empty('No race wins recorded yet')
    else:
        fig_wins = _wins_chart(wins.head(8), driver_teams, year)
        if fig_wins is None:
            ui.empty('No race wins recorded yet')
        else:
            show_plotly_chart(fig_wins)
            ui.caption('Race wins only - a sprint win is counted in the points, '
                       'not here.')

    # --- 7. what to watch --------------------------------------------------
    ui.section('What to watch',
               hint='Who has been scoring the most in the last rounds, and whether '
                    'the title is still open.')

    rounds = sorted(pd.to_numeric(df['Round'], errors='coerce').dropna().unique().tolist())
    recent_rounds = rounds[-FORM_ROUNDS:] if rounds else []
    recent = (df[df['Round'].isin(recent_rounds)]
              .groupby('Driver')['Points'].sum().sort_values(ascending=False)
              if recent_rounds else pd.Series(dtype=float))

    left, right = st.columns([1.1, 1])

    with left:
        if recent.empty or recent.iloc[0] <= 0:
            ui.verdict('No points have been scored in the recent rounds yet, so '
                       'there is no form to report.', label='Recent form')
        else:
            form_driver = recent.index[0]
            form_pts = int(recent.iloc[0])
            season_pts = int(drv_points.get(form_driver, 0))
            if form_driver == leader:
                form_text = (f'{form_driver} has scored {form_pts} points in the '
                             f'last {len(recent_rounds)} rounds - more than any '
                             f'other driver - and still leads the championship '
                             f'by {gap} points.')
            else:
                form_text = (f'{form_driver} has scored {form_pts} points in the '
                             f'last {len(recent_rounds)} rounds - more than any '
                             f'other driver - and is {leader_pts - season_pts} '
                             f'points behind {leader} in the championship.')
            ui.verdict(form_text, label='Recent form')
            level = _form_confidence(len(recent_rounds))
            ui.confidence_tag(level)
            level_name = ui.CONFIDENCE_LABELS.get(level, ('', ''))[0].lower()
            ui.estimate_note(
                f'Form, not a forecast: the last {len(recent_rounds)} rounds '
                f'only ({level_name} confidence).')

    with right:
        ui.section('Points in the recent rounds')
        if recent.empty:
            ui.empty('No points in the recent rounds yet')
        else:
            form_colours = {name: team_colour(driver_teams.get(name, ''), year)
                            for name in recent.head(FORM_SHOWN).index}
            fig_form = _bars_chart(
                {name: int(points) for name, points in
                 recent.head(FORM_SHOWN).items()},
                form_colours, 'Points in the last '
                f'{len(recent_rounds)} round{"s" if len(recent_rounds) != 1 else ""}')
            if fig_form is not None:
                show_plotly_chart(fig_form)
            ui.caption(f'Top {min(FORM_SHOWN, len(recent))} of {len(recent)} '
                       f'drivers, last {len(recent_rounds)} rounds.')

    # --- 8. how to read this dashboard -------------------------------------
    ui.section('How to read this dashboard',
               hint='Four steps if this is your first time looking at a Formula 1 '
                    'season.')
    ui.steps([
        'Start with the band at the top: it says in one sentence who is '
        'leading the championship and by how much.',
        'The tiles below it are the season in four numbers: races run, points '
        'scored, and the leaders of the drivers and the teams.',
        'The tables under "Where the title stands" are the championship '
        'standings. A driver\'s points add up from every race and every '
        'sprint of the season.',
        'Everything below that explains those numbers: the finishing order of '
        'the last race, what the gap to the leader is worth, and how the '
        'season has unfolded.',
    ], title='A four-step guide')
    ui.glossary(['points', 'race', 'sprint', 'grid', 'estimated', 'confidence'])

    _honest_positioning()


def _honest_positioning():
    """Compact, honest 'About & Limitations' block with data framing."""
    ui.section("About this tool")
    with st.expander("What F1 Lab is, and what it is not"):
        st.markdown(
            "**What this is.** F1 Lab is a public-data analytics workbench for "
            "post-session analysis, strategy rehearsal, race briefing and competitor "
            "research. It reads publicly available season results and, where requested, "
            "lap-level session data, and turns them into explainable pace, strategy and "
            "prediction views.\n\n"
            "**What it is not.** It is not a race-critical tool. It is not live pit-wall "
            "software, and it does **not** include any team's internal telemetry, setup or "
            "car information. Do not treat any readout here as a substitute for a team's "
            "operational data or for official timing.\n\n"
            "**How to read the numbers.** Anything derived rather than directly measured is "
            "labelled *estimated* and carries a confidence level. When a sample is too small, "
            "too noisy or simply missing, the app says *insufficient data* instead of "
            "fabricating an answer. Predictions and strategy recommendations are validated "
            "by historical walk-forward backtesting where possible - and their remaining "
            "assumptions and failures are shown, not hidden.\n\n"
            "**Data.** Season, sprint and qualifying results come from the "
            "[OpenF1](https://openf1.org/) API, which publishes its data under "
            "CC BY-NC-SA 4.0 (non-commercial). Lap-level session data comes from "
            "[FastF1](https://docs.fastf1.dev/)."
        )


page()
