# -*- coding: utf-8 -*-
"""Head-to-head: driver comparisons, teammate battles, form and the title race.

The page is written for someone who does not follow Formula 1 closely: the
hero states what the season looks like, the first tab answers the question a
newcomer asks first ("how do these two drivers compare?"), and every section
carries a plain-language hint that says what the block below it shows.

All the maths comes from ``analysis``; this file only decides what is shown
and in which order.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from shared import load_race_data, show_plotly_chart, team_colour
from config import FASTF1_CONFIG
from styles import (
    chart_height,
    fig_layout as _fig_layout,
    LABEL_FONT, TEXT_SECONDARY,
    scaled_height,
    compare_colours,
)
from analysis import (
    calculate_teammate_comparison, calculate_matchup,
    calculate_form_trend, championship_trajectory,
)
from pace import data_quality
import glossary as glossary_terms

# The shared components. The four that carry the plain-language contract of
# the page (the section, the verdict, the steps and the glossary) are called
# through the ``ui`` module; the smaller helpers are imported directly.
import ui
from ui import (
    caption, confidence_tag, data_table, empty, estimate_note,
    hero, scoreboard,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _driver_team_map(race_df):
    """Map each driver to their team in their **most recent** race."""
    if race_df is None or race_df.empty or 'Driver' not in race_df.columns \
            or 'Team' not in race_df.columns:
        return {}
    teams = race_df['Team'].replace('', pd.NA)
    return teams.groupby(race_df['Driver']).last().dropna().to_dict()


def _driver_colour(driver, team_map, year):
    """Livery colour of a driver's team for the selected season."""
    return team_colour(team_map.get(driver, ''), year)


def _race_count(race_df):
    """Races in the frame (by round when the frame has one)."""
    if race_df is None or race_df.empty:
        return 0
    if 'Round' in race_df.columns:
        return int(pd.to_numeric(race_df['Round'], errors='coerce').nunique())
    if 'Track' in race_df.columns:
        return int(race_df['Track'].nunique())
    return 0


def _standings(frame):
    """Championship points per driver (race *and* sprint), best first."""
    if frame is None or frame.empty or 'Points' not in frame.columns \
            or 'Driver' not in frame.columns:
        return pd.Series(dtype=float)
    return frame.groupby('Driver')['Points'].sum().sort_values(ascending=False)


def _leader_high(left, right):
    """Which side leads a "more is better" metric."""
    if left > right:
        return 'left'
    if right > left:
        return 'right'
    return ''


def _leader_low(left, right):
    """Which side leads a "lower is better" metric."""
    if left < right:
        return 'left'
    if right < left:
        return 'right'
    return ''


def _best_finish(positions):
    """The best (lowest) finishing position in a series of positions."""
    values = pd.to_numeric(pd.Series(list(positions)), errors='coerce').dropna()
    if values.empty:
        return None
    return int(values.min())


def _finish_cell(value, rival):
    """One finishing position in the race-by-race table.

    The better of the two finishes in a race is highlighted, so the eye finds
    the rows that decided the comparison.
    """
    if pd.isna(value):
        return {'value': 'not classified', 'tone': 'dim'}
    text = f"P{int(value)}"
    if pd.isna(rival):
        return {'value': text, 'tone': 'strong'}
    return {'value': text, 'tone': 'strong' if value < rival else ''}


# The status of a driver in a race, in words a newcomer understands.
_STATUS_WORDS = {
    'DNF': 'did not finish',
    'DNS': 'did not start',
    'DSQ': 'disqualified',
}


def _gap_index(race_df):
    """``(track, driver) -> (gap to the winner, status)`` from the race rows."""
    out = {}
    if race_df is None or race_df.empty:
        return out
    gaps = (race_df['GapToLeader'] if 'GapToLeader' in race_df.columns
            else [None] * len(race_df))
    statuses = (race_df['Status'] if 'Status' in race_df.columns
                else [''] * len(race_df))
    for track, driver, gap, status in zip(race_df['Track'], race_df['Driver'],
                                           gaps, statuses):
        out[(str(track), str(driver))] = (gap, status)
    return out


def _gap_text(gap, position, status):
    """The gap to the race winner, in plain words."""
    if position is not None and pd.notna(position) and float(position) == 1:
        return 'Winner'
    label = str(status or '').strip().upper()
    if label and label != 'FINISHED':
        return _STATUS_WORDS.get(label, label.title())
    if gap is None or pd.isna(gap):
        return 'n/a'
    try:
        value = float(gap)
    except (TypeError, ValueError):
        return str(gap)
    return f'+{value:.3f}s'


def _verdict_text(d1, d2, races, wins1, wins2, pts1, pts2, avg1, avg2):
    """One sentence: who is ahead, and on what."""
    if pts1 > pts2:
        lead, other = d1, d2
        lead_pts, other_pts = pts1, pts2
        lead_wins, other_wins = wins1, wins2
    elif pts2 > pts1:
        lead, other = d2, d1
        lead_pts, other_pts = pts2, pts1
        lead_wins, other_wins = wins2, wins1
    else:
        lead = other = ''
        lead_pts = other_pts = pts1
        lead_wins, other_wins = wins1, wins2

    if lead:
        return (f"Over the {races} races they ran together, {lead} is ahead of "
                f"{other}: {lead_pts} points to {other_pts}, and finished ahead "
                f"of the other in {lead_wins} of the {races} races against "
                f"{other_wins}.")

    if wins1 != wins2:
        lead = d1 if wins1 > wins2 else d2
        other = d2 if wins1 > wins2 else d1
        lead_wins = max(wins1, wins2)
        other_wins = min(wins1, wins2)
        return (f"Over the {races} races they ran together the points are level "
                f"({pts1} each), but {lead} finished ahead more often "
                f"({lead_wins} races to {other_wins}).")

    tail = ''
    if (not pd.isna(avg1) and not pd.isna(avg2) and avg1 != avg2):
        better = d1 if avg1 < avg2 else d2
        tail = (f" The estimated average finish separates them: {better} at "
                f"P{min(avg1, avg2):.1f} against P{max(avg1, avg2):.1f}.")
    return (f"Over the {races} races they ran together they are level: "
            f"{pts1} points each, and each finished ahead in {wins1} of the "
            f"{races} races.{tail}")


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

def page():
    year = st.session_state.get('selected_year', FASTF1_CONFIG.default_year)
    df = load_race_data(year)

    if df is None or df.empty:
        empty(f"No {year} results yet",
              "Results appear here once races are completed. Live session "
              "data is available in the Analysis Center.")
        return

    # Race sessions only for the head-to-head views (a sprint result is not a
    # race result). The championship trajectory uses the full frame, because
    # the championship counts sprint points.
    race_df = (df[df['SessionType'] == 'Race'].copy()
               if 'SessionType' in df.columns else df.copy())

    n_races = _race_count(race_df)
    n_drivers = int(race_df['Driver'].nunique()) if 'Driver' in race_df.columns else 0

    # The championship counts race AND sprint points, so the leader comes from
    # the full frame.
    standings = _standings(df)
    leader = standings.index[0] if not standings.empty else '-'
    leader_pts = int(standings.iloc[0]) if not standings.empty else 0

    hero(
        f"{year} season - {n_races} Grand Prix races run",
        "Head-to-Head",
        headline=(f"{leader} leads the championship with {leader_pts} points "
                  f"after {n_races} races."),
        subline=("Compare any two drivers over the races they ran together, "
                 "then look at teammates, form and the title race in the "
                 "other tabs. Every number comes from the official race "
                 "results."),
        stats=[
            {'label': 'Races run', 'value': str(n_races),
             'sub': 'Grand Prix results'},
            {'label': 'Drivers', 'value': str(n_drivers),
             'sub': 'in the results'},
            {'label': 'Championship leader', 'value': leader,
             'sub': f'{leader_pts} points (race + sprint)'},
        ],
    )

    # Every panel is built on every rerun: lazy tabs left the panel blank for
    # a moment while the second run arrived, which reads as a broken page.
    tabs = st.tabs(["Driver vs Driver", "Teammate battles", "Form",
                    "Championship race"])

    with tabs[0]:
        _tab_matchup(race_df, year, standings)
    with tabs[1]:
        _tab_teammates(race_df, year)
    with tabs[2]:
        _tab_form(race_df, year)
    with tabs[3]:
        _tab_trajectory(df, year)

    ui.steps([
        "Start at the top: the headline says who leads the championship and "
        "how many races the numbers cover.",
        "In 'Driver vs Driver', pick the two drivers you want to compare. "
        "Only the races they both ran are counted.",
        "Read the scoreboard row by row. The highlighted number is the one "
        "that leads the row: more wins or points is better, a lower finishing "
        "position is better.",
        "Use the chart and the table under it to see where the gap came from "
        "- one race can change a total.",
        "The other tabs widen the view: teammates (the same car), form (the "
        "last three races) and the championship race (points over the season).",
    ], title="How to read a comparison")
    ui.glossary(['race', 'sprint', 'qualifying', 'grid', 'points',
                 'rolling average', 'pace', 'estimated'])


# ---------------------------------------------------------------------------
# Tab 1: Driver vs Driver
# ---------------------------------------------------------------------------

@st.fragment
def _tab_matchup(race_df, year, standings):
    ui.section("Driver vs Driver",
               note_text="Two drivers compared",
               hint=("Pick two drivers. Only the races both of them started are "
                     "counted, and every number below comes from those race "
                     "results."))
    drivers = sorted(race_df['Driver'].dropna().unique())
    if len(drivers) < 2:
        empty("Need at least two drivers.",
              "The season needs at least two race results before two drivers "
              "can be compared.")
        return

    team_map = _driver_team_map(race_df)
    order = [d for d in standings.index if d in drivers]
    default_a = order[0] if order else drivers[0]
    default_b = order[1] if len(order) > 1 else drivers[1]

    def _option(driver):
        return f"{driver} - {team_map.get(driver, 'Unknown team')}"

    left_col, right_col = st.columns(2)
    with left_col:
        d1 = st.selectbox("First driver", drivers,
                          index=drivers.index(default_a),
                          format_func=_option, key="h2h_d1")
    with right_col:
        d2 = st.selectbox("Second driver", drivers,
                          index=drivers.index(default_b),
                          format_func=_option, key="h2h_d2")

    if d1 == d2:
        empty("Pick two different drivers.",
              "The comparison needs one driver on each side.")
        return

    result = calculate_matchup(race_df, d1, d2)
    if result is None:
        empty(f"{d1} and {d2} have no shared races in {year}.",
              "Pick two drivers who both raced in this season.")
        return

    summary = result['summary']
    details = result['details']

    races = int(summary['races_together'])
    race_w1 = int(summary['race_record']['driver1'])
    race_w2 = int(summary['race_record']['driver2'])
    grid_w1 = int(summary['quali_record']['driver1'])
    grid_w2 = int(summary['quali_record']['driver2'])
    pts1 = int(summary['points']['driver1'])
    pts2 = int(summary['points']['driver2'])
    avg1 = summary['avg_position']['driver1']
    avg2 = summary['avg_position']['driver2']

    pos1 = pd.to_numeric(details[d1], errors='coerce')
    pos2 = pd.to_numeric(details[d2], errors='coerce')
    best1 = _best_finish(pos1)
    best2 = _best_finish(pos2)
    podium1 = int((pos1 <= 3).sum())
    podium2 = int((pos2 <= 3).sum())

    team1 = team_map.get(d1, 'Unknown team')
    team2 = team_map.get(d2, 'Unknown team')

    ui.section("The scoreboard",
               note_text=f"{d1} ({team1}) vs {d2} ({team2})",
               hint=("One row per measure. The highlighted number leads that "
                     "row: more wins and more points is better, a lower "
                     "finishing position is better."))

    avg_left = 'n/a' if pd.isna(avg1) else f"P{avg1:.1f}"
    avg_right = 'n/a' if pd.isna(avg2) else f"P{avg2:.1f}"
    avg_leader = '' if pd.isna(avg1) or pd.isna(avg2) else _leader_low(avg1, avg2)

    scoreboard([
        {'label': 'Races run together', 'left': str(races), 'right': str(races),
         'leader': '', 'note': 'shared races'},
        {'label': 'Races ahead of the other', 'left': str(race_w1),
         'right': str(race_w2),
         'leader': _leader_high(race_w1, race_w2),
         'note': 'head to head, not a Grand Prix win'},
        {'label': 'Grid wins (started ahead)', 'left': str(grid_w1),
         'right': str(grid_w2), 'leader': _leader_high(grid_w1, grid_w2),
         'note': 'compared on the starting grid'},
        {'label': 'Points scored', 'left': str(pts1), 'right': str(pts2),
         'leader': _leader_high(pts1, pts2), 'note': 'in the shared races'},
        {'label': 'Average finishing position', 'left': avg_left,
         'right': avg_right, 'leader': avg_leader, 'note': 'estimated'},
        {'label': 'Best finish',
         'left': 'n/a' if best1 is None else f"P{best1}",
         'right': 'n/a' if best2 is None else f"P{best2}",
         'leader': '' if best1 is None or best2 is None else _leader_low(best1, best2)},
        {'label': 'Podiums', 'left': str(podium1), 'right': str(podium2),
         'leader': _leader_high(podium1, podium2)},
    ], d1, d2)

    quality = data_quality(races)
    estimate_note(
        "Average finishing position is estimated from the shared race results "
        f"({races} races). The 'races ahead of the other' count comes from the "
        "classified race results: it is the number of races one driver finished "
        "in front of the other. Grid wins are compared on the starting grid, "
        "because the qualifying session itself is not part of this data.")
    confidence_tag(quality['level'])
    caption("Grid wins are counted on the starting grid: the driver who "
            "started further up the grid is ahead, whichever qualifying "
            "session produced it. 'Races ahead of the other' is the head-to-head "
            "count - it is not the same as winning the Grand Prix.")

    # ------------------------------------------------------------------
    # Race by race
    # ------------------------------------------------------------------
    ui.section("Finishing position, race by race",
               note_text="Lower is better",
               hint=("One mark per driver per race. The mark nearer the top of a "
                     "race is the driver who finished better that weekend; the "
                     "line joins the two of them."))
    colour_a, colour_b = compare_colours(
        _driver_colour(d1, team_map, year),
        _driver_colour(d2, team_map, year),
    )
    tracks = list(details['Track'])
    connector_x, connector_y = [], []
    for track, p1, p2 in zip(tracks, pos1, pos2):
        if pd.isna(p1) or pd.isna(p2):
            continue
        connector_x += [track, track, None]
        connector_y += [p1, p2, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=connector_x, y=connector_y, mode='lines',
        line=dict(color=TEXT_SECONDARY, width=2),
        hoverinfo='skip', showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=tracks, y=pos1, mode='markers', name=d1,
        marker=dict(symbol='circle', size=11, color=colour_a,
                    line=dict(width=1.2, color='white')),
        hovertemplate=(f"<b>{d1}</b><br>%{{x}}<br>Finished P%{{y:.0f}}"
                       "<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(
        x=tracks, y=pos2, mode='markers', name=d2,
        marker=dict(symbol='diamond', size=11, color=colour_b,
                    line=dict(width=1.2, color='white')),
        hovertemplate=(f"<b>{d2}</b><br>%{{x}}<br>Finished P%{{y:.0f}}"
                       "<extra></extra>"),
    ))
    fig.update_layout(**_fig_layout(
        height=chart_height('feature'),
        xaxis_title=dict(text="Grand Prix", font=LABEL_FONT),
        yaxis_title=dict(text="Finishing position", font=LABEL_FONT),
    ))
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(autorange='reversed')
    show_plotly_chart(fig)
    caption("The colours follow the team livery when the two drivers drive "
            "for different teams; when they share a team the app uses two "
            "colours that can be told apart, and the two mark shapes differ "
            "as well.")

    # ------------------------------------------------------------------
    # The table
    # ------------------------------------------------------------------
    ui.section("Race by race",
               note_text=f"{races} shared races",
               hint=("Every race the two drivers ran together: where each of "
                     "them finished and how far behind the winner they were."))
    gaps = _gap_index(race_df)
    rows = []
    for track, p1, p2 in zip(tracks, pos1, pos2):
        gap1, status1 = gaps.get((str(track), str(d1)), (None, ''))
        gap2, status2 = gaps.get((str(track), str(d2)), (None, ''))
        rows.append([
            track,
            _finish_cell(p1, p2),
            _finish_cell(p2, p1),
            _gap_text(gap1, p1, status1),
            _gap_text(gap2, p2, status2),
        ])
    data_table(['Grand Prix', f'{d1} finish', f'{d2} finish',
                f'{d1} gap to winner', f'{d2} gap to winner'], rows)
    caption("The gap is the time behind the race winner; it is only "
            "available for drivers who were classified on the road.")

    ui.verdict(_verdict_text(d1, d2, races, race_w1, race_w2, pts1, pts2,
                              avg1, avg2),
               label="What this means")

    ui.glossary(['qualifying', 'grid', 'pace'])


# ---------------------------------------------------------------------------
# Tab 2: Teammate battles
# ---------------------------------------------------------------------------

@st.fragment
def _tab_teammates(race_df, year):
    ui.section("Teammate battles",
               note_text="Drivers who share a car",
               hint=("Teammates drive the same car on the same track, so their "
                     "results are the closest comparison in the sport. Every "
                     "number below compares the two drivers of a team with "
                     "each other."))
    comp = calculate_teammate_comparison(race_df)
    if comp is None or comp.empty:
        empty("No teammate pairs with shared races found.",
              "The season needs race results from at least two drivers of "
              "the same team.")
        return

    rows = []
    for _, row in comp.iterrows():
        rows.append([
            row['Team'],
            row['Driver 1'],
            row['Driver 2'],
            int(row['Races Together']),
            f"{int(row['D1 Race Wins'])} - {int(row['D2 Race Wins'])}",
            f"{int(row['Pts 1'])} - {int(row['Pts 2'])}",
        ])
    data_table(['Team', 'Driver 1', 'Driver 2', 'Races together',
                'Race wins (1st - 2nd)', 'Points (1st - 2nd)'], rows)
    caption("In the last two columns the first number belongs to the driver "
            "listed first and the second number to the other driver. A race "
            "win between teammates is a race where one of them finished ahead "
            "of the other, not a Grand Prix win.")

    ui.section("Race wins by pairing",
               note_text="Solid = first driver, faded = second",
               hint=("How often each teammate finished ahead of the other in "
                     "the races they ran together."))
    fig = go.Figure()
    for _, row in comp.iterrows():
        colour = team_colour(row['Team'], year)
        fig.add_trace(go.Bar(
            x=[int(row['D1 Race Wins'])], y=[row['Team']], orientation='h',
            name='First driver', marker_color=colour, marker_line_width=0,
            hovertemplate=(f"<b>{row['Driver 1']}</b><br>Race wins: "
                           f"{int(row['D1 Race Wins'])}<extra></extra>"),
        ))
        fig.add_trace(go.Bar(
            x=[int(row['D2 Race Wins'])], y=[row['Team']], orientation='h',
            name='Second driver', marker_color=colour, opacity=0.45,
            marker_line_width=0,
            hovertemplate=(f"<b>{row['Driver 2']}</b><br>Race wins: "
                           f"{int(row['D2 Race Wins'])}<extra></extra>"),
        ))
    fig.update_layout(**_fig_layout(
        height=scaled_height('series', len(comp), 36),
        xaxis_title=dict(text="Race wins", font=LABEL_FONT),
        barmode='group', showlegend=False,
    ))
    fig.update_xaxes(rangemode='tozero')
    fig.update_yaxes(showgrid=False)
    show_plotly_chart(fig)
    caption("Each bar is coloured with the livery of the team; the faded bar "
            "belongs to the second driver of that team. A bar counts the races "
            "where that driver finished ahead of the other driver of the same "
            "team.")


# ---------------------------------------------------------------------------
# Tab 3: Form
# ---------------------------------------------------------------------------

@st.fragment
def _tab_form(race_df, year):
    ui.section("Rolling finishing position",
               note_text="Lower is better",
               hint=glossary_terms.short('rolling average',
                                         'The average of the last three results.'))
    drivers = sorted(race_df['Driver'].dropna().unique())
    top = _standings(race_df).index.tolist()
    defaults = [d for d in top if d in drivers][:5]
    selected = st.multiselect("Drivers", drivers, default=defaults,
                              max_selections=8, key="form_drv")
    if not selected:
        empty("Pick at least one driver.",
              "The rolling chart needs one or more drivers to follow.")
        return

    result = calculate_form_trend(race_df, selected)
    form = result['form']
    if form.empty:
        empty("Not enough data for a form trend.",
              "A driver needs at least one race result to appear here.")
        return

    team_map = _driver_team_map(race_df)
    fig = go.Figure()
    for driver in selected:
        if driver not in form.columns:
            continue
        colour = _driver_colour(driver, team_map, year)
        fig.add_trace(go.Scatter(
            x=form.index, y=form[driver], name=driver, mode='lines+markers',
            line=dict(color=colour, width=3, shape='spline', smoothing=0.4),
            marker=dict(size=7, color=colour,
                        line=dict(width=1.2, color='white')),
            hovertemplate=("<b>%{fullData.name}</b><br>%{x}: %{y:.2f}"
                           "<extra></extra>"),
        ))
    fig.update_layout(**_fig_layout(
        height=chart_height('feature'),
        xaxis_title=dict(text="Grand Prix", font=LABEL_FONT),
        yaxis_title=dict(text="Average position", font=LABEL_FONT),
        hovermode='x unified',
    ))
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(autorange='reversed')
    show_plotly_chart(fig)
    caption("Each point is the average of the last three race finishes, so "
            "the line shows a run of form rather than one lucky or unlucky "
            "race. A line that falls is a driver improving.")

    positions = result['positions']
    if not positions.empty and positions.shape[0] > 1:
        ui.section("Finishing position by race",
                   note_text="Green is a win",
                   hint=("One row per driver and one column per race: the "
                         "colour shows how well the driver did, green for a win "
                         "and red for the back of the field."))
        values = pd.to_numeric(positions.values.ravel(), errors='coerce')
        maximum = values.max() if values.size else None
        zmax = max(20, int(maximum)) if maximum is not None and maximum > 0 else 20
        fig2 = go.Figure(go.Heatmap(
            z=positions.T.values, x=positions.index, y=positions.columns,
            colorscale='RdYlGn_r', zmin=1, zmax=zmax,
            hovertemplate="%{y}<br>%{x}: P%{z}<extra></extra>",
        ))
        fig2.update_layout(**_fig_layout(
            height=scaled_height('matrix', len(positions.columns)),
            xaxis_title=dict(text="Grand Prix", font=LABEL_FONT),
            yaxis_title=dict(text="Driver", font=LABEL_FONT),
        ))
        fig2.update_xaxes(tickangle=-30)
        fig2.update_yaxes(showgrid=False)
        show_plotly_chart(fig2)
        caption("The number in each cell is the position the driver finished "
                "in that race.")


# ---------------------------------------------------------------------------
# Tab 4: Championship race
# ---------------------------------------------------------------------------

@st.fragment
def _tab_trajectory(season_df, year):
    """Championship trajectory: race *and* sprint points, one step per round."""
    ui.section("Championship points trajectory",
               note_text="Race and sprint points",
               hint=("Each line is a driver's running points total after every "
                     "round. The steeper the line, the more points that driver "
                     "took."))
    drivers = sorted(season_df['Driver'].dropna().unique())
    top = _standings(season_df).index.tolist()
    defaults = [d for d in top if d in drivers][:5]
    selected = st.multiselect("Drivers", drivers, default=defaults,
                              max_selections=8, key="traj_drv")
    if not selected:
        empty("Pick at least one driver.",
              "The points chart needs one or more drivers to follow.")
        return

    traj = championship_trajectory(season_df, selected)
    if traj.empty:
        empty("Not enough data for a points trajectory.",
              "A driver needs at least one result to appear here.")
        return

    team_map = _driver_team_map(season_df)
    fig = go.Figure()
    for driver in selected:
        if driver not in traj.columns:
            continue
        colour = _driver_colour(driver, team_map, year)
        fig.add_trace(go.Scatter(
            x=traj.index, y=traj[driver], name=driver, mode='lines+markers',
            line=dict(color=colour, width=3, shape='spline', smoothing=0.4),
            marker=dict(size=8, color=colour,
                        line=dict(width=1.5, color='white')),
            hovertemplate=("<b>%{fullData.name}</b><br>%{x}: %{y:.0f} pts"
                           "<extra></extra>"),
        ))
    fig.update_layout(**_fig_layout(
        height=chart_height('feature'),
        xaxis_title=dict(text="Grand Prix", font=LABEL_FONT),
        yaxis_title=dict(text="Points in the championship", font=LABEL_FONT),
        hovermode='x unified',
    ))
    fig.update_xaxes(tickangle=-30)
    show_plotly_chart(fig)
    caption("Running championship total after each round. Points are real "
            "results; the curve is their accumulation.")


page()
