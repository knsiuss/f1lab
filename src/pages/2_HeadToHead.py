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

from shared import load_race_data, season_progress, show_plotly_chart, team_colour
from config import FASTF1_CONFIG
from styles import (
    chart_height,
    fig_layout as _fig_layout,
    LABEL_FONT, TEXT_SECONDARY,
    scaled_height,
    compare_colours,
    DRIVER_COLORS,
    HEATMAP_COLORS,
)
from analysis import (
    calculate_driver_stats, calculate_teammate_comparison, calculate_matchup,
    calculate_form_trend,
    form_annotations, form_driver_stats, form_heatmap, form_kpis,
    form_takeaways, race_abbreviation,
)
from pace import data_quality
import championship as champ
import jolpica
from permutations import title_clinch_checks, explain_clinch
import glossary as glossary_terms

# The shared components. The four that carry the plain-language contract of
# the page (the section, the verdict, the steps and the glossary) are called
# through the ``ui`` module; the smaller helpers are imported directly.
import ui
from ui import (
    caption, confidence_tag, data_table, duel_header, empty, estimate_note,
    hero, kpis, scoreboard, takeaways,
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
    """Colour for one driver: the fixed Form palette wins, else team livery.

    The five Form drivers keep the same colour in every chart on the tab
    (teammates are not both turquoise); everyone else uses their team colour
    from the API.
    """
    key = str(driver or '').strip().lower()
    if key in DRIVER_COLORS:
        return DRIVER_COLORS[key]
    return team_colour(team_map.get(driver, ''), year)


def _driver_number_map(race_df):
    """Map each driver to their car number in their most recent race."""
    if race_df is None or race_df.empty or 'Driver' not in race_df.columns \
            or 'No' not in race_df.columns:
        return {}
    numbers = pd.to_numeric(race_df['No'], errors='coerce')
    return numbers.groupby(race_df['Driver']).last().dropna().astype(int).to_dict()


def _parse_h2h(text):
    """``"9 - 10"`` -> ``(9, 10)``; anything else -> ``(0, 0)``."""
    try:
        left, right = str(text).split('-')
        return int(left.strip()), int(right.strip())
    except (ValueError, AttributeError):
        return 0, 0


def _norm_high(value, best):
    """Higher-is-better score in 0..1 (0 when the best is 0)."""
    if best and best > 0:
        return max(0.0, min(1.0, float(value) / float(best)))
    return 0.0


def _norm_low(value, best, worst):
    """Lower-is-better score in 0..1 (best -> 1, worst -> 0, tie -> 1)."""
    if pd.isna(value) or pd.isna(best) or pd.isna(worst):
        return 0.0
    if best == worst:
        return 1.0
    return max(0.0, min(1.0, (float(worst) - float(value))
                        / (float(worst) - float(best))))


def _duel_metrics(d1, d2, summary, pos1, pos2, grid_w1, grid_w2,
                  podium1, podium2, champ_pts):
    """Normalised 0..1 scores for the radar + dumbbell (one dataset, two views).

    Raw values stay alongside: the dumbbell labels show them, so the
    normalisation never hides the real number. Lower-is-better metrics
    (average finish, finish variability) are inverted before plotting.
    """
    pts = (float(champ_pts.get(d1, 0)), float(champ_pts.get(d2, 0)))
    wins = (float(summary['race_record']['driver1']),
            float(summary['race_record']['driver2']))
    grid = (float(grid_w1), float(grid_w2))
    pods = (float(podium1), float(podium2))
    avg = (summary['avg_position']['driver1'], summary['avg_position']['driver2'])
    stds = tuple(float(pd.to_numeric(p, errors='coerce').std())
                 if pd.to_numeric(p, errors='coerce').notna().sum() > 1 else 0.0
                 for p in (pos1, pos2))
    cons = tuple(1.0 / s if s and s > 0 else 0.0 for s in stds)

    rows = [
        ('Points', pts, 'high', f"{int(pts[0])} – {int(pts[1])}"),
        ('Races ahead', wins, 'high', f"{int(wins[0])} – {int(wins[1])}"),
        ('Grid wins', grid, 'high', f"{int(grid[0])} – {int(grid[1])}"),
        ('Podiums', pods, 'high', f"{int(pods[0])} – {int(pods[1])}"),
        ('Avg finish', avg, 'low', f"P{avg[0]:.1f} – P{avg[1]:.1f}" if pd.notna(avg[0]) and pd.notna(avg[1]) else 'n/a'),
        ('Consistency', cons, 'high', f"σ{stds[0]:.1f} – σ{stds[1]:.1f}"),
    ]
    out = []
    for label, (v1, v2), kind, raw in rows:
        if kind == 'low':
            best, worst = min(v1, v2), max(v1, v2)
            s1, s2 = _norm_low(v1, best, worst), _norm_low(v2, best, worst)
        else:
            best = max(v1, v2)
            s1, s2 = _norm_high(v1, best), _norm_high(v2, best)
        out.append({'label': label, 'raw': raw, 's1': s1, 's2': s2})
    return out


def _progress_bar(done, total):
    """Championship progress bar (races completed of the calendar)."""
    if not total:
        return
    pct = max(0.0, min(100.0, 100.0 * done / total))
    st.markdown(
        '<div class="f1-progress">'
        f'<div class="f1-progress-label"><span>Championship progress</span>'
        f'<span>{done} of {total} races · {pct:.0f}%</span></div>'
        f'<div class="f1-progress-track"><div class="f1-progress-fill" '
        f'style="width:{pct:.0f}%"></div></div></div>',
        unsafe_allow_html=True,
    )


def _insight_card(title, value, sub):
    """One derived insight: title, key number, one-line context."""
    st.markdown(
        f'<div class="f1-insight"><div class="f1-insight-title">{ui.esc(title)}</div>'
        f'<div class="f1-insight-value">{ui.esc(value)}</div>'
        f'<div class="f1-insight-sub">{ui.esc(sub)}</div></div>',
        unsafe_allow_html=True,
    )


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


def _takeaways(d1, d2, races, pts1, pts2, avg1, avg2, podium1, podium2,
               best1, best2, grid_w1, grid_w2, race_w1, race_w2, wins1, wins2):
    """Key takeaways, each derived from the shared-race statistics.

    Nothing here is a fixed conclusion: every line compares the numbers it
    was handed, so the same code tells a different story for every pairing.
    """
    items = []

    if pts1 > pts2:
        lead, other, lead_pts, other_pts = d1, d2, pts1, pts2
    elif pts2 > pts1:
        lead, other, lead_pts, other_pts = d2, d1, pts2, pts1
    else:
        lead = other = ''
        lead_pts = other_pts = pts1
    if lead:
        items.append((
            "Championship lead",
            f"{lead} leads {other} by {lead_pts - other_pts} points "
            f"({lead_pts} vs {other_pts}) across {races} shared races.",
        ))
    else:
        items.append((
            "Championship lead",
            f"The pair is level on points ({pts1} each) after {races} "
            "shared races.",
        ))

    if pd.notna(avg1) and pd.notna(avg2):
        steady, other = (d1, d2) if avg1 < avg2 else (d2, d1)
        items.append((
            "Consistency",
            f"{steady} has the better average finish "
            f"(P{min(avg1, avg2):.1f} vs P{max(avg1, avg2):.1f}) and "
            f"{max(podium1, podium2)} podiums to {min(podium1, podium2)} "
            f"for {other}.",
        ))
    elif podium1 != podium2:
        rich, other = (d1, d2) if podium1 > podium2 else (d2, d1)
        items.append((
            "Consistency",
            f"{rich} collected more podiums "
            f"({max(podium1, podium2)} vs {min(podium1, podium2)}).",
        ))

    if best1 is not None and best2 is not None:
        if best1 == best2:
            items.append((
                "Peak performance",
                f"Both drivers have a best finish of P{best1}, so outright "
                "pace at the front is level.",
            ))
        else:
            peak, other = (d1, d2) if best1 < best2 else (d2, d1)
            items.append((
                "Peak performance",
                f"{peak} owns the better best finish (P{min(best1, best2)} vs "
                f"P{max(best1, best2)}).",
            ))

    if grid_w1 != grid_w2 or race_w1 != race_w2:
        grid_lead = d1 if grid_w1 > grid_w2 else d2
        race_lead = d1 if race_w1 > race_w2 else d2
        if grid_lead == race_lead:
            items.append((
                "Qualifying vs race",
                f"{race_lead} starts ahead more often "
                f"({max(grid_w1, grid_w2)} grids) and converts it into more "
                f"races ahead ({max(race_w1, race_w2)}).",
            ))
        else:
            items.append((
                "Qualifying vs race",
                f"{grid_lead} wins the grid battle "
                f"({max(grid_w1, grid_w2)} vs {min(grid_w1, grid_w2)}) but "
                f"{race_lead} finishes ahead more often "
                f"({max(race_w1, race_w2)} vs {min(race_w1, race_w2)}).",
            ))

    if wins1 != wins2:
        winner, other = (d1, d2) if wins1 > wins2 else (d2, d1)
        items.append((
            "Race-winning trend",
            f"{winner} has won {max(wins1, wins2)} of the shared races to "
            f"{other}'s {min(wins1, wins2)}.",
        ))

    return items


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

    _done, total_races = season_progress(year)
    n_races = n_races or _done

    hero(
        f"{year} season — {n_races} of {total_races} Grand Prix races run",
        "Head-to-Head Analysis",
        headline=(f"{leader} leads the championship with {leader_pts} points "
                  f"after {n_races} races."),
        subline="Driver duels, teammate battles, form and the title race — shown, not just described.",
    )
    kpis([
        {'label': 'Races run', 'value': str(n_races),
         'sub': 'Grand Prix results', 'icon': 'flag', 'tone': 'accent'},
        {'label': 'Drivers', 'value': str(n_drivers),
         'sub': 'in the results', 'icon': 'layers'},
        {'label': 'Championship leader', 'value': leader,
         'sub': f'{leader_pts} points (race + sprint)',
         'icon': 'crown', 'tone': 'good'},
        {'label': 'Season points', 'value': f"{int(standings.sum()):,}",
         'sub': 'all drivers, race + sprint', 'icon': 'trophy'},
    ])
    _progress_bar(n_races, total_races)
    ui.steps([
        'Pick a tab: two drivers, teammates, recent form, or the title race.',
        'On Championship Race, choose the drivers to follow and the '
        'points source (race, sprint or both).',
        'Read the charts top to bottom: the trajectory first, then the '
        'per-race, position and gap charts, and finally the scenarios.',
        'Every figure is calculated from the season results - nothing '
        'is projected or hard-coded.',
    ], title='How to read this page')

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

    ui.glossary(['race', 'grid', 'points'])


# ---------------------------------------------------------------------------
# Tab 1: Driver vs Driver
# ---------------------------------------------------------------------------

@st.fragment
def _tab_matchup(race_df, year, standings):
    ui.section("Driver vs Driver",
               note_text="Two drivers compared",
               hint="Only the races both started are counted.")
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
    colour_a, colour_b = compare_colours(
        _driver_colour(d1, team_map, year),
        _driver_colour(d2, team_map, year),
    )
    number_map = _driver_number_map(race_df)
    champ_rank = {name: i + 1 for i, name in enumerate(standings.index)}

    def _tile_team(driver, team):
        no = number_map.get(driver)
        rank = champ_rank.get(driver)
        bits = [f"#{no}" if no else '', team,
                f"P{rank}" if rank else '']
        return ' · '.join(bit for bit in bits if bit)

    # Visual first: the duel tiles, then the share donuts. Text only says
    # what the picture cannot.
    duel_header(
        d1, _tile_team(d1, team1), f"{pts1} pts", f"{race_w1} races ahead · P{avg1:.1f} avg" if pd.notna(avg1) else f"{race_w1} races ahead",
        d2, _tile_team(d2, team2), f"{pts2} pts", f"{race_w2} races ahead · P{avg2:.1f} avg" if pd.notna(avg2) else f"{race_w2} races ahead",
        colour_a=colour_a, colour_b=colour_b,
    )
    # "Same position" is its own slice: when neither driver finished ahead,
    # the race belongs to neither, and hiding it would make the two shares
    # look bigger than they are.
    same_position = int(((pos1 == pos2) & pos1.notna()).sum())

    def _donut(title, slices, unit):
        """One share donut with counts and percentages in the legend."""
        total = sum(value for _label, value, _colour in slices)
        labels = [
            f"{label} — {value} ({100.0 * value / total:.1f}%)" if total else label
            for label, value, _colour in slices
        ]
        fig = go.Figure(go.Pie(
            labels=labels,
            values=[value for _label, value, _colour in slices],
            hole=0.62,
            marker=dict(
                colors=[colour for _label, _value, colour in slices],
                line=dict(width=2, color='#11151C'),
            ),
            textinfo='none',
            hovertemplate="<b>%{label}</b><extra></extra>",
        ))
        fig.update_layout(**_fig_layout(
            height=chart_height('tile'),
            showlegend=True,
            legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center'),
        ))
        fig.add_annotation(
            text=f"<b>{total}</b><br><sub>{unit}</sub>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color='#F3F5F8'), align='center',
        )
        ui.section(title)
        show_plotly_chart(fig)

    donut_cols = st.columns(2)
    with donut_cols[0]:
        _donut("Races ahead of the other", [
            (f"{d1} ahead", race_w1, colour_a),
            (f"{d2} ahead", race_w2, colour_b),
            ("Same position", same_position, '#6B7280'),
        ], "shared races")
    with donut_cols[1]:
        _donut("Points in shared races", [
            (d1, pts1, colour_a),
            (d2, pts2, colour_b),
        ], "points")

    # Radar + dumbbell share one normalised dataset (0..1 per metric, raw
    # values kept as labels). Lower-is-better metrics are already inverted.
    champ_pts = standings.to_dict()
    metrics = _duel_metrics(d1, d2, summary, pos1, pos2, grid_w1, grid_w2,
                            podium1, podium2, champ_pts)
    radar_col, dumb_col = st.columns(2)
    with radar_col:
        ui.section("Performance radar",
                   note_text="Normalised 0–100",
                   hint="Each axis is scaled to the better of the two — shape beats size.")
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[m['s1'] * 100 for m in metrics] + [metrics[0]['s1'] * 100],
            theta=[m['label'] for m in metrics] + [metrics[0]['label']],
            fill='toself', name=d1, line=dict(color=colour_a, width=2),
            hovertemplate=f"<b>{d1}</b><br>%{{theta}}: %{{r:.0f}}<extra></extra>",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[m['s2'] * 100 for m in metrics] + [metrics[0]['s2'] * 100],
            theta=[m['label'] for m in metrics] + [metrics[0]['label']],
            fill='toself', name=d2, line=dict(color=colour_b, width=2),
            hovertemplate=f"<b>{d2}</b><br>%{{theta}}: %{{r:.0f}}<extra></extra>",
        ))
        fig_radar.update_layout(**_fig_layout(
            height=chart_height('row'),
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100],
                                tickfont=dict(size=9, color=TEXT_SECONDARY),
                                gridcolor='rgba(255,255,255,0.10)'),
                angularaxis=dict(tickfont=dict(size=11, color='#E8EAF0'),
                                 gridcolor='rgba(255,255,255,0.10)'),
            ),
            showlegend=True,
            legend=dict(orientation='h', y=-0.06, x=0.5, xanchor='center'),
        ))
        show_plotly_chart(fig_radar)
    with dumb_col:
        ui.section("Metric dumbbell",
                   note_text="Raw values labelled",
                   hint="One dot per driver; the dot further right leads the metric.")
        fig_dumb = go.Figure()
        for m in metrics:
            y = m['label']
            fig_dumb.add_trace(go.Scatter(
                x=[m['s1'] * 100, m['s2'] * 100], y=[y, y],
                mode='lines', line=dict(color=TEXT_SECONDARY, width=2),
                hoverinfo='skip', showlegend=False,
            ))
            fig_dumb.add_trace(go.Scatter(
                x=[m['s1'] * 100], y=[y], mode='markers+text', name=d1,
                marker=dict(size=13, color=colour_a,
                            line=dict(width=2, color='white')),
                text=[m['raw'].split(' – ')[0]], textposition='top center',
                textfont=dict(size=10, color='#E8EAF0'),
                showlegend=False,
                hovertemplate=f"<b>{d1}</b><br>{y}: %{{text}}<extra></extra>",
            ))
            fig_dumb.add_trace(go.Scatter(
                x=[m['s2'] * 100], y=[y], mode='markers+text', name=d2,
                marker=dict(size=13, color=colour_b,
                            line=dict(width=2, color='white')),
                text=[m['raw'].split(' – ')[1] if ' – ' in m['raw'] else m['raw']],
                textposition='bottom center',
                textfont=dict(size=10, color='#E8EAF0'),
                showlegend=False,
                hovertemplate=f"<b>{d2}</b><br>{y}: %{{text}}<extra></extra>",
            ))
        fig_dumb.update_layout(**_fig_layout(
            height=chart_height('row'),
            xaxis_title=dict(text="Normalised score (100 = better on that metric)",
                             font=LABEL_FONT),
            showlegend=True,
            legend=dict(orientation='h', y=1.02, x=1, xanchor='right'),
        ))
        fig_dumb.update_xaxes(range=[0, 115])
        fig_dumb.update_yaxes(showgrid=False)
        show_plotly_chart(fig_dumb)

    ui.section("The scoreboard",
               note_text=f"{d1} vs {d2}",
               hint="Bars grow from the centre · longer bar leads the row.")

    avg_left = 'n/a' if pd.isna(avg1) else f"P{avg1:.1f}"
    avg_right = 'n/a' if pd.isna(avg2) else f"P{avg2:.1f}"
    avg_leader = '' if pd.isna(avg1) or pd.isna(avg2) else _leader_low(avg1, avg2)

    scoreboard([
        {'label': 'Shared races', 'left': str(races), 'right': str(races),
         'leader': ''},
        {'label': 'Races ahead of the other', 'left': str(race_w1),
         'right': str(race_w2),
         'leader': _leader_high(race_w1, race_w2)},
        {'label': 'Grid wins', 'left': str(grid_w1),
         'right': str(grid_w2), 'leader': _leader_high(grid_w1, grid_w2)},
        {'label': 'Points', 'left': str(pts1), 'right': str(pts2),
         'leader': _leader_high(pts1, pts2)},
        {'label': 'Average finish', 'left': avg_left,
         'right': avg_right, 'leader': avg_leader, 'lower_is_better': True},
        {'label': 'Best finish',
         'left': 'n/a' if best1 is None else f"P{best1}",
         'right': 'n/a' if best2 is None else f"P{best2}",
         'leader': '' if best1 is None or best2 is None else _leader_low(best1, best2),
         'lower_is_better': True},
        {'label': 'Podiums', 'left': str(podium1), 'right': str(podium2),
         'leader': _leader_high(podium1, podium2)},
    ], d1, d2, colour_a=colour_a, colour_b=colour_b)

    quality = data_quality(races)
    estimate_note(
        f"Avg. finish is estimated from {races} shared races.")
    confidence_tag(quality['level'])

    # ------------------------------------------------------------------
    # Race by race
    # ------------------------------------------------------------------
    ui.section("Finishing position, race by race",
               note_text="Lower is better · green band = podium",
               hint="Each mark is one race; the line joins the pair.")
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
        marker=dict(symbol='circle', size=14, color=colour_a,
                    line=dict(width=2, color='white')),
        hovertemplate=(f"<b>{d1}</b><br>%{{x}}<br>Finished P%{{y:.0f}}"
                       "<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(
        x=tracks, y=pos2, mode='markers', name=d2,
        marker=dict(symbol='diamond', size=14, color=colour_b,
                    line=dict(width=2, color='white')),
        hovertemplate=(f"<b>{d2}</b><br>%{{x}}<br>Finished P%{{y:.0f}}"
                       "<extra></extra>"),
    ))
    fig.update_layout(**_fig_layout(
        height=chart_height('feature'),
        xaxis_title=dict(text="Grand Prix", font=LABEL_FONT),
        yaxis_title=dict(text="Finishing position", font=LABEL_FONT),
        shapes=[dict(type='rect', xref='paper', yref='y', x0=0, x1=1,
                     y0=0.5, y1=3.5, fillcolor='rgba(62,207,142,0.10)',
                     line_width=0, layer='below')],
        legend=dict(orientation='h', y=1.02, x=1, xanchor='right'),
    ))
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(autorange='reversed')
    show_plotly_chart(fig)

    # Cumulative points gap: the area between the two lines is the whole
    # story of who pulled away and when - no paragraph needed.
    cum1 = pd.to_numeric(details[f'{d1} Pts'], errors='coerce').fillna(0).cumsum()
    cum2 = pd.to_numeric(details[f'{d2} Pts'], errors='coerce').fillna(0).cumsum()
    fig_gap = go.Figure()
    fig_gap.add_trace(go.Scatter(
        x=tracks, y=cum1, mode='lines+markers', name=d1,
        line=dict(color=colour_a, width=3), fill='tonexty',
        fillcolor='rgba(255,255,255,0.06)',
        marker=dict(size=7, color=colour_a, line=dict(width=1.2, color='white')),
        hovertemplate=f"<b>{d1}</b><br>%{{x}}: %{{y:.0f}} pts<extra></extra>",
    ))
    fig_gap.add_trace(go.Scatter(
        x=tracks, y=cum2, mode='lines+markers', name=d2,
        line=dict(color=colour_b, width=3),
        marker=dict(size=7, color=colour_b, line=dict(width=1.2, color='white')),
        hovertemplate=f"<b>{d2}</b><br>%{{x}}: %{{y:.0f}} pts<extra></extra>",
    ))
    fig_gap.update_layout(**_fig_layout(
        height=chart_height('row'),
        xaxis_title=dict(text="Grand Prix", font=LABEL_FONT),
        yaxis_title=dict(text="Cumulative points (shared races)", font=LABEL_FONT),
        hovermode='x unified',
    ))
    fig_gap.update_xaxes(tickangle=-30)
    ui.section("Points gap, race by race",
               note_text=f"{d1} vs {d2}",
               hint="The wider the gap, the further ahead that driver is.")
    show_plotly_chart(fig_gap)

    # Finishing-position distribution: the box shows the typical finish and
    # the spread -- a driver with a tight box is consistent, a long box is
    # feast-or-famine, even at the same average.
    ui.section("Finishing position distribution",
               note_text="Box = typical range · line = median",
               hint="Lower is better. A tighter box means a more consistent driver.")
    fig_box = go.Figure()
    for driver, pos, colour in ((d1, pos1, colour_a), (d2, pos2, colour_b)):
        fig_box.add_trace(go.Box(
            y=pd.to_numeric(pos, errors='coerce').dropna(), name=driver,
            marker_color=colour, line_color=colour, boxpoints='all',
            jitter=0.3, pointpos=0, hovertemplate=f"<b>{driver}</b><br>P%{{y:.0f}}<extra></extra>",
        ))
    fig_box.update_layout(**_fig_layout(
        height=chart_height('row'),
        yaxis_title=dict(text="Finishing position", font=LABEL_FONT),
        showlegend=False,
    ))
    fig_box.update_yaxes(autorange='reversed')
    show_plotly_chart(fig_box)

    # ------------------------------------------------------------------
    # The table
    # ------------------------------------------------------------------
    ui.section("Race by race",
               note_text=f"{races} shared races",
               hint="Winner of each race is highlighted.")
    gaps = _gap_index(race_df)
    round_map = (race_df.groupby('Track')['Round'].first().to_dict()
                 if 'Round' in race_df.columns else {})
    winner_map = {}
    if 'Position' in race_df.columns:
        winners = race_df[pd.to_numeric(race_df['Position'],
                                        errors='coerce') == 1]
        winner_map = winners.set_index('Track')['Driver'].to_dict()
    rows = []
    for index, (track, p1, p2) in enumerate(zip(tracks, pos1, pos2), start=1):
        gap1, status1 = gaps.get((str(track), str(d1)), (None, ''))
        gap2, status2 = gaps.get((str(track), str(d2)), (None, ''))
        race_no = round_map.get(track, index)
        try:
            race_no = int(race_no)
        except (TypeError, ValueError):
            race_no = index
        winner = winner_map.get(track)
        winner_cell = ({'value': str(winner),
                        'tone': 'strong' if winner in (d1, d2) else ''}
                       if winner else {'value': '—', 'tone': 'dim'})
        rows.append([
            f"{race_no} · {track}",
            _finish_cell(p1, p2),
            _finish_cell(p2, p1),
            _gap_text(gap1, p1, status1),
            _gap_text(gap2, p2, status2),
            winner_cell,
        ])
    data_table(['Grand Prix', f'{d1} finish', f'{d2} finish',
                f'{d1} gap to winner', f'{d2} gap to winner', 'Winner'], rows)

    wins1 = int((pos1 == 1).sum())
    wins2 = int((pos2 == 1).sum())
    takeaway_col, verdict_col = st.columns(2)
    with takeaway_col:
        ui.section("Key Takeaways", note_text=f"{d1} vs {d2}")
        takeaways(_takeaways(
            d1, d2, races, pts1, pts2, avg1, avg2, podium1, podium2,
            best1, best2, grid_w1, grid_w2, race_w1, race_w2, wins1, wins2,
        ))
    with verdict_col:
        ui.verdict(_verdict_text(d1, d2, races, race_w1, race_w2, pts1, pts2,
                                 avg1, avg2),
                   label="What This Means")

    ui.glossary(['grid'])


# ---------------------------------------------------------------------------
# Tab 2: Teammate battles
# ---------------------------------------------------------------------------

@st.fragment
def _tab_teammates(race_df, year):
    ui.section("Teammate battles",
               note_text="Same car, same track",
               hint="The closest comparison in the sport.")
    comp = calculate_teammate_comparison(race_df)
    if comp is None or comp.empty:
        empty("No teammate pairs with shared races found.",
              "The season needs race results from at least two drivers of "
              "the same team.")
        return
    comp = comp[comp['Races Together'] > 0].copy()
    if comp.empty:
        empty("No teammate pairs with shared races found.",
              "The season needs race results from at least two drivers of "
              "the same team.")
        return

    # Derived columns (presentation-level, from the comparison frame).
    comp['Gap'] = (comp['Pts 1'] - comp['Pts 2']).abs()
    comp['Q1'], comp['Q2'] = zip(*comp['Quali H2H'].map(_parse_h2h))
    comp = comp.sort_values('Gap', ascending=False).reset_index(drop=True)

    # --- Summary cards (reference layout): overall donut + two insights ---
    ahead1 = int((comp['D1 Race Wins'] > comp['D2 Race Wins']).sum())
    ahead2 = int((comp['D2 Race Wins'] > comp['D1 Race Wins']).sum())
    tied = int((comp['D1 Race Wins'] == comp['D2 Race Wins']).sum())
    biggest = comp.iloc[0]
    most_balanced = comp.iloc[-1]
    card_cols = st.columns(3)
    with card_cols[0]:
        ui.section("Overall teammate record", note_text=f"{len(comp)} pairings")
        fig_rec = go.Figure(go.Pie(
            labels=['Driver 1 ahead', 'Driver 2 ahead', 'Tied'],
            values=[ahead1, ahead2, tied], hole=0.62,
            marker=dict(colors=['#00D2BE', '#E10600', '#6B7280'],
                        line=dict(width=2, color='#12151C')),
            textinfo='none',
            hovertemplate="<b>%{label}</b>: %{value}<extra></extra>",
        ))
        fig_rec.update_layout(**_fig_layout(
            height=chart_height('tile'), showlegend=True,
            legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center'),
        ))
        fig_rec.add_annotation(
            text=f"<b>{len(comp)}</b><br><sub>pairings</sub>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color='#F3F5F8'), align='center')
        show_plotly_chart(fig_rec)
    with card_cols[1]:
        ui.section("Biggest teammate gap", note_text="Points")
        _insight_card("Biggest gap",
                      f"+{int(biggest['Gap'])} pts",
                      f"{biggest['Driver 1']} vs {biggest['Driver 2']} "
                      f"({biggest['Team']})")
    with card_cols[2]:
        ui.section("Most balanced pairing", note_text="Points")
        _insight_card("Closest battle",
                      f"+{int(most_balanced['Gap'])} pts",
                      f"{most_balanced['Driver 1']} vs {most_balanced['Driver 2']} "
                      f"({most_balanced['Team']})")

    # --- Battle matrix: one row per pairing, bars show the share ---------
    ui.section("Teammate battle matrix",
               note_text="Bar = share of the pair",
               hint="Longer bar leads that metric.")
    rows = []
    for _, row in comp.iterrows():
        colour = team_colour(row['Team'], year)
        w1, w2 = int(row['D1 Race Wins']), int(row['D2 Race Wins'])
        p1, p2 = int(row['Pts 1']), int(row['Pts 2'])
        wtot, ptot = w1 + w2, p1 + p2
        rows.append([
            {'value': row['Team'], 'colour': colour},
            row['Driver 1'],
            row['Driver 2'],
            int(row['Races Together']),
            {'value': f"{w1} - {w2}",
             'bar_pct': 100.0 * w1 / wtot if wtot else 0,
             'bar_colour': colour},
            {'value': f"{p1} - {p2}",
             'bar_pct': 100.0 * p1 / ptot if ptot else 0,
             'bar_colour': colour},
            {'value': f"{int(row['Q1'])} - {int(row['Q2'])}"},
            f"+{int(row['Gap'])}",
        ])
    data_table(['Team', 'Driver 1', 'Driver 2', 'Races together',
                'Race wins (1st - 2nd)', 'Points (1st - 2nd)',
                'Grid wins (1st - 2nd)', 'Points gap'], rows)

    # --- Win ratio: 100% stacked -----------------------------------------
    ui.section("Head-to-head win ratio",
               note_text="Share of races finished ahead",
               hint="100% of the races the pair ran together (ties excluded).")
    fig_ratio = go.Figure()
    for _, row in comp.iterrows():
        colour = team_colour(row['Team'], year)
        w1, w2 = int(row['D1 Race Wins']), int(row['D2 Race Wins'])
        tot = w1 + w2
        if not tot:
            continue
        fig_ratio.add_trace(go.Bar(
            x=[100.0 * w1 / tot], y=[row['Team']], orientation='h',
            name=f"{row['Driver 1']} (D1)", marker_color=colour,
            marker_line_width=0,
            hovertemplate=(f"<b>{row['Driver 1']}</b><br>{w1} of {tot} "
                           f"({100.0 * w1 / tot:.0f}%)<extra></extra>"),
        ))
        fig_ratio.add_trace(go.Bar(
            x=[100.0 * w2 / tot], y=[row['Team']], orientation='h',
            name=f"{row['Driver 2']} (D2)", marker_color=colour, opacity=0.4,
            marker_line_width=0,
            hovertemplate=(f"<b>{row['Driver 2']}</b><br>{w2} of {tot} "
                           f"({100.0 * w2 / tot:.0f}%)<extra></extra>"),
        ))
    fig_ratio.update_layout(**_fig_layout(
        height=scaled_height('series', len(comp), 30),
        xaxis_title=dict(text="Share of races finished ahead (%)",
                         font=LABEL_FONT),
        barmode='relative',
        legend=dict(orientation='h', y=1.02, x=1, xanchor='right'),
    ))
    fig_ratio.update_xaxes(range=[0, 100], ticksuffix='%')
    fig_ratio.update_yaxes(showgrid=False)
    show_plotly_chart(fig_ratio)

    # --- Race wins + points by pairing (grouped, labelled) ----------------
    pair_cols = st.columns(2)
    for col, kind in zip(pair_cols, ('wins', 'points')):
        with col:
            title = ("Race wins by pairing" if kind == 'wins'
                     else "Points by pairing")
            ui.section(title,
                       note_text="Solid = driver 1 · faded = driver 2",
                       hint="How often each teammate finished ahead." if kind == 'wins'
                       else "Total points scored by each teammate.")
            fig_pair = go.Figure()
            for _, row in comp.iterrows():
                colour = team_colour(row['Team'], year)
                v1 = int(row['D1 Race Wins']) if kind == 'wins' else int(row['Pts 1'])
                v2 = int(row['D2 Race Wins']) if kind == 'wins' else int(row['Pts 2'])
                fig_pair.add_trace(go.Bar(
                    x=[v1], y=[row['Team']], orientation='h',
                    name='Driver 1 wins' if kind == 'wins' else 'Driver 1 points',
                    marker_color=colour, marker_line_width=0, text=[v1],
                    textposition='outside', textfont=dict(size=10, color='#E8EAF0'),
                    hovertemplate=(f"<b>{row['Driver 1']}</b><br>"
                                   f"{'Race wins' if kind == 'wins' else 'Points'}: {v1}<extra></extra>"),
                ))
                fig_pair.add_trace(go.Bar(
                    x=[v2], y=[row['Team']], orientation='h',
                    name='Driver 2 wins' if kind == 'wins' else 'Driver 2 points',
                    marker_color=colour, opacity=0.45, marker_line_width=0,
                    text=[v2], textposition='outside',
                    textfont=dict(size=10, color='#E8EAF0'),
                    hovertemplate=(f"<b>{row['Driver 2']}</b><br>"
                                   f"{'Race wins' if kind == 'wins' else 'Points'}: {v2}<extra></extra>"),
                ))
            fig_pair.update_layout(**_fig_layout(
                height=scaled_height('series', len(comp), 30),
                xaxis_title=dict(text="Race wins" if kind == 'wins' else "Points",
                                 font=LABEL_FONT),
                barmode='group',
                legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center'),
            ))
            fig_pair.update_xaxes(rangemode='tozero')
            fig_pair.update_yaxes(showgrid=False)
            show_plotly_chart(fig_pair)

    # --- Qualifying head-to-head ------------------------------------------
    ui.section("Qualifying head-to-head",
               note_text="Started ahead on the grid",
               hint="On the starting grid — whichever qualifying session produced it.")
    fig_quali = go.Figure()
    for _, row in comp.iterrows():
        colour = team_colour(row['Team'], year)
        fig_quali.add_trace(go.Bar(
            x=[int(row['Q1'])], y=[row['Team']], orientation='h',
            name='Driver 1 grid wins', marker_color=colour,
            marker_line_width=0, text=[int(row['Q1'])],
            textposition='outside', textfont=dict(size=10, color='#E8EAF0'),
            hovertemplate=(f"<b>{row['Driver 1']}</b><br>Grid wins: "
                           f"{int(row['Q1'])}<extra></extra>"),
        ))
        fig_quali.add_trace(go.Bar(
            x=[int(row['Q2'])], y=[row['Team']], orientation='h',
            name='Driver 2 grid wins', marker_color=colour, opacity=0.45,
            marker_line_width=0, text=[int(row['Q2'])],
            textposition='outside', textfont=dict(size=10, color='#E8EAF0'),
            hovertemplate=(f"<b>{row['Driver 2']}</b><br>Grid wins: "
                           f"{int(row['Q2'])}<extra></extra>"),
        ))
    fig_quali.update_layout(**_fig_layout(
        height=scaled_height('series', len(comp), 30),
        xaxis_title=dict(text="Grid wins", font=LABEL_FONT),
        barmode='group',
        legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center'),
    ))
    fig_quali.update_xaxes(rangemode='tozero')
    fig_quali.update_yaxes(showgrid=False)
    show_plotly_chart(fig_quali)

    # --- Key takeaways (all derived above) ---------------------------------
    ui.section("Key takeaways", note_text="Derived from the table above")
    takeaways([
        ("Biggest teammate gap",
         f"{biggest['Driver 1']} vs {biggest['Driver 2']}: "
         f"+{int(biggest['Gap'])} points "
         f"({int(biggest['Pts 1'])} vs {int(biggest['Pts 2'])})."),
        ("Most balanced",
         f"{most_balanced['Driver 1']} vs {most_balanced['Driver 2']}: "
         f"only +{int(most_balanced['Gap'])} points apart "
         f"({int(most_balanced['Pts 1'])} vs {int(most_balanced['Pts 2'])})."),
        ("Even battle",
         f"{comp.iloc[0]['Driver 1']} vs {comp.iloc[0]['Driver 2']}: "
         f"{int(comp.iloc[0]['D1 Race Wins'])}–{int(comp.iloc[0]['D2 Race Wins'])} "
         "race wins across the shared races."),
    ])


# ---------------------------------------------------------------------------
# Tab 3: Form
# ---------------------------------------------------------------------------

@st.fragment
def _tab_form(race_df, year):
    ui.section("Rolling finishing position",
               note_text="Lower is better",
               hint=glossary_terms.short('rolling average',
                                         'The average of the last few finishes.'))
    drivers = sorted(race_df['Driver'].dropna().unique())
    top = _standings(race_df).index.tolist()
    defaults = [d for d in top if d in drivers][:5]
    selected = st.multiselect("Drivers", drivers, default=defaults,
                              max_selections=8, key="form_drv")
    if not selected:
        empty("Pick at least one driver.",
              "The rolling chart needs one or more drivers to follow.")
        return

    r3 = calculate_form_trend(race_df, selected, window=3)
    r5 = calculate_form_trend(race_df, selected, window=5)
    positions = r3['positions']
    if r3['form'].empty or r5['form'].empty or positions.empty:
        empty("Not enough data for a form trend.",
              "A driver needs at least one finished race to appear here.")
        return

    team_map = _driver_team_map(race_df)
    colours = {d: _driver_colour(d, team_map, year) for d in selected}

    mode_col, _legend_col = st.columns([2, 3])
    with mode_col:
        mode = st.segmented_control(
            "Form view", ["Rolling (3)", "Rolling (5)", "Race result"],
            default="Rolling (3)", key="form_mode",
            label_visibility="collapsed")
    view = r5['form'] if mode == "Rolling (5)" else (
        positions if mode == "Race result" else r3['form'])
    view_label = {"Rolling (3)": "Rolling average of the last 3 finishes",
                  "Rolling (5)": "Rolling average of the last 5 finishes",
                  "Race result": "Finishing position in each race"}[mode]

    fig = go.Figure()
    for driver in selected:
        if driver not in view.columns:
            continue
        hover_bits = []
        for track in view.index:
            value = view.loc[track, driver]
            actual = positions.loc[track, driver] \
                if track in positions.index and driver in positions.columns else None
            actual_text = (f"Finished P{float(actual):.0f}" if pd.notna(actual)
                           else "Did not finish")
            hover_bits.append(f"<b>{driver}</b><br>{track}<br>"
                              + (f"Average {float(value):.1f}<br>"
                                 if pd.notna(value) else "")
                              + f"{actual_text}")
        fig.add_trace(go.Scatter(
            x=list(view.index), y=view[driver], name=driver,
            mode='lines+markers', line=dict(color=colours[driver], width=2.5),
            marker=dict(size=6, color=colours[driver],
                        line=dict(width=1, color='white')),
            connectgaps=False,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_bits,
        ))
    fig.update_layout(**_fig_layout(
        height=chart_height('feature'),
        xaxis_title=dict(text="Grand Prix", font=LABEL_FONT),
        yaxis_title=dict(text="Average position", font=LABEL_FONT),
        hovermode='x unified',
        shapes=[
            dict(type='rect', xref='paper', yref='y', x0=0, x1=1,
                 y0=0.5, y1=1.5, fillcolor='rgba(0,210,190,0.08)',
                 line_width=0, layer='below'),
            dict(type='rect', xref='paper', yref='y', x0=0, x1=1,
                 y0=1.5, y1=3.5, fillcolor='rgba(62,207,142,0.06)',
                 line_width=0, layer='below'),
            dict(type='rect', xref='paper', yref='y', x0=0, x1=1,
                 y0=3.5, y1=10.5, fillcolor='rgba(255,255,255,0.02)',
                 line_width=0, layer='below'),
        ],
        legend=dict(orientation='v', y=1.0, x=1.0, xanchor='left'),
    ))
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(autorange='reversed', tickvals=[1, 2, 3, 4, 6, 8, 10,
                                                      12, 15, 20])
    show_plotly_chart(fig)
    caption(f"{view_label}. Gaps are rounds without a finish. "
            "Click a driver name in the legend to hide or show them.")

    annotations = form_annotations(race_df, selected)
    if annotations:
        rows = []
        for item in annotations:
            rows.append([item['label'],
                         {'value': item['text'], 'tone': 'strong'}])
        data_table(['Annotation', 'What it means'], rows, compact=True)

    kpis_data = form_kpis(race_df, selected)
    if kpis_data:
        cards = []
        if 'most_consistent' in kpis_data:
            c = kpis_data['most_consistent']
            sub = f"Std {c['std']:.2f} over {c['finishes']} finished races"
            if c.get('field_median') is not None:
                sub += f" · field median {c['field_median']:.2f}"
            cards.append({'label': 'Most Consistent',
                          'value': c['driver'], 'sub': sub,
                          'icon': 'activity', 'tone': 'good'})
        if 'best_recent' in kpis_data:
            b = kpis_data['best_recent']
            sub = f"Avg P{b['avg']:.1f} over last {min(5, b['finishes'])}"
            if b['trend']:
                sub += f" · {b['trend']}"
            cards.append({'label': 'Best Recent Form (Last 5)',
                          'value': b['driver'], 'sub': sub,
                          'icon': 'trophy', 'tone': 'good'})
        if 'biggest_drop' in kpis_data:
            d = kpis_data['biggest_drop']
            cards.append({'label': 'Biggest Drop',
                          'value': d['driver'],
                          'sub': f"P{d['prior']:.1f} to P{d['last']:.1f} "
                                 f"({_track_span_text(d['prev_tracks'])} to "
                                 f"{_track_span_text(d['last_tracks'])})",
                          'icon': 'trend-down', 'tone': 'warn'})
        if 'strongest_recovery' in kpis_data:
            r = kpis_data['strongest_recovery']
            cards.append({'label': 'Strongest Recovery',
                          'value': r['driver'],
                          'sub': f"P{r['prior']:.1f} to P{r['last']:.1f} "
                                 f"({_track_span_text(r['prev_tracks'])} to "
                                 f"{_track_span_text(r['last_tracks'])})",
                          'icon': 'trend-up', 'tone': 'good'})
        if cards:
            kpis(cards, columns=4)

    _form_heatmap(race_df, selected, colours)
    _form_intelligence(race_df, selected, colours)
    _form_takeaways(race_df, selected)
    _form_methodology()
    _form_distribution(race_df, selected, colours)
    _form_consistency(race_df, selected, colours)


def _form_distribution(race_df, selected, colours):
    """Finishing-position distribution (box) for the selected drivers.

    The box shows the typical finish and the spread; the jittered points are
    the individual races, so consistency reads at a glance.
    """
    ui.section("Finishing position distribution",
               note_text="Lower is better",
               hint="Box = typical range · dots = individual races.")
    fig = go.Figure()
    for driver in selected:
        values = pd.to_numeric(
            race_df.loc[race_df['Driver'] == driver, 'Position'],
            errors='coerce').dropna()
        if values.empty:
            continue
        colour = colours.get(driver, '#9AA1AE')
        fig.add_trace(go.Box(
            y=values, name=driver, marker_color=colour, line_color=colour,
            boxpoints='all', jitter=0.3, pointpos=0,
            hovertemplate=f"<b>{driver}</b><br>P%{{y:.0f}}<extra></extra>",
        ))
    if not fig.data:
        return
    fig.update_layout(**_fig_layout(
        height=scaled_height('series', len(selected), 28),
        yaxis_title=dict(text="Finishing position", font=LABEL_FONT),
        showlegend=True,
        legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center'),
    ))
    fig.update_yaxes(autorange='reversed')
    show_plotly_chart(fig)


def _form_consistency(race_df, selected, colours):
    """Average finish vs finish variability (one point per driver).

    Bottom-left is the sweet spot: fast *and* consistent. Bottom-right is
    fast but erratic; top-left is steady but slow.
    """
    ui.section("Performance consistency",
               note_text="Avg finish vs variability",
               hint="Bottom-left = fast and consistent.")
    stats = calculate_driver_stats(race_df)
    rows = []
    for driver in selected:
        if driver not in stats.index:
            continue
        values = pd.to_numeric(
            race_df.loc[race_df['Driver'] == driver, 'Position'],
            errors='coerce').dropna()
        if len(values) < 2:
            continue
        rows.append((driver, float(values.mean()), float(values.std())))
    if not rows:
        return
    fig = go.Figure()
    for driver, avg, std in rows:
        colour = colours.get(driver, '#9AA1AE')
        fig.add_trace(go.Scatter(
            x=[avg], y=[std], mode='markers+text', name=driver,
            marker=dict(size=13, color=colour,
                        line=dict(width=2, color='white')),
            text=[driver], textposition='top center',
            textfont=dict(size=10, color='#E8EAF0'),
            hovertemplate=(f"<b>{driver}</b><br>Avg finish: P{avg:.1f}"
                           f"<br>Std dev: {std:.1f}<extra></extra>"),
        ))
    fig.update_layout(**_fig_layout(
        height=chart_height('row'),
        xaxis_title=dict(text="Average finishing position (lower = faster)",
                         font=LABEL_FONT),
        yaxis_title=dict(text="Std dev of finishing position (lower = steadier)",
                         font=LABEL_FONT),
        showlegend=False,
    ))
    fig.update_xaxes(autorange='reversed')
    show_plotly_chart(fig)


def _track_span_text(tracks):
    """``AUS-CHN`` span for a track list ('' when empty)."""
    codes = [race_abbreviation(t) for t in (tracks or []) if t]
    if not codes:
        return ''
    return codes[0] if len(codes) == 1 else f"{codes[0]}-{codes[-1]}"


def _form_heatmap(race_df, selected, colours):
    ui.section("Finishing position by race",
               note_text="Green is a win",
               hint="One row per driver, one column per Grand Prix. "
                    "DNF, DNS and DSQ have their own colours and are not "
                    "counted as finishing positions.")
    sort_col, _filter_col = st.columns([2, 3])
    with sort_col:
        sort_by = st.segmented_control(
            "Sort drivers", ["Avg. Pos", "Wins", "Podiums", "Name"],
            default="Avg. Pos", key="form_hm_sort",
            label_visibility="collapsed")
    heat = form_heatmap(race_df, selected)
    if heat['codes'].empty:
        empty("Not enough data for the heatmap.",
              "The matrix needs at least one race result.")
        return
    stats = heat['stats'].copy()
    if sort_by == "Avg. Pos":
        stats = stats.sort_values(['AvgPos', 'Driver'], kind='stable')
    elif sort_by == "Wins":
        stats = stats.sort_values(['Wins', 'Driver'], ascending=[False, True],
                                  kind='stable')
    elif sort_by == "Podiums":
        stats = stats.sort_values(['Podiums', 'Driver'],
                                  ascending=[False, True], kind='stable')
    else:
        stats = stats.sort_values('Driver', kind='stable')
    ordered = list(stats['Driver'])
    codes = heat['codes'].reindex(ordered)
    text = heat['text'].reindex(ordered)
    hover_grid = heat['hover'].reindex(ordered)
    levels = [key for key, _colour in HEATMAP_COLORS]
    level_index = {key: i for i, key in enumerate(levels)}
    z = [[level_index.get(code, len(levels) - 1) for code in row]
         for row in codes.values]
    stops = []
    for i, (_key, colour) in enumerate(HEATMAP_COLORS):
        stops.append([i / len(HEATMAP_COLORS), colour])
        stops.append([(i + 1) / len(HEATMAP_COLORS), colour])
    fig = go.Figure(go.Heatmap(
        z=z, x=heat['abbr'], y=[f"{d} · {team}" for d, team in
                                zip(stats['Driver'], stats['Driver'].map(
                                    _driver_team_map(race_df)))],
        text=text.values, texttemplate="%{text}",
        hovertext=hover_grid.values,
        colorscale=stops, zmin=0, zmax=len(HEATMAP_COLORS),
        showscale=False,
        hovertemplate="%{hovertext}<extra></extra>",
        xgap=1, ygap=1,
    ))
    fig.update_layout(**_fig_layout(
        height=scaled_height('matrix', len(ordered), 30),
        xaxis_title=dict(text="Grand Prix", font=LABEL_FONT),
        yaxis_title=dict(text="Driver", font=LABEL_FONT),
    ))
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(showgrid=False, autorange='reversed')
    show_plotly_chart(fig)
    ui.legend([(colour, label) for (_key, colour), (_bkey, label, _fn)
               in zip(HEATMAP_COLORS, _heatmap_bucket_labels())])
    caption("Avg. Pos uses finished races only. "
            "Wins = P1 finishes. Podiums = P1-P3 finishes.")

    rows = []
    for _, row in stats.iterrows():
        avg = 'n/a' if pd.isna(row['AvgPos']) else f"P{row['AvgPos']:.1f}"
        rows.append([str(row['Driver']), avg, str(int(row['Wins'])),
                     str(int(row['Podiums']))])
    data_table(['Driver', 'Avg. Pos', 'Wins', 'Podiums'], rows)


def _heatmap_bucket_labels():
    from analysis import HEATMAP_BUCKETS
    return HEATMAP_BUCKETS


def _form_intelligence(race_df, selected, colours):
    ui.section("Performance Intelligence",
               note_text="Finished races only",
               hint="Consistency, podium conversion, points efficiency, "
                    "where the finishes land, and recent momentum.")
    stats = form_driver_stats(race_df, selected)
    stats = stats[stats['Finishes'] > 0]
    if stats.empty:
        empty("Not enough finished races for Performance Intelligence.",
              "These charts need finished results.")
        return
    stats = stats.sort_values('Driver', kind='stable')
    names = list(stats['Driver'])
    col = [colours.get(d, '#8993A3') for d in names]

    def bar(title, values, unit, lower=False):
        ui.section(title)
        fig = go.Figure(go.Bar(
            x=names, y=values, marker_color=col, marker_line_width=0,
            text=[f"{v:.0f}{unit}" if pd.notna(v) else "n/a"
                  for v in values],
            textposition='outside',
            hovertemplate=f"<b>%{{x}}</b><br>{title}: %{{y}}"
                          f"{unit}<extra></extra>",
        ))
        fig.update_layout(**_fig_layout(
            height=scaled_height('series', len(names), 34),
            showlegend=False,
            xaxis_title=dict(text="Driver", font=LABEL_FONT),
            yaxis_title=dict(text=title, font=LABEL_FONT),
        ))
        fig.update_xaxes(tickangle=-20)
        fig.update_yaxes(showgrid=False, autorange='reversed' if lower else True)
        show_plotly_chart(fig)

    bar("Consistency (finish std)", stats['StdPos'].tolist(), "", lower=True)
    bar("Podium conversion", stats['PodiumRate'].tolist(), "%")
    bar("Points per finished race", stats['PointsPerFinish'].tolist(), "")

    ui.section("Position distribution")
    fig = go.Figure()
    buckets = [('BucketP1', 'P1', '#0E9F8A'),
               ('BucketP2P3', 'P2-P3', '#3ECF8E'),
               ('BucketP4P10', 'P4-P10', '#7BD389'),
               ('BucketP11Plus', 'P11+', '#FF8700')]
    for column, label, colour in buckets:
        fig.add_trace(go.Bar(
            x=names, y=stats[column], name=label, marker_color=colour,
            marker_line_width=0,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>",
        ))
    fig.update_layout(**_fig_layout(
        height=scaled_height('series', len(names), 34),
        barmode='stack',
        xaxis_title=dict(text="Driver", font=LABEL_FONT),
        yaxis_title=dict(text="Finished races", font=LABEL_FONT),
    ))
    fig.update_xaxes(tickangle=-20)
    fig.update_yaxes(showgrid=False)
    show_plotly_chart(fig)

    momentum = stats.dropna(subset=['Momentum'])
    if not momentum.empty:
        ui.section("Recent momentum (last 5 vs previous 5)",
                   note_text="Left = improving",
                   hint="Negative means the recent average position is "
                        "better (lower) than the previous window.")
        fig = go.Figure(go.Bar(
            x=momentum['Driver'], y=momentum['Momentum'], orientation='h',
            marker_color=[colours.get(d, '#8993A3')
                          for d in momentum['Driver']],
            marker_line_width=0,
            text=[f"{v:+.1f}" for v in momentum['Momentum']],
            textposition='outside',
            hovertemplate=("<b>%{y}</b><br>Momentum: %{x:+.1f}<extra></extra>"),
        ))
        fig.update_layout(**_fig_layout(
            height=scaled_height('series', len(momentum), 30),
            showlegend=False,
            xaxis_title=dict(text="Average-position change", font=LABEL_FONT),
            yaxis_title=dict(text="Driver", font=LABEL_FONT),
        ))
        fig.update_xaxes(zerolinecolor='rgba(255,255,255,0.25)')
        fig.update_yaxes(showgrid=False, autorange='reversed')
        show_plotly_chart(fig)


def _form_takeaways(race_df, selected):
    ui.section("Key Takeaways",
               note_text="Calculated from finished races",
               hint="Each line is computed from the selected drivers' "
                    "results — nothing is hard-coded.")
    items = form_takeaways(race_df, selected)
    if items:
        takeaways(items)


def _form_methodology():
    with st.expander("Methodology and metric definitions"):
        st.markdown(
            "Race results only: sprint results never enter finishing-position "
            "metrics. Only finished races count toward averages — DNF, DNS "
            "and DSQ appear as gaps in the trend lines and as distinct cells "
            "in the heatmap, never as an invented position.\n\n"
            "- **Rolling average (3 / 5):** mean of the driver's last 3 (or 5) "
            "finished races, aligned to the race in which the window ends. "
            "Gaps are rounds without a finish.\n"
            "- **Most Consistent:** lowest sample standard deviation of "
            "finished positions (3+ finishes).\n"
            "- **Best Recent Form:** lowest average position over the last 5 "
            "finished races.\n"
            "- **Biggest Drop / Strongest Recovery:** last-5 average minus "
            "previous-5 average (3v3 fallback early in the season). Positive "
            "is a drop, negative is a recovery.\n"
            "- **Podium Conversion:** podiums (P1-P3) as a share of finished "
            "races. **Points Efficiency:** race points per finished race. "
            "**Momentum:** last-5 minus previous-5 average position."
        )


# ---------------------------------------------------------------------------
# Tab 4: Championship race
# ---------------------------------------------------------------------------

def _remaining_sprints(year, completed_tracks):
    """Sprint weekends still to come, from the calendar when available."""
    try:
        from season_config import get_event_schedule_cached
        schedule = get_event_schedule_cached(year)
        if schedule is None or schedule.empty:
            return 0
        events = schedule.copy()
        if events['EventDate'].dt.tz is None:
            events['EventDate'] = events['EventDate'].dt.tz_localize('UTC')
        future = events[events['EventDate'] >= pd.Timestamp.now(tz='UTC')]
        return int((future['EventFormat'] == 'sprint_weekend').sum())
    except Exception:
        return 0


@st.fragment
def _tab_trajectory(season_df, year):
    """Championship Race: trajectory plus the full analytics suite.

    Every figure is derived in ``championship`` from the same season
    frame; this function only picks drivers and draws the charts.
    """
    completed, total_races = season_progress(year)
    remaining_races = max(0, int(total_races or 0) - int(completed or 0))
    totals = champ.standings_total(season_df)
    if totals.empty:
        empty("Not enough data for a championship race.",
              "A driver needs at least one result to appear here.")
        return
    leader = str(totals.index[0])
    leader_pts = float(totals.iloc[0])
    gap_p1_p2 = float(totals.iloc[0] - totals.iloc[1]) if len(totals) > 1 else 0.0
    cum_all = champ.cumulative(season_df)
    lead_changes = champ.count_lead_changes(cum_all)
    max_available = (remaining_races * champ.MAX_POINTS_PER_GP
                     + _remaining_sprints(year, None)
                     * champ.MAX_POINTS_PER_SPRINT)

    ui.section("Championship Race",
               note_text=f"{year} season",
               hint=("How the title fight has unfolded, round by round: "
                     "who gained, who slipped, and how much is still "
                     "mathematically on offer."))
    hero(
        f"{year} season - {completed} of {total_races} Grand Prix completed",
        "Championship Race",
        headline=champ.analytical_summary(season_df),
        subline=(f"Only completed rounds are charted; the remaining "
                 f"{remaining_races} round(s) are not projected."),
        stats=[
            {'label': 'Championship leader', 'value': leader,
             'sub': f'{leader_pts:.0f} points (race + sprint)'},
            {'label': 'Points gap (P1-P2)', 'value': f'{gap_p1_p2:.0f}',
             'sub': 'points between first and second'},
            {'label': 'Races remaining', 'value': str(remaining_races),
             'sub': f'of {total_races} on the calendar'},
        ],
    )
    kpis([
        {'label': 'Championship Leader', 'value': totals.index[0],
         'sub': f'{leader_pts:.0f} points', 'icon': 'crown', 'tone': 'accent'},
        {'label': 'Points Gap', 'value': f'{gap_p1_p2:.0f}',
         'sub': 'leader to second place', 'icon': 'layers'},
        {'label': 'Races Remaining', 'value': str(remaining_races),
         'sub': f'{completed} completed', 'icon': 'flag'},
        {'label': 'Maximum Available Points', 'value': f'{max_available:.0f}',
         'sub': '26 per Grand Prix, 8 per sprint', 'icon': 'trophy',
         'tone': 'good'},
        {'label': 'Championship Lead Change', 'value': str(lead_changes),
         'sub': 'times the lead changed hands', 'icon': 'flag'},
    ])

    # --- Driver selection: presets + searchable multi-select ---------------
    meta = champ.driver_meta(season_df)
    team_map = _driver_team_map(season_df)
    drivers = [d for d in totals.index.tolist()]
    defaults = drivers[:5]

    def _set_drivers(values):
        st.session_state['traj_drv'] = list(values)[:8]

    preset_cols = st.columns(4)
    preset_defs = [
        ('Championship Top 5', defaults),
        ('Teammates', _teammate_preset(season_df, totals, meta)),
        ('Title Contenders',
         _contender_preset(season_df, remaining_races, year)),
        ('Custom Selection', None),
    ]
    for col, (label, values) in zip(preset_cols, preset_defs):
        if values is not None:
            col.button(label, key=f'traj_preset_{label}',
                       on_click=lambda v=values: _set_drivers(v))
    st.caption(
        'Presets set the selection below; use the selector for a custom '
        'comparison. Up to 8 drivers are charted.')
    col_all, col_clear = st.columns(2)
    col_all.button('Select all (up to 8)', key='traj_all',
                   on_click=lambda: _set_drivers(drivers[:8]))
    col_clear.button('Clear all', key='traj_clear',
                     on_click=lambda: _set_drivers([]))

    def _option(driver):
        info = meta.get(driver, {})
        number = f"#{info['number']}" if info.get('number') else ''
        abbrev = info.get('abbrev', '')
        team = info.get('team') or team_map.get(driver, 'Unknown team')
        return f"{abbrev} {number} {driver} - {team}".strip()

    selected = st.multiselect(
        'Drivers to compare (search by name, team or number)',
        drivers, default=defaults, max_selections=8, key='traj_drv',
        format_func=_option)
    if not selected:
        empty('Pick at least one driver.',
              'The championship charts need one or more drivers to follow.')
        return

    colours = {d: _driver_colour(d, team_map, year) for d in selected}
    ui.legend([(colours[d], d) for d in selected])

    mode_label = st.radio(
        'Points source', ['Combined (race + sprint)', 'Race only',
                          'Sprint only'], horizontal=True,
        key='traj_mode')
    mode = {'Combined (race + sprint)': 'combined', 'Race only': 'race',
            'Sprint only': 'sprint'}[mode_label]

    cum = champ.cumulative(season_df, mode, selected)
    if cum.empty:
        empty('Not enough data for a points trajectory.',
              'A driver needs at least one result to appear here.')
        return
    positions_all = champ.championship_positions(
        champ.cumulative(season_df, mode))
    per_round = champ.round_points(season_df, mode)
    labels = list(cum.columns)

    # --- Insight panel (derived, traceable) --------------------------------
    found = champ.insights(season_df, selected, champ.cumulative(season_df))
    if found:
        ui.section('Analytical insights',
                   note_text='Calculated from this season\'s results',
                   hint=('Each line is computed from the results below - '
                         'no hard-coded drivers or figures.'))
        takeaways([(item['title'], item['text']) for item in found])

    # --- Main chart: cumulative trajectory ---------------------------------
    ui.section('Championship points trajectory',
               note_text=mode_label,
               hint=('Steeper = more points taken in that round. '
                     'Hover a point for the points gained and the '
                     'championship position after that round.'))
    fig = go.Figure()
    for driver in selected:
        if driver not in cum.index:
            continue
        colour = colours[driver]
        is_leader = driver == leader
        gained, pos_after = [], []
        for label in labels:
            gained.append(per_round.loc[driver, label]
                          if driver in per_round.index
                          and label in per_round.columns else None)
            pos_after.append(
                positions_all.loc[driver, label]
                if driver in positions_all.index
                and label in positions_all.columns else None)
        fig.add_trace(go.Scatter(
            x=labels, y=cum.loc[driver], name=driver,
            mode='lines+markers',
            line=dict(color=colour, width=4 if is_leader else 2.5,
                      shape='linear'),
            marker=dict(size=8 if is_leader else 6, color=colour,
                        line=dict(width=1.5, color='white')),
            customdata=[[g, p] for g, p in zip(gained, pos_after)],
            hovertemplate=(
                f"<b>{driver}</b><br>%{{x}} Grand Prix<br>"
                "Cumulative: %{y:.0f} pts<br>"
                "Gained this round: %{customdata[0]:.0f} pts<br>"
                "Championship position: P%{customdata[1]:.0f}"
                "<extra></extra>"),
        ))
    fig.update_layout(**_fig_layout(
        height=chart_height('feature'),
        xaxis_title=dict(text='Grand Prix', font=LABEL_FONT),
        yaxis_title=dict(text='Cumulative championship points',
                         font=LABEL_FONT),
        hovermode='x unified',
    ))
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=10))
    show_plotly_chart(fig)
    caption(f'Running total after each completed round ({len(labels)} '
            f'rounds). The thickest line is the current leader, {leader}.')
    with st.expander('How to read this chart'):
        st.write(
            'Each line is one driver\'s cumulative championship total. '
            'A steep segment means a big points haul that round; a flat '
            'segment means few or no points. Hovering shows the points '
            'gained in that round and the driver\'s championship '
            'position after it. Only completed rounds are drawn - '
            'future rounds are never projected. Drag on the chart to '
            'zoom, and double-click to reset the view.')

    # --- Row: per-race points | position evolution -------------------------
    left, right = st.columns(2)
    with left:
        ui.section('Points progression per race',
                   note_text=mode_label,
                   hint='Points scored in each Grand Prix. Taller = better weekend.')
        bar = go.Figure()
        for driver in selected:
            if driver not in per_round.index:
                continue
            bar.add_trace(go.Bar(
                x=labels, y=per_round.loc[driver].fillna(0),
                name=driver, marker_color=colours[driver],
                marker_line_width=0,
                hovertemplate=(f"<b>{driver}</b><br>%{{x}}: "
                               "%{y:.0f} pts<extra></extra>"),
            ))
        bar.update_layout(**_fig_layout(
            height=chart_height('row'),
            xaxis_title=dict(text='Grand Prix', font=LABEL_FONT),
            yaxis_title=dict(text='Points in the round', font=LABEL_FONT),
            barmode='group',
        ))
        bar.update_xaxes(tickangle=-35, tickfont=dict(size=10))
        show_plotly_chart(bar)
        with st.expander('How to read this chart'):
            st.write('One bar per driver per Grand Prix, using the same '
                     'points source as the trajectory above. Switch the '
                     'points source to compare race, sprint and combined '
                     'scoring.')

    with right:
        ui.section('Championship position evolution',
                   note_text='1 = championship leader',
                   hint=('How each driver\'s championship ranking moved '
                         'after every round.'))
        bump = go.Figure()
        for driver in selected:
            if driver not in positions_all.index:
                continue
            bump.add_trace(go.Scatter(
                x=labels, y=positions_all.loc[driver], name=driver,
                mode='lines+markers',
                line=dict(color=colours[driver], width=2.5, shape='linear'),
                marker=dict(size=6, color=colours[driver],
                            line=dict(width=1, color='white')),
                hovertemplate=(f"<b>{driver}</b><br>%{{x}}: "
                               "P%{y:.0f}<extra></extra>"),
            ))
        bump.update_layout(**_fig_layout(
            height=chart_height('row'),
            xaxis_title=dict(text='Grand Prix', font=LABEL_FONT),
            yaxis_title=dict(text='Championship position', font=LABEL_FONT),
        ))
        bump.update_xaxes(tickangle=-35, tickfont=dict(size=10))
        bump.update_yaxes(autorange='reversed', dtick=1)
        show_plotly_chart(bump)
        with st.expander('How to read this chart'):
            st.write('The y-axis is inverted on purpose: P1 sits at the '
                     'top. A line climbing towards the top means the '
                     'driver gained championship places; crossing lines '
                     'show overtakes in the standings.')

    # --- Row: gap evolution | consistency ----------------------------------
    left, right = st.columns(2)
    with left:
        ui.section('Points gap evolution',
                   note_text='Gap to the reference driver',
                   hint=('Positive = ahead of the reference driver, '
                         'negative = behind, zero = level.'))
        reference = st.selectbox(
            'Reference driver', selected,
            index=selected.index(leader) if leader in selected else 0,
            key='traj_ref')
        gaps = champ.gap_vs_reference(cum, reference)
        gap_fig = go.Figure()
        for driver in selected:
            if driver == reference or driver not in gaps.index:
                continue
            gap_fig.add_trace(go.Scatter(
                x=labels, y=gaps.loc[driver], name=driver,
                mode='lines+markers',
                line=dict(color=colours[driver], width=2.5),
                marker=dict(size=6, color=colours[driver],
                            line=dict(width=1, color='white')),
                hovertemplate=(f"<b>{driver}</b> vs {reference}<br>"
                               "%{x}: %{y:+.0f} pts<extra></extra>"),
            ))
        gap_fig.update_layout(**_fig_layout(
            height=chart_height('row'),
            xaxis_title=dict(text='Grand Prix', font=LABEL_FONT),
            yaxis_title=dict(text='Points gap to reference driver',
                             font=LABEL_FONT),
        ))
        gap_fig.update_xaxes(tickangle=-35, tickfont=dict(size=10))
        show_plotly_chart(gap_fig)
        with st.expander('How to read this chart'):
            st.write('Each line is the points difference between that '
                     'driver and the reference driver after every round. '
                     'Above zero the driver leads the reference driver; '
                     'below zero they trail. The reference driver is '
                     'always level with themselves at zero.')

    with right:
        ui.section('Consistency analysis',
                   note_text='Combined points per Grand Prix',
                   hint=('Average, spread and big/results counts for '
                         'each selected driver.'))
        table = champ.consistency(season_df, selected)
        if not table.empty:
            table = table.sort_values('AvgPts', ascending=False)
            rows = []
            for _, row in table.iterrows():
                rows.append([
                    {'value': row['Driver'], 'colour': colours[row['Driver']]},
                    f"{row['AvgPts']:.1f}",
                    f"{row['MedianPts']:.1f}",
                    f"{row['StdPts']:.1f}",
                    str(int(row['Wins'])),
                    str(int(row['Podiums'])),
                    str(int(row['DNFs'])),
                ])
            data_table(['Driver', 'Avg pts', 'Median pts', 'Std dev',
                        'Wins', 'Podiums', 'DNFs'], rows)
        caption('Wins, podiums and DNFs count Grand Prix races only; '
                'points averages use combined race + sprint points '
                'per weekend the driver entered.')
        with st.expander('How to read this table'):
            st.write('A high average with a low standard deviation is a '
                     'consistent scorer; a high average with a high '
                     'deviation is a boom-or-bust driver. DNF counts '
                     'races the driver did not finish.')

    # --- Heatmap -------------------------------------------------------------
    ui.section('Race-by-race performance heatmap',
               note_text=mode_label,
               hint='Each cell is the points earned in that Grand Prix; brighter = more points.')
    heat = per_round.reindex(index=[d for d in selected
                                   if d in per_round.index])
    if not heat.empty:
        fig_heat = go.Figure(go.Heatmap(
            z=heat.values, x=list(heat.columns), y=list(heat.index),
            colorscale='Viridis', zmin=0,
            hovertemplate='%{y}<br>%{x}: %{z:.0f} pts<extra></extra>',
        ))
        fig_heat.update_layout(**_fig_layout(
            height=scaled_height('matrix', len(heat.index)),
            xaxis_title=dict(text='Grand Prix', font=LABEL_FONT),
            yaxis_title=dict(text='Driver', font=LABEL_FONT),
        ))
        fig_heat.update_xaxes(tickangle=-35, tickfont=dict(size=10))
        fig_heat.update_yaxes(showgrid=False)
        show_plotly_chart(fig_heat)
        caption('Blank cells mean the driver did not race that weekend.')
    with st.expander('How to read this chart'):
        st.write('Rows are drivers, columns are Grands Prix, and the '
                 'colour intensity is the points scored that weekend. '
                 'Hover a cell for the exact points.')

    # --- Scenarios + insights -----------------------------------------------
    ui.section('Championship scenarios',
               note_text='Mathematical analysis, not a prediction',
               hint=('Who still needs how many points, and who is '
                     'mathematically out of the title.'))
    scenario_table = champ.scenarios(
        season_df, remaining_races, _remaining_sprints(year, None))
    if not scenario_table.empty:
        rows = []
        for _, row in scenario_table.iterrows():
            driver = row['Driver']
            colour = colours.get(driver) or _driver_colour(
                driver, team_map, year)
            tone = ('good' if row['Status'] == 'Leads the championship'
                    else 'bad' if 'eliminated' in row['Status'] else '')
            rows.append([
                str(int(row['Rank'])),
                {'value': driver, 'colour': colour},
                f"{row['Points']:.0f}",
                f"{row['GapToLeader']:.0f}",
                str(int(row['PointsNeeded'])),
                {'value': row['Status'], 'tone': tone},
            ])
        data_table(['Pos', 'Driver', 'Points', 'Gap to leader',
                    'Points needed', 'Mathematical status'], rows)
    caption(f'Maximum still available: {max_available:.0f} points '
            f'({remaining_races} Grand Prix at 26 points plus any '
            f'remaining sprint weekends at 8). Points needed = gap + 1 '
            f'to overtake the leader outright.')
    clinch_rows = title_clinch_checks(
        totals, remaining_races, _remaining_sprints(year, None))
    if clinch_rows:
        ui.section('Title clinch check',
                   note_text='Mathematical analysis, not a prediction',
                   hint='Who can still win the title, and what each '
                        'driver would need to outscore the leader by.')
        rows = []
        for row in clinch_rows[:10]:
            rows.append([
                str(int(row['position'])), row['driver'],
                f"{row['points']}", f"{row['gap_to_leader']}",
                explain_clinch(row, leader, remaining_races),
            ])
        data_table(['Pos', 'Driver', 'Points', 'Gap', 'What it takes'],
                   rows)
    with st.expander('How to read this table'):
        st.write('This is arithmetic, not a forecast: it assumes the '
                 'driver scores the maximum at every remaining round. '
                 'A driver marked eliminated cannot overtake the leader '
                 'even with a perfect run of results.')

    # --- Points distribution: ranked championship points ------------------
    ui.section("Points distribution",
               note_text="Championship points, ranked",
               hint="Every driver, best first. Bar colour = team.")
    ranked = totals.sort_values(ascending=True)
    fig_pts = go.Figure(go.Bar(
        x=ranked.values, y=ranked.index, orientation='h',
        marker_color=[_driver_colour(d, team_map, year) for d in ranked.index],
        marker_line_width=0,
        text=[f"{v:.0f}" for v in ranked.values],
        textposition='outside', textfont=dict(size=10, color='#E8EAF0'),
        hovertemplate="<b>%{y}</b><br>%{x:.0f} pts<extra></extra>",
    ))
    fig_pts.update_layout(**_fig_layout(
        height=scaled_height('series', len(ranked), 26),
        xaxis_title=dict(text="Championship points (race + sprint)",
                         font=LABEL_FONT),
        showlegend=False,
    ))
    fig_pts.update_xaxes(rangemode='tozero')
    fig_pts.update_yaxes(showgrid=False)
    show_plotly_chart(fig_pts)

    ui.section('Constructor championship',
               note_text='Jolpica, fallback: computed from results',
               hint=('The constructors\' standings next to the drivers\'. '
                     'Jolpica is the primary source; when it is '
                     'unreachable the table is computed from the season '
                     'results and labelled estimated.'))
    ctor_table, ctor_source = jolpica.constructor_standings_with_fallback(
        year, season_df)
    if ctor_table is not None and not ctor_table.empty:
        rows = []
        for _, row in ctor_table.iterrows():
            rows.append([
                str(int(row['Position'])),
                row['Constructor'],
                f"{row['Points']:.0f}",
                str(int(row['Wins'])),
            ])
        data_table(['Pos', 'Constructor', 'Points', 'Wins'], rows)
        caption(f'Source: {ctor_source}.')
    else:
        empty('Constructor standings unavailable.',
              'Neither Jolpica nor the season results could provide '
              'the constructor table.')

    ui.section('Analytical insights',
               note_text='All figures from the season results',
               hint=('The same derived findings as the panel above, '
                     'listed with their evidence.'))
    if found:
        takeaways([(item['title'],
                    f"{item['text']} (Evidence: {item['evidence']}.)")
                   for item in found])
    else:
        caption('No insights could be derived from the current selection.')

    ui.glossary(['championship', 'points', 'podium', 'sprint'])


def _teammate_preset(season_df, totals, meta):
    """The championship leader and their teammate, when both are known."""
    if totals.empty:
        return []
    leader = str(totals.index[0])
    leader_team = (meta.get(leader) or {}).get('team')
    mates = [d for d in totals.index.tolist()
             if (meta.get(d) or {}).get('team') == leader_team]
    return mates[:2] if len(mates) >= 2 else totals.index.tolist()[:2]


def _contender_preset(season_df, remaining_races, year):
    """Drivers who can still mathematically overtake the leader (max 8)."""
    table = champ.scenarios(season_df, remaining_races,
                           _remaining_sprints(year, None))
    if table.empty:
        return []
    alive = table[table['Status'] != 'Mathematically eliminated']
    return alive['Driver'].tolist()[:8]


page()
