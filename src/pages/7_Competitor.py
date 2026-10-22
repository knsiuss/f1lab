# -*- coding: utf-8 -*-
"""Competitor analysis: rival watchlist, head-to-head and circuit insights.

The page is written for someone who does not follow the sport: every section
says in one plain sentence what it shows, every table uses everyday words for
its columns, and every block ends with a one-sentence verdict that says what
the numbers add up to.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

import glossary as glossary_terms
import teams as team_stats
from shared import load_race_data
from config import FASTF1_CONFIG
from season_config import get_race_names, resolve_csv_track
from rival import (
    build_watchlist,
    circuit_insights,
    head_to_head,
)
import ui
from ui import (
    caption, chips, confidence_tag, data_table, empty, estimate_note, kpis, note, page_header,
)


# The wording for the sample-size levels of `pace.data_quality`, in the
# words a newcomer would use. The chip under each table says the same thing
# in the design system's own wording.
_QUALITY_WORDS = {
    'good': 'Good',
    'moderate': 'Medium',
    'low': 'Low',
    'insufficient': 'Not enough data',
}

_QUALITY_TONES = {
    'good': 'good',
    'moderate': 'accent',
    'low': 'dim',
    'insufficient': 'dim',
}


def _distinct(frame, column):
    """Distinct values in a column (0 when the column is missing)."""
    try:
        return frame[column].nunique()
    except (KeyError, TypeError, ValueError):
        return 0


def _as_int(value):
    """An int, or None when the value is missing or not a number."""
    try:
        if value is None or value != value:  # NaN never equals itself
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value):
    """A float, or None when the value is missing or not a number."""
    try:
        if value is None or value != value:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _quality_cell(level):
    """A table cell that says how much data a figure is based on."""
    key = str(level or '').strip().lower()
    return {'value': _QUALITY_WORDS.get(key, 'Unknown'),
            'tone': _QUALITY_TONES.get(key, 'dim')}


def _position_word(rank):
    """A finishing or championship position in words: '3rd', '1st'."""
    try:
        number = int(rank)
    except (TypeError, ValueError):
        return '?'
    if 10 <= number % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
    return f"{number}{suffix}"


def _watchlist_table(watch):
    """The watchlist as an app table with everyday column titles."""
    rows = []
    for _, row in watch.iterrows():
        gap = _as_float(row.get('GapToFocus'))
        if gap is None:
            gap_cell = {'value': 'n/a', 'tone': 'dim'}
        else:
            gap_cell = {'value': f"{gap:+.0f}",
                        'tone': 'accent' if gap > 0 else 'dim'}
        points = _as_int(row.get('Points'))
        rows.append([
            str(row.get('Driver', '')),
            str(row.get('Team', '')),
            f"{_position_word(row.get('Rank'))} in the championship",
            'n/a' if points is None else str(points),
            gap_cell,
            _quality_cell(row.get('Level')),
        ])
    data_table(
        ['Driver', 'Team', 'Championship position', 'Points',
         'Points ahead (+) or behind (-) of the focus driver',
         'How much of the season is covered'],
        rows,
    )


def _watchlist_verdict(watch, focus):
    """What the watchlist adds up to, in one sentence."""
    if watch is None or watch.empty:
        return ''
    focus_row = watch[watch['Driver'] == focus]
    rivals = watch[watch['Driver'] != focus]
    if focus_row.empty:
        return ''
    focus_points = _as_int(focus_row.iloc[0].get('Points'))
    rank = _position_word(focus_row.iloc[0].get('Rank'))
    if rivals.empty:
        return (f"{focus} is {rank} in the championship on "
                f"{'no' if focus_points is None else focus_points} points; "
                f"no other driver is close enough on points to show as a rival "
                f"yet.")
    nearest = rivals.iloc[0]
    gap = _as_float(nearest.get('GapToFocus'))
    gap_text = 'level on points' if gap == 0 else f"{gap:+.0f} points"
    return (f"{focus} is {rank} in the championship on "
            f"{'no' if focus_points is None else focus_points} points, and the "
            f"closest rival on points is {nearest.get('Driver')} "
            f"({gap_text}).")


def _head_to_head_verdict(h2h, driver_a, driver_b):
    """What the head-to-head adds up to, in one sentence."""
    races = _as_int(h2h.get('races')) or 0
    if h2h.get('level') == 'insufficient':
        return (f"{driver_a} and {driver_b} have only {races} race(s) in "
                f"common this season, so there is not enough data to say "
                f"which of them is ahead.")
    a_wins = _as_int(h2h.get('a_wins')) or 0
    b_wins = _as_int(h2h.get('b_wins')) or 0
    draws = _as_int(h2h.get('draws')) or 0
    points_diff = _as_int(h2h.get('points_diff'))
    lead = driver_a if a_wins > b_wins else (driver_b if b_wins > a_wins else None)
    if lead:
        text = (f"In the {races} race(s) where both finished, {lead} came "
                f"ahead more often ({a_wins} to {b_wins}"
                + (f", {draws} tie(s)" if draws else "") + ").")
    else:
        text = (f"In the {races} race(s) where both finished they finished "
                f"level every time ({a_wins} to {b_wins}).")
    if points_diff is not None:
        text += (f" On championship points {driver_a} is "
                 f"{abs(points_diff)} point(s) "
                 f"{'ahead' if points_diff > 0 else 'behind'} {driver_b}.")
    return text


def _circuit_verdict(ins, track):
    """What the circuit figures add up to, in one sentence."""
    races = _as_int(ins.get('races')) or 0
    if ins.get('level') == 'insufficient' or races < 2:
        return (f"Only {races} race(s) have been run at {track} so far, so "
                f"there is not enough data to say what this circuit rewards.")
    winner = ins.get('winner')
    gained = _as_float(ins.get('avg_places_gained'))
    text = f"{races} race(s) have been run at {track}"
    if winner:
        text += f", and {winner} won the most recent one"
    if gained is not None:
        text += (f"; on average the field finished "
                 f"{abs(gained):.1f} place(s) "
                 f"{'further back' if gained < 0 else 'further forward'} than "
                 f"where it started")
    return text + "."


def _body():
    """Everything on the page except the closing reading guide."""
    year = st.session_state.get('selected_year', FASTF1_CONFIG.default_year)
    df = load_race_data(year)
    if df is None or df.empty:
        empty(f"No {year} results yet",
              "Results appear here once races are completed. Live session "
              "data is available in the Analysis Center.")
        return

    # The championship counts race AND sprint points, so the watchlist uses
    # the full frame. The head-to-head and circuit views compare race
    # results only.
    race_df = df[df['SessionType'] == 'Race'].copy() if 'SessionType' in df.columns else df.copy()

    drivers = sorted(df['Driver'].dropna().unique().tolist())
    if not drivers:
        empty("No drivers found in season data.")
        return

    n_races = (int(_distinct(race_df, 'Round'))
               if 'Round' in race_df.columns
               else int(_distinct(race_df, 'Track')))

    page_header(
        "Competitor Analysis",
        "Who is closest to a driver on points, how two drivers have finished "
        "against each other, and what a circuit has rewarded this season.",
        chips=[('season', str(year)), ('', f'{n_races} races'),
               ('', 'public results')],
    )

    note("Everything on this page comes from public race results. It is a "
         "study tool, not a live pit-wall tool. Figures worked out from the "
         "results are marked as estimated, and the confidence chip under each "
         "table says how many races the figure is based on.")

    # -- Rival Watchlist --------------------------------------------------
    ui.section("Rival watchlist",
            note_text=f"{n_races} race(s) covered",
            hint="Pick a driver. The list then shows that driver and the "
                 "rivals nearest to them on championship points, with the "
                 "gap in points between each of them. "
                 + glossary_terms.short('points'))
    c1, c2 = st.columns([2, 3])
    with c1:
        focus = st.selectbox("Focus driver", drivers, key="rival_focus")
    with c2:
        top_n = st.slider("Watchlist size", 3, 10, 5, key="rival_top")

    watch = build_watchlist(df, focus=focus, top=top_n)
    if watch.empty:
        empty("Insufficient data to build a watchlist.",
              "The championship table is empty for this season, so there is "
              "nothing to compare yet.")
    else:
        _watchlist_table(watch)
        level = watch['Level'].iloc[0] if 'Level' in watch.columns else ''
        confidence = watch['Confidence'].iloc[0] if 'Confidence' in watch.columns else ''
        confidence_tag(level)
        chips([('points', 'real results'),
               ('rival status', 'estimated'),
               ('confidence', str(confidence))])
        caption("A positive number in the points column means that driver has "
                "more points than the focus driver; a negative number means "
                "fewer. The championship position is where the driver sits in "
                "the standings, not the order of this list.")
        ui.verdict(_watchlist_verdict(watch, focus))

    # -- Head-to-Head -----------------------------------------------------
    ui.section("Head-to-head",
            hint="How the two drivers you pick have finished against each "
                 "other. Only the races where both of them finished are "
                 "counted.")
    h1, h2 = st.columns(2)
    with h1:
        driver_a = st.selectbox("Driver A", drivers, index=0, key="rival_a")
    with h2:
        driver_b = st.selectbox("Driver B", [d for d in drivers if d != driver_a],
                                index=0, key="rival_b")

    h2h = head_to_head(race_df, driver_a, driver_b)
    races = _as_int(h2h.get('races')) or 0
    if h2h.get('level') == 'insufficient':
        note(f"{driver_a} and {driver_b} have finished {races} race(s) "
             f"together this season. Two races are needed before a "
             f"comparison is shown, so there is not enough data yet.",
             kind='warn')
        confidence_tag(h2h.get('level'))
    else:
        pts = h2h.get('points_diff')
        a_avg = _as_float(h2h.get('a_avg_pos'))
        b_avg = _as_float(h2h.get('b_avg_pos'))
        kpis([
            {'label': 'Races where both finished', 'value': str(races),
             'sub': 'counted for the record', 'tone': 'info'},
            {'label': f'Times {driver_a} finished ahead',
             'value': str(_as_int(h2h.get('a_wins')) or 0), 'tone': 'accent'},
            {'label': f'Times {driver_b} finished ahead',
             'value': str(_as_int(h2h.get('b_wins')) or 0)},
            {'label': 'Points difference (A minus B)',
             'value': f"{pts:+d}" if pts is not None else 'n/a',
             'sub': 'championship points, whole season'},
        ])
        confidence_tag(h2h.get('level'))
        estimate_note(
            f"Average finishing position (estimated): {driver_a} finished "
            f"{'no average available' if a_avg is None else f'on average in position {a_avg:.1f}'} "
            f"and {driver_b} finished "
            f"{'no average available' if b_avg is None else f'on average in position {b_avg:.1f}'} "
            f"(1 is the win, a bigger number is further back).")
        ui.verdict(_head_to_head_verdict(h2h, driver_a, driver_b))

    # -- Pit stops and reliability ----------------------------------------
    _pit_and_reliability(year)

    # -- Circuit Insights -------------------------------------------------
    ui.section("What this circuit rewards",
            hint="The pattern this one circuit has shown so far this season: "
                 "who wins there, who starts from the front and wins, and how "
                 "many places drivers usually gain or lose over the race.")
    race_names = get_race_names(year)
    all_races = list(race_names.keys()) if race_names else []
    selected_track = st.selectbox("Circuit", all_races, key="rival_track") if all_races else None
    if selected_track:
        # The frame's Track column holds the FastF1 event name, so the
        # selector value usually matches directly; the resolution keeps the
        # lookup working with either spelling.
        csv_tracks = (list(dict.fromkeys(race_df['Track'].astype(str)))
                      if 'Track' in race_df.columns else [])
        track = resolve_csv_track(year, selected_track, csv_tracks) or selected_track
        ins = circuit_insights(race_df, track)
        if not ins.get('winner') and not ins.get('races'):
            empty(f"Insufficient data for {selected_track}.",
                  "No race at this circuit is in the season results yet.")
        else:
            gained = _as_float(ins.get('avg_places_gained'))
            kpis([
                {'label': 'Most recent winner', 'value': str(ins.get('winner', 'n/a')),
                 'tone': 'accent'},
                {'label': 'Started from the front',
                 'value': str(ins.get('pole', 'n/a')),
                 'sub': 'fastest lap in qualifying'},
                {'label': 'Won from the front',
                 'value': 'Yes' if ins.get('pole_to_win') else 'No',
                 'sub': 'started first and won'},
                {'label': 'Places gained on average',
                 'value': f"{gained:+.2f}" if gained is not None else 'n/a',
                 'sub': 'starting grid to the finish'},
            ])
            assumptions = ' · '.join(ins.get('assumptions', []))
            caption(
                f"Fastest lap: {ins.get('fastest_lap', 'not loaded')} — "
                f"{ins.get('races', 0)} race(s) at this circuit, confidence "
                f"{ins.get('confidence')} (estimated). " + assumptions)
            confidence_tag(ins.get('level'))
            ui.verdict(_circuit_verdict(ins, selected_track))


@st.cache_data(ttl=3600, show_spinner=False)
def _pit_league(year):
    """Per-team pit stop numbers (cached: the feed does not change per rerun)."""
    return team_stats.pit_stop_league(year)


@st.cache_data(ttl=3600, show_spinner=False)
def _reliability(year):
    """Per-team reliability numbers, cached like the pit league."""
    return team_stats.reliability(year)


@st.fragment
def _pit_and_reliability(year):
    """Who has the quickest pit crew and who keeps retiring."""
    ui.section(
        "Pit stops and reliability",
        hint="How long each team takes to change four tyres, and how often "
             "its cars actually reach the finish.")
    ui.explain('pit stop', glossary_terms.long('pit stop'))

    league = _pit_league(year)
    if league.empty:
        empty("No pit stop times published yet",
              "The feed has not published a measured stop for this season, so "
              "there is nothing to rank.")
    else:
        rows = []
        for _, row in league.iterrows():
            rows.append([
                row['team'],
                str(int(row['stops'])),
                f"{row['median_s']:.2f} s",
                f"{row['best_s']:.2f} s",
                {'value': _QUALITY_WORDS.get(row['level'], str(row['level'])),
                 'tone': _QUALITY_TONES.get(row['level'], 'dim')},
            ])
        data_table(["Team", "Stops", "Typical stop", "Quickest stop",
                    "How much data"], rows)
        quickest = league.iloc[0]
        ui.verdict(
            f"{quickest['team']} has the quickest typical stop at "
            f"{quickest['median_s']:.2f} s, measured over "
            f"{int(quickest['stops'])} stops.",
            label="The pit crews")
        caption("A stop that the feed did not time is skipped, never filled in "
                "with an average. 'Typical' is the median, so one slow stop "
                "does not move it.")
        confidence_tag(league.iloc[0]['level'])

    reliability = _reliability(year)
    if reliability.empty:
        empty("No classification published yet",
              "Reliability is counted from the official classification, which "
              "is not available for this season yet.")
    else:
        rows = []
        for _, row in reliability.iterrows():
            rows.append([
                row['team'],
                str(int(row['starts'])),
                str(int(row['finishes'])),
                str(int(row['retirements'])),
                f"{row['finish_rate']:.1f}%",
                {'value': _QUALITY_WORDS.get(row['level'], str(row['level'])),
                 'tone': _QUALITY_TONES.get(row['level'], 'dim')},
            ])
        data_table(["Team", "Car entries", "Finished", "Did not finish",
                    "Finish rate", "How much data"], rows)
        best = reliability.iloc[0]
        ui.verdict(
            f"{best['team']} has finished {best['finish_rate']:.0f}% of its "
            f"car entries this season ({int(best['finishes'])} of "
            f"{int(best['starts'])}).",
            label="The most reliable team")
        caption("Counted from the official classification: a car that was "
                "classified despite retiring counts as a finish.")


def page():
    """The page: the analysis, then the reading guide."""
    _body()

    ui.steps([
        "Pick a focus driver. The watchlist then shows that driver and the "
        "rivals nearest to them on championship points.",
        "Read the points column: a positive number means that driver has "
        "more points than the focus driver, a negative number means fewer.",
        "In the head-to-head, only races where both drivers finished are "
        "counted, so a driver who retired does not count as a loss for "
        "anyone.",
        "In the circuit block, 'started from the front' means the driver was "
        "fastest in qualifying and began the race first.",
        "Every estimate carries a confidence chip, which says how much data "
        "the estimate is based on. When it says 'Not enough data', the page "
        "will not draw a conclusion from it.",
    ], title="How to read this page")

    ui.glossary(['points', 'race', 'sprint', 'grid', 'estimated', 'confidence'],
             title="What the words mean")


page()
