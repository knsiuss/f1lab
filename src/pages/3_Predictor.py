# -*- coding: utf-8 -*-
"""Predictor & Fantasy: race forecast, fantasy lineup and model accuracy.

The page is written for a reader who does not follow Formula 1 closely.
Every section says in one plain sentence what it is showing, every number
the model makes up is labelled estimated, and the model is scored against a
deliberately simple guess so the reader can see how much the forecast is
really worth.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from shared import load_race_data, load_race_grid, show_plotly_chart, team_colour
from config import FASTF1_CONFIG, STREAMLIT_CONFIG
from season_config import get_race_names, latest_race_index, resolve_csv_track
from styles import (
    fig_layout as _fig_layout,
    LABEL_FONT, TEXT_SECONDARY,
    scaled_height,
)
from prediction import (
    predict_race, add_predicted_points, add_price,
    compute_driver_prices, optimal_lineup, forecast_vs_actual,
    rows_before_race, feature_report,
    DEFAULT_BUDGET, DEFAULT_DRIVERS,
)
from analysis import top3_overlap
from backtest import walk_forward_backtest, METRICS as BACKTEST_METRICS
import ui
from ui import CONFIDENCE_LABELS


# ---------------------------------------------------------------------------
# Cached data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl, show_spinner=False)
def _cached_forecast(history_df, grid):
    """Predictions rerun on every widget interaction otherwise."""
    return predict_race(history_df, grid)


@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl, show_spinner=False)
def _cached_prices(race_df):
    return compute_driver_prices(race_df)


@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl, show_spinner=False)
def _cached_backtest(race_df):
    """Walk-forward backtest is the heaviest call on this page."""
    return walk_forward_backtest(race_df)


# ---------------------------------------------------------------------------
# Plain language
# ---------------------------------------------------------------------------

# The measures of the walk-forward backtest, in words a newcomer can follow.
BACKTEST_LABELS = {
    'exact_match_rate': 'Drivers on the exact finishing place',
    'mae': 'Average error in positions (lower is better)',
    'spearman': 'How well the order matched the real order (1.0 is perfect)',
    'podium_overlap': 'Same three drivers on the podium',
    'points_mae': 'Average error in fantasy points (lower is better)',
}

# Two drivers whose scores are this close can swap places easily. The score
# runs from 0 to 1, so 0.02 is a two-percent gap in the model's own rating.
RISK_GAPS = (0.02, 0.05)

# The backtest module states its verdict in the vocabulary of the sport. These
# swaps keep that module the single source of the verdict while leaving the
# sentence in plain words for the reader.
_VERDICT_PLAIN = (
    ('does NOT beat the grid baseline',
     'does not beat simply copying the starting order'),
    ('only matches the grid baseline',
     'only matches simply copying the starting order'),
    ('beats the grid baseline',
     'does better than simply copying the starting order'),
    ('compare the model with the grid baseline',
     'compare the model with simply copying the starting order'),
    ("'the result is the grid'", 'simply copying the starting order'),
    ('the grid remains', 'the starting order remains'),
    ('position MAE', 'average error in positions'),
    ('grid baseline', 'the starting order'),
)


def _plain_verdict(raw: str) -> str:
    """The backtest verdict with the sport's vocabulary taken out."""
    text = str(raw or '')
    for old, new in _VERDICT_PLAIN:
        text = text.replace(old, new)
    return text


# The frame of the model, in plain words. The accuracy numbers are NOT written
# here: they come from the walk-forward run (``summary``) so the sentence on
# this page can never drift away from what was measured.
_MODEL_PLAIN = (
    'This is a statistical model, not a real forecast. It learns the finishing '
    'order from things known before the race: the starting order, how quick '
    'the team has been in the rounds before this one, how the driver has been '
    'finishing lately, how many places the driver usually gains, and how often '
    'the driver is classified at the finish.'
)


def _plain_caveat(summary=None) -> str:
    """The model's own weakness, in plain words, with the measured numbers."""
    if not summary or summary.get('n_races') in (None, 0):
        return (
            f'{_MODEL_PLAIN} How close it gets is measured race by race on the '
            'accuracy tab: each race is scored with only the rounds before it, '
            'against simply copying the starting order.'
        )
    races = int(summary['n_races'])
    model_error = summary.get('mae_model')
    start_error = summary.get('mae_baseline')
    wins = summary.get('mae_model_beats_baseline')
    if model_error is None or start_error is None:
        return (
            f'{_MODEL_PLAIN} Over the {races} races that could be checked it '
            'was not possible to compare it with the starting order.'
        )
    closer = (f' It was closer than simply copying the starting order in '
              f'{int(wins)} of {races} races.' if wins is not None else '')
    if float(model_error) < float(start_error):
        outcome = (' It does beat simply copying the starting order on '
                   'average, but by a small margin: the order of the names '
                   'carries more weight than the numbers attached to them.')
    elif float(model_error) == float(start_error):
        outcome = (' It only matches simply copying the starting order, so '
                   'treat it as an ordering guide rather than a result.')
    else:
        outcome = (' It did not beat simply copying the starting order, so '
                   'the starting order is still the better guess: treat it as '
                   'an ordering guide rather than a result.')
    return (
        f'{_MODEL_PLAIN} Run back over the {races} races that could be '
        f'checked it was on average {float(model_error):.2f} places away from '
        f'the real finishing order, against {float(start_error):.2f} for simply '
        f'copying the starting order.' + closer + outcome
    )


def _ordinal(number: int) -> str:
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 11 -> '11th'."""
    value = int(number)
    if 10 <= value % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(value % 10, 'th')
    return f'{value}{suffix}'


def _risk_labels(predicted) -> list:
    """How easily each driver can change place, read from the score gaps.

    The forecast is sorted by score, so the gap to the driver above and the
    gap to the driver below say how crowded the driver's place is. A driver
    whose score is almost the same as a neighbour can finish either way.
    """
    scores = pd.to_numeric(predicted['Score'], errors='coerce').tolist()
    labels = []
    for index, score in enumerate(scores):
        gaps = []
        if index > 0:
            gaps.append(scores[index - 1] - score)
        if index < len(scores) - 1:
            gaps.append(score - scores[index + 1])
        gap = min(gaps) if gaps else 1.0
        if gap < RISK_GAPS[0]:
            labels.append('High')
        elif gap < RISK_GAPS[1]:
            labels.append('Medium')
        else:
            labels.append('Low')
    return labels


def _metric_value(metric: str, value) -> str:
    """A backtest measure in plain units: rates as percentages, errors as places."""
    if value is None:
        return 'not enough races'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(number):
        return 'not enough races'
    if metric in ('exact_match_rate', 'podium_overlap'):
        return f'{number * 100:.0f}%'
    return f'{number:.2f}'


def _better_side(metric: str, model, baseline):
    """Which of the two numbers is the better one for this measure."""
    if model is None or baseline is None:
        return None
    try:
        model_value = float(model)
        baseline_value = float(baseline)
    except (TypeError, ValueError):
        return None
    if pd.isna(model_value) or pd.isna(baseline_value):
        return None
    if BACKTEST_METRICS.get(metric) == 'lower':
        if model_value < baseline_value:
            return 'model'
        if model_value > baseline_value:
            return 'baseline'
        return None
    if model_value > baseline_value:
        return 'model'
    if model_value < baseline_value:
        return 'baseline'
    return None


def _forecast_verdict(predicted, summary) -> str:
    """One plain sentence about what the forecast adds up to."""
    names = [str(row['Driver']) for _, row in predicted.head(3).iterrows()]
    favourite = predicted.iloc[0]
    text = (f"The model expects {', '.join(names)} on the podium, with "
            f"{favourite['Driver']} the favourite at "
            f"{float(favourite['WinProb']):.1f}% chance of winning.")
    races = summary.get('n_races')
    model_error = summary.get('mae_model')
    start_error = summary.get('mae_baseline')
    if races and model_error is not None:
        if start_error is not None and float(model_error) >= float(start_error):
            text += (
                f" Over the {races} earlier races of this season that were "
                f"checked it was on "
                f"average {float(model_error):.2f} places away from the real "
                f"finishing order, while simply copying the starting order "
                f"was {float(start_error):.2f} places away, so the model does "
                f"not beat the starting order: read the order of the names, "
                f"not the places."
            )
        else:
            wins = summary.get('mae_model_beats_baseline')
            closer = (f" It was closer in {int(wins)} of {races} races, so "
                      f"the two are close: read the order of the names, not "
                      f"the places." if wins is not None else '')
            text += (
                f" Over the {races} earlier races of this season that were "
                f"checked it was on "
                f"average {float(model_error):.2f} places away from the real "
                f"finishing order, against {float(start_error):.2f} for "
                f"simply copying the starting order." + closer
            )
    return text


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def page():
    year = st.session_state.get('selected_year', FASTF1_CONFIG.default_year)
    df = load_race_data(year)
    if df is None or df.empty:
        ui.empty(f"No {year} results yet",
              "Championship results appear here once races are completed. "
              "Live session data is available in the Analysis Center.")
        return

    race_df = df[df['SessionType'] == 'Race'].copy() if 'SessionType' in df.columns else df.copy()

    race_names = get_race_names(year)
    all_races = list(race_names.keys()) if race_names else []
    if not all_races:
        ui.empty("No races available for this season")
        return

    # The frame's Track column holds the FastF1 event name, so the dropdown
    # value matches it directly; the resolution keeps the page working with
    # either spelling.
    csv_tracks = (list(dict.fromkeys(race_df['Track'].astype(str)))
                  if 'Track' in race_df.columns else [])

    # Default to the most recent race with a result: a forecast for the first
    # race of the season has no history behind it (zero rounds, "insufficient").
    selected_race = st.selectbox("Which Grand Prix?", all_races,
                                 index=latest_race_index(year, all_races, csv_tracks),
                                 key="pred_race")
    csv_race = resolve_csv_track(year, selected_race, csv_tracks)

    grid = load_race_grid(year, csv_race) if csv_race else {}
    if not grid:
        ui.empty(f"No starting order found for {selected_race}",
              "The qualifying grid for this Grand Prix is not available yet.")
        return

    # The forecast is built from the rounds BEFORE the target race only. With
    # the whole season the team-pace and driver-form features would already
    # contain the race it is predicting.
    history = rows_before_race(df, csv_race or selected_race)
    predicted = add_predicted_points(_cached_forecast(history, grid))
    if predicted.empty:
        ui.empty("Could not build a forecast for this race")
        return

    prices = _cached_prices(race_df)
    predicted = add_price(predicted, prices)

    favourite = predicted.iloc[0]
    quality = predicted['Level'].iloc[0] if 'Level' in predicted.columns else 'insufficient'
    quality_label = CONFIDENCE_LABELS.get(quality, (str(quality), 'dim'))[0]
    n_history = int(history['Round'].nunique()) if 'Round' in history.columns else 0

    # The backtest is the honest check on everything else on this page, so it
    # is run once here and handed to the tabs that need it.
    per_race, summary = _cached_backtest(race_df)

    ui.page_header(
        "Predictor and fantasy",
        "Who the model expects to win the Grand Prix you pick, how sure it "
        "is, the cheapest fantasy line-up, and how accurate the model has "
        "actually been.",
        chips=[('season', str(year)), ('', selected_race),
               ('', f'{len(predicted)} drivers'),
               ('estimated', 'forecast'),
               ('', 'OpenF1 live data')],
    )

    ui.kpis([
        {'label': 'The model expects', 'value': str(favourite['Driver']),
         'sub': f"to win, at {float(favourite['WinProb']):.1f}% chance "
                 f"(estimated)",
         'tone': 'accent'},
        {'label': 'How sure the forecast is', 'value': quality_label,
         'sub': f"from {n_history} earlier races", 'tone': 'info'},
        {'label': 'Races used to build it', 'value': f'{n_history}',
         'sub': 'the rounds before this race only', 'tone': ''},
        {'label': 'Drivers on the grid', 'value': f'{len(grid)}',
         'sub': 'the starting order from qualifying', 'tone': ''},
    ])

    # Every panel is built on every rerun: lazy tabs left the panel blank for a
    # moment while the second run arrived, which reads as a broken page.
    tabs = st.tabs(["Race forecast", "Fantasy team",
                     "Forecast and the real result",
                     "How accurate is the model?"])

    with tabs[0]:
        _tab_forecast(predicted, year, n_history, summary)
    with tabs[1]:
        _tab_fantasy(predicted, year)
    with tabs[2]:
        _tab_actual(race_df, csv_race or selected_race, predicted)
    with tabs[3]:
        _tab_backtest(per_race, summary)

    ui.steps([
        "Pick the Grand Prix you want to read about. The forecast is built "
        "from the rounds before it, so an early race has little to work with "
        "and the page says so.",
        "Start with the podium: the three drivers the model expects at the "
        "front, each with the colour of the team they drive for.",
        "Then read the risk column. A driver whose rating is nearly the same "
        "as the driver next to them can easily finish the other way round.",
        "The last tab is the honest check. It runs the forecast over the "
        "races of the season and compares it with simply copying the "
        "starting order.",
        "Every number the model produces is estimated from public results, "
        "and none of them is a guaranteed result.",
    ], title="How to read this page")

    ui.glossary(["qualifying", "grid", "points", "pace", "backtest", "baseline",
              "estimated", "confidence"])


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

@st.fragment
def _tab_forecast(predicted, year, n_history, summary):
    ui.section(
        "The podium the model expects",
        "estimated by a small regression from the starting order plus the "
        "team's season and the driver's recent results",
        hint="A forecast is a guess at the finishing order. The model learns "
             "from the rounds before this one how much the starting order, "
             "the team's pace, the driver's recent finishes, the places the "
             "driver usually gains and how often the driver finishes actually "
             "matter.",
    )

    ui.estimate_note(_plain_caveat(summary))

    podium = []
    for _, row in predicted.head(3).iterrows():
        team = str(row.get('Team', ''))
        podium.append({
            'position': int(row['Predicted']),
            'name': row['Driver'],
            'team': team,
            'value': f"{float(row['WinProb']):.1f}% chance of winning",
            'sub': f"expected to finish {_ordinal(int(row['Predicted']))}",
            'colour': team_colour(team, year),
        })
    ui.result_rows(podium, empty_text="No forecast available.")

    ui.section(
        "The forecast for every driver",
        "the order the model expects and how risky each place is",
        hint="The expected outcome is the finishing place the model expects "
             "and the championship points it would be worth. The risk says "
             "how easily the order could change: two drivers with almost the "
             "same rating are close to swapping places.",
    )

    risks = _risk_labels(predicted)
    rows = []
    for (_, row), risk in zip(predicted.iterrows(), risks):
        level = str(row.get('Level', ''))
        label = CONFIDENCE_LABELS.get(level, (level, 'dim'))
        rows.append([
            f"P{int(row['Predicted'])}",
            row['Driver'],
            row.get('Team', ''),
            f"P{int(row['Grid'])}",
            f"Finishes {_ordinal(int(row['Predicted']))}, worth "
            f"{int(row['PredictedPts'])} points",
            f"{float(row['WinProb']):.1f}%",
            {'value': risk,
             'tone': {'High': 'bad', 'Medium': 'dim', 'Low': 'good'}[risk]},
            {'value': f"{label[0]} ({n_history} races of history)",
             'tone': label[1]},
        ])
    ui.data_table(
        ['Predicted finish', 'Driver', 'Team', 'Starting place',
         'Expected outcome', 'Chance of winning (estimated)',
         'Risk of a different order', 'How sure the forecast is'],
        rows,
    )

    ui.verdict(_forecast_verdict(predicted, summary))

    ui.section(
        "Chance of winning driver by driver",
        "a ranking of the field and not a precise probability",
        hint="The bars put the drivers in the order the model thinks they "
             "will finish. The differences between the bars are small, so "
             "read the order of the names rather than the lengths.",
    )

    fig = go.Figure()
    for _, row in predicted.sort_values('WinProb', ascending=True).iterrows():
        fig.add_trace(go.Bar(
            x=[row['WinProb']], y=[row['Driver']], orientation='h',
            marker_color=team_colour(row.get('Team', ''), year),
            text=f"{float(row['WinProb']):.1f}%", textposition='outside',
            textfont=dict(size=10, color=TEXT_SECONDARY),
            hovertemplate=(f"<b>{row['Driver']}</b><br>{row.get('Team', '')}"
                           f"<br>Chance of winning: {row['WinProb']}%"
                           f"<extra></extra>"),
        ))
    fig.update_layout(**_fig_layout(
        height=scaled_height('bars', len(predicted)),
        xaxis_title=dict(text="Chance of winning (%)", font=LABEL_FONT),
        showlegend=False,
    ))
    fig.update_xaxes(rangemode='tozero')
    fig.update_yaxes(showgrid=False)
    show_plotly_chart(fig)
    ui.caption(
        "The whole field spans roughly 3-8%. A difference of a tenth of a "
        "point is noise here, so read these bars as an ordering of the field "
        "and not as the chance of a result."
    )


@st.fragment
def _tab_fantasy(predicted, year):
    ui.section(
        "Build a fantasy team",
        "the drivers who give the most projected points for the money",
        hint="A fantasy team is a made-up line-up of real drivers inside a "
             "budget. The app picks the drivers with the most projected "
             "points per million dollars, one after the other, and leaves out "
             "anyone it cannot price.",
    )

    c1, c2 = st.columns(2)
    with c1:
        budget = st.slider("Budget ($M)", 20, 100, DEFAULT_BUDGET, step=5,
                            key="f_budget")
    with c2:
        drv_count = st.slider("Drivers per team", 3, 8, DEFAULT_DRIVERS,
                              key="f_count")

    chosen = optimal_lineup(predicted, budget=budget, count=drv_count)
    if not chosen:
        ui.empty("No affordable lineup within budget",
              "Raise the budget, or pick a race with more priced drivers.")
        return

    team = predicted[predicted['Driver'].isin(chosen)]
    cost = int(pd.to_numeric(team['Price'], errors='coerce').fillna(0).sum())
    pts = int(team['PredictedPts'].sum())

    ui.kpis([
        {'label': 'Projected points', 'value': str(pts),
         'sub': 'if the forecast comes true (estimated)', 'tone': 'accent'},
        {'label': 'Lineup cost', 'value': f'${cost}M',
         'sub': f'budget ${budget}M', 'tone': ''},
        {'label': 'Drivers picked', 'value': f'{len(team)}/{drv_count}',
         'sub': 'slots filled', 'tone': 'warn' if len(team) < drv_count else ''},
    ])

    if len(team) < drv_count:
        ui.note(
            f"Only {len(team)} of {drv_count} drivers fit the ${budget}M budget "
            f"(or have a derivable price). Raise the budget or lower the "
            f"driver count for a full lineup.",
            kind='warn',
        )

    ui.section(
        "The recommended lineup",
        "the drivers to pick and what each one is expected to score",
        hint="The expected outcome is the finishing place the forecast gives "
             "that driver and the fantasy points it would earn. Every number "
             "here is estimated from the forecast above.",
    )

    rows = []
    for _, row in team.sort_values('Predicted').iterrows():
        team_name = str(row.get('Team', ''))
        rows.append({
            'position': int(row['Predicted']),
            'name': row['Driver'],
            'team': team_name,
            'value': f"${int(row['Price'])}M" if pd.notna(row.get('Price')) else '—',
            'sub': f"expected P{int(row['Predicted'])}, "
                   f"{int(row['PredictedPts'])} points (estimated)",
            'colour': team_colour(team_name, year),
        })
    ui.result_rows(rows, empty_text="No lineup selected.")

    table_rows = []
    for _, row in team.sort_values('Predicted').iterrows():
        table_rows.append([
            f"P{int(row['Predicted'])}",
            row['Driver'],
            row.get('Team', ''),
            f"P{int(row['Grid'])}",
            f"{int(row['PredictedPts'])} points",
            f"${int(row['Price'])}M" if pd.notna(row.get('Price')) else '—',
        ])
    ui.data_table(
        ['Expected finish', 'Driver', 'Team', 'Starting place',
         'Projected points (estimated)', 'Price'],
        table_rows,
    )

    ui.verdict(
        f"This line-up of {len(team)} drivers costs ${cost}M of the ${budget}M "
        f"budget and is projected to score {pts} fantasy points (estimated) "
        f"if the forecast holds, which the accuracy tab shows it may not."
    )

    ui.section(
        "What every driver costs",
        "the price the app works out from the season so far",
        hint="The price of a driver is worked out from the points they have "
             "scored so far. A driver with no points has no price that can be "
             "worked out, so they are left out rather than given a made-up "
             "price.",
    )
    ui.caption(
        "Prices are derived from season points. A driver without season "
        "points has no derivable price, shown as a dash and not as a floor "
        "price."
    )
    price_rows = []
    for _, row in predicted.sort_values('Price', ascending=False,
                                        na_position='last').iterrows():
        price = row.get('Price')
        price_rows.append([
            row['Driver'],
            row.get('Team', ''),
            f"${int(price)}M" if pd.notna(price) else '—',
            f"{int(row['PredictedPts'])} points",
        ])
    ui.data_table(['Driver', 'Team', 'Price (estimated from points so far)',
                'Projected points this race'], price_rows)


@st.fragment
def _tab_actual(race_df, selected_race, predicted):
    ui.section(
        "What the forecast said and what actually happened",
        "the model never saw the race it is scored against",
        hint="This compares the forecast with the real finishing order of "
             "the Grand Prix you picked. The forecast was built from the "
             "rounds before that race, so it could not have seen the result.",
    )

    ui.estimate_note(
        "The forecast below was built from the rounds **before** this race, so "
        "the comparison is honest: the model never saw the result it is "
        "scored against. All figures are estimated."
    )

    sub = race_df[race_df['Track'] == selected_race].copy() if 'Track' in race_df.columns else race_df
    if sub.empty:
        ui.empty(f"No recorded result found for {selected_race}")
        return

    actual = {}
    for _, r in sub.iterrows():
        pos = pd.to_numeric(r.get('Position'), errors='coerce')
        if pd.notna(pos):
            actual[r['Driver']] = int(pos)

    joined = forecast_vs_actual(predicted, actual)
    if not actual:
        ui.empty("No completed result to compare against")
        return

    hits = int((joined['Predicted'] == joined['Actual']).sum())
    # "Shared in top 3" = drivers who are in the predicted top 3 AND actually
    # finished in the top 3. The set must not be built with dropna() before
    # head(3): a predicted driver without a recorded finish would pull the
    # 4th predicted driver into the comparison.
    shared = top3_overlap(list(joined['Driver']), actual)
    errors = (joined['Predicted'] - joined['Actual']).abs().dropna()
    mean_error = float(errors.mean()) if not errors.empty else None

    scored = joined[joined['Actual'].notna()]
    points_gap = int(scored['PredictedPts'].sum() - scored['ActualPts'].sum())

    ui.kpis([
        {'label': 'Drivers on the exact place', 'value': f'{hits}/{len(scored)}',
         'sub': 'the forecast said exactly this', 'tone': 'info'},
        {'label': 'Same drivers on the podium', 'value': f'{shared}/3',
         'sub': 'forecast and real podium', 'tone': 'info'},
        {'label': 'Points difference', 'value': f'{points_gap:+d}',
         'sub': 'projected minus actually scored', 'tone': ''},
    ])

    rows = []
    for _, row in joined.iterrows():
        actual_position = row.get('Actual')
        rows.append([
            f"P{int(row['Predicted'])}",
            row['Driver'],
            f"P{int(row['Grid'])}",
            f"P{int(actual_position)}" if pd.notna(actual_position) else '—',
            f"{int(row['PredictedPts'])} points",
            f"{int(row['ActualPts'])} points",
        ])
    ui.data_table(
        ['Predicted finish', 'Driver', 'Starting place', 'Real finish',
         'Points if the forecast was right (estimated)', 'Points actually scored'],
        rows,
    )

    if mean_error is None:
        ui.verdict(
            f"The forecast put {hits} of {len(scored)} drivers on the exact "
            f"finishing place and {shared} of the 3 podium places were right."
        )
    else:
        ui.verdict(
            f"The forecast put {hits} of {len(scored)} drivers on the exact "
            f"finishing place and {shared} of the 3 podium places were right; "
            f"on average it was {mean_error:.1f} places away from the real "
            f"finishing order, which is why the forecast is read as an order "
            f"of names and not as a result."
        )


@st.fragment
def _comparison_confidence(summary) -> str:
    """How much weight the model-vs-baseline comparison can carry."""
    races = int(summary.get('n_races') or 0)
    if races >= 20:
        return 'good'
    if races >= 10:
        return 'moderate'
    if races >= 2:
        return 'low'
    return 'insufficient'


def _closer_label(row) -> str:
    """Which of the two guesses was closer to the real finishing order."""
    try:
        model = float(row.get('mae_model'))
        baseline = float(row.get('mae_baseline'))
    except (TypeError, ValueError):
        return '—'
    if pd.isna(model) or pd.isna(baseline):
        return '—'
    if model < baseline:
        return 'The model'
    if model > baseline:
        return 'The starting order'
    return 'A tie'


def _method_label(engine) -> str:
    """How the race was scored: learned from history, or not enough to learn."""
    if str(engine) == 'heuristic':
        return 'Starting order only'
    return 'Ridge regression'


def _tab_backtest(per_race, summary):
    ui.section(
        "How accurate has the model been?",
        "the forecast is run over the season one race at a time",
        hint="The forecast is replayed over the races of the season. For "
             "every race it only sees the rounds before it, so it cannot "
             "cheat by looking at the result. It is then compared with a "
             "simple guess: that the finishing order is the starting order.",
    )

    ridges = int(summary.get('n_engine_ridge') or 0)
    fallbacks = int(summary.get('n_engine_fallback') or 0)
    ui.estimate_note(
        "Each race is scored with the model trained only on the races before "
        "it, so no race is ever predicted with information from itself or "
        "from later. The comparison is deliberately simple: simply copying "
        "the starting order. All figures are estimated and cover only the "
        "races with enough history to score"
        + (f" ({summary.get('n_races_skipped', 0)} earlier races had too "
           f"little history used)." if summary.get('n_races_skipped') else '.')
        + (f" {ridges} of the {summary['n_races']} scored races used the "
           f"regression; the other {fallbacks} had too little history and "
           f"fell back to the starting order." if fallbacks else '')
    )

    if summary.get('n_races', 0) == 0:
        ui.empty("Not enough completed races to check the model",
              "The check needs at least two races of history before the race "
              "being scored.")
        return

    mae_wins = summary.get('mae_model_beats_baseline')
    ui.kpis([
        {'label': 'Races checked', 'value': str(summary['n_races']),
         'sub': 'each scored with the earlier races only', 'tone': 'info'},
        {'label': 'Races where the model was closer',
         'value': f'{int(mae_wins)} of {summary["n_races"]}' if mae_wins is not None
                  else '—',
         'sub': 'against copying the starting order', 'tone': ''},
        {'label': 'Order match (the forecast)',
         'value': _metric_value('spearman', summary.get('spearman_model')),
         'sub': '1.0 would be a perfect match', 'tone': ''},
        {'label': 'Order match (the starting order)',
         'value': _metric_value('spearman', summary.get('spearman_baseline')),
         'sub': 'the simple comparison', 'tone': ''},
    ])

    # The verdict is the honest headline: whether the model beats the simple
    # "copy the starting order" guess, stated in plain words rather than
    # left to the reader to work out from the table below.
    ui.verdict(_plain_verdict(summary.get('verdict')), label="What the check says")
    ui.confidence_tag(_comparison_confidence(summary))
    if summary.get('beats_baseline') is False:
        ui.note(
            'So read this page as an ordering guide: the names in the right '
            'order carry more weight than the numbers attached to them.',
            kind='warn',
        )

    ui.section(
        "The model against the starting order",
        "the same races scored twice with the model and with the simple "
        "guess",
        hint="Each row is one way of measuring the forecast. The column with "
             "the better number is the one the model wins on: lower is better "
             "for the errors, higher is better for the rest.",
    )

    rows = []
    for metric in BACKTEST_METRICS:
        model_value = summary.get(f'{metric}_model')
        baseline_value = summary.get(f'{metric}_baseline')
        better = _better_side(metric, model_value, baseline_value)
        model_cell = {'value': _metric_value(metric, model_value),
                      'tone': 'good' if better == 'model' else
                              ('bad' if better == 'baseline' else 'dim')}
        baseline_cell = {'value': _metric_value(metric, baseline_value),
                         'tone': 'good' if better == 'baseline' else
                                 ('bad' if better == 'model' else 'dim')}
        wins = summary.get(f'{metric}_model_beats_baseline')
        rows.append([
            BACKTEST_LABELS.get(metric, metric),
            model_cell,
            baseline_cell,
            f"{wins} of {summary.get('n_races')}" if wins is not None else '—',
        ])
    ui.data_table(
        ['What was measured', 'The forecast', 'Simply copying the starting order',
         'Races where the forecast was better'],
        rows,
    )

    ui.section(
        "Race by race",
        "what the model and the simple guess did in each round of the season",
        hint="One row per race that could be scored. The two error columns "
             "say how far each guess was from the real finishing order, "
             "counted in places. The last columns say which guess was closer "
             "in that race and how the forecast was built.",
    )

    detail_rows = []
    for _, row in per_race.iterrows():
        closer = _closer_label(row)
        detail_rows.append([
            str(row.get('Race', '—')),
            str(row.get('N', '—')),
            _metric_value('mae', row.get('mae_model')),
            _metric_value('mae', row.get('mae_baseline')),
            {'value': closer,
             'tone': 'good' if closer == 'The model' else
                     ('bad' if closer == 'The starting order' else 'dim')},
            _method_label(row.get('Engine')),
            str(row.get('Estimated', 'estimated')),
        ])
    ui.data_table(
        ['Race', 'Drivers scored', 'Average error in positions (forecast)',
         'Average error in positions (starting order)', 'Which was closer',
         'How the forecast was built', 'Status'],
        detail_rows,
    )

    scored = int(summary.get('n_races') or 0)
    if mae_wins is not None and scored:
        model_closer = int(mae_wins)
        losses = scored - model_closer
        count = (f'and the starting order was closer in {losses}.' if losses
                 else 'and never the other way round.')
        beats = bool(summary.get('beats_baseline'))
        ui.verdict(
            f"The model was closer to the real finishing order than simply "
            f"copying the starting order in {model_closer} of the {scored} "
            f"races that could be scored, {count}"
            + (' That is a real improvement over the starting order, though a '
               'small one: treat the order of the names as the message and the '
               'places as a rough guide.'
               if beats else
               ' The model is not ahead of the starting order overall, so '
               'treat it as an ordering guide rather than a result.'),
            label="The win-loss count",
        )

    ui.section(
        "What the model can and cannot see",
        "the information behind the forecast, and the information the season "
        "data does not contain",
        hint="Every feature is built from the rounds before the race being "
             "predicted. The rows marked as not used are the ones the season "
             "data simply does not carry, so they are left out rather than "
             "guessed.",
    )

    inventory = feature_report()
    inventory_rows = []
    for entry in inventory:
        used = bool(entry.get('used'))
        inventory_rows.append([
            str(entry.get('feature', '')),
            str(entry.get('source', '')),
            {'value': 'Yes' if used else 'No',
             'tone': 'good' if used else 'dim'},
            str(entry.get('note', '')),
        ])
    ui.data_table(
        ['What it uses', 'Where it comes from', 'In the forecast?', 'Why'],
        inventory_rows,
    )

    unused = [str(entry.get('feature', '')) for entry in inventory
              if not entry.get('used')]
    if unused:
        ui.verdict(
            'The model sees the starting order and the season so far, but not '
            f"{', '.join(unused)}: the season data does not carry it, so the "
            'forecast cannot use it and does not pretend to.',
            label='What is missing',
        )


page()
