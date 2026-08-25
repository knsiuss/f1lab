import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from shared import load_race_data, show_plotly_chart
from config import DATA_DIR, TEAM_COLORS, FASTF1_CONFIG, STREAMLIT_CONFIG
from loader import load_race_grid
from season_config import get_race_names
from styles import (
    GRID, LABEL_FONT, TEXT_PRIMARY, TEXT_SECONDARY, TITLE_FONT,
    fig_layout as _fig_layout,
)
from prediction import (
    predict_race, add_predicted_points, add_price,
    compute_driver_prices, optimal_lineup, forecast_vs_actual,
    DEFAULT_BUDGET, DEFAULT_DRIVERS,
)
from backtest import walk_forward_backtest, METRICS as BACKTEST_METRICS


@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl, show_spinner=False)
def _cached_forecast(race_df, grid):
    """Predictions rerun on every widget interaction otherwise."""
    return predict_race(race_df, grid)


@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl, show_spinner=False)
def _cached_prices(race_df):
    return compute_driver_prices(race_df)


@st.cache_data(ttl=STREAMLIT_CONFIG.cache_ttl, show_spinner=False)
def _cached_backtest(race_df):
    """Walk-forward backtest is the heaviest call on this page."""
    return walk_forward_backtest(race_df)


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
            Predictor &amp; Fantasy
        </h1>
        <p style="font-size:0.9rem;color:#666;margin-top:0.3rem;letter-spacing:0.5px;">
            {year} Season &mdash; Race Forecast &middot; Win Probabilities &middot; Fantasy Team
        </p>
    </div>
    """, unsafe_allow_html=True)

    race_names = get_race_names(year)
    all_races = list(race_names.keys()) if race_names else []
    if not all_races:
        st.warning("No races available.")
        return

    selected_race = st.selectbox("Grand Prix", all_races, key="pred_race",
                                 label_visibility="collapsed")

    grid = load_race_grid(str(DATA_DIR), year, selected_race, race_df)
    if not grid:
        st.info(f"No grid information found for {selected_race}.")
        return

    predicted = add_predicted_points(_cached_forecast(race_df, grid))
    if predicted.empty:
        st.info("Could not build a forecast for this race.")
        return

    prices = _cached_prices(race_df)
    predicted = add_price(predicted, prices)

    tabs = st.tabs(["Race Forecast", "Fantasy Team", "Forecast vs Actual",
                    "Model Accuracy"])

    with tabs[0]:
        _tab_forecast(predicted)
    with tabs[1]:
        _tab_fantasy(predicted)
    with tabs[2]:
        _tab_actual(race_df, selected_race, predicted)
    with tabs[3]:
        _tab_backtest(race_df)


def _tab_forecast(predicted):
    st.markdown("##### Predicted Finishing Order")
    cols = {'Predicted': 'Pred', 'Driver': 'Driver', 'Team': 'Team', 'Grid': 'Grid',
            'WinProb': 'Win %', 'PredictedPts': 'Pts'}
    shown = predicted[[c for c in cols if c in predicted.columns]].copy()
    shown = shown.rename(columns=cols)
    st.dataframe(shown, use_container_width=True, hide_index=True)

    st.markdown("##### Win Probability by Driver")
    fig = go.Figure()
    for _, row in predicted.sort_values('WinProb', ascending=True).iterrows():
        fig.add_trace(go.Bar(
            x=[row['WinProb']], y=[row['Driver']], orientation='h',
            marker_color=TEAM_COLORS.get(row['Team'], '#555'),
            text=f"{row['WinProb']:.1f}%", textposition='outside',
            textfont=dict(size=10, color=TEXT_SECONDARY),
            hovertemplate=f"<b>{row['Driver']}</b><br>{row['Team']}<br>Win: {row['WinProb']}%<extra></extra>",
        ))
    fig.update_layout(**_fig_layout(
        height=70 + 26 * len(predicted),
        title=dict(text="Model win probability", font=TITLE_FONT, x=0.01),
        xaxis=dict(title=dict(text="Probability (%)", font=LABEL_FONT),
                   gridcolor=GRID, tickfont=dict(size=10, color=TEXT_SECONDARY)),
        yaxis=dict(tickfont=dict(size=11, color=TEXT_PRIMARY)),
        showlegend=False, margin=dict(l=60, r=60, t=45, b=30),
    ))
    show_plotly_chart(fig)


def _tab_fantasy(predicted):
    st.markdown("### Fantasy Team Builder")
    c1, c2 = st.columns(2)
    with c1:
        budget = st.slider("Budget ($M)", 20, 100, DEFAULT_BUDGET, step=5, key="f_budget")
    with c2:
        drv_count = st.slider("Drivers per team", 3, 8, DEFAULT_DRIVERS, key="f_count")

    chosen = optimal_lineup(predicted, budget=budget, count=drv_count)
    if not chosen:
        st.info("No affordable lineup within budget.")
        return

    team = predicted[predicted['Driver'].isin(chosen)]
    cost = int(team['Price'].sum())
    pts = int(team['PredictedPts'].sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Projected Points", pts)
    m2.metric("Lineup Cost", f"${cost}M")
    m3.metric("Drivers", len(team))

    st.markdown("##### Recommended Lineup")
    disp = team[['Driver', 'Team', 'Grid', 'Predicted', 'PredictedPts', 'Price']].copy()
    disp.columns = ['Driver', 'Team', 'Grid', 'Pred', 'Pts', 'Price ($M)']
    st.dataframe(disp.sort_values('Pred'), use_container_width=True, hide_index=True)

    st.markdown("##### Market Prices")
    price_df = predicted[['Driver', 'Team', 'Price', 'PredictedPts']].sort_values(
        'Price', ascending=False)
    price_df.columns = ['Driver', 'Team', 'Price ($M)', 'Predicted Pts']
    st.dataframe(price_df, use_container_width=True, hide_index=True)


def _tab_actual(race_df, selected_race, predicted):
    st.markdown("### Forecast vs Actual Result")
    sub = race_df[race_df['Track'] == selected_race].copy() if 'Track' in race_df.columns else race_df
    if sub.empty:
        st.info(f"No recorded result found for {selected_race}.")
        return

    actual = {}
    for _, r in sub.iterrows():
        pos = pd.to_numeric(r.get('Position'), errors='coerce')
        if pd.notna(pos):
            actual[r['Driver']] = int(pos)

    joined = forecast_vs_actual(predicted, actual)
    if not actual:
        st.info("No completed result to compare against.")
        return

    hits = int((joined['Predicted'] == joined['Actual']).sum())
    top3 = {"predicted": set(joined['Predicted'].head(3)),
            "actual": set(joined['Actual'].dropna().astype(int).head(3))}
    shared = len(top3["predicted"] & top3["actual"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Exact Position Hits", f"{hits}/{len(joined)}")
    m2.metric("Shared in Top 3", f"{shared}/3")
    m3.metric("Points Delta (pred - act)",
              int(joined['PredictedPts'].sum() - joined['ActualPts'].sum()))

    disp = joined[['Predicted', 'Driver', 'Grid', 'Actual', 'PredictedPts', 'ActualPts']].copy()
    disp.columns = ['Pred', 'Driver', 'Grid', 'Actual', 'Pred Pts', 'Actual Pts']
    st.dataframe(disp, use_container_width=True, hide_index=True)


def _tab_backtest(race_df):
    st.markdown("### Model Accuracy &mdash; Walk-Forward Backtest")
    st.caption(
        "Each race is scored with the model trained only on the races before it "
        "(no look-ahead, no leakage). The baseline is deliberately naive: "
        "&ldquo;the finishing order equals the grid order.&rdquo; All figures are "
        "estimated and cover only the races with enough history to score."
    )

    per_race, summary = _cached_backtest(race_df)
    if summary['n_races'] == 0:
        st.info("Not enough completed races to backtest.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Races backtested", summary['n_races'])
    m2.metric("Races skipped (insufficient)", summary['n_races_skipped'])
    m3.metric("Spearman vs baseline",
              f"{summary.get('spearman_model')} vs {summary.get('spearman_baseline')}")

    labels = {
        'exact_match_rate': 'Exact position hits',
        'mae': 'Position MAE (lower = better)',
        'spearman': 'Spearman rank correlation',
        'podium_overlap': 'Podium overlap (top 3)',
        'points_mae': 'Fantasy points MAE',
    }
    rows = []
    for metric in BACKTEST_METRICS:
        rows.append({
            'Metric': labels.get(metric, metric),
            'Model': summary.get(f'{metric}_model'),
            'Baseline (grid)': summary.get(f'{metric}_baseline'),
            'Model better (races)': summary.get(f'{metric}_model_beats_baseline'),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("##### Per-race detail")
    detail_cols = [c for c in
                   ['Race', 'N', 'mae_model', 'mae_baseline', 'spearman_model',
                    'spearman_baseline', 'Estimated']
                   if c in per_race.columns]
    show = per_race[detail_cols].copy()
    show.columns = ['Race', 'Drivers', 'MAE (model)', 'MAE (grid)',
                    'Spearman (model)', 'Spearman (grid)', 'Estimated']
    st.dataframe(show, use_container_width=True, hide_index=True)


page()