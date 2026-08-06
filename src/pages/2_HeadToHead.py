import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from shared import load_race_data, show_plotly_chart
from config import TEAM_COLORS, FASTF1_CONFIG
from styles import (
    GRID, LABEL_FONT, TEXT_PRIMARY, TEXT_SECONDARY, TITLE_FONT,
    fig_layout as _fig_layout,
)
from analysis import (
    calculate_teammate_comparison, calculate_matchup,
    calculate_form_trend, calculate_points_trajectory,
)


def _driver_team_map(race_df):
    """Map each driver to their primary team for color lookup."""
    return race_df.groupby('Driver')['Team'].first().to_dict()


def _driver_color(driver, team_map):
    return TEAM_COLORS.get(team_map.get(driver, ''), '#888')


def page():
    year = st.session_state.get('selected_year', FASTF1_CONFIG.default_year)
    df = load_race_data(year)

    if df is None or df.empty:
        st.error("No data available")
        return

    # Race sessions only (sprint double-counts tracks in season aggregates)
    race_df = df[df['SessionType'] == 'Race'].copy() if 'SessionType' in df.columns else df.copy()

    st.markdown(f"""
    <div style="text-align:center;padding:1.2rem 0 0.8rem 0;">
        <h1 style="font-size:2.2rem;font-weight:800;color:#E10600;margin:0;
                   text-transform:uppercase;letter-spacing:3px;">
            Season &amp; Head-to-Head
        </h1>
        <p style="font-size:0.9rem;color:#666;margin-top:0.3rem;letter-spacing:0.5px;">
            {year} Season &mdash; Teammate Battles &middot; Matchups &middot; Form &middot; Title Race
        </p>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["Teammate Battles", "Driver Matchup", "Form & Consistency", "Points Trajectory"])

    with tabs[0]:
        _tab_teammates(race_df)
    with tabs[1]:
        _tab_matchup(race_df)
    with tabs[2]:
        _tab_form(race_df)
    with tabs[3]:
        _tab_trajectory(race_df)


def _tab_teammates(race_df):
    st.markdown("##### Teammate Head-to-Head")
    comp = calculate_teammate_comparison(race_df)
    if comp is None or comp.empty:
        st.info("No teammate pairs with shared races found.")
        return

    display = comp[
        ['Team', 'Driver 1', 'Driver 2', 'Races Together',
         'Race H2H', 'Quali H2H', 'Pts 1', 'Pts 2']
    ].copy()
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("##### Race Wins by Pairing")
    fig = go.Figure()
    for _, row in comp.iterrows():
        team = row['Team']
        color = TEAM_COLORS.get(team, '#555')
        fig.add_trace(go.Bar(
            x=[int(row['D1 Race Wins'])], y=[team], orientation='h',
            name=row['Driver 1'], marker_color=color, marker_line_width=0,
            hovertemplate=f"<b>{row['Driver 1']}</b><br>Wins: {int(row['D1 Race Wins'])}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=[int(row['D2 Race Wins'])], y=[team], orientation='h',
            name=row['Driver 2'], marker_color=color, opacity=0.45, marker_line_width=0,
            hovertemplate=f"<b>{row['Driver 2']}</b><br>Wins: {int(row['D2 Race Wins'])}<extra></extra>",
        ))
    fig.update_layout(**_fig_layout(
        height=60 + 36 * len(comp),
        title=dict(text="Race wins per pairing (solid = Driver 1, translucent = Driver 2)",
                   font=TITLE_FONT, x=0.01),
        xaxis=dict(title=dict(text="Wins", font=LABEL_FONT),
                   tickfont=dict(size=10, color=TEXT_SECONDARY), gridcolor=GRID),
        yaxis=dict(tickfont=dict(size=11, color=TEXT_PRIMARY)),
        barmode='stack', showlegend=False,
        margin=dict(l=120, r=40, t=40, b=30),
    ))
    show_plotly_chart(fig)


def _tab_matchup(race_df):
    st.markdown("##### Driver vs Driver Matchup")
    drivers = sorted(race_df['Driver'].dropna().unique())
    if len(drivers) < 2:
        st.info("Need at least two drivers.")
        return

    c1, c2 = st.columns(2)
    with c1:
        d1 = st.selectbox("Driver A", drivers, key="h2h_d1")
    with c2:
        d2 = st.selectbox("Driver B", drivers, index=1, key="h2h_d2")

    if d1 == d2:
        st.info("Pick two different drivers.")
        return

    result = calculate_matchup(race_df, d1, d2)
    if result is None:
        st.info(f"{d1} and {d2} have no shared races in {st.session_state.get('selected_year', FASTF1_CONFIG.default_year)}.")
        return

    s = result['summary']
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Races Together", int(s['races_together']))
    with m2:
        st.metric("Race H2H",
                  f"{s['race_record']['driver1']} - {s['race_record']['driver2']}",
                  help=f"{d1} wins - {d2} wins on race day")
    with m3:
        st.metric("Quali H2H",
                  f"{s['quali_record']['driver1']} - {s['quali_record']['driver2']}",
                  help=f"{d1} wins - {d2} wins in qualifying")
    with m4:
        st.metric("Points", f"{s['points']['driver1']} - {s['points']['driver2']}")

    st.markdown("##### Race-by-Race")
    st.dataframe(result['details'], use_container_width=True, hide_index=True)


def _tab_form(race_df):
    st.markdown("##### Rolling Finishing Position (Form)")
    drivers = sorted(race_df['Driver'].dropna().unique())
    top = race_df.groupby('Driver')['Points'].sum().sort_values(ascending=False).index.tolist()
    defaults = [d for d in top if d in drivers][:5]
    selected = st.multiselect("Drivers", drivers, default=defaults,
                              max_selections=8, key="form_drv")
    if not selected:
        return

    result = calculate_form_trend(race_df, selected)
    form = result['form']
    if form.empty:
        st.info("Not enough data for a form trend.")
        return

    team_map = _driver_team_map(race_df)
    fig = go.Figure()
    for d in selected:
        if d not in form.columns:
            continue
        color = _driver_color(d, team_map)
        fig.add_trace(go.Scatter(
            x=form.index, y=form[d], name=d, mode='lines+markers',
            line=dict(color=color, width=3, shape='spline', smoothing=0.4),
            marker=dict(size=7, color=color, line=dict(width=1.2, color='white')),
            hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.2f}<extra></extra>",
        ))
    fig.update_layout(**_fig_layout(
        height=420,
        title=dict(text="Rolling average finishing position (lower = better)", font=TITLE_FONT, x=0.01),
        xaxis=dict(title=dict(text="Race", font=LABEL_FONT),
                   gridcolor=GRID, tickfont=dict(size=10, color=TEXT_SECONDARY), tickangle=-30),
        yaxis=dict(title=dict(text="Avg Position", font=LABEL_FONT), autorange='reversed',
                   gridcolor=GRID, tickfont=dict(size=11, color=TEXT_SECONDARY)),
        legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
        margin=dict(l=55, r=30, t=55, b=90),
    ))
    show_plotly_chart(fig)

    positions = result['positions']
    if not positions.empty and positions.shape[0] > 1:
        st.markdown("##### Finishing Position Heatmap")
        vals = pd.to_numeric(positions.values.ravel(), errors='coerce')
        zmax = int(vals.max()) if vals.max() is not None and not pd.isna(vals.max()) else 20
        zmax = max(20, zmax) if zmax > 0 else 20
        fig2 = go.Figure(go.Heatmap(
            z=positions.T.values, x=positions.index, y=positions.columns,
            colorscale='RdYlGn_r', zmin=1, zmax=zmax,
            hovertemplate="%{y}<br>%{x}: P%{z}<extra></extra>",
        ))
        fig2.update_layout(**_fig_layout(
            height=80 + 26 * len(positions.columns),
            title=dict(text="Finishing position by race (green = P1)", font=TITLE_FONT, x=0.01),
            xaxis=dict(tickfont=dict(size=9, color=TEXT_SECONDARY), gridcolor=GRID, tickangle=-30),
            yaxis=dict(tickfont=dict(size=10, color=TEXT_PRIMARY)),
            coloraxis=dict(colorbar=dict(tickfont=dict(size=10, color=TEXT_SECONDARY))),
            margin=dict(l=55, r=30, t=40, b=80),
        ))
        show_plotly_chart(fig2)


def _tab_trajectory(race_df):
    st.markdown("##### Championship Points Trajectory")
    drivers = sorted(race_df['Driver'].dropna().unique())
    top = race_df.groupby('Driver')['Points'].sum().sort_values(ascending=False).index.tolist()
    defaults = [d for d in top if d in drivers][:5]
    selected = st.multiselect("Drivers", drivers, default=defaults,
                              max_selections=8, key="traj_drv")
    if not selected:
        return

    traj = calculate_points_trajectory(race_df, selected)
    if traj.empty:
        st.info("Not enough data for a points trajectory.")
        return

    team_map = _driver_team_map(race_df)
    fig = go.Figure()
    for d in selected:
        if d not in traj.columns:
            continue
        color = _driver_color(d, team_map)
        fig.add_trace(go.Scatter(
            x=traj.index, y=traj[d], name=d, mode='lines+markers',
            line=dict(color=color, width=3, shape='spline', smoothing=0.4),
            marker=dict(size=8, color=color, line=dict(width=1.5, color='white')),
            hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.0f} pts<extra></extra>",
        ))
    fig.update_layout(**_fig_layout(
        height=430,
        title=dict(text="Cumulative points over the season", font=TITLE_FONT, x=0.01),
        xaxis=dict(title=dict(text="Race", font=LABEL_FONT),
                   gridcolor=GRID, tickfont=dict(size=10, color=TEXT_SECONDARY), tickangle=-30),
        yaxis=dict(title=dict(text="Points", font=LABEL_FONT),
                   gridcolor=GRID, tickfont=dict(size=11, color=TEXT_SECONDARY)),
        hovermode='x unified',
        legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
        margin=dict(l=55, r=30, t=55, b=90),
    ))
    show_plotly_chart(fig)


page()