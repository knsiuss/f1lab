import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shared import load_race_data, show_plotly_chart, format_f1_time
from config import TEAM_COLORS
from season_config import get_completed_races, get_race_names
import fastf1
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

# Design System
BG_PAPER = "rgba(0,0,0,0)"
BG_PLOT = "rgba(0,0,0,0)"
GRID = "rgba(255,255,255,0.05)"
GRID_EMPH = "rgba(255,255,255,0.1)"
ZERO_LINE = "rgba(255,255,255,0.15)"
TEXT_PRIMARY = "#e0e0e0"
TEXT_SECONDARY = "#888"
TEXT_DIM = "#555"
FONT = "Inter, sans-serif"
TITLE_FONT = dict(family=FONT, size=15, color=TEXT_PRIMARY)
LABEL_FONT = dict(family=FONT, size=11, color=TEXT_SECONDARY)

COMPOUND_COLORS = {
    'SOFT': '#FF3333', 'MEDIUM': '#FFD700', 'HARD': '#BBBBBB',
    'INTERMEDIATE': '#43B02A', 'WET': '#0067AD',
}

def _fig_layout(height=400, **overrides):
    base = dict(
        paper_bgcolor=BG_PAPER, plot_bgcolor=BG_PLOT,
        font=dict(family=FONT, color=TEXT_PRIMARY, size=12),
        margin=dict(l=55, r=30, t=55, b=50), height=height,
        hoverlabel=dict(bgcolor="rgba(15,15,25,0.92)",
                       font=dict(family=FONT, size=12, color="white"),
                       bordercolor="rgba(255,255,255,0.15)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=TEXT_PRIMARY),
                   orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1,
                   zerolinecolor=GRID_EMPH, tickfont=dict(size=11, color=TEXT_SECONDARY)),
        yaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1,
                   zerolinecolor=GRID_EMPH, tickfont=dict(size=11, color=TEXT_SECONDARY)),
    )
    base.update(overrides)
    return base

def _team_color_map(df):
    """Build team color map from data."""
    colors = {}
    if 'Team' in df.columns:
        for t in df['Team'].unique():
            colors[t] = TEAM_COLORS.get(t, '#888')
    return colors

def page():
    year = st.session_state.get('selected_year', 2025)
    df = load_race_data(year)

    if df is None or df.empty:
        st.error("No data available")
        return

    # Get championship data
    race_names = get_race_names(year)
    races_done = get_completed_races(year)
    n_races = max(len(race_names), len(races_done)) if race_names is not None else len(races_done)
    races_completed = len(races_done)
    is_sprint = 'SessionType' in df.columns

    # Filter to race sessions for standings
    if is_sprint:
        race_df = df[df['SessionType'] == 'Race'].copy()
    else:
        race_df = df.copy()

    # Driver standings
    if 'Points' in race_df.columns:
        drv_pts = race_df.groupby('Driver')['Points'].sum().sort_values(ascending=False)
    else:
        drv_pts = pd.Series(dtype=float)

    # Constructor standings
    if 'Team' in race_df.columns and 'Points' in race_df.columns:
        team_pts = race_df.groupby('Team')['Points'].sum().sort_values(ascending=False)
    else:
        team_pts = pd.Series(dtype=float)

    # Fastest lap
    fastest_lap_time = race_df['Time/Retired'].dropna().str.extract(r'(\d+:\d+\.\d+)').iloc[0, 0] if 'Time/Retired' in race_df.columns else '-'
    if pd.isna(fastest_lap_time):
        fastest_lap_time = '-'

    # Total points
    total_pts = int(drv_pts.sum()) if not drv_pts.empty else 0

    # === HEADER ===
    st.markdown(f"""
    <div style="text-align:center;padding:1.0rem 0 0.5rem 0;">
        <h1 style="font-size:2.4rem;font-weight:800;color:#E10600;margin:0;
                   text-transform:uppercase;letter-spacing:3px;">
            F1 {year} Dashboard
        </h1>
        <p style="font-size:0.85rem;color:#666;margin-top:0.2rem;letter-spacing:0.3px;">
            Season Overview & Championship Standings
        </p>
    </div>
    """, unsafe_allow_html=True)

    # === KPI ROW ===
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(225,6,0,0.15),rgba(225,6,0,0.05));
                    border-radius:12px;padding:1rem;text-align:center;
                    border:1px solid rgba(225,6,0,0.2);">
            <div style="font-size:0.7rem;color:#E10600;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;">Races</div>
            <div style="font-size:2.0rem;font-weight:800;color:white;">{races_completed}<span style="font-size:1rem;color:#666;">/{n_races}</span></div>
            <div style="font-size:0.7rem;color:#555;">Completed</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(0,210,190,0.12),rgba(0,210,190,0.04));
                    border-radius:12px;padding:1rem;text-align:center;
                    border:1px solid rgba(0,210,190,0.2);">
            <div style="font-size:0.7rem;color:#00D2BE;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;">Total Points</div>
            <div style="font-size:2.0rem;font-weight:800;color:white;">{total_pts:,}</div>
            <div style="font-size:0.7rem;color:#555;">Season Total</div>
        </div>
        """, unsafe_allow_html=True)

    drv_leader = drv_pts.index[0] if not drv_pts.empty else '-'
    drv_leader_pts = int(drv_pts.iloc[0]) if not drv_pts.empty else 0
    with k3:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(255,215,0,0.12),rgba(255,215,0,0.04));
                    border-radius:12px;padding:1rem;text-align:center;
                    border:1px solid rgba(255,215,0,0.2);">
            <div style="font-size:0.7rem;color:#FFD700;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;">Leader</div>
            <div style="font-size:1.4rem;font-weight:800;color:white;">{drv_leader}</div>
            <div style="font-size:1.0rem;color:#FFD700;">{drv_leader_pts} pts</div>
        </div>
        """, unsafe_allow_html=True)

    team_leader = team_pts.index[0] if not team_pts.empty else '-'
    team_leader_pts = int(team_pts.iloc[0]) if not team_pts.empty else 0
    with k4:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(107,203,119,0.12),rgba(107,203,119,0.04));
                    border-radius:12px;padding:1rem;text-align:center;
                    border:1px solid rgba(107,203,119,0.2);">
            <div style="font-size:0.7rem;color:#6BCB77;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;">Top Team</div>
            <div style="font-size:1.4rem;font-weight:800;color:white;">{team_leader}</div>
            <div style="font-size:1.0rem;color:#6BCB77;">{team_leader_pts} pts</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # === MAIN CONTENT - Two columns ===
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        # Driver Standings Leaderboard
        st.markdown("##### Championship Standings")
        if not drv_pts.empty:
            standings = []
            for rank, (driver, pts) in enumerate(drv_pts.head(10).items(), 1):
                team = race_df[race_df['Driver'] == driver]['Team'].iloc[0] if 'Team' in race_df.columns else ''
                color = TEAM_COLORS.get(team, '#555')
                wins = int((race_df[race_df['Driver'] == driver]['Position'] == 1).sum()) if 'Position' in race_df.columns else 0
                standings.append({'Rank': rank, 'Driver': driver, 'Team': team, 'Points': int(pts), 'Wins': wins})

            sdf = pd.DataFrame(standings)
            fig_stand = go.Figure()
            for _, row in sdf.iterrows():
                team_color = TEAM_COLORS.get(row['Team'], '#555')
                fig_stand.add_trace(go.Bar(
                    x=[row['Points']], y=[row['Driver']],
                    orientation='h',
                    marker=dict(color=team_color,
                                line=dict(color='rgba(255,255,255,0.3)', width=1)),
                    text=f"{row['Points']} pts",
                    textposition='outside', textfont=dict(size=10, color=TEXT_SECONDARY),
                    hovertemplate=(
                        f"<b>{row['Driver']}</b><br>"
                        f"Team: {row['Team']}<br>"
                        f"Points: {row['Points']}<br>"
                        f"Wins: {row['Wins']}"
                        "<extra></extra>"
                    ),
                ))

            fig_stand.update_layout(**_fig_layout(height=420,
                title=dict(text="Top 10 Drivers", font=TITLE_FONT, x=0.01),
                xaxis=dict(title=dict(text="Points", font=LABEL_FONT),
                           gridcolor=GRID, tickfont=dict(size=10, color=TEXT_SECONDARY)),
                yaxis=dict(autorange='reversed', tickfont=dict(size=11, color=TEXT_PRIMARY)),
                barmode='group', showlegend=False,
                margin=dict(l=80, r=80, t=40, b=30),
            ))
            show_plotly_chart(fig_stand)

    with col_right:
        # Constructor Standings
        st.markdown("##### Constructor Standings")
        if not team_pts.empty:
            tstand = []
            for rank, (team, pts) in enumerate(team_pts.items(), 1):
                tstand.append({'Rank': rank, 'Team': team, 'Points': int(pts)})

            tdf = pd.DataFrame(tstand)
            fig_t = go.Figure()
            for _, row in tdf.iterrows():
                team_color = TEAM_COLORS.get(row['Team'], '#555')
                fig_t.add_trace(go.Bar(
                    x=[row['Points']], y=[row['Team']],
                    orientation='h',
                    marker=dict(color=team_color,
                                line=dict(color='rgba(255,255,255,0.3)', width=1)),
                    text=f"{row['Points']} pts",
                    textposition='outside', textfont=dict(size=10, color=TEXT_SECONDARY),
                    hovertemplate=f"<b>{row['Team']}</b><br>Points: {row['Points']}<extra></extra>",
                ))

            fig_t.update_layout(**_fig_layout(height=420,
                title=dict(text="Constructors", font=TITLE_FONT, x=0.01),
                xaxis=dict(title=dict(text="Points", font=LABEL_FONT),
                           gridcolor=GRID, tickfont=dict(size=10, color=TEXT_SECONDARY)),
                yaxis=dict(autorange='reversed', tickfont=dict(size=11, color=TEXT_PRIMARY)),
                barmode='group', showlegend=False,
                margin=dict(l=150, r=80, t=40, b=30),
            ))
            show_plotly_chart(fig_t)

    st.markdown("---")

    # === Bottom Row: Race Results + Wins ===
    b1, b2 = st.columns([1, 1])

    with b1:
        st.markdown("##### Recent Race Results")
        if 'Position' in race_df.columns and 'Driver' in race_df.columns:
            last_race = race_df[race_df['Race'] == race_df['Race'].unique()[-1]] if 'Race' in race_df.columns else race_df
            top5 = last_race.sort_values('Position').head(5)
            if not top5.empty:
                for _, r in top5.iterrows():
                    pos = int(r['Position'])
                    drv = r['Driver']
                    team = r.get('Team', '')
                    pts = int(r['Points'])
                    color = TEAM_COLORS.get(team, '#555')
                    medal = {1: '#FFD700', 2: '#C0C0C0', 3: '#CD7F32'}.get(pos, 'transparent')
                    medal_icon = {1: '🥇', 2: '🥈', 3: '🥉'}.get(pos, f'#{pos}')
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:12px;padding:6px 8px;
                                margin:1px 0;border-radius:6px;
                                background:rgba(255,255,255,0.03);">
                        <span style="font-size:1.1rem;">{medal_icon}</span>
                        <span style="font-weight:700;color:white;min-width:36px;">{drv}</span>
                        <span style="font-size:0.8rem;color:#888;flex:1;">{team}</span>
                        <span style="font-weight:600;color:{color};">{pts} pts</span>
                    </div>
                    """, unsafe_allow_html=True)

    with b2:
        st.markdown("##### Win Count")
        if 'Driver' in race_df.columns and 'Position' in race_df.columns:
            winners = race_df[race_df['Position'] == 1]['Driver'].value_counts().head(8)
            if not winners.empty:
                fig_w = go.Figure()
                for drv, wins in winners.items():
                    team = race_df[race_df['Driver'] == drv]['Team'].iloc[0] if 'Team' in race_df.columns else ''
                    color = TEAM_COLORS.get(team, '#555')
                    fig_w.add_trace(go.Bar(
                        x=[wins], y=[drv], orientation='h',
                        marker=dict(color=color,
                                    line=dict(color='rgba(255,255,255,0.3)', width=1)),
                        text=str(int(wins)), textposition='outside',
                        textfont=dict(size=12, color='white', weight='bold'),
                        hovertemplate=f"<b>{drv}</b><br>Wins: {int(wins)}<extra></extra>",
                    ))
                fig_w.update_layout(**_fig_layout(height=320,
                    title=dict(text="Race Winners", font=TITLE_FONT, x=0.01),
                    xaxis=dict(title=dict(text="Wins", font=LABEL_FONT),
                               tickfont=dict(size=10, color=TEXT_SECONDARY)),
                    yaxis=dict(autorange='reversed', tickfont=dict(size=11, color=TEXT_PRIMARY)),
                    showlegend=False,
                    margin=dict(l=60, r=50, t=40, b=30),
                ))
                show_plotly_chart(fig_w)

    # === Points Progression ===
    st.markdown("---")
    st.markdown("##### Points Progression")

    if 'Driver' in race_df.columns and 'Points' in race_df.columns:
        try:
            # Get top drivers that actually exist in race_df
            available_drivers = [d for d in drv_pts.head(8).index if d in race_df['Driver'].values]
            top_drivers = available_drivers[:5]
            if not top_drivers:
                st.info("Insufficient data for progression chart.")
                return

            race_key = 'Race' if 'Race' in race_df.columns else 'GrandPrix'

            prog = {}
            for d in top_drivers:
                dd = race_df[race_df['Driver'] == d].sort_values(race_key) if race_key in race_df.columns else race_df[race_df['Driver'] == d]
                if dd.empty:
                    continue
                running = 0
                prog[d] = []
                for _, r in dd.iterrows():
                    running += r['Points']
                    race_name = r.get(race_key, '') if race_key in r else ''
                    prog[d].append({'RaceNum': len(prog[d]) + 1, 'RaceName': str(race_name), 'Points': running})

            if not prog:
                st.info("Could not build progression data.")
                return

            fig_p = go.Figure()
            for d, pts_list in prog.items():
                pdf = pd.DataFrame(pts_list)
                team_rows = race_df[race_df['Driver'] == d]
                team = team_rows['Team'].iloc[0] if 'Team' in team_rows.columns and not team_rows.empty else ''
                color = TEAM_COLORS.get(team, '#888')
                fig_p.add_trace(go.Scatter(
                    x=pdf['RaceNum'], y=pdf['Points'], mode='lines+markers',
                    name=d, line=dict(color=color, width=3, shape='spline', smoothing=0.4),
                    marker=dict(size=9, color=color, line=dict(width=1.5, color='white')),
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "%{customdata[0]}: %{y} pts"
                        "<extra></extra>"
                    ),
                    customdata=pdf[['RaceName', 'Points']].values.tolist(),
                ))

            # Build readable tick labels using the driver with most races
            max_races = max(len(v) for v in prog.values()) if prog else 0
            tick_vals = list(range(1, max_races + 1))
            # Find which driver has data for all races
            best_driver = max(prog, key=lambda k: len(prog[k])) if prog else None
            if best_driver and best_driver in prog:
                tick_text = [p['RaceName'][:10] for p in prog[best_driver]]
            else:
                tick_text = [str(i) for i in tick_vals]

            fig_p.update_layout(**_fig_layout(height=450,
                title=dict(text="Points Accumulation Over Season", font=TITLE_FONT, x=0.01),
                xaxis=dict(title=dict(text="Race", font=LABEL_FONT),
                           tickmode='array', tickvals=tick_vals, ticktext=tick_text,
                           gridcolor=GRID, tickfont=dict(size=10, color=TEXT_SECONDARY),
                           tickangle=-30),
                yaxis=dict(title=dict(text="Points", font=LABEL_FONT),
                           gridcolor=GRID, tickfont=dict(size=11, color=TEXT_SECONDARY),
                           zeroline=True, zerolinecolor='rgba(255,255,255,0.1)'),
                hovermode='x unified',
                legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
                margin=dict(l=55, r=30, t=55, b=80),
            ))
            show_plotly_chart(fig_p)
        except Exception as e:
            st.info(f"Progression unavailable: {e}")


page()
