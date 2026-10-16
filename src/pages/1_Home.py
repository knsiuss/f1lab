import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config import FASTF1_CONFIG, TEAM_COLORS
from season_config import get_completed_races, get_race_names
from shared import empty_state, load_race_data, show_plotly_chart
from styles import (
    GRID, hero_data_uri, icon, panel_head, podium_color,
)

def _team_color(team: str, fallback: str = "#8a8f98") -> str:
    return TEAM_COLORS.get(team, fallback)




def _sparkline_svg(points, width=140, height=34) -> str:
    """Tiny cumulative-points sparkline as pure SVG — no JS chart overhead."""
    if not points:
        return ""
    low, high = min(points), max(points)
    span = (high - low) or 1
    step = width / max(len(points) - 1, 1)
    coords = [
        (i * step, height - 4 - (p - low) / span * (height - 10))
        for i, p in enumerate(points)
    ]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area = f"0,{height} {polyline} {width},{height}"
    return f"""<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none">
  <defs><linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#D6F84C" stop-opacity=".32"/>
    <stop offset="100%" stop-color="#D6F84C" stop-opacity="0"/>
  </linearGradient></defs>
  <polygon points="{area}" fill="url(#sparkGrad)"/>
  <polyline points="{polyline}" fill="none" stroke="#D6F84C" stroke-width="2"
            stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


def _kpi(label, name, value, sub="", delta="", fill=None, spark="") -> str:
    spark_html = f'<div class="kpi-spark">{spark}</div>' if spark else ""
    track_html = ""
    if fill is not None:
        track_html = (
            f'<div class="kpi-track"><div class="kpi-fill" '
            f'style="width:{min(max(fill, 0), 100):.0f}%"></div></div>'
        )
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""<div class="kpi">
  <div class="kpi-head">
    <div class="kpi-icon">{icon(name)}</div>
    <div class="kpi-label">{label}</div>
  </div>
  <div class="kpi-value">{value}</div>
  {delta_html}{sub_html}{track_html}{spark_html}
</div>"""


def _stand_row(rank, driver, team, points, bar_pct, bar_color) -> str:
    medal = f"p{rank}" if rank <= 3 else ""
    sub_html = f'<div class="stand-team">{team}</div>' if team else ""
    return f"""<div class="stand-row">
  <div class="stand-rank {medal}">{rank}</div>
  <div>
    <div class="stand-driver">{driver}</div>
    {sub_html}
  </div>
  <div>
    <div class="stand-bar"><div class="stand-fill" style="width:{bar_pct:.1f}%;background:linear-gradient(90deg,{bar_color}99,{bar_color})"></div></div>
  </div>
  <div class="stand-pts">{points}</div>
</div>"""




def page():
    year = st.session_state.get("selected_year", FASTF1_CONFIG.default_year)
    df = load_race_data(year)

    if df is None or df.empty:
        empty_state(
            f"No {year} results yet",
            "Results appear here once races complete. For live session data, open Analysis Center.",
        )
        return

    race_names = get_race_names(year)
    races_done = get_completed_races(year)
    n_races = max(len(race_names), len(races_done)) if race_names is not None else len(races_done)
    races_completed = len(races_done)

    race_df = df[df["SessionType"] == "Race"].copy() if "SessionType" in df.columns else df.copy()

    drv_pts = (
        race_df.groupby("Driver")["Points"].sum().sort_values(ascending=False)
        if "Points" in race_df.columns
        else pd.Series(dtype=float)
    )
    team_pts = (
        race_df.groupby("Team")["Points"].sum().sort_values(ascending=False)
        if "Team" in race_df.columns and "Points" in race_df.columns
        else pd.Series(dtype=float)
    )

    # Delays are read straight off the standings so the copy stays honest.
    leader_gap = int(drv_pts.iloc[0] - drv_pts.iloc[1]) if len(drv_pts) > 1 else 0
    team_gap = int(team_pts.iloc[0] - team_pts.iloc[1]) if len(team_pts) > 1 else 0
    total_pts = int(drv_pts.sum()) if not drv_pts.empty else 0

    _render_hero(year, races_completed, n_races)
    _render_kpis(race_df, drv_pts, team_pts, races_completed, n_races,
                 total_pts, leader_gap, team_gap)
    _render_standings(race_df, drv_pts, team_pts)
    _render_bottom(race_df, drv_pts)
    _render_about()


def _render_hero(year, races_completed, n_races):
    hero_uri = hero_data_uri()
    bg = (
        f"background-image:linear-gradient(90deg,rgba(8,8,12,.97) 0%,rgba(8,8,12,.86) 42%,"
        f"rgba(8,8,12,.35) 78%),url('{hero_uri}');"
        if hero_uri
        else "background:linear-gradient(120deg,#101018,#0a0a10);"
    )
    st.markdown(f"""
    <div class="hero-top">
        <span class="eyebrow">Championship Data Analysis</span>
        <span class="pill">{icon('calendar', 13)}&nbsp; {year} Season</span>
    </div>
    <div class="hero" style="{bg}">
        <h1 class="hero-title">F1 {year}<br><span class="accent">Season Overview</span></h1>
        <p class="hero-sub">Season results, standings, and key insights from the
        {year} Formula 1 Championship — {races_completed} of {n_races} races on the books.</p>
        <div class="hero-tag">Faster data · deeper insights</div>
    </div>
    """, unsafe_allow_html=True)


def _render_kpis(race_df, drv_pts, team_pts, races_completed, n_races,
                 total_pts, leader_gap, team_gap):
    completion = (races_completed / n_races * 100) if n_races else 0
    spark = _sparkline_svg(_points_per_round(race_df))

    drv_leader = drv_pts.index[0] if not drv_pts.empty else "—"
    team_leader = team_pts.index[0] if not team_pts.empty else "—"

    cards = [
        _kpi("Races", "flag", f'{races_completed}<span class="kpi-dim">/{n_races}</span>',
             sub="Completed", fill=completion, spark=spark),
        _kpi("Total Points", "trending", f"{total_pts:,}", sub="Season total", fill=completion),
        _kpi("Championship Leader", "award", drv_leader,
             sub=f'{int(drv_pts.iloc[0]) if not drv_pts.empty else 0} pts',
             delta=f"+{leader_gap} pts vs P2"),
        _kpi("Top Team", "users", team_leader,
             sub=f'{int(team_pts.iloc[0]) if not team_pts.empty else 0} pts',
             delta=f"+{team_gap} pts vs P2"),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _points_per_round(race_df):
    """Season points scored per round, in race order — feeds the KPI sparkline."""
    if "Points" not in race_df.columns or "Race" not in race_df.columns:
        return []
    order = list(dict.fromkeys(race_df["Race"]))
    per_round = race_df.groupby("Race")["Points"].sum()
    return [int(per_round.get(r, 0)) for r in order]


def _render_standings(race_df, drv_pts, team_pts):
    left, right = st.columns(2)

    with left:
        st.markdown(
            panel_head("trophy", "Driver Championship Standings",
                        f"Top 10 · {len(drv_pts)} drivers scored", "Top 10")
            + "".join(_driver_rows(race_df, drv_pts))
            + "</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            panel_head("users", "Constructor Standings",
                        f"Top 10 · {len(team_pts)} teams", "Full standings")
            + "".join(_constructor_rows(team_pts))
            + "</div>",
            unsafe_allow_html=True,
        )


def _driver_rows(race_df, drv_pts):
    rows = []
    top_points = drv_pts.iloc[0] if not drv_pts.empty else 1
    for rank, (driver, pts) in enumerate(drv_pts.head(10).items(), 1):
        team = ""
        if "Team" in race_df.columns:
            team_rows = race_df[race_df["Driver"] == driver]["Team"]
            team = team_rows.iloc[0] if not team_rows.empty else ""
        color = _team_color(team)
        rows.append(_stand_row(rank, driver, team, int(pts),
                               int(pts) / max(top_points, 1) * 100, color))
    return rows


def _constructor_rows(team_pts):
    rows = []
    top_points = team_pts.iloc[0] if not team_pts.empty else 1
    for rank, (team, pts) in enumerate(team_pts.head(10).items(), 1):
        rows.append(_stand_row(rank, team, "", int(pts),
                               int(pts) / max(top_points, 1) * 100, _team_color(team)))
    return rows


def _render_bottom(race_df, drv_pts):
    left, right = st.columns([0.85, 1.6])

    with left:
        st.markdown(
            panel_head("flag", "Recent Race Results", "Latest completed round")
            + _result_rows(race_df)
            + "</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            panel_head("trending", "Points Progression",
                        "Cumulative points, top 5 drivers", "All drivers")
            + "</div>",
            unsafe_allow_html=True,
        )
        _progression_chart(race_df, drv_pts)


def _result_rows(race_df):
    if "Position" not in race_df.columns or "Driver" not in race_df.columns:
        return '<div class="result-row"><div class="stand-team">No results available</div></div>'
    race_col = "Race" if "Race" in race_df.columns else None
    last_race = race_df[race_col].iloc[-1] if race_col else None
    latest = race_df[race_df[race_col] == last_race] if race_col else race_df
    top5 = latest.sort_values("Position").head(5)

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    rows = []
    for _, r in top5.iterrows():
        pos = int(r["Position"])
        team = r.get("Team", "")
        color = _team_color(team)
        medal = medals.get(pos, f"#{pos}")
        rank_style = f"color:{podium_color(pos)};font-weight:800;" if pos <= 3 else "font-weight:700;"
        rows.append(f"""<div class="result-row">
  <div class="result-medal" style="{rank_style}">{medal}</div>
  <div>
    <div class="result-driver">{r['Driver']}</div>
    <div class="result-team">{team}</div>
  </div>
  <div class="result-pts" style="color:{color}">{int(r['Points'])} pts</div>
  <div class="result-chevron">{icon('chevron', 14)}</div>
</div>""")
    return "".join(rows)


def _progression_chart(race_df, drv_pts):
    if "Driver" not in race_df.columns or "Points" not in race_df.columns:
        return

    race_key = "Race" if "Race" in race_df.columns else "GrandPrix"
    top_drivers = [
        d for d in drv_pts.head(8).index if d in race_df["Driver"].values
    ][:5]
    if not top_drivers:
        return

    fig = go.Figure()
    for driver in top_drivers:
        dd = race_df[race_df["Driver"] == driver]
        if race_key in race_df.columns:
            dd = dd.sort_values(race_key)
        running, points, labels = 0, [], []
        for _, r in dd.iterrows():
            running += r["Points"]
            points.append(running)
            labels.append(str(r.get(race_key, "")) if race_key in r else "")
        team_rows = race_df[race_df["Driver"] == driver]["Team"]
        color = _team_color(team_rows.iloc[0] if not team_rows.empty else "")
        fig.add_trace(go.Scatter(
            x=list(range(1, len(points) + 1)),
            y=points,
            mode="lines+markers",
            name=driver,
            line=dict(color=color, width=3, shape="spline", smoothing=0.4),
            marker=dict(size=8, color=color, line=dict(width=1.5, color="#0b0b10")),
            customdata=labels,
            hovertemplate="<b>%{customdata}</b><br>%{y} pts<extra></extra>",
        ))

    fig.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#A4AAB4", size=11),
        margin=dict(l=45, r=15, t=20, b=40),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(14,14,20,.94)", font_size=12,
                        font_family="Inter", bordercolor="rgba(255,255,255,.12)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#A4AAB4")),
        xaxis=dict(showgrid=False, color="#5C6470", tickfont=dict(size=10, color="#5C6470")),
        yaxis=dict(showgrid=True, gridcolor=GRID, color="#5C6470",
                   zerolinecolor="rgba(214,248,76,.25)",
                   tickfont=dict(size=10, color="#5C6470")),
    )
    show_plotly_chart(fig, width="stretch")


def _render_about():
    st.markdown(f"""
    <div class="panel" style="margin-top:1.1rem;">
        {panel_head("info", "About F1 Lab & What It Is Not", "How to read the numbers")}
        <div class="about-grid">
            <div class="about-col">
                <h5>What this is</h5>
                <p>A public-data analytics workbench for post-session analysis, strategy
                rehearsal, race briefing and competitor research. It reads publicly
                available season results and lap-level session data, and turns them into
                explainable pace, strategy and prediction views.</p>
            </div>
            <div class="about-col">
                <h5>What it is not</h5>
                <p>Not a race-critical tool. Not live pit-wall software, and it includes no
                team's internal telemetry, setup or car information. Nothing here
                substitutes official timing or a team's operational data.</p>
            </div>
            <div class="about-col">
                <h5>Estimated means estimated</h5>
                <p>Anything derived rather than directly measured is labelled
                <em>estimated</em> with a confidence level. When a sample is too small,
                too noisy or missing, the app says <em>insufficient data</em> instead of
                fabricating an answer.</p>
            </div>
        </div>
        <div style="margin-top:0.9rem;font-size:0.66rem;color:#5C6470;">
            Hero photo: Ricciardo/Magnussen at Luffield, 2022 British GP — fuji.tim,
            CC BY-SA 2.0, via Flickr/Openverse.
        </div>
    </div>
    """, unsafe_allow_html=True)


page()
