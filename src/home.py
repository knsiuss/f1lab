# -*- coding: utf-8 -*-
"""
home.py
~~~~~~~
Dashboard home tab with schedule display.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Tuple

import fastf1
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from config import TEAM_COLORS
except Exception:  # pragma: no cover
    TEAM_COLORS = {}

try:
    from f1_2026_updates import OFFICIAL_2026_CALENDAR
except Exception:  # pragma: no cover
    OFFICIAL_2026_CALENDAR = []


def _get_active_season_year() -> int:
    mode = str(st.session_state.get("sidebar_season_mode", "2025"))
    return 2026 if mode.startswith("2026") else 2025


def _get_user_utc_offset() -> str:
    return str(st.session_state.get("user_utc_offset", "+00:00"))


def _apply_user_utc_offset(ts: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(ts):
        return ts
    if not isinstance(ts, pd.Timestamp):
        ts = pd.to_datetime(ts, errors="coerce", utc=True)
    if pd.isna(ts):
        return ts
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    offset = _get_user_utc_offset()
    sign = -1 if offset.startswith("-") else 1
    hh_mm = offset[1:] if offset and offset[0] in "+-" else offset
    try:
        hours, minutes = hh_mm.split(":")
        delta = pd.Timedelta(hours=int(hours), minutes=int(minutes)) * sign
        return ts + delta
    except Exception:
        return ts


def _fmt_user_time(ts: pd.Timestamp, fmt: str = "%Y-%m-%d %H:%M") -> str:
    adj = _apply_user_utc_offset(ts)
    if pd.isna(adj):
        return "N/A"
    return f"{adj.strftime(fmt)} (UTC{_get_user_utc_offset()})"


def _countdown_parts(target_ts: pd.Timestamp, now: pd.Timestamp) -> dict:
    target_ts = pd.to_datetime(target_ts, errors="coerce", utc=True)
    now = pd.to_datetime(now, errors="coerce", utc=True)
    if pd.isna(target_ts) or pd.isna(now):
        return {"total_seconds": None, "days": None, "hours": None, "minutes": None, "seconds": None}
    total_seconds = max(0, int((target_ts - now).total_seconds()))
    return {
        "total_seconds": total_seconds,
        "days": total_seconds // 86400,
        "hours": (total_seconds % 86400) // 3600,
        "minutes": (total_seconds % 3600) // 60,
        "seconds": total_seconds % 60,
    }


def _countdown_text(target_ts: pd.Timestamp, now: pd.Timestamp) -> str:
    p = _countdown_parts(target_ts, now)
    if p["total_seconds"] is None:
        return "N/A"
    return f"{p['days']}d {p['hours']}h {p['minutes']}m {p['seconds']}s"


@st.cache_data(ttl=1800)
def _get_schedule_cached(year: int) -> pd.DataFrame:
    try:
        schedule = fastf1.get_event_schedule(year)
        return schedule.copy() if schedule is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _fallback_schedule_2026() -> pd.DataFrame:
    if not OFFICIAL_2026_CALENDAR:
        return pd.DataFrame()
    rows = []
    for item in OFFICIAL_2026_CALENDAR:
        rows.append(
            {
                "RoundNumber": item.get("round"),
                "EventName": item.get("event"),
                "Location": item.get("location"),
                "Country": item.get("country"),
                "EventDate": pd.to_datetime(item.get("race_date"), errors="coerce", utc=True),
            }
        )
    return pd.DataFrame(rows)


def _get_schedule_with_fallback(year: int) -> Tuple[pd.DataFrame, str]:
    schedule = _get_schedule_cached(year)
    if isinstance(schedule, pd.DataFrame) and not schedule.empty:
        return schedule.copy(), "Live Schedule"
    if year == 2026:
        fb = _fallback_schedule_2026()
        if not fb.empty:
            return fb, "Official Fallback"
    return pd.DataFrame(), "Unavailable"


@st.cache_data(ttl=3600)
def _load_completed_season_results(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(__file__).parent.parent / "data"
    race_file = data_dir / f"Formula1_{year}Season_RaceResults.csv"
    sprint_file = data_dir / f"Formula1_{year}Season_SprintResults.csv"
    if not race_file.exists():
        return pd.DataFrame(), pd.DataFrame()
    race_df = pd.read_csv(race_file)
    sprint_df = pd.read_csv(sprint_file) if sprint_file.exists() else pd.DataFrame()
    return race_df, sprint_df


def _render_completed_season_home(year: int) -> bool:
    race_df, sprint_df = _load_completed_season_results(year)
    if race_df.empty:
        return False

    race_df = race_df.copy()
    if "Track" in race_df.columns:
        race_order = race_df["Track"].dropna().astype(str).drop_duplicates().tolist()
    else:
        race_order = []

    sprint_df = sprint_df.copy() if isinstance(sprint_df, pd.DataFrame) else pd.DataFrame()

    race_points = race_df.groupby("Driver", dropna=False)["Points"].sum() if "Points" in race_df.columns else pd.Series(dtype=float)
    sprint_points = sprint_df.groupby("Driver", dropna=False)["Points"].sum() if not sprint_df.empty and "Points" in sprint_df.columns else pd.Series(dtype=float)
    driver_points = race_points.add(sprint_points, fill_value=0).sort_values(ascending=False).reset_index()
    driver_points.columns = ["Driver", "TotalPoints"]

    race_team_points = race_df.groupby("Team", dropna=False)["Points"].sum() if "Points" in race_df.columns else pd.Series(dtype=float)
    sprint_team_points = sprint_df.groupby("Team", dropna=False)["Points"].sum() if not sprint_df.empty and "Points" in sprint_df.columns else pd.Series(dtype=float)
    constructor_points = race_team_points.add(sprint_team_points, fill_value=0).sort_values(ascending=False).reset_index()
    constructor_points.columns = ["Team", "TotalPoints"]

    champion_driver = driver_points.iloc[0]["Driver"] if not driver_points.empty else "N/A"
    champion_points = int(driver_points.iloc[0]["TotalPoints"]) if not driver_points.empty else 0
    champion_team = constructor_points.iloc[0]["Team"] if not constructor_points.empty else "N/A"
    champion_team_points = int(constructor_points.iloc[0]["TotalPoints"]) if not constructor_points.empty else 0
    total_races = len(race_order) if race_order else int(race_df["Track"].nunique()) if "Track" in race_df.columns else 0
    total_drivers = int(race_df["Driver"].nunique()) if "Driver" in race_df.columns else 0
    total_teams = int(race_df["Team"].nunique()) if "Team" in race_df.columns else 0

    st.markdown(
        f"""
        <div class="home-hero">
          <div class="home-badges">
            <span class="home-badge red">FINAL RESULTS</span>
            <span class="home-badge teal">Season {year}</span>
            <span class="home-badge amber">Completed</span>
          </div>
          <div class="home-title">{year} Final Championship Recap</div>
          <div class="home-sub">Season complete. Home dashboard now shows final standings and interactive season-end analytics.</div>
          <div class="home-count-grid">
            <div class="home-count-card">
              <div class="home-count-label">Drivers' Champion</div>
              <div class="home-count-value" style="font-size:16px;">{champion_driver}</div>
              <div class="home-count-note">{champion_points} points</div>
            </div>
            <div class="home-count-card">
              <div class="home-count-label">Constructors' Champion</div>
              <div class="home-count-value" style="font-size:16px;">{champion_team}</div>
              <div class="home-count-note">{champion_team_points} points</div>
            </div>
            <div class="home-count-card">
              <div class="home-count-label">Races</div>
              <div class="home-count-value">{total_races}</div>
              <div class="home-count-note">completed rounds</div>
            </div>
            <div class="home-count-card">
              <div class="home-count-label">Drivers</div>
              <div class="home-count-value">{total_drivers}</div>
              <div class="home-count-note">scored in season</div>
            </div>
            <div class="home-count-card">
              <div class="home-count-label">Teams</div>
              <div class="home-count-value">{total_teams}</div>
              <div class="home-count-note">constructors</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Race Winner Events", int((pd.to_numeric(race_df.get("Position"), errors="coerce") == 1).sum()) if "Position" in race_df.columns else 0)
    k2.metric("Sprint Rounds", int(sprint_df["Track"].nunique()) if not sprint_df.empty and "Track" in sprint_df.columns else 0)
    k3.metric("Total Race Points", int(pd.to_numeric(race_df.get("Points"), errors="coerce").fillna(0).sum()) if "Points" in race_df.columns else 0)

    left, right = st.columns(2)
    with left:
        top_n = st.slider("Top Drivers", 5, 20, 10, key=f"home_final_{year}_top_drivers")
        show_drivers = driver_points.head(top_n).copy()
        if not show_drivers.empty:
            team_map = race_df.groupby("Driver")["Team"].first().to_dict() if {"Driver", "Team"}.issubset(race_df.columns) else {}
            show_drivers["Team"] = show_drivers["Driver"].map(team_map).fillna("")
            fig = px.bar(
                show_drivers,
                x="TotalPoints",
                y="Driver",
                orientation="h",
                color="Team" if "Team" in show_drivers.columns else None,
                title="Final Drivers' Championship",
                text="TotalPoints",
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=max(360, 32 * len(show_drivers)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(l=10, r=10, t=55, b=20),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

    with right:
        top_t = st.slider("Top Teams", 5, max(5, min(12, len(constructor_points))), min(10, max(5, len(constructor_points))), key=f"home_final_{year}_top_teams")
        show_teams = constructor_points.head(top_t).copy()
        if not show_teams.empty:
            show_teams["Color"] = show_teams["Team"].map(lambda t: TEAM_COLORS.get(str(t), "#888888"))
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=show_teams["TotalPoints"],
                    y=show_teams["Team"],
                    orientation="h",
                    marker_color=show_teams["Color"],
                    text=show_teams["TotalPoints"].astype(int),
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Final Constructors' Championship",
                yaxis={"categoryorder": "total ascending"},
                height=max(360, 34 * len(show_teams)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                margin=dict(l=10, r=20, t=55, b=20),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

    st.markdown("#### Championship Progression (Final Season)")
    if race_order and not driver_points.empty:
        n_progress = st.slider("Drivers in Progression Chart", 3, min(12, len(driver_points)), min(6, len(driver_points)), key=f"home_final_{year}_progress_n")
        top_drivers = driver_points.head(n_progress)["Driver"].tolist()

        fig = go.Figure()
        for drv in top_drivers:
            r = race_df[race_df["Driver"] == drv].copy()
            if r.empty:
                continue
            r["Track"] = r["Track"].astype(str)
            r["Track"] = pd.Categorical(r["Track"], categories=race_order, ordered=True)
            r = r.sort_values("Track")
            base_points = pd.to_numeric(r["Points"], errors="coerce").fillna(0)
            if not sprint_df.empty and {"Driver", "Track", "Points"}.issubset(sprint_df.columns):
                s = sprint_df[sprint_df["Driver"] == drv].copy()
                sprint_map = s.groupby("Track")["Points"].sum().to_dict()
                sprint_add = r["Track"].astype(str).map(lambda x: sprint_map.get(x, 0)).fillna(0)
            else:
                sprint_add = pd.Series([0] * len(r), index=r.index)
            cum_points = (base_points + sprint_add).cumsum()
            team_name = r["Team"].iloc[0] if "Team" in r.columns and not r.empty else ""
            color = TEAM_COLORS.get(str(team_name), "#888888")
            fig.add_trace(
                go.Scatter(
                    x=[str(x) for x in r["Track"].tolist()],
                    y=cum_points,
                    mode="lines+markers",
                    name=drv,
                    line=dict(color=color, width=2),
                    marker=dict(size=7),
                    hovertemplate="<b>%{x}</b><br>%{y:.0f} pts<extra>" + drv + "</extra>",
                )
            )
        fig.update_layout(
            height=420,
            xaxis_title="Grand Prix",
            yaxis_title="Cumulative Points",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(l=10, r=10, t=10, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Race Winners Timeline")
        if {"Track", "Driver", "Team", "Position"}.issubset(race_df.columns):
            winners = race_df[pd.to_numeric(race_df["Position"], errors="coerce") == 1].copy()
            if not winners.empty:
                if race_order:
                    winners["Track"] = pd.Categorical(winners["Track"].astype(str), categories=race_order, ordered=True)
                    winners = winners.sort_values("Track")
                winners["Round"] = range(1, len(winners) + 1)
                fig = px.scatter(
                    winners,
                    x="Round",
                    y="Driver",
                    color="Team",
                    size_max=14,
                    hover_data=["Track", "Points"],
                    title="Grand Prix Winners by Round",
                )
                fig.update_layout(
                    height=360,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    margin=dict(l=10, r=10, t=55, b=20),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})
    with c2:
        st.markdown("#### Wins / Podiums Distribution")
        if {"Driver", "Position"}.issubset(race_df.columns):
            pos = pd.to_numeric(race_df["Position"], errors="coerce")
            dist = pd.DataFrame(
                {
                    "Wins": race_df[pos == 1].groupby("Driver").size(),
                    "Podiums": race_df[pos <= 3].groupby("Driver").size(),
                    "Top10": race_df[pos <= 10].groupby("Driver").size(),
                }
            ).fillna(0).reset_index()
            if not dist.empty:
                metric_pick = st.selectbox("Distribution Metric", ["Wins", "Podiums", "Top10"], key=f"home_final_{year}_dist_metric")
                dist = dist.sort_values(metric_pick, ascending=False).head(10)
                fig = px.bar(dist, x="Driver", y=metric_pick, color=metric_pick, title=f"Top 10 Drivers by {metric_pick}")
                fig.update_layout(
                    height=360,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    margin=dict(l=10, r=10, t=55, b=30),
                    xaxis_title="",
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

    with st.expander("Final Championship Tables"):
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Drivers Final Standings (local results)**")
            st.dataframe(driver_points, hide_index=True, use_container_width=True)
        with t2:
            st.markdown("**Constructors Final Standings (local results)**")
            st.dataframe(constructor_points, hide_index=True, use_container_width=True)

    return True


def _inject_home_styles() -> None:
    st.markdown(
        """
        <style>
        .home-hero {
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 18px;
            padding: 16px 16px 12px 16px;
            margin-bottom: 14px;
            background:
                radial-gradient(900px 260px at 0% 0%, rgba(225,6,0,0.20), transparent 60%),
                radial-gradient(700px 220px at 100% 0%, rgba(0,210,190,0.14), transparent 60%),
                linear-gradient(135deg, rgba(17,19,27,0.95), rgba(26,31,42,0.92));
        }
        .home-badges { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px; }
        .home-badge {
            display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px;
            font-weight:700; border:1px solid rgba(255,255,255,0.14);
        }
        .home-badge.red { background: rgba(225,6,0,0.18); color: #ffd1ce; border-color: rgba(225,6,0,0.45); }
        .home-badge.teal { background: rgba(0,210,190,0.16); color: #bff8f2; border-color: rgba(0,210,190,0.40); }
        .home-badge.amber { background: rgba(245,185,66,0.16); color: #ffe5a8; border-color: rgba(245,185,66,0.40); }
        .home-title { font-size: 24px; font-weight: 800; line-height:1.15; margin: 0 0 6px 0; }
        .home-sub { color:#b8c2d4; font-size: 14px; margin-bottom: 10px; }
        .home-count-grid { display:grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap:10px; }
        .home-count-card { border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:10px 12px; background:rgba(255,255,255,0.02); }
        .home-count-label { color:#aeb8c8; font-size:11px; text-transform: uppercase; letter-spacing:1px; }
        .home-count-value { color:#f2f5fb; font-size:20px; font-weight:800; }
        .home-count-note { color:#b8c2d4; font-size:12px; }
        .home-inline-badges { display:flex; flex-wrap:wrap; gap:8px; margin: 8px 0 10px 0; }
        .home-status-chip {
            display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:700;
            border:1px solid rgba(255,255,255,0.12);
        }
        .home-status-chip.live { background: rgba(225,6,0,0.18); color:#ffd1ce; border-color: rgba(225,6,0,0.45); }
        .home-status-chip.upcoming { background: rgba(0,210,190,0.16); color:#bff8f2; border-color: rgba(0,210,190,0.40); }
        .home-status-chip.done { background: rgba(255,255,255,0.06); color:#d8deea; border-color: rgba(255,255,255,0.16); }
        .home-radar-grid { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:10px; margin-top:8px; }
        .home-radar-card {
            border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:10px 12px;
            background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        }
        .home-radar-title { font-weight:700; color:#f2f5fb; margin-bottom:2px; }
        .home-radar-meta { font-size:12px; color:#b8c2d4; margin-bottom:6px; }
        .home-radar-count { font-size:13px; color:#e9eef7; }
        .home-radar-chip {
            display:inline-block; margin-bottom:6px; padding:3px 8px; border-radius:999px; font-size:11px; font-weight:700;
            border:1px solid rgba(255,255,255,0.12);
        }
        .home-radar-chip.urgent { background: rgba(225,6,0,0.18); color:#ffd1ce; border-color: rgba(225,6,0,0.45); }
        .home-radar-chip.soon { background: rgba(245,185,66,0.16); color:#ffe5a8; border-color: rgba(245,185,66,0.38); }
        .home-radar-chip.normal { background: rgba(0,210,190,0.14); color:#bff8f2; border-color: rgba(0,210,190,0.35); }
        @media (max-width: 900px) {
            .home-count-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
            .home-radar-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_hero_badges(active_year: int, is_live: bool, days_to_next: int | None, schedule_source: str) -> str:
    badges = []
    badges.append('<span class="home-badge red">LIVE NOW</span>' if is_live else '<span class="home-badge teal">COUNTDOWN ACTIVE</span>')
    if days_to_next is not None and 0 <= days_to_next <= 7:
        badges.append('<span class="home-badge amber">RACE WEEK</span>')
    badges.append(f'<span class="home-badge teal">Season {active_year}</span>')
    badges.append(f'<span class="home-badge amber">{schedule_source}</span>')
    return "".join(badges)


def render_home_tab():
    """Tab 0: Home - Dynamic Content based on Event Status."""
    active_year = _get_active_season_year()
    _inject_home_styles()
    st.header(f"F1 Dashboard Home ({active_year})")

    now = pd.Timestamp.now(tz="UTC")
    current_year = int(now.year)

    # Completed seasons should show final results dashboard instead of upcoming countdown.
    if active_year < current_year:
        if _render_completed_season_home(active_year):
            return

    try:
        schedule, schedule_source = _get_schedule_with_fallback(active_year)
        if schedule is None or schedule.empty:
            st.info(f"Calendar sync is not available right now for {active_year}.")
            return

        if "EventDate" not in schedule.columns:
            st.info(f"Calendar sync is loading for {active_year}.")
            return
        if schedule["EventDate"].dt.tz is None:
            schedule["EventDate"] = schedule["EventDate"].dt.tz_localize("UTC")

        upcoming_events = schedule[schedule["EventDate"] >= (now - timedelta(days=3))].sort_values("EventDate")
        next_future_events = schedule[schedule["EventDate"] >= now].sort_values("EventDate")

        if upcoming_events.empty:
            if _render_completed_season_home(active_year):
                return
            st.info(f"No upcoming events found in {active_year} calendar.")
            return

        target_event = upcoming_events.iloc[0]
        next_race_event = next_future_events.iloc[0] if not next_future_events.empty else target_event
        event_name = str(target_event.get("EventName", "Next Event"))
        round_num = target_event.get("RoundNumber", "N/A")
        location = str(target_event.get("Location", "TBA"))
        event_date = pd.to_datetime(target_event.get("EventDate"), errors="coerce", utc=True)
        next_race_dt = pd.to_datetime(next_race_event.get("EventDate"), errors="coerce", utc=True)
        countdown = _countdown_parts(next_race_dt, now)
        days_to_next = None if pd.isna(next_race_dt) else int((next_race_dt - now).days)
        round_num_num = pd.to_numeric(pd.Series([round_num]), errors="coerce").iloc[0]
        round_series = pd.to_numeric(schedule.get("RoundNumber"), errors="coerce") if "RoundNumber" in schedule.columns else pd.Series(dtype=float)
        total_rounds = int(max(24, round_series.dropna().max())) if not round_series.dropna().empty else 24

        is_live = False
        live_session_name = ""
        for s in ["Session1", "Session2", "Session3", "Session4", "Session5"]:
            date_key = f"{s}Date"
            if date_key in target_event and pd.notna(target_event[date_key]):
                s_date = pd.to_datetime(target_event[date_key], errors="coerce", utc=True)
                if pd.isna(s_date):
                    continue
                if s_date <= now <= (s_date + timedelta(hours=2.5)):
                    is_live = True
                    live_session_name = str(target_event.get(s, s))
                    break

        st.markdown(
            f"""
            <div class="home-hero">
              <div class="home-badges">{_build_hero_badges(active_year, is_live, days_to_next, schedule_source)}</div>
              <div class="home-title">{event_name}</div>
              <div class="home-sub">Round {round_num} | {location} | Race start {_fmt_user_time(next_race_dt, "%d %b %Y %H:%M")}</div>
              <div class="home-count-grid">
                <div class="home-count-card">
                  <div class="home-count-label">Days</div>
                  <div class="home-count-value">{countdown.get("days") if countdown.get("days") is not None else "N/A"}</div>
                  <div class="home-count-note">until lights out</div>
                </div>
                <div class="home-count-card">
                  <div class="home-count-label">Hours</div>
                  <div class="home-count-value">{countdown.get("hours") if countdown.get("hours") is not None else "N/A"}</div>
                  <div class="home-count-note">countdown clock</div>
                </div>
                <div class="home-count-card">
                  <div class="home-count-label">Minutes</div>
                  <div class="home-count-value">{countdown.get("minutes") if countdown.get("minutes") is not None else "N/A"}</div>
                  <div class="home-count-note">updates on rerun</div>
                </div>
                <div class="home-count-card">
                  <div class="home-count-label">Seconds</div>
                  <div class="home-count-value">{countdown.get("seconds") if countdown.get("seconds") is not None else "N/A"}</div>
                  <div class="home-count-note">live countdown detail</div>
                </div>
                <div class="home-count-card">
                  <div class="home-count-label">Round Progress</div>
                  <div class="home-count-value">{int(round_num_num) if pd.notna(round_num_num) else "N/A"}/{total_rounds}</div>
                  <div class="home-count-note">season tracker</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if pd.notna(round_num_num) and total_rounds > 0:
            progress_ratio = min(max((float(round_num_num) - 1.0) / float(total_rounds), 0.0), 1.0)
            st.progress(progress_ratio, text=f"Season progress tracker: round {int(round_num_num)} / {total_rounds}")

        quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
        quick_col1.metric("Calendar Source", schedule_source)
        quick_col2.metric("UTC Display", f"UTC{_get_user_utc_offset()}")
        quick_col3.metric("Upcoming Events", len(next_future_events))
        quick_col4.metric("Countdown", _countdown_text(next_race_dt, now))

        if is_live:
            st.success(f"{event_name} is live now ({live_session_name})")
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"### {event_name}")
                st.markdown(f"**Round {round_num}** | {location}")
                st.markdown("---")
                st.markdown("##### Quick Actions")
                if st.button("Go to Live Timing >", type="primary"):
                    st.session_state.selected_tab = "Live Timing"
                    st.rerun()
            with c2:
                st.info("Session is active now. Open the Live tab for timing, track status, and updates.")
        else:
            days_until_target = (event_date - now).days if pd.notna(event_date) else None
            if days_until_target is not None and days_until_target < 0:
                st.info(f"Recently completed: {event_name}. Open Analysis for recap and charts.")
                if not next_future_events.empty:
                    upcoming_name = str(next_future_events.iloc[0].get("EventName", "Next Race"))
                    upcoming_dt = pd.to_datetime(next_future_events.iloc[0].get("EventDate"), errors="coerce", utc=True)
                    st.markdown(f"### Next Countdown: {upcoming_name}")
                    st.markdown(f"**{_countdown_text(upcoming_dt, now)}** until lights out")
            else:
                st.markdown(f"### Next Up: {event_name}")
                st.markdown(f"**{location}** | {_fmt_user_time(event_date, '%d %b %Y %H:%M')}")
                st.markdown(f"**Countdown:** {_countdown_text(event_date, now)} until lights out")
                if days_until_target is not None and days_until_target <= 7:
                    st.warning(f"Race week countdown is live ({days_until_target} days to go)")

        # Weekend session timeline (available when full schedule feed includes sessions)
        session_rows = []
        for s in ["Session1", "Session2", "Session3", "Session4", "Session5"]:
            date_col = f"{s}Date"
            if s in target_event and date_col in target_event and pd.notna(target_event.get(date_col)):
                s_date = pd.to_datetime(target_event[date_col], errors="coerce", utc=True)
                if pd.isna(s_date):
                    continue
                session_rows.append(
                    {
                        "Session": str(target_event.get(s, s)),
                        "Start": _fmt_user_time(s_date, "%d %b %H:%M"),
                        "Countdown": _countdown_text(s_date, now),
                        "Status": "Live" if s_date <= now <= (s_date + timedelta(hours=2.5)) else ("Upcoming" if s_date >= now else "Completed"),
                    }
                )
        if session_rows:
            st.markdown("#### Weekend Session Timeline")
            status_counts = pd.Series([str(r.get("Status", "Upcoming")) for r in session_rows]).value_counts()
            st.markdown(
                f"""
                <div class="home-inline-badges">
                  <span class="home-status-chip live">Live: {int(status_counts.get('Live', 0))}</span>
                  <span class="home-status-chip upcoming">Upcoming: {int(status_counts.get('Upcoming', 0))}</span>
                  <span class="home-status-chip done">Completed: {int(status_counts.get('Completed', 0))}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            session_df = pd.DataFrame(session_rows)
            st.dataframe(session_df, hide_index=True, use_container_width=True)

        st.divider()
        st.subheader(f"{active_year} Season Schedule")
        display_cols = [c for c in ["RoundNumber", "EventDate", "EventName", "Location"] if c in upcoming_events.columns]
        display_schedule = upcoming_events[display_cols].head(8).copy()
        if "EventDate" in display_schedule.columns:
            display_schedule["Countdown"] = display_schedule["EventDate"].apply(lambda x: _countdown_text(x, now))
            display_schedule["EventDate"] = display_schedule["EventDate"].apply(lambda x: _fmt_user_time(x, "%Y-%m-%d %H:%M"))
        rename_map = {
            "RoundNumber": "Round",
            "EventDate": f"Date (UTC{_get_user_utc_offset()})",
            "EventName": "Event",
            "Location": "Location",
        }
        display_schedule = display_schedule.rename(columns=rename_map)
        st.dataframe(display_schedule, hide_index=True, use_container_width=True)

        radar_events = next_future_events.head(4).copy()
        if not radar_events.empty:
            st.markdown("#### Countdown Radar (Next 4 Races)")
            radar_cards = []
            for _, radar_row in radar_events.iterrows():
                radar_dt = pd.to_datetime(radar_row.get("EventDate"), errors="coerce", utc=True)
                radar_cd = _countdown_parts(radar_dt, now)
                radar_days = radar_cd.get("days")
                if radar_days is None:
                    chip_class = "normal"
                    chip_label = "SCHEDULED"
                elif radar_cd.get("total_seconds", 0) == 0:
                    chip_class = "urgent"
                    chip_label = "TODAY"
                elif radar_days <= 7:
                    chip_class = "urgent"
                    chip_label = "RACE WEEK"
                elif radar_days <= 30:
                    chip_class = "soon"
                    chip_label = "COMING SOON"
                else:
                    chip_class = "normal"
                    chip_label = "UPCOMING"
                radar_cards.append(
                    (
                        f'<div class="home-radar-card">'
                        f'<div class="home-radar-chip {chip_class}">{chip_label}</div>'
                        f'<div class="home-radar-title">{str(radar_row.get("EventName", "Race"))}</div>'
                        f'<div class="home-radar-meta">{str(radar_row.get("Location", "TBA"))} | {_fmt_user_time(radar_dt, "%d %b %Y %H:%M")}</div>'
                        f'<div class="home-radar-count">{_countdown_text(radar_dt, now)} until lights out</div>'
                        f"</div>"
                    )
                )
            st.markdown(f'<div class="home-radar-grid">{"".join(radar_cards)}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Could not load home dashboard: {e}")
