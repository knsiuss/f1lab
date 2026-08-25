import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go

from shared import empty_state, load_race_data, load_fastf1_session
from config import DATA_DIR, TEAM_COLORS, FASTF1_CONFIG
from loader import load_race_grid
from season_config import get_race_names
from report import build_report
from prediction import predict_race
from briefing import pre_race_briefing, post_race_debrief


def _session_extras(year, race):
    """Best-effort extra stats + report sections from a FastF1 session."""
    extras = {}
    sections = []
    try:
        session = load_fastf1_session(year, race, 'Race')
        if session is None:
            return extras, sections

        try:
            from fastf1_extended import get_pit_stops
            pits = get_pit_stops(session)
            extras['pit_stops'] = int(len(pits)) if pits is not None and not pits.empty else 0
        except Exception:
            pass

        try:
            from fastf1_extended import get_tyre_stints
            stints = get_tyre_stints(session)
            if stints is not None and not stints.empty:
                lines = ["### Tyre Strategy", "", "Driver | Stint | Compound | Laps |",
                         "---|---|---|---:"]
                for _, r in stints.head(16).iterrows():
                    lines.append(
                        f"{r.get('Driver','')} | {r.get('Stint','')} | {r.get('Compound','')} | {r.get('Laps','')}"
                    )
                sections.append("\n".join(lines))
        except Exception:
            pass
    except Exception:
        pass
    return extras, sections


def _standings_chart_html(df):
    """Dark standings bar chart embedded as self-contained HTML."""
    g = df.groupby('Driver')['Points'].sum().sort_values(ascending=False).head(10)
    fig = go.Figure(go.Bar(
        x=g.values, y=g.index, orientation='h',
        marker_color=[TEAM_COLORS.get(df[df['Driver'] == d]['Team'].iloc[0], '#555')
                      if 'Team' in df.columns else '#E10600' for d in g.index],
        marker_line=dict(width=1, color='rgba(255,255,255,0.3)'),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#eee', size=12),
        height=420, yaxis=dict(autorange='reversed'),
        margin=dict(l=90, r=30, t=30, b=30),
        title=dict(text='Season Standings', x=0.02),
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def _html_report(report_md, chart_html):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>F1 Lab — Weekend Report</title>
<style>
  body {{ background:#0e0e1a; color:#e8e8f0; font-family:Inter, Segoe UI, sans-serif;
         margin:0; padding:2.5rem max(1rem, calc(50vw - 30rem)); line-height:1.6; }}
  h1 {{ color:#E10600; }}
  pre {{ white-space:pre-wrap; font-family:ui-monospace, Consolas, monospace;
         background:#151522; border:1px solid #2a2a3a; border-radius:10px; padding:1.2rem; }}
</style>
</head>
<body>
{chart_html}
<pre>{report_md}</pre>
</body>
</html>"""


def page():
    year = st.session_state.get('selected_year', FASTF1_CONFIG.default_year)
    df = load_race_data(year)
    if df is None or df.empty:
        empty_state(f"No {year} report yet", "Results appear here once races complete. For live session data, open Analysis Center.")
        return

    race_df = df[df['SessionType'] == 'Race'].copy() if 'SessionType' in df.columns else df.copy()

    st.markdown(f"""
    <div style="text-align:center;padding:1.2rem 0 0.8rem 0;">
        <h1 style="font-size:2.2rem;font-weight:800;color:#E10600;margin:0;
                   text-transform:uppercase;letter-spacing:3px;">
            Weekend Report
        </h1>
        <p style="font-size:0.9rem;color:#666;margin-top:0.3rem;letter-spacing:0.5px;">
            {year} Season &mdash; Shareable Race &middot; Standings &middot; Insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    race_names = get_race_names(year)
    all_races = list(race_names.keys()) if race_names else []
    if not all_races:
        st.warning("No races available.")
        return
    selected_race = st.selectbox("Grand Prix", all_races, key="report_race",
                                 label_visibility="collapsed")

    with st.spinner("Building report..."):
        extras, sections = _session_extras(year, selected_race)
        report_md = build_report(
            year, selected_race, race_df,
            extra=extras, extra_sections=sections,
        )

    st.markdown(report_md)

    grid = load_race_grid(str(DATA_DIR), year, selected_race, race_df)
    forecast = predict_race(race_df, grid) if grid else None

    with st.expander("Pre-Race Briefing"):
        if grid:
            st.markdown(pre_race_briefing(race_df, grid, race=selected_race))
        else:
            st.info("No grid available to build a pre-race briefing.")

    with st.expander("Post-Race Debrief"):
        st.markdown(post_race_debrief(race_df, selected_race, forecast=forecast))

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download Markdown", data=report_md,
            file_name=f"F1_{year}_{selected_race.replace(' ', '_')}_report.md",
            mime="text/markdown",
        )
    with c2:
        chart_html = _standings_chart_html(race_df)
        html = _html_report(report_md, chart_html)
        st.download_button(
            "Download HTML (.html)", data=html,
            file_name=f"F1_{year}_{selected_race.replace(' ', '_')}_report.html",
            mime="text/html",
        )


page()