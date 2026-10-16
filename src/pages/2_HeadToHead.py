import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    calculate_form_trend, calculate_points_trajectory, calculate_teammate_comparison,
)
from config import FASTF1_CONFIG, TEAM_COLORS
from h2h import (
    head_to_head, load_season_results, result_label, season_points,
    short_race_name, with_alpha,
)
from season_config import get_completed_races, get_race_names
from shared import empty_state, load_race_data, show_plotly_chart
from styles import (
    GRID, LABEL_FONT, TEXT_PRIMARY, TEXT_SECONDARY, TITLE_FONT,
    fig_layout as _fig_layout,
    hero_data_uri, icon, panel_head, readable_text,
)

# How much of the season the race-by-race table shows.
TABLE_WINDOWS = {"All races": None, "Last 10 races": 10, "Last 5 races": 5}


def page():
    year = st.session_state.get('selected_year', FASTF1_CONFIG.default_year)
    race_df = _race_rows(load_race_data(year))

    _render_page_head(year)

    tabs = st.tabs(["Teammate Battles", "Driver Matchup", "Form & Consistency", "Points Trajectory"])
    with tabs[0]:
        _tab_teammates(race_df)
    with tabs[1]:
        _tab_matchup(year)
    with tabs[2]:
        _tab_form(race_df)
    with tabs[3]:
        _tab_trajectory(race_df)


def _render_page_head(year):
    st.markdown(f"""
    <div class="page-head">
        <span class="eyebrow">Head-to-Head &amp; Season</span>
        <h1 class="page-title">Season <span class="accent">Head-to-Head</span></h1>
        <p class="page-sub">{year} &middot; teammate battles &middot; driver matchup
            &middot; form &amp; consistency &middot; points trajectory</p>
    </div>
    """, unsafe_allow_html=True)


def _race_rows(df):
    """Race sessions only; sprints double-count tracks in season aggregates."""
    if df is None or df.empty:
        return None
    return df[df['SessionType'] == 'Race'].copy() if 'SessionType' in df.columns else df.copy()


def _tab_teammates(race_df):
    if race_df is None or race_df.empty:
        empty_state("No season results yet", "Results appear here once races complete.")
        return

    st.markdown("##### Teammate Head-to-Head")
    comp = calculate_teammate_comparison(race_df)
    if comp is None or comp.empty:
        st.info("No teammate pairs with shared races found.")
        return

    display = comp[
        ['Team', 'Driver 1', 'Driver 2', 'Races Together',
         'Race H2H', 'Quali H2H', 'Pts 1', 'Pts 2']
    ].copy()
    st.dataframe(display, width='stretch', hide_index=True)

    st.markdown("##### Race Wins by Pairing")
    fig = go.Figure()
    for _, row in comp.iterrows():
        color = TEAM_COLORS.get(row['Team'], '#555')
        fig.add_trace(go.Bar(
            x=[int(row['D1 Race Wins'])], y=[row['Team']], orientation='h',
            marker_color=color, marker_line_width=0,
            hovertemplate=f"<b>{row['Driver 1']}</b><br>Wins: {int(row['D1 Race Wins'])}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=[int(row['D2 Race Wins'])], y=[row['Team']], orientation='h',
            marker_color=color, opacity=0.45, marker_line_width=0,
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


# -- Driver matchup ------------------------------------------------------

def _tab_matchup(year):
    """Driver vs driver comparison built from FastF1 season results.

    The season frame comes from the timing API rather than the static CSVs in
    data/, so a weekend that has just been classified shows up without waiting
    for an export.
    """
    races = _load_season_frame(year)
    if races is None or races.empty:
        empty_state(
            f"No {year} session results yet",
            "FastF1 returns results once a race weekend is classified; completed "
            "rounds then load automatically.",
        )
        return

    driver_a, driver_b = _matchup_pickers(races)
    if driver_a == driver_b:
        st.info("Pick two different drivers.")
        return

    matchup = head_to_head(races, None, driver_a, driver_b)
    if matchup is None:
        st.info(f"{driver_a} and {driver_b} share no races in {year}.")
        return

    st.caption(
        f"{races['Round'].nunique()} completed rounds loaded from the FastF1 timing "
        "API; cached for the session."
    )

    _matchup_arena(matchup)
    _matchup_stats(matchup, _season_rounds(year, matchup))
    _matchup_summary(matchup)
    _matchup_table(matchup)


def _load_season_frame(year):
    """Load the season's race results, showing progress while FastF1 warms up.

    FastF1 caches every session on disk, so this is a one-off cost per season
    and a batch of cache hits on every later rerun.
    """
    rounds = get_completed_races(year) or []
    bar = st.progress(
        0.0,
        text=f"Loading {len(rounds)} rounds from FastF1 - the first load of a season takes a minute",
    )

    def _advance(event_name, index, total):
        bar.progress((index + 1) / max(total, 1), text=f"Loading {index + 1}/{total} · {event_name}")

    races, _ = load_season_results(year, progress=_advance)
    bar.empty()
    return races


def _season_rounds(year, matchup):
    """Calendar size for the 'out of N rounds' wording."""
    try:
        return len(get_race_names(year) or {})
    except Exception:
        return matchup['races_shared']


def _matchup_pickers(races):
    """Two driver pickers, defaulting to the top two in the standings."""
    order = season_points(races).index.tolist()
    drivers = sorted(races['Driver'].dropna().unique().tolist())

    left, right = st.columns(2, gap="medium")
    with left:
        driver_a = st.selectbox(
            "Driver A", drivers, index=drivers.index(order[0]) if order else 0, key="h2h_d1",
        )
    with right:
        default_b = order[1] if len(order) > 1 else drivers[-1]
        driver_b = st.selectbox(
            "Driver B", drivers, index=drivers.index(default_b), key="h2h_d2",
        )
    return driver_a, driver_b


def _matchup_arena(matchup):
    """Two driver cards facing off over a team-coloured arena."""
    a, b = matchup['driver1'], matchup['driver2']
    photo = hero_data_uri()
    photo_html = (f'<div class="h2h-arena-photo" style="background-image:url(\'{photo}\')"></div>'
                  if photo else '')
    st.markdown(f"""
    <div class="h2h-arena">
        {photo_html}
        {_arena_side(a, 'left')}
        <div class="h2h-vs">VS</div>
        {_arena_side(b, 'right')}
    </div>
    """, unsafe_allow_html=True)


def _arena_side(side, align):
    """One side of the arena: number watermark, name, team, season line."""
    flag = f'<span class="h2h-flag">{side["flag"]}</span>' if side['flag'] else ''
    avg = _position_label(side['avg_finish'])
    return f"""
    <div class="h2h-side h2h-side--{align}">
        <div class="h2h-numeral">{side['number']}</div>
        <div class="h2h-first">{side['first']}</div>
        <div class="h2h-last" style="border-color:{side['color']}">{side['last']}</div>
        <div class="h2h-meta">{flag}<span>{side['team']}</span></div>
        <div class="h2h-side-stats">{side['races']} races &middot; {side['points']} pts
            &middot; avg {avg}</div>
    </div>"""


def _position_label(value):
    return f"P{value}" if value is not None else "n/a"


def _matchup_stats(matchup, rounds_total):
    """Four comparison cards: shared races, race wins, qualifying, points."""
    a, b = matchup['driver1'], matchup['driver2']
    record, quali = matchup['race_record'], matchup['quali_record']

    cards = [
        _stat_card(
            "Races together", "flag", f"{matchup['races_shared']}",
            sub=f"of {rounds_total} rounds",
            track=_progress_bar(matchup['races_shared'], rounds_total),
        ),
        _stat_card(
            "Race wins", "trophy",
            _score(record['wins1'], record['wins2'], a, b),
            sub=_legend_names(a, b, record['ties']),
            track=_dual_bar(record['wins1'], record['wins2'], a['color'], b['color']),
        ),
        _stat_card(
            "Qualifying", "clock",
            _score(quali['wins1'], quali['wins2'], a, b),
            sub=_legend_names(a, b, quali['ties']),
            track=_dual_bar(quali['wins1'], quali['wins2'], a['color'], b['color']),
        ),
        _stat_card(
            "Points", "chart",
            _score(matchup['points']['1'], matchup['points']['2'], a, b),
            sub=_legend_names(a, b),
            track=_dual_bar(matchup['points']['1'], matchup['points']['2'],
                            a['color'], b['color']),
        ),
    ]
    st.markdown(f'<div class="h2h-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _stat_card(label, icon_name, value, sub="", track=""):
    track_html = f'<div class="h2h-track">{track}</div>' if track else ""
    sub_html = f'<div class="h2h-card-sub">{sub}</div>' if sub else ""
    return f"""<div class="h2h-card">
  <div class="h2h-card-head">
    <span class="h2h-card-icon">{icon(icon_name)}</span>
    <span class="h2h-card-label">{label}</span>
  </div>
  <div class="h2h-card-value">{value}</div>
  {sub_html}{track_html}
</div>"""


def _score(value_a, value_b, side_a, side_b):
    """'11 - 5' with each number painted in its driver's colour."""
    return (f'<span style="color:{side_a["color"]}">{value_a}</span>'
            f'<span class="h2h-dash">&ndash;</span>'
            f'<span style="color:{side_b["color"]}">{value_b}</span>')


def _legend_names(side_a, side_b, ties=0):
    names = f"{side_a['short']} &middot; {side_b['short']}"
    return f"{names} &middot; {ties} equal" if ties else names


def _dual_bar(value_a, value_b, color_a, color_b):
    """Split bar showing each side's share of the compared count."""
    total = value_a + value_b
    if total <= 0:
        return ""
    return (f'<span style="width:{value_a / total * 100:.1f}%;background:{color_a}"></span>'
            f'<span style="width:{value_b / total * 100:.1f}%;background:{color_b}"></span>')


def _progress_bar(value, total):
    if total <= 0:
        return ""
    return f'<span style="width:{min(value / total * 100, 100):.0f}%"></span>'


def _matchup_summary(matchup):
    """Plain-language read of the numbers plus three head-to-head mini stats."""
    a, b = matchup['driver1'], matchup['driver2']
    st.markdown(f"""
    <div class="panel">
        {panel_head("bulb", "Performance Summary",
                    f"{matchup['races_shared']} races together")}
        <p class="h2h-summary">{_summary_sentences(matchup)}</p>
        <div class="h2h-mini">
            {_mini_stat("Avg finish", _position_label(a['avg_finish']),
                        _position_label(b['avg_finish']), a, b)}
            {_mini_stat("Best finish", _position_label(a['best']),
                        _position_label(b['best']), a, b)}
            {_mini_stat("Unclassified", a['dnfs'], b['dnfs'], a, b)}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _mini_stat(label, value_a, value_b, side_a, side_b):
    return f"""<div class="h2h-mini-stat">
  <div class="h2h-mini-label">{label}</div>
  <div class="h2h-mini-value">
    <span style="color:{side_a['color']}">{value_a}</span>
    <span class="h2h-mini-sep">&middot;</span>
    <span style="color:{side_b['color']}">{value_b}</span>
  </div>
</div>"""


def _summary_sentences(matchup):
    """Points lead, race record, qualifying edge and consistency, in prose."""
    a, b = matchup['driver1'], matchup['driver2']
    record, quali = matchup['race_record'], matchup['quali_record']
    points_a, points_b = matchup['points']['1'], matchup['points']['2']
    sides = (a, b)
    race_wins = (record['wins1'], record['wins2'])

    lead = 0 if points_a >= points_b else 1
    if points_a == points_b:
        opening = f"{a['name']} and {b['name']} are level on {points_a} points"
    else:
        opening = (f'<span class="h2h-hl">{sides[lead]["name"]} leads by '
                   f'{abs(points_a - points_b)} points</span> ({points_a}\u2013{points_b})')

    sentences = [f"{opening}, and {_race_record_clause(race_wins, sides)}."]

    if quali['wins1'] == quali['wins2']:
        sentences.append(f"Qualifying is dead even at {quali['wins1']}\u2013{quali['wins2']}.")
    else:
        fastest = 0 if quali['wins1'] > quali['wins2'] else 1
        sentences.append(
            f"{sides[fastest]['short']} owns the qualifying advantage, "
            f"{quali['wins1']}\u2013{quali['wins2']}."
        )

    sentences.append(_consistency_sentence(matchup))
    return " ".join(sentences)


def _race_record_clause(race_wins, sides):
    better = 0 if race_wins[0] >= race_wins[1] else 1
    other = 1 - better
    if race_wins[0] == race_wins[1]:
        return f"they have split the races {race_wins[0]}\u2013{race_wins[1]}"
    return (f"{sides[better]['short']} has won {race_wins[better]} to {race_wins[other]} "
            "of their head-to-head races")


def _consistency_sentence(matchup):
    """Third sentence: say whichever of retirements / average finish is louder."""
    a, b = matchup['driver1'], matchup['driver2']
    if a['dnfs'] != b['dnfs']:
        worse, better = (a, b) if a['dnfs'] > b['dnfs'] else (b, a)
        return (f"{worse['short']} failed to classify in {worse['dnfs']} races, "
                f"{better['short']} in {better['dnfs']}.")
    if a['avg_finish'] is not None and b['avg_finish'] is not None and a['avg_finish'] != b['avg_finish']:
        better, worse = (a, b) if a['avg_finish'] < b['avg_finish'] else (b, a)
        return (f"On average {better['short']} finishes {_position_label(better['avg_finish'])} "
                f"against {_position_label(worse['avg_finish'])} for {worse['short']}, "
                "counting classified races only.")
    return "Across the shared races neither side has a clear edge."


def _matchup_table(matchup):
    """Race-by-race finish, grid and points, with the better of each cell marked."""
    a, b = matchup['driver1'], matchup['driver2']
    rows = matchup['race_table']

    left, right = st.columns([2.5, 1], gap="medium")
    with left:
        st.markdown(_table_legend(a, b))
    with right:
        window_size = TABLE_WINDOWS[st.selectbox(
            "Races shown", list(TABLE_WINDOWS), index=0, key="h2h_window",
        )]

    shown = rows.tail(window_size) if window_size else rows
    st.markdown(f"""
    <div class="panel">
        {panel_head("list", "Race-by-Race Comparison",
                    f"{len(shown)} of {len(rows)} shared races")}
        <div class="h2h-table-wrap">{_table_html(shown, a, b)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Finish, grid and points are FastF1 session results. Qualifying is "
        "compared on grid slots, which carry penalties, so it stands in for the "
        "qualifying battle rather than replacing it."
    )


def _table_legend(a, b):
    return f"""<div class="h2h-legend">
  <span class="h2h-legend-item"><span class="h2h-dot" style="background:{a['color']}"></span>{a['short']} better result</span>
  <span class="h2h-legend-item"><span class="h2h-dot" style="background:{b['color']}"></span>{b['short']} better result</span>
  <span class="h2h-legend-item"><span class="h2h-dot h2h-dot--muted"></span>Equal</span>
  <span class="h2h-legend-item"><span class="h2h-pill h2h-pill--na">DNF</span>No result</span>
</div>"""


def _table_html(rows, a, b):
    headers = "".join([
        _th("#"), _th("Grand Prix"),
        _th(f"{a['last']} finish", a['color']),
        _th(f"{b['last']} finish", b['color']),
        _th(f"{a['last']} grid", a['color']),
        _th(f"{b['last']} grid", b['color']),
        _th(f"{a['last']} pts", a['color']),
        _th(f"{b['last']} pts", b['color']),
    ])
    body = "".join(_table_row(row, a, b) for _, row in rows.iterrows())
    return f'<table class="h2h-table"><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>'


def _th(label, color=""):
    style = (f' style="color:{with_alpha(color, .9)};border-bottom-color:{with_alpha(color, .4)}"'
             if color else "")
    return f"<th{style}>{label}</th>"


def _table_row(row, a, b):
    winner = row['Winner']
    cells = (
        f'<td class="h2h-num">{int(row["Round"])}</td>',
        f'<td class="h2h-gp">{short_race_name(row["Track"])}</td>',
        _finish_cell(row['Pos1'], row['Status1'], winner == 1, a),
        _finish_cell(row['Pos2'], row['Status2'], winner == 2, b),
        _grid_cell(row['Grid1'], winner == 1, a),
        _grid_cell(row['Grid2'], winner == 2, b),
        _points_cell(row['Pts1'], winner == 1, a),
        _points_cell(row['Pts2'], winner == 2, b),
    )
    return f"<tr>{''.join(cells)}</tr>"


def _finish_cell(position, status, won, side):
    """Classified position as a pill; the winner gets the team colour."""
    label = result_label(position, status)
    if pd.isna(position):
        return f'<td><span class="h2h-pill h2h-pill--na">{label}</span></td>'
    if won:
        style = f"background:{side['color']};color:{readable_text(side['color'])}"
        return f'<td><span class="h2h-pill" style="{style}">{label}</span></td>'
    return f'<td><span class="h2h-pill h2h-pill--flat">{label}</span></td>'


def _grid_cell(grid, won, side):
    if pd.isna(grid) or int(grid) < 1:
        return '<td class="h2h-na">&ndash;</td>'
    style = f"color:{side['color']};font-weight:800" if won else ""
    return f'<td class="h2h-cell" style="{style}">P{int(grid)}</td>'


def _points_cell(points, won, side):
    style = f"color:{side['color']};font-weight:800" if points else ""
    return f'<td class="h2h-cell h2h-pts" style="{style}">{int(points)}</td>'


# -- CSV-backed tabs ------------------------------------------------------

def _tab_form(race_df):
    if race_df is None or race_df.empty:
        empty_state("No season results yet", "Results appear here once races complete.")
        return

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
        fig.add_trace(go.Scatter(
            x=form.index, y=form[d], name=d, mode='lines+markers',
            line=dict(color=_driver_color(d, team_map), width=3, shape='spline', smoothing=0.4),
            marker=dict(size=7, color=_driver_color(d, team_map),
                        line=dict(width=1.2, color='white')),
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
    if race_df is None or race_df.empty:
        empty_state("No season results yet", "Results appear here once races complete.")
        return

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
        fig.add_trace(go.Scatter(
            x=traj.index, y=traj[d], name=d, mode='lines+markers',
            line=dict(color=_driver_color(d, team_map), width=3, shape='spline', smoothing=0.4),
            marker=dict(size=8, color=_driver_color(d, team_map),
                        line=dict(width=1.5, color='white')),
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


def _driver_team_map(race_df):
    """Map each driver to their primary team for color lookup."""
    return race_df.groupby('Driver')['Team'].first().to_dict()


def _driver_color(driver, team_map):
    return TEAM_COLORS.get(team_map.get(driver, ''), '#888')


page()
