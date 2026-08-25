# -*- coding: utf-8 -*-
"""
replay.py
~~~~~~~~~
Browser-based race replay (animated position vs lap) for a FastF1 session.

The core transform (`positions_by_lap`) is pure pandas and unit-testable; the
Plotly figure builder wraps it into a playable rank animation, replacing the
never-built Arcade desktop player.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config import TEAM_COLORS, VIZ_CONFIG
except ImportError:
    from .config import TEAM_COLORS, VIZ_CONFIG


def positions_by_lap(
    laps: Optional[pd.DataFrame], drivers: Optional[List[str]] = None
):
    """Build position per (lap, driver), forward-filled across gaps.

    Returns a tuple ``(pivot_df, lap_order)`` where ``pivot_df`` is indexed by
    lap number with one column per driver holding their race position, and
    ``lap_order`` is the sorted list of lap numbers.
    """
    empty = (pd.DataFrame(), [])
    if laps is None or laps.empty:
        return empty
    for col in ('LapNumber', 'Driver', 'Position'):
        if col not in laps.columns:
            return empty

    sub = laps[['Driver', 'LapNumber', 'Position']].copy()
    sub['LapNumber'] = pd.to_numeric(sub['LapNumber'], errors='coerce')
    sub['Position'] = pd.to_numeric(sub['Position'], errors='coerce')
    sub = sub.dropna(subset=['LapNumber'])

    if drivers:
        sub = sub[sub['Driver'].isin(drivers)]

    if sub.empty:
        return empty

    pivot = sub.pivot(index='LapNumber', columns='Driver', values='Position')
    pivot = pivot.sort_index().ffill()
    # Drop drivers who never appear with a valid position
    pivot = pivot.dropna(axis=1, how='all')
    return pivot, pivot.index.tolist()


def _driver_color(driver: str, team_map: Dict[str, str]) -> str:
    return TEAM_COLORS.get(team_map.get(driver, ''), '#888')


def build_position_replay(
    pivot: pd.DataFrame,
    team_map: Dict[str, str],
    title: str = "Race Position Replay",
) -> go.Figure:
    """Plotly animated rank chart: drivers move down the inverted Y axis per lap.

    Each frame is a snapshot of driver positions at one lap. A lap counter sits
    in the top-right corner of every frame.
    """
    if pivot is None or pivot.empty:
        fig = go.Figure()
        fig.update_layout(title=dict(text=title, x=0.02))
        return fig

    drivers = pivot.columns.tolist()
    lap_list = pivot.index.tolist()
    max_pos = max(2, int(pivot.max().max()) + 2)

    frames = []
    for lap in lap_list:
        row = pivot.loc[lap]
        data = []
        for d in drivers:
            if d in row.index and pd.notna(row[d]):
                color = _driver_color(d, team_map)
                data.append(go.Scatter(
                    x=[d], y=[float(row[d])], mode='markers+text',
                    text=[d], textposition='top center',
                    textfont=dict(size=9, color='white'),
                    marker=dict(size=15, color=color,
                                line=dict(width=1.2, color='white')),
                    name=d,
                    hovertemplate=f"<b>{d}</b> &mdash; P{int(row[d])}<extra></extra>",
                ))
        # lap counter trace
        data.append(go.Scatter(
            x=[lap_list[0]], y=[max_pos - 1], mode='text',
            text=[f"Lap {int(lap)}"], textposition='middle left',
            textfont=dict(size=18, color='#E10600'),
            hoverinfo='skip', showlegend=False,
        ))
        frames.append(go.Frame(data=data, name=str(int(lap))))

    first = frames[0].data if frames else []
    fig = go.Figure(data=first, frames=frames)

    steps = [
        dict(method='animate', label=str(int(lap)), args=[
            [str(int(lap))],
            dict(mode='immediate',
                 frame=dict(duration=int(VIZ_CONFIG['replay_slider_ms']), redraw=True)),
        ])
        for lap in lap_list
    ]
    sliders = [dict(
        currentvalue=dict(prefix='Lap '),
        pad=dict(t=10),
        steps=steps,
    )]
    play_ms = int(VIZ_CONFIG['replay_play_ms'])
    play = dict(label='Play', method='animate', args=[None, dict(
        frame=dict(duration=play_ms, redraw=True), fromcurrent=True)])
    pause = dict(label='Pause', method='animate', args=[[None], dict(
        frame=dict(duration=play_ms, redraw=True), mode='immediate')])

    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=16, color='white')),
        yaxis=dict(autorange='reversed', range=[1, max_pos],
                   title=dict(text='Position', font=dict(size=12, color='#888')),
                   tickfont=dict(size=11, color='#888')),
        xaxis=dict(title=dict(text='Driver', font=dict(size=12, color='#888')),
                   tickfont=dict(size=10, color='#ccc')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=30, t=60, b=80),
        updatemenus=[dict(type='buttons', showactive=False, x=0.02, y=1.12,
                          buttons=[play, pause])],
        sliders=sliders,
        legend=dict(font=dict(size=10, color='#aaa')),
    )
    return fig