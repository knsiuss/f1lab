# -*- coding: utf-8 -*-
"""
styles.py
Shared Plotly design-system tokens and chart helpers.

The dashboard pages each used to define their own copy of these constants
and layout helpers. They now live here so every chart reads as one system
and theme tweaks land in a single place.
"""

# Design tokens
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


def fig_layout(height=400, **overrides):
    """Base Plotly layout -- consistent across every chart."""
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


def time_axis_layout(label, total_range):
    """Return yaxis kwargs for a mm:ss formatted time axis.

    total_range: (min_seconds, max_seconds)
    """
    mn, mx = total_range
    step = 5
    start = int(mn) // step * step
    stop = int(mx) // step * step + step + 1
    tickvals = list(range(start, stop, step))
    ticktext = [f"{v // 60}:{v % 60:05.2f}" for v in tickvals]
    return dict(
        title=dict(text=label, font=LABEL_FONT),
        tickvals=tickvals,
        ticktext=ticktext,
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zerolinecolor=GRID_EMPH,
        tickfont=dict(size=11, color=TEXT_SECONDARY),
    )
