# -*- coding: utf-8 -*-
"""
styles.py
Shared Plotly design-system tokens, icon set and layout helpers.

The pages used to define their own copies of these constants and helpers.
They now live here so every page reads as one system and a theme tweak lands
in a single place.
"""

import base64
from pathlib import Path

ASSET_DIR = Path(__file__).parent / "assets"
HERO_IMAGE = ASSET_DIR / "hero-track.jpg"

# Stroke icons (Feather-style) so panels across pages read as one system.
ICONS = {
    "flag": '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/>',
    "trophy": '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>',
    "award": '<circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/>',
    "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "trending": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "chart": '<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>',
    "list": '<line x1="8" x2="21" y1="6" y2="6"/><line x1="8" x2="21" y1="12" y2="12"/><line x1="8" x2="21" y1="18" y2="18"/><line x1="3" x2="3.01" y1="6" y2="6"/><line x1="3" x2="3.01" y1="12" y2="12"/><line x1="3" x2="3.01" y1="18" y2="18"/>',
    "bulb": '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "chevron": '<polyline points="9 18 15 12 9 6"/>',
}


def icon(name: str, size: int = 17) -> str:
    """Inline SVG for a system icon, stroked in the current text colour."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round">{ICONS[name]}</svg>'
    )


def panel_head(name: str, title: str, sub: str, action: str = "") -> str:
    """Panel header block: icon + title + sub, with optional right-side action."""
    action_html = f'<div class="panel-action">{action}</div>' if action else ""
    return f"""<div class="panel-head">
  <div class="panel-icon">{icon(name)}</div>
  <div><div class="panel-title">{title}</div><div class="panel-sub">{sub}</div></div>
  {action_html}
</div>"""


def hero_data_uri() -> str:
    """Hero photo inlined as base64: one request fewer and no path guessing,
    since the CSS is injected into the page rather than served from /assets.
    """
    if not HERO_IMAGE.exists():
        return ""
    encoded = base64.b64encode(HERO_IMAGE.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def readable_text(color: str) -> str:
    """'#0B0B10' or '#FFFFFF', whichever has more contrast on ``color``."""
    value = color.lstrip("#")
    if len(value) != 6:
        return "#FFFFFF"
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#0B0B10" if luminance > 0.55 else "#FFFFFF"

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

# Podium medal colours by finishing position.
PODIUM_COLORS = {
    1: '#FFD700',   # Gold
    2: '#C0C0C0',   # Silver
    3: '#CD7F32',   # Bronze
}


def podium_color(position) -> str:
    """Colour for a podium position; 'transparent' for anything else."""
    try:
        return PODIUM_COLORS[int(position)]
    except (TypeError, ValueError, KeyError):
        return 'transparent'


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


# Fixed seconds between ticks on mm:ss formatted time axes.
TIME_TICK_STEP_S = 5


def time_axis_layout(label, total_range, step=TIME_TICK_STEP_S):
    """Return yaxis kwargs for a mm:ss formatted time axis.

    total_range: (min_seconds, max_seconds)
    step: seconds between ticks
    """
    mn, mx = total_range
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
