# -*- coding: utf-8 -*-
"""
styles.py
~~~~~~~~~
Shared Plotly design-system tokens and chart helpers.

The colour values mirror the CSS custom properties in ``assets/style.css``;
keep the two in step so charts and page chrome look like one system.

Charts carry no title of their own: the section header above them (from
``ui.section``) names them. A chart title repeated in a bold plotly font was
one of the things that made the pages feel cluttered.
"""

# Design tokens (mirror assets/style.css)
# Solid surfaces, not transparency: charts sit on the card surface so the
# quiet page background never shows *through* a plot.
BG_PAPER = "#11151C"
BG_PLOT = "#171D26"
GRID = "rgba(255,255,255,0.06)"
GRID_EMPH = "rgba(255,255,255,0.13)"
ZERO_LINE = "rgba(255,255,255,0.18)"
TEXT_PRIMARY = "#F4F5F7"
TEXT_SECONDARY = "#929BAA"
TEXT_DIM = "#626D7D"
ACCENT = "#E10600"
FONT = "Inter, system-ui, sans-serif"
TITLE_FONT = dict(family=FONT, size=13, color=TEXT_PRIMARY)
LABEL_FONT = dict(family=FONT, size=11, color=TEXT_SECONDARY)

# Categorical palette for multi-series charts (colour-blind aware ordering).
PLOT_PALETTE = [
    "#4388E8", "#00D2BE", "#FF8700", "#E10600", "#B47CFF",
    "#25C2D9", "#FF7A9A", "#9AA1AE", "#5B8DEF", "#7BD389",
]

COMPOUND_COLORS = {
    'SOFT': '#FF3333', 'MEDIUM': '#FFD700', 'HARD': '#BBBBBB',
    'INTERMEDIATE': '#43B02A', 'WET': '#0067AD',
}

# Colours for an A-versus-B comparison (the head-to-head page). When the two
# drivers share a team their livery colours are identical, so the pair falls
# back to these two, which are far enough apart to tell at a glance.
COMPARE_COLOURS = ["#00D6B5", "#4285F4"]

# Driver colours for the Form tab. These override the team livery colour so
# teammates stay tellable apart and the colour means the same driver in every
# chart on the page. Drivers not listed here use their team colour.
DRIVER_COLORS = {
    'kimi antonelli': '#00D2BE',
    'andrea kimi antonelli': '#00D2BE',
    'antonelli': '#00D2BE',
    'max verstappen': '#4388E8',
    'verstappen': '#4388E8',
    'lewis hamilton': '#E10600',
    'hamilton': '#E10600',
    'george russell': '#B47CFF',
    'russell': '#B47CFF',
    'charles leclerc': '#FF8700',
    'leclerc': '#FF8700',
}

# Heatmap buckets for the Form tab (finishing-position matrix). The order
# matches ``analysis.HEATMAP_BUCKETS``; DNF/DNS/DSQ/no-entry are distinct
# treatments, never a position colour.
HEATMAP_COLORS = [
    ('p1', '#0E9F8A'),
    ('p2p3', '#3ECF8E'),
    ('p4p6', '#7BD389'),
    ('p7p10', '#F2D16B'),
    ('p11p15', '#FF8700'),
    ('p16p20', '#E10600'),
    ('dnf', '#8A3B3B'),
    ('dns', '#3A4150'),
    ('dsq', '#8B5CF6'),
    ('none', '#1A202B'),
]


def compare_colours(colour_a, colour_b):
    """The two colours of a head-to-head chart.

    Args:
        colour_a: the livery colour of the first driver (may be empty).
        colour_b: the livery colour of the second driver (may be empty).

    Returns:
        A ``(colour_a, colour_b)`` tuple. When both drivers are on the same
        team the livery colours cannot tell them apart, so the fixed
        comparison pair is used instead.
    """
    first = str(colour_a or '').strip()
    second = str(colour_b or '').strip()
    if first and second and first.lower() != second.lower():
        return first, second
    return COMPARE_COLOURS[0], COMPARE_COLOURS[1]

# Chart height scale. Every chart in the app picks its height from this scale,
# so the same kind of chart has the same height on every page and two charts
# that stand side by side line up.
CHART_HEIGHT = {
    'tile': 260,      # a small chart next to a KPI row
    'row': 320,       # the standard chart of a two-column row
    'block': 380,     # the default chart
    'feature': 440,   # a full-width feature chart
    'tall': 560,      # dense lap-level charts
    # Base heights for the charts that grow with the number of rows they show
    # (one bar or one line per driver). The rows are added on top of these.
    'bars': 70,
    'series': 60,
    'matrix': 80,
}

# Height of one row in a chart that grows with the number of drivers.
CHART_ROW_H = 26

# Height of the replay player (an iframe, not a chart) -- it belongs to the
# same layout system, so it lives here next to the chart sizes.
PLAYER_HEIGHT = 1180

# Height of the rain-radar panel (also an iframe): a map plus its timeline.
RADAR_HEIGHT = 620


def chart_height(kind: str = 'block') -> int:
    """Height of a chart, from the shared scale.

    Args:
        kind: a key of :data:`CHART_HEIGHT` (``tile``, ``row``, ``block``,
            ``feature``, ``tall``).

    Raises:
        KeyError: for an unknown kind -- a typo must not silently produce a
            different chart size.
    """
    return CHART_HEIGHT[kind]


def scaled_height(kind: str, rows: int, row_height: int = CHART_ROW_H) -> int:
    """Height of a chart that grows with the number of rows it shows.

    The result is anchored on the scale (so the chart still looks like the
    others) and grows in whole rows.
    """
    return chart_height(kind) + max(0, int(rows)) * int(row_height)
# Podium medal colours by finishing position.
PODIUM_COLORS = {
    1: '#FFD166',   # Gold
    2: '#C0C0C0',   # Silver
    3: '#CD7F32',   # Bronze
}


def podium_color(position) -> str:
    """Colour for a podium position; 'transparent' for anything else."""
    try:
        return PODIUM_COLORS[int(position)]
    except (TypeError, ValueError, KeyError):
        return 'transparent'


def fig_layout(height=None, **overrides):
    """Base Plotly layout -- consistent across every chart.

    The defaults are deliberately solid: a dark plot card, thin grid
    lines, a top-right horizontal legend and margins that leave room for
    axis labels without wasting space.

    Args:
        height: the chart height in pixels. ``None`` uses the shared default
            from the height scale; pass ``chart_height('row')`` and friends
            for the other sizes.
        overrides: layout keys to replace.
    """
    base = dict(
        paper_bgcolor=BG_PAPER, plot_bgcolor=BG_PLOT,
        font=dict(family=FONT, color=TEXT_PRIMARY, size=12),
        margin=dict(l=56, r=22, t=26, b=42),
        height=chart_height() if height is None else int(height),
        hoverlabel=dict(bgcolor="rgba(18,21,28,0.95)",
                        font=dict(family=FONT, size=12, color="white"),
                        bordercolor="rgba(255,255,255,0.14)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=TEXT_SECONDARY),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1,
                   zerolinecolor=GRID_EMPH, tickfont=dict(size=11, color=TEXT_SECONDARY),
                   title=dict(font=LABEL_FONT), automargin=True),
        yaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1,
                   zerolinecolor=GRID_EMPH, tickfont=dict(size=11, color=TEXT_SECONDARY),
                   title=dict(font=LABEL_FONT), automargin=True),
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
