# -*- coding: utf-8 -*-
"""
ui.py
~~~~~
Shared user-interface components.

Every page used to hand-write its own inline HTML for the page header, the
KPI tiles, the result rows and the notes. The copies drifted apart -- different
padding, different colours, a different title treatment per page -- which is
what made the app look cluttered. These components are the single source of
that layout: pages compose them instead of repeating markup.

Three rules the components enforce:

* **escaping** -- every dynamic value (driver names, team names, messages from
  the API) is HTML-escaped before it is injected, so a name from the data
  source can never become markup;
* **one visual language** -- spacing, radius, type scale and colour come from
  the classes in ``assets/style.css``;
* **compact markup** -- the HTML is emitted on a single line. Streamlit parses
  a markdown block as HTML only until the first blank line, and a line that
  holds nothing but spaces counts as a blank one: multi-line markup with an
  empty line in the middle silently turns into a rendered code block.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import base64
import html
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import streamlit as st

from config import FLAG_CODES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def esc(value: Any) -> str:
    """HTML-escape a value for safe interpolation into markup."""
    if value is None:
        return ''
    return html.escape(str(value), quote=True)


# ---------------------------------------------------------------------------
# Images (driver portraits + country flags, served from assets/)
# ---------------------------------------------------------------------------
#
# Portraits live in ``assets/drivers/`` and flags in ``assets/flags/`` (see
# ATTRIBUTION.md for sources). They are embedded as data URIs so no page
# hotlinks anything at render time; unknown drivers simply fall back to
# initials, never to a broken image.

_ASSET_DIR = Path(__file__).parent / 'assets'


def _data_uri(path) -> str:
    """A local image file as a base64 data URI (empty when missing)."""
    try:
        if path and Path(path).exists():
            mime = {'.png': 'image/png', '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg', '.svg': 'image/svg+xml'}.get(
                        Path(path).suffix.lower(), '')
            if mime:
                encoded = base64.b64encode(Path(path).read_bytes()).decode()
                return f'data:{mime};base64,{encoded}'
    except OSError:
        pass
    return ''


def driver_portrait(name: str) -> str:
    """Portrait data URI for a driver name ('' when no portrait is stored).

    Files are named by the driver's full name in lower case; a match on the
    last name alone is the fallback (the feed calls a driver by one name
    form while the local files use another, e.g. Kimi Antonelli).
    """
    wanted = str(name or '').strip().lower()
    if not wanted:
        return ''
    folder = _ASSET_DIR / 'drivers'
    exact = folder / f'{wanted.replace(" ", "_")}.png'
    uri = _data_uri(exact) or _data_uri(
        folder / f'{wanted.replace(" ", "_")}.jpg')
    if uri:
        return uri
    last = wanted.split()[-1]
    try:
        for candidate in sorted(folder.glob('*')):
            if candidate.is_file() and candidate.stem.endswith(last):
                uri = _data_uri(candidate)
                if uri:
                    return uri
    except OSError:
        pass
    return ''


def portrait_html(name: str, colour: str = '') -> str:
    """Avatar markup: the portrait, or initials when none is stored."""
    uri = driver_portrait(name)
    if uri:
        return (f'<span class="f1-portrait">'
                f'<img src="{esc(uri)}" alt="{esc(name)}" loading="lazy">'
                f'</span>')
    initials = ''.join(word[0] for word in str(name or '').split()[:2]).upper()
    border = f';border-color:{esc(colour)}' if colour else ''
    return (f'<span class="f1-portrait f1-portrait--initials"'
            f' style="{border}">{esc(initials)}</span>')


def flag_html(country: str) -> str:
    """A small flag image for a calendar country ('' when not stored)."""
    code = FLAG_CODES.get(str(country or '').strip(), '')
    if not code:
        return ''
    uri = _data_uri(_ASSET_DIR / 'flags' / f'{code}.png')
    if not uri:
        return ''
    return (f'<span class="f1-flag"><img src="{esc(uri)}" '
            f'alt="{esc(country)} flag" loading="lazy"></span>')


def _join_chips(chips: Iterable[Any], tone: str = '') -> str:
    """Render a list of chips.

    Each chip is either a ``(label, value)`` pair or a plain string.
    """
    parts: List[str] = []
    for chip in chips or []:
        if isinstance(chip, (tuple, list)) and len(chip) == 2:
            label, value = chip
            body = f'<strong>{esc(value)}</strong> {esc(label)}' if label \
                else f'<strong>{esc(value)}</strong>'
        else:
            body = esc(chip)
        css = 'f1-chip' + (f' f1-chip--{tone}' if tone else '')
        parts.append(f'<span class="{css}">{body}</span>')
    return ''.join(parts)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def page_header(title: str, subtitle: str = '',
                chips: Optional[Sequence[Any]] = None) -> None:
    """The page title block: title, optional subtitle, optional status chips."""
    chips_html = ''
    if chips:
        chips_html = f'<div class="f1-chip-row">{_join_chips(chips)}</div>'
    subtitle_html = f'<div class="f1-page-subtitle">{esc(subtitle)}</div>' \
        if subtitle else ''
    st.markdown(
        f'<div class="f1-page-header"><div>'
        f'<h1 class="f1-page-title">{esc(title)}</h1>{subtitle_html}'
        f'</div>{chips_html}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, note_text: str = '', hint: str = '') -> None:
    """A section heading with an accent rule and an optional right-hand note.

    Args:
        title: what the section is about.
        note_text: a short qualifier on the right of the heading.
        hint: a plain-language line under the heading -- for the reader who
            does not know the words of the sport ("The average of the last
            three results"). It answers "what am I looking at?" before the
            chart asks the reader to read it.
    """
    note_html = f'<span class="f1-section-note">{esc(note_text)}</span>' \
        if note_text else ''
    hint_html = f'<div class="f1-section-hint">{esc(hint)}</div>' if hint else ''
    st.markdown(
        f'<div class="f1-section">'
        f'<div class="f1-section-row">'
        f'<h2 class="f1-section-title">{esc(title)}</h2>{note_html}</div>'
        f'{hint_html}</div>',
        unsafe_allow_html=True,
    )


def card(content: str, flat: bool = False) -> None:
    """Wrap arbitrary (already escaped) markup in a surface card."""
    css = 'f1-card' + (' f1-card--flat' if flat else '')
    st.markdown(f'<div class="{css}">{content}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# KPI tiles
# ---------------------------------------------------------------------------

# Restrained KPI glyphs (Lucide-style line icons, inline so no icon
# dependency is needed). Keys are short names pages pass as ``icon``.
_KPI_ICONS = {
    'flag': ('M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z',
             'M4 22v-7'),
    'layers': ('m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z',
               'm22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65',
               'm22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65'),
    'crown': ('M11.56 3.27a.5.5 0 0 1 .88 0l2.95 5.38a1 1 0 0 0 1.52.28l3.5-3.6a.5.5 0 0 1 .84.35l1.23 8.58a1 1 0 0 1-.99 1.14H2.51a1 1 0 0 1-.99-1.14l1.23-8.58a.5.5 0 0 1 .84-.35l3.5 3.6a1 1 0 0 0 1.52-.28Z',
              'M5 21h14'),
    'trophy': ('M6 9H4.5a2.5 2.5 0 0 1 0-5H6',
               'M18 9h1.5a2.5 2.5 0 0 0 0-5H18',
               'M4 22h16',
               'M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22',
               'M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22',
               'M18 2H6v7a6 6 0 0 0 12 0V2Z'),
    'activity': ('M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2',),
    'trend-down': ('M16 17h6v-6', 'm22 17-8.5-8.5-5 5L2 7'),
    'trend-up': ('M16 7h6v6', 'm22 7-8.5 8.5-5-5L2 17'),
    'target': ('M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Z',
               'M12 18a6 6 0 1 0 0-12 6 6 0 0 0 0 12Z',
               'M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z'),
    'chart': ('M3 3v16a2 2 0 0 0 2 2h16', 'M18 17V9', 'M13 17V5', 'M8 17v-3'),
}


def _kpi_icon_html(name: str) -> str:
    """One small line-icon glyph for a KPI tile (empty when unknown)."""
    paths = _KPI_ICONS.get(str(name or '').strip())
    if not paths:
        return ''
    body = ''.join(f'<path d="{path}"/>' for path in paths)
    return ('<span class="f1-kpi-icon" aria-hidden="true">'
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'{body}</svg></span>')


def kpis(items: Sequence[Dict[str, Any]], columns: Optional[int] = None) -> None:
    """A row of KPI tiles.

    Each item is a dict with ``label``, ``value`` and optional ``sub``,
    ``tone`` (``accent``, ``good``, ``warn``, ``info``) and ``icon``
    (``flag``, ``layers``, ``crown``, ``trophy``).
    """
    if not items:
        return
    tiles: List[str] = []
    for item in items:
        tone = str(item.get('tone') or '').strip()
        css = 'f1-kpi' + (f' f1-kpi--{tone}' if tone else '')
        sub = item.get('sub')
        sub_html = f'<div class="f1-kpi-sub">{esc(sub)}</div>' if sub else ''
        tiles.append(
            f'<div class="{css}">'
            f'{_kpi_icon_html(str(item.get("icon") or ""))}'
            f'<div class="f1-kpi-label">{esc(item.get("label"))}</div>'
            f'<div class="f1-kpi-value">{esc(item.get("value"))}</div>'
            f'{sub_html}</div>'
        )
    st.markdown(f'<div class="f1-kpi-grid">{"".join(tiles)}</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Result rows
# ---------------------------------------------------------------------------

def result_rows(rows: Sequence[Dict[str, Any]],
                empty_text: str = 'Nothing to show yet.') -> None:
    """Classification-style rows (position, driver, team, value).

    Each row is a dict with ``position``, ``name`` and optional ``team``,
    ``value``, ``sub``, ``colour`` (a hex team colour) and ``highlight``.
    """
    if not rows:
        st.info(empty_text)
        return
    markup: List[str] = []
    for row in rows:
        try:
            position = int(row.get('position'))
        except (TypeError, ValueError):
            position = 0
        pos_css = 'f1-row-pos'
        if position in (1, 2, 3):
            pos_css += f' f1-row-pos--p{position}'
        row_css = 'f1-row' + (' f1-row--podium' if position in (1, 2, 3) else '')
        colour = row.get('colour')
        swatch = (f'<span class="f1-swatch" style="background:{esc(colour)}"></span>'
                  if colour else '')
        team = row.get('team')
        value = row.get('value')
        sub = row.get('sub')
        markup.append(
            f'<div class="{row_css}">{swatch}'
            f'<span class="{pos_css}">{esc(row.get("position"))}</span>'
            f'<span class="f1-row-name">{esc(row.get("name"))}</span>'
            + (f'<span class="f1-row-team">{esc(team)}</span>' if team else '')
            + (f'<span class="f1-row-value">{esc(value)}</span>'
               if value is not None else '')
            + (f'<span class="f1-row-team">{esc(sub)}</span>' if sub else '')
            + '</div>'
        )
    st.markdown(f'<div class="f1-rows">{"".join(markup)}</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

def note(text: str, kind: str = 'info') -> None:
    """A note panel: ``info`` (default), ``warn`` or ``estimate``."""
    css = 'f1-note'
    if kind in ('warn', 'warning'):
        css += ' f1-note--warn'
    elif kind in ('estimate', 'estimated'):
        css += ' f1-note--estimate'
    st.markdown(f'<div class="{css}">{esc(text)}</div>', unsafe_allow_html=True)


def estimate_note(text: str) -> None:
    """The standard disclosure for a derived (estimated) figure."""
    note(text, kind='estimate')


def chips(items: Sequence[Any], tone: str = '') -> None:
    """A row of status chips."""
    st.markdown(f'<div class="f1-chip-row">{_join_chips(items, tone)}</div>',
                unsafe_allow_html=True)


def legend(items: Sequence[Tuple[str, str]]) -> None:
    """A colour legend: a list of ``(hex_colour, label)`` pairs."""
    parts: List[str] = []
    for colour, label in items or []:
        parts.append(
            f'<span class="f1-chip">'
            f'<span class="f1-swatch" style="background:{esc(colour)};'
            f'width:8px;height:8px;border-radius:2px;"></span>'
            f'{esc(label)}</span>'
        )
    if parts:
        st.markdown(f'<div class="f1-chip-row">{"".join(parts)}</div>',
                    unsafe_allow_html=True)


def caption(text: str) -> None:
    """A quiet caption under a chart or table."""
    st.markdown(f'<div class="f1-section-note">{esc(text)}</div>',
                unsafe_allow_html=True)


def empty(title: str, hint: str = '') -> None:
    """The calm empty state used when a season or session has no data."""
    hint_html = (f'<div class="f1-section-note f1-empty-hint">{esc(hint)}</div>'
                 if hint else '')
    st.markdown(
        f'<div class="f1-card f1-empty">'
        f'<div class="f1-empty-title">{esc(title)}</div>{hint_html}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Dashboard band
# ---------------------------------------------------------------------------

def hero(eyebrow: str, title: str, headline: str = '', subline: str = '',
         stats: Optional[Sequence[Dict[str, Any]]] = None) -> None:
    """The band at the top of a dashboard.

    A dashboard has to answer one question before anything else: *what is the
    story right now?* The hero states it in one sentence -- in plain words --
    and carries the handful of numbers the reader needs to go on.

    Args:
        eyebrow: the context line ("2026 season - round 17 of 23").
        title: what this screen is.
        headline: the story, in one sentence.
        subline: a second line of context.
        stats: optional inline stats -- dicts with ``label``, ``value`` and
            an optional ``sub``.
    """
    stat_items: List[str] = []
    for item in stats or []:
        value = esc(item.get('value'))
        label = esc(item.get('label'))
        sub = item.get('sub')
        sub_html = f'<span class="f1-hero-stat-sub">{esc(sub)}</span>' if sub else ''
        stat_items.append(
            f'<div class="f1-hero-stat"><span class="f1-hero-stat-value">{value}</span>'
            f'<span class="f1-hero-stat-label">{label}</span>{sub_html}</div>'
        )
    stats_html = (f'<div class="f1-hero-stats">{"".join(stat_items)}</div>'
                  if stat_items else '')
    headline_html = (f'<div class="f1-hero-headline">{esc(headline)}</div>'
                     if headline else '')
    subline_html = f'<div class="f1-hero-subline">{esc(subline)}</div>' if subline else ''
    st.markdown(
        f'<div class="f1-hero"><div class="f1-hero-eyebrow">{esc(eyebrow)}</div>'
        f'<div class="f1-hero-title">{esc(title)}</div>{headline_html}{subline_html}'
        f'{stats_html}</div>',
        unsafe_allow_html=True,
    )


def verdict(text: str, label: str = 'What this means') -> None:
    """The takeaway line under a chart or a table.

    The charts say what happened; this says what it adds up to, in a sentence
    a newcomer can follow.
    """
    st.markdown(
        f'<div class="f1-verdict"><div class="f1-verdict-label">{esc(label)}</div>'
        f'<div class="f1-verdict-text">{esc(text)}</div></div>',
        unsafe_allow_html=True,
    )


def steps(items: Sequence[str], title: str = '') -> None:
    """A numbered "how to read this" list (the first-time-user helper)."""
    if not items:
        return
    title_html = f'<div class="f1-steps-title">{esc(title)}</div>' if title else ''
    rows = ''.join(
        f'<li><span class="f1-step-num">{index}</span>'
        f'<span class="f1-step-text">{esc(text)}</span></li>'
        for index, text in enumerate(items, start=1)
    )
    st.markdown(
        f'<div class="f1-steps-card">{title_html}<ol class="f1-steps">{rows}</ol></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tables and scoreboards
# ---------------------------------------------------------------------------

def _cell_html(cell: Any) -> str:
    """One table cell.

    A cell is a plain value, or a dict with a ``tone`` (how the value is
    coloured) and a ``colour`` (a hex colour, drawn as a swatch in front of the
    value -- the way a team livery is shown in the rest of the app).
    """
    if isinstance(cell, dict):
        value = cell.get('value', '')
        tone = str(cell.get('tone') or '').strip()
        colour = str(cell.get('colour') or '').strip()
        bar_pct = cell.get('bar_pct')
        bar_colour = str(cell.get('bar_colour') or '').strip()
    else:
        value = cell
        tone = ''
        colour = ''
        bar_pct = None
        bar_colour = ''
    css = 'f1-table-cell' + (f' f1-table-cell--{tone}' if tone else '')
    swatch = (f'<span class="f1-swatch" style="background:{esc(colour)}"></span>'
              if colour else '')
    bar = ''
    if bar_pct is not None:
        try:
            pct = max(0.0, min(100.0, float(bar_pct)))
        except (TypeError, ValueError):
            pct = 0.0
        bar_colour = bar_colour or '#4C8DFF'
        bar = (f'<span class="f1-cell-bar"><span style="width:{pct:.0f}%;'
               f'background:{esc(bar_colour)}"></span></span>')
    return f'<td class="{css}">{swatch}{esc(value)}{bar}</td>'


def data_table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
               first_column_as_label: bool = True,
               compact: bool = False) -> None:
    """A table in the app's own style.

    ``st.dataframe`` brings its own look (a spreadsheet grid with its own
    header, borders and font), which is one of the reasons the pages read as
    cluttered. This renders the same information as a plain table that matches
    the rest of the app.

    Args:
        headers: the column titles.
        rows: the rows. A cell is a value, or a dict with ``value``, ``tone``
            (``good``, ``bad``, ``dim``, ``accent``, ``strong``),
            ``colour`` (a hex colour drawn as a swatch in front of the value),
            and optionally ``bar_pct`` (0-100) with ``bar_colour`` (a mini
            share bar under the value -- the teammate matrix).
        first_column_as_label: render the first column as a row label
            (left aligned, muted) instead of a value.
        compact: hug the column widths to their content instead of spreading
            to the full width -- for narrow tables (a few short columns)
            where the spread would leave a void next to right-aligned
            values.
    """
    head = ''.join(f'<th>{esc(header)}</th>' for header in headers or [])
    body: List[str] = []
    for row in rows or []:
        cells = list(row)
        if first_column_as_label and cells:
            label = cells[0]
            body.append(
                f'<tr><th class="f1-table-label">{esc(label)}</th>'
                f'{"".join(_cell_html(cell) for cell in cells[1:])}</tr>'
            )
        else:
            body.append(f'<tr>{"".join(_cell_html(cell) for cell in cells)}</tr>')
    table_class = 'f1-table f1-table--compact' if compact else 'f1-table'
    st.markdown(
        f'<div class="f1-table-wrap"><table class="{table_class}">'
        f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def _score_number(value: Any) -> Optional[float]:
    """Best-effort float for a scoreboard value (``P3.5`` counts as 3.5)."""
    try:
        return float(str(value).strip().lstrip('Pp').replace(',', ''))
    except (TypeError, ValueError):
        return None


def scoreboard(metrics: Sequence[Dict[str, Any]], left: str, right: str,
               colour_a: str = '', colour_b: str = '') -> None:
    """An A-versus-B scoreboard (the head-to-head comparison).

    Two columns of unrelated tiles are hard to read side by side: the eye has
    to jump between blocks to find the same metric. The scoreboard puts the
    same metric on one row, with the number that leads it highlighted and a
    proportional bar under the row so the gap is visible, not just read.

    Args:
        metrics: one dict per row -- ``label``, ``left``, ``right``,
            ``leader`` (``left``, ``right`` or empty) and an optional ``note``.
        left: the name of the driver on the left.
        right: the name of the driver on the right.
        colour_a / colour_b: livery colours for the bars (may be empty).
    """
    head = (
        f'<div class="f1-score-head">'
        f'<div class="f1-score-name f1-score-name--a">{esc(left)}</div>'
        f'<div class="f1-score-name f1-score-name--b">{esc(right)}</div>'
        f'</div>'
    )
    rows: List[str] = []
    for metric in metrics or []:
        leader = str(metric.get('leader') or '').strip().lower()
        left_css = 'f1-score-val f1-score-val--lead' if leader == 'left' else 'f1-score-val'
        right_css = 'f1-score-val f1-score-val--lead' if leader == 'right' else 'f1-score-val'
        note = metric.get('note')
        note_html = (f'<span class="f1-score-note">{esc(note)}</span>'
                     if note else '')
        # Mirrored bar: each half is scaled to the larger value of the pair,
        # so the longer bar always belongs to the driver ahead in that row.
        # Lower-is-better metrics (average position, best finish) are
        # inverted so "ahead" still means the longer bar.
        num_l, num_r = _score_number(metric.get('left')), _score_number(metric.get('right'))
        bar_html = ''
        if num_l is not None and num_r is not None and (num_l + num_r) > 0:
            if metric.get('lower_is_better'):
                num_l, num_r = (1.0 / num_l if num_l else 0.0), (1.0 / num_r if num_r else 0.0)
            peak = max(num_l, num_r)
            pct_l, pct_r = 100.0 * num_l / peak, 100.0 * num_r / peak
            bar_html = (
                f'<div class="f1-score-mirror">'
                f'<div class="f1-score-half f1-score-half--a" '
                f'style="--f1-duel-a:{esc(colour_a or "#4285F4")}"><span style="width:{pct_l:.0f}%"></span></div>'
                f'<div class="f1-score-half f1-score-half--b" '
                f'style="--f1-duel-b:{esc(colour_b or "#FFB020")}"><span style="width:{pct_r:.0f}%"></span></div>'
                f'</div>'
            )
        rows.append(
            f'<div class="f1-score-row">'
            f'<div class="{left_css}">{esc(metric.get("left"))}</div>'
            f'<div class="f1-score-label">{esc(metric.get("label"))}{note_html}</div>'
            f'<div class="{right_css}">{esc(metric.get("right"))}</div>'
            f'</div>{bar_html}'
        )
    st.markdown(
        f'<div class="f1-score">{head}{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


def _initials(name: str) -> str:
    """Two-letter initials for a driver name ("Kimi Antonelli" -> "KA")."""
    words = [w for w in str(name or '').strip().split() if w]
    if not words:
        return '?'
    if len(words) == 1:
        return words[0][:2].upper()
    return (words[0][0] + words[-1][0]).upper()


def duel_header(left: str, left_team: str, left_big: str, left_sub: str,
                right: str, right_team: str, right_big: str, right_sub: str,
                colour_a: str = '', colour_b: str = '') -> None:
    """Two livery tiles around a checkered VS badge (visual before text).

    Each tile carries the driver's initials as its profile mark -- there is
    no portrait asset in the app, so the avatar stands in for one.
    """
    st.markdown(
        '<div class="f1-duel">'
        f'<div class="f1-duel-tile" style="--f1-duel-colour:{esc(colour_a or "#4285F4")}">'
        f'<div class="f1-duel-avatar" aria-hidden="true">{esc(_initials(left))}</div>'
        f'<div class="f1-duel-name">{esc(left)}</div>'
        f'<div class="f1-duel-team">{esc(left_team)}</div>'
        f'<div class="f1-duel-big">{esc(left_big)}</div>'
        f'<div class="f1-duel-sub">{esc(left_sub)}</div></div>'
        '<div class="f1-duel-vs">VS</div>'
        f'<div class="f1-duel-tile f1-duel-tile--b" style="--f1-duel-colour:{esc(colour_b or "#FFB020")}">'
        f'<div class="f1-duel-avatar" aria-hidden="true">{esc(_initials(right))}</div>'
        f'<div class="f1-duel-name">{esc(right)}</div>'
        f'<div class="f1-duel-team">{esc(right_team)}</div>'
        f'<div class="f1-duel-big">{esc(right_big)}</div>'
        f'<div class="f1-duel-sub">{esc(right_sub)}</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def takeaways(items: Sequence[Tuple[str, str]]) -> None:
    """Derived insights for a comparison: ``(title, text)`` pairs.

    The page derives every line from the dataset; this only renders them.
    """
    if not items:
        return
    rows = ''.join(
        f'<div class="f1-takeaway">'
        f'<div class="f1-takeaway-title">{esc(title)}</div>'
        f'<div class="f1-takeaway-text">{esc(text)}</div></div>'
        for title, text in items
    )
    st.markdown(f'<div class="f1-takeaways">{rows}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Words and confidence
# ---------------------------------------------------------------------------

def explain(term: str, text: str) -> None:
    """One term, explained in one line (used next to a chart)."""
    st.markdown(
        f'<div class="f1-explain"><span class="f1-explain-term">{esc(term)}</span>'
        f'<span class="f1-explain-text">{esc(text)}</span></div>',
        unsafe_allow_html=True,
    )


def explainer_language() -> str:
    """The language the reader picked for the explanation layer (en or id)."""
    import glossary as glossary_module

    try:
        chosen = st.session_state.get('explainer_lang')
    except Exception:  # pragma: no cover - outside a Streamlit run
        chosen = None
    return glossary_module.language(chosen)


_GLOSSARY_TITLES = {
    'en': 'What the words mean',
    'id': 'Arti istilah',
}


def glossary(terms: Optional[Sequence[str]] = None,
             title: Optional[str] = None) -> None:
    """A collapsible plain-language guide to the terms used on the page.

    Args:
        terms: the terms to explain. ``None`` explains every term.
        title: the label of the collapsible block; defaults to the reader's
            language ("What the words mean" / "Arti istilah").
    """
    import glossary as glossary_module

    lang = explainer_language()
    entries = (glossary_module.all_entries(lang=lang) if terms is None
               else glossary_module.terms_for(terms, lang=lang))
    if not entries:
        return
    heading = title or _GLOSSARY_TITLES.get(lang, _GLOSSARY_TITLES['en'])
    items = ''.join(
        f'<div class="f1-glossary-item"><div class="f1-glossary-term">'
        f'{esc(entry["term"])}</div><div class="f1-glossary-text">'
        f'{esc(entry["long"])}</div></div>'
        for entry in entries
    )
    with st.expander(heading):
        st.markdown(f'<div class="f1-glossary">{items}</div>',
                    unsafe_allow_html=True)


# The friendly names of the confidence levels of `pace.data_quality`.
CONFIDENCE_LABELS = {
    'good': ('Good confidence', 'good'),
    'moderate': ('Medium confidence', 'warn'),
    'low': ('Low confidence', 'warn'),
    'insufficient': ('Not enough data', 'dim'),
}


def confidence_tag(level: str) -> None:
    """A chip that says how much data an estimate is based on."""
    key = str(level or '').strip().lower()
    label, tone = CONFIDENCE_LABELS.get(key, (str(level or 'unknown'), 'info'))
    chips([('', label)], tone=tone)
