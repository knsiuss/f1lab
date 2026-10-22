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

import html
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import streamlit as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def esc(value: Any) -> str:
    """HTML-escape a value for safe interpolation into markup."""
    if value is None:
        return ''
    return html.escape(str(value), quote=True)


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
        f'<div class="f1-section"><h2 class="f1-section-title">{esc(title)}</h2>'
        f'{note_html}{hint_html}</div>',
        unsafe_allow_html=True,
    )


def card(content: str, flat: bool = False) -> None:
    """Wrap arbitrary (already escaped) markup in a surface card."""
    css = 'f1-card' + (' f1-card--flat' if flat else '')
    st.markdown(f'<div class="{css}">{content}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# KPI tiles
# ---------------------------------------------------------------------------

def kpis(items: Sequence[Dict[str, Any]], columns: Optional[int] = None) -> None:
    """A row of KPI tiles.

    Each item is a dict with ``label``, ``value`` and optional ``sub`` and
    ``tone`` (``accent``, ``good``, ``warn``, ``info``).
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
    else:
        value = cell
        tone = ''
        colour = ''
    css = 'f1-table-cell' + (f' f1-table-cell--{tone}' if tone else '')
    swatch = (f'<span class="f1-swatch" style="background:{esc(colour)}"></span>'
              if colour else '')
    return f'<td class="{css}">{swatch}{esc(value)}</td>'


def data_table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
               first_column_as_label: bool = True) -> None:
    """A table in the app's own style.

    ``st.dataframe`` brings its own look (a spreadsheet grid with its own
    header, borders and font), which is one of the reasons the pages read as
    cluttered. This renders the same information as a plain table that matches
    the rest of the app.

    Args:
        headers: the column titles.
        rows: the rows. A cell is a value, or a dict with ``value``, ``tone``
            (``good``, ``bad``, ``dim``, ``accent``, ``strong``) and
            ``colour`` (a hex colour drawn as a swatch in front of the value).
        first_column_as_label: render the first column as a row label
            (left aligned, muted) instead of a value.
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
    st.markdown(
        f'<div class="f1-table-wrap"><table class="f1-table">'
        f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def scoreboard(metrics: Sequence[Dict[str, Any]], left: str, right: str) -> None:
    """An A-versus-B scoreboard (the head-to-head comparison).

    Two columns of unrelated tiles are hard to read side by side: the eye has
    to jump between blocks to find the same metric. The scoreboard puts the
    same metric on one row, with the number that leads it highlighted.

    Args:
        metrics: one dict per row -- ``label``, ``left``, ``right``,
            ``leader`` (``left``, ``right`` or empty) and an optional ``note``.
        left: the name of the driver on the left.
        right: the name of the driver on the right.
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
        rows.append(
            f'<div class="f1-score-row">'
            f'<div class="{left_css}">{esc(metric.get("left"))}</div>'
            f'<div class="f1-score-label">{esc(metric.get("label"))}{note_html}</div>'
            f'<div class="{right_css}">{esc(metric.get("right"))}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div class="f1-score">{head}{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


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
