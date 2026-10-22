# -*- coding: utf-8 -*-
"""
report.py
~~~~~~~~~
Race-weekend report generator.

Builds a shareable, dependency-free markdown report from season race results
(offline) and a handful of optional extras from a FastF1 session. Pure functions so
the whole builder is unit-testable without Streamlit.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

from typing import Dict, List, Optional

import pandas as pd


def _num(value, default=None):
    try:
        val = float(value)
        return None if pd.isna(val) else val
    except (TypeError, ValueError):
        return default


def _text(value, default=''):
    """A cell as text, with missing values shown as the default."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return str(value)


def _int_or_dash(value) -> str:
    """An integer cell, or a dash when the number is missing."""
    number = _num(value)
    return '—' if number is None else str(int(number))


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """A numeric view of a column, or all-NaN when the column is missing.

    ``df.get(column)`` returns ``None`` for a missing column and
    ``pd.to_numeric(None)`` returns the boolean ``False`` -- indexing a frame
    with that raises ``KeyError: False``. Every numeric lookup goes through
    this helper instead.
    """
    if column in df.columns:
        return pd.to_numeric(df[column], errors='coerce')
    return pd.Series(float('nan'), index=df.index, dtype=float)


def _sub_race(df: pd.DataFrame, race: str) -> pd.DataFrame:
    """Race-session rows for a given track, handling sprint/race slices."""
    out = df.copy()
    if 'SessionType' in out.columns:
        out = out[out['SessionType'] == 'Race']
    if 'Track' in out.columns:
        out = out[out['Track'] == race]
    return out


def season_standings_md(df: pd.DataFrame, top: int = 10) -> str:
    """Championship standings as a Markdown table.

    The frame is expected to hold race **and** sprint rows: the championship
    counts both, so the table shows the real championship total. A note says
    so whenever sprint rows are present.
    """
    if df is None or df.empty or 'Driver' not in df.columns:
        return "_No season data available._"
    grouped = df.groupby('Driver')
    g = grouped.agg(Points=('Points', 'sum') if 'Points' in df.columns
                    else ('Driver', 'size'))
    g['GP'] = (grouped['Track'].nunique() if 'Track' in df.columns else 0)
    g['Team'] = (grouped['Team'].first() if 'Team' in df.columns else '')
    g = g.sort_values('Points', ascending=False).head(top).reset_index()

    lines = ["Position | Driver | Team | GPs | Points",
             "---:|---|---|---:|---:"]
    for i, row in g.iterrows():
        lines.append(f"{i + 1} | {row['Driver']} | {_text(row['Team'])} "
                     f"| {int(row['GP'])} | {_int_or_dash(row['Points'])}")
    if _has_sprints(df):
        lines.append("")
        lines.append("_Standings include sprint points._")
    return "\n".join(lines)


def _has_sprints(df: pd.DataFrame) -> bool:
    """Does the frame carry sprint rows (so the totals are championship ones)?"""
    if df is None or df.empty or 'SessionType' not in df.columns:
        return False
    return bool((df['SessionType'].astype(str) == 'Sprint').any())


def race_result_md(df: pd.DataFrame, race: str) -> str:
    """Official race result table for a single race."""
    sub = _sub_race(df, race)
    if sub.empty:
        return "_No recorded result for this race._"

    sub = sub.copy()
    sub['_posnum'] = _numeric_column(sub, 'Position')
    sub['_grid'] = _numeric_column(sub, 'Starting Grid')
    sub = sub.sort_values('_posnum')

    lines = ["Pos | Driver | Team | Grid | Laps | Pts |", "---:|---|---|---:|---:|---:"]
    for _, r in sub.iterrows():
        pos = r.get('Position')
        pnum = _num(pos)
        # 'NC' (not classified) is not the same result as 'DNF': it is kept as
        # it is reported.
        pos_s = str(int(pnum)) if pnum is not None else str(pos).strip().upper()
        grid_s = _int_or_dash(r.get('Starting Grid'))
        laps_s = _int_or_dash(r.get('Laps'))
        points_s = _int_or_dash(r.get('Points'))
        lines.append(
            f"{pos_s} | {_text(r.get('Driver'))} | {_text(r.get('Team'))} "
            f"| {grid_s} | {laps_s} | {points_s}"
        )
    return "\n".join(lines)


def build_insights(df: pd.DataFrame, race: str, extra: Optional[Dict] = None) -> List[str]:
    """Human-readable auto insights for a race weekend."""
    sub = _sub_race(df, race)
    insights: List[str] = []
    if sub.empty:
        insights.append(f"No result data available for **{race}**.")
        return insights

    positions = _numeric_column(sub, 'Position')
    grids = _numeric_column(sub, 'Starting Grid')
    win = sub[positions == 1]
    pole = sub[grids == 1]
    if not win.empty:
        insights.append(f"**{win.iloc[0]['Driver']}** took the win at **{race}**.")
    if not pole.empty:
        insights.append(f"Pole position went to **{pole.iloc[0]['Driver']}**.")

    sub2 = sub.copy()
    sub2['_pos'] = positions
    sub2['_grid'] = grids
    sub2 = sub2.dropna(subset=['_pos', '_grid'])
    if not sub2.empty:
        sub2['_gain'] = sub2['_grid'] - sub2['_pos']
        best = sub2.loc[sub2['_gain'].idxmax()]
        if int(best['_gain']) > 0:
            insights.append(
                f"**{best['Driver']}** gained **{int(best['_gain'])}** places "
                f"({int(best['_grid'])} → P{int(best['_pos'])})."
            )

    if extra:
        if extra.get('pit_stops') is not None:
            insights.append(f"A total of **{int(extra['pit_stops'])}** pit stops recorded.")
        if extra.get('lead_changes') is not None:
            insights.append(f"A total of **{int(extra['lead_changes'])}** lead changes.")
    return insights


def build_report(
    year: int,
    race: str,
    df: pd.DataFrame,
    extra: Optional[Dict] = None,
    extra_sections: Optional[List[str]] = None,
) -> str:
    """Assemble the full Markdown weekend report.

    Args:
        year: season year.
        race: grand prix name.
        df: season race results.
        extra: optional dict of KPI counts (pit_stops, lead_changes).
        extra_sections: optional extra Markdown blocks appended (e.g. tyre stints).
    """
    lines = [
        f"# F1 {year} Weekend Report",
        "",
        f"## {race}",
        "",
        "### Season Standings",
        season_standings_md(df),
        "",
        "### Race Result",
        "",
        race_result_md(df, race),
        "",
        "### Insights",
        "",
    ]
    insights = build_insights(df, race, extra)
    lines.extend(f"- {i}" for i in insights)
    if not insights:
        lines.append("_No insights generated._")

    if extra_sections:
        lines.append("")
        for block in extra_sections:
            lines.append(block)
            lines.append("")

    return "\n".join(lines)