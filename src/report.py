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


def _sub_race(df: pd.DataFrame, race: str) -> pd.DataFrame:
    """Race-session rows for a given track, handling sprint/race slices."""
    out = df.copy()
    if 'SessionType' in out.columns:
        out = out[out['SessionType'] == 'Race']
    if 'Track' in out.columns:
        out = out[out['Track'] == race]
    return out


def season_standings_md(df: pd.DataFrame, top: int = 10) -> str:
    """Championship standings as a Markdown table."""
    if df is None or df.empty or 'Driver' not in df.columns:
        return "_No season data available._"
    g = df.groupby('Driver').agg(
        Team=('Team', 'first'),
        GP=('Track', 'nunique'),
        Points=('Points', 'sum'),
    ).sort_values('Points', ascending=False).head(top).reset_index()

    lines = ["Position | Driver | Team | GPs | Points",
             "---:|---|---|---:|---:"]
    for i, row in g.iterrows():
        lines.append(f"{i + 1} | {row['Driver']} | {row['Team']} | {int(row['GP'])} | {int(row['Points'])}")
    return "\n".join(lines)


def race_result_md(df: pd.DataFrame, race: str) -> str:
    """Official race result table for a single race."""
    sub = _sub_race(df, race)
    if sub.empty:
        return "_No recorded result for this race._"

    sub = sub.copy()
    sub['_posnum'] = pd.to_numeric(sub.get('Position'), errors='coerce')
    sub['_grid'] = pd.to_numeric(sub.get('Starting Grid'), errors='coerce')
    sub = sub.sort_values('_posnum')

    lines = ["Pos | Driver | Team | Grid | Laps | Pts |", "---:|---|---|---:|---:|---:"]
    dnf_flags = ('DNF', 'DNS', 'DSQ', 'NC')
    for _, r in sub.iterrows():
        pos = r.get('Position')
        pnum = _num(pos)
        pos_s = str(int(pnum)) if pnum is not None else str(pos).strip().upper()
        if pnum is None and pos_s in dnf_flags:
            pos_s = pos_s if pos_s != 'NC' else 'DNF'
        grid = r.get('Starting Grid')
        grid_s = '—' if _num(grid) is None else str(int(_num(grid)))
        lines.append(
            f"{pos_s} | {r['Driver']} | {r.get('Team', '')} | {grid_s} | {int(r['Laps'])} | {int(r['Points'])}"
        )
    return "\n".join(lines)


def build_insights(df: pd.DataFrame, race: str, extra: Optional[Dict] = None) -> List[str]:
    """Human-readable auto insights for a race weekend."""
    sub = _sub_race(df, race)
    insights: List[str] = []
    if sub.empty:
        insights.append(f"No result data available for **{race}**.")
        return insights

    win = sub[pd.to_numeric(sub.get('Position'), errors='coerce') == 1]
    pole = sub[pd.to_numeric(sub.get('Starting Grid'), errors='coerce') == 1]
    if not win.empty:
        insights.append(f"**{win.iloc[0]['Driver']}** took the win at **{race}**.")
    if not pole.empty:
        insights.append(f"Pole position went to **{pole.iloc[0]['Driver']}**.")

    sub2 = sub.copy()
    sub2['_pos'] = pd.to_numeric(sub2.get('Position'), errors='coerce')
    sub2['_grid'] = pd.to_numeric(sub2.get('Starting Grid'), errors='coerce')
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