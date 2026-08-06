# -*- coding: utf-8 -*-
"""
briefing.py
~~~~~~~~~~~
Data-driven race briefing and debrief.

Pre-race briefing: a concise narrative built only from real season data and
the qualifying grid — championship context, current form, team pace and the
model's predicted favourites. Post-race debrief: what actually happened vs the
grid — winner, podium, biggest gainer/loser, championship movement and (when a
forecast is supplied) how the forecast performed.

Both are pure functions returning Markdown, so they are unit-testable without
Streamlit and reuse the prediction model's transparent helpers. The Data
Honesty contract applies: every projected number is flagged estimated and
derived from real results; when a race has no result we say so instead of
inventing one.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Ensure src/ is importable no matter how this module is reached
# (as `briefing` from a page, or as `src.briefing` from the test suite).
sys.path.insert(0, str(Path(__file__).parent))

from prediction import compute_driver_form, compute_team_pace, predict_race

ESTIMATED = "estimated"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _race_rows(df: pd.DataFrame, race: str) -> pd.DataFrame:
    """Race-session rows for one track (drops sprint rows when present)."""
    out = df.copy()
    if 'SessionType' in out.columns:
        out = out[out['SessionType'] == 'Race']
    if 'Track' in out.columns:
        out = out[out['Track'] == race]
    return out


def _standings(df: pd.DataFrame, up_to_race: Optional[str] = None) -> Optional[pd.Series]:
    """Cumulative driver points; ``up_to_race`` excludes that race and after."""
    if df is None or df.empty or 'Driver' not in df.columns or 'Points' not in df.columns:
        return None
    g = df
    if up_to_race and 'Track' in g.columns:
        order = list(dict.fromkeys(g['Track'].astype(str)))
        if up_to_race in order:
            g = g[g['Track'].isin(order[:order.index(up_to_race)])]
    if g.empty:
        return None
    s = g.groupby('Driver')['Points'].sum().sort_values(ascending=False)
    return s if not s.empty else None


def _grid_delta_md(df: pd.DataFrame, race: str) -> List[str]:
    """Biggest gainer and loser versus the grid, as Markdown lines."""
    sub = _race_rows(df, race)
    if sub.empty:
        return []
    g = sub.copy()
    g['_pos'] = pd.to_numeric(g.get('Position'), errors='coerce')
    g['_grid'] = pd.to_numeric(g.get('Starting Grid'), errors='coerce')
    g = g.dropna(subset=['_pos', '_grid'])
    if g.empty:
        return []
    g['_gain'] = g['_grid'] - g['_pos']
    lines: List[str] = []
    best = g.loc[g['_gain'].idxmax()]
    if int(best['_gain']) > 0:
        lines.append(f"**{best['Driver']}** gained **{int(best['_gain'])}** places "
                     f"(P{int(best['_grid'])} → P{int(best['_pos'])}).")
    worst = g.loc[g['_gain'].idxmin()]
    if int(worst['_gain']) < 0:
        lines.append(f"**{worst['Driver']}** lost **{int(-worst['_gain'])}** places "
                     f"(P{int(worst['_grid'])} → P{int(worst['_pos'])}).")
    return lines


# ---------------------------------------------------------------------------
# Pre-race briefing
# ---------------------------------------------------------------------------

def championship_leader_md(df: pd.DataFrame,
                           up_to_race: Optional[str] = None) -> str:
    """Current championship leader and their points gap, as Markdown."""
    s = _standings(df, up_to_race)
    if s is None or len(s) < 1:
        return ""
    leader, pts = s.index[0], int(s.iloc[0])
    if len(s) > 1 and int(s.iloc[1]) > 0:
        gap = pts - int(s.iloc[1])
        return (f"**{leader}** leads the championship on **{pts}** pts, "
                f"**{gap}** clear of **{s.index[1]}**.")
    return f"**{leader}** leads the championship on **{pts}** pts."


def form_guide_md(df: pd.DataFrame, top: int = 5) -> str:
    """Current form and team-pace leaders (estimated), as Markdown."""
    form = compute_driver_form(df)
    if not form:
        return ""
    top_drivers = sorted(form, key=form.get, reverse=True)[:top]
    lines = [f"Current form leaders: **{', '.join(top_drivers)}** (estimated)."]
    teams = compute_team_pace(df)
    if teams:
        top_teams = sorted(teams, key=teams.get, reverse=True)[:3]
        lines.append(f"Team pace leaders: **{', '.join(top_teams)}** (estimated).")
    return "\n".join(lines)


def pre_race_briefing(df: pd.DataFrame, grid: Dict[str, int],
                      race: Optional[str] = None) -> str:
    """Concise pre-race briefing from season data and the qualifying grid.

    Args:
        df: season race results (pace/form context).
        grid: {driver: grid position} for the upcoming race.
        race: grand prix name for the header (optional).

    Returns:
        Markdown string; every model figure is flagged estimated.
    """
    sections: List[str] = []
    if race:
        sections.append(f"**Pre-race briefing — {race}**")

    leader = championship_leader_md(df)
    if leader:
        sections.append(leader)

    form = form_guide_md(df)
    if form:
        sections.append(form)

    pred = predict_race(df, grid)
    if not pred.empty:
        top3 = ", ".join(
            f"**{r['Driver']}** ({r['WinProb']}%)" for _, r in pred.head(3).iterrows()
        )
        sections.append(
            f"Model favourites (estimated): {top3}."
        )
        sections.append(
            "_Forecast is estimated from season pace and form — a ranking "
            "guide, not a guaranteed result._"
        )
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Post-race debrief
# ---------------------------------------------------------------------------

def post_race_debrief(df: pd.DataFrame, race: str,
                      forecast: Optional[pd.DataFrame] = None) -> str:
    """Data-driven debrief of what actually happened at ``race``.

    Args:
        df: season race results.
        race: grand prix name.
        forecast: optional ``predict_race`` output for this race; when given,
            its podium call is checked against the real podium.

    Returns:
        Markdown string; model figures are flagged estimated.
    """
    sub = _race_rows(df, race)
    if sub.empty:
        return f"**No recorded result for {race}.**"

    sections: List[str] = [f"**Post-race debrief — {race}**"]

    sub = sub.copy()
    sub['_posnum'] = pd.to_numeric(sub.get('Position'), errors='coerce')
    win = sub[sub['_posnum'] == 1]
    if not win.empty:
        sections.append(f"**{win.iloc[0]['Driver']}** wins at **{race}**.")

    podium = sub.dropna(subset=['_posnum']).sort_values('_posnum').head(3)
    if not podium.empty:
        sections.append(
            "Podium: " + ", ".join(f"P{i + 1} **{r['Driver']}**"
                                   for i, (_, r) in enumerate(podium.iterrows())) + "."
        )

    sections.extend(_grid_delta_md(df, race))

    before = _standings(df, up_to_race=race)
    after = _standings(df)
    if before is not None and after is not None and len(after) > 0:
        lbefore, lafter = before.index[0], after.index[0]
        if lbefore != lafter:
            sections.append(f"Championship lead changes hands: **{lbefore}** → **{lafter}**.")
        else:
            sections.append(f"Championship leader unchanged: **{lafter}**.")

    if forecast is not None and not forecast.empty:
        pred_top3 = set(forecast.head(3)['Driver'])
        act_top3 = set(podium['Driver'])
        matched = len(pred_top3 & act_top3)
        sections.append(
            f"Forecast (estimated) matched **{matched}/3** of the podium."
        )

    return "\n\n".join(sections)
