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

from prediction import (
    compute_driver_form,
    compute_team_pace,
    predict_race,
    rows_before_race,
)

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


def _track_order(df: pd.DataFrame) -> list:
    """Calendar order of the tracks in a frame.

    The ``Round`` column is authoritative when the frame has one; otherwise
    the first-appearance order of the Track column is used.
    """
    if df is None or df.empty or 'Track' not in df.columns:
        return []
    if 'Round' in df.columns:
        rounds = pd.to_numeric(df['Round'], errors='coerce')
        if rounds.notna().any():
            ordered = (df.assign(_round=rounds)
                         .dropna(subset=['_round'])
                         .sort_values('_round', kind='stable'))
            tracks = list(dict.fromkeys(ordered['Track'].dropna().astype(str).tolist()))
            if tracks:
                return tracks
    return list(dict.fromkeys(df['Track'].dropna().astype(str).tolist()))


def _standings(df: pd.DataFrame, up_to_race: Optional[str] = None) -> Optional[pd.Series]:
    """Cumulative driver points; ``up_to_race`` excludes that race and after."""
    if df is None or df.empty or 'Driver' not in df.columns or 'Points' not in df.columns:
        return None
    g = df
    if up_to_race and 'Track' in g.columns:
        order = _track_order(g)
        if up_to_race in order:
            g = g[g['Track'].isin(order[:order.index(up_to_race)])]
    if g.empty:
        return None
    s = g.groupby('Driver')['Points'].sum().sort_values(ascending=False)
    return s if not s.empty else None


def _df_before(df: pd.DataFrame, race: Optional[str]) -> pd.DataFrame:
    """Rows for the rounds strictly before ``race``.

    Uses the shared look-ahead cut (:func:`prediction.rows_before_race`) so the
    briefing can never see the target race or anything after it. Unknown/None
    race falls back to the full frame so callers keep working when the target
    GP is not in the data (e.g. a future race with no rows).
    """
    if not race or df is None or df.empty or 'Track' not in df.columns:
        return df
    earlier = rows_before_race(df, race)
    if earlier is None or earlier.empty:
        # The race is not in the frame: nothing before it can be identified.
        return df
    return earlier


def _standings_through(df: pd.DataFrame, race: Optional[str]) -> Optional[pd.Series]:
    """Cumulative driver points up to and including ``race`` itself."""
    if df is None or df.empty or 'Driver' not in df.columns or 'Points' not in df.columns:
        return None
    g = df
    if race and 'Track' in g.columns:
        order = _track_order(g)
        if race in order:
            g = g[g['Track'].isin(order[:order.index(race) + 1])]
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
    g['_pos'] = pd.to_numeric(g['Position'], errors='coerce') if 'Position' in g.columns \
        else pd.Series(float('nan'), index=g.index)
    g['_grid'] = pd.to_numeric(g['Starting Grid'], errors='coerce') if 'Starting Grid' in g.columns \
        else pd.Series(float('nan'), index=g.index)
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

    Only races strictly before ``race`` feed the standings, form, pace and
    forecast — the briefing must not peek at the target race or later ones.
    """
    sections: List[str] = []
    if race:
        sections.append(f"**Pre-race briefing — {race}**")

    past = _df_before(df, race)

    leader = championship_leader_md(past)
    if leader:
        sections.append(leader)

    form = form_guide_md(past)
    if form:
        sections.append(form)

    pred = predict_race(past, grid)
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
                      forecast: Optional[pd.DataFrame] = None,
                      forecast_includes_race: bool = False) -> str:
    """Data-driven debrief of what actually happened at ``race``.

    Args:
        df: season race results.
        race: grand prix name.
        forecast: optional ``predict_race`` output for this race; when given,
            its podium call is checked against the real podium.
        forecast_includes_race: set when the forecast was built from rows
            that already contain this race's own result. Such a forecast is
            retrospective, not a prediction, and the debrief says so instead
            of reporting it as a pre-race call.

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
    after = _standings_through(df, race)
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
        if forecast_includes_race:
            # A forecast that saw the race's own result is a retrospective
            # check, not a pre-race call: saying so is the honest wording.
            sections.append(
                f"Retrospective model check (estimated, forecast built *after* "
                f"the race and therefore including its result): matched "
                f"**{matched}/3** of the podium. This is not a forecast "
                f"accuracy figure."
            )
        else:
            sections.append(
                f"Pre-race forecast (estimated, built from the rounds before "
                f"this race) matched **{matched}/3** of the podium."
            )

    return "\n\n".join(sections)
