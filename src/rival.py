# -*- coding: utf-8 -*-
"""
rival.py
~~~~~~~~
Competitor / rival intelligence from season results.

Pure pandas over the offline season CSV, so it is deterministic and unit
testable. Answers the questions a team analyst asks about a rival: who is
nearest in the championship, how does a specific head-to-head look, and what
does a circuit tend to reward (pole-to-win, overtakes, fastest lap).

Data Honesty contract: head-to-head and circuit figures are only shown when
enough completed races exist, sample sizes are reported, and every derived
figure is labelled ``estimated``. With too little data the helpers return an
``insufficient`` level instead of inventing a rivalry.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Ensure src/ is importable no matter how this module is reached
# (as `rival` from a page, or as `src.rival` from the test suite).
sys.path.insert(0, str(Path(__file__).parent))

from pace import data_quality

ESTIMATED = "estimated"

# Completed races two drivers must share before a head-to-head is shown.
MIN_RACES_FOR_RIVALRY = 2


def _finishers(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a numeric finishing position (excludes DNF/DNS/DSQ)."""
    out = df.copy()
    if 'Position' not in out.columns:
        return out.iloc[0:0]
    out['_pos'] = pd.to_numeric(out['Position'], errors='coerce')
    return out.dropna(subset=['_pos'])


def _standings(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Per-driver season points and team, sorted best-first."""
    if df is None or df.empty or 'Driver' not in df.columns:
        return None
    if 'Points' in df.columns:
        g = df.groupby('Driver').agg(
            Team=('Team', 'first'), Points=('Points', 'sum')
        ).sort_values('Points', ascending=False).reset_index()
    else:
        return None
    return g if not g.empty else None


def head_to_head(season_df: pd.DataFrame,
                 driver_a: str, driver_b: str) -> Dict[str, object]:
    """Rival head-to-head across completed races both drivers finished.

    Uses only races where both drivers have a finishing position. Counts who
    finished ahead, reports average positions and the points difference, and
    labels the result with a confidence level from the sample size.

    Returns:
        dict with wins, averages, ``points_diff``, ``level``/``confidence``
        and ``estimated``; a too-small sample returns ``level=insufficient``.
    """
    fin = _finishers(season_df)
    if fin.empty or 'Track' not in fin.columns:
        return _insufficient(driver_a, driver_b)

    a = fin[fin['Driver'] == driver_a][['Track', '_pos']].drop_duplicates('Track')
    b = fin[fin['Driver'] == driver_b][['Track', '_pos']].drop_duplicates('Track')
    merged = a.merge(b, on='Track', suffixes=('_a', '_b'))
    n = len(merged)
    if n < MIN_RACES_FOR_RIVALRY:
        return _insufficient(driver_a, driver_b, n=n)

    a_wins = int((merged['_pos_a'] < merged['_pos_b']).sum())
    b_wins = int((merged['_pos_b'] < merged['_pos_a']).sum())
    draws = n - a_wins - b_wins
    a_avg = float(merged['_pos_a'].mean())
    b_avg = float(merged['_pos_b'].mean())

    pts_diff = None
    if 'Points' in season_df.columns:
        pa = season_df.loc[season_df['Driver'] == driver_a, 'Points'].sum()
        pb = season_df.loc[season_df['Driver'] == driver_b, 'Points'].sum()
        pts_diff = int(pa - pb)

    quality = data_quality(n)
    return {
        "driver_a": driver_a, "driver_b": driver_b,
        "races": n, "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
        "a_avg_pos": round(a_avg, 2), "b_avg_pos": round(b_avg, 2),
        "net_pos": round(a_avg - b_avg, 2), "points_diff": pts_diff,
        "level": quality["level"], "confidence": quality["confidence"],
        "estimated": ESTIMATED,
    }


def _insufficient(driver_a: str, driver_b: str, n: int = 0) -> Dict[str, object]:
    return {
        "driver_a": driver_a, "driver_b": driver_b, "races": n,
        "level": "insufficient", "confidence": "low",
        "estimated": ESTIMATED,
    }


def build_watchlist(season_df: pd.DataFrame,
                    focus: Optional[str] = None,
                    top: int = 5) -> pd.DataFrame:
    """
    Championship watchlist: the drivers nearest a focus driver.

    With ``focus``, returns that driver plus the nearest rivals by points gap
    (closest above/below). Without a focus, returns the championship leaders.
    ``GapToFocus`` is points ahead (positive) or behind (negative) of the
    focus driver; the row set carries a confidence label from the season
    length (points themselves are real results, rival status is judgement).

    Returns:
        DataFrame (possibly empty) with Driver, Team, Points, GapToFocus,
        Level, Confidence, Estimated.
    """
    standings = _standings(season_df)
    if standings is None or standings.empty:
        return pd.DataFrame()

    n_races = season_df['Track'].nunique() if 'Track' in season_df.columns else 0
    quality = data_quality(n_races)

    if focus is not None and focus in set(standings['Driver']):
        focus_pts = int(standings.loc[standings['Driver'] == focus, 'Points'].iloc[0])
        others = standings[standings['Driver'] != focus].copy()
        others['_gap'] = (others['Points'] - focus_pts).astype(int)
        others['_dist'] = others['_gap'].abs()
        nearest = others.sort_values(['_dist', 'Points'],
                                     ascending=[True, False]).head(top - 1)
        watch = pd.concat([
            standings[standings['Driver'] == focus].copy(),
            nearest,
        ])
    else:
        watch = standings.head(top).copy()
        watch['_gap'] = (watch['Points'] - watch['Points'].max()).astype(int)

    out = watch.sort_values('Points', ascending=False).reset_index(drop=True)
    out.insert(0, 'Rank', range(1, len(out) + 1))
    out = out.rename(columns={'_gap': 'GapToFocus'})
    out['Level'] = quality["level"]
    out['Confidence'] = quality["confidence"]
    out['Estimated'] = ESTIMATED
    return out[['Rank', 'Driver', 'Team', 'Points', 'GapToFocus',
                'Level', 'Confidence', 'Estimated']]


def circuit_insights(season_df: pd.DataFrame, track: str) -> Dict[str, object]:
    """What a circuit has rewarded so far this season (estimated).

    Reports the latest winner, the pole position, pole-to-win conversion,
    average places gained by finishers and the fastest-lap setter, all with a
    data-quality level from the number of completed races at the track.

    Returns:
        dict (possibly ``level=insufficient``) with the figures and
        assumptions used.
    """
    if season_df is None or season_df.empty:
        return {"track": track, "level": "insufficient",
                "confidence": "low", "estimated": ESTIMATED}

    sub = season_df.copy()
    if 'SessionType' in sub.columns:
        sub = sub[sub['SessionType'] == 'Race']
    if 'Track' in sub.columns:
        sub = sub[sub['Track'] == track]
    if sub.empty:
        return {"track": track, "level": "insufficient",
                "confidence": "low", "estimated": ESTIMATED}

    n = sub['Driver'].nunique() if 'Driver' in sub.columns else 0
    quality = data_quality(n)

    out: Dict[str, object] = {
        "track": track, "races": 1, "level": quality["level"],
        "confidence": quality["confidence"], "estimated": ESTIMATED,
    }

    sub2 = sub.copy()
    sub2['_pos'] = pd.to_numeric(sub2.get('Position'), errors='coerce')
    sub2['_grid'] = pd.to_numeric(sub2.get('Starting Grid'), errors='coerce')

    win = sub2[sub2['_pos'] == 1]
    if not win.empty:
        out['winner'] = win.iloc[0]['Driver']

    pole = sub2[sub2['_grid'] == 1]
    if not pole.empty:
        out['pole'] = pole.iloc[0]['Driver']
        out['pole_to_win'] = bool(
            not win.empty and win.iloc[0]['Driver'] == pole.iloc[0]['Driver']
        )

    fin = sub2.dropna(subset=['_pos', '_grid'])
    if not fin.empty:
        out['avg_places_gained'] = round(
            float((fin['_grid'] - fin['_pos']).mean()), 2
        )

    if 'Set Fastest Lap' in sub2.columns:
        fl = sub2[sub2['Set Fastest Lap'].astype(str).str.strip().str.lower() == 'yes']
        if not fl.empty:
            out['fastest_lap'] = fl.iloc[0]['Driver']

    out['assumptions'] = [
        "figures cover completed races at this track in the selected season",
        f"{n} drivers scored",
    ]
    return out


def rival_attribution_md(h2h: Dict[str, object]) -> str:
    """One-line honest summary of a head-to-head result, as Markdown."""
    if h2h.get("level") == "insufficient":
        return (f"**{h2h.get('driver_a')} vs {h2h.get('driver_b')}**: "
                f"insufficient data ({h2h.get('races', 0)} shared finishes).")
    diff = h2h['a_avg_pos'] - h2h['b_avg_pos']
    leader = h2h['driver_a'] if diff < 0 else h2h['driver_b']
    line = (f"**{h2h['driver_a']} vs {h2h['driver_b']}** "
            f"({h2h['races']} races, confidence {h2h['confidence']}): "
            f"{h2h['driver_a']} ahead {h2h['a_wins']}–{h2h['b_wins']}, "
            f"avg finish P{h2h['a_avg_pos']} vs P{h2h['b_avg_pos']} "
            f"(est); {leader} faster on average by {abs(diff):.2f} places.")
    return line
