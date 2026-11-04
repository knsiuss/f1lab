# -*- coding: utf-8 -*-
"""Championship Race analytics: pure data layer (no Streamlit).

One source of truth for the "Championship race" tab: every figure on the
page (trajectory, per-race points, position evolution, gaps, heatmap,
consistency, scenarios, insights) is derived here from the season frame
produced by ``shared.load_race_data`` so the charts can never disagree.

Conventions (matching ``analysis``):
    * The frame carries one "Race" row and, on sprint weekends, one
      "Sprint" row per driver for the same ``Track``.
    * Championship points are race + sprint points summed per round
      ("combined") unless a caller asks for "race" or "sprint" only.
    * Championship position after a round is ranked on cumulative points
      over *all* drivers, not only the selected ones.
    * A driver with no row in a round is absent, not on zero points, so
      late debuts do not read as scoreless races -- see ``round_points``
      for how absence versus a genuine zero is represented.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# Maximum points a driver can score in one Grand Prix weekend: 25 for the
# race win plus the 1 point for fastest lap. Sprint-weekend maxima add
# the sprint win (8 points; no fastest-lap point in a sprint).
MAX_POINTS_PER_GP = 26
MAX_POINTS_PER_SPRINT = 8
POINTS_MODES = ("combined", "race", "sprint")


def short_track(track) -> str:
    """Short axis label for a Grand Prix ("Mexico City", not the full name)."""
    name = str(track or "").strip()
    for suffix in (" Grand Prix", " GP", " Grand Prix".lower()):
        if name.lower().endswith(suffix.lower()):
            name = name[: -len(suffix)]
            break
    name = name.strip()
    return name or str(track or "")


def ordered_rounds(frame: pd.DataFrame) -> list:
    """Chronological ``[(round_number, track), ...]`` for the rounds in frame."""
    if frame is None or frame.empty or "Track" not in frame.columns:
        return []
    rounds = []
    if "Round" in frame.columns:
        work = frame.assign(
            _round=pd.to_numeric(frame["Round"], errors="coerce")
        ).dropna(subset=["_round", "Track"])
        work = work.sort_values("_round", kind="stable")
        seen = {}
        for rnd, track in zip(work["_round"], work["Track"].astype(str)):
            seen.setdefault(track, int(rnd))
        rounds = sorted(seen.items(), key=lambda item: item[1])
        if rounds:
            return [(rnd, track) for track, rnd in rounds]
    tracks = list(dict.fromkeys(frame["Track"].dropna().astype(str)))
    return [(i + 1, track) for i, track in enumerate(tracks)]


def round_labels(frame: pd.DataFrame) -> list:
    """Short labels in chronological order, e.g. ``["Australia", "China"]``."""
    return [short_track(track) for _rnd, track in ordered_rounds(frame)]


def driver_meta(frame: pd.DataFrame) -> dict:
    """Per-driver ``{team, number, abbrev}`` from their most recent rows.

    The season frame has no broadcast abbreviation column, so the
    abbreviation is the first three letters of the surname (upper case);
    the car number comes from the frame's ``No`` column when present.
    """
    meta = {}
    if frame is None or frame.empty or "Driver" not in frame.columns:
        return meta
    work = frame.copy()
    if "Round" in work.columns:
        work = work.assign(
            _round=pd.to_numeric(work["Round"], errors="coerce")
        ).sort_values("_round", kind="stable")
    for driver, group in work.groupby("Driver"):
        last = group.iloc[-1]
        team = str(last.get("Team") or "").strip()
        number = None
        if "No" in group.columns:
            raw = pd.to_numeric(group["No"], errors="coerce").dropna()
            if not raw.empty:
                number = int(raw.iloc[-1])
        surname = str(driver).split()[-1] if str(driver).split() else str(driver)
        meta[str(driver)] = {
            "team": team,
            "number": number,
            "abbrev": surname[:3].upper(),
        }
    return meta


def _points_frame(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Rows of ``frame`` contributing points under ``mode``."""
    if mode not in POINTS_MODES:
        raise ValueError(f"unknown points mode: {mode!r}")
    if frame is None or frame.empty or "SessionType" not in frame.columns:
        return frame
    if mode == "combined":
        return frame
    if mode == "race":
        return frame[frame["SessionType"] == "Race"]
    return frame[frame["SessionType"] == "Sprint"]


def round_points(frame: pd.DataFrame, mode: str = "combined") -> pd.DataFrame:
    """Points per driver per round: index = driver, columns = short labels.

    Absent rounds are ``NaN`` (the driver has no row that weekend) while a
    driver who raced and scored nothing is ``0``; only the selected
    drivers are *not* reindexed away here, the full field is returned.
    """
    if frame is None or frame.empty or "Driver" not in frame.columns:
        return pd.DataFrame()
    labels = round_labels(frame)
    if not labels:
        return pd.DataFrame()
    work = _points_frame(frame.copy(), mode)
    work = work.assign(_pts=pd.to_numeric(work["Points"], errors="coerce")
                       .fillna(0))
    # Map each full track name to its short label, keeping duplicates apart
    # by round order (two rounds never share one short label in practice).
    track_order = [track for _rnd, track in ordered_rounds(frame)]
    short_of = {track: short_track(track) for track in track_order}
    work["_short"] = work["Track"].astype(str).map(short_of)
    pivot = work.pivot_table(index="Driver", columns="_short", values="_pts",
                             aggfunc="sum")
    pivot = pivot.reindex(columns=[lab for lab in labels if lab in pivot.columns])
    return pivot


def cumulative(frame: pd.DataFrame, mode: str = "combined",
               drivers=None) -> pd.DataFrame:
    """Running championship total per driver (same shape as round_points)."""
    points = round_points(frame, mode)
    if points.empty:
        return points
    cum = points.cumsum(axis=1, skipna=True)
    # A driver who enters late keeps NaN before their first round instead
    # of a misleading zero line: restore NaN where the driver was absent.
    cum = cum.where(points.notna(), other=pd.NA)
    cum = cum.astype(float)
    if drivers:
        keep = [d for d in drivers if d in cum.index]
        cum = cum.loc[keep]
    return cum


def championship_positions(cum: pd.DataFrame) -> pd.DataFrame:
    """Championship rank (1 = leader) after each round, from cumulative pts."""
    if cum is None or cum.empty:
        return pd.DataFrame()
    return cum.rank(axis=0, ascending=False, method="min").astype(float)


def leader_per_round(cum: pd.DataFrame) -> list:
    """Driver leading the championship after each round (``None`` if tied)."""
    if cum is None or cum.empty:
        return []
    leaders = []
    for column in cum.columns:
        values = cum[column].dropna()
        if values.empty:
            leaders.append(None)
            continue
        top = values.max()
        winners = values[values == top].index.tolist()
        leaders.append(winners[0] if len(winners) == 1 else None)
    return leaders


def count_lead_changes(cum: pd.DataFrame) -> int:
    """How many times the championship leader changed hands this season."""
    leaders = [leader for leader in leader_per_round(cum) if leader]
    if len(leaders) < 2:
        return 0
    return int(sum(1 for a, b in zip(leaders, leaders[1:]) if a != b))


def standings_total(frame: pd.DataFrame, mode: str = "combined"):
    """Total points per driver under ``mode``, best first (a Series)."""
    if frame is None or frame.empty or "Driver" not in frame.columns:
        return pd.Series(dtype=float)
    work = _points_frame(frame, mode)
    totals = work.groupby("Driver")["Points"].apply(
        lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()
    )
    return totals.sort_values(ascending=False)


def gap_vs_reference(cum: pd.DataFrame, reference: str):
    """Gap of every driver to ``reference`` per round (positive = ahead)."""
    if cum is None or cum.empty or reference not in cum.index:
        return pd.DataFrame()
    return cum.sub(cum.loc[reference], axis=1)


def consistency(frame: pd.DataFrame, drivers) -> pd.DataFrame:
    """Per-driver consistency table over combined points per Grand Prix.

    Wins, podiums and DNFs come from Race rows only (a sprint podium is
    not a Grand Prix podium); average, median and spread come from the
    combined (race + sprint) points of each weekend the driver entered.
    """
    rows = []
    if frame is None or frame.empty or "Driver" not in frame.columns:
        return pd.DataFrame()
    points = round_points(frame, "combined")
    race_rows = frame
    if "SessionType" in frame.columns:
        race_rows = frame[frame["SessionType"] == "Race"]
    for driver in drivers or []:
        if driver not in points.index:
            continue
        per_race = points.loc[driver].dropna()
        sub = race_rows[race_rows["Driver"] == driver]
        positions = (pd.to_numeric(sub["Position"], errors="coerce")
                     if "Position" in sub.columns else pd.Series(dtype=float))
        dnf = 0
        if "Finished" in sub.columns:
            dnf = int((~sub["Finished"].astype(bool)).sum())
        elif "Status" in sub.columns:
            status = sub["Status"].astype(str).str.strip().str.upper()
            dnf = int((~status.isin(["FINISHED", "CLASSIFIED", ""])).sum())
        rows.append({
            "Driver": driver,
            "Races": int(per_race.shape[0]),
            "Points": float(per_race.sum()),
            "AvgPts": float(per_race.mean()) if len(per_race) else None,
            "MedianPts": float(per_race.median()) if len(per_race) else None,
            "StdPts": float(per_race.std()) if len(per_race) > 1 else 0.0,
            "Wins": int((positions == 1).sum()),
            "Podiums": int(positions.between(1, 3).sum()),
            "DNFs": dnf,
        })
    return pd.DataFrame(rows)


def scenarios(frame: pd.DataFrame, remaining_races: int,
              remaining_sprints: int = 0, mode: str = "combined") -> pd.DataFrame:
    """Mathematical title scenarios: who can still overtake, and by how much.

    Mathematical analysis, not a prediction: it compares the points a
    driver still needs with the maximum points still on offer
    (``remaining_races`` Grand Prix at 26 points plus
    ``remaining_sprints`` sprint weekends at 8). A driver whose gap to
    the leader exceeds that maximum is mathematically eliminated.
    """
    totals = standings_total(frame, mode)
    if totals.empty:
        return pd.DataFrame()
    remaining_races = max(0, int(remaining_races or 0))
    remaining_sprints = max(0, int(remaining_sprints or 0))
    max_available = (remaining_races * MAX_POINTS_PER_GP
                     + remaining_sprints * MAX_POINTS_PER_SPRINT)
    leader_pts = float(totals.iloc[0])
    rows = []
    for rank, (driver, pts) in enumerate(totals.items(), start=1):
        gap = leader_pts - float(pts)
        needed = int(gap + 1) if gap > 0 else 0
        if rank == 1:
            status = "Leads the championship"
        elif needed > max_available:
            status = "Mathematically eliminated"
        elif gap == 0:
            status = "Level on points"
        else:
            status = "Can still overtake"
        rows.append({
            "Rank": rank,
            "Driver": driver,
            "Points": float(pts),
            "GapToLeader": float(gap),
            "PointsNeeded": needed,
            "MaxAvailable": max_available,
            "Status": status,
        })
    return pd.DataFrame(rows)


def _fmt_pts(value) -> str:
    return f"{float(value):.0f}"


def insights(frame: pd.DataFrame, drivers, cum_all: pd.DataFrame = None) -> list:
    """Automatically generated, data-derived observations for the page.

    Each insight is computed from the season frame (never hard-coded) and
    carries the figures behind it as ``evidence`` so a reader can trace
    every sentence back to the dataset.
    """
    out = []
    if frame is None or frame.empty:
        return out
    selected = [d for d in (drivers or [])]
    if cum_all is None:
        cum_all = cumulative(frame)
    totals = standings_total(frame)
    if totals.empty:
        return out
    leader = str(totals.index[0])
    leader_pts = float(totals.iloc[0])
    points = round_points(frame, "combined")
    labels = list(points.columns)
    if points.empty or not labels:
        return out

    # Gap between the top two after the latest round, and its change.
    final = cum_all.iloc[:, -1].dropna().sort_values(ascending=False)
    if len(final) >= 2:
        gap_now = float(final.iloc[0] - final.iloc[1])
        if cum_all.shape[1] >= 2:
            prev = cum_all.iloc[:, -2].dropna().sort_values(ascending=False)
            if len(prev) >= 2:
                change = gap_now - float(prev.iloc[0] - prev.iloc[1])
                if change != 0:
                    direction = "grown" if change > 0 else "shrunk"
                    out.append({
                        "title": "Gap movement",
                        "text": (
                            f"The gap between {final.index[0]} and "
                            f"{final.index[1]} has {direction} by "
                            f"{abs(change):.0f} points since the previous "
                            f"round, to {gap_now:.0f} points."
                        ),
                        "evidence": f"{gap_now:.0f} pts after {labels[-1]}",
                    })
        # Did the leader extend the lead over the last three rounds?
        if cum_all.shape[1] >= 4 and final.index[0] in cum_all.index:
            first_of_three = cum_all.loc[final.index[0]].iloc[-4]
            other = final.index[1]
            gap_then = float(first_of_three
                             - cum_all.loc[other].iloc[-4])
            if gap_now > gap_then:
                out.append({
                    "title": "Lead extension",
                    "text": (
                        f"{final.index[0]} has extended the championship "
                        f"lead over {other} by {gap_now - gap_then:.0f} "
                        f"points across the last three rounds."
                    ),
                    "evidence": f"{gap_then:.0f} to {gap_now:.0f} pts",
                })

    # Best scorer in the last five rounds among the selected drivers.
    window = labels[-5:]
    if selected and window:
        last_five = points.reindex(index=[d for d in selected
                                          if d in points.index],
                                   columns=window).sum(axis=1).dropna()
        if not last_five.empty:
            best = str(last_five.idxmax())
            out.append({
                "title": "Recent form",
                "text": (
                    f"{best} has scored more points than any other "
                    f"selected driver in the last {len(window)} rounds "
                    f"({last_five.max():.0f} points)."
                ),
                "evidence": f"last {len(window)} rounds",
            })

    # Most consistent selected scorer (lowest spread of per-race points).
    table = consistency(frame, selected)
    if not table.empty and table["StdPts"].notna().any():
        sound = table[table["Races"] >= 3]
        if sound.empty:
            sound = table
        row = sound.sort_values(["StdPts", "AvgPts"],
                                ascending=[True, False]).iloc[0]
        out.append({
            "title": "Most consistent",
            "text": (
                f"{row['Driver']} is the most consistent selected "
                f"scorer: {row['AvgPts']:.1f} points per race on "
                f"average with a spread of {row['StdPts']:.1f} points."
            ),
            "evidence": f"{int(row['Races'])} races",
        })

    # Largest single-round lead swing between the top two.
    if cum_all.shape[1] >= 2 and len(final) >= 2:
        top_two = [str(final.index[0]), str(final.index[1])]
        gaps = (cum_all.loc[top_two[0]] - cum_all.loc[top_two[1]]).dropna()
        swings = gaps.diff().abs().dropna()
        if not swings.empty and swings.max() > 0:
            round_label = swings.idxmax()
            out.append({
                "title": "Biggest swing",
                "text": (
                    f"The largest single-round swing in the gap between "
                    f"{top_two[0]} and {top_two[1]} was {swings.max():.0f} "
                    f"points at the {round_label} Grand Prix."
                ),
                "evidence": f"{round_label} GP",
            })

    # Lead changes across the season.
    changes = count_lead_changes(cum_all)
    if changes:
        out.append({
            "title": "Lead changes",
            "text": (
                f"The championship lead has changed hands {changes} "
                f"time{'s' if changes != 1 else ''} this season, with "
                f"{leader} leading on {leader_pts:.0f} points."
            ),
            "evidence": f"{changes} lead change(s)",
        })
    return out


def analytical_summary(frame: pd.DataFrame, drivers=None) -> str:
    """One-sentence season story, derived from the data (for the header)."""
    totals = standings_total(frame)
    if totals.empty:
        return "No championship data yet."
    leader = str(totals.index[0])
    leader_pts = float(totals.iloc[0])
    labels = round_labels(frame)
    rounds_done = len(labels)
    text = (f"{leader} leads the championship on {leader_pts:.0f} points "
            f"after {rounds_done} rounds")
    if len(totals) >= 2:
        text += (f", {float(totals.iloc[0] - totals.iloc[1]):.0f} points "
                 f"clear of {totals.index[1]}")
    text += "."
    if drivers:
        wanted = [d for d in drivers if d in totals.index]
        if wanted:
            best_selected = max(wanted, key=lambda d: totals[d])
            if best_selected != leader:
                text = (f"{leader} leads the championship on "
                        f"{leader_pts:.0f} points after {rounds_done} "
                        f"rounds; of the selected drivers, "
                        f"{best_selected} is best placed on "
                        f"{float(totals[best_selected]):.0f} points "
                        f"({float(leader_pts - totals[best_selected]):.0f} "
                        f"behind the leader).")
    return text
