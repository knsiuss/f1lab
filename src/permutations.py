# -*- coding: utf-8 -*-
"""What still has to happen for the title to be decided.

Pure arithmetic on the championship table - no data access, no Streamlit - so
a page can explain the title race in plain words and every number can be
tested.

Points model (2025/2026 rules), per driver per weekend:

* Grand Prix: 25 for the win, 18-15-12-10-8-6-4-2-1 behind it, plus 1 point for
  the fastest lap when the driver finishes in the top ten (we count it in the
  maximum: it is available to a contender).
* Sprint (when the weekend has one): 8 for the win, 7-6-5-4-3-2-1 behind it.

So one normal weekend is worth at most 26 points, a sprint weekend 34.

Everything here assumes no penalties and no disqualifications, and does not
model countback tie-breaks - a tie is reported as a tie, not as a win.
"""
from __future__ import annotations

from typing import Optional

RACE_WIN = 25
FASTEST_LAP = 1
SPRINT_WIN = 8

RACE_WEEKEND_MAX = RACE_WIN + FASTEST_LAP          # 26
SPRINT_WEEKEND_MAX = RACE_WEEKEND_MAX + SPRINT_WIN  # 34

ESTIMATED = 'estimated'


def weekend_max(has_sprint: bool = False) -> int:
    """Most points one driver can take from a single weekend."""
    return SPRINT_WEEKEND_MAX if has_sprint else RACE_WEEKEND_MAX


def remaining_points(races_left: int, sprints_left: int = 0) -> int:
    """Most points a single driver can still score this season."""
    races_left = max(0, int(races_left or 0))
    sprints_left = max(0, min(int(sprints_left or 0), races_left))
    return races_left * RACE_WEEKEND_MAX + sprints_left * SPRINT_WIN


def _as_pairs(standings) -> list:
    """Accept a dict, a pandas Series or a list of (name, points) pairs."""
    if standings is None:
        return []
    if hasattr(standings, 'items'):
        items = list(standings.items())
    else:
        items = list(standings)
    pairs = []
    for name, points in items:
        try:
            value = float(points)
        except (TypeError, ValueError):
            continue
        pairs.append((str(name), value))
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    return pairs


def clinch_round(gap: float, races_left: int) -> Optional[int]:
    """The round after which a gap can no longer be closed *before the finale*.

    ``gap`` is the leader's advantage over the best rival. Returns the 1-based
    remaining round at which the title is mathematically settled with at least
    one race still to run, or None when the fight goes to the last race.

    A normal weekend is assumed (26 points available); a sprint weekend gives
    the chasing rival eight more, so this is the optimistic-for-the-leader
    case and it is labelled estimated wherever it is shown.
    """
    races_left = max(0, int(races_left or 0))
    try:
        gap = float(gap)
    except (TypeError, ValueError):
        return None
    if races_left < 2:
        # Nothing left to clinch "early": the title goes to the final race.
        return None
    for round_index in range(1, races_left):
        points_after = (races_left - round_index) * RACE_WEEKEND_MAX
        if gap > points_after:
            return round_index
    return None


def title_clinch_checks(standings, races_left: int,
                        sprints_left: int = 0) -> list:
    """One row per driver: can they still win, and what do they need?

    Returns rows sorted by points (leader first). Each row carries the plain
    numbers plus a ``verdict`` string so a page does not have to re-word them.
    """
    pairs = _as_pairs(standings)
    if not pairs:
        return []

    races_left = max(0, int(races_left or 0))
    max_left = remaining_points(races_left, sprints_left)
    leader, leader_pts = pairs[0]

    rows = []
    for position, (name, points) in enumerate(pairs, start=1):
        gap = leader_pts - points
        can_win = points + max_left >= leader_pts
        # To finish ahead on points a contender has to outscore the leader by
        # one more than the current gap; a tie is decided by countback, which
        # this model does not know.
        needs = int(gap) + 1 if gap >= 0 else 0
        if position == 1:
            verdict = 'Leads the championship'
            needs = 0
        elif not can_win:
            verdict = 'Cannot win the title any more'
        elif needs <= max_left:
            verdict = f'Still possible - needs {needs} more points than {leader}'
        else:
            verdict = 'Cannot win the title any more'
        rows.append({
            'position': position,
            'driver': name,
            'points': int(points),
            'gap_to_leader': int(gap),
            'max_points_left': max_left,
            'can_win': bool(can_win) and needs <= max_left,
            'points_needed': needs,
            'verdict': verdict,
        })
    return rows


def explain_clinch(row, leader: str = '', races_left: int = 0) -> str:
    """One plain sentence about a single row of ``title_clinch_checks``."""
    if not row:
        return ''
    name = row.get('driver', '')
    gap = row.get('gap_to_leader', 0)
    need = row.get('points_needed', 0)
    if row.get('position') == 1:
        return (f'{name} leads by {gap} points; every rival has to outscore '
                f'{name} from here.')
    if not row.get('can_win', False):
        return (f'{name} is {gap} points behind with {races_left} races left - '
                f'not enough points remain to catch the leader.')
    leader_text = leader or 'the leader'
    return (f'{name} is {gap} points behind {leader_text} and must outscore '
            f'them by {need} over the {races_left} races left; a tie would be '
            f'settled by countback, which this model does not predict.')


def title_is_settled(standings, races_left: int, sprints_left: int = 0) -> bool:
    """True when the leader can no longer be caught by anybody."""
    rows = title_clinch_checks(standings, races_left, sprints_left)
    if len(rows) < 2:
        return bool(rows)
    return not any(row['can_win'] for row in rows[1:])
