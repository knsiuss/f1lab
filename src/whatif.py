# -*- coding: utf-8 -*-
"""
whatif.py
~~~~~~~~~
What-if race strategy simulator.

Turns the old static "1-stop vs 2-stop" question into a scenario comparison:
compound choices, pit-stop laps, pit windows, undercut/overcut timing, Safety
Car / VSC delay and weather all feed a projected race time per scenario.

Every projected time is an estimate ("estimated") built on explicit
assumptions and paired with a confidence label. We never present a projection
as fact; the caller decides how much weight to give it.

Assumptions (all overridable, all surfaced in the result):
    * tyre degradation per compound (from config.MODEL_CONFIG)
    * constant fuel gain per lap (from config.MODEL_CONFIG)
    * average pit loss in seconds (from config.MODEL_CONFIG)
    * a safety car / VSC event is a one-off additive delay (no re-stack logic)
    * weather is an additive per-lap penalty while it lasts
    * traffic is an additive per-lap penalty while it lasts
    * a rival response is a one-off additive delay; the interaction itself
      (mirror stops, track-position swaps) is not modelled

Stop-plan rules (single source of truth, shared with :mod:`pitwindow`):

    * the number of stops is ``len(scenario.stop_laps)`` -- a scenario that
      declares two stops pays for two stops, always;
    * a stop lap must be inside the race: ``1 <= stop_lap <= total_laps - 1``
      (at least one lap on the old tyre and one on the new tyre). A stop on
      lap 0 (or beyond the flag) is **rejected**, not clamped: it would
      describe a strategy that never actually runs, and clamping it silently
      turns a declared stop into a no-stop run that looks fastest because it
      is charged no pit loss;
    * stop laps must strictly increase (the plan is chronological);
    * a scenario needs one compound per stop plus the starting tyre; more
      stops than compounds is rejected instead of dropping a stop.

An invalid plan returns the ``level: insufficient`` shape with a ``reason``
so the caller can say what was wrong instead of scoring a nonsense time.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Ensure src/ is importable no matter how this module is reached
# (as `whatif` from a page, or as `src.whatif` from the test suite).
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_CONFIG

ESTIMATED = "estimated"

# Qualitative risk label by number of pit stops. More stops = more exposure to
# an ill-timed Safety Car, a botched stop or traffic; it is a heuristic, not a
# probability.
_RISK_BY_STOPS = {0: "very low", 1: "low", 2: "medium", 3: "high"}


@dataclass
class Scenario:
    """A candidate strategy for one driver.

    ``compounds[0]`` is the starting tyre; ``stop_laps[i]`` is the lap of the
    pit stop that switches to ``compounds[i + 1]``. A one-stop SOFT->HARD at
    lap 18 is Scenario(compounds=["SOFT", "HARD"], stop_laps=[18]).

    The number of stops is ``len(stop_laps)``: every declared stop is paid
    for. See the stop-plan rules in the module docstring.
    """

    name: str
    compounds: List[str]
    stop_laps: List[int] = field(default_factory=list)
    confidence: str = "medium"


def _deg_for(compound: str, deg_rates: Dict[str, float]) -> float:
    """Degradation rate for a compound, matching model.py's fallback rules."""
    c = str(compound).upper()
    for key, rate in deg_rates.items():
        if key != "DEFAULT" and key in c:
            return rate
    return deg_rates["DEFAULT"]


def stop_lap_problem(stop_laps: List[int], compounds: List[str],
                     total_laps: int) -> Optional[str]:
    """Reason why a stop plan cannot be simulated, or ``None`` when it is fine.

    The rules (one implementation, shared with :mod:`pitwindow`):

    * a stop must be inside the race -- ``1 <= stop_lap <= total_laps - 1`` --
      so there is at least one lap on the old tyre and one on the new tyre.
      A stop on lap 0 describes a strategy that never runs and would be
      charged no pit loss, which is why it is rejected instead of clamped;
    * stop laps must strictly increase (the plan is chronological);
    * a scenario needs one compound per stop plus the starting tyre.

    Args:
        stop_laps: the declared stop laps.
        compounds: the tyre plan.
        total_laps: race distance in laps.

    Returns:
        A human-readable reason string, or ``None`` when the plan is valid.
    """
    stops = list(stop_laps or [])
    tyres = list(compounds or [])
    if not tyres:
        return "no compound in the tyre plan"
    if len(stops) > len(tyres) - 1:
        return (f"{len(stops)} stops declared but only {len(tyres)} compounds "
                f"in the tyre plan")
    if not stops:
        return None
    try:
        laps = [int(s) for s in stops]
    except (TypeError, ValueError):
        return "stop lap is not a number"
    for lap in laps:
        if lap < 1 or lap > int(total_laps) - 1:
            return (f"stop lap {lap} is outside the race "
                    f"(1..{int(total_laps) - 1} for {int(total_laps)} laps)")
    for previous, current in zip(laps, laps[1:]):
        if current <= previous:
            return (f"stop laps must increase ({previous} then {current})")
    return None


def _expand_stints(total_laps: int, scenario: Scenario) -> List[tuple]:
    """Turn a Scenario into an ordered list of (compound, start_lap, end_lap).

    Only called for a plan that passed :func:`stop_lap_problem`, so every
    declared stop produces exactly one stint boundary.
    """
    stints: List[tuple] = []
    start = 0
    n_stops = len(scenario.stop_laps)
    for i, compound in enumerate(scenario.compounds):
        if i < n_stops:
            end = max(start, min(int(scenario.stop_laps[i]), total_laps))
        else:
            end = total_laps
        if end > start:
            stints.append((compound, start, end))
        start = end
    # Guarantee the final stint always reaches the chequered flag.
    if stints and stints[-1][2] < total_laps:
        stints.append((scenario.compounds[-1], stints[-1][2], total_laps))
    return stints


def stint_time(
    compound: str,
    start_lap: int,
    end_lap: int,
    base_lap_time: float,
    deg_rates: Dict[str, float],
    fuel_gain: float,
) -> float:
    """
    Projected time for one stint, laps [start_lap, end_lap).

    lap_time = base + deg * tyre_age + fuel_gain * lap; tyres are assumed fresh
    at ``start_lap``. Returns seconds (estimated).

    The sum is written in closed form (same value as the per-lap loop, but
    fast enough for the pit-window grid search to evaluate thousands of stop
    plans per driver).
    """
    deg = _deg_for(compound, deg_rates)
    n = max(0, int(end_lap) - int(start_lap))
    if n <= 0:
        return 0.0
    # sum(tyre_age) over the stint = 0 + 1 + ... + (n - 1)
    ages = n * (n - 1) / 2.0
    # sum(lap) over the stint = start_lap + ... + (end_lap - 1)
    laps = (int(start_lap) + int(end_lap) - 1) * n / 2.0
    return n * base_lap_time + deg * ages + fuel_gain * laps


def simulate_scenario(
    scenario: Scenario,
    total_laps: int,
    base_lap_time: float = MODEL_CONFIG["default_base_lap_time"],
    deg_rates: Dict[str, float] = MODEL_CONFIG["degradation_rates"],
    fuel_gain: float = MODEL_CONFIG["fuel_gain_per_lap"],
    pit_loss: float = MODEL_CONFIG["pit_loss_sec"],
    weather_penalty: float = 0.0,
    sc_delay: float = 0.0,
    traffic_penalty: float = 0.0,
    rival_response: float = 0.0,
) -> Dict[str, object]:
    """
    Project total race time for a Scenario.

    Args:
        scenario: the strategy to evaluate.
        total_laps: race length in laps.
        base_lap_time: clean-air lap time on a fresh tyre (estimated).
        deg_rates: degradation per compound (seconds per lap of tyre age).
        fuel_gain: per-lap time gain from fuel burn (negative = faster).
        pit_loss: average seconds lost per pit stop.
        weather_penalty: additive seconds per lap while a weather window lasts.
        sc_delay: one-off additive seconds from a Safety Car / VSC event.
        traffic_penalty: additive seconds per lap lost to traffic (e.g. being
            held behind a slower car after an overcut).
        rival_response: one-off additive seconds from a rival reacting to this
            strategy (e.g. matching the stop or defending track position).

    Returns:
        dict with ``total_time`` (est, seconds), ``n_stops``, ``stints`` and
        every assumption used, so the projection is transparent. An invalid
        stop plan returns ``level: insufficient`` with a ``reason``.
    """
    problem = stop_lap_problem(scenario.stop_laps, scenario.compounds,
                               total_laps)
    if problem is not None:
        return {"level": "insufficient", "estimated": ESTIMATED,
                "reason": problem, "n_stops": len(scenario.stop_laps),
                "assumptions": _assumptions(base_lap_time, fuel_gain, pit_loss,
                                            weather_penalty, sc_delay,
                                            traffic_penalty, rival_response)}

    stints = _expand_stints(total_laps, scenario)
    if not stints:
        return {"level": "insufficient", "estimated": ESTIMATED,
                "reason": "the tyre plan covers no laps",
                "n_stops": len(scenario.stop_laps),
                "assumptions": _assumptions(base_lap_time, fuel_gain, pit_loss,
                                            weather_penalty, sc_delay,
                                            traffic_penalty, rival_response)}

    # Every declared stop is paid for, whether or not the tyre plan can
    # absorb it: the stop count comes from the scenario, not from the
    # stints that happened to come out of it.
    n_stops = len(scenario.stop_laps)
    stop_loss = n_stops * pit_loss
    weather_total = weather_penalty * total_laps
    traffic_total = traffic_penalty * total_laps

    total = (stop_loss + weather_total + traffic_total
             + sc_delay + rival_response)
    for compound, start_lap, end_lap in stints:
        total += stint_time(compound, start_lap, end_lap, base_lap_time,
                            deg_rates, fuel_gain)

    return {
        "name": scenario.name,
        "total_time": round(total, 3),
        "n_stops": n_stops,
        "compounds": [s[0] for s in stints],
        "stops_laps": list(scenario.stop_laps),
        "estimated": ESTIMATED,
        "confidence": scenario.confidence,
        "assumptions": _assumptions(base_lap_time, fuel_gain, pit_loss,
                                    weather_penalty, sc_delay,
                                    traffic_penalty, rival_response),
    }


def _assumptions(base_lap_time, fuel_gain, pit_loss, weather_penalty, sc_delay,
                 traffic_penalty=0.0, rival_response=0.0) -> List[str]:
    return [
        f"base lap time (fresh tyre): {base_lap_time}s (estimated)",
        f"fuel gain per lap: {fuel_gain}s",
        f"pit loss per stop: {pit_loss}s",
        f"weather penalty: {weather_penalty}s/lap",
        f"safety car / VSC delay: {sc_delay}s",
        f"traffic penalty: {traffic_penalty}s/lap",
        f"rival response delay: {rival_response}s",
    ]


def scenario_risk(n_stops: int) -> str:
    """Heuristic risk label for a number of pit stops."""
    return _RISK_BY_STOPS.get(int(n_stops), "high")


def compare_scenarios(
    scenarios: List[Scenario],
    total_laps: int,
    **sim_kwargs,
) -> pd.DataFrame:
    """
    Simulate every Scenario and return a ranked comparison table.

    Ranks scenarios by projected total time; columns expose the expected
    outcome (projected time + gap to best), the risk heuristic, the confidence
    the caller attached to the inputs, and the assumptions used. The fastest
    scenario is not a recommendation: it is only the fastest under the stated
    assumptions.

    Returns:
        DataFrame sorted best-first with columns Name, TotalTime (est),
        GapToBest (est), Stops, Compounds, Risk, Confidence, Estimated.
    """
    rows = []
    results = []
    for sc in scenarios:
        res = simulate_scenario(sc, total_laps, **sim_kwargs)
        results.append(res)
        rows.append({
            "Name": sc.name,
            "TotalTime": res.get("total_time"),
            "Stops": res.get("n_stops", 0),
            "Compounds": " > ".join(res.get("compounds", [])),
            "Risk": scenario_risk(res.get("n_stops", 0)),
            "Confidence": res.get("confidence", "medium"),
            "Estimated": res.get("estimated", ESTIMATED),
        })

    df = pd.DataFrame(rows)
    if df.empty or df["TotalTime"].isna().all():
        return df

    df["TotalTime"] = pd.to_numeric(df["TotalTime"], errors="coerce")
    best = df["TotalTime"].min()
    df["GapToBest"] = (df["TotalTime"] - best).round(3)
    df = df.sort_values("TotalTime").reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def undercut_delta(
    total_laps: int,
    compound_new: str,
    compound_old: str,
    base_lap_time: float = MODEL_CONFIG["default_base_lap_time"],
    deg_rates: Dict[str, float] = MODEL_CONFIG["degradation_rates"],
    fuel_gain: float = MODEL_CONFIG["fuel_gain_per_lap"],
    pit_loss: float = MODEL_CONFIG["pit_loss_sec"],
    earlier_by: int = 1,
    traffic_penalty: float = 0.0,
    rival_response: float = 0.0,
) -> Dict[str, float]:
    """
    Projected time effect of stopping ``earlier_by`` laps sooner (undercut).

    Compares two one-stop strategies that differ only in the pit-stop lap.
    A negative delta means stopping earlier projects to a faster race under
    the stated assumptions.

    ``traffic_penalty`` applies to both variants (held-up traffic is symmetric
    by assumption); ``rival_response`` is a one-off penalty charged to the
    earlier variant only, modelling a rival reacting to the undercut by
    covering or defending. Both are surfaced in the returned assumptions.

    Returns:
        dict: ``delta_sec`` (earlier minus later, negative = earlier faster),
        ``earlier_lap``, ``later_lap``, ``estimated``, ``assumptions``.
        A race too short to hold a valid stop returns ``level: insufficient``
        with ``delta_sec`` set to ``None`` and a ``reason``.
    """
    earlier_lap = total_laps // 2 - earlier_by
    later_lap = total_laps // 2
    assumptions = [
        "identical tyre-age history before the stop",
        f"traffic penalty: {traffic_penalty}s/lap (applied to both)",
        f"rival response to the earlier stop: {rival_response}s (earlier only)",
        "pit loss identical for both",
    ]

    for lap in (earlier_lap, later_lap):
        problem = stop_lap_problem([lap], [compound_old, compound_new],
                                   total_laps)
        if problem is not None:
            return {"delta_sec": None, "earlier_lap": earlier_lap,
                    "later_lap": later_lap, "level": "insufficient",
                    "reason": problem, "estimated": ESTIMATED,
                    "assumptions": assumptions}

    earlier = Scenario(name="earlier", compounds=[compound_old, compound_new],
                       stop_laps=[earlier_lap])
    later = Scenario(name="later", compounds=[compound_old, compound_new],
                     stop_laps=[later_lap])
    t_early = simulate_scenario(earlier, total_laps, base_lap_time=base_lap_time,
                                deg_rates=deg_rates, fuel_gain=fuel_gain,
                                pit_loss=pit_loss, traffic_penalty=traffic_penalty,
                                rival_response=rival_response).get("total_time")
    t_late = simulate_scenario(later, total_laps, base_lap_time=base_lap_time,
                               deg_rates=deg_rates, fuel_gain=fuel_gain,
                               pit_loss=pit_loss,
                               traffic_penalty=traffic_penalty).get("total_time")
    if t_early is None or t_late is None:
        return {"delta_sec": None, "earlier_lap": earlier_lap,
                "later_lap": later_lap, "level": "insufficient",
                "reason": "the undercut could not be simulated",
                "estimated": ESTIMATED, "assumptions": assumptions}
    return {
        "delta_sec": round(t_early - t_late, 3),
        "earlier_lap": earlier_lap,
        "later_lap": later_lap,
        "estimated": ESTIMATED,
        "assumptions": assumptions,
    }
