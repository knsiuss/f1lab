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

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

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


def _expand_stints(total_laps: int, scenario: Scenario) -> List[tuple]:
    """Turn a Scenario into an ordered list of (compound, start_lap, end_lap)."""
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
    """
    deg = _deg_for(compound, deg_rates)
    total = 0.0
    for lap in range(start_lap, end_lap):
        tyre_age = lap - start_lap
        total += base_lap_time + deg * tyre_age + fuel_gain * lap
    return total


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
        every assumption used, so the projection is transparent.
    """
    stints = _expand_stints(total_laps, scenario)
    if not stints:
        return {"level": "insufficient", "estimated": ESTIMATED,
                "assumptions": _assumptions(base_lap_time, fuel_gain, pit_loss,
                                            weather_penalty, sc_delay,
                                            traffic_penalty, rival_response)}

    stop_loss = max(0, len(stints) - 1) * pit_loss
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
        "n_stops": len(stints) - 1,
        "compounds": [s[0] for s in stints],
        "stops_laps": scenario.stop_laps,
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
    """
    earlier = Scenario(name="earlier", compounds=[compound_old, compound_new],
                       stop_laps=[total_laps // 2 - earlier_by])
    later = Scenario(name="later", compounds=[compound_old, compound_new],
                     stop_laps=[total_laps // 2])
    t_early = simulate_scenario(earlier, total_laps, base_lap_time=base_lap_time,
                                deg_rates=deg_rates, fuel_gain=fuel_gain,
                                pit_loss=pit_loss, traffic_penalty=traffic_penalty,
                                rival_response=rival_response)["total_time"]
    t_late = simulate_scenario(later, total_laps, base_lap_time=base_lap_time,
                               deg_rates=deg_rates, fuel_gain=fuel_gain,
                               pit_loss=pit_loss,
                               traffic_penalty=traffic_penalty)["total_time"]
    return {
        "delta_sec": round(t_early - t_late, 3),
        "earlier_lap": total_laps // 2 - earlier_by,
        "later_lap": total_laps // 2,
        "estimated": ESTIMATED,
        "assumptions": [
            "identical tyre-age history before the stop",
            f"traffic penalty: {traffic_penalty}s/lap (applied to both)",
            f"rival response to the earlier stop: {rival_response}s (earlier only)",
            "pit loss identical for both",
        ],
    }
