# -*- coding: utf-8 -*-
"""
backtest.py
~~~~~~~~~~~
Walk-forward historical backtest of the prediction model vs a baseline.

The prediction model (src/prediction.py) is a ridge regression on lagged
features: grid position, team pace, driver form, positions gained and finish
reliability. A backtest that trained on the same season it scored would
flatter it, so this module walks the season race by race: each race is scored
using a model trained only on races *before* it (no look-ahead / no leakage),
and the result is compared with the race's actual finishing order. The
``Engine`` column says which engine produced each forecast -- ``ridge`` or the
documented ``heuristic`` fallback used when the history is too short to fit.

The baseline is deliberately naive: "the finishing order equals the
qualifying grid order." F1's grid is strongly predictive, so a model that
cannot beat "predict the grid" on average is not adding value.

Data Honesty contract:
    * every number is labelled ``estimated`` and derived from real results,
    * each backtested race reports how many drivers it was scored on,
    * races with too few drivers, or without enough prior history, are
      skipped and counted, not silently averaged in,
    * the summary reports sample sizes and both model and baseline, so a
      misleading "we're better" claim is visible at a glance,
    * the summary says plainly whether the model beat the baseline
      (``beats_baseline`` / ``verdict``) instead of leaving the caller to
      compare two numbers and hope they notice.

The season frame holds race *and* sprint rows. Every helper here says which
session it uses: the race helpers default to the **race** rows so a sprint
result can never override the race result for the same weekend.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Ensure src/ is importable no matter how this module is reached
# (as `backtest` from a page, or as `src.backtest` from the test suite).
sys.path.insert(0, str(Path(__file__).parent))

from prediction import fantasy_points, predict_race, rows_before_race  # noqa: E402

ESTIMATED = "estimated"

# Minimum drivers a race must be scored on before its metrics are shown.
MIN_DRIVERS = 2
# Default history required before a race can be predicted (walk-forward).
DEFAULT_MIN_TRAIN_RACES = 2

# Metric -> direction of "better". Lower-is-better metrics (errors) win when
# the model value is below the baseline; higher-is-better metrics win when above.
METRICS: Dict[str, str] = {
    "exact_match_rate": "higher",
    "mae": "lower",
    "spearman": "higher",
    "podium_overlap": "higher",
    "points_mae": "lower",
}


# ---------------------------------------------------------------------------
# Race helpers
# ---------------------------------------------------------------------------

def _session_rows(race_df: pd.DataFrame,
                  session_type: Optional[str] = 'Race') -> pd.DataFrame:
    """Rows of the requested session type.

    The season frame carries race and sprint rows for the same weekend. The
    default is the **race** rows: a sprint result must never override the
    race result of the same round. Pass ``session_type=None`` to use every
    row, or another value (e.g. ``"Sprint"``) to pick a session explicitly.
    """
    if race_df is None or race_df.empty:
        return pd.DataFrame()
    if session_type is not None and 'SessionType' in race_df.columns:
        return race_df[race_df['SessionType'] == session_type]
    return race_df


def race_grid(race_df: pd.DataFrame,
              session_type: Optional[str] = 'Race') -> Dict[str, int]:
    """Map {driver: starting grid position} for one race.

    Args:
        race_df: rows of the race (may contain several session types).
        session_type: which session to read ('Race' by default; None uses
            every row).
    """
    rows = _session_rows(race_df, session_type)
    if rows.empty or 'Driver' not in rows.columns:
        return {}
    if 'Starting Grid' in rows.columns:
        grid = pd.to_numeric(rows['Starting Grid'], errors='coerce')
        out = {d: int(g) for d, g in zip(rows['Driver'], grid)
               if pd.notna(g)}
    else:
        out = {}
    return {d: g for d, g in out.items()}


def race_actual(race_df: pd.DataFrame,
                session_type: Optional[str] = 'Race') -> Dict[str, int]:
    """Map {driver: finishing position} for one race.

    Args:
        race_df: rows of the race (may contain several session types).
        session_type: which session to read ('Race' by default; None uses
            every row).
    """
    rows = _session_rows(race_df, session_type)
    if (rows.empty or 'Driver' not in rows.columns
            or 'Position' not in rows.columns):
        return {}
    pos = pd.to_numeric(rows['Position'], errors='coerce')
    return {d: int(p) for d, p in zip(rows['Driver'], pos) if pd.notna(p)}


def grid_baseline(grid: Dict[str, int]) -> List[str]:
    """Baseline predicted finish: drivers in grid order (P1 on pole).

    Predicting "the result is the grid" is the honest null model for F1.
    """
    return [d for d, _ in sorted(grid.items(), key=lambda kv: kv[1])]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _common(pred: Dict[str, int], actual: Dict[str, int]) -> Optional[Dict]:
    """Drivers present in both, as {driver: (pred, actual)}; None if too few."""
    common = {d: (pred[d], actual[d]) for d in pred if d in actual}
    if len(common) < MIN_DRIVERS:
        return None
    return common


def exact_match_rate(pred: Dict[str, int], actual: Dict[str, int]) -> Optional[float]:
    """Fraction of drivers whose finishing position is predicted exactly."""
    common = _common(pred, actual)
    if common is None:
        return None
    hits = sum(1 for p, a in common.values() if p == a)
    return round(hits / len(common), 3)


def mean_abs_error(pred: Dict[str, int], actual: Dict[str, int]) -> Optional[float]:
    """Mean absolute error in predicted finishing position."""
    common = _common(pred, actual)
    if common is None:
        return None
    return round(sum(abs(p - a) for p, a in common.values()) / len(common), 3)


def spearman_rank(pred: Dict[str, int], actual: Dict[str, int]) -> Optional[float]:
    """Spearman rank correlation between predicted and actual order."""
    common = _common(pred, actual)
    if common is None or len(common) < 2:
        return None
    ps = pd.Series({d: v[0] for d, v in common.items()})
    ac = pd.Series({d: v[1] for d, v in common.items()})
    r = ps.corr(ac, method='spearman')
    return None if pd.isna(r) else round(float(r), 3)


def podium_overlap(pred: Dict[str, int], actual: Dict[str, int], n: int = 3) -> Optional[float]:
    """Fraction of the predicted top-n that actually finished top-n.

    The denominator is the number of drivers actually considered
    (``min(n, len(common))``): a race scored on two drivers -- the minimum
    this module accepts -- must not report a perfect overlap as 2/3.
    """
    common = _common(pred, actual)
    if common is None:
        return None
    considered = min(int(n), len(common))
    if considered <= 0:
        return None
    pred_top = set(d for d, v in sorted(common.items(), key=lambda kv: kv[1][0])[:considered])
    act_top = set(d for d, v in sorted(common.items(), key=lambda kv: kv[1][1])[:considered])
    return round(len(pred_top & act_top) / considered, 3)


def points_mae(pred: Dict[str, int], actual: Dict[str, int]) -> Optional[float]:
    """Mean absolute error in fantasy points earned (uses official table)."""
    common = _common(pred, actual)
    if common is None:
        return None
    err = sum(abs(fantasy_points(p) - fantasy_points(a)) for p, a in common.values())
    return round(err / len(common), 3)


def _metrics(pred: Dict[str, int], actual: Dict[str, int]) -> Dict[str, Optional[float]]:
    return {m: fn(pred, actual) for m, fn in {
        "exact_match_rate": exact_match_rate,
        "mae": mean_abs_error,
        "spearman": spearman_rank,
        "podium_overlap": podium_overlap,
        "points_mae": points_mae,
    }.items()}


def evaluate_race(
    pred_df: pd.DataFrame, actual: Dict[str, int]
) -> Optional[Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]], int]]:
    """Score one race: model vs grid baseline, sharing the same actuals.

    Args:
        pred_df: output of ``predict_race`` (has Driver, Predicted, Grid).
        actual: {driver: finishing position}.

    Returns:
        ``(model_metrics, baseline_metrics, n_drivers)`` or ``None`` when the
        race has too few drivers to score honestly.
    """
    if pred_df is None or pred_df.empty or not actual:
        return None
    model_pred = {
        d: int(p) for d, p in zip(pred_df['Driver'], pred_df['Predicted'])
    }
    grid = {d: int(g) for d, g in zip(pred_df['Driver'], pred_df['Grid'])}
    baseline_pred = {d: i + 1 for i, d in enumerate(grid_baseline(grid))}

    common = {d for d in model_pred if d in actual}
    if len(common) < MIN_DRIVERS:
        return None
    return _metrics(model_pred, actual), _metrics(baseline_pred, actual), len(common)


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def _race_order(season_df: pd.DataFrame) -> List[str]:
    """Chronological race list.

    With a ``Round`` column (the new season frame) the order is the calendar
    order of the rounds; without one it falls back to the first-appearance
    order of ``Track``.
    """
    if 'Round' in season_df.columns:
        rounds = pd.to_numeric(season_df['Round'], errors='coerce')
        ordered = (season_df.assign(_r=rounds)
                   .dropna(subset=['_r'])
                   .sort_values('_r', kind='stable'))
        order = list(dict.fromkeys(ordered['Track'].astype(str).tolist()))
        if order:
            return order
    return list(dict.fromkeys(season_df['Track'].astype(str).tolist()))


def walk_forward_backtest(
    season_df: pd.DataFrame,
    min_train_races: int = DEFAULT_MIN_TRAIN_RACES,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Score every race with a model trained only on the races before it.

    Each race from ``min_train_races`` onward is predicted with a model fit on
    the preceding rounds only (no leakage), compared against the grid baseline
    on the same actuals. Races without enough history, without a grid, or with
    too few drivers are skipped and counted.

    The season is walked in calendar order (the ``Round`` column when the
    frame has it), and the training set is cut on the same round boundary as
    the target race, so a race is never predicted from data that includes
    itself.

    Returns:
        ``(per_race_df, summary)``. ``per_race_df`` has one row per backtested
        race with model and baseline metrics; ``summary`` reports means, how
        often the model beat the baseline per metric, the sample sizes and
        whether the model beat the baseline overall (``beats_baseline`` plus
        a human-readable ``verdict``).
    """
    empty_summary = {"n_races": 0, "n_races_skipped": 0, "estimated": ESTIMATED,
                     "beats_baseline": None,
                     "verdict": "Not enough scored races to compare the model "
                                "with the grid baseline."}
    if season_df is None or season_df.empty or 'Track' not in season_df.columns:
        return pd.DataFrame(), empty_summary

    order = _race_order(season_df)
    rows: List[Dict[str, object]] = []
    skipped = 0

    for i in range(min_train_races, len(order)):
        target_track = order[i]
        # No look-ahead: the training set is cut on the round boundary.
        train = rows_before_race(season_df, target_track)
        target = season_df[season_df['Track'] == target_track]

        grid = race_grid(target)
        if not grid:
            skipped += 1
            continue
        pred = predict_race(train, grid)
        actual = race_actual(target)
        ev = evaluate_race(pred, actual)
        if ev is None:
            skipped += 1
            continue

        model_m, baseline_m, n = ev
        row: Dict[str, object] = {
            "Race": target_track, "N": n, "Estimated": ESTIMATED,
            "Engine": pred.attrs.get("engine", ""),
        }
        for m in METRICS:
            row[f"{m}_model"] = model_m.get(m)
            row[f"{m}_baseline"] = baseline_m.get(m)
        rows.append(row)

    per_race = pd.DataFrame(rows)
    summary = _summarize(per_race, skipped)
    return per_race, summary


def summarise(per_race: pd.DataFrame,
              skipped: int = 0) -> Dict[str, object]:
    """Public summary step: the same aggregation ``walk_forward_backtest`` uses.

    Kept as a named function so a caller holding its own ``per_race`` frame
    (a page, a notebook, a test) can get the honest comparison -- means, the
    per-metric win counts, the engine split and the plain verdict -- without
    re-running the whole walk-forward.
    """
    return _summarize(per_race, skipped)


def _summarize(per_race: pd.DataFrame, skipped: int) -> Dict[str, object]:
    """Aggregate per-race metrics into an honest comparison summary."""
    if per_race is None or per_race.empty:
        return {"n_races": 0, "n_races_skipped": skipped, "estimated": ESTIMATED,
                "beats_baseline": None,
                "verdict": "Not enough scored races to compare the model with "
                           "the grid baseline."}

    summary: Dict[str, object] = {
        "n_races": len(per_race),
        "n_races_skipped": skipped,
        "estimated": ESTIMATED,
    }
    if "Engine" in per_race.columns:
        engines = per_race["Engine"].fillna("").astype(str)
        summary["n_engine_ridge"] = int((engines == "ridge").sum())
        summary["n_engine_fallback"] = int((engines != "ridge").sum())
    for m, direction in METRICS.items():
        col_model = f"{m}_model"
        col_base = f"{m}_baseline"
        summary[f"{m}_model"] = _mean(per_race[col_model])
        summary[f"{m}_baseline"] = _mean(per_race[col_base])
        summary[f"{m}_model_beats_baseline"] = _win_count(
            per_race[col_model], per_race[col_base], direction
        )
    summary.update(_verdict(summary))
    return summary


def _verdict(summary: Dict[str, object]) -> Dict[str, object]:
    """Did the model beat the grid baseline? Say so in plain words.

    The honest comparison is the position error (MAE, lower is better): a
    model that cannot predict the finishing order better than "the result is
    the grid" is not adding value, and the caller should be told rather than
    left to compare two numbers.
    """
    model = summary.get("mae_model")
    baseline = summary.get("mae_baseline")
    n_races = summary.get("n_races")
    if model is None or baseline is None:
        return {"beats_baseline": None,
                "verdict": "Not enough scored races to compare the model with "
                           "the grid baseline."}
    if model < baseline:
        beats = True
        verdict = (f"The model beats the grid baseline over {n_races} scored "
                   f"races (position MAE {model} vs {baseline}).")
    elif model == baseline:
        beats = False
        verdict = (f"The model only matches the grid baseline over {n_races} "
                   f"scored races (position MAE {model} vs {baseline}): it adds "
                   f"nothing over 'the result is the grid'.")
    else:
        beats = False
        verdict = (f"The model does NOT beat the grid baseline over {n_races} "
                   f"scored races (position MAE {model} vs {baseline}): the "
                   f"grid remains the stronger predictor, so the forecast is a "
                   f"ranking guide, not a prediction.")
    return {"beats_baseline": beats, "verdict": verdict}


def _mean(col: pd.Series) -> Optional[float]:
    vals = col.dropna()
    return None if vals.empty else round(float(vals.mean()), 3)


def _win_count(model_col: pd.Series, base_col: pd.Series,
               direction: str) -> int:
    """Races where the model is strictly better than the baseline."""
    both = pd.concat([model_col, base_col], axis=1).dropna()
    if both.empty:
        return 0
    better = both.iloc[:, 0] < both.iloc[:, 1] if direction == "lower" \
        else both.iloc[:, 0] > both.iloc[:, 1]
    return int(better.sum())


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

