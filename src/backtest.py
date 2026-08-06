# -*- coding: utf-8 -*-
"""
backtest.py
~~~~~~~~~~~
Walk-forward historical backtest of the prediction model vs a baseline.

The prediction model (src/prediction.py) is a transparent heuristic: grid
advantage, team pace and driver form weighted into a composite score. A
backtest that trained on the same season it scored would flatter it, so this
module walks the season race by race: each race is scored using a model
trained only on races *before* it (no look-ahead / no leakage), and the
result is compared with the race's actual finishing order.

The baseline is deliberately naive: "the finishing order equals the
qualifying grid order." F1's grid is strongly predictive, so a model that
cannot beat "predict the grid" on average is not adding value.

Data Honesty contract:
    * every number is labelled ``estimated`` and derived from real results,
    * each backtested race reports how many drivers it was scored on,
    * races with too few drivers, or without enough prior history, are
      skipped and counted, not silently averaged in,
    * the summary reports sample sizes and both model and baseline, so a
      misleading "we're better" claim is visible at a glance.

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

from prediction import fantasy_points, predict_race

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

def race_grid(race_df: pd.DataFrame) -> Dict[str, int]:
    """Map {driver: starting grid position} for one race."""
    if race_df is None or race_df.empty or 'Driver' not in race_df.columns:
        return {}
    if 'Starting Grid' in race_df.columns:
        grid = pd.to_numeric(race_df['Starting Grid'], errors='coerce')
        out = {d: int(g) for d, g in zip(race_df['Driver'], grid)
               if pd.notna(g)}
    else:
        out = {}
    return {d: g for d, g in out.items()}


def race_actual(race_df: pd.DataFrame) -> Dict[str, int]:
    """Map {driver: finishing position} for one race."""
    if (race_df is None or race_df.empty or 'Driver' not in race_df.columns
            or 'Position' not in race_df.columns):
        return {}
    pos = pd.to_numeric(race_df['Position'], errors='coerce')
    return {d: int(p) for d, p in zip(race_df['Driver'], pos) if pd.notna(p)}


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
    """Fraction of the predicted top-n that actually finished top-n."""
    common = _common(pred, actual)
    if common is None:
        return None
    pred_top = set(d for d, v in sorted(common.items(), key=lambda kv: kv[1][0])[:n])
    act_top = set(d for d, v in sorted(common.items(), key=lambda kv: kv[1][1])[:n])
    return round(len(pred_top & act_top) / n, 3)


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
    """Chronological race list = first-appearance order of the Track column."""
    return list(dict.fromkeys(season_df['Track'].astype(str).tolist()))


def walk_forward_backtest(
    season_df: pd.DataFrame,
    min_train_races: int = DEFAULT_MIN_TRAIN_RACES,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Score every race with a model trained only on the races before it.

    Each race from ``min_train_races`` onward is predicted with a model fit on
    the preceding races only (no leakage), compared against the grid baseline
    on the same actuals. Races without enough history or too few drivers are
    skipped and counted.

    Returns:
        ``(per_race_df, summary)``. ``per_race_df`` has one row per backtested
        race with model and baseline metrics; ``summary`` reports means, how
        often the model beat the baseline per metric, and sample sizes.
    """
    empty_summary = {"n_races": 0, "n_races_skipped": 0, "estimated": ESTIMATED}
    if season_df is None or season_df.empty or 'Track' not in season_df.columns:
        return pd.DataFrame(), empty_summary

    order = _race_order(season_df)
    rows: List[Dict[str, object]] = []
    skipped = 0

    for i in range(min_train_races, len(order)):
        train = season_df[season_df['Track'].isin(order[:i])]
        target = season_df[season_df['Track'] == order[i]]

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
            "Race": order[i], "N": n, "Estimated": ESTIMATED,
        }
        for m in METRICS:
            row[f"{m}_model"] = model_m.get(m)
            row[f"{m}_baseline"] = baseline_m.get(m)
        rows.append(row)

    per_race = pd.DataFrame(rows)
    summary = _summarize(per_race, skipped)
    return per_race, summary


def _summarize(per_race: pd.DataFrame, skipped: int) -> Dict[str, object]:
    """Aggregate per-race metrics into an honest comparison summary."""
    if per_race is None or per_race.empty:
        return {"n_races": 0, "n_races_skipped": skipped, "estimated": ESTIMATED}

    summary: Dict[str, object] = {
        "n_races": len(per_race),
        "n_races_skipped": skipped,
        "estimated": ESTIMATED,
    }
    for m, direction in METRICS.items():
        col_model = f"{m}_model"
        col_base = f"{m}_baseline"
        summary[f"{m}_model"] = _mean(per_race[col_model])
        summary[f"{m}_baseline"] = _mean(per_race[col_base])
        summary[f"{m}_model_beats_baseline"] = _win_count(
            per_race[col_model], per_race[col_base], direction
        )
    return summary


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

def explain_prediction(row: pd.Series, field: int) -> List[str]:
    """
    Human-readable breakdown of why the model placed one driver where it did.

    The three weighted components (grid advantage, team pace, driver form) are
    shown with their contribution, so a prediction is auditable rather than a
    black-box number. ``field`` is the grid size used to normalise the grid
    advantage (the same value ``predict_race`` used).

    Returns:
        List of lines, e.g. ``["VER: predicted P1 (Score 0.880)", ...]``.
    """
    grid_pos = row['Grid']
    field = max(1, int(field))
    grid_adv = max(0.0, 1.0 - (grid_pos - 1) / field)
    from prediction import W_FORM, W_GRID, W_TEAM
    return [
        f"{row['Driver']}: predicted P{int(row['Predicted'])} "
        f"(Score {float(row['Score']):.3f})",
        f"  grid P{grid_pos}: {grid_adv:.3f} x {W_GRID:.0%}",
        f"  team pace: {float(row['TeamPace']):.3f} x {W_TEAM:.0%}",
        f"  driver form: {float(row['Form']):.3f} x {W_FORM:.0%}",
    ]
