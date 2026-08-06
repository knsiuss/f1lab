# -*- coding: utf-8 -*-
"""
prediction.py
~~~~~~~~~~~~~
Race prediction and fantasy scoring model.

Pure logic (no Streamlit) so it is unit-testable. The model is a transparent,
documented heuristic rather than an opaque ML black-box:
  - grid advantage  (from qualifying)    40%
  - team pace       (season finishing)   35%
  - driver form     (recent results)     25%

A driver's composite score drives a softmax over the field to yield win
probabilities and a predicted finishing order. Fantasy helpers use the official
F1 2025 points table and a budget-constrained greedy lineup builder.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Official F1 2025 race points for positions 1..10
RACE_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]

# Attribution weights for the composite performance score (sum to 1.0)
W_GRID = 0.40
W_TEAM = 0.35
W_FORM = 0.25

DEFAULT_BUDGET = 50  # fantasy budget in $M
DEFAULT_DRIVERS = 5  # drivers in a fantasy team
PRICE_MIN = 5
PRICE_MAX = 30


def compute_team_pace(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Per-team pace score in [0,1]: better average finishing = closer to 1."""
    if (
        df is None or df.empty
        or 'Team' not in df.columns or 'Position' not in df.columns
    ):
        return {}
    avg = df.groupby('Team')['Position'].apply(
        lambda s: pd.to_numeric(s, errors='coerce').mean()
    )
    mx, mn = avg.max(), avg.min()
    if pd.isna(mx) or mx == mn:
        return {t: 0.5 for t in avg.index}
    return {t: round((mx - v) / (mx - mn), 3) for t, v in avg.items()}


def compute_driver_form(df: Optional[pd.DataFrame], window: int = 3) -> Dict[str, float]:
    """Per-driver form score in [0,1] from recent rolling finishing position."""
    if (
        df is None or df.empty
        or 'Driver' not in df.columns or 'Position' not in df.columns
    ):
        return {}

    order = (
        list(dict.fromkeys(df['Track'].astype(str).tolist()))
        if 'Track' in df.columns else []
    )
    ord_map = {t: i for i, t in enumerate(order)}

    rolling = {}
    for d in df['Driver'].dropna().unique():
        dd = df[df['Driver'] == d].copy()
        dd['_pos'] = pd.to_numeric(dd['Position'], errors='coerce')
        if order:
            dd['_ord'] = dd['Track'].map(ord_map)
            dd = dd.dropna(subset=['_pos', '_ord']).sort_values('_ord')
        else:
            dd = dd.dropna(subset=['_pos'])
        if dd.empty:
            continue
        rolling[d] = dd['_pos'].rolling(window, min_periods=1).mean().iloc[-1]

    if not rolling:
        return {}
    mx, mn = max(rolling.values()), min(rolling.values())
    if mx == mn:
        return {d: 0.5 for d in rolling}
    return {d: round((mx - v) / (mx - mn), 3) for d, v in rolling.items()}


def compute_driver_prices(df: Optional[pd.DataFrame]) -> Dict[str, int]:
    """Fantasy prices ($M) derived from season points, scaled $5M–$30M."""
    if df is None or df.empty or 'Driver' not in df.columns:
        return {}
    if 'Points' in df.columns:
        pts = df.groupby('Driver')['Points'].sum()
    else:
        pts = pd.Series(1.0, index=df['Driver'].unique())
    if pts.max() <= 0 or pd.isna(pts.max()):
        return {d: PRICE_MIN for d in pts.index}
    return {
        d: int(max(PRICE_MIN, min(PRICE_MAX, PRICE_MIN + (v / pts.max()) * (PRICE_MAX - PRICE_MIN))))
        for d, v in pts.items()
    }


def predict_race(df: Optional[pd.DataFrame], grid: Dict[str, int]) -> pd.DataFrame:
    """Predict finishing order and win probabilities from a qualifying grid.

    Args:
        df: season race results (for team pace and driver form).
        grid: mapping {driver_code: grid_position}, 1 = pole.

    Returns a DataFrame sorted by predicted finish with a ``WinProb`` percent.
    """
    if not grid:
        return pd.DataFrame()

    team_pace = compute_team_pace(df)
    form = compute_driver_form(df)
    team_map = (
        df.groupby('Driver')['Team'].first().to_dict()
        if df is not None and 'Team' in df.columns else {}
    )

    rows = []
    for d, g in grid.items():
        grid_pos = max(1, int(g))
        grid_adv = max(0.0, 1.0 - (grid_pos - 1) / 20.0)
        team = team_map.get(d, '')
        pace = team_pace.get(team, 0.5)
        fm = form.get(d, 0.5)
        score = W_GRID * grid_adv + W_TEAM * pace + W_FORM * fm
        rows.append({
            'Driver': d, 'Grid': grid_pos, 'Team': team,
            'TeamPace': round(pace, 3), 'Form': round(fm, 3),
            'Score': round(score, 4),
        })

    pdf = pd.DataFrame(rows)
    if pdf.empty:
        return pdf

    shifted = np.exp(pdf['Score'] - pdf['Score'].max())
    pdf['WinProb'] = (shifted / shifted.sum() * 100).round(1)
    pdf = pdf.sort_values('Score', ascending=False).reset_index(drop=True)
    pdf['Predicted'] = pdf.index + 1

    reorder = ['Predicted', 'Driver', 'Team', 'Grid', 'TeamPace', 'Form', 'WinProb', 'Score']
    return pdf[[c for c in reorder if c in pdf.columns]]


def fantasy_points(position: int) -> int:
    """F1 2025 fantasy points for a finishing position (P1 = position 1)."""
    pos = int(position)
    if 1 <= pos <= len(RACE_POINTS):
        return RACE_POINTS[pos - 1]
    return 0


def add_predicted_points(pdf: pd.DataFrame) -> pd.DataFrame:
    """Add a ``PredictedPts`` column using predicted finishing position."""
    out = pdf.copy()
    out['PredictedPts'] = out['Predicted'].apply(fantasy_points)
    return out


def add_price(pdf: pd.DataFrame, prices: Dict[str, int]) -> pd.DataFrame:
    """Add a ``Price`` ($M) column from a driver-price mapping."""
    out = pdf.copy()
    out['Price'] = out['Driver'].map(prices).fillna(PRICE_MIN).astype(int)
    return out


def optimal_lineup(
    entries: pd.DataFrame,
    budget: int = DEFAULT_BUDGET,
    count: int = DEFAULT_DRIVERS,
    price_key: str = 'Price',
    points_key: str = 'PredictedPts',
    driver_key: str = 'Driver',
) -> List[str]:
    """Approximate best fantasy team via greedy value (points per $M).

    Repeatedly adds the affordable driver with the highest points/price ratio,
    stopping at a full lineup or an exhausted budget. Simple and explainable.
    Returns the list of chosen driver names in selection order.
    """
    if entries is None or entries.empty:
        return []

    pool = entries.copy()
    pool['_ratio'] = pool[points_key] / pool[price_key].astype(float)
    selected: List[str] = []
    spent = 0

    for _ in range(count):
        affordable = pool[(pool[price_key].astype(int) <= budget - spent)]
        if affordable.empty:
            break
        affordable = affordable.sort_values(
            ['_ratio', points_key], ascending=[False, False]
        )
        pick = affordable.iloc[0]
        selected.append(pick[driver_key])
        spent += int(pick[price_key])
        pool = pool.drop(index=pick.name)

    return selected


def forecast_vs_actual(
    predicted: pd.DataFrame, actual: Dict[str, int]
) -> pd.DataFrame:
    """Join predicted finish with the actual finish for accuracy comparison."""
    out = predicted.copy()
    out['Actual'] = out['Driver'].map(actual)
    out['PredictedPts'] = out['Predicted'].apply(fantasy_points)
    out['ActualPts'] = out['Actual'].apply(lambda p: fantasy_points(p) if pd.notna(p) else 0)
    return out