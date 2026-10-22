# -*- coding: utf-8 -*-
"""
prediction.py
~~~~~~~~~~~~~
Race prediction and fantasy scoring model.

The finishing order is predicted by a **ridge regression written with numpy
only** -- no scikit-learn, no lightgbm, no new dependency.

Features (each one is knowable *before* the race it predicts):

    * ``Grid``      the starting grid position from qualifying -- the
                    strongest single signal in Formula 1;
    * ``TeamPace``  the team's mean race finish in the rounds **before** the
                    race, rescaled to [0, 1] (1 = best team);
    * ``Form``      the driver's mean finish over their last three races;
    * ``Gain``      the driver's mean grid-to-finish change (places gained);
    * ``Dnf``       the share of earlier races the driver was not classified
                    in, read from the ``Finished`` flag (or a classified
                    position when the frame carries no flag).

The target is the finishing position. Every feature is z-scored on the
training window, an unpenalised intercept is added, and the coefficients
solve ``(X'X + lambda I)^-1 X'y`` with a small ``RIDGE_LAMBDA``. A missing
feature value is imputed with that column's **training median**, and the
number of imputed cells is published in ``attrs['imputed']``; a row is never
silently dropped.

Walk-forward, no look-ahead
---------------------------
:func:`predict_race` fits on the rounds of the frame it is handed. Every
caller cuts that frame with :func:`rows_before_race`, so a race is scored only
from the rounds before it. Inside the fit, each training race is itself given
features built from the rounds before *it*, so the model cannot learn a
driver's finishing position from the result it is trying to predict.

When there is too little history to fit honestly (fewer than
``MIN_TRAIN_ROUNDS`` rounds, or fewer than two training rows per coefficient)
the model falls back to the documented hand-weighted heuristic
(``W_GRID`` / ``W_TEAM`` / ``W_FORM``) and the ``Engine`` column says so. The
fallback is never silent.

Measured weakness
-----------------
The forecast is scored by :mod:`backtest` against the naive "the result is
the grid" baseline. The numbers are *measured*, not written by hand:
:func:`model_caveat` renders whatever the walk-forward run found, so the text
under a forecast can never claim more than the code has shown.

Features that are NOT available are reported by :func:`feature_report` instead
of being invented: the season frame carries no qualifying lap times, so there
is no qualifying *gap*; it holds one race per circuit, so there is no earlier
visit to a circuit to learn from; and it carries no weather, so a wet race
cannot be flagged.

Frame expectations
------------------
The season frame carries both race and sprint rows. Every helper here states
explicitly which rows it uses:

    * race results only (team pace, driver form, the sample size behind a
      forecast) -- a sprint result is not a race result and must never stand
      in for one;
    * calendar order via the ``Round`` column when the frame has it, so a
      team change or a form window follows the season, not the row layout.

Data Honesty contract
---------------------
Every number this module produces is an estimate and is labelled as such:
the forecast frame carries an ``Estimated`` marker plus a ``Level`` and
``Confidence`` derived from the number of race results behind it
(``pace.data_quality``: good >= 15, moderate >= 8, low >= 2, insufficient
< 2). The module also publishes its own weakness -- the walk-forward backtest
in :mod:`backtest` -- so the caller shows that instead of discovering it.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Ensure src/ is importable no matter how this module is reached
# (as `prediction` from a page, or as `src.prediction` from the test suite).
sys.path.insert(0, str(Path(__file__).parent))

from pace import data_quality  # noqa: E402

# Championship points for a Grand Prix finishing position 1..10. The table
# has been unchanged since 2010, so it holds for every season the app serves.
# Sprint points are never hardcoded: the season frame carries the points the
# API reported for the sprint session.
RACE_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]

# Attribution weights of the DOCUMENTED FALLBACK heuristic (sum to 1.0). They
# are only used when there is too little history for the ridge fit; the ridge
# model does not use them.
W_GRID = 0.40
W_TEAM = 0.35
W_FORM = 0.25

DEFAULT_BUDGET = 50  # fantasy budget in $M
DEFAULT_DRIVERS = 5  # drivers in a fantasy team
PRICE_MIN = 5
PRICE_MAX = 30

# Banner applied to every derived figure (project convention).
ESTIMATED = "estimated"

# -- Ridge model ------------------------------------------------------------

# The regression features, in the order they enter the design matrix. Every
# one of them is derived from rows strictly before the race being predicted.
FEATURES: Sequence[str] = ('Grid', 'TeamPace', 'Form', 'Gain', 'Dnf')

# Small ridge penalty. The features are z-scored, so lambda = 1.0 shrinks a
# coefficient to about half of its unpenalised value -- enough to keep a
# twenty-row sample from producing wild coefficients, small enough that the
# grid signal is not thrown away.
RIDGE_LAMBDA = 1.0

# A fit below this many completed rounds, or below this many rows per
# coefficient, is noise: the model falls back to the documented heuristic and
# the forecast says which engine produced it.
MIN_TRAIN_ROUNDS = 2
MIN_ROWS_PER_COEFFICIENT = 2

# Display scale of the win probability. The softmax is a monotone transform of
# the model's own rating, NOT a calibrated probability; the page says so.
PROB_SCALE = 8.0

ENGINE_RIDGE = 'ridge'
ENGINE_FALLBACK = 'heuristic'

# What a forecast is made of, and what it honestly cannot see. The report is
# published to the page so the limitation is visible next to the numbers.
_FEATURE_SPECS: List[Dict[str, object]] = [
    {'feature': 'Starting grid', 'source': 'the grid position from qualifying',
     'used': True,
     'note': 'the strongest signal: the car that starts ahead usually stays '
             'ahead'},
    {'feature': 'Team pace', 'source': "the team's mean race finish in the "
                                       'rounds before the race',
     'used': True,
     'note': 'computed on earlier rounds only, never on the race being '
             'predicted'},
    {'feature': 'Driver form', 'source': "the driver's last three race "
                                         'finishes',
     'used': True,
     'note': 'race results only; a sprint result never stands in for one'},
    {'feature': 'Positions gained', 'source': 'mean change from starting grid '
                                              'to finish in earlier rounds',
     'used': True,
     'note': 'separates a driver who moves forward from one who loses places'},
    {'feature': 'Finish reliability', 'source': 'share of earlier races the '
                                                'driver was classified in',
     'used': True,
     'note': 'read from the Finished flag, never guessed'},
    {'feature': 'Qualifying gap', 'source': 'the lap-time gap in qualifying',
     'used': False,
     'note': 'the season frame carries no qualifying lap times, only the '
             'starting order, so a gap is not available and is not invented'},
    {'feature': 'Circuit history', 'source': "the driver's earlier results at "
                                             'this circuit',
     'used': False,
     'note': 'the frame holds one race per circuit, so there is no earlier '
             'visit to that circuit to learn from'},
    {'feature': 'Wet or dry', 'source': 'the weather flag of the session',
     'used': False,
     'note': 'the season frame carries no weather, so a wet race cannot be '
             'flagged and is not guessed'},
]


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def _race_rows(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Race-session rows of a frame.

    The season frame holds race *and* sprint rows; a sprint result is not a
    race result, so every race-based metric uses this filter. Frames without
    a ``SessionType`` column (older CSV exports, unit-test fixtures) are
    returned untouched.
    """
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df
    if 'SessionType' in df.columns:
        return df[df['SessionType'] == 'Race']
    return df


def _order_keys(df: pd.DataFrame) -> pd.Series:
    """Calendar-order key per row: the ``Round`` when the frame has one.

    Rows without a round keep their relative order but sort after the rows
    that have one -- a missing round must never reorder the season.
    """
    if 'Round' in df.columns:
        return pd.to_numeric(df['Round'], errors='coerce')
    return pd.Series(np.arange(len(df)), index=df.index, dtype=float)


def _sorted_by_round(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of ``df`` in calendar order (round when known, else frame order)."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    out['_order'] = _order_keys(out)
    out = out.sort_values('_order', kind='stable')
    return out.drop(columns=['_order'])


def _history_races(df: Optional[pd.DataFrame]) -> int:
    """Number of completed races behind a prediction (rounds, not rows)."""
    races = _race_rows(df)
    if races.empty or 'Driver' not in races.columns:
        return 0
    if 'Round' in races.columns:
        rounds = pd.to_numeric(races['Round'], errors='coerce').dropna().nunique()
        if rounds:
            return int(rounds)
    if 'Track' in races.columns:
        return int(races['Track'].nunique())
    return 0


def _driver_teams(df: Optional[pd.DataFrame]) -> Dict[str, str]:
    """Each driver's team in their **most recent** race.

    A mid-season team change must not be attributed to the old team, so the
    mapping uses the last round a driver appears in (calendar order via
    ``Round`` when available). Teams that are unknown (empty string) do not
    overwrite a known earlier team.
    """
    if df is None or df.empty or 'Driver' not in df.columns or 'Team' not in df.columns:
        return {}
    d = df.copy()
    d['Driver'] = d['Driver'].astype(str)
    ordered = _sorted_by_round(d)
    teams = ordered['Team'].replace('', np.nan)
    if teams.isna().all():
        return {}
    return teams.groupby(ordered['Driver']).last().dropna().to_dict()


def rows_before_race(df: pd.DataFrame, track: str,
                     session_type: Optional[str] = None) -> pd.DataFrame:
    """Rows of the season strictly before the given track's round.

    The cut is on the **calendar round**: the target race's own rows and its
    sprint share one round, so both are excluded together with everything
    after it. That is what makes a pre-race briefing free of look-ahead.

    Args:
        df: season frame. Uses ``Track`` and, when present, ``Round`` and
            ``SessionType``.
        track: the ``Track`` value to cut at (the FastF1 event name, e.g.
            "Australian Grand Prix").
        session_type: optional filter, e.g. ``"Race"``. ``None`` (default)
            keeps every session type of the earlier rounds.

    Returns:
        The rows of the earlier rounds in calendar order. An unknown track
        returns an empty frame -- never the whole frame.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return (df.iloc[0:0].copy() if isinstance(df, pd.DataFrame)
                else pd.DataFrame())
    if 'Track' not in df.columns or not track:
        return df.iloc[0:0].copy()

    matches = df[df['Track'] == track]
    if matches.empty:
        return df.iloc[0:0].copy()

    earlier = None
    if 'Round' in df.columns:
        rounds = pd.to_numeric(matches['Round'], errors='coerce').dropna()
        if not rounds.empty:
            target_round = float(rounds.min())
            keep = pd.to_numeric(df['Round'], errors='coerce') < target_round
            earlier = df[keep.fillna(False)].copy()
    if earlier is None:
        # No usable round: fall back to the first-appearance order of Track.
        order = list(dict.fromkeys(df['Track'].astype(str).tolist()))
        key = str(track)
        if key not in order:
            return df.iloc[0:0].copy()
        earlier = df[df['Track'].isin(order[:order.index(key)])].copy()

    if session_type is not None and 'SessionType' in earlier.columns:
        earlier = earlier[earlier['SessionType'] == session_type].copy()
    return _sorted_by_round(earlier)


# ---------------------------------------------------------------------------
# Model inputs
# ---------------------------------------------------------------------------

def compute_team_pace(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Per-team pace score in [0,1]: better average finishing = closer to 1.

    Race sessions only: a sprint result is not a race result.
    """
    if (
        df is None or df.empty
        or 'Team' not in df.columns or 'Position' not in df.columns
    ):
        return {}
    races = _race_rows(df)
    if races.empty:
        return {}
    avg = races.groupby('Team')['Position'].apply(
        lambda s: pd.to_numeric(s, errors='coerce').mean()
    )
    # A team whose positions are all non-numeric averages to NaN. Drop it so
    # it falls back to the neutral 0.5 in predict_race instead of poisoning
    # the composite score and the softmax with NaN.
    avg = avg.dropna()
    if avg.empty:
        return {}
    mx, mn = avg.max(), avg.min()
    if mx == mn:
        return {t: 0.5 for t in avg.index}
    return {t: round((mx - v) / (mx - mn), 3) for t, v in avg.items()}


def compute_driver_form(df: Optional[pd.DataFrame], window: int = 3) -> Dict[str, float]:
    """Per-driver form score in [0,1] from recent race finishing positions.

    The rolling window covers **races only** (the frame holds sprint rows as
    well) and follows the calendar: rounds in order when the frame has a
    ``Round`` column, otherwise the first-appearance order of ``Track``.
    """
    if (
        df is None or df.empty
        or 'Driver' not in df.columns or 'Position' not in df.columns
    ):
        return {}

    races = _race_rows(df)
    if races.empty:
        return {}
    ordered = _sorted_by_round(races)

    rolling = {}
    for d in ordered['Driver'].dropna().unique():
        dd = ordered[ordered['Driver'] == d].copy()
        dd['_pos'] = pd.to_numeric(dd['Position'], errors='coerce')
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


def compute_driver_gain(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Per-driver mean places gained between the grid and the flag.

    ``Starting Grid`` minus ``Position``: positive means the driver finishes
    ahead of where they started. DNF rows have no position and are left out of
    the mean -- a retirement is not a place lost on track.
    """
    if (
        df is None or df.empty or 'Driver' not in df.columns
        or 'Starting Grid' not in df.columns or 'Position' not in df.columns
    ):
        return {}
    races = _race_rows(df)
    if races.empty:
        return {}
    ordered = _sorted_by_round(races)
    grid = pd.to_numeric(ordered['Starting Grid'], errors='coerce')
    pos = pd.to_numeric(ordered['Position'], errors='coerce')
    gained = (grid - pos).dropna()
    if gained.empty:
        return {}
    means = gained.groupby(ordered['Driver']).mean()
    return {d: round(float(v), 3) for d, v in means.items()}


def compute_dnf_rate(df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Per-driver share of earlier races they were not classified in.

    The ``Finished`` flag written by the loader is authoritative; a frame
    without it falls back to "a classified position exists", which is the only
    evidence such a frame carries.
    """
    if df is None or df.empty or 'Driver' not in df.columns:
        return {}
    races = _race_rows(df)
    if races.empty:
        return {}
    ordered = _sorted_by_round(races)
    if 'Finished' in ordered.columns:
        finished = pd.to_numeric(ordered['Finished'], errors='coerce')
    elif 'Position' in ordered.columns:
        finished = (pd.to_numeric(ordered['Position'], errors='coerce')
                    .notna().astype(float))
    else:
        return {}
    rates = (1.0 - finished.groupby(ordered['Driver']).mean()).dropna()
    if rates.empty:
        return {}
    return {d: round(float(v), 3) for d, v in rates.items()}


def _feature_rows(history: Optional[pd.DataFrame],
                  target: pd.DataFrame) -> pd.DataFrame:
    """One feature row per target row, derived from ``history`` only.

    Args:
        history: rows strictly before the race (may be empty).
        target: the race's own rows -- only their ``Driver``, ``Team`` and
            ``Starting Grid`` are read, so nothing about the result leaks in.

    Returns:
        A frame indexed like ``target`` with the ``Driver``, ``Team`` and one
        column per name in :data:`FEATURES`. A feature that cannot be derived
        stays ``NaN``; it is imputed with the training median inside the fit,
        not filled here, so the caller can count the missing cells.
    """
    if target is None or len(target) == 0:
        return pd.DataFrame()
    out = pd.DataFrame(index=target.index)
    out['Driver'] = (target['Driver'].astype(str) if 'Driver' in target.columns
                     else '')
    if 'Team' in target.columns:
        out['Team'] = target['Team'].fillna('').astype(str)
    else:
        out['Team'] = ''
    if 'Starting Grid' in target.columns:
        out['Grid'] = pd.to_numeric(target['Starting Grid'], errors='coerce')
    elif 'Grid' in target.columns:
        out['Grid'] = pd.to_numeric(target['Grid'], errors='coerce')
    else:
        out['Grid'] = np.nan

    pace = compute_team_pace(history)
    form = compute_driver_form(history)
    gain = compute_driver_gain(history)
    dnf = compute_dnf_rate(history)
    out['TeamPace'] = out['Team'].map(pace)
    out['Form'] = out['Driver'].map(form)
    out['Gain'] = out['Driver'].map(gain)
    out['Dnf'] = out['Driver'].map(dnf)
    return out


def _round_frames(df: Optional[pd.DataFrame]) -> List[pd.DataFrame]:
    """The race of each completed round, in calendar order.

    One frame per round, so the walk-forward can give every training race the
    features of the rounds *before* it. Rounds are identified by ``Track``
    (one race per circuit in a season); when the frame has no ``Track`` the
    ``Round`` column is used, and a frame with neither is one anonymous block.
    """
    races = _race_rows(df)
    if races.empty or 'Driver' not in races.columns:
        return []
    ordered = _sorted_by_round(races)
    if 'Track' in ordered.columns:
        keys = ordered['Track'].astype(str)
    elif 'Round' in ordered.columns:
        keys = ordered['Round']
    else:
        return [ordered]
    return [group for _, group in ordered.groupby(keys, sort=False)]


def _training_table(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Walk-forward training rows: one row per driver-race, no look-ahead.

    The first round of the frame has no history, so it contributes no row.
    Every later round contributes its drivers, with the features computed from
    the rounds before that round and the finishing position as the target.
    """
    frames = _round_frames(df)
    parts: List[pd.DataFrame] = []
    prefix: Optional[pd.DataFrame] = None
    for frame in frames:
        if prefix is not None and not prefix.empty:
            features = _feature_rows(prefix, frame)
            features['_y'] = (pd.to_numeric(frame['Position'], errors='coerce')
                              if 'Position' in frame.columns else np.nan)
            if 'Round' in frame.columns:
                features['_round'] = frame['Round']
            parts.append(features)
        prefix = frame if prefix is None else pd.concat([prefix, frame])
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Ridge regression (numpy only)
# ---------------------------------------------------------------------------

def _impute(matrix: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Fill non-finite cells of ``matrix`` with the given column medians."""
    filled = np.array(matrix, dtype=float, copy=True)
    if filled.ndim == 1:
        filled = filled.reshape(1, -1)
    for column in range(filled.shape[1]):
        bad = ~np.isfinite(filled[:, column])
        if bad.any():
            filled[bad, column] = medians[column]
    return filled


def ridge_fit(X, y, lam: float = RIDGE_LAMBDA) -> Optional[Dict[str, object]]:
    """Fit ``y = intercept + X @ beta`` with a ridge penalty.

    The columns of ``X`` are z-scored on the training window (zero-variance
    columns are left unscaled), the intercept is not penalised, and the
    coefficients solve ``(X'X + lambda I)^-1 X'y``. Non-finite cells are
    imputed with the column's training median and counted.

    Args:
        X: (n_samples, n_features) array-like.
        y: (n_samples,) targets. Rows with a missing target are dropped.
        lam: ridge penalty (``0`` is ordinary least squares).

    Returns:
        A model dict (``intercept``, ``beta``, ``mean``, ``std``, ``medians``,
        ``n``, ``n_imputed``) or ``None`` when nothing can be fitted honestly
        (no rows, mismatched shapes, a singular system).
    """
    features = np.asarray(X, dtype=float)
    targets = np.asarray(y, dtype=float).ravel()
    if features.ndim != 2 or features.shape[0] == 0:
        return None
    if features.shape[0] != targets.shape[0]:
        return None

    usable = np.isfinite(targets)
    features, targets = features[usable], targets[usable]
    if features.shape[0] == 0:
        return None

    medians = np.nanmedian(features, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    n_imputed = int((~np.isfinite(features)).sum())
    filled = _impute(features, medians)

    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    design = np.hstack([np.ones((filled.shape[0], 1)),
                        (filled - mean) / std])

    penalty = float(lam) * np.eye(design.shape[1])
    penalty[0, 0] = 0.0  # the intercept is not shrunk
    try:
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    except np.linalg.LinAlgError:
        return None

    return {
        'intercept': float(beta[0]),
        'beta': beta[1:],
        'mean': mean,
        'std': std,
        'medians': medians,
        'n': int(features.shape[0]),
        'n_imputed': n_imputed,
    }


def ridge_predict(model: Optional[Dict[str, object]], X) -> np.ndarray:
    """Predicted target for ``X`` under a :func:`ridge_fit` model.

    Returns an empty array when the model is ``None``, so a caller that could
    not fit degrades to its fallback instead of crashing.
    """
    if not model:
        return np.array([], dtype=float)
    features = np.asarray(X, dtype=float)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    filled = _impute(features, np.asarray(model['medians'], dtype=float))
    design = (filled - np.asarray(model['mean'], dtype=float)) \
        / np.asarray(model['std'], dtype=float)
    return float(model['intercept']) + design @ np.asarray(model['beta'], dtype=float)


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

def _heuristic_strength(features: pd.DataFrame) -> np.ndarray:
    """The documented fallback score: grid 40%, team pace 35%, form 25%.

    Only used when there is too little history for the ridge fit. Higher is
    better. A missing team-pace or form value degrades to the neutral 0.5.
    """
    grid = pd.to_numeric(features['Grid'], errors='coerce')
    field = max(1, len(features))
    grid_adv = (1.0 - (grid - 1.0) / field).clip(lower=0.0).fillna(0.5)
    pace = pd.to_numeric(features['TeamPace'], errors='coerce').fillna(0.5)
    form = pd.to_numeric(features['Form'], errors='coerce').fillna(0.5)
    blended = W_GRID * grid_adv + W_TEAM * pace + W_FORM * form
    return blended.to_numpy(dtype=float)


def _normalise_strength(values: np.ndarray) -> np.ndarray:
    """Rescale a strength vector to [0, 1] where 1 is the strongest entry.

    A flat vector (every driver identical) degrades to 0.5 for all, never to a
    made-up ordering.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.full(array.shape, 0.5)
    low, high = float(finite.min()), float(finite.max())
    if not np.isfinite(high) or high == low:
        return np.full(array.shape, 0.5)
    out = (array - low) / (high - low)
    return np.where(np.isfinite(out), out, 0.5)


def _win_probabilities(scores: np.ndarray) -> np.ndarray:
    """Softmax of the model's own rating, as percentages.

    A monotone transform of the rating, **not** a calibrated probability: the
    page reads it as an ordering of the field.
    """
    if np.asarray(scores).size == 0:
        return np.array([], dtype=float)
    logits = np.asarray(scores, dtype=float) * PROB_SCALE
    shifted = np.exp(logits - float(np.max(logits)))
    total = float(shifted.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(np.asarray(scores).shape, 100.0 / max(1, len(scores)))
    return np.round(shifted / total * 100.0, 1)


def feature_report() -> List[Dict[str, object]]:
    """What the model uses, and what it honestly cannot see.

    Returns one dict per feature with ``feature``, ``source``, ``used`` and
    ``note``. A feature that is not available in the season frame is listed
    with ``used`` False and the reason, so the page can show the limitation
    instead of hiding it.
    """
    return [dict(spec) for spec in _FEATURE_SPECS]


def model_caveat(summary: Optional[Dict[str, object]] = None) -> str:
    """The model's own caveat, with the measured walk-forward numbers in it.

    Args:
        summary: output of ``backtest.walk_forward_backtest`` (or any mapping
            with ``mae_model``, ``mae_baseline``, ``n_races`` and
            ``mae_model_beats_baseline``). ``None`` returns the description
            without a measurement -- no number is invented.

    The text is generated from whatever the backtest measured, so it cannot
    claim more than the code has shown.
    """
    base = (
        "Ridge regression (numpy only) on the starting grid, the team's pace "
        "in the rounds before the race, the driver's last three finishes, "
        "positions gained and finish reliability. It is a ranking estimate, "
        "not a calibrated forecast, and its win probabilities are a display "
        "transform rather than a real chance."
    )
    if not summary:
        return base + (
            " Its walk-forward accuracy against the 'result = grid' baseline "
            "is measured by backtest.walk_forward_backtest, which reports the "
            "average error and the win/loss count race by race."
        )

    def _number(key: str) -> Optional[float]:
        value = summary.get(key)
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    model_error = _number('mae_model')
    baseline_error = _number('mae_baseline')
    races = _number('n_races')
    if model_error is None or baseline_error is None or not races:
        return base + (
            " Its walk-forward accuracy cannot be quoted from this run: there "
            "were not enough scored races to compare it with the baseline."
        )

    wins = _number('mae_model_beats_baseline')
    win_text = (f". It was closer than the baseline in {int(wins)} of "
                f"{int(races)} races" if wins is not None else '')
    scored = (f" Over {int(races)} races, each scored using only the rounds "
              f"before it, the average error in finishing positions is "
              f"{model_error:.3f} for the model against {baseline_error:.3f} "
              f"for simply copying the starting order")
    if model_error < baseline_error:
        outcome = (" -- so the model does beat the baseline on average"
                   + win_text
                   + ", but the margin is small and race by race it is close "
                     "to a coin flip. Read the order of the names, not the "
                     "numbers attached to them.")
    elif model_error == baseline_error:
        outcome = (": the model only matches the baseline, so it adds "
                   "nothing over copying the starting order."
                   + (f" {win_text.strip('. ')}." if win_text else ''))
    else:
        outcome = (" -- the model does NOT beat the baseline, so the "
                   "starting order remains the better guess and the forecast "
                   "is only an ordering guide"
                   + win_text + ".")
    return base + scored + outcome


MODEL_CAVEAT = model_caveat(None)


def predict_race(df: Optional[pd.DataFrame], grid: Dict[str, int]) -> pd.DataFrame:
    """Predict finishing order and win probabilities from a qualifying grid.

    A ridge regression is fitted on the rounds of ``df`` (walk-forward: each
    training race is described by the rounds before it), and the target race
    is described by all of ``df``. Callers pass the rows *before* the race --
    :func:`rows_before_race` -- so nothing from the race itself is used.

    Args:
        df: season race results up to (not including) the target race. Used
            for the fit and for the team/driver history of the target race.
        grid: mapping {driver_code: grid_position}, 1 = pole.

    Returns:
        DataFrame sorted by predicted finish with a ``WinProb`` percent and an
        ``Engine`` column (``ridge`` when a fit was possible, ``heuristic``
        when the history was too short). Every row carries ``Estimated`` plus
        the ``Level``/``Confidence`` derived from the number of race results
        behind the forecast (``pace.data_quality``).
        ``out.attrs`` carries ``caveat``, ``engine``, ``features``,
        ``coefficients``, ``n_train``, ``imputed``, ``level``,
        ``confidence`` and ``n_races``.
    """
    if not grid:
        return pd.DataFrame()

    team_map = _driver_teams(df)
    quality = data_quality(_history_races(df))

    drivers = [str(driver) for driver in grid]
    target = pd.DataFrame({
        'Driver': drivers,
        'Team': [str(team_map.get(driver, '') or '') for driver in drivers],
        'Starting Grid': [max(1, int(position)) for position in grid.values()],
    })
    features = _feature_rows(df, target)

    training = _training_table(df) if df is not None else pd.DataFrame()
    model = None
    if not training.empty:
        rounds = (training['_round'].nunique()
                  if '_round' in training.columns else 1)
        needed = max(MIN_ROWS_PER_COEFFICIENT * (len(FEATURES) + 1),
                     MIN_ROWS_PER_COEFFICIENT)
        if rounds >= MIN_TRAIN_ROUNDS and len(training) >= needed:
            model = ridge_fit(training[list(FEATURES)].to_numpy(dtype=float),
                              training['_y'].to_numpy(dtype=float))

    if model is not None:
        # The ridge target is the finishing position: lower is better, so the
        # negated prediction is the strength that orders the field.
        strength = -ridge_predict(
            model, features[list(FEATURES)].to_numpy(dtype=float))
        engine = ENGINE_RIDGE
        coefficients = {name: round(float(coefficient), 4)
                        for name, coefficient in zip(FEATURES, model['beta'])}
        n_imputed = int(model['n_imputed'])
        n_train = int(model['n'])
    else:
        strength = _heuristic_strength(features)
        engine = ENGINE_FALLBACK
        coefficients = {}
        n_imputed = int(features[list(FEATURES)].isna().sum().sum())
        n_train = 0

    scores = _normalise_strength(strength)
    pdf = pd.DataFrame({
        'Driver': drivers,
        'Team': target['Team'].to_numpy(),
        'Grid': target['Starting Grid'].to_numpy(),
        'TeamPace': pd.to_numeric(features['TeamPace'], errors='coerce')
                      .fillna(0.5).round(3).to_numpy(),
        'Form': pd.to_numeric(features['Form'], errors='coerce')
                  .fillna(0.5).round(3).to_numpy(),
        'Score': np.round(scores, 4),
    })
    if pdf.empty:
        return pdf

    pdf['WinProb'] = _win_probabilities(scores)
    pdf = pdf.sort_values(['Score', 'Driver'],
                          ascending=[False, True], kind='stable')
    pdf = pdf.reset_index(drop=True)
    pdf['Predicted'] = pdf.index + 1
    pdf['Engine'] = engine
    pdf['Estimated'] = ESTIMATED
    pdf['Level'] = quality['level']
    pdf['Confidence'] = quality['confidence']

    reorder = ['Predicted', 'Driver', 'Team', 'Grid', 'TeamPace', 'Form',
               'WinProb', 'Score', 'Engine', 'Estimated', 'Level', 'Confidence']
    out = pdf[[c for c in reorder if c in pdf.columns]]
    out.attrs['caveat'] = MODEL_CAVEAT
    out.attrs['engine'] = engine
    out.attrs['features'] = list(FEATURES)
    out.attrs['coefficients'] = coefficients
    out.attrs['n_train'] = n_train
    out.attrs['imputed'] = n_imputed
    out.attrs['level'] = quality['level']
    out.attrs['confidence'] = quality['confidence']
    out.attrs['n_races'] = quality['n']
    return out


# ---------------------------------------------------------------------------
# Fantasy
# ---------------------------------------------------------------------------

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
    """Add a ``Price`` ($M) column using driver-price mapping.

    A driver without a derivable price (no season points) gets ``NaN`` and
    ``HasPrice=False`` -- not a fabricated floor price -- and is skipped by
    :func:`optimal_lineup`.
    """
    out = pdf.copy()
    out['Price'] = out['Driver'].map(prices) if prices else np.nan
    out['HasPrice'] = pd.to_numeric(out['Price'], errors='coerce').notna()
    return out


def compute_driver_prices(df: Optional[pd.DataFrame],
                           min_points: int = 1) -> Dict[str, int]:
    """Fantasy prices ($M) derived from season points, scaled $5M-$30M.

    A driver with fewer than ``min_points`` season points has no derivable
    market price and is left out of the mapping instead of being priced at
    the floor as if they were a cheap pick. When no driver has scored, the
    mapping is empty rather than a table of invented $5M values.
    """
    if df is None or df.empty or 'Driver' not in df.columns:
        return {}
    if 'Points' not in df.columns:
        return {}
    pts = (pd.to_numeric(df['Points'], errors='coerce').fillna(0)
           .groupby(df['Driver']).sum())
    pts = pts[pts >= max(1, int(min_points))]
    if pts.empty:
        return {}
    top = float(pts.max())
    if top <= 0 or pd.isna(top):
        return {}
    return {
        d: int(max(PRICE_MIN, min(PRICE_MAX,
                                  PRICE_MIN + (v / top) * (PRICE_MAX - PRICE_MIN))))
        for d, v in pts.items()
    }


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
    Drivers without a derivable price are skipped: they have no market price,
    so they cannot be selected on value. Returns the list of chosen driver
    names in selection order.
    """
    if entries is None or entries.empty:
        return []

    pool = entries.copy()
    if price_key in pool.columns:
        pool = pool[pd.to_numeric(pool[price_key], errors='coerce').notna()]
        if pool.empty:
            return []
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
