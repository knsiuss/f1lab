# -*- coding: utf-8 -*-
"""
setupfingerprint.py
~~~~~~~~~~~~~~~~~~~
Setup fingerprint: a first-principles setup proxy from public telemetry.

Physics: a lap is corners + straights, and public telemetry observes both.
That lets us estimate the two ends of the classic drag/downforce trade-off
per driver without any team data:

* Top-speed proxy -- median of the K best clean-lap ``SpeedST`` trap values
  (drag + power limited end of the straight).
* Cornering proxies -- mean apex speed in slow / medium / fast buckets,
  detected as smoothed speed minima on the driver's fastest clean lap,
  plus mean peak braking deceleration into those apexes (grip-limited
  ends of the corners).

Normalised against the session best this becomes a comparable fingerprint;
diffing fingerprints across races exposes setup or upgrade shifts ("what did
they bring to this round?").

Data Honesty contract: every figure is estimated, labelled ``estimated``,
and gated on sample quality via :mod:`pace`. Drivers without enough clean
laps keep their row with ``None`` cornering metrics instead of fabricated
ones. Trap speeds include slipstream and DRS effects that cannot be fully
removed from public data; within-session comparison only partially cancels
this and the UI must say so.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import AERO_CONFIG
from pace import clean_laps_for_analysis, data_quality

ESTIMATED = "estimated"

# Metric columns produced by a fingerprint row (raw units).
METRIC_COLS = ['TopSpeed', 'SlowApex', 'MediumApex', 'FastApex', 'BrakeDecelG']

_G = 9.81  # m/s^2 per g


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean; short traces degrade gracefully."""
    s = pd.Series(np.asarray(values, dtype=float))
    return s.rolling(window, min_periods=max(1, window // 2), center=True).mean().to_numpy()


def classify_corner(apex_speed: float, cfg: Dict) -> str:
    """Bucket an apex speed into slow / medium / fast (mechanical vs aero regime)."""
    if apex_speed <= cfg['slow_max_kmh']:
        return 'slow'
    if apex_speed <= cfg['medium_max_kmh']:
        return 'medium'
    return 'fast'


def detect_corners(speed: np.ndarray, cfg: Optional[Dict] = None) -> List[Dict]:
    """Detect corner events as smoothed-speed local minima with real braking.

    A minimum counts when the highest smoothed speed within the entry scan
    window before it exceeds the apex by at least ``min_speed_drop_kmh``.

    Returns a list of dicts sorted by index:
        ``apex_idx, entry_idx, apex_speed, entry_speed``
    """
    cfg = {**AERO_CONFIG, **(cfg or {})}
    s = np.asarray(speed, dtype=float)
    w = int(cfg['smooth_window'])
    if s.size < max(3 * w, 10):
        return []

    sm = _smooth(s, w)
    sm = np.nan_to_num(sm, nan=np.nanmax(sm) if np.isfinite(sm).any() else 0.0)
    drop_min = float(cfg['min_speed_drop_kmh'])
    scan_back = int(cfg['entry_scan_samples'])

    corners: List[Dict] = []
    for i in range(1, len(sm) - 1):
        if not (sm[i] <= sm[i - 1] and sm[i] < sm[i + 1]):
            continue  # not a local minimum
        lo = max(0, i - scan_back)
        window = sm[lo:i]
        if window.size == 0:
            continue
        entry_rel = int(np.argmax(window))
        entry_speed = float(window[entry_rel])
        if entry_speed - float(sm[i]) >= drop_min:
            corners.append({
                'apex_idx': int(i),
                'entry_idx': int(lo + entry_rel),
                'apex_speed': float(s[i]),
                'entry_speed': entry_speed,
            })
    return corners


def _cornering_metrics(car_data: pd.DataFrame,
                       cfg: Optional[Dict] = None) -> Optional[Dict[str, float]]:
    """Corner-bucket apex means plus mean peak braking g from one lap's telemetry.

    Returns None when the frame is unusable; returns partial results with
    ``corners=0`` style guards when too few corners are detected.
    """
    cfg = {**AERO_CONFIG, **(cfg or {})}
    try:
        if car_data is None or car_data.empty or 'Speed' not in car_data.columns:
            return None
        speed = pd.to_numeric(car_data['Speed'], errors='coerce').to_numpy(float)
        corners = detect_corners(speed, cfg)
        if len(corners) < int(cfg['min_corners_total']):
            return {'corners': len(corners), 'SlowApex': None, 'MediumApex': None,
                    'FastApex': None, 'BrakeDecelG': None}

        # Longitudinal acceleration in g along the lap (for braking severity).
        brake_g = None
        if 'Time' in car_data.columns:
            t = pd.to_numeric(pd.Series(car_data['Time']).astype('int64'),
                              errors='coerce').to_numpy(float) / 1e9
            if len(t) == len(speed) and np.isfinite(t).all() and np.nanstd(t) > 0:
                accel = np.gradient(speed / 3.6, t)  # km/h -> m/s before gradient
                peak_gs = []
                win = int(cfg['brake_window_samples'])
                for c in corners:
                    lo = max(0, c['apex_idx'] - win)
                    seg = accel[lo:c['apex_idx'] + 1]
                    if seg.size:
                        peak_gs.append(-float(np.min(seg)) / _G)
                if peak_gs:
                    brake_g = float(np.mean(peak_gs))

        buckets: Dict[str, List[float]] = {'slow': [], 'medium': [], 'fast': []}
        for c in corners:
            buckets[classify_corner(c['apex_speed'], cfg)].append(c['apex_speed'])

        def mean_or_none(vals):
            return round(float(np.mean(vals)), 2) if vals else None

        return {
            'corners': len(corners),
            'SlowApex': mean_or_none(buckets['slow']),
            'MediumApex': mean_or_none(buckets['medium']),
            'FastApex': mean_or_none(buckets['fast']),
            'BrakeDecelG': round(brake_g, 3) if brake_g else None,
        }
    except Exception:
        return None


def _top_speed(clean: pd.DataFrame, cfg: Dict) -> Optional[float]:
    """Median of the K best clean SpeedST trap readings (km/h), robust to spikes."""
    cfg = {**AERO_CONFIG, **(cfg or {})}
    k = int(cfg['top_trap_laps'])
    if 'SpeedST' not in clean.columns or clean.empty:
        return None
    vals = pd.to_numeric(clean['SpeedST'], errors='coerce').dropna()
    if vals.empty:
        return None
    top = vals.nlargest(min(k, len(vals)))
    return round(float(top.median()), 1)


def _driver_car_data(session, clean: pd.DataFrame, driver: str):
    """Telemetry for the driver's fastest clean lap (or None)."""
    try:
        sub = clean[(clean['Driver'] == driver) & clean['LapTime'].notna()]
        if sub.empty or session is None:
            return None
        lap = sub.loc[sub['LapTime'].idxmin()]
        car = lap.get_car_data()
        if car is None or car.empty:
            return None
        if 'Distance' not in car.columns:
            car = car.add_distance()
        return car
    except Exception:
        return None


def fingerprint(laps: pd.DataFrame,
                session=None,
                drivers: Optional[List[str]] = None,
                cfg: Optional[Dict] = None) -> pd.DataFrame:
    """Per-driver setup fingerprint for one session.

    Args:
        laps: FastF1-style lap table (needs Driver, LapTime; SpeedST for traps).
        session: FastF1 Session for telemetry-derived cornering metrics;
            when None only trap-based metrics are filled.
        drivers: restrict to these driver codes (default: all present).
        cfg: overrides merged over ``config.AERO_CONFIG``.

    Returns:
        DataFrame (possibly empty): Driver, Team, TopSpeed, SlowApex,
        MediumApex, FastApex, BrakeDecelG, CornersDetected, Level,
        Confidence, Estimated. Rows exist even when metrics are missing.
    """
    cfg = {**AERO_CONFIG, **(cfg or {})}
    if laps is None or laps.empty or 'Driver' not in laps.columns:
        return pd.DataFrame()

    clean = clean_laps_for_analysis(laps)
    if clean.empty:
        return pd.DataFrame()

    scope = sorted(set(drivers)) if drivers else sorted(clean['Driver'].dropna().unique())
    team_map = (laps.dropna(subset=['Driver'])
                .groupby('Driver')['Team'].first().to_dict()
                if 'Team' in laps.columns else {})

    rows = []
    for drv in scope:
        dclean = clean[clean['Driver'] == drv]
        n = len(dclean)
        quality = data_quality(n)

        top = _top_speed(dclean, cfg)

        corner_block: Optional[Dict[str, float]] = None
        if session is not None:
            car = _driver_car_data(session, clean, drv)
            corner_block = _cornering_metrics(car, cfg)

        rows.append({
            'Driver': drv,
            'Team': team_map.get(drv),
            'TopSpeed': top,
            'SlowApex': corner_block.get('SlowApex') if corner_block else None,
            'MediumApex': corner_block.get('MediumApex') if corner_block else None,
            'FastApex': corner_block.get('FastApex') if corner_block else None,
            'BrakeDecelG': corner_block.get('BrakeDecelG') if corner_block else None,
            'CornersDetected': corner_block.get('corners') if corner_block else 0,
            'Level': quality['level'],
            'Confidence': quality['confidence'],
            'Estimated': ESTIMATED,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values('TopSpeed', ascending=False, na_position='last') \
              .reset_index(drop=True)


def normalise_fingerprint(fp: pd.DataFrame,
                          cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Add ``*_Idx`` columns scaled so the session best sits at 1.000.

    Drivers lacking a metric get NaN in its index column rather than a fake
    score. Raw columns are preserved alongside the indices.
    """
    if fp is None or fp.empty:
        return pd.DataFrame()
    cols = cols or [c for c in METRIC_COLS if c in fp.columns]
    out = fp.copy()
    for col in cols:
        series = pd.to_numeric(out[col], errors='coerce')
        best = series.max(skipna=True)
        out[f'{col}Idx'] = (series / best).round(3) if best and best > 0 else np.nan
    return out


def fingerprint_shift(fp_a: pd.DataFrame, fp_b: pd.DataFrame,
                      cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Race-to-race shift per driver: metric_b - metric_a (positive = bigger in B).

    Only drivers present with numeric values in both fingerprints are kept --
    no interpolation, no invention.
    """
    cols = cols or [c for c in METRIC_COLS if c in fp_a.columns and c in fp_b.columns]
    if not cols:
        return pd.DataFrame()

    a = fp_a[['Driver'] + [c for c in cols if c in fp_a.columns]].copy()
    b = fp_b[['Driver'] + [c for c in cols if c in fp_b.columns]].copy()
    for frame in (a, b):
        for c in cols:
            if c in frame.columns:
                frame[c] = pd.to_numeric(frame[c], errors='coerce')

    merged = a.merge(b, on='Driver', how='inner', suffixes=('_a', '_b'))
    if merged.empty:
        return pd.DataFrame()

    out = pd.DataFrame({'Driver': merged['Driver']})
    for c in cols:
        ca, cb = f'{c}_a', f'{c}_b'
        if ca in merged.columns and cb in merged.columns:
            out[f'{c}_delta'] = (merged[cb] - merged[ca]).round(2)

    delta_cols = [c for c in out.columns if c.endswith('_delta')]
    if not delta_cols:
        return pd.DataFrame()
    out['AbsShift'] = out[delta_cols].abs().sum(axis=1, skipna=True).round(2)
    out['Estimated'] = ESTIMATED
    return out.sort_values('AbsShift', ascending=False).reset_index(drop=True)
