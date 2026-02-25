# -*- coding: utf-8 -*-
"""Pre-season testing datasets and feature engineering helpers for F1 2026."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PRESEASON_TEAM_SUMMARY_PATH = (
    Path(__file__).parent.parent / "data" / "F1_2026_PreseasonTesting_TeamSummary.csv"
)


TEAM_NAME_ALIASES: Dict[str, str] = {
    "mclaren": "McLaren",
    "mclaren mercedes": "McLaren",
    "mclaren formula 1 team": "McLaren",
    "mercedes": "Mercedes",
    "mercedes-amg": "Mercedes",
    "mercedes-amg petronas f1 team": "Mercedes",
    "ferrari": "Ferrari",
    "scuderia ferrari": "Ferrari",
    "scuderia ferrari hp": "Ferrari",
    "red bull": "Red Bull",
    "red bull racing": "Red Bull",
    "red bull racing honda rbpt": "Red Bull",
    "williams": "Williams",
    "williams mercedes": "Williams",
    "williams racing": "Williams",
    "racing bulls": "Racing Bulls",
    "rb": "Racing Bulls",
    "vcarb": "Racing Bulls",
    "visa cash app rb": "Racing Bulls",
    "racing bulls honda rbpt": "Racing Bulls",
    "aston martin": "Aston Martin",
    "aston martin aramco": "Aston Martin",
    "aston martin aramco mercedes": "Aston Martin",
    "alpine": "Alpine",
    "alpine renault": "Alpine",
    "bwt alpine f1 team": "Alpine",
    "haas": "Haas",
    "haas ferrari": "Haas",
    "moneygram haas f1 team": "Haas",
    "audi": "Audi",
    "kick sauber": "Audi",
    "kick sauber ferrari": "Audi",
    "sauber": "Audi",
    "stake f1 team kick sauber": "Audi",
    "cadillac": "Cadillac",
    "cadillac formula 1 team": "Cadillac",
}

TRACK_TEST_VENUE_KEYWORDS = {
    "Bahrain": ["bahrain", "sakhir"],
    "Barcelona-Catalunya": ["spanish", "spain", "barcelona", "catalunya", "catalunya"],
}


def normalize_team_name(name: object) -> str:
    """Map project/team variants to the 2026 pre-season summary naming."""
    if name is None:
        return ""
    raw = str(name).strip()
    if not raw:
        return ""
    key = raw.lower()
    if key in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[key]
    for alias, canonical in TEAM_NAME_ALIASES.items():
        if alias in key or key in alias:
            return canonical
    return raw


def parse_lap_time_to_seconds(value: object) -> float:
    """Convert an F1 lap time string like 1:29.348 to seconds."""
    if value is None:
        return float("nan")
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return float("nan")
    try:
        if ":" in s:
            mins, secs = s.split(":", 1)
            return float(mins) * 60.0 + float(secs)
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def load_preseason_testing_team_summary(
    file_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load curated 2026 pre-season team-level testing summary."""
    path = Path(file_path) if file_path else DEFAULT_PRESEASON_TEAM_SUMMARY_PATH
    if not path.exists():
        logger.warning("Pre-season team summary file not found: %s", path)
        return pd.DataFrame()

    df = pd.read_csv(path)
    if df.empty:
        return df

    for col in ("start_date", "end_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    for col in ("participated", "late_start", "official_timing_available"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "total_laps" in df.columns:
        df["total_laps"] = pd.to_numeric(df["total_laps"], errors="coerce")

    if "fastest_lap_time" in df.columns:
        df["fastest_lap_seconds"] = df["fastest_lap_time"].apply(parse_lap_time_to_seconds)
    else:
        df["fastest_lap_seconds"] = float("nan")

    if "team" in df.columns:
        df["team_norm"] = df["team"].apply(normalize_team_name)
    else:
        df["team_norm"] = ""

    return df


def _aggregate_preseason_team_features(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate curated team summary rows into numeric ML features."""
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    df = summary_df.copy()
    df["total_laps_num"] = pd.to_numeric(df.get("total_laps"), errors="coerce")
    df["timed"] = pd.to_numeric(df.get("official_timing_available"), errors="coerce").fillna(0).astype(int)
    df["participated_num"] = pd.to_numeric(df.get("participated"), errors="coerce").fillna(0).astype(int)
    df["late_start_num"] = pd.to_numeric(df.get("late_start"), errors="coerce").fillna(0).astype(int)

    group = df.groupby("team_norm", dropna=False)
    agg = group.agg(
        preseason_entries=("test_label", "size"),
        preseason_tests_participated=("participated_num", "sum"),
        preseason_timed_entries=("timed", "sum"),
        preseason_total_laps_all=("total_laps_num", "sum"),
        preseason_best_lap_sec_all=("fastest_lap_seconds", "min"),
        preseason_avg_best_lap_sec_all=("fastest_lap_seconds", "mean"),
        preseason_unique_sources=("source_url", "nunique"),
    ).reset_index()

    # Bahrain official tests (timed and most relevant to early 2026 races).
    bahrain = df[
        df["venue"].astype(str).str.contains("bahrain", case=False, na=False)
        & (df["timed"] == 1)
    ].copy()
    if not bahrain.empty:
        b_group = bahrain.groupby("team_norm", dropna=False).agg(
            preseason_bahrain_test_entries=("test_label", "size"),
            preseason_bahrain_total_laps=("total_laps_num", "sum"),
            preseason_bahrain_best_lap_sec=("fastest_lap_seconds", "min"),
            preseason_bahrain_avg_best_lap_sec=("fastest_lap_seconds", "mean"),
        ).reset_index()
        agg = agg.merge(b_group, on="team_norm", how="left")
    else:
        for col in (
            "preseason_bahrain_test_entries",
            "preseason_bahrain_total_laps",
            "preseason_bahrain_best_lap_sec",
            "preseason_bahrain_avg_best_lap_sec",
        ):
            agg[col] = pd.NA

    # Spain private shakedown metadata (no public timing).
    spain = df[df["venue"].astype(str).str.contains("barcelona", case=False, na=False)].copy()
    if not spain.empty:
        s_group = spain.groupby("team_norm", dropna=False).agg(
            preseason_spain_shakedown_participated=("participated_num", "max"),
            preseason_spain_shakedown_late_start=("late_start_num", "max"),
            preseason_spain_shakedown_private_no_times=("timed", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) == 0).any())),
        ).reset_index()
        agg = agg.merge(s_group, on="team_norm", how="left")
    else:
        agg["preseason_spain_shakedown_participated"] = pd.NA
        agg["preseason_spain_shakedown_late_start"] = pd.NA
        agg["preseason_spain_shakedown_private_no_times"] = pd.NA

    # Ranks computed across Bahrain timed test summaries.
    if "preseason_bahrain_best_lap_sec" in agg.columns:
        agg["preseason_bahrain_pace_rank"] = (
            pd.to_numeric(agg["preseason_bahrain_best_lap_sec"], errors="coerce")
            .rank(method="dense", ascending=True)
        )
    if "preseason_bahrain_total_laps" in agg.columns:
        agg["preseason_bahrain_mileage_rank"] = (
            pd.to_numeric(agg["preseason_bahrain_total_laps"], errors="coerce")
            .rank(method="dense", ascending=False)
        )

    # Track-match fallbacks use overall best/laps when a direct venue match is absent.
    agg["preseason_trackmatch_default_best_lap_sec"] = pd.to_numeric(
        agg.get("preseason_best_lap_sec_all"), errors="coerce"
    )
    agg["preseason_trackmatch_default_total_laps"] = pd.to_numeric(
        agg.get("preseason_total_laps_all"), errors="coerce"
    )
    return agg


def load_preseason_testing_team_features(
    file_path: Optional[str] = None,
) -> pd.DataFrame:
    """Public helper for aggregated numeric pre-season features."""
    summary_df = load_preseason_testing_team_summary(file_path=file_path)
    return _aggregate_preseason_team_features(summary_df)


def _track_to_test_venue(track_name: object) -> Optional[str]:
    if track_name is None:
        return None
    name = str(track_name).strip().lower()
    if not name:
        return None
    for venue, keywords in TRACK_TEST_VENUE_KEYWORDS.items():
        if any(k in name for k in keywords):
            return venue
    return None


def merge_preseason_features(
    df: pd.DataFrame,
    file_path: Optional[str] = None,
) -> pd.DataFrame:
    """Merge aggregated 2026 pre-season testing features onto race/prediction rows."""
    if df is None or df.empty:
        return df

    feature_df = load_preseason_testing_team_features(file_path=file_path)
    if feature_df.empty:
        return df.copy()

    out = df.copy()
    out["__team_norm_preseason"] = out.get("Team", pd.Series(index=out.index, dtype="object")).apply(
        normalize_team_name
    )
    out = out.merge(
        feature_df,
        left_on="__team_norm_preseason",
        right_on="team_norm",
        how="left",
    )

    if "Track" in out.columns:
        out["__track_test_venue"] = out["Track"].apply(_track_to_test_venue)
    else:
        out["__track_test_venue"] = None

    # Direct venue-specific features for Bahrain/Barcelona.
    bahrain_mask = out["__track_test_venue"].eq("Bahrain")
    spain_mask = out["__track_test_venue"].eq("Barcelona-Catalunya")

    out["preseason_track_match_has_direct_test"] = (bahrain_mask | spain_mask).astype(int)
    out["preseason_track_match_fastest_lap_sec"] = pd.to_numeric(
        out.get("preseason_trackmatch_default_best_lap_sec"), errors="coerce"
    )
    out["preseason_track_match_total_laps"] = pd.to_numeric(
        out.get("preseason_trackmatch_default_total_laps"), errors="coerce"
    )
    if "preseason_bahrain_best_lap_sec" in out.columns:
        out.loc[bahrain_mask, "preseason_track_match_fastest_lap_sec"] = pd.to_numeric(
            out.loc[bahrain_mask, "preseason_bahrain_best_lap_sec"], errors="coerce"
        )
    if "preseason_bahrain_total_laps" in out.columns:
        out.loc[bahrain_mask, "preseason_track_match_total_laps"] = pd.to_numeric(
            out.loc[bahrain_mask, "preseason_bahrain_total_laps"], errors="coerce"
        )
    # Spain shakedown has no public timings; retain defaults and expose coverage/flags.
    out.loc[spain_mask, "preseason_track_match_has_direct_test"] = 1

    # Cleanup join helper columns but keep engineered features.
    drop_cols = ["__team_norm_preseason", "team_norm", "__track_test_venue"]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])
    return out

