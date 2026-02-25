# -*- coding: utf-8 -*-
"""
test_preseason_testing.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for 2026 pre-season testing ingestion and ML feature enrichment.
"""

from pathlib import Path
import sys

import pandas as pd

# Add project root to path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preseason_testing import (  # noqa: E402
    load_preseason_testing_team_summary,
    load_preseason_testing_team_features,
    merge_preseason_features,
)
from src.features import prepare_features  # noqa: E402


def test_default_preseason_dataset_loads():
    df = load_preseason_testing_team_summary()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 33
    assert "venue" in df.columns
    assert "Bahrain" in set(df["venue"].astype(str))
    assert any("Barcelona" in v for v in df["venue"].astype(str).tolist())


def test_aggregated_preseason_team_features_cover_all_teams():
    features_df = load_preseason_testing_team_features()
    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) >= 11
    assert "team_norm" in features_df.columns
    assert "preseason_bahrain_best_lap_sec" in features_df.columns
    assert "preseason_spain_shakedown_participated" in features_df.columns


def test_merge_preseason_features_adds_trackmatch_signals():
    race_df = pd.DataFrame(
        {
            "Driver": ["Lando Norris", "Carlos Sainz", "George Russell"],
            "Team": ["McLaren Mercedes", "Williams Mercedes", "Mercedes"],
            "Track": ["Bahrain", "Spanish Grand Prix", "Australia"],
            "Starting Grid": [1, 5, 2],
            "Position": [1, 3, 2],
            "Finished": [True, True, True],
        }
    )
    enriched = merge_preseason_features(race_df)
    preseason_cols = [c for c in enriched.columns if c.startswith("preseason_")]
    assert preseason_cols
    assert "preseason_track_match_has_direct_test" in enriched.columns
    assert int(enriched.loc[0, "preseason_track_match_has_direct_test"]) == 1  # Bahrain
    assert int(enriched.loc[1, "preseason_track_match_has_direct_test"]) == 1  # Spain/Barcelona
    assert "preseason_bahrain_best_lap_sec" in enriched.columns


def test_prepare_features_supports_preseason_enrichment(tmp_path):
    race_df = pd.DataFrame(
        {
            "Driver": ["Lando Norris", "George Russell", "Carlos Sainz", "Lando Norris"],
            "Team": ["McLaren Mercedes", "Mercedes", "Williams Mercedes", "McLaren Mercedes"],
            "Track": ["Bahrain", "Australia", "Spanish Grand Prix", "Bahrain"],
            "Starting Grid": [1, 2, 4, 3],
            "Position": [1, 2, 5, 3],
            "Finished": [True, True, True, True],
        }
    )
    original_cwd = Path.cwd()
    try:
        # Use isolated models dir for encoder/schema artifacts.
        (tmp_path / "models").mkdir(exist_ok=True)
        os_cwd = str(tmp_path)
        import os
        os.chdir(os_cwd)
        X, y, encoders = prepare_features(race_df, train_mode=True, include_preseason_features=True)
    finally:
        import os
        os.chdir(original_cwd)

    assert len(X.columns) > 4
    assert any(c.startswith("preseason_") for c in X.columns)
    assert y is not None and len(y) == 4
    assert {"Driver", "Team", "Track"}.issubset(encoders.keys())
