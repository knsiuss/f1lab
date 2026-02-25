# -*- coding: utf-8 -*-
"""
features.py
~~~~~~~~~~~
Feature engineering for race prediction models.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

try:
    from src.preseason_testing import merge_preseason_features
except ImportError:  # pragma: no cover - fallback when running from src/
    try:
        from preseason_testing import merge_preseason_features
    except ImportError:  # pragma: no cover - optional module unavailable
        merge_preseason_features = None

logger = logging.getLogger(__name__)


def _feature_schema_path(model_dir: str) -> str:
    return os.path.join(model_dir, "feature_columns.json")


def _save_feature_schema(model_dir: str, feature_columns: List[str]) -> None:
    try:
        with open(_feature_schema_path(model_dir), "w", encoding="utf-8") as f:
            json.dump(feature_columns, f)
    except Exception as e:  # pragma: no cover - non-critical persistence
        logger.warning("Could not save feature schema: %s", e)


def _load_feature_schema(model_dir: str) -> Optional[List[str]]:
    path = _feature_schema_path(model_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    except Exception as e:  # pragma: no cover - non-critical persistence
        logger.warning("Could not load feature schema: %s", e)
    return None


def prepare_features(
    df: pd.DataFrame, 
    train_mode: bool = True,
    include_preseason_features: bool = False,
    preseason_csv_path: Optional[str] = None,
    persist_artifacts: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, LabelEncoder]]:
    """Transform race data into encoded feature matrix for model input."""
    logger.info(
        "Preparing features (train_mode=%s, include_preseason_features=%s, persist_artifacts=%s)...",
        train_mode,
        include_preseason_features,
        persist_artifacts,
    )
    
    try:
        # Core feature/target definitions used throughout the function
        base_features = ['Starting Grid', 'Driver', 'Team', 'Track']
        features = list(base_features)
        target = 'Position'
        model_dir = 'models'
        
        def _empty_result(feature_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, LabelEncoder]]:
            """Return consistent empty outputs when no usable data is present."""
            X_empty = pd.DataFrame(columns=feature_columns or features)
            y_empty = pd.Series(name=target, dtype='float64') if train_mode else None
            return X_empty, y_empty, {}
        
        # Validate input
        if df is None or df.empty:
            logger.warning("Received empty DataFrame; returning empty features without error")
            return _empty_result()

        # Ensure models directory exists early (schema alignment depends on it)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f"Created directory: {model_dir}")

        stored_schema = None if train_mode else _load_feature_schema(model_dir)
        schema_requires_preseason = bool(
            stored_schema and any(str(col).startswith("preseason_") for col in stored_schema)
        )

        # Optional 2026 pre-season testing enrichment (team-level local CSV; performance-safe)
        if (include_preseason_features or schema_requires_preseason) and merge_preseason_features is not None:
            try:
                df = merge_preseason_features(df, file_path=preseason_csv_path)
            except Exception as e:
                logger.warning("Pre-season feature merge failed; continuing with core features only: %s", e)
        elif (include_preseason_features or schema_requires_preseason) and merge_preseason_features is None:
            logger.warning("Pre-season feature merge helper unavailable; continuing with core features only")
        
        # Filter to finished races in training mode
        if train_mode:
            if 'Finished' not in df.columns:
                logger.error("'Finished' column required for training mode")
                raise ValueError("Missing 'Finished' column")
            df = df[df['Finished'] == True].copy()
            logger.info(f"Filtered to {len(df)} finished races")
            if df.empty:
                logger.warning("No finished races found; returning empty features")
                return _empty_result()

        # Build expanded feature list after optional enrichment
        if include_preseason_features or schema_requires_preseason:
            preseason_cols = sorted(
                c for c in df.columns
                if isinstance(c, str) and c.startswith("preseason_")
            )
            # Keep only numeric (or coercible numeric) engineered columns
            numeric_preseason_cols: List[str] = []
            for col in preseason_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_preseason_cols.append(col)
                    continue
                if pd.to_numeric(df[col], errors="coerce").notna().any():
                    numeric_preseason_cols.append(col)
            features = base_features + [c for c in numeric_preseason_cols if c not in base_features]

        # In inference mode, honor the exact feature schema used at training time
        if not train_mode and stored_schema:
            features = list(stored_schema)
        
        # Validate required columns exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            missing_core = [f for f in missing_features if f in base_features]
            if missing_core:
                logger.error(f"Missing feature columns: {missing_core}")
                raise ValueError(f"Missing columns: {missing_core}")
        
        if train_mode and target not in df.columns:
            logger.error(f"Target column '{target}' not found")
            raise ValueError(f"Missing target column: {target}")

        # Add absent engineered columns for inference compatibility
        for col in missing_features:
            if col not in df.columns:
                df[col] = 0

        X = df[features].copy()
        y = df[target] if train_mode and target in df.columns else None

        # Normalize numeric columns (including optional pre-season features)
        numeric_cols = [c for c in X.columns if c not in ['Driver', 'Team', 'Track']]
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        if numeric_cols:
            X[numeric_cols] = X[numeric_cols].fillna(0)
        
        # Encode categorical columns
        encoders: Dict[str, LabelEncoder] = {}
        categorical_cols = [c for c in ['Driver', 'Team', 'Track'] if c in X.columns]
        
        for col in categorical_cols:
            le = LabelEncoder()
            encoder_path = os.path.join(model_dir, f'{col}_encoder.pkl')
            
            if train_mode:
                # Fit encoder and save
                try:
                    X[col] = le.fit_transform(X[col])
                    if persist_artifacts:
                        with open(encoder_path, 'wb') as f:
                            pickle.dump(le, f)
                    encoders[col] = le
                    logger.debug(f"Fitted and saved encoder for {col} ({len(le.classes_)} classes)")
                except Exception as e:
                    logger.error(f"Error encoding {col}: {e}")
                    raise
            else:
                # Load existing encoder
                try:
                    with open(encoder_path, 'rb') as f:
                        le = pickle.load(f)
                    
                    # Handle unknown categories gracefully
                    unknown_values = set(X[col].unique()) - set(le.classes_)
                    if unknown_values:
                        logger.warning(f"Unknown values in {col}: {unknown_values}")
                    
                    X[col] = X[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                    encoders[col] = le
                    logger.debug(f"Loaded encoder for {col} ({len(le.classes_)} classes)")
                except FileNotFoundError:
                    logger.error(f"Encoder for {col} not found at {encoder_path}")
                    logger.error("Please train the model first using train_mode=True")
                    raise
                except Exception as e:
                    logger.error(f"Error loading encoder for {col}: {e}")
                    raise

        if train_mode and persist_artifacts:
            _save_feature_schema(model_dir, X.columns.tolist())
        
        logger.info(f"Features prepared: {X.shape[0]} samples, {X.shape[1]} features")
        if y is not None:
            logger.debug(f"Target range: {y.min()}-{y.max()}")
        
        return X, y, encoders
        
    except Exception as e:
        logger.exception(f"Error preparing features: {e}")
        raise


def calculate_degradation_curve(compound: str) -> float:
    """
    Calculate time loss per lap due to tire wear (seconds).
    Returns the degradation factor (s/lap).
    """
    c = str(compound).upper()
    if 'SOFT' in c: return 0.12   # High deg
    if 'MEDIUM' in c: return 0.08 # Medium deg
    if 'HARD' in c: return 0.04   # Low deg
    if 'INTER' in c: return 0.05
    if 'WET' in c: return 0.05
    return 0.08


def calculate_fuel_correction(current_lap: int, total_laps: int) -> float:
    """
    Calculate time gained due to fuel burn (seconds).
    Returns negative time delta (time gained) relative to heavy start.
    Avg gain ~0.06s per lap driven.
    """
    return float(current_lap) * -0.06
