# -*- coding: utf-8 -*-
"""
model.py
~~~~~~~~
Race position prediction model.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

try:
    from src.features import prepare_features
except ImportError:
    from features import prepare_features

logger = logging.getLogger(__name__)

MODEL_PATH = 'models/f1_model.pkl'
N_ESTIMATORS = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42


def train_model(
    df: pd.DataFrame,
    enable_preseason_features: bool = True,
    preseason_csv_path: Optional[str] = None,
    persist: bool = True,
) -> Tuple[HistGradientBoostingRegressor, pd.DataFrame, pd.Series]:
    """Train position prediction model on race data."""
    logger.info("Starting model training...")
    
    # Prepare features
    logger.info("Preparing features...")
    X, y, _ = prepare_features(
        df,
        train_mode=True,
        include_preseason_features=enable_preseason_features,
        preseason_csv_path=preseason_csv_path,
        persist_artifacts=persist,
    )
    if X is None or X.empty or y is None or y.empty:
        raise ValueError("Training data produced empty feature matrix/target")
    
    # Split data
    if len(X) < 3:
        raise ValueError("Need at least 3 samples to train/test split a model")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train model
    logger.info(f"Training HistGradientBoostingRegressor...")
    model = HistGradientBoostingRegressor(
        max_iter=N_ESTIMATORS,
        learning_rate=0.1,
        max_depth=10,
        random_state=RANDOM_STATE,
        early_stopping=True
    )
    model.fit(X_train, y_train)
    # Persist feature names for UI/debug display.
    try:
        setattr(model, "feature_names_in_", np.array(X.columns.tolist(), dtype=object))
    except Exception:
        pass
    # HistGradientBoostingRegressor may not expose feature_importances_ depending on sklearn version.
    if not hasattr(model, "feature_importances_"):
        fallback_importance = np.ones(X.shape[1], dtype=float)
        if fallback_importance.sum() > 0:
            fallback_importance = fallback_importance / fallback_importance.sum()
        try:
            setattr(model, "feature_importances_", fallback_importance)
        except Exception:
            pass
    
    # Save model
    if persist:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {MODEL_PATH}")
    
    # Log feature importances
    feature_names = X.columns.tolist()
    importances_raw = getattr(model, "feature_importances_", None)
    if importances_raw is not None and len(importances_raw) == len(feature_names):
        importances = dict(zip(feature_names, importances_raw))
        logger.info(f"Feature importances: {importances}")
    else:
        logger.info("Feature importances unavailable for this model/sklearn version")
    
    return model, X_test, y_test


def load_trained_model() -> Optional[HistGradientBoostingRegressor]:
    """Load persisted model from disk."""
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
    except:
        pass
    return None


class RaceStrategySimulator:
    """Race strategy optimizer using tyre degradation and fuel models."""
    
    def __init__(self, base_lap_time: float = 90.0, total_laps: int = 57):
        self.base_lap_time = base_lap_time
        self.total_laps = total_laps
        self.pit_loss = 22.0 # Avg pit loss in seconds
        
    def predict_strategy(self, driver: str, start_tire: str, current_lap: int = 0) -> dict:
        """
        Predict optimal strategy and finish time.
        """
        # Try 1-stop
        t_1stop = self._simulate_stint(start_tire, current_lap, self.total_laps)
        
        # Try 2-stop (simplified split)
        stop_lap = current_lap + (self.total_laps - current_lap) // 2
        t_2stop_s1 = self._simulate_stint(start_tire, current_lap, stop_lap)
        t_2stop_s2 = self._simulate_stint('MEDIUM', stop_lap, self.total_laps) # Assume switch to Med
        t_2stop = t_2stop_s1 + t_2stop_s2 + self.pit_loss
        
        return {
            '1_stop_time': t_1stop,
            '2_stop_time': t_2stop,
            'recommended': '1 Stop' if t_1stop < t_2stop else '2 Stops',
            'delta': abs(t_1stop - t_2stop)
        }

    def _simulate_stint(self, compound: str, start_lap: int, end_lap: int) -> float:
        """Simulate total time for a stint."""
        from features import calculate_degradation_curve, calculate_fuel_correction
        
        total_time = 0.0
        deg_factor = calculate_degradation_curve(compound)
        
        for lap in range(start_lap, end_lap):
            # Physics: Base + Deg - Fuel
            tyre_age = lap - start_lap
            deg_penalty = tyre_age * deg_factor
            fuel_gain = calculate_fuel_correction(lap, self.total_laps)
            
            lap_time = self.base_lap_time + deg_penalty + fuel_gain
            total_time += lap_time
            
        return total_time

    def catch_up_prediction(self, chaser_gap: float, chaser_tire: str, leader_tire: str, laps_remaining: int) -> int:
        """
        Predict lap when chaser catches leader.
        Returns lap number or -1 if never.
        """
        from features import calculate_degradation_curve
        
        deg_chaser = calculate_degradation_curve(chaser_tire)
        deg_leader = calculate_degradation_curve(leader_tire)
        
        current_gap = chaser_gap
        
        for i in range(laps_remaining):
            # Relative pace (neg = chaser faster)
            # Simplified: Assume chaser has fresher tires (-0.5s avg advantage)
            pace_delta = -0.5 + (deg_chaser * i) - (deg_leader * i)
            current_gap += pace_delta
            
            if current_gap <= 0:
                return i # Laps to catch
                
        return -1
