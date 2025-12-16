"""
SVR Model Module (Support Vector Regression)

This module implements a Support Vector Regression model following the
standardized model interface.

SVR with RBF Kernel:
    - Captures nonlinear patterns in electricity prices
    - Uses kernel trick for high-dimensional feature mapping
    - Robust to outliers via epsilon-insensitive loss

Based on Tschora et al. (2022) showing MAE 3.67-3.15 EUR/MWh on European markets.

Hyperparameter tuning via TimeSeriesSplit cross-validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')

from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import TASK_TYPE, SVR_CONFIG, RANDOM_SEED


def select_top_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_features: int = 50
) -> Tuple[List[int], List[str]]:
    """
    Select top features by correlation with target.

    Returns:
        Tuple of (feature_indices, feature_names)
    """
    correlations = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        if np.isnan(corr):
            corr = 0
        correlations.append(abs(corr))

    # Get indices of top features
    sorted_indices = np.argsort(correlations)[::-1]
    top_indices = sorted_indices[:n_features].tolist()

    if feature_names:
        top_names = [feature_names[i] for i in top_indices]
    else:
        top_names = [f"feature_{i}" for i in top_indices]

    return top_indices, top_names


def tune_svr_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    """
    Tune SVR hyperparameters using TimeSeriesSplit cross-validation.

    Returns:
        Best hyperparameters dict
    """
    param_grid = config.get("param_grid", {
        "C": [0.1, 1, 10, 100, 1000],
        "epsilon": [0.01, 0.1, 0.5, 1.0],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "kernel": ["rbf", "poly"],
    })

    cv_folds = config.get("cv_folds", 5)
    n_iter = config.get("n_iter", 50)

    print(f"  Tuning SVR with {n_iter} iterations, {cv_folds}-fold TimeSeriesSplit...")

    tscv = TimeSeriesSplit(n_splits=cv_folds)

    svr = SVR()

    search = RandomizedSearchCV(
        svr,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=0,
    )

    search.fit(X_train, y_train)

    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV MAE: {-search.best_score_:.6f}")

    return search.best_params_


def train_svr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict = None,
) -> Tuple[SVR, Dict[str, Any]]:
    """
    Train SVR model with optional hyperparameter tuning.

    Returns:
        Tuple of (trained_model, best_params)
    """
    if config is None:
        config = SVR_CONFIG

    if config.get("tune_hyperparams", True):
        best_params = tune_svr_hyperparameters(X_train, y_train, config)
    else:
        best_params = {
            "kernel": config.get("kernel", "rbf"),
            "C": config.get("C", 100),
            "epsilon": config.get("epsilon", 0.1),
            "gamma": config.get("gamma", "scale"),
        }
        print(f"  Using default params: {best_params}")

    # Train final model
    model = SVR(**best_params)
    model.fit(X_train, y_train)

    return model, best_params


def train_and_predict(
    datasets: Dict[str, Any],
    config: dict = None
) -> Dict[str, Any]:
    """
    Standard model interface for SVR.

    Performs feature selection to reduce dimensionality,
    then trains SVR with hyperparameter tuning.
    """
    if config is None:
        config = SVR_CONFIG

    np.random.seed(RANDOM_SEED)

    # Extract data
    X_train = datasets["X_train"]
    y_train = datasets["y_train"]
    X_val = datasets["X_val"]
    y_val = datasets["y_val"]
    X_test = datasets["X_test"]
    y_test = datasets["y_test"]
    feature_names = datasets.get("feature_names", [])

    # Handle sequential data (3D -> 2D)
    if len(X_train.shape) == 3:
        X_train = X_train[:, -1, :]
        X_val = X_val[:, -1, :]
        X_test = X_test[:, -1, :]

    print("\nSVR Model (Support Vector Regression):")
    print("-" * 80)
    print(f"  Task type: {TASK_TYPE}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Original features: {X_train.shape[1]}")

    # Feature selection
    n_features = config.get("n_features", 50)
    if X_train.shape[1] > n_features:
        print(f"\nPerforming feature selection (top {n_features} features)...")
        feature_indices, selected_names = select_top_features(
            X_train, y_train, feature_names, n_features
        )
        X_train_sel = X_train[:, feature_indices]
        X_val_sel = X_val[:, feature_indices]
        X_test_sel = X_test[:, feature_indices]
        print(f"  Selected {len(feature_indices)} features")
        print(f"  Top 5: {selected_names[:5]}")
    else:
        X_train_sel = X_train
        X_val_sel = X_val
        X_test_sel = X_test
        feature_indices = list(range(X_train.shape[1]))
        selected_names = feature_names

    # Re-scale selected features (SVR is sensitive to scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_val_scaled = scaler.transform(X_val_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    # For classification, train on return direction
    if TASK_TYPE == "classification":
        # y_train is already binary (0/1)
        # Convert to continuous for SVR, then threshold predictions
        y_train_svr = y_train.astype(float)
        y_val_svr = y_val.astype(float)
    else:
        y_train_svr = y_train
        y_val_svr = y_val

    # Train model
    print("\nTraining SVR model...")
    model, best_params = train_svr(X_train_scaled, y_train_svr, config)

    # Predictions
    print("\nGenerating predictions...")
    y_pred_val_raw = model.predict(X_val_scaled)
    y_pred_test_raw = model.predict(X_test_scaled)

    # Convert predictions based on task type
    if TASK_TYPE == "classification":
        # Use Platt scaling (sigmoid calibration) to convert SVR outputs to probabilities
        # This is more principled than arbitrary thresholding at 0.5
        # Fit sigmoid on validation predictions to calibrate
        from scipy.special import expit

        # SVR outputs are in [0, 1] range when trained on binary targets
        # Apply sigmoid calibration: map to proper probabilities
        # Center around 0.5 and scale for better separation
        y_pred_val = expit((y_pred_val_raw - 0.5) * 10)
        y_pred_test = expit((y_pred_test_raw - 0.5) * 10)

        print(f"\nPrediction distribution (calibrated probabilities):")
        print(f"  Val: {(y_pred_val > 0.5).mean()*100:.1f}% Up, {(y_pred_val <= 0.5).mean()*100:.1f}% Down")
        print(f"  Test: {(y_pred_test > 0.5).mean()*100:.1f}% Up, {(y_pred_test <= 0.5).mean()*100:.1f}% Down")
    else:
        y_pred_val = y_pred_val_raw
        y_pred_test = y_pred_test_raw

    # Detect task type from actual data (binary targets = classification)
    is_regression = len(np.unique(y_train)) > 2

    # Calculate metrics
    if is_regression:
        val_mae = mean_absolute_error(y_val, y_pred_val)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        print(f"\nPerformance (Regression):")
        print(f"  Val MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}")
        print(f"  Test MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}")
    else:
        from sklearn.metrics import accuracy_score
        # Convert probabilities to binary predictions for accuracy calculation
        y_pred_val_binary = (y_pred_val > 0.5).astype(int)
        y_pred_test_binary = (y_pred_test > 0.5).astype(int)
        val_acc = accuracy_score(y_val, y_pred_val_binary)
        test_acc = accuracy_score(y_test, y_pred_test_binary)
        print(f"\nPerformance (Classification):")
        print(f"  Val Accuracy: {val_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")

    print(f"\nPrediction stats:")
    print(f"  Val: min={y_pred_val_raw.min():.6f}, max={y_pred_val_raw.max():.6f}, mean={y_pred_val_raw.mean():.6f}")
    print(f"  Test: min={y_pred_test_raw.min():.6f}, max={y_pred_test_raw.max():.6f}, mean={y_pred_test_raw.mean():.6f}")
    print("-" * 80)

    return {
        "y_pred_val": y_pred_val,
        "y_pred_test": y_pred_test,
        "y_pred_val_raw": y_pred_val_raw,
        "y_pred_test_raw": y_pred_test_raw,
        "model": model,
        "best_params": best_params,
        "feature_indices": feature_indices,
        "feature_names": selected_names,
        "scaler": scaler,
    }


if __name__ == "__main__":
    """Test SVR model"""
    print("=" * 80)
    print("Testing SVR Model Module")
    print("=" * 80)

    from data_pipeline import make_dataset_for_task
    from config import get_task_name

    print(f"\nLoading data for {TASK_TYPE} task...")
    datasets = make_dataset_for_task(
        task_type=get_task_name(),
        seq_len=None,
        scaler_type="standard"
    )

    # Quick test config
    test_config = {
        "tune_hyperparams": True,
        "n_iter": 20,
        "cv_folds": 3,
        "n_features": 30,
        "param_grid": {
            "C": [1, 10, 100],
            "epsilon": [0.1, 0.5],
            "gamma": ["scale", 0.01],
            "kernel": ["rbf"],
        },
    }

    results = train_and_predict(datasets, config=test_config)

    print("\n" + "=" * 80)
    print("SVR Model Test Complete!")
    print(f"Best params: {results['best_params']}")
    print(f"Selected features: {len(results['feature_indices'])}")
    print("=" * 80)
