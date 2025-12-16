"""
LightGBM Model Module

This module implements a LightGBM (Light Gradient Boosting Machine) model
following the standardized model interface.

LightGBM Features:
    - Handles 305 features efficiently (faster than XGBoost)
    - Native missing value support (addresses 31% target missingness)
    - Built-in feature importance for interpretability
    - Robust to regime changes (COVID-2020, 2022 energy crisis)
    - Leaf-wise tree growth for better accuracy

Hyperparameter tuning via Optuna Bayesian optimization with TimeSeriesSplit CV.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import warnings
import json
import os

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Hyperparameter tuning disabled.")

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

from config import TASK_TYPE, LIGHTGBM_CONFIG, RANDOM_SEED


def get_feature_importance(
    model: Any,  # lgb.LGBMModel when available
    feature_names: List[str],
    top_n: int = 20
) -> Dict[str, float]:
    """
    Get top N most important features from trained model.
    """
    importances = model.feature_importances_
    importance_dict = dict(zip(feature_names, importances))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_importance[:top_n])


def tune_lightgbm_optuna(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    is_classification: bool = False,
) -> Dict[str, Any]:
    """
    Tune LightGBM hyperparameters using Optuna Bayesian optimization.

    Returns:
        Best hyperparameters dict
    """
    if not OPTUNA_AVAILABLE:
        print("  Optuna not available, using default params")
        return {}

    n_trials = config.get("n_trials", 100)
    cv_folds = config.get("cv_folds", 5)
    param_bounds = config.get("param_bounds", {})

    print(f"  Tuning LightGBM with Optuna ({n_trials} trials)...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int(
                'n_estimators',
                param_bounds.get('n_estimators', (100, 2000))[0],
                param_bounds.get('n_estimators', (100, 2000))[1]
            ),
            'learning_rate': trial.suggest_float(
                'learning_rate',
                param_bounds.get('learning_rate', (0.01, 0.3))[0],
                param_bounds.get('learning_rate', (0.01, 0.3))[1],
                log=True
            ),
            'max_depth': trial.suggest_int(
                'max_depth',
                param_bounds.get('max_depth', (3, 12))[0],
                param_bounds.get('max_depth', (3, 12))[1]
            ),
            'num_leaves': trial.suggest_int(
                'num_leaves',
                param_bounds.get('num_leaves', (20, 300))[0],
                param_bounds.get('num_leaves', (20, 300))[1]
            ),
            'min_child_samples': trial.suggest_int(
                'min_child_samples',
                param_bounds.get('min_child_samples', (5, 100))[0],
                param_bounds.get('min_child_samples', (5, 100))[1]
            ),
            'subsample': trial.suggest_float(
                'subsample',
                param_bounds.get('subsample', (0.5, 1.0))[0],
                param_bounds.get('subsample', (0.5, 1.0))[1]
            ),
            'colsample_bytree': trial.suggest_float(
                'colsample_bytree',
                param_bounds.get('colsample_bytree', (0.5, 1.0))[0],
                param_bounds.get('colsample_bytree', (0.5, 1.0))[1]
            ),
            'reg_alpha': trial.suggest_float(
                'reg_alpha',
                param_bounds.get('reg_alpha', (1e-8, 10.0))[0],
                param_bounds.get('reg_alpha', (1e-8, 10.0))[1],
                log=True
            ),
            'reg_lambda': trial.suggest_float(
                'reg_lambda',
                param_bounds.get('reg_lambda', (1e-8, 10.0))[0],
                param_bounds.get('reg_lambda', (1e-8, 10.0))[1],
                log=True
            ),
            'random_state': RANDOM_SEED,
            'verbose': -1,
            'n_jobs': -1,
        }

        # Add GPU if configured
        if config.get('device', 'cpu') == 'gpu':
            params['device'] = 'gpu'
            params['gpu_use_dp'] = config.get('gpu_use_dp', False)

        if is_classification:
            model = lgb.LGBMClassifier(**params)
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')
            return scores.mean()
        else:
            model = lgb.LGBMRegressor(**params)
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
            return -scores.mean()  # Return positive MAE

    # Create Optuna study
    sampler = TPESampler(seed=RANDOM_SEED)
    direction = 'maximize' if is_classification else 'minimize'
    study = optuna.create_study(direction=direction, sampler=sampler)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best value: {study.best_value:.6f}")
    print(f"  Best params: {study.best_params}")

    return study.best_params


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict = None,
    is_classification: bool = False,
) -> Tuple[lgb.LGBMModel, Dict[str, Any]]:
    """
    Train LightGBM model with optional hyperparameter tuning.

    Returns:
        Tuple of (trained_model, best_params)
    """
    if config is None:
        config = LIGHTGBM_CONFIG

    if config.get("tune_hyperparams", True) and OPTUNA_AVAILABLE:
        best_params = tune_lightgbm_optuna(
            X_train, y_train, X_val, y_val, config, is_classification
        )
        # Add fixed params
        best_params['random_state'] = RANDOM_SEED
        best_params['verbose'] = config.get('verbose', -1)
        best_params['n_jobs'] = -1
    else:
        best_params = {
            'n_estimators': config.get('n_estimators', 1000),
            'learning_rate': config.get('learning_rate', 0.05),
            'max_depth': config.get('max_depth', 7),
            'num_leaves': config.get('num_leaves', 63),
            'min_child_samples': config.get('min_child_samples', 20),
            'subsample': config.get('subsample', 0.8),
            'colsample_bytree': config.get('colsample_bytree', 0.8),
            'reg_alpha': config.get('reg_alpha', 0.1),
            'reg_lambda': config.get('reg_lambda', 1.0),
            'random_state': RANDOM_SEED,
            'verbose': config.get('verbose', -1),
            'n_jobs': -1,
        }
        print(f"  Using default params")

    # Add GPU settings if specified
    device = config.get('device', 'cpu')
    if device == 'gpu':
        best_params['device'] = 'gpu'
        best_params['gpu_use_dp'] = config.get('gpu_use_dp', False)
        print(f"  Using GPU acceleration")

    # Train final model with early stopping
    early_stopping_rounds = config.get('early_stopping_rounds', 50)

    if is_classification:
        model = lgb.LGBMClassifier(**best_params)
    else:
        model = lgb.LGBMRegressor(**best_params)

    # Fit with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),  # Suppress output
        ],
    )

    print(f"  Best iteration: {model.best_iteration_}")

    return model, best_params


def train_and_predict(
    datasets: Dict[str, Any],
    config: dict = None
) -> Dict[str, Any]:
    """
    Standard model interface for LightGBM.

    Handles both classification and regression tasks.
    Performs Optuna-based hyperparameter tuning.
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM required. Install with: pip install lightgbm")

    if config is None:
        config = LIGHTGBM_CONFIG

    np.random.seed(RANDOM_SEED)

    # Extract data
    X_train = datasets["X_train"]
    y_train = datasets["y_train"]
    X_val = datasets["X_val"]
    y_val = datasets["y_val"]
    X_test = datasets["X_test"]
    y_test = datasets["y_test"]
    feature_names = datasets.get("feature_names", [f"f_{i}" for i in range(X_train.shape[-1])])

    # Handle sequential data (3D -> 2D)
    if len(X_train.shape) == 3:
        X_train = X_train[:, -1, :]
        X_val = X_val[:, -1, :]
        X_test = X_test[:, -1, :]

    # Detect task type from actual data (binary targets = classification)
    is_classification = len(np.unique(y_train)) <= 2
    task_type_detected = "classification" if is_classification else "regression"

    print("\nLightGBM Model:")
    print("-" * 80)
    print(f"  Task type: {task_type_detected}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Hyperparameter tuning: {config.get('tune_hyperparams', True)}")

    # Train model
    print("\nTraining LightGBM model...")
    model, best_params = train_lightgbm(
        X_train, y_train, X_val, y_val, config, is_classification
    )

    # Get feature importance
    importance = get_feature_importance(model, feature_names, top_n=20)
    print("\nTop 10 Feature Importances:")
    for i, (name, imp) in enumerate(list(importance.items())[:10]):
        name_short = name[:40] + "..." if len(name) > 40 else name
        print(f"  {i+1}. {name_short}: {imp:.0f}")

    # Predictions
    print("\nGenerating predictions...")
    if is_classification:
        y_pred_val_proba = model.predict_proba(X_val)[:, 1]
        y_pred_test_proba = model.predict_proba(X_test)[:, 1]
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        val_acc = accuracy_score(y_val, y_pred_val)
        test_acc = accuracy_score(y_test, y_pred_test)

        print(f"\nPerformance:")
        print(f"  Val Accuracy: {val_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"\nPrediction distribution:")
        print(f"  Val: {y_pred_val.mean()*100:.1f}% Up")
        print(f"  Test: {y_pred_test.mean()*100:.1f}% Up")

        y_pred_val_raw = y_pred_val_proba
        y_pred_test_raw = y_pred_test_proba
    else:
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        val_mae = mean_absolute_error(y_val, y_pred_val)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        print(f"\nPerformance:")
        print(f"  Val MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}")
        print(f"  Test MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}")

        y_pred_val_raw = y_pred_val
        y_pred_test_raw = y_pred_test

    print(f"\nPrediction stats:")
    print(f"  Val: min={y_pred_val_raw.min():.6f}, max={y_pred_val_raw.max():.6f}, mean={y_pred_val_raw.mean():.6f}")
    print(f"  Test: min={y_pred_test_raw.min():.6f}, max={y_pred_test_raw.max():.6f}, mean={y_pred_test_raw.mean():.6f}")
    print("-" * 80)

    return {
        "y_pred_val": y_pred_val.astype(float),
        "y_pred_test": y_pred_test.astype(float),
        "y_pred_val_raw": y_pred_val_raw,
        "y_pred_test_raw": y_pred_test_raw,
        "model": model,
        "best_params": best_params,
        "feature_importance": importance,
        "best_iteration": model.best_iteration_,
    }


def save_best_params(params: Dict[str, Any], filepath: str) -> None:
    """Save best hyperparameters to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"  Best params saved to {filepath}")


def load_best_params(filepath: str) -> Dict[str, Any]:
    """Load best hyperparameters from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    """Test LightGBM model"""
    print("=" * 80)
    print("Testing LightGBM Model Module")
    print("=" * 80)

    if not LIGHTGBM_AVAILABLE:
        print("LightGBM not installed. Skipping test.")
        exit(1)

    from data_pipeline import make_dataset_for_task
    from config import get_task_name

    print(f"\nLoading data for {TASK_TYPE} task...")
    datasets = make_dataset_for_task(
        task_type=get_task_name(),
        seq_len=None,
        scaler_type="standard"
    )

    # Quick test config (reduced trials for speed)
    test_config = {
        "tune_hyperparams": True,
        "n_trials": 10,
        "cv_folds": 3,
        "early_stopping_rounds": 20,
        "param_bounds": {
            "n_estimators": (100, 500),
            "learning_rate": (0.05, 0.2),
            "max_depth": (3, 7),
            "num_leaves": (20, 100),
            "min_child_samples": (10, 50),
            "subsample": (0.7, 1.0),
            "colsample_bytree": (0.7, 1.0),
            "reg_alpha": (0.01, 1.0),
            "reg_lambda": (0.01, 1.0),
        },
        "verbose": -1,
    }

    results = train_and_predict(datasets, config=test_config)

    print("\n" + "=" * 80)
    print("LightGBM Model Test Complete!")
    print(f"Best params: {results['best_params']}")
    print(f"Best iteration: {results['best_iteration']}")
    print(f"Top 5 features: {list(results['feature_importance'].keys())[:5]}")
    print("=" * 80)
