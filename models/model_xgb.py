"""
XGBoost Model Module

This module implements an XGBoost model following the standardized model interface.

XGBoost (eXtreme Gradient Boosting) is a powerful gradient boosting algorithm that:
- Often outperforms Random Forest on tabular data
- Uses boosting (sequential learning) instead of bagging
- Has built-in regularization to prevent overfitting
- Handles missing values automatically
- Provides feature importance scores

Key characteristics:
- Works with tabular data (no sequences needed)
- Faster training than Random Forest for same accuracy
- Better performance on structured/tabular data
- Built-in cross-validation support
- Regularization parameters (L1, L2)

The model performs a simple hyperparameter search over n_estimators and max_depth
to find the best configuration on the validation set.
"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import accuracy_score, mean_absolute_error
import optuna

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None
    XGBRegressor = None

from config import TASK_TYPE, XGB_CONFIG
from training_utils import set_global_seed

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_and_predict(
    datasets: Dict[str, Any],
    config: dict | None = None
) -> Dict[str, Any]:
    """
    Standard model interface for XGBoost.

    This function follows the standardized interface contract:
    - Takes datasets dict from make_dataset_for_task()
    - Trains the model with simple hyperparameter search
    - Returns predictions on validation and test sets

    Training Process:
    1. Extract data from datasets dict (must be tabular, not sequences)
    2. Set random seed for reproducibility
    3. Perform simple hyperparameter search on n_estimators and max_depth
    4. Select best model based on validation performance
    5. Generate predictions on validation and test sets
    6. Return results in standard format

    Hyperparameter Search:
    - Tests multiple values of n_estimators and max_depth from config
    - Evaluates each on validation set
    - Selects model with best validation performance
    - For classification: highest accuracy
    - For regression: lowest MAE

    Args:
        datasets: dict from make_dataset_for_task() containing:
            - X_train, y_train: Training data (must be 2D tabular)
            - X_val, y_val: Validation data
            - X_test, y_test: Test data
            - returns_test: Raw returns for backtesting
            - scaler: Fitted scaler
            - n_features: Number of features

        config: Optional model-specific config dict with keys:
            - n_estimators_options: List of n_estimators to try
            - max_depth_options: List of max_depth values to try
            - learning_rate: Learning rate (eta)
            - subsample: Row sampling ratio
            - colsample_bytree: Column sampling ratio
            - reg_alpha: L1 regularization
            - reg_lambda: L2 regularization
            - n_jobs: Number of parallel jobs (-1 = use all cores)

    Returns:
        dict containing:
            - y_pred_val: np.ndarray of predictions on validation set
            - y_pred_test: np.ndarray of predictions on test set
            - model: Trained XGBoost model object
            - best_params: Dict of best hyperparameters found
            - feature_importance: Array of feature importance scores

    Example:
        ```python
        from data_pipeline import make_dataset_for_task
        from models import model_xgb

        # Get tabular data for XGBoost
        datasets = make_dataset_for_task(task_type="sign", seq_len=None)

        # Train and predict
        results = model_xgb.train_and_predict(datasets)

        # Access predictions
        y_pred_test = results["y_pred_test"]
        trained_model = results["model"]

        # Check feature importance
        importance = results["feature_importance"]
        feature_names = datasets["feature_names"]
        for name, imp in zip(feature_names, importance):
            print(f"{name}: {imp:.4f}")
        ```
    """
    # Check if XGBoost is available
    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "XGBoost is not installed. Install it with: pip install xgboost"
        )

    # Extract data from datasets dict
    X_train = datasets["X_train"]
    y_train = datasets["y_train"]
    X_val = datasets["X_val"]
    y_val = datasets["y_val"]
    X_test = datasets["X_test"]

    # Handle 3D sequential data by flattening to 2D
    if len(X_train.shape) == 3:
        print(f"  Got 3D sequential data with shape {X_train.shape}")
        print(f"  Flattening to 2D tabular format for XGBoost...")

        # Flatten: (samples, timesteps, features) -> (samples, timesteps * features)
        n_samples_train, n_timesteps, n_features = X_train.shape
        X_train = X_train.reshape(n_samples_train, n_timesteps * n_features)
        X_val = X_val.reshape(X_val.shape[0], n_timesteps * n_features)
        X_test = X_test.reshape(X_test.shape[0], n_timesteps * n_features)

        print(f"  New shape: {X_train.shape}")

    # Validate data shape (must be 2D for XGBoost)
    if len(X_train.shape) != 2:
        raise ValueError(
            f"XGBoost requires 2D tabular input (samples, features). "
            f"Got shape: {X_train.shape}."
        )

    # Set random seed for reproducibility
    set_global_seed()

    # Use provided config or fall back to default config
    if config is None:
        config = XGB_CONFIG

    # Extract Optuna settings
    n_trials = config.get("n_trials", 50)
    n_jobs = config.get("n_jobs", -1)

    print(f"\nXGBoost for {TASK_TYPE} task")
    print("-" * 80)
    print(f"Data shape: {X_train.shape}")
    print(f"Using Optuna for hyperparameter optimization ({n_trials} trials)")

    # Select appropriate model class
    if TASK_TYPE == "classification":
        ModelClass = XGBClassifier
        metric_name = "accuracy"
        direction = "maximize"
    else:  # regression
        ModelClass = XGBRegressor
        metric_name = "MAE"
        direction = "minimize"

    # Define Optuna objective function
    def objective(trial):
        # Suggest hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "random_state": 42,
            "n_jobs": n_jobs,
            "verbosity": 0,
        }

        # Create and train model
        model = ModelClass(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate on validation set
        val_pred = model.predict(X_val)

        if TASK_TYPE == "classification":
            score = accuracy_score(y_val, val_pred)
        else:
            score = mean_absolute_error(y_val, val_pred)

        return score

    # Run Optuna optimization
    print(f"\nOptimizing hyperparameters...")
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Get best parameters
    best_params = study.best_params.copy()
    best_params["random_state"] = 42
    best_params["n_jobs"] = n_jobs
    best_params["verbosity"] = 0

    print(f"\nBest hyperparameters found:")
    for key, value in best_params.items():
        if key not in ["random_state", "n_jobs", "verbosity"]:
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    print(f"  Best val {metric_name}: {study.best_value:.6f}")

    # Train final model with best parameters
    print(f"\nTraining final model with best parameters...")
    best_model = ModelClass(**best_params)
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Generate predictions with best model
    print("\nGenerating predictions...")
    y_pred_val = best_model.predict(X_val)
    y_pred_test = best_model.predict(X_test)

    # For classification, also get probabilities if available
    if TASK_TYPE == "classification":
        y_pred_val_proba = best_model.predict_proba(X_val)[:, 1]
        y_pred_test_proba = best_model.predict_proba(X_test)[:, 1]

    # Get feature importance
    feature_importance = best_model.feature_importances_

    # Return results in standard format
    results = {
        "y_pred_val": y_pred_val_proba if TASK_TYPE == "classification" else y_pred_val,
        "y_pred_test": y_pred_test_proba if TASK_TYPE == "classification" else y_pred_test,
        "model": best_model,
        "best_params": best_params,
        "feature_importance": feature_importance,
        "optuna_study": study,  # Include study for visualization
    }

    return results


if __name__ == "__main__":
    """
    Test XGBoost model with data pipeline
    """
    print("=" * 80)
    print("Testing XGBoost Model Module")
    print("=" * 80)

    if not XGBOOST_AVAILABLE:
        print("\nERROR: XGBoost is not installed!")
        print("Install it with: pip install xgboost")
        exit(1)

    # Import data pipeline
    from data_pipeline import make_dataset_for_task
    from config import get_task_name

    # Get data for XGBoost (tabular, no sequences)
    print(f"\nLoading data for {TASK_TYPE} task...")
    datasets = make_dataset_for_task(
        task_type=get_task_name(),
        seq_len=None,  # XGBoost needs tabular data
        scaler_type="standard"
    )

    print(f"Data loaded:")
    print(f"  X_train shape: {datasets['X_train'].shape}")
    print(f"  X_val shape: {datasets['X_val'].shape}")
    print(f"  X_test shape: {datasets['X_test'].shape}")
    print(f"  Features: {datasets['n_features']}")

    # Train and predict
    print("\n" + "=" * 80)
    print("Training XGBoost Model")
    print("=" * 80)

    # Use small config for quick testing
    test_config = {
        "n_estimators_options": [50, 100],  # Quick test with fewer trees
        "max_depth_options": [3, 5],  # Quick test with fewer depths
        "learning_rate": 0.1,
        "n_jobs": -1,
    }

    results = train_and_predict(datasets, config=test_config)

    # Evaluate results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)

    y_pred_test = results["y_pred_test"]
    y_test = datasets["y_test"]

    if TASK_TYPE == "classification":
        from sklearn.metrics import accuracy_score, precision_score, recall_score

        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred_test > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary, zero_division=0)
        recall = recall_score(y_test, y_pred_binary, zero_division=0)

        print(f"\nClassification Metrics (Test Set):")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
    else:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)

        print(f"\nRegression Metrics (Test Set):")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R²:  {r2:.4f}")

    # Show top 10 most important features
    print(f"\nTop 10 Most Important Features:")
    feature_importance = results["feature_importance"]
    feature_names = datasets["feature_names"]

    # Sort by importance
    indices = np.argsort(feature_importance)[::-1]
    for i, idx in enumerate(indices[:10], 1):
        print(f"  {i:2d}. {feature_names[idx]:30s} {feature_importance[idx]:.6f}")

    print("\n" + "=" * 80)
    print("✓ XGBoost Model Test Complete!")
    print("=" * 80)