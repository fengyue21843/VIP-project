"""
XGBoost Model Module

This module implements an XGBoost model following the standardized model interface.

XGBoost is a gradient boosting algorithm that works with tabular data.
It uses an ensemble of decision trees to make predictions.

Key characteristics:
- Works with tabular data (no sequences needed)
- Gradient boosting with regularization
- Handles non-linear relationships well
- Provides feature importance scores
- Fast training with GPU support

The model performs a simple hyperparameter search over n_estimators and max_depth
to find the best configuration on the validation set.
"""

import numpy as np
from typing import Dict, Any
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_absolute_error

from config import TASK_TYPE, XGBOOST_CONFIG, RANDOM_SEED
from training_utils import set_global_seed


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
            - max_depth_options: List of max_depth to try
            - learning_rate: Learning rate for boosting
            - subsample: Subsample ratio of training instances
            - colsample_bytree: Subsample ratio of columns
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
        from data_pipeline_v2 import make_dataset_v2
        from models import model_xgboost

        # Get tabular data for XGBoost
        datasets = make_dataset_v2(task_type="sign", seq_len=None)

        # Train and predict
        results = model_xgboost.train_and_predict(datasets)

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
    # Extract data from datasets dict
    X_train = datasets["X_train"]
    y_train = datasets["y_train"]
    X_val = datasets["X_val"]
    y_val = datasets["y_val"]
    X_test = datasets["X_test"]

    # Validate data shape (must be 2D for XGBoost)
    if len(X_train.shape) != 2:
        raise ValueError(
            f"XGBoost requires 2D tabular input (samples, features). "
            f"Got shape: {X_train.shape}. "
            f"Use make_dataset_for_task() with seq_len=None."
        )

    # Set random seed for reproducibility
    set_global_seed()

    # Use provided config or fall back to global config
    if config is None:
        config = XGBOOST_CONFIG

    # Extract hyperparameters
    n_estimators_options = config.get("n_estimators_options", [100, 200, 500])
    max_depth_options = config.get("max_depth_options", [3, 5, 7])
    learning_rate = config.get("learning_rate", 0.1)
    subsample = config.get("subsample", 0.8)
    colsample_bytree = config.get("colsample_bytree", 0.8)
    n_jobs = config.get("n_jobs", -1)

    print(f"\nXGBoost for {TASK_TYPE} task")
    print("-" * 80)
    print(f"Data shape: {X_train.shape}")
    print(f"Hyperparameter search over:")
    print(f"  n_estimators: {n_estimators_options}")
    print(f"  max_depth: {max_depth_options}")
    print(f"Other params: learning_rate={learning_rate}, subsample={subsample}")

    # Select appropriate model class
    if TASK_TYPE == "classification":
        ModelClass = XGBClassifier
        metric_name = "accuracy"
        best_score = -float("inf")  # Higher is better for accuracy
        maximize = True
        eval_metric = 'logloss'
    else:  # regression
        ModelClass = XGBRegressor
        metric_name = "MAE"
        best_score = float("inf")  # Lower is better for MAE
        maximize = False
        eval_metric = 'rmse'

    # Perform simple hyperparameter search
    best_model = None
    best_n_estimators = None
    best_max_depth = None

    print(f"\nSearching for best hyperparameters...")
    for n_estimators in n_estimators_options:
        for max_depth in max_depth_options:
            print(f"  Training with n_estimators={n_estimators}, max_depth={max_depth}...", end=" ")

            # Create and train model
            model = ModelClass(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=RANDOM_SEED,
                n_jobs=n_jobs,
                eval_metric=eval_metric
            )

            # Train with early stopping on validation set
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Evaluate on validation set
            if TASK_TYPE == "classification":
                # Get probability predictions for classification
                val_pred_proba = model.predict_proba(X_val)[:, 1]
                val_pred = (val_pred_proba > 0.5).astype(int)
            else:
                val_pred = model.predict(X_val)

            if TASK_TYPE == "classification":
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_val, val_pred)
            else:
                score = mean_absolute_error(y_val, val_pred)

            print(f"Val {metric_name}: {score:.6f}")

            # Update best model
            if maximize:
                is_better = score > best_score
            else:
                is_better = score < best_score

            if is_better:
                best_score = score
                best_model = model
                best_n_estimators = n_estimators
                best_max_depth = max_depth

    print(f"\nBest configuration:")
    print(f"  n_estimators: {best_n_estimators}")
    print(f"  max_depth: {best_max_depth}")
    print(f"  Best val {metric_name}: {best_score:.6f}")

    # Generate predictions with best model
    print("\nGenerating predictions...")
    if TASK_TYPE == "classification":
        # Get probability predictions for classification
        y_pred_val = best_model.predict_proba(X_val)[:, 1]
        y_pred_test = best_model.predict_proba(X_test)[:, 1]
    else:
        y_pred_val = best_model.predict(X_val)
        y_pred_test = best_model.predict(X_test)

    # Get feature importance
    feature_importance = best_model.feature_importances_

    # Return results in standard format
    results = {
        "y_pred_val": y_pred_val,
        "y_pred_test": y_pred_test,
        "model": best_model,
        "best_params": {
            "n_estimators": best_n_estimators,
            "max_depth": best_max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        },
        "feature_importance": feature_importance,
    }

    return results


if __name__ == "__main__":
    """
    Test XGBoost model with data pipeline
    """
    print("=" * 80)
    print("Testing XGBoost Model Module")
    print("=" * 80)

    # Import data pipeline
    from data_pipeline_v2 import make_dataset_v2
    from config import get_task_name

    # Get data for XGBoost (tabular, no sequences)
    print(f"\nLoading data for {TASK_TYPE} task...")
    datasets = make_dataset_v2(
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
    print("XGBoost Model Test Complete!")
    print("=" * 80)
