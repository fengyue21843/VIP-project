
"""
Random Forest Model Module

This module implements a Random Forest model following the standardized model interface.

Random Forest is a traditional machine learning model that works with tabular data.
It uses an ensemble of decision trees to make predictions.

Key characteristics:
- Works with tabular data (no sequences needed)
- Non-parametric (no assumptions about data distribution)
- Handles non-linear relationships well
- Provides feature importance scores
- Less prone to overfitting than single decision trees

The model performs a simple hyperparameter search over n_estimators
to find the best configuration on the validation set.
"""

import numpy as np
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
import optuna

from config import TASK_TYPE, RF_CONFIG
from training_utils import set_global_seed

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_and_predict(
    datasets: Dict[str, Any],
    config: dict | None = None
) -> Dict[str, Any]:
    """
    Standard model interface for Random Forest.
    
    This function follows the standardized interface contract:
    - Takes datasets dict from make_dataset_for_task()
    - Trains the model with simple hyperparameter search
    - Returns predictions on validation and test sets
    
    Training Process:
    1. Extract data from datasets dict (must be tabular, not sequences)
    2. Set random seed for reproducibility
    3. Perform simple hyperparameter search on n_estimators
    4. Select best model based on validation performance
    5. Generate predictions on validation and test sets
    6. Return results in standard format
    
    Hyperparameter Search:
    - Tests multiple values of n_estimators from config
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
            - max_depth: Maximum tree depth (None = unlimited)
            - min_samples_split: Min samples to split node
            - min_samples_leaf: Min samples in leaf node
            - n_jobs: Number of parallel jobs (-1 = use all cores)
    
    Returns:
        dict containing:
            - y_pred_val: np.ndarray of predictions on validation set
            - y_pred_test: np.ndarray of predictions on test set
            - model: Trained sklearn model object
            - best_params: Dict of best hyperparameters found
            - feature_importance: Array of feature importance scores
    
    Example:
        ```python
        from data_pipeline import make_dataset_for_task
        from models import model_rf
        
        # Get tabular data for Random Forest
        datasets = make_dataset_for_task(task_type="sign", seq_len=None)
        
        # Train and predict
        results = model_rf.train_and_predict(datasets)
        
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

    # Handle 3D sequential data by flattening to 2D
    if len(X_train.shape) == 3:
        print(f"  Got 3D sequential data with shape {X_train.shape}")
        print(f"  Flattening to 2D tabular format for Random Forest...")

        # Flatten: (samples, timesteps, features) -> (samples, timesteps * features)
        n_samples_train, n_timesteps, n_features = X_train.shape
        X_train = X_train.reshape(n_samples_train, n_timesteps * n_features)
        X_val = X_val.reshape(X_val.shape[0], n_timesteps * n_features)
        X_test = X_test.reshape(X_test.shape[0], n_timesteps * n_features)

        print(f"  New shape: {X_train.shape}")

    # Validate data shape (must be 2D for Random Forest)
    if len(X_train.shape) != 2:
        raise ValueError(
            f"Random Forest requires 2D tabular input (samples, features). "
            f"Got shape: {X_train.shape}."
        )
    
    # Set random seed for reproducibility
    set_global_seed()
    
    # Use provided config or fall back to global config
    if config is None:
        config = RF_CONFIG

    # Extract Optuna settings
    n_trials = config.get("n_trials", 50)  # Number of Optuna trials
    n_jobs = config.get("n_jobs", -1)

    print(f"\nRandom Forest for {TASK_TYPE} task")
    print("-" * 80)
    print(f"Data shape: {X_train.shape}")
    print(f"Using Optuna for hyperparameter optimization ({n_trials} trials)")

    # Select appropriate model class
    if TASK_TYPE == "classification":
        ModelClass = RandomForestClassifier
        metric_name = "accuracy"
        direction = "maximize"
    else:  # regression
        ModelClass = RandomForestRegressor
        metric_name = "MAE"
        direction = "minimize"

    # Define Optuna objective function
    def objective(trial):
        # Suggest hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 10, 50) if trial.suggest_categorical("use_max_depth", [True, False]) else None,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": 42,
            "n_jobs": n_jobs,
        }

        # Create and train model
        model = ModelClass(**params)
        model.fit(X_train, y_train)

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

    # Handle max_depth (convert use_max_depth boolean to actual value)
    if not best_params.pop("use_max_depth", False):
        best_params["max_depth"] = None

    best_params["random_state"] = 42
    best_params["n_jobs"] = n_jobs

    print(f"\nBest hyperparameters found:")
    for key, value in best_params.items():
        if key not in ["random_state", "n_jobs"]:
            print(f"  {key}: {value}")
    print(f"  Best val {metric_name}: {study.best_value:.6f}")

    # Train final model with best parameters
    print(f"\nTraining final model with best parameters...")
    best_model = ModelClass(**best_params)
    best_model.fit(X_train, y_train)
    
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
    Test Random Forest model with data pipeline
    """
    print("=" * 80)
    print("Testing Random Forest Model Module")
    print("=" * 80)
    
    # Import data pipeline
    from data_pipeline import make_dataset_for_task
    from config import get_task_name
    
    # Get data for Random Forest (tabular, no sequences)
    print(f"\nLoading data for {TASK_TYPE} task...")
    datasets = make_dataset_for_task(
        task_type=get_task_name(),
        seq_len=None,  # Random Forest needs tabular data
        scaler_type="standard"
    )
    
    print(f"Data loaded:")
    print(f"  X_train shape: {datasets['X_train'].shape}")
    print(f"  X_val shape: {datasets['X_val'].shape}")
    print(f"  X_test shape: {datasets['X_test'].shape}")
    print(f"  Features: {datasets['n_features']}")
    
    # Train and predict
    print("\n" + "=" * 80)
    print("Training Random Forest Model")
    print("=" * 80)
    
    # Use small config for quick testing
    test_config = {
        "n_estimators_options": [50, 100],  # Quick test with fewer trees
        "max_depth": None,
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
    print("✓ Random Forest Model Test Complete!")
    print("=" * 80)
