"""
Global Configuration Module

This module contains global settings and configuration parameters
that are shared across all models and scripts.

All models should import configuration from this module to ensure consistency.
"""
from typing import Literal

# =============================================================================
# Task Configuration
# =============================================================================

# Task type: "classification" for sign prediction, "regression" for price prediction
# Change this to switch between tasks globally
TASK_TYPE: Literal["classification", "regression"] = "classification"

# Mapping of task types to their string identifiers in data_pipeline
TASK_TYPE_MAPPING = {
    "classification": "sign",  # Binary classification: 0 (down) or 1 (up)
    "regression": "price",     # Continuous: next-day return prediction
}

# =============================================================================
# Data Configuration
# =============================================================================

# Data file path
DATA_PATH = "datasets/Data_cleaned_Dataset.csv"  # Path to the data file

# Sequence length for RNN models (LSTM, GRU)
# None = tabular data for traditional ML models (RF, etc.)
# Integer = number of time steps for sequential models
SEQUENCE_LENGTH = 14  # Use 14 days of history for RNN models

# Train/val/test split ratios
TEST_SIZE = 0.15   # 15% for test set
VAL_SIZE = 0.15    # 15% for validation set
# Remaining 70% for training

# Rolling window configuration for time series cross-validation
USE_ROLLING_WINDOW = False  # If True, use rolling window; if False, use expanding window
ROLLING_WINDOW_SIZE = 0.5   # Proportion of data for training window (e.g., 0.5 = 50%)
ROLLING_STEP_SIZE = 0.1     # Step size for moving the window (e.g., 0.1 = 10%)

# Feature scaling method
SCALER_TYPE: Literal["standard", "minmax"] = "standard"

# Model saving configuration
MODEL_SAVE_DIR = "saved_models"  # Directory to save trained models
SAVE_BEST_MODEL = True           # Automatically save best model during training
MODEL_FILE_FORMAT = "pth"        # Format: 'pth' (PyTorch) or 'pkl' (pickle)

# =============================================================================
# Training Configuration
# =============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Deep learning training parameters (shared)
MAX_EPOCHS = 100          # Maximum training epochs (increased for better convergence)

# Task-specific training parameters
# Classification: faster convergence, less regularization needed
# Regression: slower convergence, more regularization needed
TRAINING_PARAMS = {
    "classification": {
        "batch_size": 32,           # Smaller batch for better generalization
        "learning_rate": 5e-4,      # Standard learning rate
        "early_stop_patience": 20,  # Classification converges faster
        "early_stop_min_delta": 0.001,  # 0.1% accuracy improvement
    },
    "regression": {
        "batch_size": 64,           # Larger batch for more stable gradients
        "learning_rate": 1e-4,      # Smaller LR to avoid gradient explosion
        "early_stop_patience": 50,  # Regression needs more time to converge
        "early_stop_min_delta": 0.0001,  # Absolute MAE improvement
    },
}

# Legacy parameters (for backward compatibility)
BATCH_SIZE = TRAINING_PARAMS[TASK_TYPE]["batch_size"]
LEARNING_RATE = TRAINING_PARAMS[TASK_TYPE]["learning_rate"]
EARLY_STOP_PATIENCE = TRAINING_PARAMS[TASK_TYPE]["early_stop_patience"] 

# =============================================================================
# Model-Specific Configurations
# =============================================================================

# LSTM Configuration (task-specific)
# Classification: simpler decision boundary (binary), moderate capacity
# Regression: complex continuous mapping, needs more capacity and regularization
LSTM_CONFIGS = {
    "classification": {
        "layer1_units": 128,
        "layer2_units": 64,
        "dropout_rate": 0.4,
        "dense_units": 32,
        "use_attention": True,
        "attention_heads": 2,
    },
    "regression": {
        "layer1_units": 256,     # More capacity for continuous mapping
        "layer2_units": 128,     # Deeper representation
        "dropout_rate": 0.5,     # Stronger regularization
        "dense_units": 64,       # Larger output layer
        "use_attention": True,
        "attention_heads": 2,
    },
}

# Legacy parameter (for backward compatibility)
LSTM_CONFIG = LSTM_CONFIGS[TASK_TYPE]

# GRU Configuration (task-specific)
GRU_CONFIGS = {
    "classification": {
        "layer1_units": 128,
        "layer2_units": 64,
        "dropout_rate": 0.3,
        "dense_units": 32,
    },
    "regression": {
        "layer1_units": 256,     # More capacity
        "layer2_units": 128,     # Deeper representation
        "dropout_rate": 0.5,     # Stronger regularization
        "dense_units": 64,       # Larger output layer
    },
}

# Legacy parameter (for backward compatibility)
GRU_CONFIG = GRU_CONFIGS[TASK_TYPE]

# Random Forest Configuration
RF_CONFIG = {
    "n_estimators_options": [100, 200, 500],
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "n_jobs": -1,
}

# SARIMAX Configuration (Seasonal ARIMA with Exogenous Variables)
SARIMAX_CONFIG = {
    "order": (2, 1, 2),                     # (p, d, q) - AR, differencing, MA orders
    "seasonal_order": (1, 1, 1, 7),         # (P, D, Q, s) - seasonal components, s=7 for weekly
    "use_auto_arima": True,                 # Auto-select best orders using AIC
    "max_p": 5,                             # Max AR order for auto selection
    "max_q": 5,                             # Max MA order for auto selection
    "max_d": 2,                             # Max differencing order
    "max_P": 2,                             # Max seasonal AR order
    "max_Q": 2,                             # Max seasonal MA order
    "seasonal_period": 7,                   # Seasonal period (7=weekly, 30=monthly)
    "exog_cols": [                          # Exogenous variables to include
        "gas_price",
        "temperature",
        "pjm_load",
        "IS_WORKDAY",
    ],
    "max_history": 1000,                    # Sliding window for rolling forecast
    "use_fast_forecast": True,              # Fast mode: single fit, multi-step forecast (50x faster)
    "refit_interval": 50,                   # Rolling mode only: refit every N steps
}

# SVR Configuration (Support Vector Regression)
SVR_CONFIG = {
    "kernel": "rbf",                        # Kernel type: 'rbf', 'linear', 'poly'
    "C": 100,                               # Regularization parameter
    "epsilon": 0.1,                         # Epsilon in epsilon-SVR model
    "gamma": "scale",                       # Kernel coefficient: 'scale', 'auto', or float
    "tune_hyperparams": True,               # Whether to tune hyperparameters
    "cv_folds": 5,                          # TimeSeriesSplit folds
    "n_iter": 50,                           # Number of iterations for RandomizedSearchCV
    "param_grid": {                         # Hyperparameter search space
        "C": [0.1, 1, 10, 100, 1000],
        "epsilon": [0.01, 0.1, 0.5, 1.0],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "kernel": ["rbf", "poly"],
    },
    "n_features": 50,                       # Number of features to use (feature selection)
}

# LightGBM Configuration
LIGHTGBM_CONFIG = {
    "n_estimators": 1000,                   # Number of boosting rounds
    "learning_rate": 0.05,                  # Learning rate
    "max_depth": 7,                         # Max tree depth (-1 for no limit)
    "num_leaves": 63,                       # Max number of leaves per tree
    "min_child_samples": 20,                # Min samples in a leaf
    "subsample": 0.8,                       # Row subsampling ratio
    "colsample_bytree": 0.8,                # Column subsampling ratio
    "reg_alpha": 0.1,                       # L1 regularization
    "reg_lambda": 1.0,                      # L2 regularization
    "early_stopping_rounds": 50,            # Early stopping patience
    "tune_hyperparams": True,               # Whether to tune with Optuna
    "n_trials": 100,                        # Number of Optuna trials
    "cv_folds": 5,                          # TimeSeriesSplit folds
    "param_bounds": {                       # Optuna search bounds
        "n_estimators": (100, 2000),
        "learning_rate": (0.01, 0.3),
        "max_depth": (3, 12),
        "num_leaves": (20, 300),
        "min_child_samples": (5, 100),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "reg_alpha": (1e-8, 10.0),
        "reg_lambda": (1e-8, 10.0),
    },
    "verbose": -1,                          # Suppress LightGBM output
    "device": "cpu",                        # Device: "cpu" or "gpu" (requires GPU build)
    "gpu_use_dp": False,                    # Use double precision on GPU (slower but more accurate)
}

# =============================================================================
# Utility Functions
# =============================================================================

def get_training_params() -> dict:
    """
    Get task-specific training parameters.
    
    Returns:
        dict with batch_size, learning_rate, early_stop_patience, early_stop_min_delta
    
    Example:
        >>> params = get_training_params()
        >>> optimizer = Adam(model.parameters(), lr=params['learning_rate'])
    """
    return TRAINING_PARAMS[TASK_TYPE]


def get_learning_rate() -> float:
    """
    Get task-specific learning rate.
    
    Returns:
        5e-4 for classification, 1e-4 for regression
    """
    return TRAINING_PARAMS[TASK_TYPE]["learning_rate"]


def get_batch_size() -> int:
    """
    Get task-specific batch size.
    
    Returns:
        32 for classification, 64 for regression
    """
    return TRAINING_PARAMS[TASK_TYPE]["batch_size"]


def get_early_stop_patience() -> int:
    """
    Get task-specific early stop patience.
    
    Returns:
        20 for classification, 50 for regression
    """
    return TRAINING_PARAMS[TASK_TYPE]["early_stop_patience"]


def get_model_config(model_type: str) -> dict:
    """
    Get task-specific model configuration.
    
    Args:
        model_type: 'lstm' or 'gru'
    
    Returns:
        Configuration dict for the specified model and current task
    
    Example:
        >>> lstm_cfg = get_model_config('lstm')
        >>> print(lstm_cfg['layer1_units'])  # 128 for classification, 256 for regression
    """
    model_type_lower = model_type.lower()
    
    if model_type_lower == "lstm":
        return LSTM_CONFIGS[TASK_TYPE]
    elif model_type_lower == "gru":
        return GRU_CONFIGS[TASK_TYPE]
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'lstm' or 'gru'.")


def get_task_name() -> str:
    """
    Get the task name for data pipeline based on TASK_TYPE.
    
    Returns:
        "sign" for classification or "price" for regression
    """
    return TASK_TYPE_MAPPING[TASK_TYPE]


def get_output_activation():
    """
    Get the appropriate output activation for PyTorch models.
    
    Returns:
        "nn.Sigmoid" for classification, None for regression
    """
    try:
        import torch.nn as nn
        if TASK_TYPE == "classification":
            return nn.Sigmoid
        else:
            return None
    except ImportError:
        # Return string names if PyTorch not available
        if TASK_TYPE == "classification":
            return "Sigmoid"
        else:
            return None

    


def get_loss_function():
    """
    Get the appropriate loss function for PyTorch models.
    
    Returns:
        "nn.BCEWithLogitsLoss" for classification, "nn.MSELoss" for regression
    """
    try:
        import torch.nn as nn
        if TASK_TYPE == "classification":
            return nn.BCEWithLogitsLoss
        else:
            return nn.MSELoss
    except ImportError:
        # Return string names if PyTorch not available
        if TASK_TYPE == "classification":
            return "BCEWithLogitsLoss"
        else:
            return "MSELoss"
    


def get_metrics() -> list:
    """
    Get the appropriate metrics for the current task.
    
    Returns:
        ["accuracy"] for classification, ["mae"] for regression
    """
    return ["accuracy"] if TASK_TYPE == "classification" else ["mae"]


def print_config() -> None:
    """
    Print current configuration settings.
    """
    print("=" * 80)
    print("Current Configuration")
    print("=" * 80)
    print(f"\nTask Configuration:")
    print(f"  TASK_TYPE: {TASK_TYPE}")
    print(f"  Data pipeline task: {get_task_name()}")

    activation = get_output_activation()
    loss_fn = get_loss_function()
    
    # Handle both PyTorch objects and string names
    if isinstance(activation, str) or activation is None:
        activation_name = activation if activation else 'None'
    else:
        activation_name = activation.__name__
    
    if isinstance(loss_fn, str):
        loss_name = loss_fn
    else:
        loss_name = loss_fn.__name__
    
    print(f"  Output activation: {activation_name}")
    print(f"  Loss function: {loss_name}")
    print(f"  Metrics: {get_metrics()}")
  
    
    print(f"\nData Configuration:")
    print(f"  SEQUENCE_LENGTH: {SEQUENCE_LENGTH}")
    print(f"  TEST_SIZE: {TEST_SIZE}")
    print(f"  VAL_SIZE: {VAL_SIZE}")
    print(f"  SCALER_TYPE: {SCALER_TYPE}")
    
    print(f"\nTraining Configuration ({TASK_TYPE}):")
    print(f"  RANDOM_SEED: {RANDOM_SEED}")
    print(f"  MAX_EPOCHS: {MAX_EPOCHS}")
    params = get_training_params()
    print(f"  BATCH_SIZE: {params['batch_size']}")
    print(f"  LEARNING_RATE: {params['learning_rate']}")
    print(f"  EARLY_STOP_PATIENCE: {params['early_stop_patience']}")
    print(f"  EARLY_STOP_MIN_DELTA: {params['early_stop_min_delta']}")
    
    print(f"\nModel Configurations ({TASK_TYPE}):")
    print(f"  LSTM: {get_model_config('lstm')}")
    print(f"  GRU: {get_model_config('gru')}")
    print(f"  Random Forest: {RF_CONFIG}")
    print(f"  SARIMAX: {SARIMAX_CONFIG}")
    print(f"  SVR: {SVR_CONFIG}")
    print(f"  LightGBM: {LIGHTGBM_CONFIG}")
    print("=" * 80)


if __name__ == "__main__":
    """
    Test configuration module
    """
    print("Testing Configuration Module\n")
    
    # Print current configuration
    print_config()
    
    # Test utility functions
    print("\nTesting Utility Functions:")
    print(f"  get_task_name() = '{get_task_name()}'")

    activation = get_output_activation()
    loss_fn = get_loss_function()
    
    # Handle both PyTorch objects and string names
    if isinstance(activation, str) or activation is None:
        activation_name = activation if activation else 'None'
    else:
        activation_name = activation.__name__
    
    if isinstance(loss_fn, str):
        loss_name = loss_fn
    else:
        loss_name = loss_fn.__name__
    
    print(f"  get_output_activation() = '{activation_name}'")
    print(f"  get_loss_function() = '{loss_name}'")
    print(f"  get_metrics() = {get_metrics()}")
    
    print("\nTesting Task-Specific Functions:")
    print(f"  get_learning_rate() = {get_learning_rate()}")
    print(f"  get_batch_size() = {get_batch_size()}")
    print(f"  get_early_stop_patience() = {get_early_stop_patience()}")
    print(f"  get_model_config('lstm') = {get_model_config('lstm')}")
    print(f"  get_model_config('gru') = {get_model_config('gru')}")



    print("\n✓ Configuration module loaded successfully!")
    print("\nTo change task type, edit TASK_TYPE in config.py:")
    print("  TASK_TYPE = 'classification'  # for sign prediction")
    print("  TASK_TYPE = 'regression'      # for price prediction")
