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
