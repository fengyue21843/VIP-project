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

# Deep learning training parameters
MAX_EPOCHS = 100          # Maximum training epochs (increased for better convergence)
BATCH_SIZE = 32           # Batch size for neural networks (smaller for better generalization)
EARLY_STOP_PATIENCE = 20  # Patience for early stopping (increased to allow more exploration)
LEARNING_RATE = 5e-4      # Learning rate (moderate for stable convergence) 

# =============================================================================
# Model-Specific Configurations
# =============================================================================

# LSTM Configuration (optimized for better performance)
LSTM_CONFIG = {
    "layer1_units": 128,  # Balanced capacity for first layer
    "layer2_units": 64,   # Balanced capacity for second layer
    "dropout_rate": 0.4,  # Increased dropout to reduce overfitting
    "dense_units": 32,    # Moderate dense layer capacity
    "use_attention": True,  # Enable attention mechanism
    "attention_heads": 2,   # Reduced attention heads to prevent overfitting
}

# GRU Configuration (optimized to match LSTM improvements)
GRU_CONFIG = {
    "layer1_units": 128,   # Increased from 64
    "layer2_units": 64,    # Increased from 32
    "dropout_rate": 0.3,
    "dense_units": 32,     # Increased from 16
}

# Random Forest Configuration (with Optuna optimization)
RF_CONFIG = {
    "n_trials": 50,  # Number of Optuna trials for hyperparameter optimization
    "n_jobs": -1,  # Use all CPU cores
}

# XGBoost Configuration (with Optuna optimization)
XGB_CONFIG = {
    "n_trials": 50,  # Number of Optuna trials for hyperparameter optimization
    "n_jobs": -1,  # Use all CPU cores
}

# TCN Configuration (with Optuna optimization)
TCN_CONFIG = {
    "n_trials": 30,  # Number of Optuna trials for hyperparameter optimization
    "kernel_size": 3,  # Fixed kernel size for temporal convolutions
}

# =============================================================================
# Utility Functions
# =============================================================================

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
    import torch.nn as nn
    
    if TASK_TYPE == "classification":
        return nn.Sigmoid
    else:
        return None

    


def get_loss_function():
    """
    Get the appropriate loss function for PyTorch models.
    
    Returns:
        "nn.BCEWithLogitsLoss" for classification, "nn.MSELoss" for regression
    """
    import torch.nn as nn

    if TASK_TYPE == "classification":
        return nn.BCEWithLogitsLoss
    else:
        return nn.MSELoss
    


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
    import torch.nn as nn
    
    print("=" * 80)
    print("Current Configuration")
    print("=" * 80)
    print(f"\nTask Configuration:")
    print(f"  TASK_TYPE: {TASK_TYPE}")
    print(f"  Data pipeline task: {get_task_name()}")

    activation = get_output_activation()
    loss_fn = get_loss_function()
    print(f"  Output activation: {activation.__name__ if activation else 'None'}")
    print(f"  Loss function: {loss_fn.__name__}")
    print(f"  Metrics: {get_metrics()}")
  
    
    print(f"\nData Configuration:")
    print(f"  SEQUENCE_LENGTH: {SEQUENCE_LENGTH}")
    print(f"  TEST_SIZE: {TEST_SIZE}")
    print(f"  VAL_SIZE: {VAL_SIZE}")
    print(f"  SCALER_TYPE: {SCALER_TYPE}")
    
    print(f"\nTraining Configuration:")
    print(f"  RANDOM_SEED: {RANDOM_SEED}")
    print(f"  MAX_EPOCHS: {MAX_EPOCHS}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    print(f"  EARLY_STOP_PATIENCE: {EARLY_STOP_PATIENCE}")
    print(f"  LEARNING_RATE: {LEARNING_RATE}")
    
    print(f"\nModel Configurations:")
    print(f"  LSTM: {LSTM_CONFIG}")
    print(f"  GRU: {GRU_CONFIG}")
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
    print(f"  get_output_activation() = '{activation.__name__ if activation else 'None'}'")
    print(f"  get_loss_function() = '{loss_fn.__name__}'")
    print(f"  get_metrics() = {get_metrics()}")



    print("\n✓ Configuration module loaded successfully!")
    print("\nTo change task type, edit TASK_TYPE in config.py:")
    print("  TASK_TYPE = 'classification'  # for sign prediction")
    print("  TASK_TYPE = 'regression'      # for price prediction")
