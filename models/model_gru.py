"""
GRU Model Module

This module implements a GRU (Gated Recurrent Unit) neural network 
following the standardized model interface.

GRU is similar to LSTM but with a simpler architecture (fewer parameters),
often resulting in faster training while maintaining comparable performance.

Architecture:
    - Input: (sequence_length, n_features)
    - GRU Layer 1: Configurable units with return_sequences=True
    - Dropout 1: Configurable dropout rate
    - GRU Layer 2: Configurable units
    - Dropout 2: Configurable dropout rate
    - Dense Hidden: Configurable units with ReLU activation
    - Output: 1 unit (no activation, handled by loss function)

The model uses the standardized training scheme from training_utils module.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any

from config import TASK_TYPE, GRU_CONFIG, MAX_EPOCHS, BATCH_SIZE, EARLY_STOP_PATIENCE
from training_utils import standard_compile_and_train


class GRUModel(nn.Module):
    """
    PyTorch GRU Model.
    
    Architecture:
    - 2-layer GRU with Dropout
    - Dense hidden layer with ReLU
    - Output layer (no activation, handled by loss)
    """
    
    def __init__(self, input_size: int, config: dict | None = None):
        """
        Initialize GRU model.
        
        Args:
            input_size: Number of input features
            config: Configuration dict with architecture params
        """
        super(GRUModel, self).__init__()
        
        if config is None:
            config = GRU_CONFIG
        
        self.layer1_units = config.get("layer1_units", 64)
        self.layer2_units = config.get("layer2_units", 32)
        dropout_rate = config.get("dropout_rate", 0.3)
        dense_units = config.get("dense_units", 16)
        
        # First GRU layer
        self.gru1 = nn.GRU(
            input_size,
            self.layer1_units,
            batch_first=True,
            dropout=dropout_rate * 0.5
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second GRU layer
        self.gru2 = nn.GRU(
            self.layer1_units,
            self.layer2_units,
            batch_first=True,
            dropout=dropout_rate * 0.5
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Dense layers
        self.fc1 = nn.Linear(self.layer2_units, dense_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_units, 1)
    
    def forward(self, x):
        """Forward pass through the network."""
        # First GRU layer
        gru_out, _ = self.gru1(x)
        gru_out = self.dropout1(gru_out)
        
        # Second GRU layer
        gru_out, h_n = self.gru2(gru_out)
        # Use last hidden state
        h_n = h_n[-1]
        h_n = self.dropout2(h_n)
        
        # Dense layers
        out = self.fc1(h_n)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out


def build_gru(input_shape: tuple, config: dict | None = None) -> nn.Module:
    """
    Build the GRU architecture.
    
    This function constructs a 2-layer GRU network with dropout and a dense layer.
    GRU typically uses slightly higher dropout than LSTM due to fewer parameters.
    
    Args:
        input_shape: Shape of input data (sequence_length, n_features)
                    Example: (14, 20) for 14 days of 20 features
        config: Optional configuration dict with keys:
                - layer1_units: Units in first GRU layer (default: 64)
                - layer2_units: Units in second GRU layer (default: 32)
                - dropout_rate: Dropout rate (default: 0.3)
                - dense_units: Units in dense hidden layer (default: 16)
    
    Returns:
        PyTorch nn.Module ready for training
    
    Example:
        ```python
        model = build_gru(input_shape=(14, 20))
        print(model)
        ```
    """
    if config is None:
        config = GRU_CONFIG
    
    # Extract n_features from input_shape
    _, n_features = input_shape
    
    # Build and return model
    model = GRUModel(n_features, config)
    
    return model


def train_and_predict(
    datasets: Dict[str, Any],
    config: dict | None = None
) -> Dict[str, Any]:
    """
    Standard model interface for GRU.
    
    This function follows the standardized interface contract:
    - Takes datasets dict from make_dataset_for_task()
    - Trains the model using standard training scheme
    - Returns predictions on validation and test sets
    
    Training Process:
    1. Extract data from datasets dict
    2. Build GRU architecture with given configuration
    3. Train using standard_compile_and_train() from training_utils
    4. Generate predictions on validation and test sets
    5. Return results in standard format
    
    Args:
        datasets: dict from make_dataset_for_task() containing:
            - X_train, y_train: Training data
            - X_val, y_val: Validation data
            - X_test, y_test: Test data
            - returns_test: Raw returns for backtesting
            - scaler: Fitted scaler
            - n_features: Number of features
        
        config: Optional model-specific config dict with keys:
            - layer1_units, layer2_units, dropout_rate, dense_units (architecture)
            - max_epochs, batch_size, patience (training)
    
    Returns:
        dict containing:
            - y_pred_val: np.ndarray of predictions on validation set
            - y_pred_test: np.ndarray of predictions on test set
            - model: Trained PyTorch model object
            - history: Training history dict (for debugging/visualization)
    
    Example:
        ```python
        from data_pipeline import make_dataset_for_task
        from models import model_gru
        
        # Get sequential data for GRU
        datasets = make_dataset_for_task(task_type="sign", seq_len=14)
        
        # Train and predict
        results = model_gru.train_and_predict(datasets)
        
        # Access predictions
        y_pred_test = results["y_pred_test"]
        trained_model = results["model"]
        ```
    """
    # Extract data from datasets dict
    X_train = datasets["X_train"]
    y_train = datasets["y_train"]
    X_val = datasets["X_val"]
    y_val = datasets["y_val"]
    X_test = datasets["X_test"]
    y_test = datasets["y_test"]
    
    # Get input shape
    if len(X_train.shape) == 3:
        seq_len, n_features = X_train.shape[1], X_train.shape[2]
    else:
        raise ValueError(f"GRU requires 3D input (samples, timesteps, features), got {X_train.shape}")
    
    input_shape = (seq_len, n_features)
    
    # Build model
    print(f"Building GRU model with input shape: {input_shape}")
    model = build_gru(input_shape, config)
    print(f"Model architecture: {model}")
    
    # Extract training parameters from config
    if config is not None:
        max_epochs = config.get("max_epochs", MAX_EPOCHS)
        batch_size = config.get("batch_size", BATCH_SIZE)
        patience = config.get("patience", EARLY_STOP_PATIENCE)
    else:
        max_epochs = MAX_EPOCHS
        batch_size = BATCH_SIZE
        patience = EARLY_STOP_PATIENCE
    
    # Train model using standard training scheme
    print(f"Training GRU for task: {TASK_TYPE}")
    model, history = standard_compile_and_train(
        model,
        X_train, y_train,
        X_val, y_val,
        task_type=TASK_TYPE,
        max_epochs=max_epochs,
        batch_size=batch_size,
        patience=patience,
        verbose=1
    )
    
    # Generate predictions
    print("Generating predictions...")
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        # Convert to tensors if needed
        if not isinstance(X_val, torch.Tensor):
            X_val_tensor = torch.FloatTensor(X_val).to(device)
        else:
            X_val_tensor = X_val.to(device)
        
        if not isinstance(X_test, torch.Tensor):
            X_test_tensor = torch.FloatTensor(X_test).to(device)
        else:
            X_test_tensor = X_test.to(device)
        
        # Get predictions
        y_pred_val = model(X_val_tensor).cpu().numpy().flatten()
        y_pred_test = model(X_test_tensor).cpu().numpy().flatten()
    
    # For classification, apply sigmoid
    if TASK_TYPE == "classification":
        y_pred_val = 1 / (1 + np.exp(-y_pred_val))  # Sigmoid
        y_pred_test = 1 / (1 + np.exp(-y_pred_test))
    
    # Return results in standard format
    results = {
        "y_pred_val": y_pred_val,
        "y_pred_test": y_pred_test,
        "model": model,
        "history": history
    }
    
    return results


if __name__ == "__main__":
    """
    Test the GRU model module
    """
    print("=" * 80)
    print("Testing GRU Model Module (PyTorch)")
    print("=" * 80)
    
    # Test 1: Build model
    print("\nTest 1: Building GRU Model")
    print("-" * 80)
    try:
        model = build_gru(input_shape=(14, 20))
        print(f"✓ Model created successfully")
        print(f"  Architecture: {model}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
    
    # Test 2: Forward pass
    print("\nTest 2: Forward Pass")
    print("-" * 80)
    try:
        X_dummy = torch.randn(32, 14, 20)  # (batch, seq_len, features)
        output = model(X_dummy)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {X_dummy.shape}")
        print(f"  Output shape: {output.shape}")
        assert output.shape == (32, 1), f"Expected (32, 1), got {output.shape}"
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
    
    print("\n" + "=" * 80)
    print("GRU Model Tests Complete!")
    print("=" * 80)
