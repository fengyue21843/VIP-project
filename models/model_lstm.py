"""
LSTM Model Module (Unidirectional)

This module implements a standard unidirectional LSTM neural network.

Architecture:
    - Input: (sequence_length, n_features)
    - LSTM Layer 1: Configurable units with return_sequences=True
    - BatchNorm + Dropout
    - LSTM Layer 2: Configurable units
    - BatchNorm + Dropout
    - Dense Hidden: Configurable units with ReLU activation
    - Output: 1 unit (no activation, handled by loss function)

The model uses the standardized training scheme from training_utils module.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TASK_TYPE, LSTM_CONFIG, MAX_EPOCHS, BATCH_SIZE, EARLY_STOP_PATIENCE
from training_utils import standard_compile_and_train


def _normalize_task_type(task_type: str) -> str:
    """
    Normalize task type to standard format.
    
    Args:
        task_type: "sign", "price", "classification", or "regression"
    
    Returns:
        "classification" or "regression"
    
    Raises:
        ValueError: If task_type is not recognized
    """
    task_mapping = {
        "sign": "classification",
        "price": "regression",
        "classification": "classification",
        "regression": "regression",
    }
    if task_type not in task_mapping:
        raise ValueError(
            f"Unknown task_type: '{task_type}'. "
            f"Must be one of {list(task_mapping.keys())}"
        )
    return task_mapping[task_type]


class LSTMModel(nn.Module):
    """
    PyTorch Unidirectional LSTM Model.
    
    Architecture:
    - 2-layer unidirectional LSTM with BatchNorm and Dropout
    - Dense hidden layer with ReLU
    - Output layer (no activation, handled by loss)
    """
    
    def __init__(self, input_size: int, config: dict | None = None):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            config: Configuration dict with architecture params
        """
        super(LSTMModel, self).__init__()
        
        if config is None:
            config = LSTM_CONFIG
        
        self.layer1_units = config.get("layer1_units", 128)
        self.layer2_units = config.get("layer2_units", 64)
        dropout_rate = config.get("dropout_rate", 0.2)
        dense_units = config.get("dense_units", 32)
        
        # First LSTM layer (unidirectional)
        self.lstm1 = nn.LSTM(
            input_size,
            self.layer1_units,
            batch_first=True,
            bidirectional=False
        )
        self.bn1 = nn.BatchNorm1d(self.layer1_units)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second LSTM layer (unidirectional)
        self.lstm2 = nn.LSTM(
            self.layer1_units,
            self.layer2_units,
            batch_first=True,
            bidirectional=False
        )
        self.bn2 = nn.BatchNorm1d(self.layer2_units)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Dense layers
        self.fc1 = nn.Linear(self.layer2_units, dense_units)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate * 0.5)
        self.fc2 = nn.Linear(dense_units, 1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
        
        Returns:
            Output tensor of shape (batch, 1)
        """
        # First LSTM layer
        lstm_out, _ = self.lstm1(x)
        # BatchNorm expects (N, C, L), LSTM outputs (N, L, C)
        lstm_out = lstm_out.transpose(1, 2)
        lstm_out = self.bn1(lstm_out)
        lstm_out = lstm_out.transpose(1, 2)
        lstm_out = self.dropout1(lstm_out)
        
        # Second LSTM layer
        lstm_out, (h_n, _) = self.lstm2(lstm_out)
        # Use last hidden state from last layer
        h_n = h_n[-1]
        
        h_n = self.bn2(h_n)
        h_n = self.dropout2(h_n)
        
        # Dense layers
        out = self.fc1(h_n)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        
        return out


def build_lstm(input_shape: tuple, config: dict | None = None) -> nn.Module:
    """
    Build the unidirectional LSTM architecture.
    
    Args:
        input_shape: Shape of input data (sequence_length, n_features)
        config: Optional configuration dict with keys:
                - layer1_units: Units in first LSTM layer (default: 128)
                - layer2_units: Units in second LSTM layer (default: 64)
                - dropout_rate: Dropout rate (default: 0.2)
                - dense_units: Units in dense hidden layer (default: 32)
    
    Returns:
        PyTorch nn.Module ready for training
    
    Example:
        ```python
        model = build_lstm(input_shape=(14, 20))
        ```
    """
    if config is None:
        config = LSTM_CONFIG
    
    # Extract n_features from input_shape
    _, n_features = input_shape
    
    # Build and return model
    model = LSTMModel(n_features, config)
    
    return model


def train_and_predict(
    datasets: Dict[str, Any],
    config: dict | None = None
) -> Dict[str, Any]:
    """
    Standard model interface for LSTM.
    
    Args:
        datasets: dict from make_dataset_v2() containing training/validation/test data
        config: Optional model-specific config dict
    
    Returns:
        dict containing predictions, trained model, and training history
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
        raise ValueError(f"LSTM requires 3D input (samples, timesteps, features), got {X_train.shape}")
    
    input_shape = (seq_len, n_features)
    
    # Build model
    print(f"Building Unidirectional LSTM with input shape: {input_shape}")
    model = build_lstm(input_shape, config)
    
    # Extract training parameters
    if config is not None:
        max_epochs = config.get("max_epochs", MAX_EPOCHS)
        batch_size = config.get("batch_size", BATCH_SIZE)
        patience = config.get("patience", EARLY_STOP_PATIENCE)
        task_type_to_use = _normalize_task_type(config.get("task_type", TASK_TYPE))
    else:
        max_epochs = MAX_EPOCHS
        batch_size = BATCH_SIZE
        patience = EARLY_STOP_PATIENCE
        task_type_to_use = _normalize_task_type(TASK_TYPE)
    
    # Train model
    print(f"Training Unidirectional LSTM for task: {task_type_to_use}")
    
    # Generate custom save path to distinguish from other LSTM variants
    from datetime import datetime
    if config is not None and config.get('save_best', False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"saved_models/LSTM_{task_type_to_use}_{timestamp}"
    else:
        save_path = None
    
    model, history = standard_compile_and_train(
        model,
        X_train, y_train,
        X_val, y_val,
        task_type=task_type_to_use,
        max_epochs=max_epochs,
        batch_size=batch_size,
        patience=patience,
        save_path=save_path,
        verbose=1
    )
    
    # Generate predictions
    print("Generating predictions...")
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).to(device) if not isinstance(X_val, torch.Tensor) else X_val.to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device) if not isinstance(X_test, torch.Tensor) else X_test.to(device)
        
        y_pred_val = model(X_val_tensor).cpu().numpy().flatten()
        y_pred_test = model(X_test_tensor).cpu().numpy().flatten()
    
    # Post-processing for classification
    if task_type_to_use == "classification":
        y_pred_val = 1.0 / (1.0 + np.exp(-y_pred_val))
        y_pred_test = 1.0 / (1.0 + np.exp(-y_pred_test))
    
    return {
        "y_pred_val": y_pred_val,
        "y_pred_test": y_pred_test,
        "model": model,
        "history": history,
        "task_type": task_type_to_use,
    }


if __name__ == "__main__":
    """Run LSTM model training"""
    print("=" * 80)
    print("Unidirectional LSTM Model - Training and Prediction")
    print("=" * 80)
    
    from data_pipeline_v2 import make_dataset_v2
    from config import get_task_name
    
    print(f"\nLoading data for {TASK_TYPE} task...")
    datasets = make_dataset_v2(task_type=get_task_name(), seq_len=14)
    
    print(f"\nData shapes:")
    print(f"  X_train: {datasets['X_train'].shape}")
    print(f"  X_val: {datasets['X_val'].shape}")
    print(f"  X_test: {datasets['X_test'].shape}")
    
    print("\n" + "=" * 80)
    print("Training Unidirectional LSTM")
    print("=" * 80)
    
    results = train_and_predict(datasets, config=LSTM_CONFIG)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Val predictions shape: {results['y_pred_val'].shape}")
    print(f"Test predictions shape: {results['y_pred_test'].shape}")
    print("=" * 80)
