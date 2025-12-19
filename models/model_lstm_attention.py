"""
LSTM Model Module

This module implements LSTM neural networks following the standardized model interface.

The LSTM architecture is designed for time-series prediction using sequences of historical data.
It can be used for both classification (direction prediction) and regression (price prediction).

Supported Architectures:
    1. Unidirectional LSTM (default)
    2. Bidirectional LSTM (bidirectional=True)

Architecture:
    - Input: (sequence_length, n_features)
    - LSTM Layer 1: Configurable units with return_sequences=True
      * Can be unidirectional or bidirectional
    - BatchNorm + Dropout
    - LSTM Layer 2: Configurable units
      * Can be unidirectional or bidirectional
    - BatchNorm + Dropout
    - Optional: Multi-head Self-Attention
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
    PyTorch LSTM Model with Unidirectional or Bidirectional support.
    
    Architecture:
    - 2-layer LSTM (unidirectional or bidirectional) with BatchNorm and Dropout
    - Optional Multi-head Self-Attention
    - Dense hidden layer with ReLU
    - Output layer (no activation, handled by loss)
    
    Config Parameters:
        - bidirectional: Enable bidirectional LSTM (default: False)
        - use_attention: Enable multi-head attention (default: False)
        - layer1_units, layer2_units: LSTM hidden units
        - dropout_rate, dense_units: Regularization and output config
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
        self.use_attention = config.get("use_attention", False)
        attention_heads = config.get("attention_heads", 4)
        self.bidirectional = config.get("bidirectional", False)  # NEW: Bidirectional option
        
        # Calculate output size multiplier for bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_size,
            self.layer1_units,
            batch_first=True,
            bidirectional=self.bidirectional  # NEW: Enable bidirectional
        )
        # BatchNorm needs to account for bidirectional output
        self.bn1 = nn.BatchNorm1d(self.layer1_units * self.num_directions)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            self.layer1_units * self.num_directions,  # Input size from layer 1
            self.layer2_units,
            batch_first=True,
            bidirectional=self.bidirectional  # NEW: Enable bidirectional
        )
        self.bn2 = nn.BatchNorm1d(self.layer2_units * self.num_directions)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Multi-head attention mechanism (optional)
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.layer2_units * self.num_directions,  # Account for bidirectional
                num_heads=attention_heads,
                dropout=dropout_rate * 0.5,
                batch_first=True
            )
            self.attention_dropout = nn.Dropout(dropout_rate)
        
        # Dense layers (input size depends on bidirectional)
        self.fc1 = nn.Linear(self.layer2_units * self.num_directions, dense_units)
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
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        
        # Apply attention mechanism if enabled
        if self.use_attention:
            # Apply multi-head attention on LSTM output
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            attn_out = self.attention_dropout(attn_out)
            
            # Extract hidden state (handle bidirectional)
            if self.bidirectional:
                # Concatenate forward and backward hidden states
                h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                h_n = h_n[-1]
            
            # Use residual connection: combine attention with LSTM hidden state
            h_n = h_n + attn_out[:, -1, :]
        else:
            # Extract hidden state (handle bidirectional)
            if self.bidirectional:
                # Concatenate forward and backward hidden states from last layer
                h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                # Use last layer's hidden state
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
    Build the LSTM architecture (unidirectional or bidirectional).
    
    This function constructs a 2-layer LSTM network with dropout and a dense layer.
    The architecture can be customized via the config parameter.
    
    Args:
        input_shape: Shape of input data (sequence_length, n_features)
                    Example: (14, 20) for 14 days of 20 features
        config: Optional configuration dict with keys:
                - layer1_units: Units in first LSTM layer (default: 128)
                - layer2_units: Units in second LSTM layer (default: 64)
                - dropout_rate: Dropout rate (default: 0.3)
                - dense_units: Units in dense hidden layer (default: 32)
                - bidirectional: Use bidirectional LSTM (default: False)
                - use_attention: Use multi-head attention (default: False)
                - attention_heads: Number of attention heads (default: 4)
    
    Returns:
        PyTorch nn.Module ready for training
    
    Examples:
        ```python
        # Unidirectional LSTM
        model = build_lstm(input_shape=(14, 20))
        
        # Bidirectional LSTM
        model = build_lstm(input_shape=(14, 20), config={'bidirectional': True})
        
        # Bidirectional LSTM with Attention
        model = build_lstm(input_shape=(14, 20), config={
            'bidirectional': True,
            'use_attention': True,
            'attention_heads': 4
        })
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
    
    This function follows the standardized interface contract:
    - Takes datasets dict from make_dataset_for_task()
    - Trains the model using standard training scheme
    - Returns predictions on validation and test sets
    
    Training Process:
    1. Extract data from datasets dict
    2. Build LSTM architecture with given configuration
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
        from data_pipeline_v2 import make_dataset_v2
        from models import model_lstm
        
        # Get sequential data for LSTM
        datasets = make_dataset_v2(task_type="sign", seq_len=14)
        
        # Train and predict
        results = model_lstm.train_and_predict(datasets)
        
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
        raise ValueError(f"LSTM requires 3D input (samples, timesteps, features), got {X_train.shape}")
    
    input_shape = (seq_len, n_features)
    
    # Build model
    print(f"Building LSTM model with input shape: {input_shape}")
    model = build_lstm(input_shape, config)
    print(f"Model architecture: {model}")
    
    # Extract training parameters from config
    if config is not None:
        max_epochs = config.get("max_epochs", MAX_EPOCHS)
        batch_size = config.get("batch_size", BATCH_SIZE)
        patience = config.get("patience", EARLY_STOP_PATIENCE)
        # Use task_type from config if provided, otherwise use parameter
        task_type_to_use = _normalize_task_type(config.get("task_type", TASK_TYPE))
    else:
        max_epochs = MAX_EPOCHS
        batch_size = BATCH_SIZE
        patience = EARLY_STOP_PATIENCE
        task_type_to_use = _normalize_task_type(TASK_TYPE)
    
    # Train model using standard training scheme
    print(f"Training LSTM for task: {task_type_to_use}")
    
    # Generate custom save path to distinguish from other LSTM variants
    from datetime import datetime
    if config is not None and config.get('save_best', False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build descriptive model name based on configuration
        model_name_parts = []
        if config.get('bidirectional', False):
            model_name_parts.append('BiLSTM')
        else:
            model_name_parts.append('LSTM')
        
        if config.get('use_attention', False):
            model_name_parts.append('Attention')
        
        model_name = '_'.join(model_name_parts)
        save_path = f"saved_models/{model_name}_{task_type_to_use}_{timestamp}"
    else:
        save_path = None
    
    model, history = standard_compile_and_train(
        model,
        X_train, y_train,
        X_val, y_val,
        task_type=task_type_to_use,  # Use the correct task type
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
        # Convert to tensors if needed
        if not isinstance(X_val, torch.Tensor):
            X_val_tensor = torch.FloatTensor(X_val).to(device)
        else:
            X_val_tensor = X_val.to(device)
        
        if not isinstance(X_test, torch.Tensor):
            X_test_tensor = torch.FloatTensor(X_test).to(device)
        else:
            X_test_tensor = X_test.to(device)
        
        # Get predictions (raw logits for classification, raw values for regression)
        y_pred_val = model(X_val_tensor).cpu().numpy().flatten()
        y_pred_test = model(X_test_tensor).cpu().numpy().flatten()
    
    # Post-processing: Apply sigmoid ONLY for classification
    if task_type_to_use == "classification":
        # Use numpy sigmoid (avoid torch conversion overhead)
        y_pred_val = 1.0 / (1.0 + np.exp(-y_pred_val))
        y_pred_test = 1.0 / (1.0 + np.exp(-y_pred_test))
    
    # Return results in standard format with task_type for consistency
    results = {
        "y_pred_val": y_pred_val,
        "y_pred_test": y_pred_test,
        "model": model,
        "history": history,
        "task_type": task_type_to_use,  # Single source of truth for evaluation
    }
    
    return results


if __name__ == "__main__":
    """
    Run LSTM model training and prediction on real data
    """
    print("=" * 80)
    print("LSTM Model - Training and Prediction")
    print("=" * 80)
    
    # Import data pipeline
    from data_pipeline_v2 import make_dataset_v2
    from config import get_task_name
    
    # Get data for LSTM (requires sequences)
    print(f"\nLoading data for {TASK_TYPE} task...")
    datasets = make_dataset_v2(
        task_type=get_task_name(),
        seq_len=14,  # LSTM needs sequences
        scaler_type="standard"
    )
    
    print(f"Data loaded:")
    print(f"  X_train shape: {datasets['X_train'].shape}")
    print(f"  X_val shape: {datasets['X_val'].shape}")
    print(f"  X_test shape: {datasets['X_test'].shape}")
    
    # Train and predict
    print("\n" + "=" * 80)
    print("Training LSTM Model")
    print("=" * 80)
    
    # Use the real LSTM configuration from config.py
    print(f"Using LSTM_CONFIG from config.py:")
    print(f"  layer1_units: {LSTM_CONFIG.get('layer1_units')}")
    print(f"  layer2_units: {LSTM_CONFIG.get('layer2_units')}")
    print(f"  dropout_rate: {LSTM_CONFIG.get('dropout_rate')}")
    print(f"  use_attention: {LSTM_CONFIG.get('use_attention')}")
    print(f"  attention_heads: {LSTM_CONFIG.get('attention_heads')}")
    print(f"  MAX_EPOCHS: {MAX_EPOCHS}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")
    
    results = train_and_predict(datasets, config=LSTM_CONFIG)
    
    # Evaluate results
    print("\n" + "=" * 80)
    print("Test Set Results")
    print("=" * 80)
    
    y_pred_test = results["y_pred_test"]
    y_test = datasets["y_test"]
    
    # Use task_type from results (ensures training/evaluation consistency)
    task_type_normalized = results["task_type"]
    
    if task_type_normalized == "classification":
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred_test > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary, zero_division=0)
        recall = recall_score(y_test, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test, y_pred_binary, zero_division=0)
        cm = confusion_matrix(y_test, y_pred_binary)
        
        print(f"\nClassification Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]:<6} FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]:<6} TP: {cm[1,1]}")
        
    else:  # regression
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        print(f"\nRegression Metrics:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  R²:   {r2:.4f}")
    
    # Training history summary
    history = results["history"]
    print(f"\nTraining Summary:")
    print(f"  Epochs trained: {len(history['loss'])}")
    print(f"  Best val_loss:  {min(history['val_loss']):.6f}")
    print(f"  Final train loss: {history['loss'][-1]:.6f}")
    
    print("\n" + "=" * 80)
    print("LSTM Training Complete!")
    print("=" * 80)
