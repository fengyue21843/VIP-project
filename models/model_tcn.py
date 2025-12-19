"""
Temporal Convolutional Network (TCN) Model Module

This module implements a Temporal Convolutional Network following the standardized model interface.

TCN is a deep learning architecture designed for sequence modeling that uses:
- Causal convolutions (no information leakage from future to past)
- Dilated convolutions for large receptive fields
- Residual connections for gradient flow

Key characteristics:
- Works with sequential/time-series data
- Parallelizable training (faster than RNNs)
- Long-range dependencies via dilations
- More stable gradients than RNNs

Architecture:
    - Input: (sequence_length, n_features)
    - Multiple TCN blocks with increasing dilation rates
    - Each block: Conv1d -> ReLU -> Dropout -> Conv1d -> ReLU -> Dropout + Residual
    - Global Average Pooling
    - Dense layers
    - Output: 1 unit (no activation, handled by loss function)

The model uses the standardized training scheme from training_utils module.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any

from config import TASK_TYPE, TCN_CONFIG, MAX_EPOCHS, BATCH_SIZE, EARLY_STOP_PATIENCE
from training_utils import set_global_seed, standard_compile_and_train


class TemporalBlock(nn.Module):
    """
    Temporal Block for TCN architecture.

    Contains two convolutional layers with residual connection.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout):
        """
        Initialize temporal block.

        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Size of convolutional kernel
            stride: Stride for convolution
            dilation: Dilation factor for convolution
            dropout: Dropout rate
        """
        super(TemporalBlock, self).__init__()

        # Calculate padding to maintain sequence length
        padding = (kernel_size - 1) * dilation // 2

        # First convolutional layer
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection (downsample if needed)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through temporal block."""
        # Main path
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network (TCN) Model.

    Architecture:
    - Multiple temporal blocks with increasing dilation
    - Global average pooling
    - Dense layers for prediction
    """

    def __init__(self, input_size: int, config: dict | None = None):
        """
        Initialize TCN model.

        Args:
            input_size: Number of input features
            config: Configuration dict with architecture params
        """
        super(TCNModel, self).__init__()

        if config is None:
            config = TCN_CONFIG

        # Extract configuration
        num_channels = config.get("num_channels", [64, 64, 128, 128])
        kernel_size = config.get("kernel_size", 3)
        dropout = config.get("dropout_rate", 0.2)
        dense_units = config.get("dense_units", 64)

        # Build TCN blocks
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponentially increasing dilation
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size, dropout=dropout
            ))

        self.network = nn.Sequential(*layers)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Dense layers
        self.fc1 = nn.Linear(num_channels[-1], dense_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_units, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, seq_len, features)

        Returns:
            Output predictions of shape (batch, 1)
        """
        # x shape: (batch, seq_len, features)
        # TCN expects (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Pass through TCN blocks
        y = self.network(x)

        # Global average pooling
        y = self.global_pool(y)
        y = y.squeeze(-1)  # (batch, channels)

        # Dense layers
        y = self.fc1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.fc2(y)

        return y


def build_tcn(input_shape: tuple, config: dict | None = None) -> nn.Module:
    """
    Build the TCN architecture.

    This function constructs a TCN network with multiple temporal blocks.
    The architecture can be customized via the config parameter.

    Args:
        input_shape: Shape of input data (sequence_length, n_features)
                    Example: (14, 20) for 14 days of 20 features
        config: Optional configuration dict with keys:
                - num_channels: List of channel sizes for each level
                - kernel_size: Size of convolutional kernel
                - dropout_rate: Dropout rate
                - dense_units: Units in dense hidden layer

    Returns:
        PyTorch nn.Module ready for training

    Example:
        ```python
        model = build_tcn(input_shape=(14, 20))
        print(model)
        ```
    """
    if config is None:
        config = TCN_CONFIG

    # Extract n_features from input_shape
    _, n_features = input_shape

    # Build and return model
    model = TCNModel(n_features, config)

    return model


def train_and_predict(
    datasets: Dict[str, Any],
    config: dict | None = None
) -> Dict[str, Any]:
    """
    Standard model interface for TCN.

    This function follows the standardized interface contract:
    - Takes datasets dict from make_dataset_for_task()
    - Trains the model using standard training scheme
    - Returns predictions on validation and test sets

    Training Process:
    1. Extract data from datasets dict
    2. Build TCN architecture with given configuration
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
            - num_channels: List of channel sizes for TCN blocks
            - kernel_size: Convolutional kernel size
            - dropout_rate: Dropout rate
            - dense_units: Dense layer units
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
        from models import model_tcn

        # Get sequential data for TCN
        datasets = make_dataset_v2(task_type="sign", seq_len=14)

        # Train and predict
        results = model_tcn.train_and_predict(datasets)

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
        raise ValueError(f"TCN requires 3D input (samples, timesteps, features), got {X_train.shape}")

    input_shape = (seq_len, n_features)

    # Build model
    print(f"Building TCN model with input shape: {input_shape}")
    model = build_tcn(input_shape, config)
    print(f"Model architecture:\n{model}")

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
    print(f"Training TCN for task: {TASK_TYPE}")
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

        # Get predictions (raw logits for classification, raw values for regression)
        y_pred_val = model(X_val_tensor).cpu().numpy().flatten()
        y_pred_test = model(X_test_tensor).cpu().numpy().flatten()

    # Post-processing: Apply sigmoid ONLY for classification
    if TASK_TYPE == "classification":
        # Use numpy sigmoid (avoid torch conversion overhead)
        y_pred_val = 1.0 / (1.0 + np.exp(-y_pred_val))
        y_pred_test = 1.0 / (1.0 + np.exp(-y_pred_test))

    # Return results in standard format
    results = {
        "y_pred_val": y_pred_val,
        "y_pred_test": y_pred_test,
        "model": model,
        "history": history,
    }

    return results


if __name__ == "__main__":
    """
    Run TCN model training and prediction on real data
    """
    print("=" * 80)
    print("TCN Model - Training and Prediction")
    print("=" * 80)

    # Import data pipeline
    from data_pipeline_v2 import make_dataset_v2
    from config import get_task_name

    # Get data for TCN (requires sequences)
    print(f"\nLoading data for {TASK_TYPE} task...")
    datasets = make_dataset_v2(
        task_type=get_task_name(),
        seq_len=14,  # TCN needs sequences
        scaler_type="standard"
    )

    print(f"Data loaded:")
    print(f"  X_train shape: {datasets['X_train'].shape}")
    print(f"  X_val shape: {datasets['X_val'].shape}")
    print(f"  X_test shape: {datasets['X_test'].shape}")

    # Train and predict
    print("\n" + "=" * 80)
    print("Training TCN Model")
    print("=" * 80)

    # Use the real TCN configuration from config.py
    print(f"Using TCN_CONFIG from config.py:")
    print(f"  num_channels: {TCN_CONFIG.get('num_channels')}")
    print(f"  kernel_size: {TCN_CONFIG.get('kernel_size')}")
    print(f"  dropout_rate: {TCN_CONFIG.get('dropout_rate')}")
    print(f"  dense_units: {TCN_CONFIG.get('dense_units')}")
    print(f"  MAX_EPOCHS: {MAX_EPOCHS}")
    print(f"  BATCH_SIZE: {BATCH_SIZE}")

    results = train_and_predict(datasets, config=TCN_CONFIG)

    # Evaluate results
    print("\n" + "=" * 80)
    print("Test Set Results")
    print("=" * 80)

    y_pred_test = results["y_pred_test"]
    y_test = datasets["y_test"]

    if TASK_TYPE == "classification":
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
    print("TCN Training Complete!")
    print("=" * 80)
