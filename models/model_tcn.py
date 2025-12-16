"""
Temporal Convolutional Network (TCN) Model Module

This module implements a TCN model following the standardized model interface.

TCN is a neural network architecture specifically designed for sequence modeling:
- Uses dilated causal convolutions for long-range dependencies
- Processes sequences in parallel (faster than RNN/LSTM)
- Maintains temporal causality (no future information leakage)
- More stable gradients than LSTM/GRU
- Often outperforms RNNs on sequence tasks

Key characteristics:
- Works with sequential data (3D input required)
- Dilated convolutions capture long-term patterns
- Residual connections for better gradient flow
- Better parallelization than recurrent models
- Fixed receptive field (controlled by dilation)

The model uses a simple architecture with 2 TCN blocks and performs
hyperparameter search over number of filters.
"""

import numpy as np
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna

from config import TASK_TYPE, MAX_EPOCHS, BATCH_SIZE, EARLY_STOP_PATIENCE, LEARNING_RATE, TCN_CONFIG
from training_utils import set_global_seed

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)




class TemporalBlock(nn.Module):
    """
    Single temporal block with dilated causal convolution.

    Architecture:
    - Dilated causal Conv1d
    - Weight normalization
    - ReLU activation
    - Dropout
    - Residual connection
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # Padding to ensure causal convolution (output length = input length)
        padding = (kernel_size - 1) * dilation

        # First conv layer
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv1 = nn.utils.weight_norm(self.conv1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second conv layer
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.utils.weight_norm(self.conv2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection (if input/output dims differ)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass with causal convolution.

        Args:
            x: (batch, channels, seq_len)
        Returns:
            out: (batch, channels, seq_len)
        """
        # First conv block
        out = self.conv1(x)
        # Remove future padding to maintain causality
        out = out[:, :, :-self.conv1.padding[0]]
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for sequence prediction.

    Architecture:
    - Multiple stacked TCN blocks with increasing dilation
    - Dense layer
    - Output layer (1 unit for regression, 2 for classification)
    """
    def __init__(self, input_size, num_filters, kernel_size, num_layers,
                 dropout, dense_units, task_type):
        super(TCNModel, self).__init__()

        self.task_type = task_type

        # Build TCN blocks with exponentially increasing dilation
        layers = []
        num_channels = [num_filters] * num_layers

        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    dilation, dropout
                )
            )

        self.tcn = nn.Sequential(*layers)

        # Dense layers
        self.dense = nn.Linear(num_filters, dense_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Output layer
        if task_type == "classification":
            self.output = nn.Linear(dense_units, 2)  # Binary classification
        else:
            self.output = nn.Linear(dense_units, 1)  # Regression

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, features)
        Returns:
            out: (batch, num_classes) or (batch, 1)
        """
        # Transpose for Conv1d: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)

        # TCN blocks
        x = self.tcn(x)

        # Take last timestep
        x = x[:, :, -1]

        # Dense layers
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output
        out = self.output(x)

        return out


def train_and_predict(
    datasets: Dict[str, Any],
    config: dict | None = None
) -> Dict[str, Any]:
    """
    Standard model interface for TCN.

    This function follows the standardized interface contract:
    - Takes datasets dict from make_dataset_for_task()
    - Trains the model with simple hyperparameter search
    - Returns predictions on validation and test sets

    Training Process:
    1. Extract data from datasets dict (must be 3D sequences)
    2. Set random seed for reproducibility
    3. Perform simple hyperparameter search on num_filters
    4. Select best model based on validation performance
    5. Generate predictions on validation and test sets
    6. Return results in standard format

    Hyperparameter Search:
    - Tests multiple values of num_filters from config
    - Evaluates each on validation set
    - Selects model with best validation performance
    - For classification: highest accuracy
    - For regression: lowest MAE

    Args:
        datasets: dict from make_dataset_for_task() containing:
            - X_train, y_train: Training data (must be 3D sequences)
            - X_val, y_val: Validation data
            - X_test, y_test: Test data
            - returns_test: Raw returns for backtesting
            - scaler: Fitted scaler
            - n_features: Number of features

        config: Optional model-specific config dict with keys:
            - num_filters_options: List of filter counts to try
            - kernel_size: Convolution kernel size
            - num_layers: Number of TCN blocks
            - dropout: Dropout rate
            - dense_units: Dense layer size

    Returns:
        dict containing:
            - y_pred_val: np.ndarray of predictions on validation set
            - y_pred_test: np.ndarray of predictions on test set
            - model: Trained PyTorch model
            - best_params: Dict of best hyperparameters found

    Example:
        ```python
        from data_pipeline import make_dataset_for_task
        from models import model_tcn

        # Get sequential data for TCN
        datasets = make_dataset_for_task(task_type="sign", seq_len=14)

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

    # Validate data shape (must be 3D for TCN)
    if len(X_train.shape) != 3:
        raise ValueError(
            f"TCN requires 3D sequential input (samples, timesteps, features). "
            f"Got shape: {X_train.shape}. "
            f"Use make_dataset_for_task() with seq_len > 0."
        )

    # Set random seed for reproducibility
    set_global_seed()

    # Use provided config or fall back to default config
    if config is None:
        config = TCN_CONFIG

    # Extract Optuna settings
    n_trials = config.get("n_trials", 30)
    kernel_size = config.get("kernel_size", 3)

    # Get input dimensions
    seq_len, n_features = X_train.shape[1], X_train.shape[2]

    print(f"\nTCN for {TASK_TYPE} task")
    print("-" * 80)
    print(f"Data shape: {X_train.shape}")
    print(f"Using Optuna for hyperparameter optimization ({n_trials} trials)")

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)

    # For classification, convert to long
    if TASK_TYPE == "classification":
        y_train_t = y_train_t.long()
        y_val_t = y_val_t.long()

    # Determine metric
    if TASK_TYPE == "classification":
        metric_name = "accuracy"
        direction = "maximize"
        criterion = nn.CrossEntropyLoss()
    else:
        metric_name = "MAE"
        direction = "minimize"
        criterion = nn.MSELoss()

    # Define Optuna objective function
    def objective(trial):
        # Suggest hyperparameters
        num_filters = trial.suggest_categorical("num_filters", [64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 2, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        dense_units = trial.suggest_categorical("dense_units", [16, 32, 64])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        # Create model
        model = TCNModel(
            input_size=n_features,
            num_filters=num_filters,
            kernel_size=kernel_size,
            num_layers=num_layers,
            dropout=dropout,
            dense_units=dense_units,
            task_type=TASK_TYPE
        )

        # Create data loader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)

                if TASK_TYPE == "classification":
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)

                if TASK_TYPE == "classification":
                    val_loss = criterion(val_outputs, y_val_t).item()
                else:
                    val_loss = criterion(val_outputs.squeeze(), y_val_t).item()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOP_PATIENCE:
                break

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_pred_t = model(X_val_t)

            if TASK_TYPE == "classification":
                val_pred = torch.argmax(val_pred_t, dim=1).numpy()
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_val, val_pred)
            else:
                val_pred = val_pred_t.squeeze().numpy()
                from sklearn.metrics import mean_absolute_error
                score = mean_absolute_error(y_val, val_pred)

        return score

    # Run Optuna optimization
    print(f"\nOptimizing hyperparameters...")
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Get best parameters
    best_params = study.best_params.copy()
    best_params["kernel_size"] = kernel_size

    print(f"\nBest hyperparameters found:")
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"  Best val {metric_name}: {study.best_value:.6f}")

    # Train final model with best parameters
    print(f"\nTraining final model with best parameters...")
    best_model = TCNModel(
        input_size=n_features,
        num_filters=best_params["num_filters"],
        kernel_size=best_params["kernel_size"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        dense_units=best_params["dense_units"],
        task_type=TASK_TYPE
    )

    # Create data loader for final training
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizer with best learning rate
    optimizer = optim.Adam(best_model.parameters(), lr=best_params["learning_rate"])

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        best_model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = best_model(batch_X)

            if TASK_TYPE == "classification":
                loss = criterion(outputs, batch_y)
            else:
                loss = criterion(outputs.squeeze(), batch_y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        best_model.eval()
        with torch.no_grad():
            val_outputs = best_model(X_val_t)

            if TASK_TYPE == "classification":
                val_loss = criterion(val_outputs, y_val_t).item()
            else:
                val_loss = criterion(val_outputs.squeeze(), y_val_t).item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{MAX_EPOCHS}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")

    # Generate predictions with best model
    print("\nGenerating predictions...")
    best_model.eval()
    with torch.no_grad():
        val_pred_t = best_model(X_val_t)
        test_pred_t = best_model(X_test_t)

        if TASK_TYPE == "classification":
            # Get probabilities for class 1
            val_pred_proba = torch.softmax(val_pred_t, dim=1)[:, 1].numpy()
            test_pred_proba = torch.softmax(test_pred_t, dim=1)[:, 1].numpy()
            y_pred_val = val_pred_proba
            y_pred_test = test_pred_proba
        else:
            y_pred_val = val_pred_t.squeeze().numpy()
            y_pred_test = test_pred_t.squeeze().numpy()

    # Return results in standard format
    results = {
        "y_pred_val": y_pred_val,
        "y_pred_test": y_pred_test,
        "model": best_model,
        "best_params": best_params,
        "optuna_study": study,  # Include study for visualization
    }

    return results


if __name__ == "__main__":
    """
    Test TCN model with data pipeline
    """
    print("=" * 80)
    print("Testing TCN Model Module")
    print("=" * 80)

    # Import data pipeline
    from data_pipeline import make_dataset_for_task
    from config import get_task_name

    # Get sequential data for TCN
    print(f"\nLoading data for {TASK_TYPE} task...")
    datasets = make_dataset_for_task(
        task_type=get_task_name(),
        seq_len=14,  # TCN needs sequential data
        scaler_type="standard"
    )

    print(f"Data loaded:")
    print(f"  X_train shape: {datasets['X_train'].shape}")
    print(f"  X_val shape: {datasets['X_val'].shape}")
    print(f"  X_test shape: {datasets['X_test'].shape}")
    print(f"  Features: {datasets['n_features']}")

    # Train and predict
    print("\n" + "=" * 80)
    print("Training TCN Model")
    print("=" * 80)

    # Use small config for quick testing
    test_config = {
        "num_filters_options": [64, 128],  # Quick test with fewer options
        "kernel_size": 3,
        "num_layers": 2,
        "dropout": 0.3,
        "dense_units": 32,
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

    print("\n" + "=" * 80)
    print("✓ TCN Model Test Complete!")
    print("=" * 80)