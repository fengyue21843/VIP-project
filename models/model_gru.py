"""
GRU Model Module

This module implements a GRU (Gated Recurrent Unit) neural network
following the standardized model interface.

GRU is similar to LSTM but with a simpler architecture (fewer parameters),
often resulting in faster training while maintaining comparable performance.

Architecture:
    - Input: (sequence_length, n_features)
    - GRU Layer 1: Configurable units (return_sequences=True conceptually)
    - Dropout 1: Configurable dropout rate
    - GRU Layer 2: Configurable units
    - Dropout 2: Configurable dropout rate
    - Dense Hidden: Configurable units with ReLU activation
    - Output: 1 unit (no activation; handled by loss function / post-processing)

Training:
    - Uses standardized training scheme from training_utils.standard_compile_and_train
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

import config as cfg  # IMPORTANT: use cfg.TASK_TYPE dynamically at runtime to avoid naming conflict
from config import GRU_CONFIG, MAX_EPOCHS, BATCH_SIZE, EARLY_STOP_PATIENCE, LEARNING_RATE
from training_utils import standard_compile_and_train, set_global_seed

# Optuna for hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None


class GRUModel(nn.Module):
    """
    PyTorch GRU Model.

    Architecture:
    - 2 GRU blocks (each single-layer GRU + external Dropout)
    - Dense hidden layer with ReLU
    - Output layer (no activation; handled by loss / post-processing)
    """

    def __init__(self, input_size: int, model_config: dict | None = None):
        super().__init__()

        if model_config is None:
            model_config = GRU_CONFIG

        layer1_units = model_config.get("layer1_units", 64)
        layer2_units = model_config.get("layer2_units", 32)
        dropout_rate = model_config.get("dropout_rate", 0.3)
        dense_units = model_config.get("dense_units", 16)

        # Note: PyTorch GRU's internal `dropout=` only applies when num_layers > 1.
        # We apply dropout explicitly via nn.Dropout layers.
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=layer1_units, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.gru2 = nn.GRU(input_size=layer1_units, hidden_size=layer2_units, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(layer2_units, dense_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        out, _ = self.gru1(x)          # (batch, seq_len, layer1_units)
        out = self.dropout1(out)

        out, h_n = self.gru2(out)      # h_n: (1, batch, layer2_units) because num_layers=1
        h_last = h_n[-1]               # (batch, layer2_units)
        h_last = self.dropout2(h_last)

        out = self.fc1(h_last)
        out = self.relu(out)
        out = self.fc2(out)            # (batch, 1) logits/values
        return out


def build_gru(input_shape: tuple, model_config: dict | None = None) -> nn.Module:
    """
    Build the GRU architecture.

    Args:
        input_shape: (sequence_length, n_features)
        model_config: Optional dict overriding GRU_CONFIG keys.

    Returns:
        nn.Module (untrained)
    """
    if model_config is None:
        model_config = GRU_CONFIG

    _, n_features = input_shape
    return GRUModel(input_size=n_features, model_config=model_config)


def _normalize_classification_labels(y: np.ndarray) -> np.ndarray:
    """
    Ensure labels are float32 in {0,1}.
    Converts {-1,+1} -> {0,1} if needed.
    """
    y = y.astype(np.float32)
    uniq = set(np.unique(y).tolist())
    if uniq == {-1.0, 1.0} or uniq == {-1, 1}:
        y = (y > 0).astype(np.float32)
    return y


def optimize_hyperparameters(
    datasets: Dict[str, Any],
    n_trials: int = 20,
    timeout: Optional[int] = None,
    direction: str = "maximize"
) -> Dict[str, Any]:
    """
    Optimize GRU hyperparameters using Optuna.
    
    This function uses Optuna to find the best hyperparameters for the GRU model
    by trying different combinations and selecting the one with the best validation performance.
    
    Args:
        datasets: Dictionary containing train/val/test data from make_dataset_for_task()
        n_trials: Number of Optuna trials to run (default: 20)
        timeout: Maximum time in seconds for optimization (None = no timeout)
        direction: "maximize" for accuracy/F1, "minimize" for loss (default: "maximize")
    
    Returns:
        Dictionary containing:
            - best_config: Best hyperparameters found
            - best_value: Best validation metric value
            - study: Optuna study object (for further analysis)
            - trials_data: List of trial results
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is not installed. Please install it with: pip install optuna"
        )
    
    task_type = cfg.TASK_TYPE
    
    # Extract data
    X_train = datasets["X_train"]
    y_train = datasets["y_train"]
    X_val = datasets["X_val"]
    y_val = datasets["y_val"]
    
    # Normalize labels
    if task_type == "classification":
        y_train = _normalize_classification_labels(y_train)
        y_val = _normalize_classification_labels(y_val)
    else:
        y_train = np.asarray(y_train, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)
    
    seq_len, n_features = X_train.shape[1], X_train.shape[2]
    input_shape = (seq_len, n_features)
    
    def objective(trial):
        """Optuna objective function"""
        # Suggest hyperparameters
        layer1_units = trial.suggest_int("layer1_units", 32, 256, step=32)
        layer2_units = trial.suggest_int("layer2_units", 16, 128, step=16)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6, step=0.1)
        dense_units = trial.suggest_int("dense_units", 16, 64, step=16)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        
        # Build model with suggested hyperparameters
        model_config = {
            "layer1_units": layer1_units,
            "layer2_units": layer2_units,
            "dropout_rate": dropout_rate,
            "dense_units": dense_units,
        }
        
        model = build_gru(input_shape, model_config=model_config)
        
        # Train with reduced epochs for faster optimization
        max_epochs_opt = min(MAX_EPOCHS, 30)  # Limit epochs during optimization
        patience_opt = min(EARLY_STOP_PATIENCE, 10)
        
        try:
            model, history = standard_compile_and_train(
                model,
                X_train, y_train,
                X_val, y_val,
                task_type=task_type,
                max_epochs=max_epochs_opt,
                batch_size=batch_size,
                patience=patience_opt,
                learning_rate=learning_rate,
                verbose=0  # Silent during optimization
            )
            
            # Get best validation metric
            if task_type == "classification":
                # Use validation accuracy
                if "val_accuracy" in history:
                    metric_value = max(history["val_accuracy"])
                else:
                    # Fallback to validation loss (inverted for maximization)
                    metric_value = -min(history["val_loss"])
            else:
                # For regression, minimize validation loss
                metric_value = -min(history["val_loss"])
            
            return metric_value
            
        except Exception as e:
            # If training fails, return a poor score
            print(f"Trial failed: {e}")
            return float('-inf') if direction == "maximize" else float('inf')
    
    # Create study
    study = optuna.create_study(
        direction=direction,
        study_name="GRU_hyperparameter_optimization"
    )
    
    # Optimize
    print(f"Starting Optuna optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    # Extract best parameters
    best_config = {
        "layer1_units": study.best_params["layer1_units"],
        "layer2_units": study.best_params["layer2_units"],
        "dropout_rate": study.best_params["dropout_rate"],
        "dense_units": study.best_params["dense_units"],
        "learning_rate": study.best_params["learning_rate"],
        "batch_size": study.best_params["batch_size"],
    }
    
    print(f"\nBest hyperparameters found:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"Best validation metric: {study.best_value:.4f}")
    
    return {
        "best_config": best_config,
        "best_value": study.best_value,
        "study": study,
        "n_trials": len(study.trials)
    }


def train_and_predict(
    datasets: Dict[str, Any],
    config: dict | None = None,
    use_optuna: bool = False,
    optuna_trials: int = 20,
    optuna_timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    Standard model interface for GRU.

    - Extract data
    - Build model
    - Train using standard_compile_and_train
    - Predict on val/test
    - Apply sigmoid for classification (probabilities)
    
    Args:
        datasets: Dictionary containing train/val/test data
        config: Optional model configuration dict. If None and use_optuna=False, uses GRU_CONFIG
        use_optuna: If True, run Optuna hyperparameter optimization first
        optuna_trials: Number of Optuna trials (if use_optuna=True)
        optuna_timeout: Timeout in seconds for Optuna (None = no timeout)
    """
    # dynamic, correct at runtime (use cfg to avoid conflict with parameter name 'config')
    task_type = cfg.TASK_TYPE

    # Extract data
    X_train = datasets["X_train"]
    y_train = datasets["y_train"]
    X_val = datasets["X_val"]
    y_val = datasets["y_val"]
    X_test = datasets["X_test"]
    y_test = datasets["y_test"]

    # Label normalization for classification (prevents 0/0/0 metrics when labels are -1/+1)
    if task_type == "classification":
        y_train = _normalize_classification_labels(y_train)
        y_val = _normalize_classification_labels(y_val)
        y_test = _normalize_classification_labels(y_test)
    else:
        # regression: keep float32 (good for torch + losses)
        y_train = np.asarray(y_train, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)

    # Sanity check input shape
    if len(X_train.shape) != 3:
        raise ValueError(
            f"GRU requires 3D input (samples, timesteps, features), got {X_train.shape}"
        )

    seq_len, n_features = X_train.shape[1], X_train.shape[2]
    input_shape = (seq_len, n_features)

    # Run Optuna optimization if requested
    optuna_results = None
    if use_optuna:
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is not installed. Please install it with: pip install optuna"
            )
        print("=" * 80)
        print("Running Optuna Hyperparameter Optimization")
        print("=" * 80)
        optuna_results = optimize_hyperparameters(
            datasets,
            n_trials=optuna_trials,
            timeout=optuna_timeout
        )
        # Use optimized config
        if config is None:
            config = {}
        config.update(optuna_results["best_config"])
        print("=" * 80)
        print("Using optimized hyperparameters for final training")
        print("=" * 80)

    # Build model
    print(f"Building GRU model with input shape: {input_shape}")
    model = build_gru(input_shape, model_config=config)
    print(f"Model architecture: {model}")

    # Training parameters
    if config is None:
        config = {}
    
    max_epochs = config.get("max_epochs", MAX_EPOCHS)
    batch_size = config.get("batch_size", BATCH_SIZE)
    patience = config.get("patience", EARLY_STOP_PATIENCE)
    learning_rate = config.get("learning_rate", LEARNING_RATE)

    # Train
    print(f"Training GRU for task: {task_type}")
    model, history = standard_compile_and_train(
        model,
        X_train, y_train,
        X_val, y_val,
        task_type=task_type,
        max_epochs=max_epochs,
        batch_size=batch_size,
        patience=patience,
        learning_rate=learning_rate,
        verbose=1
    )

    # Predict
    print("Generating predictions...")
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=device)
        X_test_tensor = torch.as_tensor(X_test, dtype=torch.float32, device=device)

        y_pred_val = model(X_val_tensor).detach().cpu().numpy().flatten()
        y_pred_test = model(X_test_tensor).detach().cpu().numpy().flatten()

    # If classification: convert logits -> probabilities
    if task_type == "classification":
        y_pred_val = 1.0 / (1.0 + np.exp(-y_pred_val))
        y_pred_test = 1.0 / (1.0 + np.exp(-y_pred_test))

        # Optional quick sanity checks (helps debug all-zeros predictions)
        try:
            print("y_test positives:", int(y_test.sum()), "out of", len(y_test))
            print("predicted positives @0.5:", int((y_pred_test > 0.5).sum()))
            print("pred min/mean/max:", float(y_pred_test.min()), float(y_pred_test.mean()), float(y_pred_test.max()))
        except Exception:
            pass

    result = {
        "y_pred_val": y_pred_val,
        "y_pred_test": y_pred_test,
        "model": model,
        "history": history
    }
    
    # Add Optuna results if available
    if optuna_results:
        result["optuna_results"] = optuna_results
        result["best_config"] = optuna_results["best_config"]
    
    return result


if __name__ == "__main__":
    print("=" * 80)
    print("Testing GRU Model Module (PyTorch)")
    print("=" * 80)

    print("\nTest 1: Building GRU Model")
    print("-" * 80)
    try:
        m = build_gru(input_shape=(14, 20))
        print(" Model created successfully")
        print("  Architecture:", m)

        total_params = sum(p.numel() for p in m.parameters())
        trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    except Exception as e:
        print(f" Model creation failed: {e}")

    print("\nTest 2: Forward Pass")
    print("-" * 80)
    try:
        X_dummy = torch.randn(32, 14, 20)
        out = m(X_dummy)
        print("  Forward pass successful")
        print("  Input shape:", X_dummy.shape)
        print("  Output shape:", out.shape)
        assert out.shape == (32, 1), f"Expected (32, 1), got {out.shape}"
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")

    print("\n" + "=" * 80)
    print("GRU Model Tests Complete!")
    print("=" * 80)
