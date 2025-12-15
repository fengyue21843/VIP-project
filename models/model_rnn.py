"""
Simple RNN Model (PyTorch)

This follows the same interface and training flow as model_template_deep_learning.py:
- build_model(input_shape, config_dict) -> nn.Module
- train_and_predict(datasets, config_dict) -> Dict[str, Any]

The architecture is a basic 1–2 layer RNN over (seq_len, features) inputs.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any

# Import global configuration
import config
from training_utils import set_global_seed, standard_compile_and_train

# Set random seeds for reproducibility
set_global_seed(config.RANDOM_SEED)


class SimpleRNNModel(nn.Module):
    """
    Base RNN model for time series.

    Input shape: (batch, seq_len, n_features)
    Output: single value/logit per sample.
    """
    def __init__(self, n_features: int, config: dict | None = None):
        super().__init__()

        # Default hyperparameters (can be overridden via config dict)
        hidden_size = 64
        num_layers = 1
        dropout_rate = 0.2
        dense_units = 32

        if config is not None:
            hidden_size = config.get("hidden_size", hidden_size)
            num_layers = config.get("num_layers", num_layers)
            dropout_rate = config.get("dropout_rate", dropout_rate)
            dense_units = config.get("dense_units", dense_units)

        self.rnn = nn.RNN(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_out, _ = self.rnn(x)      
        last_hidden = rnn_out[:, -1, :]  
        x = self.dropout(last_hidden)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                   
        return x


def build_model(input_shape: tuple, config_dict: dict | None = None) -> nn.Module:
    """
    Build the SimpleRNN model.

    Args:
        input_shape: (timesteps, features)
        config: optional hyperparameter overrides

    Returns:
        PyTorch model instance
    """
    _, n_features = input_shape
    model = SimpleRNNModel(n_features, config_dict)
    return model



def train_and_predict(
    datasets: Dict[str, Any],
    config_dict: dict | None = None
) -> Dict[str, Any]:
    """
    Standard model interface for training and prediction.
    
    This function MUST be implemented with this exact signature.
    It will be called by run_all_models.py for fair comparison.
    
    Args:
        datasets: Dictionary from make_dataset_for_task() containing:
                  - X_train, y_train: Training data
                  - X_val, y_val: Validation data
                  - X_test, y_test: Test data
                  - returns_test: Returns for backtesting
                  - scaler: Fitted scaler
                  - feature_names: Feature names
        
        config_dict: Optional config override (usually None, uses global config)
    
    Returns:
        Dictionary containing:
        - y_pred_test: Test predictions (REQUIRED)
        - y_pred_val: Validation predictions (recommended)
        - model: Trained model (recommended)
    
    Example:
        >>> from data_pipeline import make_dataset_for_task
        >>> datasets = make_dataset_for_task("sign", seq_len=14)
        >>> result = train_and_predict(datasets)
        >>> print(result['y_pred_test'].shape)
        (300,)
    """
    # ========================================================================
    # Step 1: Extract data from datasets dictionary
    # ========================================================================
    
    X_train = datasets['X_train']
    y_train = datasets['y_train']
    X_val = datasets['X_val']
    y_val = datasets['y_val']
    X_test = datasets['X_test']
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    
    # Verify data is 3D for sequence models
    assert len(X_train.shape) == 3, (
        f"Expected 3D data (samples, timesteps, features), "
        f"got shape {X_train.shape}. "
        f"Make sure seq_len is set correctly in MODEL_REGISTRY."
    )
    
    # ========================================================================
    # Step 2: Build model
    # ========================================================================
    
    input_shape = X_train.shape[1:]  # (timesteps, features)
    print(f"\nBuilding model with input shape: {input_shape}")
    
    model = build_model(input_shape, config_dict)
    print(f"Model: {model}")
    
    # ========================================================================
    # Step 3: Train model using standard training scheme
    # ========================================================================
    
    print(f"\nTraining model...")
    print(f"  Epochs: {config.MAX_EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    
    task_type = config.TASK_TYPE
    model, history = standard_compile_and_train(
        model,
        X_train, y_train,
        X_val, y_val,
        task_type=task_type,
        max_epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        patience=config.EARLY_STOP_PATIENCE,
        verbose=1
    )
    
    print(f"\nTraining completed!")
    
    # ========================================================================
    # Step 4: Make predictions
    # ========================================================================
    
    print(f"\nGenerating predictions...")
    
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
        
        # Predict on validation and test sets
        y_pred_val_raw = model(X_val_tensor).cpu().numpy().flatten()
        y_pred_test_raw = model(X_test_tensor).cpu().numpy().flatten()
    
    # Convert predictions to correct format
    if config.TASK_TYPE == "classification":
        # For classification: apply sigmoid and convert to probabilities
        y_pred_val = 1 / (1 + np.exp(-y_pred_val_raw))
        y_pred_test = 1 / (1 + np.exp(-y_pred_test_raw))
    else:
        # For regression: use raw predictions
        y_pred_val = y_pred_val_raw
        y_pred_test = y_pred_test_raw
    
    print(f"  Validation predictions: {y_pred_val.shape}")
    print(f"  Test predictions: {y_pred_test.shape}")
    
    # ========================================================================
    # Step 5: Return results in standard format
    # ========================================================================
    
    return {
        'y_pred_test': y_pred_test,   # REQUIRED: Test predictions
        'y_pred_val': y_pred_val,     # Recommended: Validation predictions
        'model': model,               # Recommended: Trained model
        'history': history            # Optional: Training history
    }


# ============================================================================
# Optional: Test your model locally
# ============================================================================

if __name__ == "__main__":
    """
    Test your model locally before adding to run_all_models.py

    Run this file (recommended, from project root):
        python -m models.model_rnn
    """
    import numpy as np
    import config
    from data_pipeline import make_dataset_for_task

    # Toggle tuning here
    USE_OPTUNA = True          # <- set False for a single normal run
    N_TRIALS = 15              # <- start small, increase later
    SEED = config.RANDOM_SEED  # use your global seed for reproducibility

    print("=" * 80)
    print("Testing RNN Deep Learning Model")
    print("=" * 80)

    # Knowing your pipeline: config.get_task_name() returns "sign" or "price"
    task_name = config.get_task_name()
    metrics_task_type = "classification" if task_name == "sign" else "regression"

    # ------------------------------------------------------------------
    # [1/3] Loading data...
    # ------------------------------------------------------------------
    print("\n[1/3] Loading data...")
    datasets = make_dataset_for_task(
        task_type=task_name,
        seq_len=config.SEQUENCE_LENGTH,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        scaler_type=config.SCALER_TYPE
    )

    # ------------------------------------------------------------------
    # [2/3] Training model... (with optional Optuna)
    # ------------------------------------------------------------------
    print("\n[2/3] Training model...")

    if USE_OPTUNA:
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is not installed. Run `pip install optuna`.")

        # Make Optuna reproducible
        sampler = optuna.samplers.TPESampler(seed=SEED)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        y_val = np.array(datasets["y_val"])

        def objective(trial):
            # --- Search space (safe, compact defaults) ---
            config_dict = {
                "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
                "num_layers": trial.suggest_int("num_layers", 1, 3),
                "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
                "dense_units": trial.suggest_categorical("dense_units", [16, 32, 64]),
            }

            result = train_and_predict(datasets, config_dict=config_dict)
            preds = np.array(result["y_pred_val"])

            if metrics_task_type == "classification":
                # assume preds are probabilities (your train_and_predict should apply sigmoid)
                labels = (preds >= 0.5).astype(int)
                acc = float(np.mean(labels == y_val))
                return -acc  # minimize negative accuracy
            else:
                # regression: MSE
                return float(np.mean((preds - y_val) ** 2))

        print(f"\nRunning Optuna tuning: N_TRIALS={N_TRIALS}  (task={metrics_task_type})\n")
        study.optimize(objective, n_trials=N_TRIALS)

        print("\n[OPTUNA] Best hyperparameters:")
        print(study.best_params)
        print(f"[OPTUNA] Best objective value: {study.best_value:.6f}")

        print("\nTraining FINAL model with Optuna best hyperparameters...")
        result = train_and_predict(datasets, config_dict=study.best_params)

    else:
        # Single normal run
        result = train_and_predict(datasets, config_dict=None)

    # ------------------------------------------------------------------
    # [3/3] Verifying results...
    # ------------------------------------------------------------------
    print("\n[3/3] Verifying results...")

    assert "y_pred_test" in result, "Missing y_pred_test in result!"
    assert len(result["y_pred_test"]) == len(datasets["y_test"]), (
        f"Wrong shape! Expected {len(datasets['y_test'])}, got {len(result['y_pred_test'])}"
    )

    print(f"\n✅ Model test passed!")
    print(f"  Test predictions shape: {result['y_pred_test'].shape}")
    print(f"  Test labels shape: {datasets['y_test'].shape}")

    print(f"\nSample predictions (first 10):")
    print(f"  Predicted: {result['y_pred_test'][:10]}")
    print(f"  Actual:    {datasets['y_test'][:10]}")

    print("\n" + "=" * 80)
    print("✅ RNN test completed successfully!")
    print("=" * 80)

