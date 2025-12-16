"""
Deep Learning Model Template (PyTorch)

This template is for sequence-based deep learning models such as:
- CNN (Convolutional Neural Network)
- Transformer
- Custom RNN variants
- Attention-based models

IMPORTANT: Follow these rules for fair comparison:
1. Use config.py for all hyperparameters
2. Set random seed using config.RANDOM_SEED
3. Get data from data_pipeline.make_dataset_for_task()
4. Return at least 'y_pred_test' in the required format

Author: [Your Name]
Date: [Date]
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


class CNNModel(nn.Module):
    """
    Example CNN model for time series.
    Replace with your own architecture.
    """
    def __init__(self, n_features: int, config_dict: dict | None = None):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        
        # Dense layers
        self.fc1 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        
        # Conv blocks
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.squeeze(-1)  # (batch, 32)
        
        # Dense layers
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


def build_model(input_shape: tuple, config_dict: dict | None = None) -> nn.Module:
    """
    Build your deep learning model architecture.
    
    Args:
        input_shape: Shape of input data (timesteps, features)
                    Example: (14, 116) for 14-day sequences with 116 features
        config_dict: Optional configuration dictionary
    
    Returns:
        PyTorch model
    
    TODO: Replace this example CNN with your own architecture
    """
    _, n_features = input_shape
    model = CNNModel(n_features, config_dict)
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
    
    Run this file directly:
        python models/model_template_deep_learning.py
    """
    print("="*80)
    print("Testing Deep Learning Model Template")
    print("="*80)
    
    # Import data pipeline
    from data_pipeline import make_dataset_for_task
    
    # Load data (use same settings as in run_all_models.py)
    print("\n[1/3] Loading data...")
    datasets = make_dataset_for_task(
        task_type=config.get_task_name(),
        seq_len=config.SEQUENCE_LENGTH,  # Use sequence data for deep learning
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        scaler_type=config.SCALER_TYPE
    )
    
    # Train and predict
    print("\n[2/3] Training model...")
    result = train_and_predict(datasets)
    
    # Verify results
    print("\n[3/3] Verifying results...")
    
    assert 'y_pred_test' in result, "Missing y_pred_test in result!"
    assert len(result['y_pred_test']) == len(datasets['y_test']), \
        f"Wrong shape! Expected {len(datasets['y_test'])}, got {len(result['y_pred_test'])}"
    
    print(f"\n✅ Model test passed!")
    print(f"  Test predictions shape: {result['y_pred_test'].shape}")
    print(f"  Test labels shape: {datasets['y_test'].shape}")
    
    # Print sample predictions
    print(f"\nSample predictions (first 10):")
    print(f"  Predicted: {result['y_pred_test'][:10]}")
    print(f"  Actual: {datasets['y_test'][:10]}")
    
    print("\n" + "="*80)
    print("✅ Template test completed successfully!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Customize build_model() with your architecture")
    print("  2. Test again with: python models/model_template_deep_learning.py")
    print("  3. Add to MODEL_REGISTRY in run_all_models.py")
    print("  4. Run: python run_all_models.py")
