"""
Unified Training Utilities Module

This module provides standardized training functions and utilities
that all models (LSTM, GRU, etc.) must use to ensure:
- Reproducible experiments (fixed random seeds)
- Consistent training schemes (optimizer, loss, early stopping)
- Fair model comparisons

All deep learning models should use these utilities for training.
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle
from datetime import datetime

from typing import Tuple, Literal, Dict, Any, Optional
import config


def set_global_seed(seed: int = 42) -> None:
    """
    Set global random seeds for reproducible experiments.
    Should be called at the start of each training run.
    
    This ensures that:
    - NumPy random operations are reproducible
    - Python random operations are reproducible
    - PyTorch operations are reproducible
    
    Args:
        seed: Random seed value (default: 42)
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    #Additional PyTorch determinism settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


def standard_compile_and_train(
    model: nn.Module,
    X_train: np.ndarray | torch.Tensor,
    y_train: np.ndarray | torch.Tensor,
    X_val: np.ndarray | torch.Tensor,
    y_val: np.ndarray | torch.Tensor,
    task_type: Literal["regression", "classification"] = "regression",
    max_epochs: int = None,
    batch_size: int = None,
    patience: int = None,
    learning_rate: float = None,
    verbose: int = 1,
    device: str = None,
    save_best: bool = None,
    save_path: str = None,
    min_delta: float = 1e-4,
    clip_grad_norm: float = 1.0,
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Standard training scheme for all deep models (LSTM/GRU/etc.).
    
    This function provides a unified training approach that ensures:
    - All models use the same optimizer (Adam)
    - All models use appropriate loss functions for their task
    - All models use early stopping with best weight restoration
    - All models are trained with the same random seed
    
    Training Configuration:
    - Optimizer: Adam with configurable learning rate
    - Loss:
        * regression: 'mse' (Mean Squared Error)
        * classification: 'binary_crossentropy'
    - Metrics:
        * regression: 'mae' (Mean Absolute Error)
        * classification: 'accuracy'
    - Early Stopping: Monitors val_loss with configurable patience
    - Best Weights: Automatically restored after training
    
    Args:
        model: PyTorch model(nn.Module) to be trained
        X_train: Training features (numpy or torch)
        y_train: Training targets (numpy or torch)
        X_val: Validation features (numpy or torch)
        y_val: Validation targets (numpy or torch)
        task_type: "regression" for continuous targets, "classification" for binary
        max_epochs: Maximum number of training epochs (default to config.MAX_EPOCHS)
        batch_size: Batch size for training(default to config.BATCH_SIZE)
        patience: Early stopping patience (default to config.EARLY_STOP_PATIENCE)
        learning_rate: Learning rate for Adam optimizer
        verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
        device: Device to train on ('cpu' or 'cuda')
        save_best: If True, save best model to disk (default to config.SAVE_BEST_MODEL)
        save_path: Path to save model (default: saved_models/{model_class}_{timestamp})
        min_delta: Minimum change in val_loss to qualify as improvement (default: 1e-4)
    
    Returns:
        Tuple of:
            - model: Trained model with best weights restored
            - history: Dictionary containing training metrics
    
    Example:
        ```python
        import torch
        import torch.nn as nn
        
        # Build model
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(20, 64, batch_first=True)
                self.fc = nn.Linear(64, 1)
            
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])
        
        model = MyModel()
        
        # Train with standard scheme
        model, history = standard_compile_and_train(
            model, X_train, y_train, X_val, y_val,
            task_type="regression"
        )
        ```
    """
    # Read defaults from config
    if max_epochs is None:
        max_epochs = config.MAX_EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if patience is None:
        patience = config.EARLY_STOP_PATIENCE
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE
    if save_best is None:
        save_best = getattr(config, 'SAVE_BEST_MODEL', False)
    if save_path is None and save_best:
        model_dir = getattr(config, 'MODEL_SAVE_DIR', 'saved_models')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(model_dir, f"{model.__class__.__name__}_{task_type}_{timestamp}")
    
    # Set random seed
    set_global_seed(config.RANDOM_SEED)
    
    # Device configuration
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose >= 1:
        print(f"Training on device: {device}")
    model = model.to(device)
    
    # Convert to tensors
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.FloatTensor(X_train)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.FloatTensor(y_train)
    if not isinstance(X_val, torch.Tensor):
        X_val = torch.FloatTensor(X_val)
    if not isinstance(y_val, torch.Tensor):
        y_val = torch.FloatTensor(y_val)
    
    # Ensure proper shapes
    if y_train.dim() == 1:
        y_train = y_train.reshape(-1, 1)
    if y_val.dim() == 1:
        y_val = y_val.reshape(-1, 1)
    
    # Create DataLoaders with pin_memory for faster GPU transfer
    use_cuda = torch.cuda.is_available() and (device is None or device == 'cuda')
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=0
    )
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=0
    )
    
    # Configure loss and metrics
    if task_type == "regression":
        criterion = nn.MSELoss()
        metric_name = "mae"
    elif task_type == "classification":
        criterion = nn.BCEWithLogitsLoss()
        metric_name = "accuracy"
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler_patience = max(1, patience // 2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience,
        min_lr=1e-6
    )
    
    # Training history
    history = {
        'loss': [],
        'val_loss': [],
        metric_name: [],
        f'val_{metric_name}': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_metric = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=use_cuda)
            batch_y = batch_y.to(device, non_blocking=use_cuda)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # CRITICAL FIX: Gradient clipping for training stability (prevents exploding gradients)
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            if task_type == "regression":
                train_metric += torch.abs(outputs - batch_y).mean().item() * batch_X.size(0)
            else:
                preds = (torch.sigmoid(outputs) > 0.5).float()
                train_metric += (preds == batch_y).float().mean().item() * batch_X.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_metric /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metric = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=use_cuda)
                batch_y = batch_y.to(device, non_blocking=use_cuda)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                if task_type == "regression":
                    val_metric += torch.abs(outputs - batch_y).mean().item() * batch_X.size(0)
                else:
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    val_metric += (preds == batch_y).float().mean().item() * batch_X.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_metric /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        # Record history
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history[metric_name].append(train_metric)
        history[f'val_{metric_name}'].append(val_metric)
        
        if verbose >= 1:
            print(f"Epoch {epoch+1}/{max_epochs} - loss: {train_loss:.4f} - "
                  f"{metric_name}: {train_metric:.4f} - val_loss: {val_loss:.4f} - "
                  f"val_{metric_name}: {val_metric:.4f}")
        
        # Early stopping with min_delta
        if val_loss < best_val_loss - min_delta:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {
                key: value.clone().detach().cpu()
                for key, value in model.state_dict().items()
            }
            if verbose >= 2:
                print(f"  → New best model (val_loss: {val_loss:.4f}, improvement: {improvement:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose >= 1:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose >= 1:
            print(f"Best model restored (val_loss: {best_val_loss:.4f})")
        
        # Save best model if requested
        if save_best and save_path:
            metadata = {
                'val_loss': best_val_loss,
                'epochs_trained': len(history['loss']),
                'task_type': task_type,
                'config': {
                    'max_epochs': max_epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'min_delta': min_delta,
                }
            }
            save_format = getattr(config, 'MODEL_FILE_FORMAT', 'pth')
            save_model(model, save_path, metadata=metadata, save_format=save_format)
    
    return model, history


def save_model(
    model: nn.Module,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None,
    save_format: str = "pth"
) -> None:
    """
    Save PyTorch model to disk.
    
    Args:
        model: PyTorch model to save
        filepath: Path to save the model (without extension)
        metadata: Optional dictionary containing model metadata
                 (architecture config, training history, etc.)
        save_format: 'pth' for PyTorch state_dict or 'pkl' for full pickle
    
    Example:
        >>> save_model(model, "saved_models/lstm_best", 
        ...            metadata={"accuracy": 0.85, "epoch": 42})
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    
    if save_format == "pth":
        # PyTorch standard format (recommended)
        filepath_with_ext = f"{filepath}.pth"
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'timestamp': datetime.now().isoformat(),
        }
        if metadata:
            save_dict['metadata'] = metadata
        
        torch.save(save_dict, filepath_with_ext)
        print(f"Model saved to: {filepath_with_ext}")
        
    elif save_format == "pkl":
        # Pickle format (saves entire model)
        filepath_with_ext = f"{filepath}.pkl"
        save_dict = {
            'model': model,
            'timestamp': datetime.now().isoformat(),
        }
        if metadata:
            save_dict['metadata'] = metadata
        
        with open(filepath_with_ext, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Model saved to: {filepath_with_ext}")
    else:
        raise ValueError(f"Invalid save_format: {save_format}. Must be 'pth' or 'pkl'")


def load_model(
    filepath: str,
    model_class: Optional[type] = None,
    model_args: Optional[Dict[str, Any]] = None,
    device: str = "cpu"
) -> Tuple[nn.Module, Optional[Dict[str, Any]]]:
    """
    Load PyTorch model from disk.
    
    Args:
        filepath: Path to the saved model file
        model_class: Model class for .pth files (not needed for .pkl)
        model_args: Arguments to instantiate model for .pth files
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        Tuple of (model, metadata)
    
    Example:
        >>> from models.model_lstm import LSTMModel
        >>> model, metadata = load_model(
        ...     "saved_models/lstm_best.pth",
        ...     model_class=LSTMModel,
        ...     model_args={"input_size": 26, "config": {...}}
        ... )
    """
    if filepath.endswith('.pth'):
        # Load PyTorch state dict
        checkpoint = torch.load(filepath, map_location=device)
        
        if model_class is None:
            raise ValueError("model_class is required for loading .pth files")
        if model_args is None:
            raise ValueError("model_args is required for loading .pth files")
        
        # Instantiate model
        model = model_class(**model_args)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        metadata = checkpoint.get('metadata', None)
        print(f"Model loaded from: {filepath}")
        if metadata:
            print(f"Metadata: {metadata}")
        
        return model, metadata
        
    elif filepath.endswith('.pkl'):
        # Load pickled model
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        model = checkpoint['model']
        model.to(device)
        model.eval()
        
        metadata = checkpoint.get('metadata', None)
        print(f"Model loaded from: {filepath}")
        if metadata:
            print(f"Metadata: {metadata}")
        
        return model, metadata
    else:
        raise ValueError(f"Invalid file format. Must be .pth or .pkl")


def get_training_summary(history: Dict[str, list]) -> dict:
    """
    Extract summary statistics from training history.
    
    Args:
        history: Dictionary from standard_compile_and_train()
    
    Returns:
        Dictionary containing training statistics
    """
    epochs_trained = len(history['loss'])
    val_losses = history['val_loss']
    best_epoch = np.argmin(val_losses)
    
    summary = {
        'epochs_trained': epochs_trained,
        'best_epoch': best_epoch + 1,
        'best_val_loss': val_losses[best_epoch],
        'final_train_loss': history['loss'][-1],
        'metrics': {}
    }
    
    for key in history.keys():
        if key.startswith('val_') and key != 'val_loss':
            metric_name = key[4:]
            summary['metrics'][metric_name] = history[key][best_epoch]
    
    return summary


if __name__ == "__main__":
    """
    Test the training utilities
    """
    print("=" * 80)
    print("Testing Training Utilities Module")
    print("=" * 80)
    
    # Test 1: Random seed setting
    print("\nTest 1: Random Seed Reproducibility")
    print("-" * 80)
    set_global_seed(42)
    rand1 = np.random.randn(5)
    
    set_global_seed(42)
    rand2 = np.random.randn(5)
    
    if np.allclose(rand1, rand2):
        print("✓ Random seed works correctly - same sequence produced")
        print(f"  Sample values: {rand1[:3]}")
    else:
        print("✗ Random seed not working properly")
    
    # Test 2: Standard training (regression)
    print("\nTest 2: Standard Training - Regression")
    print("-" * 80)
    
    X_train = np.random.randn(100, 10, 5)
    y_train = np.random.randn(100)
    X_val = np.random.randn(20, 10, 5)
    y_val = np.random.randn(20)
    
    class SimpleLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(5, 16, batch_first=True)
            self.fc = nn.Linear(16, 1)
        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return self.fc(h_n[-1])
    
    model = SimpleLSTM()
    try:
        model, history = standard_compile_and_train(
            model, X_train, y_train, X_val, y_val,
            task_type="regression", max_epochs=5, verbose=1
        )
        print(f"✓ Training successful")
        print(f"  Epochs: {len(history['loss'])}")
        print(f"  Final val_loss: {history['val_loss'][-1]:.6f}")
        summary = get_training_summary(history)
        print(f"  Best epoch: {summary['best_epoch']}")
    except Exception as e:
        print(f"✗ Training failed: {e}")
    
    # Test 3: Classification
    print("\nTest 3: Standard Training - Classification")
    print("-" * 80)
    y_train_cls = np.random.randint(0, 2, size=100)
    y_val_cls = np.random.randint(0, 2, size=20)
    
    model_cls = SimpleLSTM()
    try:
        model_cls, history_cls = standard_compile_and_train(
            model_cls, X_train, y_train_cls, X_val, y_val_cls,
            task_type="classification", max_epochs=5, verbose=0
        )
        print(f"✓ Classification training successful")
        print(f"  Epochs: {len(history_cls['loss'])}")
        print(f"  Final val_accuracy: {history_cls['val_accuracy'][-1]:.4f}")
    except Exception as e:
        print(f"✗ Classification training failed: {e}")
    
    print("\n" + "=" * 80)
    print("Training Utilities Tests Complete!")
    print("=" * 80)
