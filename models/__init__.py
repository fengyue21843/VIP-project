"""
Models Package

This package contains all model implementations following the standardized interface.

Each model module must implement:
    train_and_predict(datasets, config=None) -> dict

Standard Interface Contract:
    Input:
        datasets: dict from make_dataset_for_task() containing:
            - X_train, y_train: Training data
            - X_val, y_val: Validation data
            - X_test, y_test: Test data
            - returns_test: Raw returns for backtesting
            - scaler: Fitted scaler instance
        config: Optional model-specific configuration dict
    
    Output:
        dict containing:
            - y_pred_val: Predictions on validation set
            - y_pred_test: Predictions on test set
            - model: Trained model object

Available Models:
    - model_lstm: LSTM neural network for sequential data
    - model_gru: GRU neural network for sequential data
    - model_rf: Random Forest for tabular data
    - model_sarimax: Seasonal ARIMA with exogenous variables (natural gas, weather)
    - model_svr: Support Vector Regression with RBF kernel
    - model_lightgbm: LightGBM gradient boosting with Optuna tuning

Example Usage:
    ```python
    from data_pipeline_v2 import make_dataset_v2
    from models import model_lstm
    
    # Get data
    datasets = make_dataset_v2(task_type="sign", seq_len=14)
    
    # Train and predict
    results = model_lstm.train_and_predict(datasets)
    
    # Access results
    y_pred_test = results["y_pred_test"]
    trained_model = results["model"]
    ```
"""

__version__ = "1.0.0"

# Import all model modules for easy access
try:
    from . import model_lstm
except ImportError:
    pass

try:
    from . import model_gru
except ImportError:
    pass

try:
    from . import model_rf
except ImportError:
    pass

try:
    from . import model_sarimax
except ImportError:
    pass

try:
    from . import model_svr
except ImportError:
    pass

try:
    from . import model_lightgbm
except ImportError:
    pass

# List of available models
__all__ = [
    "model_lstm",
    "model_gru",
    "model_rf",
    "model_sarimax",
    "model_svr",
    "model_lightgbm",
]
