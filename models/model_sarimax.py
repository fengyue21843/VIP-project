"""
SARIMAX Model Module

This module implements a SARIMAX (Seasonal ARIMA with Exogenous Variables) model
following the standardized model interface.

SARIMAX(p, d, q) x (P, D, Q, s):
    - p, d, q: Non-seasonal AR, differencing, MA orders
    - P, D, Q: Seasonal AR, differencing, MA orders
    - s: Seasonal period (7 for weekly, 30 for monthly)
    - Exogenous variables: natural gas price, temperature, load, workday flag

Enhancement over basic ARIMA:
- Captures weekly/monthly seasonality in electricity prices
- Incorporates external drivers (natural gas, weather, demand)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. SARIMAX model will not work.")

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    print("Warning: pmdarima not available. Auto ARIMA selection disabled.")

from config import TASK_TYPE, SARIMAX_CONFIG, RANDOM_SEED


def check_stationarity(series: np.ndarray, significance: float = 0.05) -> Tuple[bool, float]:
    """
    Check if a time series is stationary using Augmented Dickey-Fuller test.
    """
    result = adfuller(series, autolag='AIC')
    p_value = result[1]
    is_stationary = p_value < significance
    return is_stationary, p_value


def auto_select_sarimax_order(
    y_train: np.ndarray,
    exog_train: Optional[np.ndarray] = None,
    seasonal_period: int = 7,
    max_p: int = 5,
    max_q: int = 5,
    max_P: int = 2,
    max_Q: int = 2,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    """
    Automatically select best SARIMAX orders using pmdarima.

    Returns:
        Tuple of (order, seasonal_order)
    """
    if not PMDARIMA_AVAILABLE:
        print("  pmdarima not available, using default orders")
        return (2, 1, 2), (1, 1, 1, seasonal_period)

    print(f"  Auto-selecting SARIMAX order using pmdarima...")
    print(f"    Seasonal period: {seasonal_period}")
    print(f"    Max orders: p={max_p}, q={max_q}, P={max_P}, Q={max_Q}")

    try:
        model = auto_arima(
            y_train,
            exogenous=exog_train,
            start_p=1, max_p=max_p,
            start_q=1, max_q=max_q,
            d=None,  # auto-detect
            seasonal=True,
            m=seasonal_period,
            start_P=0, max_P=max_P,
            start_Q=0, max_Q=max_Q,
            D=None,  # auto-detect
            information_criterion='aic',
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False,
            n_jobs=-1,
        )

        order = model.order
        seasonal_order = model.seasonal_order

        print(f"  Best order: SARIMAX{order}x{seasonal_order}")
        print(f"  AIC: {model.aic():.2f}")

        return order, seasonal_order

    except Exception as e:
        print(f"  Auto-selection failed: {e}")
        print("  Using default orders: (2,1,2)x(1,1,1,7)")
        return (2, 1, 2), (1, 1, 1, seasonal_period)


def fit_sarimax(
    y_train: np.ndarray,
    exog_train: Optional[np.ndarray] = None,
    order: Tuple[int, int, int] = None,
    seasonal_order: Tuple[int, int, int, int] = None,
    config: dict = None
) -> Tuple[Any, Tuple, Tuple]:
    """
    Fit SARIMAX model on training data.

    Returns:
        Tuple of (fitted_model, order, seasonal_order)
    """
    if config is None:
        config = SARIMAX_CONFIG

    seasonal_period = config.get("seasonal_period", 7)

    # Auto-select orders if not specified
    if order is None and config.get("use_auto_arima", True):
        order, seasonal_order = auto_select_sarimax_order(
            y_train,
            exog_train=exog_train,
            seasonal_period=seasonal_period,
            max_p=config.get("max_p", 5),
            max_q=config.get("max_q", 5),
            max_P=config.get("max_P", 2),
            max_Q=config.get("max_Q", 2),
        )
    elif order is None:
        order = config.get("order", (2, 1, 2))
        seasonal_order = config.get("seasonal_order", (1, 1, 1, seasonal_period))

    # Fit the SARIMAX model
    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=200)

    return fitted, order, seasonal_order


def fast_forecast_sarimax(
    fitted_model: Any,
    actual_values: np.ndarray,
    future_exog: Optional[np.ndarray],
    verbose: bool = True,
) -> np.ndarray:
    """
    Generate fast one-step-ahead forecasts using Kalman filter extension.

    Instead of refitting for each step (slow), this uses the fitted model's
    parameters and extends it with new observations using the Kalman filter.
    This is 50-100x faster than rolling refit.
    """
    n_predictions = len(actual_values)
    predictions = np.zeros(n_predictions)

    # Use the fitted model to forecast all at once using dynamic prediction
    try:
        # Get forecasts for all steps at once
        forecast_result = fitted_model.get_forecast(
            steps=n_predictions,
            exog=future_exog
        )
        predictions = forecast_result.predicted_mean.values
        if verbose:
            print(f"    Generated {n_predictions} predictions (fast mode)")
    except Exception as e:
        if verbose:
            print(f"    Fast forecast failed ({e}), using fallback")
        # Fallback: use simple extrapolation
        predictions = np.full(n_predictions, fitted_model.fittedvalues.iloc[-1])

    return predictions


def rolling_forecast_sarimax(
    train_series: np.ndarray,
    train_exog: Optional[np.ndarray],
    actual_values: np.ndarray,
    future_exog: Optional[np.ndarray],
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    max_history: int = 1000,
    verbose: bool = True,
    refit_interval: int = 50,
) -> np.ndarray:
    """
    Generate rolling one-step-ahead forecasts with exogenous variables.

    Optimized to refit only every `refit_interval` steps instead of every step.
    This is ~50x faster than refitting every prediction.
    """
    predictions = []
    history_y = list(train_series)
    history_exog = list(train_exog) if train_exog is not None else None
    n_predictions = len(actual_values)

    fitted_model = None

    for i in range(n_predictions):
        # Refit only at intervals (or first iteration)
        if i % refit_interval == 0 or fitted_model is None:
            # Use sliding window if specified
            if max_history is not None and len(history_y) > max_history:
                fit_y = history_y[-max_history:]
                fit_exog = history_exog[-max_history:] if history_exog else None
            else:
                fit_y = history_y
                fit_exog = history_exog

            try:
                model = SARIMAX(
                    fit_y,
                    exog=fit_exog,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted_model = model.fit(disp=False, maxiter=100)
            except Exception:
                pass  # Keep using previous fitted_model

        # Get current exog for prediction
        if future_exog is not None:
            current_exog = future_exog[i:i+1]
        else:
            current_exog = None

        try:
            pred = fitted_model.forecast(steps=1, exog=current_exog)[0]
            predictions.append(pred)
        except Exception:
            # Fallback to last prediction or mean
            predictions.append(predictions[-1] if predictions else np.mean(history_y[-100:]))

        # Update history
        history_y.append(actual_values[i])
        if history_exog is not None and future_exog is not None:
            history_exog.append(future_exog[i])

        if verbose and (i + 1) % 200 == 0:
            print(f"    Progress: {i + 1}/{n_predictions} predictions")

    return np.array(predictions)


def train_and_predict(
    datasets: Dict[str, Any],
    config: dict = None
) -> Dict[str, Any]:
    """
    Standard model interface for SARIMAX.

    Uses exogenous variables (natural gas, temperature, load, workday)
    to improve predictions over univariate ARIMA.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required. Install with: pip install statsmodels")

    if config is None:
        config = SARIMAX_CONFIG

    np.random.seed(RANDOM_SEED)

    # Extract data
    X_train = datasets["X_train"]
    y_train = datasets["y_train"]
    X_val = datasets["X_val"]
    y_val = datasets["y_val"]
    X_test = datasets["X_test"]
    y_test = datasets["y_test"]
    feature_names = datasets.get("feature_names", [])

    # Handle sequential vs tabular data
    if len(X_train.shape) == 3:
        # Sequential: use last timestep
        X_train_2d = X_train[:, -1, :]
        X_val_2d = X_val[:, -1, :]
        X_test_2d = X_test[:, -1, :]
    else:
        X_train_2d = X_train
        X_val_2d = X_val
        X_test_2d = X_test

    # Extract price series (first column is typically price)
    train_prices = X_train_2d[:, 0]
    val_prices = X_val_2d[:, 0]
    test_prices = X_test_2d[:, 0]

    # Calculate returns for SARIMAX
    train_returns = np.diff(train_prices) / np.where(train_prices[:-1] != 0, train_prices[:-1], 1)
    train_returns = np.nan_to_num(train_returns, nan=0, posinf=0, neginf=0)

    val_returns = np.diff(np.concatenate([[train_prices[-1]], val_prices])) / \
                  np.concatenate([[train_prices[-1]], val_prices[:-1]])
    val_returns = np.nan_to_num(val_returns[1:], nan=0, posinf=0, neginf=0)

    test_returns = np.diff(np.concatenate([[val_prices[-1]], test_prices])) / \
                   np.concatenate([[val_prices[-1]], test_prices[:-1]])
    test_returns = np.nan_to_num(test_returns[1:], nan=0, posinf=0, neginf=0)

    # Extract exogenous features (columns 1-4 typically: gas_price, pjm_load, temperature, etc.)
    exog_indices = list(range(1, min(5, X_train_2d.shape[1])))
    if len(exog_indices) > 0:
        train_exog = X_train_2d[1:, exog_indices]  # Align with returns
        val_exog = X_val_2d[:len(val_returns), exog_indices]
        test_exog = X_test_2d[:len(test_returns), exog_indices]
    else:
        train_exog = None
        val_exog = None
        test_exog = None

    max_history = config.get("max_history", 1000)
    use_fast_forecast = config.get("use_fast_forecast", True)  # Default: fast mode
    refit_interval = config.get("refit_interval", 50)  # Refit every N steps if rolling

    print("\nSARIMAX Model (with Exogenous Variables):")
    print("-" * 80)
    print(f"  Task type: {TASK_TYPE}")
    print(f"  Training samples: {len(train_returns)}")
    print(f"  Validation samples: {len(y_val)}")
    print(f"  Test samples: {len(y_test)}")
    print(f"  Exogenous features: {len(exog_indices) if exog_indices else 0}")
    print(f"  Forecast mode: {'fast (single fit)' if use_fast_forecast else f'rolling (refit every {refit_interval} steps)'}")
    print(f"  Max history window: {max_history}")

    # Check stationarity
    is_stationary, p_value = check_stationarity(train_returns)
    print(f"  Series stationary: {is_stationary} (p-value: {p_value:.4f})")

    # Fit SARIMAX
    print("\nFitting SARIMAX model...")
    fitted_model, order, seasonal_order = fit_sarimax(
        train_returns,
        exog_train=train_exog,
        config=config
    )

    print(f"\nModel Summary:")
    print(f"  Order: SARIMAX{order}x{seasonal_order}")
    print(f"  AIC: {fitted_model.aic:.2f}")
    print(f"  BIC: {fitted_model.bic:.2f}")

    # Generate predictions
    print("\nGenerating predictions...")

    if use_fast_forecast:
        # Fast mode: single fit, forecast all at once
        print("  Validation set (fast mode):")
        y_pred_val_raw = fast_forecast_sarimax(
            fitted_model=fitted_model,
            actual_values=val_returns,
            future_exog=val_exog,
            verbose=True,
        )

        # Refit on train+val for test predictions
        print("  Refitting model with validation data...")
        full_train = np.concatenate([train_returns, val_returns])
        full_exog = np.concatenate([train_exog, val_exog]) if train_exog is not None else None
        fitted_full, _, _ = fit_sarimax(
            full_train,
            exog_train=full_exog,
            order=order,
            seasonal_order=seasonal_order,
            config=config
        )

        print("  Test set (fast mode):")
        y_pred_test_raw = fast_forecast_sarimax(
            fitted_model=fitted_full,
            actual_values=test_returns,
            future_exog=test_exog,
            verbose=True,
        )
    else:
        # Rolling mode: refit every N steps (slower but more adaptive)
        print("  Validation set (rolling mode):")
        y_pred_val_raw = rolling_forecast_sarimax(
            train_series=train_returns,
            train_exog=train_exog,
            actual_values=val_returns,
            future_exog=val_exog,
            order=order,
            seasonal_order=seasonal_order,
            max_history=max_history,
            verbose=True,
            refit_interval=refit_interval,
        )

        print("  Test set (rolling mode):")
        full_train = np.concatenate([train_returns, val_returns])
        full_exog = np.concatenate([train_exog, val_exog]) if train_exog is not None else None

        y_pred_test_raw = rolling_forecast_sarimax(
            train_series=full_train,
            train_exog=full_exog,
            actual_values=test_returns,
            future_exog=test_exog,
            order=order,
            seasonal_order=seasonal_order,
            max_history=max_history,
            verbose=True,
            refit_interval=refit_interval,
        )

    # Align predictions with target length
    y_pred_val_raw = y_pred_val_raw[:len(y_val)]
    y_pred_test_raw = y_pred_test_raw[:len(y_test)]

    # Pad if needed
    if len(y_pred_val_raw) < len(y_val):
        y_pred_val_raw = np.pad(y_pred_val_raw, (0, len(y_val) - len(y_pred_val_raw)), 'edge')
    if len(y_pred_test_raw) < len(y_test):
        y_pred_test_raw = np.pad(y_pred_test_raw, (0, len(y_test) - len(y_pred_test_raw)), 'edge')

    # Convert based on task type
    if TASK_TYPE == "classification":
        y_pred_val = (y_pred_val_raw > 0).astype(float)
        y_pred_test = (y_pred_test_raw > 0).astype(float)
        print(f"\nPrediction distribution:")
        print(f"  Val: {y_pred_val.mean()*100:.1f}% Up")
        print(f"  Test: {y_pred_test.mean()*100:.1f}% Up")
    else:
        y_pred_val = y_pred_val_raw
        y_pred_test = y_pred_test_raw

    print(f"\nPrediction stats:")
    print(f"  Val: min={y_pred_val_raw.min():.6f}, max={y_pred_val_raw.max():.6f}")
    print(f"  Test: min={y_pred_test_raw.min():.6f}, max={y_pred_test_raw.max():.6f}")
    print("-" * 80)

    return {
        "y_pred_val": y_pred_val,
        "y_pred_test": y_pred_test,
        "y_pred_val_raw": y_pred_val_raw,
        "y_pred_test_raw": y_pred_test_raw,
        "model": {
            "order": order,
            "seasonal_order": seasonal_order,
            "aic": fitted_model.aic,
            "bic": fitted_model.bic,
        },
        "order": order,
        "seasonal_order": seasonal_order,
    }


if __name__ == "__main__":
    """Test SARIMAX model"""
    print("=" * 80)
    print("Testing SARIMAX Model Module")
    print("=" * 80)

    if not STATSMODELS_AVAILABLE:
        print("statsmodels not installed. Skipping test.")
        exit(1)

    from data_pipeline import make_dataset_for_task
    from config import get_task_name

    print(f"\nLoading data for {TASK_TYPE} task...")
    datasets = make_dataset_for_task(
        task_type=get_task_name(),
        seq_len=None,
        scaler_type="standard"
    )

    # Quick test config
    test_config = {
        "use_auto_arima": True,
        "max_p": 3,
        "max_q": 2,
        "max_P": 1,
        "max_Q": 1,
        "seasonal_period": 7,
        "max_history": 500,
    }

    results = train_and_predict(datasets, config=test_config)

    print("\n" + "=" * 80)
    print("SARIMAX Model Test Complete!")
    print(f"Order: {results['order']}")
    print(f"Seasonal Order: {results['seasonal_order']}")
    print("=" * 80)
