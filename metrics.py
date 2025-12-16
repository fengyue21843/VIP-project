"""
Unified Metrics and Evaluation Module

This module provides standardized evaluation functions for both:
1. Prediction metrics (regression and classification)
2. Trading/backtesting metrics (ROI, Win Rate, Sharpe Ratio)

All models use these functions to ensure fair and consistent comparisons.

Key Functions:
- evaluate_regression(): MAE, RMSE, MSE for price models
- evaluate_classification(): Accuracy, Precision, Recall, F1, AUC for sign models
- evaluate_trading_from_returns(): ROI, Win Rate, Sharpe for trading strategies
- evaluate_model_outputs(): Master evaluator combining prediction + trading metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Union
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    classification_report, 
    roc_auc_score,
    confusion_matrix
)


# =============================================================================
# 1. PREDICTION METRICS
# =============================================================================

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Standard regression metrics for price/return prediction models.
    
    These metrics evaluate how accurately the model predicts continuous values
    (e.g., next-day returns, price changes).
    
    Args:
        y_true: 1D array of true values (e.g., actual next-day returns)
        y_pred: 1D array of predicted values
    
    Returns:
        dict with:
            - MAE: Mean Absolute Error (average absolute difference)
            - RMSE: Root Mean Squared Error (penalizes large errors more)
            - MSE: Mean Squared Error
            - MAPE: Mean Absolute Percentage Error (if no zeros in y_true)
    
    Example:
        ```python
        y_true = np.array([0.01, -0.02, 0.03, -0.01])
        y_pred = np.array([0.012, -0.018, 0.028, -0.015])
        metrics = evaluate_regression(y_true, y_pred)
        print(f"MAE: {metrics['MAE']:.6f}")
        ```
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE only if no zeros in y_true
    mape = None
    if not np.any(y_true == 0):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
    }
    
    if mape is not None:
        metrics["MAPE"] = mape
    
    return metrics


def evaluate_classification(
    y_true: np.ndarray, 
    y_prob_or_label: np.ndarray
) -> Dict[str, float]:
    """
    Standard classification metrics for sign_* models (direction prediction).
    
    These metrics evaluate how accurately the model predicts binary outcomes
    (e.g., price will go up or down).
    
    Args:
        y_true: 1D array of true binary labels (0 or 1)
        y_prob_or_label: 1D array of probabilities (0-1) or hard labels (0/1)
                        If probabilities, threshold at 0.5 for predictions
    
    Returns:
        dict with:
            - Accuracy: Overall correct predictions
            - Precision: Of predicted ups, how many were actually up
            - Recall: Of actual ups, how many were predicted
            - F1: Harmonic mean of Precision and Recall
            - AUC: Area Under ROC Curve (only if probabilities provided)
            - Confusion_Matrix: 2x2 matrix [[TN, FP], [FN, TP]]
    
    Example:
        ```python
        y_true = np.array([1, 0, 1, 1, 0])
        y_prob = np.array([0.8, 0.3, 0.6, 0.9, 0.2])
        metrics = evaluate_classification(y_true, y_prob)
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"AUC: {metrics['AUC']:.4f}")
        ```
    """
    y_true = np.asarray(y_true).flatten()
    arr = np.asarray(y_prob_or_label).flatten()
    
    # Determine if input is probability or hard label
    y_prob = None
    if ((arr >= 0) & (arr <= 1)).all() and not np.array_equal(arr, arr.astype(int)):
        # Input looks like probabilities
        y_prob = arr
        y_pred = (arr > 0.5).astype(int)
    else:
        # Input is hard labels
        y_pred = arr.astype(int)
    
    # Get classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Extract metrics
    metrics = {
        "Accuracy": report["accuracy"],
        "Precision": report.get("1", {}).get("precision", 0.0),
        "Recall": report.get("1", {}).get("recall", 0.0),
        "F1": report.get("1", {}).get("f1-score", 0.0),
        "Confusion_Matrix": cm.tolist(),
    }
    
    # Add AUC if probabilities were provided
    if y_prob is not None:
        try:
            metrics["AUC"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            # Can happen if y_true has only one class
            metrics["AUC"] = None
    
    return metrics


# =============================================================================
# 2. TRADING METRICS
# =============================================================================

def evaluate_trading_from_returns(
    actual_returns: np.ndarray,
    predicted_returns: np.ndarray,
    trading_rule: str = "sign"
) -> Dict[str, Any]:
    """
    Evaluate trading performance from actual and predicted returns.
    
    This function simulates a trading strategy based on predictions and
    evaluates its performance using standard financial metrics.
    
    Trading Logic:
    - For each period, determine position (+1 long, -1 short) based on prediction
    - Calculate strategy return = position * actual_return
    - Aggregate into performance metrics
    
    Args:
        actual_returns: Realized returns (e.g., next-day returns)
        predicted_returns: Model predictions (same length as actual_returns)
        trading_rule: How to convert predictions to positions
            - "sign": position = sign(predicted_return) ∈ {+1, -1}
            - "threshold": position = +1 if pred > 0, else 0 (long only)
    
    Returns:
        dict with:
            - WinRate: Fraction of trades with positive return
            - ROI: Total return (sum of all strategy returns)
            - Sharpe: Annualized Sharpe ratio (assumes daily returns)
            - TotalTrades: Number of trades executed
            - AvgReturn: Average return per trade
            - algo_returns_series: Array of per-period strategy returns
    
    Example:
        ```python
        actual = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        predicted = np.array([0.015, -0.01, 0.025, 0.005, 0.018])
        metrics = evaluate_trading_from_returns(actual, predicted)
        print(f"ROI: {metrics['ROI']:.4%}")
        print(f"Sharpe: {metrics['Sharpe']:.2f}")
        ```
    """
    # Align lengths
    n = min(len(actual_returns), len(predicted_returns))
    actual = np.asarray(actual_returns[:n]).flatten()
    pred = np.asarray(predicted_returns[:n]).flatten()
    
    # Remove NaN values
    valid_mask = ~(np.isnan(actual) | np.isnan(pred))
    actual = actual[valid_mask]
    pred = pred[valid_mask]
    
    if len(actual) == 0:
        return {
            "WinRate": 0.0,
            "ROI": 0.0,
            "Sharpe": 0.0,
            "TotalTrades": 0,
            "AvgReturn": 0.0,
            "algo_returns_series": np.array([]),
        }
    
    # Determine position based on trading rule
    if trading_rule == "sign":
        # Take position based on sign of prediction
        position = np.where(pred > 0, 1, -1)
    elif trading_rule == "threshold":
        # Long only when prediction is positive
        position = np.where(pred > 0, 1, 0)
    else:
        raise ValueError(f"Unknown trading_rule: {trading_rule}")
    
    # Calculate strategy returns
    algo_return = position * actual
    
    # Trading metrics
    total_trades = len(algo_return)
    wins = (algo_return > 0).sum()
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    roi = algo_return.sum()
    avg_return = algo_return.mean()
    
    # Sharpe ratio (annualized, assuming daily returns)
    if algo_return.std() > 0:
        sharpe = (algo_return.mean() / algo_return.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    return {
        "WinRate": win_rate,
        "ROI": roi,
        "Sharpe": sharpe,
        "TotalTrades": total_trades,
        "AvgReturn": avg_return,
        "algo_returns_series": algo_return,
    }


def build_trading_dataframe(
    actual_returns: np.ndarray,
    predicted_returns: np.ndarray,
    dates: pd.DatetimeIndex = None
) -> pd.DataFrame:
    """
    Build a DataFrame with trading details for analysis and visualization.
    
    This is useful for:
    - Detailed inspection of trading decisions
    - Plotting cumulative returns
    - Identifying periods of good/bad performance
    
    Args:
        actual_returns: Realized returns
        predicted_returns: Model predictions
        dates: Optional datetime index for the DataFrame
    
    Returns:
        DataFrame with columns:
            - Date (if dates provided)
            - Actual: Actual return
            - Predicted: Predicted return
            - Position: Trading position (+1 long, -1 short)
            - AlgoReturn: Strategy return for this period
            - CumulativeReturn: Cumulative strategy return
            - CumulativeActual: Cumulative buy-and-hold return
    
    Example:
        ```python
        df = build_trading_dataframe(actual_returns, predicted_returns)
        df['CumulativeReturn'].plot(title='Strategy Performance')
        ```
    """
    n = min(len(actual_returns), len(predicted_returns))
    
    # Create base dataframe
    df = pd.DataFrame({
        "Actual": actual_returns[:n],
        "Predicted": predicted_returns[:n],
    })
    
    if dates is not None:
        df.index = dates[:n]
        df.index.name = "Date"
    
    # Calculate position and strategy returns
    df["Position"] = np.where(df["Predicted"] > 0, 1, -1)
    df["AlgoReturn"] = df["Position"] * df["Actual"]
    
    # Calculate cumulative returns
    df["CumulativeReturn"] = (1 + df["AlgoReturn"]).cumprod() - 1
    df["CumulativeActual"] = (1 + df["Actual"]).cumprod() - 1
    
    return df


def evaluate_trading_from_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate trading metrics from a trading DataFrame.
    
    This is a convenience function for when you already have a DataFrame
    with AlgoReturn column (e.g., from build_trading_dataframe).
    
    Args:
        df: DataFrame with at least 'AlgoReturn' column
    
    Returns:
        dict with trading metrics (same as evaluate_trading_from_returns)
    """
    algo_returns = df["AlgoReturn"].values
    
    total_trades = len(algo_returns)
    wins = (algo_returns > 0).sum()
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    roi = algo_returns.sum()
    avg_return = algo_returns.mean()
    
    if algo_returns.std() > 0:
        sharpe = (algo_returns.mean() / algo_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    return {
        "WinRate": win_rate,
        "ROI": roi,
        "Sharpe": sharpe,
        "TotalTrades": total_trades,
        "AvgReturn": avg_return,
    }


# =============================================================================
# 3. MASTER EVALUATOR
# =============================================================================

def evaluate_model_outputs(
    task_type: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    returns_test: np.ndarray,
    trading_rule: str = "sign"
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Master evaluator combining prediction metrics and trading metrics.
    
    This is the main function you'll use to evaluate any model's performance.
    It automatically selects the appropriate metrics based on task type.
    
    Args:
        task_type: "regression" or "classification"
        y_test: True target values
        y_pred: Model predictions
        returns_test: Actual returns for trading evaluation
        trading_rule: Trading rule to use (default: "sign")
    
    Returns:
        Tuple of (base_metrics, trading_metrics):
            - base_metrics: Prediction metrics (MAE/RMSE or Accuracy/F1/AUC)
            - trading_metrics: Trading performance (ROI, Win Rate, Sharpe)
    
    Example:
        ```python
        from data_pipeline import make_dataset_for_task
        from models import model_lstm
        from metrics import evaluate_model_outputs
        
        # Get data and train model
        datasets = make_dataset_for_task("sign", seq_len=14)
        results = model_lstm.train_and_predict(datasets)
        
        # Evaluate
        base_metrics, trading_metrics = evaluate_model_outputs(
            task_type="classification",
            y_test=datasets["y_test"],
            y_pred=results["y_pred_test"],
            returns_test=datasets["returns_test"]
        )
        
        print("Prediction Metrics:", base_metrics)
        print("Trading Metrics:", trading_metrics)
        ```
    """
    # Evaluate prediction quality
    if task_type == "regression" or task_type == "price":
        base_metrics = evaluate_regression(y_test, y_pred)
        # For regression, predictions are returns, use them directly for trading
        trading_metrics = evaluate_trading_from_returns(
            actual_returns=returns_test,
            predicted_returns=y_pred,
            trading_rule=trading_rule
        )
    elif task_type == "classification" or task_type == "sign":
        base_metrics = evaluate_classification(y_test, y_pred)
        # For classification, convert predictions to position signals
        # If y_pred is probability, convert to direction
        y_pred_array = np.asarray(y_pred).flatten()
        if ((y_pred_array >= 0) & (y_pred_array <= 1)).all():
            # Probabilities: convert to -1/+1
            direction = np.where(y_pred_array > 0.5, 1, -1)
        else:
            # Already binary: convert 0/1 to -1/+1
            direction = np.where(y_pred_array > 0.5, 1, -1)
        
        # Use direction as "predicted returns" for trading
        # This means: if predict up, position = +1; if predict down, position = -1
        trading_metrics = evaluate_trading_from_returns(
            actual_returns=returns_test,
            predicted_returns=direction,
            trading_rule=trading_rule
        )
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Use 'regression' or 'classification'.")
    
    return base_metrics, trading_metrics


def print_evaluation_results(
    model_name: str,
    base_metrics: Dict[str, float],
    trading_metrics: Dict[str, Any],
    task_type: str
) -> None:
    """
    Pretty-print evaluation results.
    
    Args:
        model_name: Name of the model
        base_metrics: Prediction metrics from evaluate_model_outputs
        trading_metrics: Trading metrics from evaluate_model_outputs
        task_type: "regression" or "classification"
    """
    print("\n" + "=" * 80)
    print(f"{model_name} - Evaluation Results")
    print("=" * 80)
    
    print("\nPrediction Metrics:")
    print("-" * 40)
    for metric, value in base_metrics.items():
        if metric == "Confusion_Matrix":
            print(f"  {metric}:")
            print(f"    {value}")
        elif isinstance(value, float):
            if metric in ["Accuracy", "Precision", "Recall", "F1", "AUC"]:
                print(f"  {metric:20s} {value:.4f}")
            else:
                print(f"  {metric:20s} {value:.6f}")
        else:
            print(f"  {metric:20s} {value}")
    
    print("\nTrading Metrics:")
    print("-" * 40)
    for metric, value in trading_metrics.items():
        if metric == "algo_returns_series":
            continue  # Skip the series
        elif metric == "WinRate":
            print(f"  {metric:20s} {value:.2%}")
        elif metric == "ROI":
            print(f"  {metric:20s} {value:.4%}")
        elif metric == "AvgReturn":
            print(f"  {metric:20s} {value:.6f}")
        elif isinstance(value, float):
            print(f"  {metric:20s} {value:.4f}")
        else:
            print(f"  {metric:20s} {value}")
    
    print("=" * 80)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the metrics module
    """
    print("=" * 80)
    print("Testing Metrics Module")
    print("=" * 80)
    
    # Test 1: Regression metrics
    print("\nTest 1: Regression Metrics")
    print("-" * 80)
    y_true_reg = np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0.00, -0.015, 0.025])
    y_pred_reg = np.array([0.012, -0.018, 0.028, -0.015, 0.022, 0.002, -0.012, 0.023])
    
    reg_metrics = evaluate_regression(y_true_reg, y_pred_reg)
    print("Input:")
    print(f"  y_true: {y_true_reg}")
    print(f"  y_pred: {y_pred_reg}")
    print("\nMetrics:")
    for k, v in reg_metrics.items():
        print(f"  {k}: {v:.6f}")
    
    # Test 2: Classification metrics
    print("\n" + "=" * 80)
    print("Test 2: Classification Metrics")
    print("-" * 80)
    y_true_cls = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_prob_cls = np.array([0.8, 0.3, 0.6, 0.9, 0.2, 0.7, 0.4, 0.1, 0.85, 0.95])
    
    cls_metrics = evaluate_classification(y_true_cls, y_prob_cls)
    print("Input:")
    print(f"  y_true: {y_true_cls}")
    print(f"  y_prob: {y_prob_cls}")
    print("\nMetrics:")
    for k, v in cls_metrics.items():
        if k == "Confusion_Matrix":
            print(f"  {k}:")
            print(f"    {v}")
        else:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Test 3: Trading metrics
    print("\n" + "=" * 80)
    print("Test 3: Trading Metrics")
    print("-" * 80)
    actual_returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0.00, -0.015, 0.025])
    predicted_returns = np.array([0.015, -0.01, 0.025, 0.005, 0.018, -0.002, -0.01, 0.02])
    
    trading_metrics = evaluate_trading_from_returns(actual_returns, predicted_returns)
    print("Input:")
    print(f"  Actual returns:    {actual_returns}")
    print(f"  Predicted returns: {predicted_returns}")
    print("\nMetrics:")
    for k, v in trading_metrics.items():
        if k == "algo_returns_series":
            print(f"  {k}: {v}")
        elif k == "WinRate":
            print(f"  {k}: {v:.2%}")
        elif k == "ROI":
            print(f"  {k}: {v:.4%}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    
    # Test 4: Trading DataFrame
    print("\n" + "=" * 80)
    print("Test 4: Trading DataFrame")
    print("-" * 80)
    df = build_trading_dataframe(actual_returns, predicted_returns)
    print(df)
    
    # Test 5: Master evaluator (regression)
    print("\n" + "=" * 80)
    print("Test 5: Master Evaluator - Regression Task")
    print("-" * 80)
    base_reg, trade_reg = evaluate_model_outputs(
        task_type="regression",
        y_test=y_true_reg,
        y_pred=y_pred_reg,
        returns_test=actual_returns
    )
    print_evaluation_results("Test Regression Model", base_reg, trade_reg, "regression")
    
    # Test 6: Master evaluator (classification)
    print("\n" + "=" * 80)
    print("Test 6: Master Evaluator - Classification Task")
    print("-" * 80)
    base_cls, trade_cls = evaluate_model_outputs(
        task_type="classification",
        y_test=y_true_cls,
        y_pred=y_prob_cls,
        returns_test=actual_returns[:len(y_true_cls)]
    )
    print_evaluation_results("Test Classification Model", base_cls, trade_cls, "classification")
    
    print("\n" + "=" * 80)
    print("✓ All Metrics Tests Complete!")
    print("=" * 80)
