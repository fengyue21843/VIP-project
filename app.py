"""
Streamlit Trading Dashboard - UI Layer

This is the Streamlit user interface that calls backend modules.
All heavy logic is in backend modules - this file only handles:
- User inputs (sliders, date pickers, buttons)
- Displaying results (tables, charts, metrics)
- Layout and styling

Backend Dependencies:
- data_pipeline.py (Step 1): Data loading and preprocessing
- models/*.py (Step 2): Model training and prediction  
- metrics.py (Step 3): Evaluation metrics
- strategies.py (Step 5): Rule-based strategies
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pickle
from pathlib import Path
from typing import Dict

# Backend imports
from data_pipeline_v2 import make_dataset_v2
from metrics import evaluate_model_outputs, evaluate_trading_from_returns, print_evaluation_results
from strategies import (
    run_percentile_strategy_backend,
    run_BOS_strategy_backend,
    compute_strategy_returns_from_positions,
    get_strategy_info,
    STRATEGY_DESCRIPTIONS
)
import config

# Check if PyTorch is available
PYTORCH_AVAILABLE = False
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    pass

# Model imports for prediction (conditional)
if PYTORCH_AVAILABLE:
    try:
        from models import model_lstm, model_gru, model_tcn, model_rf, model_xgboost
        MODELS_AVAILABLE = True
    except ImportError:
        MODELS_AVAILABLE = False
else:
    MODELS_AVAILABLE = False
    model_lstm = None
    model_gru = None
    model_tcn = None
    model_rf = None
    model_xgboost = None


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Electricity Price Trading Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# Styling and Helpers
# =============================================================================

def waiting_statement():
    """Display loading message"""
    st.info("Processing... Please wait.")


def success_statement():
    """Display success message"""
    st.success("Backtest complete!")


# =============================================================================
# Data Loading
# =============================================================================

@st.cache_data
def load_data():
    """Load electricity price data using unified pipeline"""
    try:
        # Use Step 1 data pipeline to load raw data
        from data_pipeline_v2 import load_dataset_v2
        data = load_dataset_v2()
        
        # Debug: Show columns
        # st.write("DEBUG - Columns loaded:", list(data.columns)[:10])
        
        # Ensure required columns exist
        required_cols = ['Trade Date', 'Electricity: Wtd Avg Price $/MWh']
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.write("Available columns:", list(data.columns)[:20])
            return None
        
        # Sort by date
        data = data.sort_values('Trade Date').reset_index(drop=True)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_price_and_signals(data: pd.DataFrame, strategy_name: str):
    """
    Plot price with trading signals.
    
    Args:
        data: DataFrame with 'Trade Date', price, 'Position', 'Signal'
        strategy_name: Name of strategy for title
    """
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=data['Trade Date'],
        y=data['Electricity: Wtd Avg Price $/MWh'],
        mode='lines',
        name='Price',
        line=dict(color='blue', width=2)
    ))
    
    # Buy signals
    buy_signals = data[data['Signal'] == 1]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals['Trade Date'],
            y=buy_signals['Electricity: Wtd Avg Price $/MWh'],
            mode='markers',
            name='Buy Signal',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))
    
    # Sell signals
    sell_signals = data[data['Signal'] == -1]
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals['Trade Date'],
            y=sell_signals['Electricity: Wtd Avg Price $/MWh'],
            mode='markers',
            name='Sell Signal',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))
    
    # Add percentile channels if they exist
    if 'Percentile_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['Trade Date'],
            y=data['Percentile_20'],
            mode='lines',
            name='Lower Channel',
            line=dict(color='lightgreen', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=data['Trade Date'],
            y=data['Percentile_80'],
            mode='lines',
            name='Upper Channel',
            line=dict(color='lightcoral', width=1, dash='dash')
        ))
    
    fig.update_layout(
        title=f"{strategy_name} - Price and Signals",
        xaxis_title="Date",
        yaxis_title="Price ($/MWh)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_cumulative_returns(actual_returns: np.ndarray, strategy_returns: np.ndarray, 
                            dates: pd.Series, strategy_name: str):
    """
    Plot cumulative returns comparison.
    
    Args:
        actual_returns: Market returns
        strategy_returns: Strategy returns
        dates: Date series
        strategy_name: Name for title
    """
    # Calculate cumulative returns
    cum_market = (1 + actual_returns).cumprod() - 1
    cum_strategy = (1 + strategy_returns).cumprod() - 1
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=cum_market * 100,
        mode='lines',
        name='Buy & Hold',
        line=dict(color='gray', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=cum_strategy * 100,
        mode='lines',
        name=strategy_name,
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title=f"{strategy_name} - Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_balance_growth(data: pd.DataFrame, initial_capital: float, final_balance: float):
    """
    Plot balance growth over time.
    
    Args:
        data: DataFrame with 'Balance' column
        initial_capital: Starting capital
        final_balance: Ending capital
    """
    if 'Balance' not in data.columns:
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Trade Date'],
        y=data['Balance'],
        mode='lines',
        name='Portfolio Balance',
        fill='tozeroy',
        line=dict(color='green', width=2)
    ))
    
    fig.add_hline(
        y=initial_capital, 
        line_dash="dash", 
        line_color="blue",
        annotation_text=f"Initial: ${initial_capital:,.2f}"
    )
    
    roi = ((final_balance - initial_capital) / initial_capital) * 100
    color = 'green' if roi >= 0 else 'red'
    
    fig.update_layout(
        title=f"Portfolio Balance Growth (ROI: {roi:.2f}%)",
        xaxis_title="Date",
        yaxis_title="Balance ($)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_training_history(history: Dict[str, list], model_name: str):
    """
    Plot training history (loss and metrics over epochs).
    
    Args:
        history: Dictionary with keys like 'loss', 'val_loss', 'accuracy', 'val_accuracy'
        model_name: Name of the model for title
    """
    if not history or 'loss' not in history:
        return
    
    fig = go.Figure()
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Training loss
    fig.add_trace(go.Scatter(
        x=list(epochs),
        y=history['loss'],
        mode='lines',
        name='Train Loss',
        line=dict(color='blue', width=2)
    ))
    
    # Validation loss
    if 'val_loss' in history:
        fig.add_trace(go.Scatter(
            x=list(epochs),
            y=history['val_loss'],
            mode='lines',
            name='Val Loss',
            line=dict(color='red', width=2)
        ))
    
    # Add metric if available
    metric_keys = [k for k in history.keys() if k not in ['loss', 'val_loss'] and not k.startswith('val_')]
    if metric_keys:
        metric_name = metric_keys[0]
        fig.add_trace(go.Scatter(
            x=list(epochs),
            y=history[metric_name],
            mode='lines',
            name=f'Train {metric_name.title()}',
            yaxis='y2',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        val_metric_key = f'val_{metric_name}'
        if val_metric_key in history:
            fig.add_trace(go.Scatter(
                x=list(epochs),
                y=history[val_metric_key],
                mode='lines',
                name=f'Val {metric_name.title()}',
                yaxis='y2',
                line=dict(color='orange', width=2, dash='dash')
            ))
    
    fig.update_layout(
        title=f"{model_name} - Training History",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis2=dict(title="Metric", overlaying='y', side='right') if metric_keys else None,
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                               task_type: str, model_name: str):
    """
    Plot predictions vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task_type: "classification" or "regression"
        model_name: Name of the model
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    if task_type == "classification":
        # For classification, show probability distribution and confusion
        fig = go.Figure()
        
        # Histogram of predicted probabilities
        fig.add_trace(go.Histogram(
            x=y_pred,
            nbinsx=50,
            name='Predicted Probabilities',
            opacity=0.7,
            marker_color='blue'
        ))
        
        fig.update_layout(
            title=f"{model_name} - Prediction Probability Distribution",
            xaxis_title="Predicted Probability",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix heatmap
        from sklearn.metrics import confusion_matrix
        y_pred_binary = (y_pred > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_binary)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted 0', 'Predicted 1'],
            y=['Actual 0', 'Actual 1'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            colorbar=dict(title="Count")
        ))
        
        fig.update_layout(
            title=f"{model_name} - Confusion Matrix",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.4f})',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"{model_name} - ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve, auc as pr_auc
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_pred)
        pr_auc_score = pr_auc(recall_curve, precision_curve)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall_curve,
            y=precision_curve,
            mode='lines',
            name=f'PR Curve (AUC = {pr_auc_score:.4f})',
            line=dict(color='green', width=2)
        ))
        fig.add_hline(
            y=y_true.mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Baseline (mean={y_true.mean():.3f})"
        )
        
        fig.update_layout(
            title=f"{model_name} - Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Threshold Sweep
        from sklearn.metrics import precision_score, recall_score, f1_score
        thresholds = np.linspace(0.1, 0.9, 50)
        precisions = []
        recalls = []
        f1_scores = []
        
        for thresh in thresholds:
            y_pred_thresh = (y_pred >= thresh).astype(int)
            precisions.append(precision_score(y_true, y_pred_thresh, zero_division=0))
            recalls.append(recall_score(y_true, y_pred_thresh, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred_thresh, zero_division=0))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=precisions,
            mode='lines',
            name='Precision',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=recalls,
            mode='lines',
            name='Recall',
            line=dict(color='green', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=f1_scores,
            mode='lines',
            name='F1 Score',
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            title=f"{model_name} - Threshold Sweep (Precision, Recall, F1)",
            xaxis_title="Threshold",
            yaxis_title="Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction Probability Distribution by Classification Result
        y_pred_binary = (y_pred >= 0.5).astype(int)
        tp_mask = (y_true == 1) & (y_pred_binary == 1)
        fp_mask = (y_true == 0) & (y_pred_binary == 1)
        fn_mask = (y_true == 1) & (y_pred_binary == 0)
        tn_mask = (y_true == 0) & (y_pred_binary == 0)
        
        fig = go.Figure()
        
        if tp_mask.sum() > 0:
            fig.add_trace(go.Histogram(
                x=y_pred[tp_mask],
                nbinsx=30,
                name='True Positives',
                opacity=0.7,
                marker_color='green'
            ))
        
        if fp_mask.sum() > 0:
            fig.add_trace(go.Histogram(
                x=y_pred[fp_mask],
                nbinsx=30,
                name='False Positives',
                opacity=0.7,
                marker_color='red'
            ))
        
        if fn_mask.sum() > 0:
            fig.add_trace(go.Histogram(
                x=y_pred[fn_mask],
                nbinsx=30,
                name='False Negatives',
                opacity=0.7,
                marker_color='orange'
            ))
        
        if tn_mask.sum() > 0:
            fig.add_trace(go.Histogram(
                x=y_pred[tn_mask],
                nbinsx=30,
                name='True Negatives',
                opacity=0.7,
                marker_color='blue'
            ))
        
        fig.add_vline(x=0.5, line_dash="dash", line_color="black", annotation_text="Threshold 0.5")
        
        fig.update_layout(
            title=f"{model_name} - Prediction Probability Distribution by Classification Result",
            xaxis_title="Predicted Probability",
            yaxis_title="Frequency",
            barmode='overlay',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calibration Curve (Reliability Diagram)
        from sklearn.calibration import calibration_curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred, n_bins=10, strategy='uniform'
            )
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=mean_predicted_value,
                y=fraction_of_positives,
                mode='lines+markers',
                name='Model Calibration',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfectly Calibrated',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f"{model_name} - Calibration Curve (Reliability Diagram)",
                xaxis_title="Mean Predicted Probability",
                yaxis_title="Fraction of Positives",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Calibration curve not available: {str(e)}")
        
        # Predictions Timeline
        indices = range(len(y_true))
        fig = go.Figure()
        
        # Actual labels
        fig.add_trace(go.Scatter(
            x=list(indices),
            y=y_true,
            mode='markers',
            name='Actual Labels',
            marker=dict(color='blue', size=4, symbol='circle'),
            opacity=0.6
        ))
        
        # Predicted probabilities
        fig.add_trace(go.Scatter(
            x=list(indices),
            y=y_pred,
            mode='lines',
            name='Predicted Probability',
            line=dict(color='green', width=1),
            opacity=0.7
        ))
        
        # Threshold line
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold 0.5")
        
        # Binary predictions
        fig.add_trace(go.Scatter(
            x=list(indices),
            y=y_pred_binary,
            mode='markers',
            name='Binary Predictions',
            marker=dict(color='orange', size=3, symbol='x'),
            opacity=0.5
        ))
        
        fig.update_layout(
            title=f"{model_name} - Predictions Timeline",
            xaxis_title="Sample Index",
            yaxis_title="Label / Probability",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # regression
        # Scatter plot: predicted vs actual
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', size=4, opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f"{model_name} - Predictions vs Actual",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series plot
        fig = go.Figure()
        
        indices = range(len(y_true))
        fig.add_trace(go.Scatter(
            x=list(indices),
            y=y_true,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=list(indices),
            y=y_pred,
            mode='lines',
            name='Predicted',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f"{model_name} - Predictions vs Actual (Time Series)",
            xaxis_title="Sample Index",
            yaxis_title="Value",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residuals vs Time
        residuals = y_true - y_pred
        time_index = np.arange(len(residuals))
        
        # Residuals over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_index,
            y=residuals,
            mode='lines',
            name='Residuals',
            line=dict(color='blue', width=1),
            opacity=0.6
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title=f"{model_name} - Residuals vs Time",
            xaxis_title="Time Index",
            yaxis_title="Residual (True - Predicted)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Rolling mean of residuals
        window = max(20, len(residuals) // 20)
        residuals_series = pd.Series(residuals)
        rolling_mean = residuals_series.rolling(window=window).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_index,
            y=rolling_mean,
            mode='lines',
            name=f'Rolling Mean (window={window})',
            line=dict(color='green', width=2)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title=f"{model_name} - Rolling Mean of Residuals",
            xaxis_title="Time Index",
            yaxis_title="Rolling Mean Residual",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residual distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=residuals,
            nbinsx=50,
            name='Residual Distribution',
            marker_color='blue',
            opacity=0.7
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Residual")
        
        fig.update_layout(
            title=f"{model_name} - Residual Distribution",
            xaxis_title="Residual Value",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Rolling RMSE
        window = max(20, len(residuals) // 20)
        rolling_rmse = []
        for i in range(0, len(residuals) - window + 1):
            window_residuals = residuals[i:i+window]
            rolling_rmse.append(np.sqrt(np.mean(window_residuals ** 2)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(rolling_rmse))),
            y=rolling_rmse,
            mode='lines',
            name=f'Rolling RMSE (window={window})',
            line=dict(color='purple', width=2)
        ))
        
        fig.update_layout(
            title=f"{model_name} - Rolling RMSE (Error Trends)",
            xaxis_title="Window Index",
            yaxis_title=f"Rolling RMSE (window={window})",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Q-Q Plot for Residuals (Normality Check)
        from scipy import stats
        try:
            sorted_residuals = np.sort(residuals)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=sorted_residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='blue', size=4, opacity=0.6)
            ))
            
            # Perfect normal line
            min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
            max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Normal',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f"{model_name} - Q-Q Plot (Normality Check)",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Q-Q plot not available: {str(e)}")
        
        # Error by Magnitude
        abs_residuals = np.abs(residuals)
        abs_actual = np.abs(y_true)
        
        # Bin by magnitude
        bins = np.linspace(abs_actual.min(), abs_actual.max(), 10)
        bin_indices = np.digitize(abs_actual, bins)
        
        bin_errors = []
        bin_centers = []
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_errors.append(np.mean(abs_residuals[mask]))
                bin_centers.append((bins[i-1] + bins[i]) / 2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=bin_errors,
            mode='lines+markers',
            name='Mean Absolute Error',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"{model_name} - Error by Magnitude",
            xaxis_title="Absolute Actual Value",
            yaxis_title="Mean Absolute Error",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def display_metrics_cards(metrics: dict):
    """
    Display metrics in card format.
    
    Args:
        metrics: Dictionary of metric names and values
    """
    cols = st.columns(len(metrics))
    
    for idx, (metric_name, value) in enumerate(metrics.items()):
        with cols[idx]:
            if isinstance(value, float):
                if 'Rate' in metric_name or 'ROI' in metric_name:
                    display_value = f"{value:.2f}%"
                else:
                    display_value = f"{value:.4f}"
            else:
                display_value = str(value)
            
            st.metric(label=metric_name, value=display_value)


# =============================================================================
# Strategy UI Functions
# =============================================================================

def run_percentile_strategy_ui(data: pd.DataFrame):
    """
    UI for Percentile Channel Breakout strategy.
    Collects user inputs and displays results.
    """
    st.header("Percentile Channel Breakout Strategy")
    
    # Strategy description
    strategy_info = get_strategy_info("Percentile Channel Breakout")
    st.info(strategy_info["description"])
    
    # User inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Strategy Parameters")
        window_size = st.slider("Window Size (days)", 5, 60, 14, 
                               help="Rolling window for percentile calculation")
        percentile_low = st.slider("Lower Percentile (Buy)", 0, 50, 20,
                                   help="Buy when price <= this percentile")
        percentile_high = st.slider("Upper Percentile (Sell)", 50, 100, 80,
                                    help="Sell when price >= this percentile")
    
    with col2:
        st.subheader("Backtest Period")
        start_date = st.date_input("Start Date", 
                                   value=data['Trade Date'].min().date(),
                                   min_value=data['Trade Date'].min().date(),
                                   max_value=data['Trade Date'].max().date())
        end_date = st.date_input("End Date",
                                value=data['Trade Date'].max().date(),
                                min_value=data['Trade Date'].min().date(),
                                max_value=data['Trade Date'].max().date())
        
        initial_capital = st.number_input("Initial Capital ($)", 
                                         min_value=1000, value=10000, step=1000)
    
    # Run backtest button
    if st.button("Run Backtest", key="percentile_run"):
        waiting_statement()
        
        # Filter date range
        mask = (data['Trade Date'] >= pd.to_datetime(start_date)) & \
               (data['Trade Date'] <= pd.to_datetime(end_date))
        data_window = data.loc[mask].copy()
        
        if len(data_window) < window_size:
            st.error(f"Insufficient data. Need at least {window_size} days.")
            return
        
        # Call backend strategy
        data_with_signals = run_percentile_strategy_backend(
            data_window,
            window_size=window_size,
            percentile_low=percentile_low,
            percentile_high=percentile_high
        )
        
        # Calculate returns and metrics
        actual_returns, strategy_returns = compute_strategy_returns_from_positions(
            data_with_signals
        )
        
        trading_metrics = evaluate_trading_from_returns(actual_returns, strategy_returns)
        
        success_statement()
        
        # Display results
        st.subheader("Performance Metrics")
        
        # Key metrics in cards
        key_metrics = {
            "ROI": trading_metrics["ROI"],
            "Win Rate": trading_metrics["WinRate"],
            "Sharpe Ratio": trading_metrics["Sharpe"],
            "Total Trades": trading_metrics["TotalTrades"]
        }
        display_metrics_cards(key_metrics)
        
        # Visualizations
        st.subheader("Strategy Visualization")
        
        tab1, tab2, tab3 = st.tabs(["Price & Signals", "Cumulative Returns", "Trade Log"])
        
        with tab1:
            plot_price_and_signals(data_with_signals, "Percentile Strategy")
        
        with tab2:
            plot_cumulative_returns(
                actual_returns, 
                strategy_returns,
                data_with_signals['Trade Date'],
                "Percentile Strategy"
            )
        
        with tab3:
            st.dataframe(
                data_with_signals[['Trade Date', 'Electricity: Wtd Avg Price $/MWh', 
                                  'Signal', 'Position', 'Percentile_20', 'Percentile_80']],
                use_container_width=True
            )
        
        # Detailed metrics
        with st.expander("Detailed Metrics"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Trading Metrics:**")
                for key, value in trading_metrics.items():
                    if key != 'algo_returns_series':
                        if isinstance(value, float):
                            st.write(f"- {key}: {value:.4f}")
                        else:
                            st.write(f"- {key}: {value}")
            
            with col2:
                st.write("**Strategy Parameters:**")
                st.write(f"- Window Size: {window_size}")
                st.write(f"- Lower Percentile: {percentile_low}")
                st.write(f"- Upper Percentile: {percentile_high}")
                st.write(f"- Date Range: {start_date} to {end_date}")


def run_BOS_strategy_ui(data: pd.DataFrame):
    """
    UI for Break of Structure strategy.
    Collects user inputs and displays results.
    """
    st.header("Break of Structure (BOS) Strategy")
    
    # Strategy description
    strategy_info = get_strategy_info("Break of Structure")
    st.info(strategy_info["description"])
    
    # User inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Strategy Parameters")
        initial_capital = st.number_input("Initial Capital ($)", 
                                         min_value=1000, value=10000, step=1000,
                                         key="bos_capital")
    
    with col2:
        st.subheader("Backtest Period")
        start_date = st.date_input("Start Date", 
                                   value=data['Trade Date'].min().date(),
                                   min_value=data['Trade Date'].min().date(),
                                   max_value=data['Trade Date'].max().date(),
                                   key="bos_start")
        end_date = st.date_input("End Date",
                                value=data['Trade Date'].max().date(),
                                min_value=data['Trade Date'].min().date(),
                                max_value=data['Trade Date'].max().date(),
                                key="bos_end")
    
    # Run backtest button
    if st.button("Run Backtest", key="bos_run"):
        waiting_statement()
        
        # Filter date range
        mask = (data['Trade Date'] >= pd.to_datetime(start_date)) & \
               (data['Trade Date'] <= pd.to_datetime(end_date))
        data_window = data.loc[mask].copy()
        
        if len(data_window) < 20:
            st.error("Insufficient data. Need at least 20 days.")
            return
        
        # Call backend strategy
        data_bos = run_BOS_strategy_backend(data_window, initial_capital=initial_capital)
        
        # Calculate returns and metrics
        actual_returns, strategy_returns = compute_strategy_returns_from_positions(data_bos)
        trading_metrics = evaluate_trading_from_returns(actual_returns, strategy_returns)
        
        success_statement()
        
        # Display results
        st.subheader("Performance Metrics")
        
        # Key metrics in cards
        key_metrics = {
            "ROI": trading_metrics["ROI"],
            "Win Rate": trading_metrics["WinRate"],
            "Sharpe Ratio": trading_metrics["Sharpe"],
            "Total Trades": trading_metrics["TotalTrades"]
        }
        display_metrics_cards(key_metrics)
        
        # Visualizations
        st.subheader("Strategy Visualization")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Price & Signals", "Cumulative Returns", 
                                          "Balance Growth", "Trade Log"])
        
        with tab1:
            plot_price_and_signals(data_bos, "BOS Strategy")
        
        with tab2:
            plot_cumulative_returns(
                actual_returns,
                strategy_returns,
                data_bos['Trade Date'],
                "BOS Strategy"
            )
        
        with tab3:
            final_balance = data_bos['Balance'].iloc[-1]
            plot_balance_growth(data_bos, initial_capital, final_balance)
        
        with tab4:
            st.dataframe(
                data_bos[['Trade Date', 'Electricity: Wtd Avg Price $/MWh',
                         'Signal', 'Position', 'Balance', 'Shares']],
                use_container_width=True
            )
        
        # Detailed metrics
        with st.expander("Detailed Metrics"):
            st.write("**Trading Metrics:**")
            for key, value in trading_metrics.items():
                if key != 'algo_returns_series':
                    if isinstance(value, float):
                        st.write(f"- {key}: {value:.4f}")
                    else:
                        st.write(f"- {key}: {value}")


# =============================================================================
# ML Model UI Functions
# =============================================================================

def run_ml_model_ui(data: pd.DataFrame):
    """
    UI for ML model backtesting.
    Uses Steps 1-3 backend for training and evaluation.
    """
    st.header("Machine Learning Models")
    
    # Check if PyTorch is available
    if not PYTORCH_AVAILABLE:
        st.warning("PyTorch Not Installed")
        st.markdown("""
        Deep learning models (LSTM, GRU) require PyTorch.
        
        **Currently Available:**
        - All trading strategies (Percentile, BOS)
        - Random Forest model (does not require PyTorch)
        
        **To use LSTM/GRU models, install PyTorch:**
        ```bash
        pip install torch
        ```
        
        Or using conda:
        ```bash
        conda install pytorch
        ```
        """)
        
        # Still allow Random Forest
        st.info("You can still use Random Forest model (does not require PyTorch)")
    else:
        st.info("Train and backtest ML models using the unified pipeline (Steps 1-3)")
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        task_type = st.selectbox(
            "Task Type",
            options=["sign", "price"],
            format_func=lambda x: "Direction Prediction (Classification)" if x == "sign" 
                                  else "Price Prediction (Regression)"
        )
    
    with col2:
        model_type = st.selectbox(
            "Model Type",
            options=["LSTM", "GRU", "TCN", "Random Forest", "XGBoost"],
            help="Select the model architecture"
        )
    
    # Optuna hyperparameter optimization (only for GRU)
    use_optuna = False
    optuna_trials = 20
    optuna_timeout = None
    
    if model_type == "GRU":
        st.subheader("Hyperparameter Optimization (Optuna)")
        use_optuna = st.checkbox(
            "Enable Optuna Hyperparameter Optimization",
            value=False,
            help="Automatically find the best hyperparameters using Optuna. This will take longer but may improve model performance."
        )
        
        if use_optuna:
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                optuna_trials = st.number_input(
                    "Number of Trials",
                    min_value=5,
                    max_value=100,
                    value=20,
                    step=5,
                    help="More trials = better results but longer time"
                )
            with col_opt2:
                optuna_timeout_min = st.number_input(
                    "Timeout (minutes, 0 = no timeout)",
                    min_value=0,
                    max_value=120,
                    value=0,
                    step=5,
                    help="Maximum time to spend on optimization"
                )
                optuna_timeout = optuna_timeout_min * 60 if optuna_timeout_min > 0 else None
            
            st.info(f"⚠️ Optimization will run {optuna_trials} trials. This may take several minutes.")
    
    # Model parameters
    st.subheader("Model Parameters")

    # Initialize seq_len BEFORE columns to ensure it's always defined
    seq_len = config.SEQUENCE_LENGTH if model_type in ["LSTM", "GRU", "TCN"] else None

    col1, col2, col3 = st.columns(3)

    with col1:
        if model_type in ["LSTM", "GRU", "TCN"]:
            seq_len = st.slider(
                "Sequence Length",
                7, 30, seq_len,
                help="Number of past days to use for prediction"
            )
        else:
            st.info(f"{model_type} uses tabular data (no sequences)")

    with col2:
        test_size = st.slider(
            "Test Split %",
            10, 30, int(config.TEST_SIZE * 100)
        ) / 100

    with col3:
        val_size = st.slider(
            "Validation Split %",
            10, 30, int(config.VAL_SIZE * 100)
        ) / 100

    # Run model button
    if st.button("Train and Evaluate Model", key="ml_run"):
        # Check if selected model requires PyTorch
        if model_type in ["LSTM", "GRU", "TCN"] and not PYTORCH_AVAILABLE:
            st.error(f"{model_type} model requires PyTorch, but PyTorch is not installed.")
            st.info("Please select Random Forest or XGBoost model, or install PyTorch: `pip install torch`")
            return

        waiting_statement()

        try:
            # Determine seq_len based on model type
            # CRITICAL: Tabular models (RF, XGBoost) need seq_len=None
            if model_type in ["LSTM", "GRU", "TCN"]:
                final_seq_len = seq_len  # Use slider value
            else:
                final_seq_len = None  # Force None for tabular models

            # Step 1: Prepare data
            st.write("Loading and preprocessing data...")
            st.write(f"DEBUG: Model={model_type}, seq_len={final_seq_len}")
            datasets = make_dataset_v2(
                task_type=task_type,
                seq_len=final_seq_len,  # None = tabular; int = sequence
                test_size=test_size,
                val_size=val_size,
                scaler_type=config.SCALER_TYPE,
                use_imputation=False,
                drop_zero_returns=True if task_type == "sign" else False,
                balance_data=True if task_type == "sign" else False,
            )
            
            # Step 2: Train model
            st.write(f"Training {model_type} model...")

            if model_type == "LSTM":
                from models.model_lstm import train_and_predict
                results = train_and_predict(datasets, config=None)
            elif model_type == "GRU":
                from models.model_gru import train_and_predict
                results = train_and_predict(
                    datasets,
                    config=None,
                    use_optuna=use_optuna,
                    optuna_trials=optuna_trials,
                    optuna_timeout=optuna_timeout
                )
            elif model_type == "TCN":
                from models.model_tcn import train_and_predict
                results = train_and_predict(datasets, config=None)
            elif model_type == "XGBoost":
                from models.model_xgboost import train_and_predict
                results = train_and_predict(datasets, config=None)
            else:  # Random Forest
                from models.model_rf import train_and_predict
                results = train_and_predict(datasets, config=None)
            
            # Step 3: Evaluate
            st.write("Evaluating performance...")
            
            metrics_task_type = "classification" if task_type == "sign" else "regression"
            
            base_metrics, trading_metrics = evaluate_model_outputs(
                task_type=metrics_task_type,
                y_test=datasets["y_test"],
                y_pred=results["y_pred_test"],
                returns_test=datasets["returns_test"]
            )
            
            success_statement()
            
            # Display Optuna results if available
            if model_type == "GRU" and "optuna_results" in results:
                st.subheader("Optuna Optimization Results")
                optuna_results = results["optuna_results"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Validation Score", f"{optuna_results['best_value']:.4f}")
                with col2:
                    st.metric("Trials Completed", optuna_results["n_trials"])
                with col3:
                    st.metric("Status", "✓ Complete")
                
                with st.expander("Best Hyperparameters Found"):
                    best_config = optuna_results["best_config"]
                    st.json(best_config)
                    
                    # Show hyperparameter importance if available
                    try:
                        import optuna.visualization as vis
                        importance_fig = vis.plot_param_importances(optuna_results["study"])
                        st.plotly_chart(importance_fig, use_container_width=True)
                    except Exception:
                        pass
            
            # Display results
            st.subheader("Model Performance")
            
            # Diagnostic information for classification models
            if metrics_task_type == "classification":
                y_test_array = np.asarray(datasets["y_test"]).flatten()
                y_pred_array = np.asarray(results["y_pred_test"]).flatten()
                
                # Show prediction statistics
                with st.expander("Prediction Diagnostics", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write("**True Labels:**")
                        st.write(f"- Total samples: {len(y_test_array)}")
                        st.write(f"- Positive (1): {int(y_test_array.sum())} ({y_test_array.mean()*100:.1f}%)")
                        st.write(f"- Negative (0): {int((1-y_test_array).sum())} ({(1-y_test_array.mean())*100:.1f}%)")
                    
                    with col2:
                        st.write("**Predictions (Probabilities):**")
                        st.write(f"- Min: {y_pred_array.min():.4f}")
                        st.write(f"- Mean: {y_pred_array.mean():.4f}")
                        st.write(f"- Max: {y_pred_array.max():.4f}")
                        st.write(f"- Std: {y_pred_array.std():.4f}")
                    
                    with col3:
                        st.write("**Predictions @ 0.5 Threshold:**")
                        y_pred_binary = (y_pred_array > 0.5).astype(int)
                        st.write(f"- Predicted Positive: {int(y_pred_binary.sum())} ({y_pred_binary.mean()*100:.1f}%)")
                        st.write(f"- Predicted Negative: {int((1-y_pred_binary).sum())} ({(1-y_pred_binary.mean())*100:.1f}%)")
                        
                        # Show confusion matrix breakdown
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(y_test_array, y_pred_binary)
                        if cm.shape == (2, 2):
                            st.write(f"- True Negatives: {cm[0,0]}")
                            st.write(f"- False Positives: {cm[0,1]}")
                            st.write(f"- False Negatives: {cm[1,0]}")
                            st.write(f"- True Positives: {cm[1,1]}")
                    
                    # Warning if no positive predictions
                    if y_pred_binary.sum() == 0:
                        st.warning("⚠️ **Model is predicting all negatives!** This suggests the model may need: more training epochs, different learning rate, class balancing, or architecture adjustments.")
            
            # Prediction metrics
            st.write("**Prediction Metrics:**")
            pred_cols = st.columns(len(base_metrics))
            for idx, (metric_name, value) in enumerate(base_metrics.items()):
                if metric_name != "Confusion_Matrix":
                    with pred_cols[idx]:
                        if isinstance(value, float):
                            st.metric(metric_name, f"{value:.4f}")
                        else:
                            st.metric(metric_name, value)
            
            # Metric Comparison Table with Baselines
            if metrics_task_type == "classification":
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                y_test_array = np.asarray(datasets["y_test"]).flatten()
                y_pred_array = np.asarray(results["y_pred_test"]).flatten()
                y_pred_binary = (y_pred_array > 0.5).astype(int)
                y_train_array = np.asarray(datasets["y_train"]).flatten()
                
                # Calculate baselines
                majority_class = int(np.round(y_train_array.mean()))
                majority_pred = np.full_like(y_test_array, majority_class)
                random_pred = np.random.RandomState(42).binomial(1, y_train_array.mean(), size=len(y_test_array))
                random_prob = np.random.RandomState(42).uniform(0, 1, size=len(y_test_array))
                
                # Model metrics
                model_acc = accuracy_score(y_test_array, y_pred_binary)
                model_prec = precision_score(y_test_array, y_pred_binary, zero_division=0)
                model_rec = recall_score(y_test_array, y_pred_binary, zero_division=0)
                model_f1 = f1_score(y_test_array, y_pred_binary, zero_division=0)
                model_auc = roc_auc_score(y_test_array, y_pred_array) if len(np.unique(y_test_array)) > 1 else 0.5
                
                # Baseline metrics
                maj_acc = accuracy_score(y_test_array, majority_pred)
                # Calculate metrics for the class that was predicted (majority class)
                # If majority is 0, calculate for class 0; if majority is 1, calculate for class 1
                if majority_class == 0:
                    # For majority class 0, calculate metrics for class 0 (negative class)
                    maj_prec = precision_score(y_test_array, majority_pred, pos_label=0, zero_division=0)
                    maj_rec = recall_score(y_test_array, majority_pred, pos_label=0, zero_division=0)
                    maj_f1 = f1_score(y_test_array, majority_pred, pos_label=0, zero_division=0)
                else:
                    # For majority class 1, calculate metrics for class 1 (positive class)
                    maj_prec = precision_score(y_test_array, majority_pred, pos_label=1, zero_division=0)
                    maj_rec = recall_score(y_test_array, majority_pred, pos_label=1, zero_division=0)
                    maj_f1 = f1_score(y_test_array, majority_pred, pos_label=1, zero_division=0)
                
                rand_acc = accuracy_score(y_test_array, random_pred)
                rand_prec = precision_score(y_test_array, random_pred, zero_division=0)
                rand_rec = recall_score(y_test_array, random_pred, zero_division=0)
                rand_f1 = f1_score(y_test_array, random_pred, zero_division=0)
                from sklearn.metrics import roc_curve, auc
                fpr_rand, tpr_rand, _ = roc_curve(y_test_array, random_prob)
                rand_auc = auc(fpr_rand, tpr_rand)
                
                comparison_data = {
                    "Model": [f"{model_type} Model", "Baseline (Majority)", "Baseline (Random)"],
                    "Accuracy": [model_acc, maj_acc, rand_acc],
                    "Precision": [model_prec, maj_prec, rand_prec],
                    "Recall": [model_rec, maj_rec, rand_rec],
                    "F1": [model_f1, maj_f1, rand_f1],
                    "ROC-AUC": [model_auc, 0.5, rand_auc]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.subheader("Metric Comparison Table")
                st.dataframe(comparison_df.style.format({
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1": "{:.4f}",
                    "ROC-AUC": "{:.4f}"
                }), use_container_width=True)
                
            else:  # regression
                y_test_array = np.asarray(datasets["y_test"]).flatten()
                y_pred_array = np.asarray(results["y_pred_test"]).flatten()
                y_train_array = np.asarray(datasets["y_train"]).flatten()
                
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
                
                # Model metrics
                model_mse = mean_squared_error(y_test_array, y_pred_array)
                model_rmse = np.sqrt(model_mse)
                model_mae = mean_absolute_error(y_test_array, y_pred_array)
                model_r2 = r2_score(y_test_array, y_pred_array)
                try:
                    model_mape = mean_absolute_percentage_error(y_test_array, y_pred_array) * 100
                except:
                    model_mape = np.mean(np.abs((y_test_array - y_pred_array) / (y_test_array + 1e-8))) * 100
                
                # Baseline: Zero (predict 0)
                zero_pred = np.zeros_like(y_test_array)
                zero_mse = mean_squared_error(y_test_array, zero_pred)
                zero_rmse = np.sqrt(zero_mse)
                zero_mae = mean_absolute_error(y_test_array, zero_pred)
                zero_r2 = r2_score(y_test_array, zero_pred)
                try:
                    zero_mape = mean_absolute_percentage_error(y_test_array, zero_pred) * 100
                except:
                    zero_mape = np.mean(np.abs((y_test_array - zero_pred) / (y_test_array + 1e-8))) * 100
                
                # Baseline: Mean (predict mean of training)
                mean_pred = np.full_like(y_test_array, y_train_array.mean())
                mean_mse = mean_squared_error(y_test_array, mean_pred)
                mean_rmse = np.sqrt(mean_mse)
                mean_mae = mean_absolute_error(y_test_array, mean_pred)
                mean_r2 = r2_score(y_test_array, mean_pred)
                try:
                    mean_mape = mean_absolute_percentage_error(y_test_array, mean_pred) * 100
                except:
                    mean_mape = np.mean(np.abs((y_test_array - mean_pred) / (y_test_array + 1e-8))) * 100
                
                comparison_data = {
                    "Model": [f"{model_type} Model", "Baseline (Zero)", "Baseline (Mean)"],
                    "MSE": [model_mse, zero_mse, mean_mse],
                    "RMSE": [model_rmse, zero_rmse, mean_rmse],
                    "MAE": [model_mae, zero_mae, mean_mae],
                    "R²": [model_r2, zero_r2, mean_r2],
                    "MAPE (%)": [model_mape, zero_mape, mean_mape]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.subheader("📊 Metric Comparison Table")
                st.dataframe(comparison_df.style.format({
                    "MSE": "{:.6f}",
                    "RMSE": "{:.6f}",
                    "MAE": "{:.6f}",
                    "R²": "{:.4f}",
                    "MAPE (%)": "{:.2f}"
                }), use_container_width=True)
            
            # Trading metrics
            st.write("**Trading Metrics:**")
            key_trading = {
                "ROI": trading_metrics["ROI"],
                "Win Rate": trading_metrics["WinRate"],
                "Sharpe Ratio": trading_metrics["Sharpe"],
                "Total Trades": trading_metrics["TotalTrades"]
            }
            display_metrics_cards(key_trading)
            
            # Visualizations
            st.subheader("Model Visualizations")
            
            viz_tabs = st.tabs(["Training History", "Predictions Analysis", "Detailed Results"])
            
            with viz_tabs[0]:
                # Training history plot
                if "history" in results and results["history"]:
                    plot_training_history(results["history"], model_type)
                else:
                    st.info("Training history not available for this model.")
            
            with viz_tabs[1]:
                # Predictions vs actual plots
                plot_predictions_vs_actual(
                    datasets["y_test"],
                    results["y_pred_test"],
                    metrics_task_type,
                    model_type
                )
            
            with viz_tabs[2]:
                # Detailed results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**All Prediction Metrics:**")
                    for key, value in base_metrics.items():
                        if isinstance(value, (int, float)):
                            st.write(f"- {key}: {value:.4f}")
                        else:
                            st.write(f"- {key}: {value}")
                
                with col2:
                    st.write("**All Trading Metrics:**")
                    for key, value in trading_metrics.items():
                        if key != 'algo_returns_series':
                            if isinstance(value, float):
                                st.write(f"- {key}: {value:.4f}")
                            else:
                                st.write(f"- {key}: {value}")
            
            # Dataset info
            with st.expander("Dataset Information"):
                st.write(f"- Train samples: {len(datasets['y_train'])}")
                st.write(f"- Validation samples: {len(datasets['y_val'])}")
                st.write(f"- Test samples: {len(datasets['y_test'])}")
                st.write(f"- Features: {datasets['X_train'].shape[-1]}")
                if seq_len:
                    st.write(f"- Sequence length: {seq_len}")
        
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("Electricity Price Trading Dashboard")
    st.markdown("""
    **Unified Trading Platform** - Rule-based strategies and ML models with standardized evaluation
    
    Built on modular backend:
    - **Step 1**: Unified data pipeline
    - **Step 2**: Standardized model interface
    - **Step 3**: Unified metrics and evaluation
    - **Step 4**: Experiment runner
    - **Step 5**: Interactive UI (this app)
    """)
    
    # Load data
    data = load_data()
    
    if data is None:
        st.error("Failed to load data. Please check the data path in config.py")
        return
    
    st.success(f"Data loaded: {len(data)} records from {data['Trade Date'].min().date()} to {data['Trade Date'].max().date()}")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Show PyTorch status
    if not PYTORCH_AVAILABLE:
        st.sidebar.warning("PyTorch Not Installed\n(LSTM/GRU Unavailable)")
    
    page = st.sidebar.radio(
        "Select Strategy/Model:",
        options=[
            "Dashboard",
            "Percentile Strategy",
            "Break of Structure",
            "ML Models",
            "Documentation"
        ]
    )
    
    # Show data preview
    with st.sidebar.expander("Data Preview"):
        st.dataframe(data.head(), use_container_width=True)
    
    # Route to selected page
    if page == "Dashboard":
        st.header("Dashboard Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Date Range", f"{(data['Trade Date'].max() - data['Trade Date'].min()).days} days")
        with col3:
            avg_price = data['Electricity: Wtd Avg Price $/MWh'].mean()
            st.metric("Avg Price", f"${avg_price:.2f}/MWh")
        
        # Price chart
        fig = px.line(data, x='Trade Date', y='Electricity: Wtd Avg Price $/MWh',
                     title='Electricity Price History')
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.subheader("Price Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Min Price", f"${data['Electricity: Wtd Avg Price $/MWh'].min():.2f}")
        with col2:
            st.metric("Max Price", f"${data['Electricity: Wtd Avg Price $/MWh'].max():.2f}")
        with col3:
            st.metric("Std Dev", f"${data['Electricity: Wtd Avg Price $/MWh'].std():.2f}")
        with col4:
            volatility = (data['Electricity: Wtd Avg Price $/MWh'].std() / 
                         data['Electricity: Wtd Avg Price $/MWh'].mean()) * 100
            st.metric("Volatility", f"{volatility:.2f}%")
    
    elif page == "Percentile Strategy":
        run_percentile_strategy_ui(data)
    
    elif page == "Break of Structure":
        run_BOS_strategy_ui(data)
    
    elif page == "ML Models":
        run_ml_model_ui(data)
    
    elif page == "Documentation":
        st.header("Documentation")
        
        st.markdown("""
        ### System Architecture
        
        This application is built on a **5-step modular architecture**:
        
        #### Step 1: Unified Data Pipeline (`data_pipeline.py`)
        - Single source of truth for data loading
        - Consistent feature engineering (20 features)
        - Time-based train/val/test splits
        - Leakage-safe scaling
        
        #### Step 2: Standardized Model Interface (`models/*.py`)
        - All models implement `train_and_predict(datasets, config)`
        - Unified training scheme
        - Consistent output format
        
        #### Step 3: Unified Metrics (`metrics.py`)
        - Prediction metrics (MAE, Accuracy, F1, etc.)
        - Trading metrics (ROI, Sharpe, Win Rate)
        - Same evaluation for all strategies/models
        
        #### Step 4: Experiment Runner (`run_all_models.py`)
        - Batch processing of multiple models
        - Comparable results table
        - CSV export for analysis
        
        #### Step 5: Interactive UI (This App)
        - Thin UI layer over backend logic
        - Real-time backtesting
        - Visual analysis
        
        ### Strategies
        
        #### 1. Percentile Channel Breakout
        - **Logic**: Trade based on rolling percentile channels
        - **Buy**: Price <= lower percentile
        - **Sell**: Price >= upper percentile
        - **Parameters**: Window size, percentile thresholds
        
        #### 2. Break of Structure (BOS)
        - **Logic**: Trade on trend breaks
        - **Buy**: Price breaks above recent high (uptrend)
        - **Sell**: Price breaks below recent low (downtrend)
        - **Parameters**: Initial capital, lookback window
        
        #### 3. Machine Learning Models
        - **LSTM**: 2-layer LSTM for sequence modeling
        - **GRU**: 2-layer GRU (faster alternative to LSTM)
        - **Random Forest**: Ensemble method for tabular data
        - **Tasks**: Classification (direction) or Regression (price)
        
        ### Metrics Explained
        
        **Prediction Metrics:**
        - **MAE**: Mean Absolute Error (regression)
        - **Accuracy**: Classification accuracy
        - **F1 Score**: Harmonic mean of precision and recall
        - **AUC**: Area Under ROC Curve
        
        **Trading Metrics:**
        - **ROI**: Return on Investment (%)
        - **Win Rate**: Percentage of profitable trades
        - **Sharpe Ratio**: Risk-adjusted returns
        - **Total Trades**: Number of trades executed
        
        ### Usage Tips
        
        1. **Start with Dashboard** to understand data characteristics
        2. **Try rule-based strategies** first (faster, interpretable)
        3. **Experiment with parameters** using sliders
        4. **Compare strategies** by saving results
        5. **Use ML models** for more sophisticated patterns
        
        ### Backend Modules
        
        All heavy computation is in backend modules:
        - `data_pipeline.py` - Data loading and preprocessing
        - `strategies.py` - Rule-based strategy logic
        - `metrics.py` - Evaluation functions
        - `models/*.py` - ML model implementations
        - `run_all_models.py` - Batch experiment runner
        
        This separation ensures:
        - Code reusability
        - Consistent evaluation
        - Easy testing
        - Maintainability
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Electricity Price Trading Dashboard**
    
    Version 1.0
    Built with Streamlit
    """)


if __name__ == "__main__":
    main()
