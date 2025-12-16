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

# Backend imports
from data_pipeline import make_dataset_for_task
from metrics import evaluate_model_outputs, evaluate_trading_from_returns, print_evaluation_results
from strategies import (
    run_percentile_strategy_backend,
    run_BOS_strategy_backend,
    compute_strategy_returns_from_positions,
    get_strategy_info,
    STRATEGY_DESCRIPTIONS
)
import config
import optuna_viz

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
        from models import model_lstm, model_gru, model_rf
        MODELS_AVAILABLE = True
    except ImportError:
        MODELS_AVAILABLE = False
else:
    MODELS_AVAILABLE = False
    model_lstm = None
    model_gru = None
    model_rf = None


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
        from data_pipeline import load_dataset
        data = load_dataset()
        
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
    
    # Model parameters
    st.subheader("Model Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if model_type in ["LSTM", "GRU", "TCN"]:
            seq_len = st.slider(
                "Sequence Length",
                7, 30, config.SEQUENCE_LENGTH,
                help="Number of past days to use for prediction"
            )
        else:
            seq_len = None
            if model_type == "XGBoost":
                st.info("XGBoost uses tabular data (no sequences)")
            else:
                st.info("Random Forest uses tabular data (no sequences)")
    
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
        if model_type in ["LSTM", "GRU"] and not PYTORCH_AVAILABLE:
            st.error(f"{model_type} model requires PyTorch, but PyTorch is not installed.")
            st.info("Please select Random Forest model, or install PyTorch: `pip install torch`")
            return
        
        waiting_statement()
        
        try:
            # Step 1: Prepare data
            st.write("Loading and preprocessing data...")

            # Force seq_len=None for tree-based models (requires 2D tabular data)
            if model_type in ["Random Forest", "XGBoost"]:
                seq_len = None

            datasets = make_dataset_for_task(
                task_type=task_type,
                seq_len=seq_len,  # None = tabular; int = sequence
                test_size=test_size,
                val_size=val_size,
                scaler_type=config.SCALER_TYPE
            )

            # Step 2: Train model
            st.write(f"Training {model_type} model...")

            if model_type == "LSTM":
                from models.model_lstm import train_and_predict
            elif model_type == "GRU":
                from models.model_gru import train_and_predict
            elif model_type == "TCN":
                from models.model_tcn import train_and_predict
            elif model_type == "XGBoost":
                from models.model_xgb import train_and_predict
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

            # Display Optuna hyperparameter optimization results
            if "optuna_study" in results and results["optuna_study"] is not None:
                st.subheader("Hyperparameter Optimization Results")

                # Display best parameters in an expandable section
                with st.expander("Finalized Hyperparameters", expanded=True):
                    st.write("**Best parameters found by Optuna:**")

                    # Display parameters in columns
                    best_params = results["best_params"]
                    param_cols = st.columns(min(3, len(best_params)))

                    for idx, (param_name, param_value) in enumerate(best_params.items()):
                        with param_cols[idx % 3]:
                            if isinstance(param_value, float):
                                st.metric(param_name, f"{param_value:.4f}")
                            else:
                                st.metric(param_name, str(param_value))

                # Create visualizations
                st.write("**Optimization Visualizations:**")

                tab1, tab2, tab3 = st.tabs(["Optimization History", "Parameter Importance", "Parameter Analysis"])

                with tab1:
                    st.write("This plot shows how the objective value improved over trials:")
                    fig_history = optuna_viz.plot_optimization_history(
                        results["optuna_study"],
                        title=f"{model_type} - Optimization History"
                    )
                    st.plotly_chart(fig_history, use_container_width=True)

                with tab2:
                    st.write("This plot ranks parameters by their impact on performance:")
                    fig_importance = optuna_viz.plot_param_importances(
                        results["optuna_study"],
                        title=f"{model_type} - Parameter Importances"
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)

                with tab3:
                    st.write("**Individual Parameter Effects** - See how each parameter affects performance:")

                    # Get all parameter names
                    import pandas as pd
                    df_trials = results["optuna_study"].trials_dataframe()
                    param_names = [col.replace('params_', '') for col in df_trials.columns if col.startswith('params_')]

                    if param_names:
                        # Create a chart for each parameter showing tested values vs results
                        num_params = len(param_names)

                        # Display in a grid (2 columns)
                        for i in range(0, num_params, 2):
                            cols = st.columns(2)

                            # First parameter in this row
                            with cols[0]:
                                param = param_names[i]
                                fig_slice = optuna_viz.plot_param_slice(
                                    results["optuna_study"],
                                    param,
                                    title=f"Effect of {param}"
                                )
                                st.plotly_chart(fig_slice, use_container_width=True)

                            # Second parameter in this row (if exists)
                            if i + 1 < num_params:
                                with cols[1]:
                                    param = param_names[i + 1]
                                    fig_slice = optuna_viz.plot_param_slice(
                                        results["optuna_study"],
                                        param,
                                        title=f"Effect of {param}"
                                    )
                                    st.plotly_chart(fig_slice, use_container_width=True)

                        # Add parallel coordinate plot at the end
                        st.write("---")
                        st.write("**Parallel Coordinate Plot** - Shows relationships between all parameters simultaneously:")
                        st.caption("Each line represents one trial. Color indicates performance (darker = better).")
                        fig_parallel = optuna_viz.plot_parallel_coordinate(
                            results["optuna_study"],
                            title=f"{model_type} - All Parameter Relationships"
                        )
                        st.plotly_chart(fig_parallel, use_container_width=True)

            # Display results
            st.subheader("Model Performance")

            # Prediction metrics - display in rows to avoid truncation
            st.write("**Prediction Metrics:**")

            # Filter out confusion matrix and separate metrics
            display_metrics = {k: v for k, v in base_metrics.items() if k != "Confusion_Matrix"}

            # Display in rows of 4 metrics to avoid label truncation
            metric_items = list(display_metrics.items())
            for i in range(0, len(metric_items), 4):
                cols = st.columns(4)
                for j, (metric_name, value) in enumerate(metric_items[i:i+4]):
                    with cols[j]:
                        if isinstance(value, float):
                            st.metric(metric_name, f"{value:.4f}")
                        else:
                            st.metric(metric_name, value)
            
            # Trading metrics
            st.write("**Trading Metrics:**")
            key_trading = {
                "ROI": trading_metrics["ROI"],
                "Win Rate": trading_metrics["WinRate"],
                "Sharpe Ratio": trading_metrics["Sharpe"],
                "Total Trades": trading_metrics["TotalTrades"]
            }
            display_metrics_cards(key_trading)
            
            # Detailed results
            with st.expander("Detailed Results"):
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
