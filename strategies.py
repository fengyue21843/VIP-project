"""
Backend Strategy Module - Pure Logic (No Streamlit Dependencies)

This module contains rule-based trading strategies with no UI dependencies.
All functions are pure Python - can be used in both offline experiments and Streamlit UI.

Strategies:
1. Percentile Channel Breakout
2. Break of Structure (BOS)

All strategies return DataFrames with standardized columns:
- 'Trade Date'
- 'Electricity: Wtd Avg Price $/MWh' 
- 'Position': +1 (long), -1 (short), 0 (neutral)
- 'Signal': Entry signals
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


# =============================================================================
# Percentile Channel Breakout Strategy
# =============================================================================

def calculate_percentiles(data: pd.DataFrame, window_size: int, 
                         percentile_low: int, percentile_high: int) -> pd.DataFrame:
    """
    Calculate rolling percentiles for price channel.
    
    Args:
        data: DataFrame with 'Electricity: Wtd Avg Price $/MWh' column
        window_size: Rolling window size (must be >= 2, in days)
        percentile_low: Lower percentile threshold (0-100, e.g., 20)
        percentile_high: Upper percentile threshold (0-100, e.g., 80)
    
    Returns:
        DataFrame with added percentile columns
    
    Raises:
        ValueError: If parameters are invalid
    """
    # Input validation
    if window_size < 2:
        raise ValueError(f"window_size must be >= 2, got {window_size}")
    if not (0 <= percentile_low < percentile_high <= 100):
        raise ValueError(
            f"percentile_low ({percentile_low}) must be < percentile_high ({percentile_high}), "
            f"both in range [0, 100]"
        )
    
    data = data.copy()
    price_col = 'Electricity: Wtd Avg Price $/MWh'
    
    data['Percentile_20'] = (
        data[price_col]
        .rolling(window=window_size)
        .apply(lambda x: np.percentile(x, percentile_low), raw=True)
    )
    data['Percentile_80'] = (
        data[price_col]
        .rolling(window=window_size)
        .apply(lambda x: np.percentile(x, percentile_high), raw=True)
    )
    
    return data


def run_percentile_strategy_backend(data: pd.DataFrame, window_size: int = 14,
                                    percentile_low: int = 20, 
                                    percentile_high: int = 80) -> pd.DataFrame:
    """
    Pure backend implementation of Percentile Channel Breakout strategy.
    No Streamlit dependencies - just data in, signals out.
    
    Strategy Logic:
    - Buy (Position = +1) when price <= lower percentile
    - Sell (Position = -1) when price >= upper percentile
    - Hold previous position otherwise
    
    Args:
        data: DataFrame with at least 'Trade Date' and 'Electricity: Wtd Avg Price $/MWh'
        window_size: Rolling window for percentile calculation
        percentile_low: Lower threshold (buy signal)
        percentile_high: Upper threshold (sell signal)
    
    Returns:
        DataFrame with added columns:
        - 'Percentile_20', 'Percentile_80': Channel boundaries
        - 'Signal': Entry signals (1=buy, -1=sell, 0=no signal)
        - 'Position': Current position (1=long, -1=short, 0=neutral)
    """
    data = data.copy()
    
    # Calculate percentile channels
    data = calculate_percentiles(data, window_size, percentile_low, percentile_high)
    
    # Initialize signal and position columns
    data['Signal'] = 0
    data['Position'] = 0
    
    price_col = 'Electricity: Wtd Avg Price $/MWh'
    
    # Generate signals based on channel breakout
    for i in range(window_size, len(data)):
        price = data[price_col].iloc[i]
        low_threshold = data['Percentile_20'].iloc[i]
        high_threshold = data['Percentile_80'].iloc[i]
        
        if pd.notna(low_threshold) and pd.notna(high_threshold):
            if price <= low_threshold:
                data.loc[data.index[i], 'Signal'] = 1  # Buy signal
            elif price >= high_threshold:
                data.loc[data.index[i], 'Signal'] = -1  # Sell signal
    
    # Convert signals to positions (forward-fill to maintain position)
    data['Position'] = data['Signal'].replace(0, np.nan).ffill().fillna(0)
    
    return data


# =============================================================================
# Break of Structure (BOS) Strategy
# =============================================================================

def detect_trend(data: pd.DataFrame, lookback: int = 5) -> str:
    """
    Detect market trend based on recent price action.
    
    Args:
        data: DataFrame with price data
        lookback: Number of periods to analyze
    
    Returns:
        'uptrend', 'downtrend', or 'sideways'
    """
    if len(data) < lookback:
        return 'sideways'
    
    recent_prices = data['Electricity: Wtd Avg Price $/MWh'].iloc[-lookback:].values
    
    # Simple trend detection: compare first and last prices
    if recent_prices[-1] > recent_prices[0] * 1.02:  # 2% threshold
        return 'uptrend'
    elif recent_prices[-1] < recent_prices[0] * 0.98:
        return 'downtrend'
    else:
        return 'sideways'


def get_latest_high_and_low(data: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
    """
    Get recent high and low prices.
    
    Args:
        data: DataFrame with price data
        window: Lookback window for finding highs/lows
    
    Returns:
        (recent_high, recent_low)
    """
    if len(data) < window:
        window = len(data)
    
    recent_data = data.iloc[-window:]
    recent_high = recent_data['Electricity: Wtd Avg Price $/MWh'].max()
    recent_low = recent_data['Electricity: Wtd Avg Price $/MWh'].min()
    
    return recent_high, recent_low


def BOS_logic(data: pd.DataFrame, initial_capital: float = 10000.0) -> pd.DataFrame:
    """
    Break of Structure (BOS) trading logic.
    
    Strategy Logic:
    - In uptrend: Buy when price breaks above recent high
    - In downtrend: Sell when price breaks below recent low
    - Track balance and positions
    
    Args:
        data: DataFrame with price and date columns
        initial_capital: Starting capital
    
    Returns:
        DataFrame with trading signals and positions
    """
    data = data.copy()
    
    # Initialize columns
    data['Signal'] = 0
    data['Position'] = 0
    data['Balance'] = initial_capital
    data['Shares'] = 0.0
    
    price_col = 'Electricity: Wtd Avg Price $/MWh'
    balance = initial_capital
    position = 0
    shares = 0.0
    
    for i in range(20, len(data)):  # Start after minimum window
        current_price = data[price_col].iloc[i]
        
        # Get market context
        historical_data = data.iloc[:i]
        trend = detect_trend(historical_data)
        recent_high, recent_low = get_latest_high_and_low(historical_data)
        
        # Trading logic
        if trend == 'uptrend' and current_price > recent_high and position <= 0:
            # Buy signal - break of structure to upside
            if position == -1:  # Close short first
                balance -= shares * current_price  # Cover short
                shares = 0
            # Go long
            shares = balance / current_price
            position = 1
            data.loc[data.index[i], 'Signal'] = 1
            
        elif trend == 'downtrend' and current_price < recent_low and position >= 0:
            # Sell signal - break of structure to downside
            if position == 1:  # Close long first
                balance = shares * current_price
                shares = 0
            # Go short
            shares = balance / current_price
            position = -1
            data.loc[data.index[i], 'Signal'] = -1
        
        # Update tracking columns
        data.loc[data.index[i], 'Position'] = position
        data.loc[data.index[i], 'Shares'] = shares
        
        # Calculate current balance
        if position == 1:
            data.loc[data.index[i], 'Balance'] = shares * current_price
        elif position == -1:
            data.loc[data.index[i], 'Balance'] = balance - shares * current_price
        else:
            data.loc[data.index[i], 'Balance'] = balance
    
    return data


def run_BOS_strategy_backend(data: pd.DataFrame, 
                             initial_capital: float = 10000.0) -> pd.DataFrame:
    """
    Pure backend implementation of Break of Structure strategy.
    No Streamlit dependencies.
    
    Args:
        data: DataFrame with 'Trade Date' and 'Electricity: Wtd Avg Price $/MWh'
        initial_capital: Starting capital for simulation
    
    Returns:
        DataFrame with signals, positions, and balance tracking
    """
    return BOS_logic(data.copy(), initial_capital)


# =============================================================================
# Strategy Returns Calculation
# =============================================================================

def compute_strategy_returns_from_positions(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute actual returns and strategy returns from position data.
    
    This function bridges rule-based strategies with ML model evaluation:
    - Calculates per-period returns from price changes
    - Applies position to get strategy returns
    - Output can be fed to metrics.evaluate_trading_from_returns()
    
    Args:
        data: DataFrame with columns:
              - 'Electricity: Wtd Avg Price $/MWh': Price series
              - 'Position': Trading positions (+1, -1, or 0)
    
    Returns:
        Tuple of (actual_returns, strategy_returns):
        - actual_returns: Market returns (% change)
        - strategy_returns: Position * actual_returns
    """
    if 'Electricity: Wtd Avg Price $/MWh' not in data.columns:
        raise ValueError("Data must contain 'Electricity: Wtd Avg Price $/MWh' column")
    if 'Position' not in data.columns:
        raise ValueError("Data must contain 'Position' column")
    
    # Calculate actual market returns (t to t+1)
    prices = data['Electricity: Wtd Avg Price $/MWh'].values
    actual_returns = np.zeros(len(prices))
    actual_returns[:-1] = (prices[1:] - prices[:-1]) / prices[:-1]
    
    # Get positions (use position at time t for return from t to t+1)
    position = data['Position'].values
    
    # Strategy returns = position * actual_returns
    strategy_returns = position * actual_returns
    
    return actual_returns, strategy_returns


def calculate_ROI_from_positions(data: pd.DataFrame, initial_capital: float = 10000.0) -> float:
    """
    Calculate total ROI from position data.
    
    Args:
        data: DataFrame with positions and returns
        initial_capital: Starting capital
    
    Returns:
        Total ROI as percentage
    """
    actual_returns, strategy_returns = compute_strategy_returns_from_positions(data)
    
    # Calculate cumulative returns
    cumulative_return = np.prod(1 + strategy_returns) - 1
    roi_percentage = cumulative_return * 100
    
    return roi_percentage


# =============================================================================
# Strategy Metadata
# =============================================================================

STRATEGY_DESCRIPTIONS = {
    "Percentile Channel Breakout": {
        "description": "Trades based on rolling percentile channels. Buy when price falls below lower percentile, sell when above upper percentile.",
        "parameters": ["window_size", "percentile_low", "percentile_high"],
        "default_params": {"window_size": 14, "percentile_low": 20, "percentile_high": 80}
    },
    "Break of Structure": {
        "description": "Identifies trend breaks by monitoring price breaking above recent highs (uptrend) or below recent lows (downtrend).",
        "parameters": ["initial_capital", "lookback_window"],
        "default_params": {"initial_capital": 10000.0, "lookback_window": 20}
    }
}


def get_strategy_info(strategy_name: str) -> Dict[str, Any]:
    """
    Get metadata about a strategy.
    
    Args:
        strategy_name: Name of strategy
    
    Returns:
        Dictionary with strategy information
    """
    return STRATEGY_DESCRIPTIONS.get(strategy_name, {
        "description": "Unknown strategy",
        "parameters": [],
        "default_params": {}
    })
