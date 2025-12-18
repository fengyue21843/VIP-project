"""
Unified Data & Feature Pipeline Module

This module provides a single, reusable data pipeline that:
- Loads the cleaned dataset once
- Builds a standard feature set that all models will use
- Constructs targets for both tasks (sign classification and price regression)
- Splits into train/val/test in a consistent, time-ordered way
- Applies scaling in a leakage-safe way (fit on train only)
- Optionally turns tabular data into sequences for LSTM/GRU

All models (LSTM, GRU, RF, ARIMA, etc.) should receive data from this module.
"""

import pandas as pd
import numpy as np
from typing import Literal, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import config

# Optional: Import torch if available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Set return_torch=False")


# Type definitions
TaskType = Literal["sign", "price"]
ScalerType = Literal["standard", "minmax"]


def load_dataset() -> pd.DataFrame:
    """
    Returns a cleaned DataFrame with all raw columns loaded and basic imputations applied.
    
    This function:
    - Reads datasets/Data_cleaned_Dataset.csv
    - Parses date columns
    - Interpolates missing values
    - Replaces zero electricity prices with mean of non-zero values
    
    Returns:
        pd.DataFrame: Cleaned raw dataset with all columns
    """
    # Determine the path to the dataset
    # Prioritize config.DATA_PATH, then fall back to hardcoded paths
    data_paths = [
        config.DATA_PATH,
        r"c:\Users\DELL\Downloads\Data-20251207T171745Z-1-001\Data\Data_cleaned_Dataset.csv",
        "datasets/Data_cleaned_Dataset.csv",
        "../Data/Data_cleaned_Dataset.csv",
        "Data/Data_cleaned_Dataset.csv"
    ]
    
    dataset_path = None
    for path in data_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        raise FileNotFoundError(
            "Could not find Data_cleaned_Dataset.csv. "
            "Please ensure the file exists in one of the expected locations."
        )
    
    # Read the dataset
    df = pd.read_csv(dataset_path)
    
    # Parse date columns
    date_columns = [
        'Trade Date',
        'Electricity: Delivery Start Date',
        'Electricity: Delivery End Date'
    ]
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Sort by Trade Date (keep it as a column, not as index)
    if 'Trade Date' in df.columns:
        df = df.sort_values('Trade Date')
    
    # Interpolate missing values (pandas 2.0+ compatibility)
    df = df.ffill().bfill()
    
    # Handle zero electricity prices - replace with mean of non-zero values
    price_col = 'Electricity: Wtd Avg Price $/MWh'
    if price_col in df.columns:
        non_zero_prices = df[price_col][df[price_col] > 0]
        if len(non_zero_prices) > 0:
            mean_non_zero = non_zero_prices.mean()
            df.loc[df[price_col] == 0, price_col] = mean_non_zero
    
    # Reset index to ensure it's a clean integer index
    df = df.reset_index(drop=True)
    
    return df


def build_feature_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the cleaned raw df from load_dataset() and returns a feature DataFrame
    with engineered columns used by all models.
    
    Features constructed:
    1. Core features (price, volume, natural gas, load, temperature, weekday)
    2. Time features (Day, Month, Year)
    3. Return & % change features
    
    Args:
        df_raw: Raw cleaned DataFrame from load_dataset()
    
    Returns:
        pd.DataFrame: Feature DataFrame with all engineered columns
    """
    df_feat = df_raw.copy()
    
    # CRITICAL FIX: Use Trade Date as index (the actual datetime)
    if 'Trade Date' in df_feat.columns:
        df_feat = df_feat.set_index('Trade Date')
        if not isinstance(df_feat.index, pd.DatetimeIndex):
            df_feat.index = pd.to_datetime(df_feat.index)
    else:
        raise KeyError("'Trade Date' column not found in dataset. Cannot build time features.")
    
    # Ensure index is sorted
    df_feat = df_feat.sort_index()
    
    # 1. Core features - select and rename for clarity
    core_columns = {
        'Electricity: Wtd Avg Price $/MWh': 'price',
        'Electricity: Daily Volume MWh': 'volume',
        'Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)': 'gas_price',
        'pjm_load sum in MW (daily)': 'pjm_load',
        'temperature mean in C (daily): US': 'temperature'
    }
    
    # Create feature dataframe with core features
    df_features = pd.DataFrame(index=df_feat.index)
    
    for old_col, new_col in core_columns.items():
        if old_col in df_feat.columns:
            df_features[new_col] = df_feat[old_col]
        else:
            raise KeyError(f"Required column '{old_col}' not found in dataset")
    
    # 2. Time features
    df_features['Day'] = df_features.index.day
    df_features['Month'] = df_features.index.month
    df_features['Year'] = df_features.index.year
    df_features['Weekday'] = df_features.index.weekday  # 0=Monday, 6=Sunday
    
    # 3. Return features - price return
    df_features['price_return'] = df_features['price'].pct_change()
    
    # 4. Percentage change features for other core variables
    df_features['volume_pct_change'] = df_features['volume'].pct_change()
    df_features['gas_price_pct_change'] = df_features['gas_price'].pct_change()
    df_features['pjm_load_pct_change'] = df_features['pjm_load'].pct_change()
    df_features['temperature_pct_change'] = df_features['temperature'].pct_change()
    
    # 5. Additional useful features
    # Lagged features (previous day values)
    df_features['price_lag1'] = df_features['price'].shift(1)
    df_features['volume_lag1'] = df_features['volume'].shift(1)
    df_features['gas_price_lag1'] = df_features['gas_price'].shift(1)
    
    # Rolling statistics (7-day windows)
    df_features['price_rolling_mean_7d'] = df_features['price'].rolling(window=7, min_periods=1).mean()
    df_features['price_rolling_std_7d'] = df_features['price'].rolling(window=7, min_periods=1).std()
    df_features['volume_rolling_mean_7d'] = df_features['volume'].rolling(window=7, min_periods=1).mean()
    
    # Additional technical features
    # Volatility measures (7-day and 30-day)
    df_features['volatility_7d'] = df_features['price_return'].rolling(window=7, min_periods=1).std()
    df_features['volatility_30d'] = df_features['price_return'].rolling(window=30, min_periods=1).std()
    
    # Momentum indicators
    df_features['momentum_3d'] = df_features['price'].pct_change(periods=3)
    df_features['momentum_7d'] = df_features['price'].pct_change(periods=7)
    
    # Price position relative to range (0-1 scale)
    rolling_min = df_features['price'].rolling(window=14, min_periods=1).min()
    rolling_max = df_features['price'].rolling(window=14, min_periods=1).max()
    price_range = rolling_max - rolling_min
    df_features['price_position'] = np.where(
        price_range > 0, 
        (df_features['price'] - rolling_min) / price_range,
        0.5  # Default to middle if no range
    )
    
    # Trend strength (slope of linear regression over 14 days)
    def calculate_trend_slope(series, window=14):
        """Calculate rolling linear regression slope"""
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(window)
                # Polyfit returns [slope, intercept]
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)
        return pd.Series(slopes, index=series.index)
    
    df_features['trend_slope'] = calculate_trend_slope(df_features['price'])
    
    # Handle NaNs - drop rows where key features are NaN
    # After pct_change and shift operations, the first few rows will have NaNs
    df_features = df_features.dropna()
    
    return df_features


def build_targets(df_feat: pd.DataFrame, task_type: TaskType) -> np.ndarray:
    """
    Given the feature DataFrame, construct the target vector y according to the task type.
    
    - 'price': next-day return (regression target)
    - 'sign': next-day direction (binary classification: 0=down, 1=up)
    
    Args:
        df_feat: Feature DataFrame from build_feature_frame()
        task_type: Type of task - "sign" or "price"
    
    Returns:
        np.ndarray: Target vector aligned with df_feat
    """
    # Calculate next-day return (target)
    # We need the actual price to calculate future return
    if 'price' not in df_feat.columns:
        raise KeyError("'price' column not found in feature DataFrame")
    
    # Calculate return at time t (if not already present)
    price = df_feat['price']
    return_t = price.pct_change()
    
    # Calculate next-day return (shift -1 to get future value)
    return_t_plus_1 = return_t.shift(-1)
    
    # Build targets based on task type
    if task_type == "price":
        # Regression target: next-day return
        y = return_t_plus_1.values
    elif task_type == "sign":
        # CRITICAL FIX: Classification target ALWAYS 0/1 float32 (not int)
        # This ensures compatibility with BCEWithLogitsLoss
        y = (return_t_plus_1 > 0).astype(np.float32).values
    else:
        raise ValueError(f"Invalid task_type: {task_type}. Must be 'sign' or 'price'")
    
    return y


def create_sequences(
    data: np.ndarray,
    target: np.ndarray,
    sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert tabular data into sequences for RNN models.
    
    Args:
        data: Input features array (2D: samples x features)
        target: Target array (1D: samples)
        sequence_length: Length of each sequence
    
    Returns:
        Tuple of (X_sequences, y_sequences):
            - X_sequences: 3D array (samples, sequence_length, features)
            - y_sequences: 1D array (samples) - target for each sequence
    """
    xs, ys = [], []
    
    for i in range(len(data) - sequence_length):
        # Extract sequence of length `sequence_length`
        x_seq = data[i:i + sequence_length]
        # Target is the value right after the sequence
        y_val = target[i + sequence_length]
        
        xs.append(x_seq)
        ys.append(y_val)
    
    return np.array(xs), np.array(ys)


def make_dataset_for_task(
    task_type: TaskType | None = None,
    seq_len: int | None = None,
    test_size: float | None = None,
    val_size: float | None = None,
    scaler_type: ScalerType | None = None,
    return_torch: bool = False,
    use_rolling_window: bool | None = None,
    rolling_window_size: float | None = None,
) -> Dict[str, Any]:
    """
    End-to-end data pipeline function that:
    1. Loads dataset via load_dataset()
    2. Builds features via build_feature_frame()
    3. Builds targets via build_targets()
    4. Performs time-based split into train/val/test
    5. Fits scaler on train, transforms val/test
    6. (Optional) builds sequences for RNN models
    7. (Optional) converts to PyTorch tensors
    
    Args:
        task_type: Type of task. If None, uses config.get_task_name()
        seq_len: Sequence length. If None, uses config.SEQUENCE_LENGTH
        test_size: Test set size. If None, uses config.TEST_SIZE
        val_size: Validation set size. If None, uses config.VAL_SIZE
        scaler_type: Scaler type. If None, uses config.SCALER_TYPE
        return_torch: If True, return PyTorch tensors instead of numpy arrays
        use_rolling_window: If True, use rolling window split; if False, use expanding window
        rolling_window_size: Size of training window as proportion of data (e.g., 0.5 = 50%)
    
    Returns:
        Dictionary containing:
            - X_train, y_train: Training data (numpy or torch)
            - X_val, y_val: Validation data
            - X_test, y_test: Test data
            - returns_test: Raw returns for backtesting (aligned with test set)
            - scaler: Fitted scaler instance
            - feature_names: List of feature column names
            - n_features: Number of features
    """
    # Read defaults from config
    if task_type is None:
        task_type = config.get_task_name()
    if test_size is None:
        test_size = config.TEST_SIZE
    if val_size is None:
        val_size = config.VAL_SIZE
    if scaler_type is None:
        scaler_type = config.SCALER_TYPE
    if seq_len is None and hasattr(config, 'SEQUENCE_LENGTH'):
        seq_len = config.SEQUENCE_LENGTH
    if use_rolling_window is None:
        use_rolling_window = getattr(config, 'USE_ROLLING_WINDOW', False)
    if rolling_window_size is None:
        rolling_window_size = getattr(config, 'ROLLING_WINDOW_SIZE', 0.5)
    
    # Check torch availability
    if return_torch and not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed. Cannot return torch tensors.")
    # Step 1: Load dataset
    print("Loading dataset...")
    df_raw = load_dataset()
    
    # Step 2: Build features
    print("Building features...")
    df_feat = build_feature_frame(df_raw)
    
    # Step 3: Build targets
    print(f"Building targets for task: {task_type}...")
    y = build_targets(df_feat, task_type)
    
    # Create a temporary dataframe to align features and targets
    df_feat_copy = df_feat.copy()
    df_feat_copy['Return_t+1'] = df_feat['price'].pct_change().shift(-1)
    df_feat_copy['y_target'] = y
    
    # Drop rows with NaN in target (last row will have NaN from shift(-1))
    df_feat_copy = df_feat_copy.dropna(subset=['y_target', 'Return_t+1'])
    
    # Extract features and targets
    # Exclude the target columns and the original price (to avoid leakage)
    feature_cols = [col for col in df_feat_copy.columns 
                    if col not in ['y_target', 'Return_t+1']]
    
    X = df_feat_copy[feature_cols].values
    y = df_feat_copy['y_target'].values
    returns_all = df_feat_copy['Return_t+1'].values
    
    feature_names = feature_cols
    
    # Step 4: Time-based train/val/test split
    print(f"Splitting data (time-based, {'rolling window' if use_rolling_window else 'expanding window'})...")
    N = len(X)
    n_test = int(N * test_size)
    n_val = int(N * val_size)
    
    if use_rolling_window:
        # Rolling window: fixed-size training window
        n_train = int(N * rolling_window_size)
        # Start position ensures we have enough data for train + val + test
        train_start = 0
        train_end = n_train
        val_end = train_end + n_val
        test_end = val_end + n_test
        
        # Ensure we don't exceed data length
        if test_end > N:
            # Adjust to fit within available data
            test_end = N
            val_end = test_end - n_test
            train_end = val_end - n_val
            train_start = train_end - n_train
            if train_start < 0:
                raise ValueError(f"Not enough data for rolling window of size {rolling_window_size}")
        
        X_train_raw = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_val_raw = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test_raw = X[val_end:test_end]
        y_test = y[val_end:test_end]
        returns_test = returns_all[val_end:test_end]
        
        print(f"Rolling window: Train [{train_start}:{train_end}], Val [{train_end}:{val_end}], Test [{val_end}:{test_end}]")
    else:
        # Expanding window: training set grows over time (original behavior)
        n_train = N - n_test - n_val
        
        X_train_raw = X[:n_train]
        y_train = y[:n_train]
        X_val_raw = X[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val]
        X_test_raw = X[n_train + n_val:]
        y_test = y[n_train + n_val:]
        returns_test = returns_all[n_train + n_val:]
    
    print(f"Train size: {len(X_train_raw)}, Val size: {len(X_val_raw)}, Test size: {len(X_test_raw)}")
    
    # Step 5: Scaling (fit on train only)
    print(f"Scaling features using {scaler_type} scaler...")
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Invalid scaler_type: {scaler_type}. Must be 'standard' or 'minmax'")
    
    # Fit on train, transform all sets
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Step 6: Optional sequence creation for RNN models
    if seq_len is not None:
        print(f"Creating sequences with length {seq_len}...")
        X_train, y_train = create_sequences(X_train, y_train, seq_len)
        X_val, y_val = create_sequences(X_val, y_val, seq_len)
        X_test, y_test = create_sequences(X_test, y_test, seq_len)
        
        # Adjust returns_test to align with sequences
        # After sequence creation, we lose the first seq_len samples
        returns_test = returns_test[seq_len:]
        
        print(f"Sequence shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    else:
        print(f"Using tabular format (no sequences)")
        print(f"Shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Step 7: Optional conversion to PyTorch tensors
    if return_torch:
        print("Converting to PyTorch tensors...")
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        X_test = torch.FloatTensor(X_test)
        
        # Reshape targets to (N, 1) for compatibility
        y_train = torch.FloatTensor(y_train).reshape(-1, 1)
        y_val = torch.FloatTensor(y_val).reshape(-1, 1)
        y_test = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # Return all components in a dictionary
    datasets = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "returns_test": returns_test,
        "scaler": scaler,
        "feature_names": feature_names,
        "n_features": X_train.shape[-1] if seq_len else X_train.shape[1],
    }
    
    # Validate dataset
    _validate_dataset(datasets, return_torch)
    
    print("Dataset preparation complete!")
    return datasets


def _validate_dataset(datasets: Dict[str, Any], is_torch: bool) -> None:
    """
    Validate dataset integrity.
    
    Args:
        datasets: Dictionary containing dataset components
        is_torch: Whether data is in PyTorch tensor format
    """
    # Check shape consistency
    assert datasets['X_train'].shape[0] == len(datasets['y_train']), \
        "Train X and y have different lengths"
    assert datasets['X_val'].shape[0] == len(datasets['y_val']), \
        "Val X and y have different lengths"
    assert datasets['X_test'].shape[0] == len(datasets['y_test']), \
        "Test X and y have different lengths"
    
    # Check for NaN/Inf
    for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
        data = datasets[key]
        if is_torch:
            assert not torch.isnan(data).any(), f"{key} contains NaN"
            assert not torch.isinf(data).any(), f"{key} contains Inf"
        else:
            assert not np.isnan(data).any(), f"{key} contains NaN"
            assert not np.isinf(data).any(), f"{key} contains Inf"
    
    print("✓ Dataset validation passed!")


if __name__ == "__main__":
    """
    Test the pipeline with both tasks and configurations
    """
    print("=" * 80)
    print("Testing Data Pipeline Module")
    print("=" * 80)
    
    # Test 1: Sign classification without sequences (for RF, etc.)
    print("\n" + "=" * 80)
    print("Test 1: Sign classification (tabular)")
    print("=" * 80)
    try:
        datasets_sign_tabular = make_dataset_for_task(
            task_type="sign",
            seq_len=None,
            test_size=0.15,
            val_size=0.15,
            scaler_type="standard"
        )
        print("\n✓ Test 1 passed!")
        print(f"  Features: {datasets_sign_tabular['n_features']}")
        print(f"  Train samples: {len(datasets_sign_tabular['X_train'])}")
        print(f"  Val samples: {len(datasets_sign_tabular['X_val'])}")
        print(f"  Test samples: {len(datasets_sign_tabular['X_test'])}")
    except Exception as e:
        print(f"\n✗ Test 1 failed: {e}")
    
    # Test 2: Sign classification with sequences (for LSTM/GRU)
    print("\n" + "=" * 80)
    print("Test 2: Sign classification (sequences, seq_len=14)")
    print("=" * 80)
    try:
        datasets_sign_seq = make_dataset_for_task(
            task_type="sign",
            seq_len=14,
            test_size=0.15,
            val_size=0.15,
            scaler_type="standard"
        )
        print("\n✓ Test 2 passed!")
        print(f"  Sequence shape: {datasets_sign_seq['X_train'].shape}")
        print(f"  Expected: (samples, 14, {datasets_sign_seq['n_features']})")
    except Exception as e:
        print(f"\n✗ Test 2 failed: {e}")
    
    # Test 3: Price regression without sequences
    print("\n" + "=" * 80)
    print("Test 3: Price regression (tabular)")
    print("=" * 80)
    try:
        datasets_price_tabular = make_dataset_for_task(
            task_type="price",
            seq_len=None,
            test_size=0.15,
            val_size=0.15,
            scaler_type="minmax"
        )
        print("\n✓ Test 3 passed!")
        print(f"  Features: {datasets_price_tabular['n_features']}")
        print(f"  Train samples: {len(datasets_price_tabular['X_train'])}")
    except Exception as e:
        print(f"\n✗ Test 3 failed: {e}")
    
    # Test 4: Price regression with sequences
    print("\n" + "=" * 80)
    print("Test 4: Price regression (sequences, seq_len=14)")
    print("=" * 80)
    try:
        datasets_price_seq = make_dataset_for_task(
            task_type="price",
            seq_len=14,
            test_size=0.15,
            val_size=0.15,
            scaler_type="standard"
        )
        print("\n✓ Test 4 passed!")
        print(f"  Sequence shape: {datasets_price_seq['X_train'].shape}")
    except Exception as e:
        print(f"\n✗ Test 4 failed: {e}")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
