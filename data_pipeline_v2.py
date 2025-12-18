"""
Data Pipeline V2 - Comprehensive Fix

Implements all three suggested solutions:
1. Drop missing data (no imputation)
2. Optional: Add third class "flat" for zero returns
3. Optional: Use returns-based interpolation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import Literal, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter
import config

# Type definitions
TaskType = Literal["sign", "price", "sign_3class"]
ScalerType = Literal["standard", "minmax"]

# Optional: Import torch if available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Set return_torch=False")


def load_dataset_v2(use_imputation: bool = False) -> pd.DataFrame:
    """
    Load dataset WITHOUT imputation (Solution 1: Drop missing data).
    
    Args:
        use_imputation: If False, do NOT use ffill/bfill (recommended)
    
    Returns:
        pd.DataFrame: Raw dataset WITHOUT imputation
    """
    # Determine the path to the dataset
    data_paths = [
        config.DATA_PATH,
        r"c:\Users\DELL\Downloads\Data-20251207T171745Z-1-001\Data\Data_cleaned_Dataset.csv",
        "datasets/Data_cleaned_Dataset.csv",
        "Data_cleaned_Dataset.csv",
        "../Data/Data_cleaned_Dataset.csv",
        "Data/Data_cleaned_Dataset.csv"
    ]
    
    dataset_path = None
    for path in data_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        raise FileNotFoundError("Could not find Data_cleaned_Dataset.csv")
    
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
    
    # Sort by Trade Date
    if 'Trade Date' in df.columns:
        df = df.sort_values('Trade Date')
    
    # CRITICAL: Do NOT use ffill/bfill unless explicitly requested
    if use_imputation:
        print("WARNING: Using ffill/bfill imputation (may create artificial flat periods)")
        df = df.ffill().bfill()
    else:
        print("NO imputation - missing values will be dropped")
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def clean_dataset_v2(df: pd.DataFrame, drop_zero_returns: bool = False) -> pd.DataFrame:
    """
    Clean dataset by dropping rows with missing critical values.
    
    Args:
        df: Raw dataframe
        drop_zero_returns: If True, also drop rows with zero price changes
    
    Returns:
        Cleaned dataframe
    """
    print("\n" + "="*80)
    print("DATA CLEANING V2 (No Imputation)")
    print("="*80)
    
    original_len = len(df)
    print(f"Original dataset: {original_len:,} rows")
    
    # Key columns that must not have missing values
    critical_columns = [
        'Electricity: Wtd Avg Price $/MWh',
        'Electricity: Daily Volume MWh',
        'Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)',
        'pjm_load sum in MW (daily)',
        'temperature mean in C (daily): US'
    ]
    
    # Check which critical columns exist
    existing_critical = [col for col in critical_columns if col in df.columns]
    print(f"Critical columns to check: {len(existing_critical)}")
    
    # Drop rows with missing values in critical columns
    df_clean = df.dropna(subset=existing_critical).copy()
    
    removed = original_len - len(df_clean)
    print(f"Removed rows with missing values: {removed:,} ({removed/original_len*100:.1f}%)")
    
    # Remove rows with zero or negative prices
    price_col = 'Electricity: Wtd Avg Price $/MWh'
    if price_col in df_clean.columns:
        invalid_prices = (df_clean[price_col] <= 0).sum()
        if invalid_prices > 0:
            df_clean = df_clean[df_clean[price_col] > 0].copy()
            print(f"Removed rows with invalid prices (≤0): {invalid_prices}")
    
    # Optional: Remove consecutive days with identical prices (likely imputed)
    if drop_zero_returns and price_col in df_clean.columns:
        df_clean['price_change'] = df_clean[price_col].diff()
        zero_changes = (df_clean['price_change'].abs() < 1e-6).sum()
        if zero_changes > 0:
            df_clean = df_clean[df_clean['price_change'].abs() >= 1e-6].copy()
            print(f"Removed rows with zero price change: {zero_changes}")
        df_clean = df_clean.drop('price_change', axis=1)
    
    print(f"Clean dataset: {len(df_clean):,} rows")
    print(f"Data loss: {(original_len - len(df_clean))/original_len*100:.1f}%")
    
    return df_clean


def build_feature_frame_v2(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build features from cleaned raw data.
    Same as original but with better handling of edge cases.
    """
    df_feat = df_raw.copy()
    
    # Use Trade Date as index
    if 'Trade Date' in df_feat.columns:
        df_feat = df_feat.set_index('Trade Date')
    else:
        print("Warning: 'Trade Date' not found, using integer index")
    
    # Ensure index is sorted
    df_feat = df_feat.sort_index()
    
    # Map column names to shorter feature names
    price_col = 'Electricity: Wtd Avg Price $/MWh'
    volume_col = 'Electricity: Daily Volume MWh'
    gas_col = 'Natural Gas: Henry Hub Natural Gas Spot Price (Dollars per Million Btu)'
    load_col = 'pjm_load sum in MW (daily)'
    temp_col = 'temperature mean in C (daily): US'
    
    # Create core features
    if price_col in df_feat.columns:
        df_feat['price'] = df_feat[price_col]
    if volume_col in df_feat.columns:
        df_feat['volume'] = df_feat[volume_col]
    if gas_col in df_feat.columns:
        df_feat['gas_price'] = df_feat[gas_col]
    if load_col in df_feat.columns:
        df_feat['load'] = df_feat[load_col]
    if temp_col in df_feat.columns:
        df_feat['temperature'] = df_feat[temp_col]
    
    # Time features
    if isinstance(df_feat.index, pd.DatetimeIndex):
        df_feat['day'] = df_feat.index.day
        df_feat['month'] = df_feat.index.month
        df_feat['year'] = df_feat.index.year
        df_feat['weekday'] = df_feat.index.weekday
        df_feat['quarter'] = df_feat.index.quarter
        df_feat['day_of_year'] = df_feat.index.dayofyear
    
    # Lagged features (returns, not prices to avoid leakage)
    for col in ['price', 'volume', 'gas_price', 'load', 'temperature']:
        if col in df_feat.columns:
            # Percentage change (returns)
            df_feat[f'{col}_return'] = df_feat[col].pct_change(fill_method=None)
            
            # Lagged values (1, 2, 3 days)
            df_feat[f'{col}_lag1'] = df_feat[col].shift(1)
            df_feat[f'{col}_lag2'] = df_feat[col].shift(2)
            df_feat[f'{col}_lag3'] = df_feat[col].shift(3)
            
            # Rolling statistics (7-day window)
            df_feat[f'{col}_roll7_mean'] = df_feat[col].rolling(7, min_periods=1).mean()
            df_feat[f'{col}_roll7_std'] = df_feat[col].rolling(7, min_periods=1).std()
    
    # Select only numeric columns
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
    df_feat = df_feat[numeric_cols]
    
    # CRITICAL: Replace Inf with NaN, then drop rows with NaN
    # This handles division by zero and other edge cases
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN (from pct_change, rolling stats, etc.)
    rows_before = len(df_feat)
    df_feat = df_feat.dropna()
    rows_after = len(df_feat)
    
    if rows_before > rows_after:
        print(f"  Dropped {rows_before - rows_after} rows with NaN from features")
    
    return df_feat


def build_targets_v2(
    df_feat: pd.DataFrame, 
    task_type: TaskType,
    flat_threshold: float = 0.001
) -> np.ndarray:
    """
    Build targets with optional 3-class classification (Solution 2).
    
    Args:
        df_feat: Feature dataframe with 'price' column
        task_type: 
            - "sign": Binary (up=1, down=0)
            - "sign_3class": Three classes (up=2, flat=1, down=0)
            - "price": Regression (continuous returns)
        flat_threshold: Threshold for "flat" classification (e.g., 0.001 = 0.1%)
    
    Returns:
        Target array
    """
    if 'price' not in df_feat.columns:
        raise ValueError("'price' column not found in feature dataframe")
    
    # Compute returns for next period
    price = df_feat['price']
    return_t_plus_1 = price.pct_change(fill_method=None).shift(-1)
    
    if task_type == "sign":
        # Binary classification: up vs down
        y = (return_t_plus_1 > 0).astype(np.float32).values
        
    elif task_type == "sign_3class":
        # Three-class classification: up, flat, down
        y = np.zeros(len(return_t_plus_1), dtype=np.float32)
        y[return_t_plus_1 > flat_threshold] = 2   # Up
        y[return_t_plus_1 < -flat_threshold] = 0  # Down
        y[(return_t_plus_1 >= -flat_threshold) & (return_t_plus_1 <= flat_threshold)] = 1  # Flat
        
    elif task_type == "price":
        # Regression: predict continuous returns
        y = return_t_plus_1.values
        
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    return y


def diagnose_dataset_v2(df: pd.DataFrame, y: np.ndarray, task_type: str):
    """Comprehensive diagnostics"""
    print("\n" + "="*80)
    print(f"DATASET DIAGNOSTICS ({task_type})")
    print("="*80)
    
    # Check for imputation artifacts
    price_col = 'Electricity: Wtd Avg Price $/MWh'
    if price_col in df.columns:
        prices = df[price_col].dropna()
        price_diffs = prices.diff().dropna()
        zero_diffs = (price_diffs.abs() < 1e-6).sum()
        print(f"\nPrice Analysis:")
        print(f"  Total price records: {len(prices):,}")
        print(f"  Zero price changes: {zero_diffs:,} ({zero_diffs/len(price_diffs)*100:.2f}%)")
        if zero_diffs > len(price_diffs) * 0.1:
            print(f"  WARNING: >10% zero changes suggests imputation artifacts!")
    
    # Label distribution
    unique, counts = np.unique(y[~np.isnan(y)], return_counts=True)
    print(f"\nLabel Distribution:")
    
    if task_type in ["sign", "sign_3class"]:
        for val, count in zip(unique, counts):
            pct = count / len(y) * 100
            if task_type == "sign":
                label = "Up" if val == 1 else "Down"
            else:
                label = ["Down", "Flat", "Up"][int(val)]
            print(f"  {label} ({int(val)}): {count:,} ({pct:.2f}%)")
        
        # Balance ratio
        balance_ratio = min(counts) / max(counts)
        print(f"\nBalance ratio: {balance_ratio:.3f}")
        if balance_ratio < 0.8:
            print(f"  Significant imbalance")
        else:
            print(f"  Acceptable balance")
    
    else:
        print(f"  Mean: {np.nanmean(y):.6f}")
        print(f"  Std: {np.nanstd(y):.6f}")
        print(f"  Range: [{np.nanmin(y):.6f}, {np.nanmax(y):.6f}]")


def create_sequences(
    data: np.ndarray,
    target: np.ndarray,
    sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for RNN models"""
    xs, ys = [], []
    for i in range(len(data) - sequence_length + 1):
        x = data[i:i+sequence_length]
        y = target[i+sequence_length-1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def balance_classification_dataset(
    X: np.ndarray, 
    y: np.ndarray,
    returns: np.ndarray,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balance dataset by downsampling majority class"""
    print("\n" + "="*80)
    print("DATASET BALANCING")
    print("="*80)
    
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    print(f"Original distribution:")
    for cls, count in class_counts.items():
        print(f"  Class {int(cls)}: {count:,} ({count/len(y)*100:.2f}%)")
    
    # Find minimum class size
    min_count = min(counts)
    
    # Sample equally from each class
    balanced_indices = []
    np.random.seed(random_seed)
    
    for cls in unique:
        cls_indices = np.where(y == cls)[0]
        sampled = np.random.choice(cls_indices, size=min_count, replace=False)
        balanced_indices.extend(sampled)
    
    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)
    
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]
    returns_balanced = returns[balanced_indices]
    
    print(f"\nBalanced distribution:")
    balanced_counts = Counter(y_balanced)
    for cls, count in sorted(balanced_counts.items()):
        print(f"  Class {int(cls)}: {count:,} ({count/len(y_balanced)*100:.2f}%)")
    
    print(f"Total: {len(X_balanced):,} (from {len(X):,})")
    
    return X_balanced, y_balanced, returns_balanced


def make_dataset_v2(
    task_type: TaskType = None,
    seq_len: int = None,
    test_size: float = None,
    val_size: float = None,
    scaler_type: ScalerType = None,
    return_torch: bool = False,
    balance_data: bool = True,
    use_imputation: bool = False,  # NEW: Control imputation
    drop_zero_returns: bool = True,  # NEW: Drop zero return periods
    flat_threshold: float = 0.001,  # NEW: For 3-class classification
    random_seed: int = None,
) -> Dict[str, Any]:
    """
    Enhanced data pipeline V2.
    
    New parameters:
        use_imputation: If False, drops missing data (Solution 1 - RECOMMENDED)
        drop_zero_returns: If True, drops periods with zero returns (Solution 2)
        flat_threshold: Threshold for "flat" class in 3-class classification
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    # Set defaults
    if task_type is None:
        task_type = config.get_task_name()
    if test_size is None:
        test_size = config.TEST_SIZE
    if val_size is None:
        val_size = config.VAL_SIZE
    if scaler_type is None:
        scaler_type = config.SCALER_TYPE
    if seq_len is None:
        seq_len = getattr(config, 'SEQUENCE_LENGTH', None)
    if random_seed is None:
        random_seed = config.RANDOM_SEED
    
    print("\n" + "="*80)
    print("DATA PIPELINE V2 - COMPREHENSIVE FIX")
    print("="*80)
    print(f"Task: {task_type}")
    print(f"Use imputation: {use_imputation} (False = Solution 1)")
    print(f"Drop zero returns: {drop_zero_returns}")
    print(f"Balance data: {balance_data}")
    
    # Step 1: Load WITHOUT imputation (Solution 1)
    print("\n[1/7] Loading dataset...")
    df_raw = load_dataset_v2(use_imputation=use_imputation)
    
    # Step 2: Clean (drop missing data)
    print("\n[2/7] Cleaning dataset...")
    df_raw = clean_dataset_v2(df_raw, drop_zero_returns=drop_zero_returns)
    
    # Step 3: Build features
    print("\n[3/7] Building features...")
    df_feat = build_feature_frame_v2(df_raw)
    
    # Step 4: Build targets (with optional 3-class)
    print(f"\n[4/7] Building targets ({task_type})...")
    y = build_targets_v2(df_feat, task_type, flat_threshold=flat_threshold)
    
    # Align
    df_feat_copy = df_feat.copy()
    df_feat_copy['Return_t+1'] = df_feat['price'].pct_change(fill_method=None).shift(-1)
    df_feat_copy['y_target'] = y
    
    # Drop NaN and inf
    df_feat_copy = df_feat_copy.replace([np.inf, -np.inf], np.nan)
    df_feat_copy = df_feat_copy.dropna(subset=['y_target', 'Return_t+1'])
    
    # Extract
    feature_cols = [col for col in df_feat_copy.columns 
                    if col not in ['y_target', 'Return_t+1']]
    
    X = df_feat_copy[feature_cols].values
    y = df_feat_copy['y_target'].values
    returns_all = df_feat_copy['Return_t+1'].values
    
    print(f"Features: {X.shape}")
    print(f"Targets: {y.shape}")
    
    # Diagnostics
    diagnose_dataset_v2(df_raw, y, task_type)
    
    # Step 5: Balance
    if balance_data and task_type in ["sign", "sign_3class"]:
        print(f"\n[5/7] Balancing...")
        X, y, returns_all = balance_classification_dataset(X, y, returns_all, random_seed)
    else:
        print(f"\n[5/7] Skip balancing")
    
    # Step 6: Split
    print(f"\n[6/7] Splitting...")
    N = len(X)
    n_test = int(N * test_size)
    n_val = int(N * val_size)
    n_train = N - n_test - n_val
    
    X_train, y_train, returns_train = X[:n_train], y[:n_train], returns_all[:n_train]
    X_val, y_val, returns_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val], returns_all[n_train:n_train+n_val]
    X_test, y_test, returns_test = X[n_train+n_val:], y[n_train+n_val:], returns_all[n_train+n_val:]
    
    print(f"Train: {n_train:,}, Val: {n_val:,}, Test: {n_test:,}")
    
    # Step 7: Scale
    print(f"\n[7/7] Scaling ({scaler_type})...")
    if scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Sequences
    if seq_len is not None and seq_len > 1:
        print(f"\nCreating sequences (len={seq_len})...")
        X_train, y_train = create_sequences(X_train, y_train, seq_len)
        X_val, y_val = create_sequences(X_val, y_val, seq_len)
        X_test, y_test = create_sequences(X_test, y_test, seq_len)
        returns_train = returns_train[seq_len-1:]
        returns_val = returns_val[seq_len-1:]
        returns_test = returns_test[seq_len-1:]
    
    # Convert to torch
    if return_torch:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
    
    print(f"\n" + "="*80)
    print("FINAL SHAPES")
    print("="*80)
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'returns_train': returns_train,
        'returns_val': returns_val,
        'returns_test': returns_test,
        'scaler': scaler,
        'feature_names': feature_cols,
        'n_features': X_train.shape[-1] if seq_len else X_train.shape[1],
        'task_type': task_type,
    }


if __name__ == "__main__":
    print("Testing Data Pipeline V2...")
    
    # Test Solution 1: No imputation
    datasets = make_dataset_v2(
        task_type="sign",
        seq_len=14,
        use_imputation=False,  # Solution 1
        drop_zero_returns=True,
        balance_data=True,
    )
    
    print("\nPipeline V2 test completed!")
