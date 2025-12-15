"""
Step 4: Unified Experiment Runner

This script orchestrates the complete ML pipeline:
1. Loads data using unified data pipeline (Step 1)
2. Trains models using standard interface (Step 2)
3. Evaluates using unified metrics (Step 3)
4. Produces comparable performance table

Usage:
    python run_all_models.py

Output:
    - Console: Per-model metrics and summary table
    - File: model_comparison_results.csv
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import traceback

# Step 1: Data pipeline
from data_pipeline import make_dataset_for_task

# Step 2: Models
# Check PyTorch availability
PYTORCH_AVAILABLE = False
try:
    import torch
    PYTORCH_AVAILABLE = True
    from models import model_lstm, model_gru
except ImportError:
    print("Warning: PyTorch not available. LSTM and GRU models will be skipped.")
    model_lstm = None
    model_gru = None

from models import model_rf

# Step 3: Evaluation
from metrics import evaluate_model_outputs

# Configuration
import config


# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY = {}

# Add PyTorch models only if available
if PYTORCH_AVAILABLE and model_lstm is not None:
    MODEL_REGISTRY["LSTM_sign"] = {
        "module": model_lstm,
        "task_type": "sign",
        "seq_len": config.SEQUENCE_LENGTH,
        "description": "2-layer LSTM for direction prediction"
    }
    MODEL_REGISTRY["LSTM_price"] = {
        "module": model_lstm,
        "task_type": "price",
        "seq_len": config.SEQUENCE_LENGTH,
        "description": "2-layer LSTM for price prediction"
    }

if PYTORCH_AVAILABLE and model_gru is not None:
    MODEL_REGISTRY["GRU_sign"] = {
        "module": model_gru,
        "task_type": "sign",
        "seq_len": config.SEQUENCE_LENGTH,
        "description": "2-layer GRU for direction prediction"
    }
    MODEL_REGISTRY["GRU_price"] = {
        "module": model_gru,
        "task_type": "price",
        "seq_len": config.SEQUENCE_LENGTH,
        "description": "2-layer GRU for price prediction"
    }

# Random Forest is always available
MODEL_REGISTRY["RF_sign"] = {
    "module": model_rf,
    "task_type": "sign",
    "seq_len": None,
    "description": "Random Forest for direction prediction"
}
MODEL_REGISTRY["RF_price"] = {
    "module": model_rf,
    "task_type": "price",
    "seq_len": None,
    "description": "Random Forest for price prediction"
}


# =============================================================================
# Helper Functions
# =============================================================================

def run_single_model(name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run one model end-to-end: data → train → evaluate
    
    Args:
        name: Model name (key in registry)
        spec: Model specification dict with keys:
              - module: Python module with train_and_predict()
              - task_type: "sign" or "price"
              - seq_len: Sequence length (int or None for tabular)
              - description: Human-readable description
    
    Returns:
        Dict with all results:
        - model_name: str
        - task_type: str
        - description: str
        - base_*: prediction metrics (MAE/RMSE or Accuracy/F1/etc)
        - trading_*: trading metrics (ROI, WinRate, Sharpe)
    """
    module = spec["module"]
    task_type = spec["task_type"]
    seq_len = spec["seq_len"]
    description = spec.get("description", "")
    
    print(f"\n{'='*80}")
    print(f"Running: {name}")
    print(f"{'='*80}")
    print(f"Task: {task_type}")
    print(f"Sequence length: {seq_len if seq_len else 'None (tabular)'}")
    print(f"Description: {description}")
    print()
    
    # Step 1: Prepare data using unified pipeline
    print(f"[1/3] Loading data...")
    datasets = make_dataset_for_task(
        task_type=task_type,
        seq_len=seq_len,  # None = tabular; int = sequence
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        scaler_type=config.SCALER_TYPE
    )
    print(f"      Train samples: {len(datasets['y_train'])}")
    print(f"      Val samples: {len(datasets['y_val'])}")
    print(f"      Test samples: {len(datasets['y_test'])}")
    
    # Step 2: Train & predict using standard interface
    print(f"\n[2/3] Training model...")
    out = module.train_and_predict(datasets, config=None)
    
    y_test = datasets["y_test"]
    y_pred_test = out["y_pred_test"]
    returns_test = datasets["returns_test"]
    
    print(f"      Predictions generated: {len(y_pred_test)} samples")
    
    # Step 3: Evaluate using unified metrics
    print(f"\n[3/3] Evaluating performance...")
    
    # Map internal task_type ("sign"/"price") to metrics task_type ("classification"/"regression")
    metrics_task_type = "classification" if task_type == "sign" else "regression"
    
    base_metrics, trading_metrics = evaluate_model_outputs(
        task_type=metrics_task_type,
        y_test=y_test,
        y_pred=y_pred_test,
        returns_test=returns_test,
    )
    
    # Combine all results in one flat dict
    result = {
        "model_name": name,
        "task_type": task_type,
        "description": description,
    }
    
    # Flatten metrics with prefixes
    for k, v in base_metrics.items():
        # Handle confusion matrix specially
        if k == "Confusion_Matrix":
            result[f"base_{k}"] = str(v.tolist()) if isinstance(v, np.ndarray) else str(v)
        else:
            result[f"base_{k}"] = v
    
    for k, v in trading_metrics.items():
        # Handle algo_returns_series - summarize instead of full array
        if k == "algo_returns_series":
            result["trading_algo_returns_mean"] = float(np.mean(v))
            result["trading_algo_returns_std"] = float(np.std(v))
        else:
            result[f"trading_{k}"] = v
    
    # Print summary
    print(f"\n      Prediction Metrics:")
    for k, v in base_metrics.items():
        if k != "Confusion_Matrix":
            if isinstance(v, (float, int)):
                print(f"        {k}: {v:.4f}")
            else:
                print(f"        {k}: {v}")
    
    print(f"\n      Trading Metrics:")
    for k, v in trading_metrics.items():
        if k != "algo_returns_series":
            if isinstance(v, float):
                print(f"        {k}: {v:.4f}")
            else:
                print(f"        {k}: {v}")
    
    print(f"\n{'='*80}")
    print(f"Completed: {name}")
    print(f"{'='*80}\n")
    
    return result


def print_summary_table(all_results: List[Dict[str, Any]]):
    """
    Print a formatted summary table of all model results
    """
    if not all_results:
        print("\n[WARNING] No results to display")
        return
    
    df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print(" " * 25 + "MODEL COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    # Display settings
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    # Create a simplified view
    display_cols = ["model_name", "task_type"]
    
    # Add key prediction metrics
    if "base_Accuracy" in df.columns:
        display_cols.extend(["base_Accuracy", "base_F1", "base_AUC"])
    if "base_MAE" in df.columns:
        display_cols.extend(["base_MAE", "base_RMSE"])
    
    # Add key trading metrics
    trading_cols = ["trading_ROI", "trading_WinRate", "trading_Sharpe", "trading_TotalTrades"]
    display_cols.extend([c for c in trading_cols if c in df.columns])
    
    # Filter to available columns
    display_cols = [c for c in display_cols if c in df.columns]
    
    if display_cols:
        df_display = df[display_cols].copy()
        
        # Format numeric columns
        for col in df_display.columns:
            if df_display[col].dtype in ['float64', 'float32']:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
        
        print(df_display.to_string(index=False))
        print()
    
    # Print full details
    print("\nFull Results:")
    print("-" * 80)
    print(df.to_string(index=False))
    print()


def save_results(all_results: List[Dict[str, Any]], filename: str = "model_comparison_results.csv"):
    """
    Save results to CSV file
    """
    if not all_results:
        print("\n[WARNING] No results to save")
        return
    
    df = pd.DataFrame(all_results)
    df.to_csv(filename, index=False)
    print(f"\n[SUCCESS] Results saved to: {filename}")
    print(f"          Total models: {len(all_results)}")
    print(f"          Columns: {len(df.columns)}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """
    Main experiment runner:
    1. Iterate over MODEL_REGISTRY
    2. Run each model end-to-end
    3. Collect results
    4. Print summary table
    5. Save to CSV
    """
    print("\n" + "="*80)
    print(" " * 20 + "STEP 4: UNIFIED EXPERIMENT RUNNER")
    print(" " * 15 + "Running All Models with Consistent Pipeline")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Total models in registry: {len(MODEL_REGISTRY)}")
    print(f"  Random seed: {config.RANDOM_SEED}")
    print(f"  Data path: {config.DATA_PATH}")
    print()
    
    all_results = []
    failed_models = []
    
    # Run each model
    for name, spec in MODEL_REGISTRY.items():
        try:
            result = run_single_model(name, spec)
            all_results.append(result)
            
        except Exception as e:
            print(f"\n[ERROR] Model {name} failed with error:")
            print(f"        {str(e)}")
            print("\n        Stack trace:")
            traceback.print_exc()
            print()
            failed_models.append({"model_name": name, "error": str(e)})
    
    # Print summary
    print("\n" + "="*80)
    print(" " * 30 + "EXECUTION SUMMARY")
    print("="*80)
    print(f"\nSuccessful models: {len(all_results)}/{len(MODEL_REGISTRY)}")
    print(f"Failed models: {len(failed_models)}/{len(MODEL_REGISTRY)}")
    
    if failed_models:
        print("\nFailed models:")
        for fail in failed_models:
            print(f"  - {fail['model_name']}: {fail['error']}")
    
    # Display results table
    if all_results:
        print_summary_table(all_results)
        
        # Save to CSV
        save_results(all_results, "model_comparison_results.csv")
        
        # Print best models
        print("\n" + "="*80)
        print(" " * 28 + "BEST MODEL BY METRIC")
        print("="*80 + "\n")
        
        df = pd.DataFrame(all_results)
        
        # Best by prediction metrics
        if "base_Accuracy" in df.columns:
            best_acc = df.loc[df["base_Accuracy"].idxmax()]
            print(f"Best Accuracy: {best_acc['model_name']} ({best_acc['base_Accuracy']:.4f})")
        
        if "base_MAE" in df.columns:
            best_mae = df.loc[df["base_MAE"].idxmin()]
            print(f"Best MAE: {best_mae['model_name']} ({best_mae['base_MAE']:.4f})")
        
        # Best by trading metrics
        if "trading_ROI" in df.columns:
            best_roi = df.loc[df["trading_ROI"].idxmax()]
            print(f"Best ROI: {best_roi['model_name']} ({best_roi['trading_ROI']:.4f}%)")
        
        if "trading_Sharpe" in df.columns:
            best_sharpe = df.loc[df["trading_Sharpe"].idxmax()]
            print(f"Best Sharpe: {best_sharpe['model_name']} ({best_sharpe['trading_Sharpe']:.4f})")
        
        if "trading_WinRate" in df.columns:
            best_winrate = df.loc[df["trading_WinRate"].idxmax()]
            print(f"Best Win Rate: {best_winrate['model_name']} ({best_winrate['trading_WinRate']:.4f}%)")
        
        print()
    
    print("\n" + "="*80)
    print(" " * 25 + "EXPERIMENT COMPLETE!")
    print("="*80 + "\n")
    
    print("Next Steps:")
    print("  1. Review model_comparison_results.csv for detailed metrics")
    print("  2. Analyze best-performing models by task type")
    print("  3. Consider ensemble strategies (Step 5)")
    print("  4. Build interactive dashboard for visualization")
    print()


if __name__ == "__main__":
    main()
