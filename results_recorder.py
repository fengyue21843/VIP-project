"""
Results Recording Utility

This module provides functions to save and load experiment results,
hyperparameters, and model comparison data.

Usage:
    from results_recorder import save_experiment, load_experiment, save_params
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


RESULTS_DIR = "results"
PARAMS_DIR = os.path.join(RESULTS_DIR, "best_params")


def ensure_dirs():
    """Ensure results directories exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PARAMS_DIR, exist_ok=True)


def _make_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_make_serializable(v) for v in obj)
    return obj


def save_params(
    model_name: str,
    params: Dict[str, Any],
    filepath: str = None
) -> str:
    """
    Save best hyperparameters for a model.

    Args:
        model_name: Name of the model (e.g., 'lightgbm', 'svr')
        params: Dictionary of hyperparameters
        filepath: Optional custom filepath

    Returns:
        Path to saved file
    """
    ensure_dirs()

    if filepath is None:
        filepath = os.path.join(PARAMS_DIR, f"{model_name}_params.json")

    params_serializable = _make_serializable(params)
    params_serializable["_saved_at"] = datetime.now().isoformat()

    with open(filepath, 'w') as f:
        json.dump(params_serializable, f, indent=2)

    print(f"Params saved: {filepath}")
    return filepath


def load_params(model_name: str, filepath: str = None) -> Dict[str, Any]:
    """
    Load saved hyperparameters for a model.

    Args:
        model_name: Name of the model
        filepath: Optional custom filepath

    Returns:
        Dictionary of hyperparameters
    """
    if filepath is None:
        filepath = os.path.join(PARAMS_DIR, f"{model_name}_params.json")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved params found: {filepath}")

    with open(filepath, 'r') as f:
        params = json.load(f)

    # Remove metadata
    params.pop("_saved_at", None)

    return params


def save_experiment(
    experiment_id: str,
    data_split: Dict[str, str],
    models: Dict[str, Dict[str, Any]],
    best_model: str = None,
    notes: str = None,
) -> str:
    """
    Save complete experiment results.

    Args:
        experiment_id: Unique identifier for the experiment
        data_split: Dict with train/val/test date ranges
        models: Dict mapping model names to their results
        best_model: Name of best performing model
        notes: Optional notes about the experiment

    Returns:
        Path to saved file
    """
    ensure_dirs()

    experiment = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "data_split": data_split,
        "models": _make_serializable(models),
        "best_model": best_model,
        "notes": notes,
    }

    filepath = os.path.join(RESULTS_DIR, f"experiment_{experiment_id}.json")

    with open(filepath, 'w') as f:
        json.dump(experiment, f, indent=2)

    print(f"Experiment saved: {filepath}")
    return filepath


def load_experiment(experiment_id: str) -> Dict[str, Any]:
    """
    Load experiment results.

    Args:
        experiment_id: Experiment identifier

    Returns:
        Experiment results dictionary
    """
    filepath = os.path.join(RESULTS_DIR, f"experiment_{experiment_id}.json")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No experiment found: {filepath}")

    with open(filepath, 'r') as f:
        return json.load(f)


def list_experiments() -> List[str]:
    """List all saved experiment IDs."""
    ensure_dirs()

    experiments = []
    for f in os.listdir(RESULTS_DIR):
        if f.startswith("experiment_") and f.endswith(".json"):
            exp_id = f.replace("experiment_", "").replace(".json", "")
            experiments.append(exp_id)

    return sorted(experiments)


def create_comparison_report(
    results_df: pd.DataFrame,
    output_path: str = None
) -> str:
    """
    Create a formatted comparison report from results DataFrame.

    Args:
        results_df: DataFrame with model comparison results
        output_path: Optional output path (default: results/comparison_report.md)

    Returns:
        Path to report file
    """
    ensure_dirs()

    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "comparison_report.md")

    lines = [
        "# Model Comparison Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nTotal models: {len(results_df)}",
        "\n## Summary Statistics\n",
    ]

    # Add key metrics summary
    if "base_MAE" in results_df.columns:
        best_mae = results_df.loc[results_df["base_MAE"].idxmin()]
        lines.append(f"**Best MAE:** {best_mae['model_name']} ({best_mae['base_MAE']:.4f})")

    if "base_Accuracy" in results_df.columns:
        best_acc = results_df.loc[results_df["base_Accuracy"].idxmax()]
        lines.append(f"\n**Best Accuracy:** {best_acc['model_name']} ({best_acc['base_Accuracy']:.4f})")

    if "trading_Sharpe" in results_df.columns:
        best_sharpe = results_df.loc[results_df["trading_Sharpe"].idxmax()]
        lines.append(f"\n**Best Sharpe:** {best_sharpe['model_name']} ({best_sharpe['trading_Sharpe']:.4f})")

    # Add full results table
    lines.append("\n## Full Results\n")
    lines.append("```")
    lines.append(results_df.to_string())
    lines.append("```")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Report saved: {output_path}")
    return output_path


def generate_experiment_id() -> str:
    """Generate a unique experiment ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    """Test the results recorder module."""
    print("Testing Results Recorder Module")
    print("=" * 50)

    # Test saving params
    test_params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 7,
    }
    save_params("test_model", test_params)

    # Test loading params
    loaded_params = load_params("test_model")
    print(f"Loaded params: {loaded_params}")

    # Test saving experiment
    exp_id = generate_experiment_id()
    save_experiment(
        experiment_id=exp_id,
        data_split={
            "train": "2001-01-01 to 2018-12-31",
            "val": "2019-01-01 to 2020-12-31",
            "test": "2021-01-01 to 2022-12-31",
        },
        models={
            "LightGBM": {"MAE": 4.5, "RMSE": 7.2},
            "SVR": {"MAE": 5.1, "RMSE": 8.0},
        },
        best_model="LightGBM",
        notes="Test experiment",
    )

    # List experiments
    print(f"\nSaved experiments: {list_experiments()}")

    print("\n" + "=" * 50)
    print("Results Recorder Test Complete!")
