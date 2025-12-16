"""
Traditional Machine Learning Model Template

This template is for tabular (non-sequence) ML models such as:
- XGBoost
- LightGBM
- SVM (Support Vector Machine)
- Tree-based models (Extra Trees, Gradient Boosting, etc.)
- Linear models (Logistic Regression, Ridge, etc.)

IMPORTANT: Follow these rules for fair comparison:
1. Use config.py for all hyperparameters
2. Set random seed using config.RANDOM_SEED
3. Get data from data_pipeline.make_dataset_for_task()
4. Return at least 'y_pred_test' in the required format

Author: [Your Name]
Date: [Date]
"""

import numpy as np
from typing import Dict, Any

# Import global configuration
import config

# Set random seed for reproducibility
np.random.seed(config.RANDOM_SEED)


def build_model(task_type: str):
    """
    Build your ML model.
    
    Args:
        task_type: "classification" or "regression"
    
    Returns:
        Model object with fit() and predict() methods
    
    TODO: Replace this example XGBoost with your own model
    """
    # Example: XGBoost model
    # Replace with your model: LightGBM, SVM, ExtraTrees, etc.
    
    if task_type == "classification":
        # Classification model
        from xgboost import XGBClassifier
        
        model = XGBClassifier(
            n_estimators=config.MAX_EPOCHS,           # Use config!
            max_depth=10,
            learning_rate=config.LEARNING_RATE,       # Use config!
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.RANDOM_SEED,          # Use config!
            n_jobs=1,                                 # Single thread for fairness
            eval_metric='logloss'
        )
        
    else:  # regression
        # Regression model
        from xgboost import XGBRegressor
        
        model = XGBRegressor(
            n_estimators=config.MAX_EPOCHS,           # Use config!
            max_depth=10,
            learning_rate=config.LEARNING_RATE,       # Use config!
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.RANDOM_SEED,          # Use config!
            n_jobs=1,                                 # Single thread for fairness
            eval_metric='rmse'
        )
    
    return model


def train_and_predict(
    datasets: Dict[str, Any],
    config_dict: dict | None = None
) -> Dict[str, Any]:
    """
    Standard model interface for training and prediction.
    
    This function MUST be implemented with this exact signature.
    It will be called by run_all_models.py for fair comparison.
    
    Args:
        datasets: Dictionary from make_dataset_for_task() containing:
                  - X_train, y_train: Training data
                  - X_val, y_val: Validation data
                  - X_test, y_test: Test data
                  - returns_test: Returns for backtesting
                  - scaler: Fitted scaler
                  - feature_names: Feature names
        
        config_dict: Optional config override (usually None, uses global config)
    
    Returns:
        Dictionary containing:
        - y_pred_test: Test predictions (REQUIRED)
        - y_pred_val: Validation predictions (recommended)
        - model: Trained model (recommended)
    
    Example:
        >>> from data_pipeline import make_dataset_for_task
        >>> datasets = make_dataset_for_task("sign", seq_len=None)
        >>> result = train_and_predict(datasets)
        >>> print(result['y_pred_test'].shape)
        (300,)
    """
    # ========================================================================
    # Step 1: Extract data from datasets dictionary
    # ========================================================================
    
    X_train = datasets['X_train']
    y_train = datasets['y_train']
    X_val = datasets['X_val']
    y_val = datasets['y_val']
    X_test = datasets['X_test']
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
    
    # Verify data is 2D for tabular models
    assert len(X_train.shape) == 2, (
        f"Expected 2D data (samples, features), got shape {X_train.shape}. "
        f"Make sure seq_len=None in MODEL_REGISTRY for tabular models."
    )
    
    # ========================================================================
    # Step 2: Build model
    # ========================================================================
    
    task_type = "classification" if config.TASK_TYPE == "classification" else "regression"
    print(f"\nBuilding {task_type} model...")
    
    model = build_model(task_type)
    
    print(f"Model: {type(model).__name__}")
    
    # ========================================================================
    # Step 3: Train model
    # ========================================================================
    
    print(f"\nTraining model...")
    print(f"  Random seed: {config.RANDOM_SEED}")
    print(f"  Max iterations/estimators: {config.MAX_EPOCHS}")
    
    # For models that support early stopping (e.g., XGBoost, LightGBM)
    if hasattr(model, 'fit') and 'eval_set' in model.fit.__code__.co_varnames:
        # Train with validation set for early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False  # Set to True to see training progress
        )
    else:
        # Train without validation set
        model.fit(X_train, y_train)
    
    print(f"Training completed!")
    
    # ========================================================================
    # Step 4: Make predictions
    # ========================================================================
    
    print(f"\nGenerating predictions...")
    
    # Predict on validation set
    y_pred_val = model.predict(X_val)
    
    # Predict on test set
    y_pred_test = model.predict(X_test)
    
    # Ensure predictions are 1D arrays
    if len(y_pred_val.shape) > 1:
        y_pred_val = y_pred_val.flatten()
    if len(y_pred_test.shape) > 1:
        y_pred_test = y_pred_test.flatten()
    
    print(f"  Validation predictions: {y_pred_val.shape}")
    print(f"  Test predictions: {y_pred_test.shape}")
    
    # ========================================================================
    # Step 5: Optional - Feature importance (for tree-based models)
    # ========================================================================
    
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        
        # Print top 10 most important features
        if 'feature_names' in datasets:
            feature_names = datasets['feature_names']
            top_indices = np.argsort(feature_importance)[-10:][::-1]
            
            print(f"\nTop 10 most important features:")
            for i, idx in enumerate(top_indices, 1):
                print(f"  {i}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    # ========================================================================
    # Step 6: Return results in standard format
    # ========================================================================
    
    result = {
        'y_pred_test': y_pred_test,        # REQUIRED: Test predictions
        'y_pred_val': y_pred_val,          # Recommended: Validation predictions
        'model': model,                    # Recommended: Trained model
    }
    
    # Add feature importance if available
    if feature_importance is not None:
        result['feature_importance'] = feature_importance
    
    return result


# ============================================================================
# Optional: Test your model locally
# ============================================================================

if __name__ == "__main__":
    """
    Test your model locally before adding to run_all_models.py
    
    Run this file directly:
        python models/model_template_traditional_ml.py
    """
    print("="*80)
    print("Testing Traditional ML Model Template")
    print("="*80)
    
    # Import data pipeline
    from data_pipeline import make_dataset_for_task
    
    # Load data (use same settings as in run_all_models.py)
    print("\n[1/3] Loading data...")
    datasets = make_dataset_for_task(
        task_type=config.get_task_name(),
        seq_len=None,  # None for tabular models
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        scaler_type=config.SCALER_TYPE
    )
    
    # Train and predict
    print("\n[2/3] Training model...")
    result = train_and_predict(datasets)
    
    # Verify results
    print("\n[3/3] Verifying results...")
    
    assert 'y_pred_test' in result, "Missing y_pred_test in result!"
    assert len(result['y_pred_test']) == len(datasets['y_test']), \
        f"Wrong shape! Expected {len(datasets['y_test'])}, got {len(result['y_pred_test'])}"
    
    print(f"\n✅ Model test passed!")
    print(f"  Test predictions shape: {result['y_pred_test'].shape}")
    print(f"  Test labels shape: {datasets['y_test'].shape}")
    
    # Print sample predictions
    print(f"\nSample predictions (first 10):")
    print(f"  Predicted: {result['y_pred_test'][:10]}")
    print(f"  Actual: {datasets['y_test'][:10]}")
    
    # Calculate simple accuracy/error
    if config.TASK_TYPE == "classification":
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(datasets['y_test'], result['y_pred_test'])
        print(f"\nTest Accuracy: {accuracy:.4f}")
    else:
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(datasets['y_test'], result['y_pred_test'])
        r2 = r2_score(datasets['y_test'], result['y_pred_test'])
        print(f"\nTest MAE: {mae:.6f}")
        print(f"Test R²: {r2:.4f}")
    
    print("\n" + "="*80)
    print("✅ Template test completed successfully!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Customize build_model() with your model (XGBoost/LightGBM/SVM/etc.)")
    print("  2. Test again with: python models/model_template_traditional_ml.py")
    print("  3. Add to MODEL_REGISTRY in run_all_models.py")
    print("  4. Run: python run_all_models.py")
