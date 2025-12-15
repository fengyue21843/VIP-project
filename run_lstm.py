"""
Run LSTM Model - Choose Classification or Regression Task

Usage:
    python run_lstm.py classification  # Price direction prediction
    python run_lstm.py regression      # Price value prediction
    python run_lstm.py                 # Default: run classification task
"""

import sys
import numpy as np
from data_pipeline import make_dataset_for_task
from models.model_lstm import train_and_predict
from config import LSTM_CONFIG, MAX_EPOCHS, BATCH_SIZE, EARLY_STOP_PATIENCE


def run_lstm(task_type: str = "classification"):
    """
    Run LSTM training and prediction
    
    Args:
        task_type: "classification" or "regression"
    """
    print("=" * 80)
    print(f"LSTM Model - {task_type.upper()} Task")
    print("=" * 80)
    
    # 加载数据
    data_task = "sign" if task_type == "classification" else "price"
    print(f"\nLoading data for {data_task} task...")
    
    datasets = make_dataset_for_task(
        task_type=data_task,
        seq_len=14,
        scaler_type="standard"
    )
    
    print(f"\nData Shapes:")
    print(f"  Train: {datasets['X_train'].shape}")
    print(f"  Val:   {datasets['X_val'].shape}")
    print(f"  Test:  {datasets['X_test'].shape}")
    
    # 显示配置
    print("\n" + "=" * 80)
    print("LSTM Configuration")
    print("=" * 80)
    print(f"  Layer 1 units:      {LSTM_CONFIG['layer1_units']}")
    print(f"  Layer 2 units:      {LSTM_CONFIG['layer2_units']}")
    print(f"  Dense units:        {LSTM_CONFIG['dense_units']}")
    print(f"  Dropout rate:       {LSTM_CONFIG['dropout_rate']}")
    print(f"  Use attention:      {LSTM_CONFIG['use_attention']}")
    print(f"  Attention heads:    {LSTM_CONFIG['attention_heads']}")
    print(f"  Max epochs:         {MAX_EPOCHS}")
    print(f"  Batch size:         {BATCH_SIZE}")
    print(f"  Early stop patience: {EARLY_STOP_PATIENCE}")
    
    # 训练模型
    print("\n" + "=" * 80)
    print("Training LSTM Model")
    print("=" * 80)
    
    # 添加任务类型到配置
    train_config = LSTM_CONFIG.copy()
    train_config['task_type'] = task_type
    
    results = train_and_predict(datasets, config=train_config)
    
    # 评估结果
    print("\n" + "=" * 80)
    print("Test Set Results")
    print("=" * 80)
    
    y_pred_test = results["y_pred_test"]
    y_test = datasets["y_test"]
    
    if task_type == "classification":
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        y_pred_binary = (y_pred_test > 0.5).astype(int)
        
        acc = accuracy_score(y_test, y_pred_binary)
        prec = precision_score(y_test, y_pred_binary, zero_division=0)
        rec = recall_score(y_test, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test, y_pred_binary, zero_division=0)
        cm = confusion_matrix(y_test, y_pred_binary)
        
        print(f"\nClassification Metrics:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]:<6} FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]:<6} TP: {cm[1,1]}")
        
    else:  # regression
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        
        print(f"\nRegression Metrics:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  R²:   {r2:.4f}")
        
        # Show sample predictions
        print(f"\nSample Predictions (first 10):")
        print(f"  True:      {y_test[:10]}")
        print(f"  Predicted: {y_pred_test[:10]}")
    
    # Training history summary
    history = results["history"]
    print(f"\nTraining Summary:")
    print(f"  Epochs trained:   {len(history['loss'])}")
    print(f"  Best val_loss:    {min(history['val_loss']):.6f}")
    print(f"  Final train loss: {history['loss'][-1]:.6f}")
    
    print("\n" + "=" * 80)
    print("LSTM Training Complete!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Get task type from command line arguments
    if len(sys.argv) > 1:
        task = sys.argv[1].lower()
    else:
        task = "classification"
    
    # Validate task type
    if task not in ["classification", "regression", "cls", "reg"]:
        print(f"Error: Unknown task type '{task}'")
        print("\nUsage:")
        print("  python run_lstm.py classification  # Price direction prediction (classification)")
        print("  python run_lstm.py regression      # Price value prediction (regression)")
        print("  python run_lstm.py cls             # Short form of classification")
        print("  python run_lstm.py reg             # Short form of regression")
        sys.exit(1)
    
    # Handle short forms
    if task == "cls":
        task = "classification"
    elif task == "reg":
        task = "regression"
    
    # Run LSTM
    run_lstm(task)
