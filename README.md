# VIP Project - Electricity Price Trading Dashboard

A modular quantitative trading system for electricity markets, featuring machine learning models (LSTM/GRU/Random Forest), rule-based strategies, comprehensive backtesting, and an interactive Streamlit dashboard.

## 🚀 Features

### Machine Learning Models
- **LSTM (Long Short-Term Memory)**: Deep learning model for time series forecasting
- **GRU (Gated Recurrent Unit)**: Efficient RNN architecture with comparable performance to LSTM
- **Random Forest**: Traditional ML model for tabular data
- **Optuna Hyperparameter Optimization**: Automatic hyperparameter tuning for GRU models

### Trading Strategies
- **Percentile Channel Breakout**: Buy/sell based on price percentiles
- **Break of Structure (BOS)**: Trend-following strategy

### Professional Visualizations
- **Classification Visualizations**:
  - Prediction Probability Distribution by Classification Result (TP/FP/TN/FN)
  - Calibration Curve (Reliability Diagram)
  - Predictions Timeline
  - ROC Curve with baselines
  - Precision-Recall Curve
  - Threshold Sweep Analysis
  - Metric Comparison Tables

- **Regression Visualizations**:
  - Scatter plots (Predicted vs Actual)
  - Time series plots
  - Residuals vs Time
  - Rolling RMSE
  - Q-Q Plot (Normality Check)
  - Error by Magnitude Analysis
  - Metric Comparison Tables

### Evaluation Metrics
- **Prediction Metrics**: Accuracy, Precision, Recall, F1-Score, AUC (classification); MAE, RMSE, MSE, MAPE, R² (regression)
- **Trading Metrics**: ROI, Win Rate, Sharpe Ratio, Total Trades
- **Baseline Comparisons**: Compare model performance against simple baselines

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/fengyue21843/VIP-project.git
cd VIP-project
```

2. **Create a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### Running the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Using the Dashboard

1. **Select Strategy/Model** from the sidebar:
   - Dashboard: Overview and data statistics
   - Percentile Strategy: Rule-based trading strategy
   - Break of Structure: Trend-following strategy
   - ML Models: Train and evaluate machine learning models
   - Documentation: Project documentation

2. **For ML Models**:
   - Choose task type: Classification (direction) or Regression (price)
   - Select model: LSTM, GRU, or Random Forest
   - Configure hyperparameters
   - Enable Optuna optimization (for GRU) for automatic tuning
   - Train and evaluate the model

3. **View Results**:
   - Prediction metrics
   - Trading metrics
   - Training history plots
   - Comprehensive visualizations

## 📁 Project Structure

```
VIP-project/
├── app.py                          # Streamlit dashboard (main UI)
├── config.py                       # Global configuration settings
├── data_pipeline.py                # Data loading and preprocessing
├── metrics.py                      # Evaluation metrics
├── strategies.py                   # Trading strategies
├── training_utils.py               # Training utilities
├── requirements.txt                # Python dependencies
├── datasets/
│   └── Data_cleaned_Dataset.csv    # Main dataset
├── models/
│   ├── model_lstm.py               # LSTM model implementation
│   ├── model_gru.py                # GRU model with Optuna
│   ├── model_rf.py                 # Random Forest model
│   └── __init__.py
├── notebooks/
│   ├── gru_forecasting_visualizations_pytorch_colab.ipynb
│   ├── gru_forecasting_visualizations_pytorch.ipynb
│   └── gru_forecasting_visualizations.ipynb
└── saved_models/                   # Trained model checkpoints
```

## 🔧 Configuration

Edit `config.py` to customize:
- Task type (classification/regression)
- Sequence length for RNN models
- Train/validation/test split ratios
- Feature scaling method
- Model hyperparameters

## 📊 Data

The project uses electricity market data from `datasets/Data_cleaned_Dataset.csv`:
- **Time Period**: 2001-01-02 to 2022-12-31
- **Features**: Price, volume, natural gas prices, load data, temperature, and engineered features
- **Preprocessing**: Automatic handling of missing values, zero prices, and feature scaling

## 🎓 Key Features

### Optuna Hyperparameter Optimization

For GRU models, enable Optuna to automatically find optimal hyperparameters:
- **Layer Units**: Optimizes GRU layer sizes (32-256 for layer1, 16-128 for layer2)
- **Dropout Rate**: Tests dropout values from 0.1 to 0.6
- **Learning Rate**: Log-uniform search from 1e-5 to 1e-2
- **Batch Size**: Tests [16, 32, 64, 128]
- **Dense Units**: Optimizes final dense layer (16-64)

The best hyperparameters are automatically used for final model training.

### Professional Visualizations

All visualizations are interactive (Plotly) and suitable for:
- Academic presentations
- Professional reports
- Model performance analysis
- Trading strategy evaluation

### Metric Comparison Tables

Compare your model against baselines:
- **Classification**: Majority Class, Random Classifier
- **Regression**: Zero Baseline, Mean Baseline

## 🐛 Bug Fixes

- **Fixed metrics calculation**: Resolved Precision/Recall/F1 showing 0.0000
- **Fixed BrokenPipeError**: Added safe_print() for Streamlit compatibility
- **Improved error handling**: Better diagnostics for model training

## 📈 Usage Examples

### Training a GRU Model with Optuna

1. Navigate to "ML Models" in the dashboard
2. Select "GRU" as model type
3. Enable "Hyperparameter Optimization (Optuna)"
4. Set number of trials (default: 20)
5. Click "Train and Evaluate Model"

### Running a Trading Strategy

1. Navigate to "Percentile Strategy" or "Break of Structure"
2. Configure strategy parameters
3. Select backtest period
4. Click "Run Strategy"
5. View results and portfolio balance chart

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## 📝 License

This project is open source. Please check the repository for license details.

## 🙏 Acknowledgments

- Built with Streamlit for the interactive dashboard
- Uses PyTorch for deep learning models
- Optuna for hyperparameter optimization
- Plotly for interactive visualizations

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🔄 Recent Updates

- ✅ Added Optuna hyperparameter optimization for GRU models
- ✅ Added comprehensive professional visualizations
- ✅ Fixed classification metrics calculation bug
- ✅ Added metric comparison tables with baselines
- ✅ Improved error handling and diagnostics
- ✅ Added Q-Q plots and residual analysis for regression
- ✅ Enhanced UI with better organization and tabs

---

**Note**: Make sure you have the dataset file (`Data_cleaned_Dataset.csv`) in the `datasets/` directory before running the application.

