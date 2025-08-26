# Improved Stock Price Prediction using LSTM with Twitter Sentiment Analysis

**Authors:** Leo Chang, Aditya Saraf, Jenjen Chen

This repository contains an improved stock price prediction system that addresses feedback from the Journal of Emerging Investigators (JEI) editorial review. The implementation provides enhanced accuracy, statistical rigor, and professional visualization suitable for academic publication.

## 🎯 Key Improvements

### Scientific Enhancements
- **Multi-stock Analysis**: Extended analysis to AAPL, TSLA, and MSFT for generalizability testing
- **Cross-Validation**: Implemented 5-fold time series cross-validation for uncertainty quantification
- **Statistical Testing**: Added paired t-tests and Wilcoxon signed-rank tests for model comparisons
- **Enhanced Feature Engineering**: Technical indicators (RSI, MACD, moving averages, volatility)

### Model Architecture Improvements
- **Advanced LSTM**: 3-layer LSTM with 128→64→32 units and regularization
- **Regularization**: L2 regularization, batch normalization, and dropout layers
- **Optimized Training**: Early stopping, learning rate scheduling, and validation splits

### Visualization & Documentation
- **Professional Figures**: High-resolution (300 DPI) publication-ready visualizations
- **Comprehensive Metrics**: RMSE, MAE, R², with uncertainty bounds
- **Results Tables**: Publication-ready tables with statistical significance
- **Model Architecture Diagram**: Clear visualization of neural network structure

## 📊 Dataset

The analysis uses two main datasets:
- **Stock Price Data** (`stock_yfinance_data.csv`): Daily OHLCV data from Yahoo Finance
- **Twitter Sentiment Data** (`stock_tweets.csv`): Tweets with sentiment analysis

### Supported Stocks
- **AAPL** (Apple Inc.)
- **TSLA** (Tesla Inc.)  
- **MSFT** (Microsoft Corporation)

## 🚀 Quick Start

### Installation
```bash
pip install numpy pandas matplotlib seaborn textblob tensorflow scikit-learn scipy
```

### Running the Analysis
```python
python main_analysis.py
```

This will automatically:
1. Load and preprocess data for all stocks
2. Train LSTM models with and without sentiment analysis
3. Perform cross-validation and statistical testing
4. Generate comprehensive visualizations
5. Create results tables in CSV format

## 🧠 Model Architecture

### Advanced LSTM with Sentiment Integration

```
Input: [60 days × Features]
├── LSTM Layer 1: 128 units (return_sequences=True)
│   ├── L2 Regularization (0.001)
│   ├── Batch Normalization  
│   └── Dropout (30%)
├── LSTM Layer 2: 64 units (return_sequences=True)
│   ├── L2 Regularization (0.001)
│   ├── Batch Normalization
│   └── Dropout (30%)  
├── LSTM Layer 3: 32 units (return_sequences=False)
│   ├── L2 Regularization (0.001)
│   ├── Batch Normalization
│   └── Dropout (20%)
├── Dense Layer: 16 units (ReLU)
│   ├── L2 Regularization (0.001)
│   └── Dropout (10%)
└── Output Layer: 1 unit (Linear)

Optimizer: Adam (lr=0.001)
Loss: Mean Squared Error
```

### Features Used

#### Price-Based Features
- Open, High, Low, Volume
- Price returns and log returns
- Price ranges and volume ratios

#### Technical Indicators  
- Moving averages (5, 10, 20, 50 days)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Volatility measures (5-day, 20-day)

#### Sentiment Features
- Daily average sentiment polarity
- Sentiment standard deviation
- Tweet count per day
- Min/max daily sentiment

## 📈 Results Summary

### Model Performance (5-Fold Cross-Validation)

| Stock | Model Type | RMSE (Mean ± Std) | MAE (Mean ± Std) | R² (Mean ± Std) |
|-------|------------|-------------------|------------------|------------------|
| AAPL  | LSTM + Sentiment | X.XXX ± X.XXX | X.XXX ± X.XXX | X.XXX ± X.XXX |
| AAPL  | LSTM Baseline | X.XXX ± X.XXX | X.XXX ± X.XXX | X.XXX ± X.XXX |
| TSLA  | LSTM + Sentiment | X.XXX ± X.XXX | X.XXX ± X.XXX | X.XXX ± X.XXX |
| TSLA  | LSTM Baseline | X.XXX ± X.XXX | X.XXX ± X.XXX | X.XXX ± X.XXX |
| MSFT  | LSTM + Sentiment | X.XXX ± X.XXX | X.XXX ± X.XXX | X.XXX ± X.XXX |
| MSFT  | LSTM Baseline | X.XXX ± X.XXX | X.XXX ± X.XXX | X.XXX ± X.XXX |

*Note: Values will be populated after running the analysis*

### Statistical Significance
- Paired t-tests compare models within each stock
- Wilcoxon signed-rank tests provide non-parametric validation
- P-values < 0.05 indicate statistically significant improvements

## 📁 Output Files

Running the analysis generates:
- `comprehensive_results_table.csv`: Detailed results table
- `comprehensive_stock_analysis.png`: Multi-panel visualization
- `{STOCK}_prediction_detailed.png`: Individual stock predictions
- Console output with statistical summaries

## 🔧 Customization

### Modifying Parameters
```python
# Initialize with custom parameters
predictor = ImprovedStockPredictor(
    sequence_length=30,  # Days of historical data
    prediction_horizon=1  # Days ahead to predict
)

# Run analysis on different stocks
stocks = ['AAPL', 'GOOGL', 'AMZN']
results = predictor.run_complete_analysis(stocks)
```

### Adding New Features
Extend the `_create_technical_features()` method:
```python
def _create_technical_features(self, df):
    # Add your custom features here
    df['custom_indicator'] = your_calculation(df)
    return df
```

## 📋 Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn, SciPy
- TextBlob for sentiment analysis

## 🏗️ Project Structure

```
├── main_analysis.py                      # Main analysis script
├── original_implementation.py            # Original implementation
├── original_hyperparameter_tuning.py     # Original hyperparameter tuning
├── sample_results_demo.py                # Demo results display
├── stock_yfinance_data.csv               # Stock price data
├── stock_tweets.csv                      # Twitter sentiment data
├── JEI_FEEDBACK_ADDRESSED.md             # Response to journal feedback
├── requirements.txt                      # Dependencies
└── [Generated outputs]
    ├── comprehensive_results_table.csv
    ├── comprehensive_stock_analysis.png
    └── {STOCK}_prediction_detailed.png
```

## 📊 Methodology

### Cross-Validation Strategy
- **Time Series Split**: 5-fold validation respecting temporal order
- **Train/Validation/Test**: 60%/20%/20% split per fold
- **Rolling Window**: Maintains chronological integrity

### Statistical Testing
- **Paired T-Test**: Tests mean difference in RMSE scores
- **Wilcoxon Test**: Non-parametric alternative for robustness
- **Effect Size**: Calculates practical significance of improvements

### Evaluation Metrics
- **RMSE**: Root Mean Square Error (primary metric)
- **MAE**: Mean Absolute Error (interpretability)
- **R²**: Coefficient of determination (explained variance)

## 🎯 Addressing JEI Feedback

### Required Changes Implemented
✅ **Generalizability**: Analysis extended to multiple stocks (AAPL, TSLA, MSFT)  
✅ **Cross-Validation**: 5-fold time series CV with uncertainty quantification  
✅ **Statistical Testing**: Paired t-tests and Wilcoxon tests implemented  
✅ **Professional Visualization**: High-quality figures with proper formatting  
✅ **Model Architecture**: Clear documentation and diagram included  
✅ **Comprehensive Metrics**: RMSE, MAE, R² with confidence intervals  

### Recommended Changes Implemented  
✅ **Enhanced Features**: Technical indicators and sentiment metrics  
✅ **Model Improvements**: Multi-layer LSTM with regularization  
✅ **Publication Quality**: Professional tables and figures  
✅ **Code Accessibility**: Well-documented, modular code structure  

## 📚 References

This implementation builds upon established methods in:
- LSTM neural networks for time series prediction
- Sentiment analysis for financial forecasting  
- Technical analysis indicators
- Cross-validation for time series data

## 🤝 Contributing

For questions, suggestions, or contributions, please contact the authors or create an issue in the repository.

## 📄 License

This project is created for academic purposes as part of research submitted to the Journal of Emerging Investigators.

---

*Generated using the Improved Stock Price Prediction System*