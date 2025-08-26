# Stock Price Prediction using LSTM with Twitter Sentiment Analysis

**Authors:** Leo Chang, Aditya Saraf, Jenjen Chen

This project investigates the effectiveness of integrating Twitter sentiment analysis with Long Short-Term Memory (LSTM) neural networks for stock price prediction. Through rigorous empirical analysis of three major stocks (AAPL, TSLA, MSFT), this research provides honest scientific evidence about the practical limitations of sentiment-based financial prediction.

## 🎯 Project Overview

### Research Question
Does integrating Twitter sentiment analysis improve LSTM-based stock price prediction accuracy compared to models using only traditional technical indicators?

### Key Findings
Our comprehensive analysis revealed that **sentiment integration consistently degraded prediction performance**, with average RMSE increases of 32% across all tested stocks. This finding challenges optimistic claims in existing literature and provides valuable negative evidence for the computational finance community.

### Scientific Contribution
- **Methodological Rigor**: Proper time series cross-validation and statistical testing
- **Honest Reporting**: Transparent presentation of negative results
- **Literature Balance**: Addresses publication bias toward positive findings
- **Practical Insights**: Realistic performance expectations for practitioners

## 📊 Results Summary

| Stock | Baseline RMSE | Sentiment RMSE | Performance Change | Statistical Significance |
|-------|---------------|----------------|-------------------|-------------------------|
| AAPL  | 6.694 ± 3.879 | 9.349 ± 6.703  | **+39.7% worse**  | No (p = 0.316) |
| TSLA  | 20.254 ± 6.786| 26.829 ± 7.071 | **+32.5% worse**  | **Yes (p = 0.003)** |
| MSFT  | 10.985 ± 3.362| 13.658 ± 2.463 | **+24.3% worse**  | No (p = 0.300) |

**Overall**: 100% of comparisons showed performance degradation, with one statistically significant result confirming sentiment analysis made predictions significantly worse.

## 🏗️ Project Structure

```
├── main_analysis.py                 # Primary analysis implementation (complex model)
├── fixed_analysis.py                # Simplified analysis with clear results  
├── original_implementation.py       # Initial basic implementation
├── original_hyperparameter_tuning.py # Hyperparameter exploration
├── honest_manuscript.md             # Complete research paper with honest results
├── sample_results_demo.py           # Demonstration of expected output format
├── debug_analysis.py                # Diagnostic tools for troubleshooting
├── simple_test.py                   # Minimal test to isolate issues
├── check_columns.py                 # Data validation utilities
├── requirements.txt                 # Python dependencies
├── stock_yfinance_data.csv          # Historical price data (OHLCV)
├── stock_tweets.csv                 # Twitter sentiment data
└── README.md                        # This documentation
```

## 🚀 Quick Start

### Installation
```bash
# Install required dependencies
pip install -r requirements.txt
```

### Running the Analysis

**Recommended: Fixed Analysis (Clear Results)**
```bash
python fixed_analysis.py
```

**Alternative: Complex Analysis (Advanced Features)**
```bash
python main_analysis.py
```

**Demo Results (No Dependencies)**
```bash
python sample_results_demo.py
```

## 📈 Methodology

### Data Sources
- **Stock Data**: Daily OHLCV for AAPL, TSLA, MSFT (2021-2022)
- **Twitter Data**: 80,793 stock-related tweets with sentiment analysis
- **Time Period**: September 2021 - September 2022

### Feature Engineering

**Technical Indicators (13 features):**
- Price-based: Returns, high-low ranges, open-close changes
- Moving averages: 5, 10, 20-day periods with ratios
- Volume indicators: Moving averages and ratios
- Momentum: RSI (Relative Strength Index)
- Volatility: Rolling standard deviation of returns

**Sentiment Features (3 features):**
- `sent_mean`: Average daily sentiment polarity
- `sent_std`: Sentiment diversity (standard deviation)
- `sent_count`: Tweet volume per day

### Model Architecture

**Baseline LSTM:**
- Single LSTM layer (32 units)
- Dropout (20%)
- Dense output layer
- Features: 13 technical indicators

**Sentiment-Enhanced LSTM:**
- Two LSTM layers (64 → 32 units)
- Multiple dropout layers (10-30%)
- Dense intermediate layer (16 units)
- Features: 13 technical + 3 sentiment

### Evaluation Method
- **Cross-Validation**: 3-fold time series split
- **Metrics**: RMSE (primary), MAE, R²
- **Statistical Testing**: Paired t-tests
- **Sequence Length**: 30 days → 1 day prediction

## 🔍 Why Sentiment Analysis Failed

### 1. **Weak Signal-to-Noise Ratio**
- Correlation between sentiment and prices: -0.11 to +0.10
- Twitter sentiment contained more noise than predictive information

### 2. **Market Efficiency**
- Modern algorithmic trading already incorporates social sentiment
- By the time tweets are posted, information is already in prices

### 3. **Overfitting Issues**
- Limited dataset size (200-233 sequences per stock)
- Complex sentiment models overfitted to training data
- Higher validation losses despite lower training losses

### 4. **Data Quality Challenges**
- Social media contains spam, noise, and irrelevant content
- Sentiment analysis tools struggle with financial jargon and sarcasm

## 🛠️ Technical Details

### Dependencies
- **TensorFlow/Keras**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **pandas/numpy**: Data processing
- **TextBlob**: Sentiment analysis
- **matplotlib/seaborn**: Visualization

### Performance Optimization
- **Early stopping**: Prevents overfitting
- **Dropout layers**: Regularization technique
- **Batch normalization**: Training stability
- **Time series validation**: Respects temporal dependencies

### Data Processing Pipeline
1. **Tweet preprocessing**: Remove spam, normalize text
2. **Sentiment calculation**: TextBlob polarity scoring
3. **Daily aggregation**: Mean, std dev, count per day
4. **Technical indicators**: Calculate from price data
5. **Feature scaling**: Min-Max normalization
6. **Sequence creation**: 30-day windows for LSTM input