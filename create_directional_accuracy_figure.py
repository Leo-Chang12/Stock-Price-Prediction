"""
Directional Accuracy Analysis for Stock Price Prediction Models
Creates Figure 7 for the Stock Price Prediction manuscript

This script calculates directional accuracy (the percentage of times the model
correctly predicted whether the stock price would increase or decrease) and
compares baseline LSTM models to sentiment-enhanced LSTM models.

Data loading, feature engineering, and model architectures are identical
to main_analysis.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Set style to match other figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Font sizes matching other figures
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)


def load_and_preprocess_data(stock_symbol):
    """
    Load and preprocess stock data with sentiment analysis.
    Identical to main_analysis.py ImprovedStockPredictor.load_and_preprocess_data().
    """
    # Load datasets
    tweets = pd.read_csv('stock_tweets.csv')
    yfinance_data = pd.read_csv('stock_yfinance_data.csv')

    # Filter data for selected stock
    tweets_selected = tweets[tweets['Stock Name'] == stock_symbol].copy()
    yfinance_selected = yfinance_data[yfinance_data['Stock Name'] == stock_symbol].copy()

    # Sentiment analysis — identical to main_analysis.py _calculate_sentiment
    def calculate_sentiment(text):
        try:
            if pd.isna(text) or text == '':
                return 0.0
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0.0

    tweets_selected['Sentiment'] = tweets_selected['Tweet'].apply(calculate_sentiment)
    tweets_selected['Date'] = pd.to_datetime(tweets_selected['Date']).dt.date

    # Aggregate sentiment metrics — identical column names to main_analysis.py
    daily_sentiment = tweets_selected.groupby('Date').agg({
        'Sentiment': ['mean', 'std', 'count', 'min', 'max']
    }).round(6)
    daily_sentiment.columns = ['sent_mean', 'sent_std', 'sent_count', 'sent_min', 'sent_max']
    daily_sentiment = daily_sentiment.reset_index()
    daily_sentiment['sent_std'] = daily_sentiment['sent_std'].fillna(0)

    # Process price data
    yfinance_selected['Date'] = pd.to_datetime(yfinance_selected['Date']).dt.date
    yfinance_selected = yfinance_selected.sort_values('Date')

    # Technical features — identical to main_analysis.py _create_technical_features
    df = yfinance_selected.copy()
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['price_range'] = (df['High'] - df['Low']) / df['Close']
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    for window in [5, 10, 20, 50]:
        df[f'ma_{window}'] = df['Close'].rolling(window).mean()
        df[f'ma_ratio_{window}'] = df['Close'] / df[f'ma_{window}']

    df['volatility_5'] = df['returns'].rolling(5).std()
    df['volatility_20'] = df['returns'].rolling(20).std()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2

    # Merge with sentiment
    combined_data = df.merge(daily_sentiment, on='Date', how='left')

    # Fill missing sentiment data — same 5 columns as main_analysis.py
    sentiment_cols = ['sent_mean', 'sent_std', 'sent_count', 'sent_min', 'sent_max']
    for col in sentiment_cols:
        combined_data[col] = combined_data[col].fillna(0)

    combined_data = combined_data.dropna()

    print(f"Data loaded: {len(combined_data)} days of data for {stock_symbol}")
    return combined_data


def prepare_sequences(data, features, target_col='Close', sequence_length=60):
    """Prepare sequential data for LSTM. Identical to main_analysis.py."""
    scaler_features = RobustScaler()
    scaler_target = RobustScaler()

    feature_data = data[features].values
    target_data = data[target_col].values.reshape(-1, 1)

    feature_scaled = scaler_features.fit_transform(feature_data)
    target_scaled = scaler_target.fit_transform(target_data)

    X, y = [], []
    for i in range(sequence_length, len(feature_scaled)):
        X.append(feature_scaled[i-sequence_length:i])
        y.append(target_scaled[i])

    return np.array(X), np.array(y), scaler_features, scaler_target


def build_advanced_lstm(input_shape):
    """
    Build sentiment-augmented LSTM architecture with regularization.
    Identical to main_analysis.py build_advanced_lstm().
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape,
             kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(64, return_sequences=True,
             kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(32, return_sequences=False,
             kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.1),
        Dense(1, activation='linear')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


def build_baseline_lstm(input_shape):
    """
    Build baseline LSTM for comparison (without sentiment).
    Identical to main_analysis.py build_baseline_lstm():
    single 50-unit LSTM layer, 0.2 dropout, linear output.
    """
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    return model


def calculate_directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy: percentage of times the model correctly
    predicted the direction of price movement (up or down).
    """
    actual_direction = np.sign(np.diff(y_true))
    predicted_direction = np.sign(np.diff(y_pred))
    correct_directions = (actual_direction == predicted_direction).astype(int)
    directional_accuracy = np.mean(correct_directions) * 100
    return directional_accuracy


def train_and_evaluate_directional_accuracy(stock_symbol, n_splits=5):
    """
    Train models with cross-validation and calculate directional accuracy.
    Uses the same CV setup as main_analysis.py.
    """
    print(f"\nAnalyzing {stock_symbol}...")

    # Load data
    data = load_and_preprocess_data(stock_symbol)

    # Define features — identical to main_analysis.py
    all_features = [col for col in data.columns
                   if col not in ['Date', 'Stock Name', 'Close']]

    # Separate sentiment and non-sentiment features
    sentiment_features = [f for f in all_features if 'sent_' in f]
    baseline_features = [f for f in all_features if f not in sentiment_features]

    results = {
        'stock': stock_symbol,
        'advanced': {'directional_accuracies': [], 'rmse': []},
        'baseline': {'directional_accuracies': [], 'rmse': []}
    }

    # Prepare sequences for both models
    X_advanced, y, scaler_feat_adv, scaler_target = prepare_sequences(
        data, all_features, 'Close'
    )
    X_baseline, _, scaler_feat_base, _ = prepare_sequences(
        data, baseline_features, 'Close'
    )

    # Time series cross-validation — identical to main_analysis.py
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=len(X_advanced)//10)

    fold = 1
    for train_idx, test_idx in tscv.split(X_advanced):
        print(f"  Fold {fold}/{n_splits}")

        y_train, y_test = y[train_idx], y[test_idx]

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        ]

        # Advanced model (with sentiment) — 3-layer LSTM
        X_train_adv, X_test_adv = X_advanced[train_idx], X_advanced[test_idx]
        model_adv = build_advanced_lstm((X_train_adv.shape[1], X_train_adv.shape[2]))
        model_adv.fit(X_train_adv, y_train, epochs=100, batch_size=32,
                     validation_split=0.2, callbacks=callbacks, verbose=0)

        y_pred_adv_scaled = model_adv.predict(X_test_adv, verbose=0)
        y_pred_adv = scaler_target.inverse_transform(y_pred_adv_scaled).flatten()
        y_true = scaler_target.inverse_transform(y_test).flatten()

        dir_acc_adv = calculate_directional_accuracy(y_true, y_pred_adv)
        rmse_adv = np.sqrt(mean_squared_error(y_true, y_pred_adv))

        results['advanced']['directional_accuracies'].append(dir_acc_adv)
        results['advanced']['rmse'].append(rmse_adv)

        # Baseline model (without sentiment) — single 50-unit LSTM
        X_train_base, X_test_base = X_baseline[train_idx], X_baseline[test_idx]
        model_base = build_baseline_lstm((X_train_base.shape[1], X_train_base.shape[2]))
        model_base.fit(X_train_base, y_train, epochs=100, batch_size=32,
                      validation_split=0.2, callbacks=callbacks, verbose=0)

        y_pred_base_scaled = model_base.predict(X_test_base, verbose=0)
        y_pred_base = scaler_target.inverse_transform(y_pred_base_scaled).flatten()

        dir_acc_base = calculate_directional_accuracy(y_true, y_pred_base)
        rmse_base = np.sqrt(mean_squared_error(y_true, y_pred_base))

        results['baseline']['directional_accuracies'].append(dir_acc_base)
        results['baseline']['rmse'].append(rmse_base)

        fold += 1

    # Calculate summary statistics
    results['advanced']['mean_dir_acc'] = np.mean(results['advanced']['directional_accuracies'])
    results['advanced']['std_dir_acc'] = np.std(results['advanced']['directional_accuracies'])
    results['baseline']['mean_dir_acc'] = np.mean(results['baseline']['directional_accuracies'])
    results['baseline']['std_dir_acc'] = np.std(results['baseline']['directional_accuracies'])

    return results


def create_directional_accuracy_figure(stocks=['AAPL', 'TSLA', 'MSFT']):
    """Create Figure 7: Directional Accuracy Comparison."""

    all_results = {}

    # Analyze each stock
    for stock in stocks:
        all_results[stock] = train_and_evaluate_directional_accuracy(stock, n_splits=5)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    stocks_list = list(all_results.keys())
    x = np.arange(len(stocks_list))
    width = 0.35

    baseline_accs = [all_results[s]['baseline']['mean_dir_acc'] for s in stocks_list]
    baseline_stds = [all_results[s]['baseline']['std_dir_acc'] for s in stocks_list]
    advanced_accs = [all_results[s]['advanced']['mean_dir_acc'] for s in stocks_list]
    advanced_stds = [all_results[s]['advanced']['std_dir_acc'] for s in stocks_list]

    # Create bars
    bars1 = ax.bar(x - width/2, baseline_accs, width, yerr=baseline_stds,
                   label='LSTM Baseline', capsize=5,
                   color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, advanced_accs, width, yerr=advanced_stds,
                   label='LSTM with Sentiment Analysis', capsize=5,
                   color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add 50% random baseline reference line
    baseline_line = ax.axhline(y=50, color='gray', linestyle='--', linewidth=2.5, alpha=0.7,
                               label='Random Baseline (50%)')

    # Customize plot
    ax.set_xlabel('Stock Symbol', fontweight='bold')
    ax.set_ylabel('Directional Accuracy (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stocks_list)
    # legend entry order matches Figure 1 (sentiment model first)
    ax.legend(handles=[bars2, bars1, baseline_line],
              frameon=True, fancybox=True, shadow=True, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    # 0-based axis so bars and error bars are fully visible, matching the other figures
    ax.set_ylim([0, 85])

    plt.tight_layout()
    plt.savefig('Figure7_Directional_Accuracy_Comparison.png', dpi=300,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print("\n[OK] Saved: Figure7_Directional_Accuracy_Comparison.png")

    # Print summary
    print("\n" + "="*80)
    print("DIRECTIONAL ACCURACY SUMMARY")
    print("="*80)

    for stock in stocks_list:
        print(f"\n{stock}:")
        print(f"  Baseline LSTM (50-unit single layer):")
        print(f"    Mean Directional Accuracy: {all_results[stock]['baseline']['mean_dir_acc']:.2f}%")
        print(f"    Std Dev: {all_results[stock]['baseline']['std_dir_acc']:.2f}%")
        print(f"  LSTM with Sentiment (128/64/32 three-layer):")
        print(f"    Mean Directional Accuracy: {all_results[stock]['advanced']['mean_dir_acc']:.2f}%")
        print(f"    Std Dev: {all_results[stock]['advanced']['std_dir_acc']:.2f}%")
        print(f"  Difference: {all_results[stock]['advanced']['mean_dir_acc'] - all_results[stock]['baseline']['mean_dir_acc']:.2f}%")

    print("\n" + "="*80)

    # Save detailed results
    summary_data = []
    for stock in stocks_list:
        summary_data.append({
            'Stock': stock,
            'Model': 'Baseline',
            'Mean_Directional_Accuracy': f"{all_results[stock]['baseline']['mean_dir_acc']:.2f}%",
            'Std_Directional_Accuracy': f"{all_results[stock]['baseline']['std_dir_acc']:.2f}%"
        })
        summary_data.append({
            'Stock': stock,
            'Model': 'With Sentiment',
            'Mean_Directional_Accuracy': f"{all_results[stock]['advanced']['mean_dir_acc']:.2f}%",
            'Std_Directional_Accuracy': f"{all_results[stock]['advanced']['std_dir_acc']:.2f}%"
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('directional_accuracy_results.csv', index=False)
    print("[OK] Detailed results saved to: directional_accuracy_results.csv")

    return all_results


if __name__ == "__main__":
    print("="*80)
    print("DIRECTIONAL ACCURACY ANALYSIS")
    print("="*80)

    results = create_directional_accuracy_figure(['AAPL', 'TSLA', 'MSFT'])

    print("\nAnalysis complete!")
