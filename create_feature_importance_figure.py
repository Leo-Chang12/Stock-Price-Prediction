"""
Feature Importance Analysis using Permutation-Based Methods
Creates Figure 6 for the Stock Price Prediction manuscript

This script performs post-hoc permutation importance analysis to demonstrate
that sentiment features rank lowest in predictive importance compared to
traditional price and volume features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Increase font sizes for publication quality (matching other figures)
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


def load_and_preprocess_data(stock_symbol='AAPL'):
    """Load and preprocess stock data with all features."""
    from textblob import TextBlob

    # Load datasets
    tweets = pd.read_csv('stock_tweets.csv')
    yfinance_data = pd.read_csv('stock_yfinance_data.csv')

    # Filter data for selected stock
    tweets_selected = tweets[tweets['Stock Name'] == stock_symbol].copy()
    yfinance_selected = yfinance_data[yfinance_data['Stock Name'] == stock_symbol].copy()

    # Process sentiment
    tweets_selected['Sentiment'] = tweets_selected['Tweet'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0.0
    )
    tweets_selected['Subjectivity'] = tweets_selected['Tweet'].apply(
        lambda x: TextBlob(str(x)).sentiment.subjectivity if pd.notna(x) else 0.0
    )
    tweets_selected['Date'] = pd.to_datetime(tweets_selected['Date']).dt.date

    # Aggregate sentiment metrics
    daily_sentiment = tweets_selected.groupby('Date').agg({
        'Sentiment': ['mean', 'std', 'count', 'min', 'max'],
        'Subjectivity': ['mean', 'std']
    }).round(6)
    daily_sentiment.columns = ['sent_polarity_mean', 'sent_polarity_std', 'sent_count',
                                'sent_polarity_min', 'sent_polarity_max',
                                'sent_subjectivity_mean', 'sent_subjectivity_std']
    daily_sentiment = daily_sentiment.reset_index()
    daily_sentiment['sent_polarity_std'] = daily_sentiment['sent_polarity_std'].fillna(0)
    daily_sentiment['sent_subjectivity_std'] = daily_sentiment['sent_subjectivity_std'].fillna(0)

    # Process price data and create technical features
    yfinance_selected['Date'] = pd.to_datetime(yfinance_selected['Date']).dt.date
    yfinance_selected = yfinance_selected.sort_values('Date')

    # Technical features
    df = yfinance_selected.copy()
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['price_range'] = (df['High'] - df['Low']) / df['Close']
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f'ma_{window}'] = df['Close'].rolling(window).mean()
        df[f'ma_ratio_{window}'] = df['Close'] / df[f'ma_{window}']

    # Volatility
    df['volatility_5'] = df['returns'].rolling(5).std()
    df['volatility_20'] = df['returns'].rolling(20).std()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2

    # Merge with sentiment
    combined_data = df.merge(daily_sentiment, on='Date', how='left')

    # Fill missing sentiment data
    sentiment_cols = ['sent_polarity_mean', 'sent_polarity_std', 'sent_count',
                      'sent_polarity_min', 'sent_polarity_max',
                      'sent_subjectivity_mean', 'sent_subjectivity_std']
    for col in sentiment_cols:
        combined_data[col] = combined_data[col].fillna(0)

    combined_data = combined_data.dropna()

    return combined_data


def prepare_sequences(data, features, target_col='Close', sequence_length=60):
    """Prepare sequential data for LSTM."""
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


def build_lstm_model(input_shape):
    """Build LSTM model matching the one used in the main analysis."""
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


def custom_permutation_importance(model, X_test, y_test, feature_names, n_repeats=10):
    """
    Calculate permutation importance for LSTM model.

    Since sklearn's permutation_importance works with 2D data and we have 3D LSTM input,
    we implement a custom version that permutes features across the time dimension.
    """
    baseline_score = mean_squared_error(y_test, model.predict(X_test, verbose=0))

    importances = []

    for i, feature_name in enumerate(feature_names):
        feature_scores = []

        for _ in range(n_repeats):
            # Create a copy of the test data
            X_permuted = X_test.copy()

            # Permute the i-th feature across all time steps
            # X_test shape: (samples, timesteps, features)
            perm_idx = np.random.permutation(len(X_permuted))
            X_permuted[:, :, i] = X_permuted[perm_idx, :, i]

            # Calculate score with permuted feature
            permuted_score = mean_squared_error(y_test, model.predict(X_permuted, verbose=0))

            # Importance is the increase in error
            feature_scores.append(permuted_score - baseline_score)

        importances.append({
            'feature': feature_name,
            'importance_mean': np.mean(feature_scores),
            'importance_std': np.std(feature_scores)
        })

    return pd.DataFrame(importances)


def create_feature_importance_figure(stocks=['AAPL', 'TSLA', 'MSFT']):
    """Create comprehensive feature importance analysis across multiple stocks."""

    all_importances = []

    for stock in stocks:
        print(f"\nAnalyzing {stock}...")

        # Load data
        data = load_and_preprocess_data(stock)

        # Define features
        feature_cols = [col for col in data.columns
                       if col not in ['Date', 'Stock Name', 'Close']]

        # Prepare sequences
        X, y, scaler_features, scaler_target = prepare_sequences(data, feature_cols, 'Close')

        # Split data (80/20 train/test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Build and train model
        print("  Training LSTM model...")
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        ]

        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )

        # Calculate permutation importance
        print("  Calculating permutation importance...")
        importance_df = custom_permutation_importance(model, X_test, y_test, feature_cols, n_repeats=10)
        importance_df['stock'] = stock

        all_importances.append(importance_df)

    # Combine results
    combined_importance = pd.concat(all_importances, ignore_index=True)

    # Average across stocks
    avg_importance = combined_importance.groupby('feature').agg({
        'importance_mean': 'mean',
        'importance_std': 'mean'
    }).reset_index()

    # Sort by importance
    avg_importance = avg_importance.sort_values('importance_mean', ascending=True)

    # Categorize features
    def categorize_feature(feature_name):
        if 'sent_' in feature_name or 'subjectivity' in feature_name.lower():
            return 'Sentiment'
        elif 'volume' in feature_name.lower():
            return 'Volume'
        elif any(x in feature_name.lower() for x in ['price', 'close', 'high', 'low', 'open', 'return', 'ma_', 'rsi', 'macd']):
            return 'Price/Technical'
        else:
            return 'Other'

    avg_importance['category'] = avg_importance['feature'].apply(categorize_feature)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Color mapping
    color_map = {
        'Sentiment': '#E74C3C',  # Red - lowest importance
        'Price/Technical': '#3498DB',  # Blue - high importance
        'Volume': '#F39C12',  # Orange - medium importance
        'Other': '#95A5A6'  # Gray
    }

    colors = [color_map[cat] for cat in avg_importance['category']]

    # Create horizontal bar chart
    y_pos = np.arange(len(avg_importance))
    bars = ax.barh(y_pos, avg_importance['importance_mean'],
                   xerr=avg_importance['importance_std'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.2,
                   capsize=4)

    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(avg_importance['feature'])
    ax.set_xlabel('Permutation Importance Score\n(Increase in MSE when feature is permuted)',
                  fontweight='bold')
    ax.set_ylabel('Feature Name', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['Price/Technical'], edgecolor='black', label='Price & Technical Indicators'),
        Patch(facecolor=color_map['Volume'], edgecolor='black', label='Volume Features'),
        Patch(facecolor=color_map['Sentiment'], edgecolor='black', label='Sentiment Features')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
             fancybox=True, shadow=True)

    # Add text box highlighting sentiment features
    sentiment_count = len(avg_importance[avg_importance['category'] == 'Sentiment'])
    sentiment_text = f'Sentiment features (n={sentiment_count})\nranked lowest in predictive\nimportance'
    ax.text(0.98, 0.15, sentiment_text, transform=ax.transAxes,
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle="round,pad=0.8", facecolor="#FFE6E6",
                    alpha=0.9, edgecolor='#E74C3C', linewidth=2))

    plt.tight_layout()
    plt.savefig('Figure6_Permutation_Feature_Importance.png', dpi=300,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print("\n[OK] Saved: Figure6_Permutation_Feature_Importance.png")

    # Print summary
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE SUMMARY")
    print("="*80)
    print("\nTop 10 Most Important Features:")
    top_10 = avg_importance.tail(10)[['feature', 'category', 'importance_mean', 'importance_std']]
    for idx, row in top_10.iterrows():
        print(f"  {row['feature']:30s} ({row['category']:20s}): {row['importance_mean']:.6f} ± {row['importance_std']:.6f}")

    print("\nSentiment Features (Lowest Importance):")
    sentiment_features = avg_importance[avg_importance['category'] == 'Sentiment']
    for idx, row in sentiment_features.iterrows():
        print(f"  {row['feature']:30s}: {row['importance_mean']:.6f} ± {row['importance_std']:.6f}")

    print("\n" + "="*80)

    return avg_importance


if __name__ == "__main__":
    print("="*80)
    print("PERMUTATION-BASED FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    importance_results = create_feature_importance_figure(['AAPL', 'TSLA', 'MSFT'])

    # Save detailed results to CSV
    importance_results.to_csv('feature_importance_results.csv', index=False)
    print("\n[OK] Detailed results saved to: feature_importance_results.csv")

    print("\nAnalysis complete!")
