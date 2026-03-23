"""
Print the test set sizes for each TimeSeriesSplit fold per stock.
Uses the same preprocessing as create_feature_importance_figure.py
and the same CV setup as main_analysis.py.
"""

import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

SEQUENCE_LENGTH = 60

tweets = pd.read_csv('stock_tweets.csv')
yfinance_data = pd.read_csv('stock_yfinance_data.csv')

for stock in ['AAPL', 'TSLA', 'MSFT']:
    # Filter
    tweets_sel = tweets[tweets['Stock Name'] == stock].copy()
    yf_sel = yfinance_data[yfinance_data['Stock Name'] == stock].copy()

    # Sentiment + Subjectivity (matching create_feature_importance_figure.py)
    tweets_sel['Sentiment'] = tweets_sel['Tweet'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0.0
    )
    tweets_sel['Subjectivity'] = tweets_sel['Tweet'].apply(
        lambda x: TextBlob(str(x)).sentiment.subjectivity if pd.notna(x) else 0.0
    )
    tweets_sel['Date'] = pd.to_datetime(tweets_sel['Date']).dt.date

    # Aggregate daily sentiment (matching create_feature_importance_figure.py column names)
    daily_sentiment = tweets_sel.groupby('Date').agg({
        'Sentiment': ['mean', 'std', 'count', 'min', 'max'],
        'Subjectivity': ['mean', 'std']
    }).round(6)
    daily_sentiment.columns = ['sent_polarity_mean', 'sent_polarity_std', 'sent_count',
                                'sent_polarity_min', 'sent_polarity_max',
                                'sent_subjectivity_mean', 'sent_subjectivity_std']
    daily_sentiment = daily_sentiment.reset_index()
    daily_sentiment['sent_polarity_std'] = daily_sentiment['sent_polarity_std'].fillna(0)
    daily_sentiment['sent_subjectivity_std'] = daily_sentiment['sent_subjectivity_std'].fillna(0)

    # Price data + technical features
    yf_sel['Date'] = pd.to_datetime(yf_sel['Date']).dt.date
    yf_sel = yf_sel.sort_values('Date')

    df = yf_sel.copy()
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

    # Merge + fill + drop
    combined = df.merge(daily_sentiment, on='Date', how='left')
    sentiment_cols = ['sent_polarity_mean', 'sent_polarity_std', 'sent_count',
                      'sent_polarity_min', 'sent_polarity_max',
                      'sent_subjectivity_mean', 'sent_subjectivity_std']
    for col in sentiment_cols:
        combined[col] = combined[col].fillna(0)
    combined = combined.dropna()

    # Prepare sequences (matching create_feature_importance_figure.py)
    feature_cols = [col for col in combined.columns
                    if col not in ['Date', 'Stock Name', 'Close']]

    feature_data = combined[feature_cols].values
    target_data = combined['Close'].values.reshape(-1, 1)

    scaler_features = RobustScaler()
    scaler_target = RobustScaler()
    feature_scaled = scaler_features.fit_transform(feature_data)
    target_scaled = scaler_target.fit_transform(target_data)

    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(feature_scaled)):
        X.append(feature_scaled[i - SEQUENCE_LENGTH:i])
        y.append(target_scaled[i])
    X = np.array(X)
    y = np.array(y)

    # TimeSeriesSplit (matching main_analysis.py)
    n_splits = 5
    test_size = len(X) // 10
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    print(f"\n{stock}: {len(combined)} rows, {len(X)} sequences, test_size={test_size}")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"  Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")
