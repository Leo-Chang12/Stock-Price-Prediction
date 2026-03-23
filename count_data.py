"""
Quick script to count rows and sequences per stock after preprocessing.
Mirrors the exact preprocessing from main_analysis.py without training any models.
"""

import numpy as np
import pandas as pd
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

SEQUENCE_LENGTH = 60

# Load datasets
tweets = pd.read_csv('stock_tweets.csv')
yfinance_data = pd.read_csv('stock_yfinance_data.csv')

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26):
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    return exp1 - exp2

for stock in ['AAPL', 'TSLA', 'MSFT']:
    # Filter tweets and price data
    tweets_sel = tweets[tweets['Stock Name'] == stock].copy()
    yf_sel = yfinance_data[yfinance_data['Stock Name'] == stock].copy()

    # Sentiment scoring
    tweets_sel['Sentiment'] = tweets_sel['Tweet'].apply(
        lambda t: TextBlob(str(t)).sentiment.polarity if pd.notna(t) and t != '' else 0.0
    )
    tweets_sel['Date'] = pd.to_datetime(tweets_sel['Date']).dt.date

    # Aggregate daily sentiment
    daily_sentiment = tweets_sel.groupby('Date').agg({
        'Sentiment': ['mean', 'std', 'count', 'min', 'max']
    }).round(6)
    daily_sentiment.columns = ['sent_mean', 'sent_std', 'sent_count', 'sent_min', 'sent_max']
    daily_sentiment = daily_sentiment.reset_index()
    daily_sentiment['sent_std'] = daily_sentiment['sent_std'].fillna(0)

    # Process price data
    yf_sel['Date'] = pd.to_datetime(yf_sel['Date']).dt.date
    yf_sel = yf_sel.sort_values('Date')

    # Technical feature engineering (exact same as main_analysis.py)
    yf_sel['returns'] = yf_sel['Close'].pct_change()
    yf_sel['log_returns'] = np.log(yf_sel['Close'] / yf_sel['Close'].shift(1))
    yf_sel['price_range'] = (yf_sel['High'] - yf_sel['Low']) / yf_sel['Close']
    yf_sel['volume_ratio'] = yf_sel['Volume'] / yf_sel['Volume'].rolling(20).mean()

    for window in [5, 10, 20, 50]:
        yf_sel[f'ma_{window}'] = yf_sel['Close'].rolling(window).mean()
        yf_sel[f'ma_ratio_{window}'] = yf_sel['Close'] / yf_sel[f'ma_{window}']

    yf_sel['volatility_5'] = yf_sel['returns'].rolling(5).std()
    yf_sel['volatility_20'] = yf_sel['returns'].rolling(20).std()
    yf_sel['rsi'] = calculate_rsi(yf_sel['Close'])
    yf_sel['macd'] = calculate_macd(yf_sel['Close'])

    # Merge with sentiment
    combined = yf_sel.merge(daily_sentiment, on='Date', how='left')

    # Fill missing sentiment
    for col in ['sent_mean', 'sent_std', 'sent_count', 'sent_min', 'sent_max']:
        combined[col] = combined[col].fillna(0)

    # Drop NaN rows
    combined = combined.dropna()

    rows = len(combined)
    sequences = rows - SEQUENCE_LENGTH

    print(f"{stock}: {rows} rows after dropna, {sequences} sequences (sequence_length={SEQUENCE_LENGTH})")
