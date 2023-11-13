import numpy as np
import pandas as pd
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import itertools

# 1. Data Loading and Preprocessing
tweets = pd.read_csv('stock_tweets.csv')
yfinance_data = pd.read_csv('stock_yfinance_data.csv')

selected_stock = 'AAPL'

# Filter based on selected stock and process data
tweets_selected = tweets[tweets['Stock Name'] == selected_stock].copy()
tweets_selected['Sentiment'] = tweets_selected['Tweet'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
tweets_selected['Date'] = pd.to_datetime(tweets_selected['Date']).dt.date

yfinance_data_selected = yfinance_data[yfinance_data['Stock Name'] == selected_stock].copy()
yfinance_data_selected['Date'] = pd.to_datetime(yfinance_data_selected['Date']).dt.date

# Merge data
average_daily_sentiment = tweets_selected.groupby('Date')['Sentiment'].mean().reset_index()
combined_data = yfinance_data_selected.merge(average_daily_sentiment, on='Date', how='left')
combined_data['Sentiment'].fillna(0, inplace=True)

# Data Splitting and Scaling
features = combined_data.drop(columns=['Close', 'Stock Name', 'Date'])
target = combined_data['Close']

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
target_scaled = scaler.fit_transform(target.values.reshape(-1, 1))

train_size = int(len(features_scaled) * 0.8)
train_data = features_scaled[:train_size].reshape(train_size, 1, features.shape[1])
test_data = features_scaled[train_size:].reshape(len(features_scaled) - train_size, 1, features.shape[1])
train_labels = target_scaled[:train_size]
test_labels = target_scaled[train_size:]

# 2. Define Grid Search Hyperparameters
hyperparameters = {
    'epochs': [20, 50, 100, 150],
    'batch_size': [16, 32, 64],
    'verbose': [0, 1, 2]  # 0 = silent, 1 = progress bar, 2 = one line per epoch
}

# 3. Function to Build and Evaluate Model
def train_and_evaluate_model(epochs, batch_size, verbose):
    # Define the model
    model = Sequential()
    # Assuming you want to keep the same model structure as before
    # First LSTM layer
    model.add(LSTM(50, activation='relu', input_shape=(train_data.shape[1], train_data.shape[2])))
    # You can add more layers or change parameters as needed
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the model
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # Evaluate the model
    predicted_labels = model.predict(test_data)
    predicted_prices = scaler.inverse_transform(predicted_labels)
    rmse = mean_squared_error(scaler.inverse_transform(test_labels), predicted_prices, squared=False)
    return rmse


# 4. Running Grid Search
results = []
for combination in itertools.product(*hyperparameters.values()):
    params = dict(zip(hyperparameters.keys(), combination))
    rmse = train_and_evaluate_model(epochs=params['epochs'], batch_size=params['batch_size'], verbose=params['verbose'])
    results.append({'params': params, 'rmse': rmse})
    print(f'Params: {params}, RMSE: {rmse}')
