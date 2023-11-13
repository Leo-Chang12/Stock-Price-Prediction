import numpy as np
import pandas as pd
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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

# 2. Data Splitting and Scaling
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

# 3. LSTM Model Building and Training
def build_lstm(neurons=32, has_sentiment=True):
    model = Sequential()
    num_features = train_data.shape[2]
    if not has_sentiment:
        num_features = train_data.shape[2] - 1
    model.add(LSTM(neurons, input_shape=(train_data.shape[1], num_features), activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

lstm_model = build_lstm(32, True)
lstm_model.fit(train_data, train_labels, epochs=150, batch_size=16, verbose=0)
lstm_predicted_labels = lstm_model.predict(test_data)
lstm_predicted_prices = scaler.inverse_transform(lstm_predicted_labels)
lstm_rmse = mean_squared_error(scaler.inverse_transform(test_labels), lstm_predicted_prices, squared=False)
print(f'lstm rmse: {lstm_rmse}')

# 4. LSTM without Sentiment Data
features_no_sentiment = features.drop(columns=['Sentiment'])
features_no_sentiment_scaled = scaler.fit_transform(features_no_sentiment)
target_scaled = scaler.fit_transform(target.values.reshape(-1, 1))
train_data_no_sentiment = features_no_sentiment_scaled[:train_size].reshape(train_size, 1, features_no_sentiment.shape[1])
test_data_no_sentiment = features_no_sentiment_scaled[train_size:].reshape(len(features_no_sentiment_scaled) - train_size, 1, features_no_sentiment.shape[1])

lstm_model_no_sentiment = build_lstm(32, False)
lstm_model_no_sentiment.fit(train_data_no_sentiment, train_labels, epochs=150, batch_size=16, verbose=0)
lstm_no_sentiment_predicted_labels = lstm_model_no_sentiment.predict(test_data_no_sentiment)
lstm_no_sentiment_predicted_prices = scaler.inverse_transform(lstm_no_sentiment_predicted_labels)
lstm_no_sentiment_rmse = mean_squared_error(scaler.inverse_transform(test_labels), lstm_no_sentiment_predicted_prices, squared=False)
print(f'lstm_no_sentiment rmse: {lstm_no_sentiment_rmse}')

# 5. Naive Prediction
naive_predicted_prices = np.roll(scaler.inverse_transform(test_labels), 1)
naive_predicted_prices[0] = scaler.inverse_transform(test_labels)[-1]
naive_rmse = mean_squared_error(scaler.inverse_transform(test_labels), naive_predicted_prices, squared=False)
print(f'naive rmse: {naive_rmse}')

# 6. Plotting Actual vs Predicted Prices
actual_prices = scaler.inverse_transform(test_labels)

# Plotting the results
plt.figure(figsize=(12, 8))
plt.plot(actual_prices, label='Actual Prices', color='blue')
plt.plot(lstm_predicted_prices, label='LSTM with Sentiment Analysis Predicted Prices', color='green')
plt.plot(lstm_no_sentiment_predicted_prices, label='LSTM without Sentiment Analysis Predicted Prices', color='red')
plt.plot(naive_predicted_prices, label='Naive Predicted Prices', color='orange')

plt.title(f'Comparison of Actual Prices and Predicted Prices for {selected_stock}')
plt.xlabel('Time (Days)')
plt.ylabel('Price')
plt.legend()
plt.show()