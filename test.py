import pandas
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from textblob import TextBlob

# Load the dataset 
# df = pd.read_csv("prices.csv")
# apple_df = df[df['symbol'] == 'AAPL']
# apple_data_array = apple_df[['date', 'open', 'low', 'high', 'volume']].values
# apple_closing_prices_array = apple_df['close'].values
#
# # Preprocess the data
# scaler = MinMaxScaler()
# apple_data_scaled = scaler.fit_transform(apple_data_array[:, 1:])
# apple_closing_prices_scaled = scaler.fit_transform(apple_closing_prices_array.reshape(-1, 1))
#
# # Splitting the data
# train_size = int(len(apple_data_scaled) * 0.8)
# train_data = apple_data_scaled[:train_size]
# test_data = apple_data_scaled[train_size:]
# train_labels = apple_closing_prices_scaled[:train_size]
# test_labels = apple_closing_prices_scaled[train_size:]
#
# # Reshape data for LSTM (if it's not already in the correct shape)
# train_data = train_data.reshape((train_data.shape[0], 1, train_data.shape[1]))
# test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))
#
# # Build the LSTM model as a function
# def build_lstm(neurons=32):
#     model = Sequential()
#     model.add(LSTM(neurons, input_shape=(train_data.shape[1], train_data.shape[2]), activation='relu'))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model
#
# # Define hyperparameters to iterate through
# neurons_list = [16, 32, 64]
# epochs_list = [50, 100, 150]
#
# best_neuron = None
# best_epochs = None
# best_rmse = np.inf
#
# # Iterate through hyperparameters
# for neurons in neurons_list:
#     for epochs in epochs_list:
#         model = build_lstm(neurons)
#         model.fit(train_data, train_labels, epochs=epochs, batch_size=16, verbose=0)
#
#         predicted_labels = model.predict(test_data)
#         predicted_prices = scaler.inverse_transform(predicted_labels)
#         actual_prices = scaler.inverse_transform(test_labels)
#
#         rmse = mean_squared_error(actual_prices, predicted_prices, squared=False)
#
#         print(f"Neurons: {neurons}, Epochs: {epochs}, RMSE: {rmse}")
#
#         if rmse < best_rmse:
#             best_rmse = rmse
#             best_neuron = neurons
#             best_epochs = epochs
#
# # Build the final LSTM model with best hyperparameters
# final_lstm_model = build_lstm(best_neuron)
# final_lstm_model.fit(train_data, train_labels, epochs=best_epochs, batch_size=16, verbose=1)
#
# # Predict using the LSTM model
# lstm_predicted_labels = final_lstm_model.predict(test_data)
# lstm_predicted_prices = scaler.inverse_transform(lstm_predicted_labels)
# actual_prices = scaler.inverse_transform(test_labels)
#
# # Naive Predictor
# naive_predicted_prices = np.roll(actual_prices, 1)
# train_prices = scaler.inverse_transform(train_labels)
# naive_predicted_prices[0] = train_prices[-1]  # Set the first prediction as the last training actual price
#
# # Calculate RMSE for Naive Predictor and LSTM Model
# naive_rmse = mean_squared_error(actual_prices, naive_predicted_prices, squared=False)
# lstm_rmse = mean_squared_error(actual_prices, lstm_predicted_prices, squared=False)
# print("Naive Predictor RMSE:", naive_rmse)
# print("LSTM Model RMSE:", lstm_rmse)
#
# # Visualize the results
# plt.figure(figsize=(12, 6))
# plt.plot(apple_df['date'][train_size:], actual_prices, label='Actual Prices')
# plt.plot(apple_df['date'][train_size:], lstm_predicted_prices, label='LSTM Predicted Prices')
# plt.plot(apple_df['date'][train_size:], naive_predicted_prices, label='Naive Predicted Prices')
# plt.title("Apple Stock Price Prediction using LSTM")
# plt.xlabel("Date")
# plt.ylabel("Stock Price")
# plt.legend()
# plt.show()

# Load the historical stock prices
# stock_prices = pd.read_csv('prices.csv')

## Reloading the datasets

# Load the stock tweets
tweets = pd.read_csv('stock_tweets.csv')

# Selecting a particular stock for analysis
selected_stock = 'AAPL'
tweets_selected = tweets[tweets['Stock Name'] == selected_stock]

# Load the yfinance data
yfinance_data = pd.read_csv('stock_yfinance_data.csv')

# Preprocessing: Generate sentiment analysis feature from the tweets
tweets_selected['Sentiment'] = tweets_selected['Tweet'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)

# Get rid of the time so we can group by just the day
tweets_selected['Date'] = pd.to_datetime(tweets_selected['Date'])
tweets_selected['Date'] = tweets_selected['Date'].dt.date

# Filter tweets and yfinance_data for the selected stock
yfinance_data_selected = yfinance_data[yfinance_data['Stock Name'] == selected_stock]
# Get rid of the time so we can group by just the day
yfinance_data_selected['Date'] = pd.to_datetime(yfinance_data_selected['Date'])
yfinance_data_selected['Date'] = yfinance_data_selected['Date'].dt.date

# Compute average sentiment for each day
average_daily_sentiment = tweets_selected.groupby('Date')['Sentiment'].mean().reset_index()

# Merge on the Date to combine sentiment data with yfinance data
combined_data = yfinance_data_selected.merge(average_daily_sentiment, on='Date', how='left')

# Filling NA values in sentiment with 0 (in case there are days with no tweets)
combined_data['Sentiment'].fillna(0, inplace=True)

# Using 'Close' as the target variable
features = combined_data.drop(columns=['Close', 'Stock Name', 'Date'])  # Removing target and non-feature columns
target = combined_data['Close']

# Train-test split and Model training
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model training and prediction
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(mean_squared_error(y_test, predictions, squared=False))