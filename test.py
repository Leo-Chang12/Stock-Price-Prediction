import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("prices.csv")
apple_df = df[df['symbol'] == 'AAPL']
data_array = apple_df[['date', 'open', 'low', 'high', 'volume']].values
closing_prices_array = apple_df['close'].values

# Preprocess the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_array[:, 1:])  # Exclude 'date' column from scaling
closing_prices_scaled = scaler.fit_transform(closing_prices_array.reshape(-1, 1))

# Splitting the data
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]
train_labels = closing_prices_scaled[:train_size]
test_labels = closing_prices_scaled[train_size:]

# Reshape data for LSTM
train_data = train_data.reshape((train_data.shape[0], 1, train_data.shape[1]))
test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(32, input_shape=(train_data.shape[1], train_data.shape[2]), activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_data, train_labels, epochs=50, batch_size=16, verbose=1)

# Predict using the model
predicted_labels = model.predict(test_data)

# Inverse transform the scaled values to get actual prices
predicted_prices = scaler.inverse_transform(predicted_labels)
actual_prices = scaler.inverse_transform(test_labels)

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(apple_df['date'][train_size:], actual_prices, label='Actual Prices')
plt.plot(apple_df['date'][train_size:], predicted_prices, label='Predicted Prices')
plt.title("Apple Stock Price Prediction using LSTM")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()