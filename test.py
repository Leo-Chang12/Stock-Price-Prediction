import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the dataset (Replace "prices.csv" with the actual path to your dataset)
df = pd.read_csv("prices.csv")
apple_df = df[df['symbol'] == 'AAPL']
apple_data_array = apple_df[['date', 'open', 'low', 'high', 'volume']].values
apple_closing_prices_array = apple_df['close'].values

# Preprocess the data
scaler = MinMaxScaler()
apple_data_scaled = scaler.fit_transform(apple_data_array[:, 1:])
apple_closing_prices_scaled = scaler.fit_transform(apple_closing_prices_array.reshape(-1, 1))

# Splitting the data
train_size = int(len(apple_data_scaled) * 0.8)
train_data = apple_data_scaled[:train_size]
test_data = apple_data_scaled[train_size:]
train_labels = apple_closing_prices_scaled[:train_size]
test_labels = apple_closing_prices_scaled[train_size:]

# Reshape data for LSTM (if it's not already in the correct shape)
train_data = train_data.reshape((train_data.shape[0], 1, train_data.shape[1]))
test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))

# Build the LSTM model as a function
def build_lstm(neurons=32):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(train_data.shape[1], train_data.shape[2]), activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Define hyperparameters to iterate through
neurons_list = [16, 32, 64]
epochs_list = [50, 100, 150]

best_neuron = None
best_epochs = None
best_rmse = np.inf

# Iterate through hyperparameters
for neurons in neurons_list:
    for epochs in epochs_list:
        model = build_lstm(neurons)
        model.fit(train_data, train_labels, epochs=epochs, batch_size=16, verbose=0)

        predicted_labels = model.predict(test_data)
        predicted_prices = scaler.inverse_transform(predicted_labels)
        actual_prices = scaler.inverse_transform(test_labels)

        rmse = mean_squared_error(actual_prices, predicted_prices, squared=False)

        print(f"Neurons: {neurons}, Epochs: {epochs}, RMSE: {rmse}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_neuron = neurons
            best_epochs = epochs

# Build the final LSTM model with best hyperparameters
final_lstm_model = build_lstm(best_neuron)
final_lstm_model.fit(train_data, train_labels, epochs=best_epochs, batch_size=16, verbose=1)

# Predict using the LSTM model
lstm_predicted_labels = final_lstm_model.predict(test_data)
lstm_predicted_prices = scaler.inverse_transform(lstm_predicted_labels)
actual_prices = scaler.inverse_transform(test_labels)

# Naive Predictor
naive_predicted_prices = np.roll(actual_prices, 1)
naive_predicted_prices[0] = actual_prices[train_size - 2]  # Set the first prediction as the second-to-last training actual price

# Calculate RMSE for Naive Predictor and LSTM Model
naive_rmse = mean_squared_error(actual_prices, naive_predicted_prices, squared=False)
lstm_rmse = mean_squared_error(actual_prices, lstm_predicted_prices, squared=False)
print("Naive Predictor RMSE:", naive_rmse)
print("LSTM Model RMSE:", lstm_rmse)

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(apple_df['date'][train_size:], actual_prices, label='Actual Prices')
plt.plot(apple_df['date'][train_size:], lstm_predicted_prices, label='LSTM Predicted Prices')
plt.plot(apple_df['date'][train_size:], naive_predicted_prices, label='Naive Predicted Prices')
plt.title("Apple Stock Price Prediction using LSTM")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()