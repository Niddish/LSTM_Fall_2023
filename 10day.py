import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed, MaxPooling1D, Flatten

# Function to fetch stock price data
def request_stock_price_list(symbol, start_date, end_date):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="30mo", start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

# Fetch and process data
symbol = "AAPL"
start_date = datetime.today() - timedelta(days=(30*30) + 1)
end_date = datetime.today() - timedelta(days=1)
data = request_stock_price_list(symbol, start_date, end_date)

# Check for missing values and fill them
data.reset_index(drop=True, inplace=True)
numeric_cols = data.select_dtypes(include='number')
data[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mean())

# Calculate moving averages and daily return percentage
ma_day = [10, 50, 100]
for ma in ma_day:
    column_name = "MA for %s days" % (str(ma))
    data[column_name] = pd.DataFrame.rolling(data['Close'], ma).mean()
data['Daily Return'] = data['Close'].pct_change()

# Split data into training set (excluding last 100 days) and testing set (last 100 days)
train_data = data[:-100]
test_data = data[-100:]

# Prepare training data
X_train = []
Y_train = []
window_size = 100
for i in range(len(train_data) - window_size - 1):
    first = train_data.iloc[i, 2]  # Assuming '2' is the index of 'Close' column
    temp = [(train_data.iloc[i + j, 2] - first) / first for j in range(window_size)]
    temp2 = (train_data.iloc[i + window_size, 2] - first) / first
    X_train.append(np.array(temp).reshape(100, 1))
    Y_train.append(np.array(temp2).reshape(1, 1))

# Convert to NumPy arrays
train_X = np.array(X_train).reshape(len(X_train), 1, 100, 1)
train_Y = np.array(Y_train)

# Build and compile the model
model = tf.keras.Sequential([
    TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', input_shape=(None, 100, 1))),
    TimeDistributed(MaxPooling1D(2)),
    TimeDistributed(Conv1D(128, kernel_size=3, activation='relu')),
    TimeDistributed(MaxPooling1D(2)),
    TimeDistributed(Conv1D(64, kernel_size=3, activation='relu')),
    TimeDistributed(MaxPooling1D(2)),
    TimeDistributed(Flatten()),
    Bidirectional(LSTM(100, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(100, return_sequences=False)),
    Dropout(0.5),
    Dense(1, activation='linear')
])

# Compile and train the model
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
model.fit(train_X, train_Y, epochs=40, batch_size=64, verbose=1, shuffle=True)

# Prediction for the first 10 days of the test dataset
if len(test_data) >= window_size:
    # Normalize the entire dataset first
    min_close = data['Close'].min()
    max_close = data['Close'].max()
    normalized_data = (data['Close'] - min_close) / (max_close - min_close)

    # Predict using the model for the first 10 days
    test_X = []
    start_index = len(normalized_data) - len(test_data) - window_size + 1
    for i in range(start_index, start_index + 10):
        test_X.append(normalized_data[i:i + window_size].values.reshape(1, 100, 1))
    test_X = np.array(test_X).reshape(-1, 1, 100, 1)
    predicted = model.predict(test_X).reshape(-1, 1)
    predicted_prices = predicted * (max_close - min_close) + min_close

    # Print predicted and real prices at the beginning and end of the 10 days
    print(f"Predicted Start Price: {predicted_prices[0][0]:.2f}, Real Start Price: {test_data['Close'].values[0]:.2f}")
    print(f"Predicted End Price: {predicted_prices[-1][0]:.2f}, Real End Price: {test_data['Close'].values[9]:.2f}")

    # Plot the predicted prices against the actual prices for the first 10 days
    plt.figure(figsize=(12, 6))
    plt.plot(predicted_prices, color='green', label='Predicted Stock Price')
    plt.plot(test_data['Close'].values[:10], color='red', label='Real Stock Price')
    plt.title('AAPL Stock Price Prediction: First 10 Days of the Test Dataset')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
else:
    print("Test dataset is too small to create sequences.")
