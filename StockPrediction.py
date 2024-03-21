import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import yfinance as yf
import math
import seaborn as sns
import datetime as dt
from datetime import datetime    
sns.set_style("whitegrid")
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use("ggplot")
from sklearn.model_selection import train_test_split

# Read the CSV file
#data = pd.read_csv('appl.txt')

from datetime import datetime, timedelta

def request_stock_price_list(symbol, start_date, end_date):
    # Define the ticker symbol
    ticker = yf.Ticker(symbol)

#Download historical data
    data123 = ticker.history(period="30mo", start=start_date, end=end_date)

#Rename columns to match your original code
    #data123.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)

#Reset index and add a 'date' column
    data123.reset_index(inplace=True)

    return data123

#Example usage:
symbol = "AAPL"  # Replace with the stock symbol you want
start_date = datetime.today() - timedelta(days=(30*30) + 1)  # Replace with your desired start date
end_date = datetime.today() - timedelta(days=1)   # Replace with your desired end date

data = request_stock_price_list(symbol, start_date, end_date)

# Check for missing values
data.isnull().sum()

# Reset the index
data.reset_index(drop=True, inplace=True)

# Fill missing values with column-wise mean
numeric_cols = data.select_dtypes(include='number')
data[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mean())

# Calculate moving averages
ma_day = [10, 50, 100]
for ma in ma_day:
    column_name = "MA for %s days" % (str(ma))
    data[column_name] = pd.DataFrame.rolling(data['Close'], ma).mean()

# Calculate daily return percentage
data['Daily Return'] = data['Close'].pct_change()

# Display the first few rows of the DataFrame
data.head()

# Print unique values in the DataFrame
data.nunique()

# Print the DataFrame
df = data
print(df)

# Check for missing values again
data.isnull().sum()

X = []
Y = []
window_size=100
for i in range(1 , len(df) - window_size -1 , 1):
    first = df.iloc[i,2]
    temp = []
    temp2 = []
    for j in range(window_size):
        temp.append((df.iloc[i + j, 2] - first) / first)
    temp2.append((df.iloc[i + window_size, 2] - first) / first)
    X.append(np.array(temp).reshape(100, 1))
    Y.append(np.array(temp2).reshape(1, 1))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

train_X = np.array(x_train)
test_X = np.array(x_test)
train_Y = np.array(y_train)
test_Y = np.array(y_test)

train_X = train_X.reshape(train_X.shape[0],1,100,1)
test_X = test_X.reshape(test_X.shape[0],1,100,1)

print(len(train_X))
print(len(test_X))

# For creating model and training
import tensorflow as tf
from tensorflow import keras

from keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from keras.layers import MaxPooling1D, Flatten
from keras.regularizers import L1, L2
from keras.metrics import Accuracy
from keras.metrics import RootMeanSquaredError

model = tf.keras.Sequential()

# Creating the Neural Network model here...
# CNN layers
model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', input_shape=(None, 100, 1))))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(128, kernel_size=3, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(256, kernel_size=3, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(128, kernel_size=3, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Flatten()))
# model.add(Dense(5, kernel_regularizer=L2(0.01)))

# LSTM layers
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(100, return_sequences=False)))
model.add(Dropout(0.5))

#Final layers
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

history = model.fit(train_X, train_Y, validation_data=(test_X,test_Y), epochs=10,batch_size=64, verbose=1, shuffle =True)

from keras.utils import plot_model
print(model.summary())
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

recent_data = data['Close'].iloc[-100:]

# Normalize this data in the same manner as was done for the training data
min_close = recent_data.min()
max_close = recent_data.max()
normalized_recent_data = (recent_data - min_close) / (max_close - min_close)

# Reshape the data to the format expected by the model
normalized_recent_data_reshaped = normalized_recent_data.values.reshape(1, 1, 100, 1)

# Make a prediction for the next day
predicted_normalized_value = model.predict(normalized_recent_data_reshaped)

# Convert the normalized prediction back to a price
predicted_price = (predicted_normalized_value * (max_close - min_close)) + min_close
print(f"Predicted stock price for tomorrow: {predicted_price[0][0]}")


# predicted  = model.predict(test_X)
# test_label = test_Y.reshape(-1,1)
# predicted = np.array(predicted[:,0]).reshape(-1,1)
# len_t = len(train_X)
# for j in range(len_t , len_t + len(test_X)):
#     temp = data.iloc[j,3]
#     test_label[j - len_t] = test_label[j - len_t] * temp + temp
#     predicted[j - len_t] = predicted[j - len_t] * temp + temp
# plt.plot(predicted, color = 'green', label = 'Predicted  Stock Price')
# plt.plot(test_label, color = 'red', label = 'Real Stock Price')
# plt.title(' Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel(' Stock Price')
# plt.legend()
# plt.show()