import yfinance as yf
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras

if os.path.exists("sp500.csv"):
    sp500 = pd.read_csv("sp500.csv", index_col=0)
else:
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    sp500.to_csv("sp500.csv")

sp500.index = pd.to_datetime(sp500.index, utc=True)

del sp500["Dividends"]
del sp500["Stock Splits"]
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()

sp500

# Assuming data is available from 1920 to the current year
starting_year = 1920
current_year = 2023  # You can update this to the current year
sequence_length = 100

# Calculate the total number of days
total_days = (current_year - starting_year) * 365

# Calculate the total number of sequences
total_sequences = total_days/sequence_length

print(f"Total number of sequences with sequence_length {sequence_length}: {total_sequences}")
