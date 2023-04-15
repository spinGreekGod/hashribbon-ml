import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from math import pi
from datetime import datetime as dt
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter
from bokeh.palettes import Spectral4
from bokeh.transform import factor_cmap
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the hash rate data from the CSV file
df = pd.read_csv('BCHAIN-HRATE.csv')
df = df.rename(columns={'Date': 'ds', 'Value': 'y'})
df['ds'] = pd.to_datetime(df['ds'])
df.set_index('ds', inplace=True)

# Download the price data from Yahoo Finance
data = yf.download("BTC-USD", start="2015-01-01", end="2023-04-15")

# Merge the price data with the hash ribbon data
merged_data = pd.merge(data['Adj Close'], df, left_index=True, right_index=True)

# Calculate the 30-day and 60-day moving averages of the hash rate
merged_data['30_day_moving_avg'] = merged_data['y'].rolling(window=30).mean()
merged_data['60_day_moving_avg'] = merged_data['y'].rolling(window=60).mean()

# Calculate the difference between the two moving averages
merged_data['hash_ribbon'] = np.where(merged_data['30_day_moving_avg'] > merged_data['60_day_moving_avg'], 1, 0)

# Define the feature and target variables
X = merged_data[['hash_ribbon']]
y = np.where(merged_data['Adj Close'] > merged_data['Adj Close'].shift(1), 1, 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier on the training data
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rfc.predict(X_test)

# Print the accuracy score and confusion matrix
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:', confusion_matrix(y_test, y_pred))

# Create a figure with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 8))
ax2 = ax1.twinx()

# Plot the price data on the first y-axis
ax1.plot(merged_data['Adj Close'], color='blue')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price ($)', color='blue')

# Plot the hash ribbon indicator on the second y-axis
ax2.plot(merged_data['hash_ribbon'], color='green')
ax2.set_ylabel('Hash Ribbon Indicator', color='green')

# Add buy and sell signals to the plot
buy_signals = merged_data[merged_data['hash_ribbon'].diff() == 1]
sell_signals = merged_data[merged_data['hash_ribbon'].diff() == -1]
ax2.scatter(buy_signals.index, buy_signals['hash_ribbon'], marker='^', s=200, c='green', label='Buy Signal')
ax2.scatter(sell_signals.index, sell_signals['hash_ribbon'], marker='v', s=200, c='red', label='Sell Signal')
ax2.legend()



# Load the hash rate data from the CSV file
df = pd.read_csv('BCHAIN-HRATE.csv')
df = df.rename(columns={'Date': 'ds', 'Value': 'y'})
df['ds'] = pd.to_datetime(df['ds'])
df.set_index('ds', inplace=True)

# Download the price data from Yahoo Finance
data = yf.download("BTC-USD", start="2015-01-01", end="2023-04-15")

# Merge the price data with the hash ribbon data
merged_data = pd.merge(data['Adj Close'], df, left_index=True, right_index=True)

# Calculate the 30-day and 60-day moving averages of the hash rate
merged_data['30_day_moving_avg'] = merged_data['y'].rolling(window=30).mean()
merged_data['60_day_moving_avg'] = merged_data['y'].rolling(window=60).mean()

# Calculate the difference between the two moving averages
merged_data['hash_ribbon'] = np.where(merged_data['30_day_moving_avg'] > merged_data['60_day_moving_avg'], 1, 0)

# Define the feature and target variables
X = merged_data[['hash_ribbon']]
y = np.where(merged_data['Adj Close'] > merged_data['Adj Close'].shift(1), 1, 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier on the training data
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rfc.predict(X_test)

# Print the accuracy score and confusion matrix
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:', confusion_matrix(y_test, y_pred))

# Create a figure with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 8))
ax2 = ax1.twinx()

# Plot the price data on the first y-axis
ax1.plot(merged_data['Adj Close'], color='blue')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price ($)', color='blue')

# Plot the hash ribbon indicator on the second y-axis
ax2.plot(merged_data['hash_ribbon'], color='green')
ax2.set_ylabel('Hash Ribbon Indicator', color='green')

# Add buy and sell signals to the plot
buy_signals = merged_data[merged_data['hash_ribbon'].diff() == 1]
sell_signals = merged_data[merged_data['hash_ribbon'].diff() == -1]
ax2.scatter(buy_signals.index, buy_signals['hash_ribbon'], marker='^', s=200, c='green', label='Buy Signal')
ax2.scatter(sell_signals.index, sell_signals['hash_ribbon'], marker='v', s=200, c='red', label='Sell Signal')
ax2.legend()

plt.show()

# Create a ColumnDataSource for the price data
source_price = ColumnDataSource(data=dict(
    date=df_price.index,
    open=df_price['Open'],
    close=df_price['Close'],
    high=df_price['High'],
    low=df_price['Low']
))

# Create a ColumnDataSource for the hash ribbon data
source_hashribbon = ColumnDataSource(data=dict(
    date=pd.to_datetime(df['date']),
    hashribbon=df['y']
))

# Define the colors for up and down candles
inc = df_price.Close > df_price.Open
dec = df_price.Open > df_price.Close
w = 12*60*60*1000 # half day in ms

# Create the candlestick figure
fig = figure(x_axis_type="datetime", plot_width=1000, title=symbol)
fig.xaxis.major_label_orientation = pi/4
fig.grid.grid_line_alpha=0.3

# Plot the candlesticks
fig.segment('date', 'high', 'date', 'low', color="black", source=source_price)
fig.vbar('date', w, 'open', 'close', fill_color="#D5E1DD", line_color="black", source=source_price, name="candle")
fig.vbar('date', w, 'open', 'close', fill_color="#F2583E", line_color="black", source=source_price, name="candle", selection_color=Spectral4[2], nonselection_color="#D5E1DD")

# Add the hover tool for candlestick data
hover = HoverTool(names=["candle"])
hover.tooltips = [
    ("date", "@date{%F}"),
    ("open", "@open{($ 0.00)}"),
    ("close", "@close{($ 0.00)}"),
    ("high", "@high{($ 0.00)}"),
    ("low", "@low{($ 0.00)}")
]
hover.formatters = {
    "@date": "datetime",
}
fig.add_tools(hover)

# Plot the hash ribbon indicator
fig.line('date', 'hashribbon', line_width=2, color='green', source=source_hashribbon)

# Format the y-axis labels for the hash ribbon indicator
fig.yaxis.formatter = NumeralTickFormatter(format="0")
fig.yaxis.axis_label = "Hash Ribbon Indicator"

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpl_dates

# Pull Bitcoin historical data from January 1st 2015 to present
btc = yf.download('BTC-USD', start='2015-01-01')

# Resample to daily timeframe
btc = btc.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})

# Reset index to obtain a column of date values
btc = btc.reset_index()

# Convert date column to mpl dates
btc['Date'] = btc['Date'].apply(mpl_dates.date2num)

# Plot candlestick chart
fig, ax = plt.subplots()

# Format x-axis ticks as dates
ax.xaxis.set_major_formatter(mpl_dates.DateFormatter('%Y-%m-%d'))

# Plot candlesticks
candlestick_ohlc(ax, btc.values, width=0.6, colorup='green', colordown='red')
