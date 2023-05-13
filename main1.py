import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
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

# Plot the price data on the first y-axis in log scale
ax1.plot(merged_data['Adj Close'], color='blue')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price ($)', color='blue')
ax1.set_yscale('log')

# Plot the hash ribbon indicator on the second y-axis
ax2.plot(merged_data['hash_ribbon'], color='green')
ax2.set_ylabel('Hash Ribbon Indicator', color='green')

# Add buy and sell signals to the plot
buy_signals = merged_data[merged_data['hash_ribbon'].diff() == 1]
sell_signals = merged_data[merged_data['hash_ribbon'].diff() == -1]
ax2.scatter(buy_signals.index, buy_signals['hash_ribbon'], marker='^', s=200, c='green', label='Buy Signal')
ax2.scatter(sell_signals.index, sell_signals['hash_ribbon'], marker='v', s=200, c='red', label='Sell Signal')
#Set the title and legend for the plot
ax1.set_title('Bitcoin Price with Hash Ribbon Indicator')
ax2.legend()

plt.show()
