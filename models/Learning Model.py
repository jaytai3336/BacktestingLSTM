import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

from tensorflow import keras
from sklearn.preprocessing import StandardScaler

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data\learning\MicrosoftStock.csv')
df['date'] = pd.to_datetime(df['date'])
df['date'] = sorted(df['date'])

# Data Visualization
def visualize_data(df):
    # Plot the closing and opening prices
    plt.figure(figsize=(14, 7))
    plt.plot(df['date'], df['close'], label='Close Price', color='blue')
    plt.plot(df['date'], df['open'], label='Open Price', color='red')
    plt.title('Microsoft Stock Prices')
    plt.legend()
    plt.show()

    # Plot the volume traded
    plt.figure(figsize=(14, 7))
    plt.bar(df['date'], df['volume'], color='green', alpha=0.5)
    plt.title('Volume Traded')
    plt.show()

    # Data Preprocessing
    numeric_data = df.select_dtypes(include=["int64", "float64"])

    # Check for correlation btw features
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation')
    plt.show()


# visualize_data(df)

prediction = df.loc[
    (df['date'] >= datetime(2013, 1, 1)) & (df['date'] <= datetime(2018, 1, 1))
]

# Prepare the data for LSTM
stock_close = df.filter(["close"])
dataset = stock_close.values #converting to numpy array
training_data_len = int(np.ceil(len(dataset) * .95 )) 

scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)
training_data = scaled_data[0:training_data_len, :]

X_train = []
y_train = []

# create a sliding window 60 days for the stock
for i in range(60, len(training_data)):
    X_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#???

# Build the LSTM model
model = keras.Sequential()
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(keras.layers.LSTM(64, return_sequences=False))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation='linear'))
model.summary()
model.compile(optimizer='adam', loss='mae', metrics=['root_mean_squared_error'])

training = model.fit(X_train, y_train, epochs=20, batch_size=32)

# Prep test data
test_data = scaled_data[training_data_len - 60:, :]
X_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Get the model's predicted price values
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(df['date'][training_data_len:], predictions, label='Predicted Price', color='orange')
plt.plot(df['date'][training_data_len:], y_test, label='Actual Price', color='blue')
plt.title('Microsoft Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
