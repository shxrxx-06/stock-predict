import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import time

ticker = "HUDCO.NS" 
stock = yf.Ticker(ticker)

print(f"Fetching for {ticker} 1min")
data = stock.history(period="5d", interval="1m")  

if data.empty:
    raise ValueError("No data fetched. Try again later.")

data = data[['Close']]  
data.dropna(inplace=True)
print("Data shape:", data.shape)


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)


def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

window_size = 60
X, y = create_sequences(scaled_data, window_size)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

print("Training the model...")
model.fit(X, y, epochs=20, batch_size=32)


def predict_next_price(latest_data):
    last_60_scaled = scaler.transform(latest_data[-60:])
    input_seq = np.reshape(last_60_scaled, (1, 60, 1))
    predicted_scaled = model.predict(input_seq)
    predicted_price = scaler.inverse_transform(predicted_scaled)
    return predicted_price[0][0]

print("\n--- Starting Real-Time Prediction Loop ---\n")

try:
    while True:
   
        new_data = stock.history(period="5d", interval="1m")
        if len(new_data) < 60:
            print("Waiting for enough data...")
            time.sleep(60)
            continue

        new_data = new_data[['Close']]
        new_data.dropna(inplace=True)

   
        predicted_price = predict_next_price(new_data)
        current_price = new_data['Close'].iloc[-1]

        print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}]")
        print(f"  Current Price: Rs. {current_price:.2f}")
        print(f"  Predicted Next Minute Price: Rs. {predicted_price:.2f}\n")

        time.sleep(60) 

except KeyboardInterrupt:
    print("Real-time prediction stopped.")
