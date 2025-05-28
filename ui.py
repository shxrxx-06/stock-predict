import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Price Predictor",page_icon="📈",layout="wide",initial_sidebar_state="expanded")
st.title("📈 Real-Time Stock Price Predictor")
st.sidebar.header("Configuration")
ticker_input = st.sidebar.text_input("Enter Stock Ticker Symbol", value="HUDCO.NS",help="Examples: RELIANCE.NS, TCS.NS, INFY.NS for Indian stocks, AAPL, GOOGL for US stocks")
st.sidebar.subheader("Model Parameters")
window_size = st.sidebar.slider("Window Size (minutes)", 30, 120, 60)
epochs = st.sidebar.slider("Training Epochs", 10, 50, 20)
prediction_interval = st.sidebar.selectbox("Prediction Interval", ["1m", "5m", "15m"], index=0)

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'ticker' not in st.session_state:
    st.session_state.ticker = ""

def fetch_stock_data(ticker, period="5d", interval="1m"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            return None, None
        return data, stock
    except Exception as e:
        return None, None

def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except Exception as e:
        return {}

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_next_price(model, scaler, latest_data, window_size):
    try:
        last_window_scaled = scaler.transform(latest_data[-window_size:])
        input_seq = np.reshape(last_window_scaled, (1, window_size, 1))
        predicted_scaled = model.predict(input_seq, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_scaled)
        return predicted_price[0][0]
    except Exception as e:
        return None

def plot_stock_data(data):
    fig = make_subplots(rows=2, cols=1,subplot_titles=('Stock Price', 'Volume'),vertical_spacing=0.1,row_width=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=data.index,y=data['Close'],mode='lines',name='Close Price',line=dict(color='blue')),row=1, col=1)
    if 'Volume' in data.columns:
        fig.add_trace(go.Bar(x=data.index,y=data['Volume'],name='Volume',marker_color='lightblue'),row=2, col=1)
    fig.update_layout(title=f"Stock Data for {ticker_input}",xaxis_title="Time",height=600,showlegend=True)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig

def main():
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button(" Fetch Stock Data", type="primary"):
            if ticker_input:
                with st.spinner(f"Fetching data for {ticker_input}..."):
                    data, stock = fetch_stock_data(ticker_input, interval=prediction_interval)
                    if data is not None and not data.empty:
                        st.success(f" Successfully fetched {len(data)} data points")
                        st.session_state.stock_data = data
                        st.session_state.stock_object = stock
                        st.session_state.ticker = ticker_input
                        info = get_stock_info(ticker_input)
                        if info:
                            st.write(f"**Company:** {info.get('longName', 'N/A')}")
                            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                        st.write(f"**Current Price:** ₹{data['Close'].iloc[-1]:.2f}")
                        fig = plot_stock_data(data)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(" Failed to fetch data. Please check the ticker symbol.")
            else:
                st.warning(" Please enter a ticker symbol")
    with col2:
        st.subheader(" Model Training")
        if st.button(" Train LSTM Model", type="secondary"):
            if 'stock_data' in st.session_state:
                data = st.session_state.stock_data[['Close']].dropna()
                if len(data) < window_size + 10:
                    st.error(f" Not enough data. Need at least {window_size + 10} data points.")
                    return
                with st.spinner("Training LSTM model..."):
                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(data)
                    X, y = create_sequences(scaled_data, window_size)
                    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                    model = build_lstm_model((X.shape[1], 1))
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.model_trained = True
                    progress_bar.progress(1.0)
                    status_text.text(" Model training completed!")
                    st.success(" Model trained successfully!")
            else:
                st.warning(" Please fetch stock data first")
    if st.session_state.model_trained:
        st.subheader(" Real-Time Predictions")
        col3, col4, col5 = st.columns(3)
        with col3:
            if st.button(" Get Single Prediction"):
                with st.spinner("Making prediction..."):
                    latest_data = yf.Ticker(st.session_state.ticker).history(period="5d", interval=prediction_interval)
                    if latest_data is not None and not latest_data.empty:
                        latest_close = latest_data[['Close']].dropna()
                        if len(latest_close) >= window_size:
                            predicted_price = predict_next_price(st.session_state.model,st.session_state.scaler,latest_close,window_size)
                            current_price = latest_close['Close'].iloc[-1]
                            st.metric(label="Current Price",value=f"₹{current_price:.2f}")
                            if predicted_price:
                                change = predicted_price - current_price
                                change_percent = (change / current_price) * 100
                                st.metric(label="Predicted Next Price",value=f"₹{predicted_price:.2f}",delta=f"{change:.2f} ({change_percent:+.2f}%)")
                            else:
                                st.error("❌ Prediction failed")
                        else:
                            st.error(f"❌ Not enough recent data for prediction")
        with col4:
            auto_predict = st.checkbox(" Auto Prediction")
        with col5:
            refresh_interval = st.selectbox("Refresh Rate", [30, 60, 120, 300], index=1)
        if auto_predict:
            placeholder = st.empty()
            while auto_predict:
                with placeholder.container():
                    latest_data = yf.Ticker(st.session_state.ticker).history(period="5d", interval=prediction_interval)
                    if latest_data is not None and not latest_data.empty:
                        latest_close = latest_data[['Close']].dropna()
                        if len(latest_close) >= window_size:
                            predicted_price = predict_next_price(st.session_state.model,st.session_state.scaler,latest_close,window_size)
                            current_price = latest_close['Close'].iloc[-1]
                            timestamp = pd.Timestamp.now().strftime('%H:%M:%S')
                            metric_col1, metric_col2 = st.columns(2)
                            with metric_col1:
                                st.metric(label=f"Current Price ({timestamp})",value=f"₹{current_price:.2f}")
                            with metric_col2:
                                if predicted_price:
                                    change = predicted_price - current_price
                                    change_percent = (change / current_price) * 100
                                    st.metric(label="Predicted Next Price",value=f"₹{predicted_price:.2f}",delta=f"{change:.2f} ({change_percent:+.2f}%)")
                time.sleep(refresh_interval)

if __name__ == "__main__":
    main()

st.markdown("---")