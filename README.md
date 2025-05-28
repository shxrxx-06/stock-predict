📊 Real-Time Stock Price Predictor using yFinance and Streamlit
This project allows users to view, analyze, and predict stock prices in real-time using various machine learning models. Built using Python, Streamlit, yFinance, and common ML libraries, it provides both static and real-time predictions for a given stock ticker (e.g., GOOG).
🚀 Features
✅ Real-time stock data fetching using yFinance
📈 Historical data visualization with volume charts
🧠 Stock price prediction using trained ML models
🔄 Real-time prediction based on recent market data
🎛️ Streamlit-based interactive UI
💾 CSV-based static data support for offline predictions
🗂️ Project Structure

.
├── templates/           # HTML templates (if Flask is used)
├── trained_models/      # Saved ML models (e.g., LSTM, Linear Regression)
├── GOOG.csv             # Raw historical data
├── GOOG-year.csv        # Aggregated yearly data
├── app.py               # (Optional) Flask/Streamlit entry point
├── model.py             # ML model creation and training
├── predictor.py         # Real-time data fetching & prediction
├── ui.py                # Streamlit user interface

🛠️ Installation
1. Clone the repository:
git clone [https://github.com/shxrxx-06/stock-predict.git](https://github.com/shxrxx-06/stock-predict)
cd stock-predict
2. Install dependencies:
pip install streamlit yfinance pandas numpy matplotlib scikit-learn keras tensorflow plotly
3. Run the Streamlit app:
streamlit run ui.py
📌 Usage

- Open the Streamlit app in your browser.
- Enter a stock ticker (e.g., GOOG, AAPL, MSFT).
- View:
  - Line chart of historical prices
  - Volume trends
  - Predicted future price (single-point or real-time)

📦 Dependencies
streamlit
yfinance
pandas
numpy
scikit-learn
matplotlib
tensorflow or torch
🧠 Models Used

You can train and switch between models like:
- LSTM / RNN (deep learning)
- Linear Regression
- Other time-series based models

Trained models are saved inside trained_models/.

📅 Data Sources

- Yahoo Finance via yfinance
- Custom CSVs (GOOG.csv, GOOG-year.csv) for offline testing and historical training

🧑‍💻 Author
[Ranusha Ramesh]
(https://www.linkedin.com/in/shxrxx-06-ranusha/)
📧 [rameshranusha4@gmail.com]
