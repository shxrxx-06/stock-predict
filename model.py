import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import dump
import os

df = pd.read_csv('./GOOG-year.csv', parse_dates=['Date'], index_col='Date')
df = df[['Close']]

def cf(data):
    for i in [1, 2, 3, 5, 7]:
        data[f'Close_lag_{i}'] = data['Close'].shift(i)
    data['MA_7'] = data['Close'].rolling(7).mean()
    data['MA_21'] = data['Close'].rolling(21).mean()
    data['Volatility'] = data['Close'].pct_change().rolling(21).std()
    data['Target'] = data['Close'].shift(-1)
    return data.dropna()
df = cf(df)
train = df.iloc[:-100]
test = df.iloc[-100:]
X_ = train.drop(['Target', 'Close'], axis=1)
y_ = train['Target']
lr = LinearRegression()
lr.fit(X_, y_)
sa = SARIMAX(train['Target'],
                        exog=train[['MA_7', 'Volatility']],
                        order=(1, 0, 1),
                        seasonal_order=(0, 0, 0, 0))
sare = sa.fit(disp=False)
os.makedirs('trained_models', exist_ok=True)
dump(lr, 'trained_models/linear_model.joblib')
sare.save('trained_models/sarimax_model.pkl')
print("Models trained and saved successfully.")
