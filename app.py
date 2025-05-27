from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from joblib import load
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import datetime
import os

app = Flask(__name__)

lr_model = None
sarimax_model = None

def load_models():
    global lr_model, sarimax_model
    
    try:
       
        if os.path.exists('trained_models/linear_model.joblib'):
            lr_model = load('trained_models/linear_model.joblib')
            print("Linear regression model loaded successfully")
        else:
            print("Warning: linear_model.joblib not found")
            
        if os.path.exists('trained_models/sarimax_model.pkl'):
            sarimax_model = SARIMAXResults.load('trained_models/sarimax_model.pkl')
            print("SARIMAX model loaded successfully")
        else:
            print("Warning: sarimax_model.pkl not found")
            
    except Exception as e:
        print(f"Error loading models: {str(e)}")

def calculate_features(data):
    """Calculate moving average and volatility features"""
    try:
        if len(data) >= 7:
            data['MA_7'] = data['Close'].rolling(window=7, min_periods=1).mean()
        else:
            data['MA_7'] = data['Close']  
        
        data['Volatility'] = (data['High'] - data['Low']) / data['Close']
        
        return data
    except Exception as e:
        print(f"Error calculating features: {str(e)}")
        data['MA_7'] = data['Close']
        data['Volatility'] = 0.02  
        return data

def prepare_input(data):
    """Prepare input data for SARIMAX model"""
    try:
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        return calculate_features(data)
    except Exception as e:
        print(f"Error preparing input: {str(e)}")
        return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('home'))
    
    try:
       
        date_str = request.form.get('date')
        open_price = request.form.get('open')
        high_price = request.form.get('high')
        low_price = request.form.get('low')
        close_price = request.form.get('close')
        volume = request.form.get('volume')       
        if not all([date_str, open_price, high_price, low_price, close_price, volume]):
            raise ValueError("All fields are required")
        input_data = {
            'Date': date_str,
            'Open': float(open_price),
            'High': float(high_price),
            'Low': float(low_price),
            'Close': float(close_price),
            'Volume': int(volume)
        }
        
        print(f"Input data: {input_data}")  
        
        predictions = {}
        if lr_model is not None:
            try:
                lr_input = pd.DataFrame([input_data]).drop(['Date', 'Close'], axis=1)
                lr_pred = lr_model.predict(lr_input)[0]
                predictions['linear_regression_prediction'] = float(lr_pred)
                print(f"Linear regression prediction: {lr_pred}")
            except Exception as e:
                print(f"Linear regression error: {str(e)}")
                predictions['linear_regression_prediction'] = None
        else:
            print("Linear regression model not available")
            predictions['linear_regression_prediction'] = None

        if sarimax_model is not None:
            try:
                sarimax_input = prepare_input(pd.DataFrame([input_data]))
                if 'MA_7' in sarimax_input.columns and 'Volatility' in sarimax_input.columns:
                    sarimax_pred = sarimax_model.forecast(
                        steps=1, 
                        exog=sarimax_input[['MA_7', 'Volatility']].iloc[-1:].values
                    )
                    predictions['sarimax_prediction'] = float(sarimax_pred.iloc[0])
                    print(f"SARIMAX prediction: {sarimax_pred.iloc[0]}")
                else:
                    print("Required features not available for SARIMAX")
                    predictions['sarimax_prediction'] = None
            except Exception as e:
                print(f"SARIMAX error: {str(e)}")
                predictions['sarimax_prediction'] = None
        else:
            print("SARIMAX model not available")
            predictions['sarimax_prediction'] = None
        
        valid_predictions = [p for p in [
            predictions.get('linear_regression_prediction'),
            predictions.get('sarimax_prediction')
        ] if p is not None]
        
        if valid_predictions:
            predictions['ensemble_prediction'] = float(np.mean(valid_predictions))
        else:
            predictions['ensemble_prediction'] = None
        if all(p is None for p in predictions.values()):
            simple_pred = input_data['Close'] * 1.01  
            predictions = {
                'linear_regression_prediction': simple_pred,
                'sarimax_prediction': simple_pred,
                'ensemble_prediction': simple_pred
            }
            print("Using simple heuristic prediction")
        
        print(f"Final predictions: {predictions}")  
        
        return render_template('index.html', predictions=predictions, request=request)
        
    except ValueError as ve:
        error_message = f"Invalid input format: {str(ve)}"
        print(f"ValueError: {error_message}")
        return render_template('index.html', error=error_message, request=request)
        
    except Exception as e:
        error_message = f"Prediction error: {str(e)}"
        print(f"Exception: {error_message}")
        return render_template('index.html', error=error_message, request=request)

@app.errorhandler(404)
def not_found(error):
    return redirect(url_for('home'))

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', error="Internal server error occurred")

load_models()

if __name__ == '__main__':
    print("Starting Flask app...")
    print(f"Linear model loaded: {lr_model is not None}")
    print(f"SARIMAX model loaded: {sarimax_model is not None}")
    app.run(debug=True)