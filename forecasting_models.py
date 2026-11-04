"""
Time-Series Forecasting Models for Energy Consumption
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ARIMAModel:
    """ARIMA-like forecasting model."""
    
    def __init__(self, p=5, d=1, q=2):
        self.p = p
        self.d = d
        self.q = q
        self.scaler = MinMaxScaler()
        self.history = []
        
    def fit(self, data):
        """Fit the model."""
        self.history = list(data)
        return self
    
    def predict(self, steps=24):
        """Make predictions."""
        predictions = []
        history = self.history.copy()
        
        for _ in range(steps):
            if len(history) >= self.p:
                # AR component: use last p values
                ar_pred = np.mean(history[-self.p:])
                predictions.append(ar_pred)
                history.append(ar_pred)
            else:
                predictions.append(np.mean(history))
                history.append(np.mean(history))
        
        return np.array(predictions)

class ExponentialSmoothing:
    """Exponential Smoothing forecasting model."""
    
    def __init__(self, alpha=0.3, beta=0.1, gamma=0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.level = None
        self.trend = None
        
    def fit(self, data):
        """Fit the model."""
        self.level = data[0]
        self.trend = (data[1] - data[0]) if len(data) > 1 else 0
        
        for value in data[1:]:
            last_level = self.level
            self.level = self.alpha * value + (1 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - last_level) + (1 - self.beta) * self.trend
        
        return self
    
    def predict(self, steps=24):
        """Make predictions."""
        predictions = []
        for i in range(1, steps + 1):
            pred = self.level + i * self.trend
            predictions.append(max(pred, 10))
        return np.array(predictions)

class LSTMSimulator:
    """Simplified LSTM-like model for energy forecasting."""
    
    def __init__(self, lookback=24):
        self.lookback = lookback
        self.weights = None
        self.scaler = MinMaxScaler()
        
    def fit(self, data):
        """Fit the model."""
        self.scaler.fit(data.reshape(-1, 1))
        scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        # Simple weight initialization
        self.weights = np.random.randn(self.lookback) * 0.1
        
        # Train weights using gradient descent
        for epoch in range(10):
            for i in range(self.lookback, len(scaled_data) - 1):
                window = scaled_data[i-self.lookback:i]
                pred = np.dot(window, self.weights)
                target = scaled_data[i]
                error = target - pred
                self.weights += 0.01 * error * window
        
        return self
    
    def predict(self, data, steps=24):
        """Make predictions."""
        scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        predictions = []
        current_window = scaled_data[-self.lookback:].copy()
        
        for _ in range(steps):
            pred_scaled = np.dot(current_window, self.weights)
            pred = self.scaler.inverse_transform([[pred_scaled]])[0][0]
            predictions.append(max(pred, 10))
            
            # Update window
            current_window = np.append(current_window[1:], pred_scaled)
        
        return np.array(predictions)

class EnsembleForecaster:
    """Ensemble of multiple forecasting models."""
    
    def __init__(self):
        self.arima = ARIMAModel()
        self.exp_smooth = ExponentialSmoothing()
        self.lstm = LSTMSimulator()
        
    def fit(self, data):
        """Fit all models."""
        self.arima.fit(data)
        self.exp_smooth.fit(data)
        self.lstm.fit(data)
        return self
    
    def predict(self, data, steps=24):
        """Make ensemble predictions."""
        pred_arima = self.arima.predict(steps)
        pred_exp = self.exp_smooth.predict(steps)
        pred_lstm = self.lstm.predict(data, steps)
        
        # Weighted average: 30% ARIMA, 30% Exp Smooth, 40% LSTM
        ensemble_pred = 0.3 * pred_arima + 0.3 * pred_exp + 0.4 * pred_lstm
        return ensemble_pred

def evaluate_forecast(actual, predicted):
    """Evaluate forecast accuracy."""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }

