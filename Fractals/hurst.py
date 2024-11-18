import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    
    # Flatten the MultiIndex columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() for col in data.columns.values]
    
    return data

def hurst_exponent(series):
    lags = range(2, 100)
    tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
    
    # Avoid log(0) or log(near-zero) by adding a small constant to tau
    tau = [t if t > 1e-10 else 1e-10 for t in tau]
    
    try:
        hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2
    except np.RankWarning:
        hurst = np.nan  # In case of any errors with polyfit
    return hurst

def calculate_hurst_exponents(stock_data, window_size=100):
    hurst_exponents = []
    
    for i in range(window_size, len(stock_data)):
        window = stock_data['Close ^GSPC'][i-window_size:i]
        hurst_value = hurst_exponent(window)
        hurst_exponents.append(hurst_value)
    
    # Pad the start with NaN to match the length of the stock data
    hurst_exponents = [np.nan] * window_size + hurst_exponents
    stock_data['Hurst Exponent'] = hurst_exponents
    return stock_data

def detect_hurst_patterns(stock_data, trend_threshold=0.6, mean_revert_threshold=0.4):
    stock_data['Pattern'] = None  # Initialize with None for no pattern
    
    for i in range(len(stock_data)):
        hurst_value = stock_data['Hurst Exponent'].iloc[i]
        
        if np.isnan(hurst_value):  # Skip NaN values
            continue
        
        if hurst_value > trend_threshold:
            stock_data.at[stock_data.index[i], 'Pattern'] = "Trend (Persistent)"
        elif hurst_value < mean_revert_threshold:
            stock_data.at[stock_data.index[i], 'Pattern'] = "Mean-Reversion (Anti-Persistent)"
        else:
            stock_data.at[stock_data.index[i], 'Pattern'] = "Random Walk"
    
    return stock_data

def calculate_accuracy(stock_data, prediction_window=5):
    """Calculates the accuracy of the detected Hurst patterns."""
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(len(stock_data) - prediction_window):
        pattern = stock_data['Pattern'].iloc[i]
        future_prices = stock_data['Close ^GSPC'].iloc[i + 1:i + 1 + prediction_window]
        
        if pattern == "Trend (Persistent)":
            if future_prices.max() > stock_data['Close ^GSPC'].iloc[i]:
                correct_predictions += 1
                print(f"Correct Trend at {stock_data.index[i]}: {future_prices.max()} > {stock_data['Close ^GSPC'].iloc[i]}")
            total_predictions += 1
        
        elif pattern == "Mean-Reversion (Anti-Persistent)":
            if future_prices.min() < stock_data['Close ^GSPC'].iloc[i]:
                correct_predictions += 1
                print(f"Correct Mean-Reversion at {stock_data.index[i]}: {future_prices.min()} < {stock_data['Close ^GSPC'].iloc[i]}")
            total_predictions += 1
        
        elif pattern == "Random Walk":
            # For random walk, we do not expect a specific direction
            total_predictions += 1
    
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    return accuracy

def plot_stock_data(stock_data, ticker):
    plt.figure(figsize=(18, 7))
    plt.plot(stock_data['Close ^GSPC'], label='Close Price')
    
    plt.title(f'{ticker} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ticker = '^GSPC'  # S&P 500 Index
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Drop NaN values
    stock_data = stock_data.dropna()
    
    # Calculate Hurst exponents and detect patterns
    stock_data = calculate_hurst_exponents(stock_data, window_size=100)  # Adjust window size if needed
    stock_data = detect_hurst_patterns(stock_data, trend_threshold=0.6, mean_revert_threshold=0.4)
    
    # Calculate accuracy of detected patterns
    accuracy = calculate_accuracy(stock_data)
    print(f"Hurst Pattern Prediction Accuracy: {accuracy:.2f}%")
    
    # Plot the stock data
    plot_stock_data(stock_data, ticker)
