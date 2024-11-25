import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# ORIGINAL DATA #

def fetch_stock_data(ticker, start, end):
    """Fetches stock data for a given ticker symbol, start date, and end date."""
    data = yf.download(ticker, start=start, end=end)
    
    # Flatten the MultiIndex columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() for col in data.columns.values]
    
    return data

def detect_fractal_patterns(stock_data, ticker, tolerance=0.02):
    """Detects various fractal patterns in the stock data and labels them."""
    stock_data['Pattern'] = None  # Initialize with None for no pattern
    avg_price = stock_data[f'Close {ticker}'].mean()  # Average price for tolerance calculation

    for i in range(2, len(stock_data) - 2):
        # Basic Bullish Fractal (Bottom)
        if (stock_data[f'Low {ticker}'][i] < stock_data[f'Low {ticker}'][i-1] and 
            stock_data[f'Low {ticker}'][i] < stock_data[f'Low {ticker}'][i-2] and
            stock_data[f'Low {ticker}'][i] < stock_data[f'Low {ticker}'][i+1] and 
            stock_data[f'Low {ticker}'][i] < stock_data[f'Low {ticker}'][i+2]):
            stock_data.at[stock_data.index[i], 'Pattern'] = "Bullish Fractal"

        # Basic Bearish Fractal (Top)
        elif (stock_data[f'High {ticker}'][i] > stock_data[f'High {ticker}'][i-1] and 
              stock_data[f'High {ticker}'][i] > stock_data[f'High {ticker}'][i-2] and
              stock_data[f'High {ticker}'][i] > stock_data[f'High {ticker}'][i+1] and 
              stock_data[f'High {ticker}'][i] > stock_data[f'High {ticker}'][i+2]):
            stock_data.at[stock_data.index[i], 'Pattern'] = "Bearish Fractal"
        
        # Double Bottom (Bullish Reversal)
        elif (i > 2 and abs(stock_data[f'Low {ticker}'][i] - stock_data[f'Low {ticker}'][i-2]) <= tolerance * avg_price):
            stock_data.at[stock_data.index[i], 'Pattern'] = "Double Bottom (Bullish)"
        
        # Double Top (Bearish Reversal)
        elif (i > 2 and abs(stock_data[f'High {ticker}'][i] - stock_data[f'High {ticker}'][i-2]) <= tolerance * avg_price):
            stock_data.at[stock_data.index[i], 'Pattern'] = "Double Top (Bearish)"

    return stock_data

def calculate_predictive_accuracy(stock_data, ticker, prediction_window=5):
    """Calculates the predictive accuracy of each detected fractal pattern."""
    
    # Initialize counts for total and correct predictions
    pattern_counts = {
        "Bullish Fractal": 0,
        "Bearish Fractal": 0,
        "Double Bottom (Bullish)": 0,
        "Double Top (Bearish)": 0
    }
    correct_predictions = {
        "Bullish Fractal": 0,
        "Bearish Fractal": 0,
        "Double Bottom (Bullish)": 0,
        "Double Top (Bearish)": 0
    }
    
    # Loop through stock data to evaluate predictions
    for i in range(len(stock_data) - prediction_window):
        pattern = stock_data['Pattern'].iloc[i]
        
        if pattern in pattern_counts:
            pattern_counts[pattern] += 1
            future_prices = stock_data[f'Close {ticker}'].iloc[i+1:i+1+prediction_window]

            # Define "correct" prediction based on pattern type
            if pattern == "Bullish Fractal" and future_prices.max() > stock_data[f'Close {ticker}'].iloc[i]:
                correct_predictions[pattern] += 1
            elif pattern == "Bearish Fractal" and future_prices.min() < stock_data[f'Close {ticker}'].iloc[i]:
                correct_predictions[pattern] += 1
            elif pattern == "Double Bottom (Bullish)" and future_prices.max() > stock_data[f'Close {ticker}'].iloc[i]:
                correct_predictions[pattern] += 1
            elif pattern == "Double Top (Bearish)" and future_prices.min() < stock_data[f'Close {ticker}'].iloc[i]:
                correct_predictions[pattern] += 1
    
    # Calculate predictive accuracy for each pattern
    pattern_accuracy = {
        pattern: (correct_predictions[pattern] / pattern_counts[pattern] * 100) if pattern_counts[pattern] > 0 else 0
        for pattern in pattern_counts
    }
    
    # Calculate overall predictive accuracy
    total_patterns = sum(pattern_counts.values())
    total_correct = sum(correct_predictions.values())
    overall_accuracy = (total_correct / total_patterns) * 100 if total_patterns > 0 else 0

    return pattern_accuracy, overall_accuracy, pattern_counts

def plot_fractal_patterns(stock_data, ticker):
    """Plots the stock price with detected fractal patterns."""
    plt.figure(figsize=(18, 7))
    plt.plot(stock_data[f'Close {ticker}'], label='Close Price')
    
    # Plot detected patterns
    for pattern, color, marker in [("Bullish Fractal", 'g', '^'), 
                                   ("Bearish Fractal", 'r', 'v'), 
                                   ("Double Bottom (Bullish)", 'b', 'o'), 
                                   ("Double Top (Bearish)", 'm', 'x')]:
        pattern_data = stock_data[stock_data['Pattern'] == pattern]
        plt.scatter(pattern_data.index, pattern_data[f'Close {ticker}'], color=color, marker=marker, label=pattern)
    
    plt.title(f'{ticker} Stock Price with Fractal Patterns')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ticker = 'DELL' #Adjust ticker here
    start_date = '2024-08-22'
    end_date = '2024-09-27'
    
    # datetime.today().strftime('%Y-%m-%d')
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Drop NaN values
    stock_data = stock_data.dropna()
    
    # Ensure the modified column names match what you access
    if f'High {ticker}' in stock_data.columns and f'Low {ticker}' in stock_data.columns:
        stock_data = detect_fractal_patterns(stock_data, ticker)
        pattern_accuracy, overall_accuracy, pattern_counts = calculate_predictive_accuracy(stock_data, ticker)  # Calculate predictive accuracy
        print("Fractal Pattern Predictive Accuracy:")
        for pattern, accuracy in pattern_accuracy.items():
            print(f"{pattern}: {accuracy:.2f}%")
        print(f"Overall Predictive Accuracy: {overall_accuracy:.2f}%")
        print(f"Pattern Counts: {pattern_counts}")
        
        plot_fractal_patterns(stock_data, ticker)
    else:
        print("Required columns are missing in the stock data.")
