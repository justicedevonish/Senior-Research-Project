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

def detect_williams_fractal(stock_data):
    stock_data['Pattern'] = None  # Initialize with None for no pattern

    for i in range(2, len(stock_data) - 2):
        # Williams Bullish Fractal (Bottom)
        if (stock_data['Low ^GSPC'][i] < stock_data['Low ^GSPC'][i-1] and 
            stock_data['Low ^GSPC'][i] < stock_data['Low ^GSPC'][i-2] and
            stock_data['Low ^GSPC'][i] < stock_data['Low ^GSPC'][i+1] and 
            stock_data['Low ^GSPC'][i] < stock_data['Low ^GSPC'][i+2]):
            stock_data.at[stock_data.index[i], 'Pattern'] = "Bullish Fractal"

        # Williams Bearish Fractal (Top)
        elif (stock_data['High ^GSPC'][i] > stock_data['High ^GSPC'][i-1] and 
              stock_data['High ^GSPC'][i] > stock_data['High ^GSPC'][i-2] and
              stock_data['High ^GSPC'][i] > stock_data['High ^GSPC'][i+1] and 
              stock_data['High ^GSPC'][i] > stock_data['High ^GSPC'][i+2]):
            stock_data.at[stock_data.index[i], 'Pattern'] = "Bearish Fractal"

    return stock_data

def calculate_predictive_accuracy(stock_data, prediction_window=5):
    # Initialize counts for total and correct predictions
    pattern_counts = {"Bullish Fractal": 0, "Bearish Fractal": 0}
    correct_predictions = {"Bullish Fractal": 0, "Bearish Fractal": 0}
    
    # Loop through stock data to evaluate predictions
    for i in range(len(stock_data) - prediction_window):
        pattern = stock_data['Pattern'].iloc[i]
        
        if pattern in pattern_counts:
            pattern_counts[pattern] += 1
            future_prices = stock_data['Close ^GSPC'].iloc[i+1:i+1+prediction_window]

            # Define "correct" prediction based on pattern type
            if pattern == "Bullish Fractal" and future_prices.max() > stock_data['Close ^GSPC'].iloc[i]:
                correct_predictions[pattern] += 1
            elif pattern == "Bearish Fractal" and future_prices.min() < stock_data['Close ^GSPC'].iloc[i]:
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

def plot_williams_fractal(stock_data, ticker):
    plt.figure(figsize=(18, 7))
    plt.plot(stock_data['Close ^GSPC'], label='Close Price', alpha=0.8)
    
    for idx, row in stock_data.iterrows():
        if row['Pattern'] == "Bullish Fractal":
            plt.scatter(idx, row['Low ^GSPC'], color='green', marker='^', label="Bullish Fractal" if "Bullish Fractal" not in plt.gca().get_legend_handles_labels()[1] else "", alpha=1)
        elif row['Pattern'] == "Bearish Fractal":
            plt.scatter(idx, row['High ^GSPC'], color='red', marker='v', label="Bearish Fractal" if "Bearish Fractal" not in plt.gca().get_legend_handles_labels()[1] else "", alpha=1)

    plt.title(f'{ticker} Stock Price with Williams Fractals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    ticker = '^GSPC'  # S&P 500 Index ticker symbol
    start_date = '2020-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Drop NaN values
    stock_data = stock_data.dropna()
    
    if 'High ^GSPC' in stock_data.columns and 'Low ^GSPC' in stock_data.columns and 'Close ^GSPC' in stock_data.columns:
        stock_data = detect_williams_fractal(stock_data)
        pattern_accuracy, overall_accuracy, pattern_counts = calculate_predictive_accuracy(stock_data)
        
        print("Williams Fractal Predictive Accuracy:")
        for pattern, accuracy in pattern_accuracy.items():
            print(f"{pattern}: {accuracy:.2f}%")
        print(f"Overall Predictive Accuracy: {overall_accuracy:.2f}%")
        print(f"Pattern Counts: {pattern_counts}")
        
        plot_williams_fractal(stock_data, ticker)
    else:
        print("Required columns are missing in the stock data.")