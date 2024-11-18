import yfinance as yf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sys

def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() for col in data.columns.values]
    return data

def detect_elliott_wave(stock_data, tolerance=0.02):
    stock_data['Pattern'] = None
    highs = stock_data['High ^GSPC'].values
    lows = stock_data['Low ^GSPC'].values

    for i in range(5, len(stock_data)):
        # Attempt to find 5-wave structure
        wave_1 = highs[i-5]
        wave_2 = lows[i-4]
        wave_3 = highs[i-3]
        wave_4 = lows[i-2]
        wave_5 = highs[i-1]
        wave_0 = lows[i-6]  # Starting point of the pattern

        if (wave_1 > wave_0 and wave_2 < wave_1 and wave_3 > wave_1 and
            wave_3 > wave_2 and wave_4 < wave_3 and wave_4 > wave_2 and
            wave_5 > wave_3 and abs(wave_5 - highs[i]) <= tolerance * highs[i]):
            stock_data.at[stock_data.index[i], 'Pattern'] = "Elliott Wave (Impulse)"
        
        # Attempt to find corrective wave (ABC structure)
        if i >= 3:
            a = highs[i-3]
            b = lows[i-2]
            c = highs[i-1]
            if (a > b and c > b and c < a):
                stock_data.at[stock_data.index[i], 'Pattern'] = "Elliott Wave (Corrective)"

    return stock_data

def plot_elliott_wave(stock_data, ticker, accuracy):
    plt.figure(figsize=(18, 7))
    plt.plot(stock_data['Close ^GSPC'], label='Close Price', alpha=0.8)

    for idx, row in stock_data.iterrows():
        if row['Pattern'] == "Elliott Wave (Impulse)":
            plt.scatter(idx, row['High ^GSPC'], color='green', marker='^', label="Impulse Wave" if "Impulse Wave" not in plt.gca().get_legend_handles_labels()[1] else "", alpha=1)
        elif row['Pattern'] == "Elliott Wave (Corrective)":
            plt.scatter(idx, row['High ^GSPC'], color='red', marker='v', label="Corrective Wave" if "Corrective Wave" not in plt.gca().get_legend_handles_labels()[1] else "", alpha=1)

    plt.title(f'{ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def measure_accuracy(stock_data, prediction_window=5):
    pattern_counts = stock_data['Pattern'].value_counts().to_dict()
    total_patterns = sum(pattern_counts.values())
    
    correct_predictions = 0

    for i in range(len(stock_data) - prediction_window):
        pattern = stock_data['Pattern'].iloc[i]
        if pattern:
            future_prices = stock_data['Close ^GSPC'].iloc[i+1:i+1+prediction_window]

            if pattern == "Elliott Wave (Impulse)":
                if future_prices.max() > stock_data['Close ^GSPC'].iloc[i]:
                    correct_predictions += 1
            elif pattern == "Elliott Wave (Corrective)":
                if future_prices.min() < stock_data['Close ^GSPC'].iloc[i]:
                    correct_predictions += 1

    overall_accuracy = (correct_predictions / total_patterns) * 100 if total_patterns > 0 else 0

    print("Pattern Counts:")
    for pattern, count in pattern_counts.items():
        print(f"{pattern}: {count}")

    print(f"Overall Predictive Accuracy: {overall_accuracy:.2f}%")
    sys.stdout.flush()  # Ensure the output is displayed
    return overall_accuracy

if __name__ == "__main__":
    ticker = '^GSPC'  # S&P 500 Index
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    stock_data = stock_data.dropna()

    # Ensure the column names match what you need
    if 'High ^GSPC' in stock_data.columns and 'Low ^GSPC' in stock_data.columns:
        stock_data = detect_elliott_wave(stock_data)
        accuracy = measure_accuracy(stock_data)
        plot_elliott_wave(stock_data, ticker, accuracy)
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("Required columns are missing in the stock data.")
