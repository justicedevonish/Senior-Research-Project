import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    
    # Flatten the MultiIndex columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() for col in data.columns.values]
    
    return data


def detect_self_affine_fractal_patterns(stock_data, scale_factor=0.05, window_size=50):
    stock_data['Pattern'] = None  # Initialize with None for no pattern

    # Loop through the data looking for repeating patterns
    for i in range(window_size, len(stock_data) - window_size):
        # Look at a segment of the stock data and compare it to the next segment of the same size
        segment_close = stock_data['Close ^GSPC'][i - window_size:i]
        next_segment_close = stock_data['Close ^GSPC'][i:i + window_size]
        segment_high = stock_data['High ^GSPC'][i - window_size:i]
        next_segment_high = stock_data['High ^GSPC'][i:i + window_size]

        # Calculate the scaling factor between the two segments (close prices and high prices)
        max_segment_close = max(segment_close)
        min_segment_close = min(segment_close)
        max_next_segment_close = max(next_segment_close)
        min_next_segment_close = min(next_segment_close)

        max_segment_high = max(segment_high)
        min_segment_high = min(segment_high)
        max_next_segment_high = max(next_segment_high)
        min_next_segment_high = min(next_segment_high)

        # Check if the next segment is scaled version of the current segment (using close price and high price)
        if max_segment_close > 0 and max_next_segment_close > 0:
            scaling_factor_current_close = (max_segment_close - min_segment_close) / max_segment_close
            scaling_factor_next_close = (max_next_segment_close - min_next_segment_close) / max_next_segment_close

            scaling_factor_current_high = (max_segment_high - min_segment_high) / max_segment_high
            scaling_factor_next_high = (max_next_segment_high - min_next_segment_high) / max_next_segment_high

            # If the scaling factors are similar, mark as self-affine fractal
            if abs(scaling_factor_current_close - scaling_factor_next_close) < scale_factor and \
               abs(scaling_factor_current_high - scaling_factor_next_high) < scale_factor:
                # Mark this as a self-affine fractal pattern
                stock_data.at[stock_data.index[i], 'Pattern'] = "Self-Affine Fractal"

    return stock_data


def calculate_accuracy(stock_data):
    total_data_points = len(stock_data)
    
    # Count occurrences of self-affine fractal pattern
    pattern_counts = {
        "Self-Affine Fractal": (stock_data['Pattern'] == "Self-Affine Fractal").sum()
    }

    # Calculate percentage accuracy for each pattern
    pattern_accuracy = {pattern: (count / total_data_points) * 100 for pattern, count in pattern_counts.items()}
    
    total_patterns = sum(pattern_counts.values())
    overall_accuracy = (total_patterns / total_data_points) * 100

    return pattern_accuracy, overall_accuracy, pattern_counts


def plot_fractal_patterns(stock_data, ticker, pattern_accuracy):
    plt.figure(figsize=(18, 7))  # Increased width for more spacing on x-axis
    plt.plot(stock_data['Close ^GSPC'], label='Close Price', linewidth=1.5)

    plotted_labels = set()  # Set to track which labels have been added to the legend

    for idx, row in stock_data.iterrows():
        if row['Pattern'] == "Self-Affine Fractal":
            label = "Self-Affine Fractal" if "Self-Affine Fractal" not in plotted_labels else ""
            plt.scatter(idx, row['Close ^GSPC'], color='orange', marker='o', label=label, alpha=1)
            plotted_labels.add("Self-Affine Fractal")

    # Display pattern counts on the right side of the plot
    text_x = stock_data.index[-1]  # Position at the far right of the graph
    y_start = stock_data['Close ^GSPC'].max() * 0.95  # Position text at the top

    for i, (pattern, count) in enumerate(pattern_accuracy.items()):
        plt.text(text_x, y_start - i * (stock_data['Close ^GSPC'].max() * 0.05), f"{pattern}: {count:.2f}%", 
                 fontsize=10, color='black', ha='right')

    plt.title(f'{ticker} Stock Price with Self-Affine Fractal Patterns')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.show()


if __name__ == "__main__":
    ticker = '^GSPC'
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    if 'Close ^GSPC' in stock_data.columns and 'High ^GSPC' in stock_data.columns:
        stock_data = detect_self_affine_fractal_patterns(stock_data)
        pattern_accuracy, overall_accuracy, pattern_counts = calculate_accuracy(stock_data)
        print(f"Overall accuracy: {overall_accuracy:.2f}%")
        print("Fractal Pattern Accuracy:")
        for pattern, accuracy in pattern_accuracy.items():
            print(f"{pattern}: {accuracy:.2f}%")
        print("Fractal Pattern Counts:")
        for pattern, count in pattern_counts.items():
            print(f"{pattern}: {count}")
        plot_fractal_patterns(stock_data, ticker, pattern_accuracy)
    else:
        print("Required columns are missing in the stock data.")
