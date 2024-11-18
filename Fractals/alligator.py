import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetches stock data for a given ticker symbol, start date, and end date.
def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    
    # Flatten the MultiIndex columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() for col in data.columns.values]
    
    print(data.columns)  # Print columns to debug
    return data

def calculate_alligator(stock_data, jaw_period=13, teeth_period=8, lips_period=5):
    # Smoothed moving averages for the Jaw, Teeth, and Lips
    stock_data['Jaw'] = stock_data[f'Close ^GSPC'].rolling(window=jaw_period).mean().shift(jaw_period // 2)
    stock_data['Teeth'] = stock_data[f'Close ^GSPC'].rolling(window=teeth_period).mean().shift(teeth_period // 2)
    stock_data['Lips'] = stock_data[f'Close ^GSPC'].rolling(window=lips_period).mean().shift(lips_period // 2)
    
    return stock_data

def detect_alligator_fractal(stock_data):
    stock_data['Pattern'] = None  # Initialize with None for no pattern
    potential_fractal_points = 0  # Track the number of potential fractal points

    for i in range(3, len(stock_data) - 3):
        # Check if a potential fractal point exists (Price above or below Alligator lines)
        if (stock_data['Close ^GSPC'][i] > stock_data['Jaw'][i] and 
            stock_data['Close ^GSPC'][i] > stock_data['Teeth'][i] and 
            stock_data['Close ^GSPC'][i] > stock_data['Lips'][i]) or \
           (stock_data['Close ^GSPC'][i] < stock_data['Jaw'][i] and 
            stock_data['Close ^GSPC'][i] < stock_data['Teeth'][i] and 
            stock_data['Close ^GSPC'][i] < stock_data['Lips'][i]):
            potential_fractal_points += 1
        
        # Bullish Fractal: Price is above Jaw, Teeth, and Lips
        if (stock_data['Close ^GSPC'][i] > stock_data['Jaw'][i] and 
            stock_data['Close ^GSPC'][i] > stock_data['Teeth'][i] and 
            stock_data['Close ^GSPC'][i] > stock_data['Lips'][i] and
            stock_data['Low ^GSPC'][i] < min(stock_data['Low ^GSPC'][i-1], stock_data['Low ^GSPC'][i-2], stock_data['Low ^GSPC'][i+1])):
            stock_data.at[stock_data.index[i], 'Pattern'] = "Bullish Fractal"
        
        # Bearish Fractal: Price is below Jaw, Teeth, and Lips
        elif (stock_data['Close ^GSPC'][i] < stock_data['Jaw'][i] and 
              stock_data['Close ^GSPC'][i] < stock_data['Teeth'][i] and 
              stock_data['Close ^GSPC'][i] < stock_data['Lips'][i] and
              stock_data['High ^GSPC'][i] > max(stock_data['High ^GSPC'][i-1], stock_data['High ^GSPC'][i-2], stock_data['High ^GSPC'][i+1])):
            stock_data.at[stock_data.index[i], 'Pattern'] = "Bearish Fractal"
    
    return stock_data, potential_fractal_points

def calculate_accuracy(stock_data, potential_fractal_points):
    
    # Count occurrences of each pattern
    pattern_counts = {
        "Bullish Fractal": (stock_data['Pattern'] == "Bullish Fractal").sum(),
        "Bearish Fractal": (stock_data['Pattern'] == "Bearish Fractal").sum()
    }

    # Calculate total detected fractal patterns
    total_patterns = pattern_counts["Bullish Fractal"] + pattern_counts["Bearish Fractal"]
    
    if total_patterns == 0 or potential_fractal_points == 0:
        return {}, 0, pattern_counts  # No patterns detected or no potential fractal points, accuracy is 0

    # Calculate percentage accuracy for each pattern, based on detected patterns only
    pattern_accuracy = {
        pattern: (count / total_patterns) * 100 
        for pattern, count in pattern_counts.items()
    }
    
    # Overall accuracy: fraction of detected patterns over potential fractal points
    overall_accuracy = (total_patterns / potential_fractal_points) * 100

    return pattern_accuracy, overall_accuracy, pattern_counts

def plot_fractal_patterns(stock_data, ticker, pattern_accuracy):
    plt.figure(figsize=(18, 7))  # Increased width for more spacing on x-axis
    plt.plot(stock_data['Close ^GSPC'], label='Close Price', linewidth=1.5)
    
    # Plot Alligator Indicator (Jaw, Teeth, Lips)
    plt.plot(stock_data['Jaw'], label='Jaw (13-period)', color='blue', linewidth=1.2)
    plt.plot(stock_data['Teeth'], label='Teeth (8-period)', color='green', linewidth=1.2)
    plt.plot(stock_data['Lips'], label='Lips (5-period)', color='red', linewidth=1.2)
    
    # Initialize variables to alternate text position and track plotted labels
    alternate = True
    plotted_labels = set()  # Set to track which labels have been added to the legend

    # Loop over detected patterns to plot them
    for idx, row in stock_data.iterrows():
        # Alternate text position to reduce overlap
        y_offset = 5 if alternate else -5
        alternate = not alternate  # Toggle position for the next label

        # Check the type of pattern and plot with label only if it hasn't been added before
        if row['Pattern'] == "Bullish Fractal":
            label = "Bullish Fractal" if "Bullish Fractal" not in plotted_labels else ""
            plt.scatter(idx, row['Low ^GSPC'], color='green', marker='v', label=label, alpha=1)
            plt.text(idx, row['Close ^GSPC'] + y_offset, row['Pattern'], color='green', fontsize=8, ha='center')
            plotted_labels.add("Bullish Fractal")
        
        elif row['Pattern'] == "Bearish Fractal":
            label = "Bearish Fractal" if "Bearish Fractal" not in plotted_labels else ""
            plt.scatter(idx, row['High ^GSPC'], color='red', marker='^', label=label, alpha=1)
            plt.text(idx, row['Close ^GSPC'] + y_offset, row['Pattern'], color='red', fontsize=8, ha='center')
            plotted_labels.add("Bearish Fractal")

    # Display pattern counts on the right side of the plot
    text_x = stock_data.index[-1]  # Position at the far right of the graph
    y_start = stock_data['Close ^GSPC'].max() * 0.95  # Position text at the top

    for i, (pattern, count) in enumerate(pattern_accuracy.items()):
        plt.text(text_x, y_start - i * (stock_data['Close ^GSPC'].max() * 0.05), f"{pattern}: {count:.2f}%", 
                 fontsize=10, color='black', ha='right')

    plt.title(f'{ticker} Stock Price with Alligator Indicator and Fractal Patterns')
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
    
    # Ensure the modified column names match what you access
    if 'High ^GSPC' in stock_data.columns and 'Low ^GSPC' in stock_data.columns:
        stock_data = calculate_alligator(stock_data)
        stock_data, potential_fractal_points = detect_alligator_fractal(stock_data)
        pattern_accuracy, overall_accuracy, pattern_counts = calculate_accuracy(stock_data, potential_fractal_points)
        
        # Print Overall Accuracy and Pattern Accuracies
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")
        print("Fractal Pattern Accuracy:")
        for pattern, accuracy in pattern_accuracy.items():
            print(f"{pattern}: {accuracy:.2f}%")
        
        print("Fractal Pattern Counts:")
        for pattern, count in pattern_counts.items():
            print(f"{pattern}: {count}")
        
        # Plot the results
        plot_fractal_patterns(stock_data, ticker, pattern_accuracy)
    else:
        print("Required columns are missing in the stock data.")
