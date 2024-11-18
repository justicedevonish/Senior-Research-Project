import pandas as pd

patterns = ['Bullish', 'Bearish', 'Double Bottom', 'Double Top', 'Alligator', 'Elliot Wave', 'Hurst Exponent', 'Self Affine', 'Williams']
headers = ['Prediction Accuracy %', 'Return on Investment %', 'Win/Loss Ratio', 'False Positives']

data = {header: [None] * len(patterns) for header in headers}
data['Pattern'] = patterns

df = pd.DataFrame(data)

df.set_index('Pattern', inplace=True)

print(df)