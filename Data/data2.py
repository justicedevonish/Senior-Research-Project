import matplotlib.pyplot as plt

patterns = ['Bullish', 'Bearish', 'Double Bottom', 'Double Top', 'Alligator', 'Elliot Wave', 'Hurst Exponent', 'Self Affine', 'Williams']
counts = [14, 119, 608, 55, 59, 588, 644, 724, 301]

plt.figure(figsize=(10, 6))
bars = plt.bar(patterns, counts, color='red')
plt.xlabel('Patterns')
plt.ylabel('Counts')
plt.title('Pattern Counts')
plt.xticks(rotation=45)
plt.tight_layout()

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom')

plt.show()