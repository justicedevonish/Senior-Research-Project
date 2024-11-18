import matplotlib.pyplot as plt

patterns = ['Bullish', 'Bearish', 'Double Bottom', 'Double Top', 'Alligator', 'Elliot Wave', 'Hurst Exponent', 'Self Affine', 'Williams']
accuracy = [99.19, 93.28, 83.55, 72.73, 91.01, 74.66, 71.48, 71.97, 96.35]

plt.figure(figsize=(12, 6))
bars = plt.barh(patterns, accuracy, color='red', height=0.6)

plt.title('Accuracy of Fractal Patterns')
plt.ylabel('Fractal Patterns')
plt.xlabel('Accuracy (%)')

plt.xlim(0, 100)

for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height() / 2.0, f'{width:.2f}%', ha='left', va='center')

plt.tight_layout()
plt.show()
