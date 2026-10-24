import pandas as pd

df = pd.read_csv('data/training_dataset_meta_v1.csv')
print("Bookie Line Stats:")
print(df['bookie_line'].describe())

zeros = (df['bookie_line'] == 0).sum()
print(f"\nRows with bookie_line = 0: {zeros}")

# Check if there are other low values
low_values = (df['bookie_line'] < 100).sum()
print(f"Rows with bookie_line < 100: {low_values}")
