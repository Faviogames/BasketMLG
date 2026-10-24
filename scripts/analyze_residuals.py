import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/training_dataset_live_v4_stacked.csv')

# Calculate residual
df['residual'] = df['final_total'] - df['base_model_prediction']

print("Residual Stats:")
print(df['residual'].describe())

# Correlation with features
numeric_df = df.select_dtypes(include=[np.number])
correlations = numeric_df.corr()['residual'].sort_values(ascending=False)

print("\nTop Positive Correlations with Residual:")
print(correlations.head(10))

print("\nTop Negative Correlations with Residual:")
print(correlations.tail(10))

# Check correlation between base_model_prediction and residual
# If high, the base model has systematic bias
print(f"\nCorrelation (Base Prediction vs Residual): {df['base_model_prediction'].corr(df['residual']):.4f}")

# Check if residual is just noise
# Plot residual distribution
try:
    plt.figure(figsize=(10, 6))
    sns.histplot(df['residual'], kde=True)
    plt.title('Distribution of Residuals (Actual - Base)')
    plt.savefig('residual_distribution.png')
    print("\nSaved residual_distribution.png")
except:
    print("Could not save plot")
