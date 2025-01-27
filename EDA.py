import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv('/content/heart.csv')

# Display first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Basic Info
print("\nDataset Info:")
print(df.info())

# Describe
print("\nBasic statistics of numerical columns:")
print(df.describe())

# Missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Correlation Heatmap

print("\nCorrelation Heatmap:")
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()  # Compute correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Heatmap')
plt.show()

# Boxplots for Numeric Features
print("\nBoxplots for Numeric Features:")
plt.figure(figsize=(14, 8))
sns.boxplot(data=df, palette='Set2')
plt.title('Boxplots of Numeric Features')
plt.xticks(rotation=45)
plt.show()