import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv('/content/heart.csv') 

# Function to detect outliers using IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)  # First quartile (25th percentile)
    Q3 = data[column].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - 1.5 * IQR  # Lower bound for outliers
    upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers
    
    # Find outliers
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Detect outliers for all numeric columns
for col in df.select_dtypes(include=np.number).columns:
    print(f"\nColumn: {col}")
    outliers, lower, upper = detect_outliers_iqr(df, col)
    print(f"Lower Bound: {lower}, Upper Bound: {upper}")
    print(f"Number of Outliers: {len(outliers)}")
    if len(outliers) > 0:
        print("Outliers:")
        print(outliers[[col]])

from scipy import stats

def detect_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))  # Compute Z-scores
    outliers = data[z_scores > threshold]  # Outliers where Z-score > threshold
    return outliers

# Detect outliers for all numeric columns
for col in df.select_dtypes(include=np.number).columns:
    print(f"\nColumn: {col} (Z-Score Method)")
    outliers = detect_outliers_zscore(df, col)
    print(f"Number of Outliers: {len(outliers)}")
    if len(outliers) > 0:
        print(outliers[[col]])


def detect_outliers_modified_zscore(data, column, threshold=3.5):
    median = np.median(data[column])  # Compute median
    mad = np.median(np.abs(data[column] - median))  # Compute MAD
    modified_z_scores = 0.6745 * (data[column] - median) / mad  # Compute modified Z-scores
    outliers = data[np.abs(modified_z_scores) > threshold]  # Detect outliers
    return outliers

# Detect outliers for all numeric columns
for col in df.select_dtypes(include=np.number).columns:
    print(f"\nColumn: {col} (Modified Z-Score Method)")
    outliers = detect_outliers_modified_zscore(df, col)
    print(f"Number of Outliers: {len(outliers)}")
    if len(outliers) > 0:
        print(outliers[[col]])


from sklearn.neighbors import LocalOutlierFactor

def detect_outliers_lof(data, column, n_neighbors=20, contamination=0.05):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    data['LOF_Score'] = lof.fit_predict(data[[column]])  # LOF model
    outliers = data[data['LOF_Score'] == -1]  # -1 indicates outliers
    return outliers

# Detect outliers for all numeric columns
for col in df.select_dtypes(include=np.number).columns:
    print(f"\nColumn: {col} (LOF Method)")
    outliers = detect_outliers_lof(df, col)
    print(f"Number of Outliers: {len(outliers)}")
    if len(outliers) > 0:
        print(outliers[[col]])
