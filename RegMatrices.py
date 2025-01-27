import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load Dataset
file_path = "/mnt/data/heart.csv"
df = pd.read_csv(file_path)

# Define Features and Target
X = df.drop(columns=['target']).values  # All columns except 'target'
y = df['target'].values  # The 'target' column

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Compute Regression Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rel_mse = mse / np.var(y_test)  # Relative MSE: MSE / Variance of actual values
coefficient_of_variation = rmse / np.mean(y_test)  # RMSE / Mean of actual values

# Print Results
print("Regression Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Relative MSE (RelMSE): {rel_mse:.2f}")
print(f"Coefficient of Variation (CV): {coefficient_of_variation:.2f}")
