import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression

# Step 1: Load Dataset
file_path = "/mnt/data/heart.csv"
df = pd.read_csv(file_path)

# Step 2: Define Features and Target
X = df.drop(columns=['target']).values  # All columns except 'target'
y = df['target'].values  # The 'target' column

# Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Logistic Regression
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Confusion Matrix and Metrics
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
misclassification_rate = 1 - accuracy
true_negative_rate = tn / (tn + fp)
false_positive_rate = fp / (fp + tn)

# Print Results
print("Confusion Matrix:")
print(conf_matrix)
print("\nMetrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Misclassification Rate: {misclassification_rate:.2f}")
print(f"True Negative Rate: {true_negative_rate:.2f}")
print(f"False Positive Rate: {false_positive_rate:.2f}")
