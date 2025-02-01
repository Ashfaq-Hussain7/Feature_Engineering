import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load a sample dataset (Iris dataset as an example)
data = load_iris()
X = data.data  # Features
y = data.target  # Target labels
target_names = data.target_names

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)
pca_explained_variance_ratio = pca.explained_variance_ratio_

#Variance Ratio
print("PCA Explained Variance Ratio:", pca_explained_variance_ratio)

# Plot PCA results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i, target_name in enumerate(target_names):
    plt.scatter(X_pca[y_train == i, 0], X_pca[y_train == i, 1], label=target_name)
plt.title("PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()

# Perform LDA
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_train, y_train)
lda_explained_variance_ratio = np.var(X_lda, axis=0) / np.sum(np.var(X_lda, axis=0))

#Variance Ratio
print("LDA Explained Variance Ratio:", lda_explained_variance_ratio)


# Plot LDA results
plt.subplot(1, 2, 2)
for i, target_name in enumerate(target_names):
    plt.scatter(X_lda[y_train == i, 0], X_lda[y_train == i, 1], label=target_name)
plt.title("LDA Visualization")
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.legend()

plt.tight_layout()
plt.show()