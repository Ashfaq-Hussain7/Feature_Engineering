import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneOut, StratifiedKFold
from sklearn.linear_model import LogisticRegression


# Step 1: Define Features and Target
X = df.drop(columns=['target'])  # All columns except the target
y = df['target']  # Target column

# Step 2: Choose a Classifier
model = LogisticRegression(max_iter=1000)

# 1. k-Fold Cross-Validation
print("\nPerforming k-Fold Cross-Validation:")
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
kfold_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(f"k-Fold Accuracy Scores: {kfold_scores}")
print(f"Mean Accuracy: {np.mean(kfold_scores):.2f}")
print(f"Standard Deviation: {np.std(kfold_scores):.2f}")

# 2. Leave-One-Out Cross-Validation (LOOCV)
print("\nPerforming Leave-One-Out Cross-Validation (LOOCV):")
loo = LeaveOneOut()
loo_scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
print(f"Mean Accuracy (LOOCV): {np.mean(loo_scores):.2f}")

# 3. Stratified k-Fold Cross-Validation
print("\nPerforming Stratified k-Fold Cross-Validation (No Repetition):")
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
stratified_scores = cross_val_score(model, X, y, cv=stratified_kfold, scoring='accuracy')
print(f"Stratified k-Fold Accuracy Scores: {stratified_scores}")
print(f"Mean Accuracy (Stratified k-Fold): {np.mean(stratified_scores):.2f}")
print(f"Standard Deviation: {np.std(stratified_scores):.2f}")
