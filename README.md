# Feature Engineering:
# 🧠 Machine Learning Project: Heart Dataset + Iris Dimensionality Reduction

This project includes various machine learning and data analysis tasks performed on the `heart.csv` and `Iris` datasets using Python and popular libraries like `pandas`, `scikit-learn`, `seaborn`, and `matplotlib`.

---

## 📁 Datasets

- `heart.csv`: Health indicators used for heart disease prediction.
- `Iris`: Classic dataset used to demonstrate dimensionality reduction techniques.

---

## 📊 1. Exploratory Data Analysis (EDA)

Performed using `pandas`, `seaborn`, and `matplotlib`.

### ✅ Includes:
- Dataset overview and structure
- Missing values
- Descriptive statistics
- Correlation heatmap
- Boxplots for outlier visualization

```python
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
sns.boxplot(data=df)
📈 2. Evaluation Metrics
🟠 Classification:
Confusion Matrix

Accuracy

Precision

Recall

F1-Score

Misclassification Rate

True Negative Rate (TNR)

False Positive Rate (FPR)

🔵 Regression:
Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

Relative MSE

Coefficient of Variation (CV)

🔄 3. Cross-Validation Techniques
Implemented using scikit-learn:

Stratified k-Fold Cross-Validation

Leave-One-Out Cross-Validation (LOOCV)

Repeated Stratified k-Fold Cross-Validation

python
Copy
Edit
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, RepeatedStratifiedKFold
🧪 4. Outlier Detection Methods
✅ Techniques Used:
Interquartile Range (IQR) Method

Z-Score

Modified Z-Score

Local Outlier Factor (LOF)

python
Copy
Edit
from sklearn.neighbors import LocalOutlierFactor
These methods identify anomalous data points based on statistical and density-based criteria.

🌺 5. PCA and LDA on Iris Dataset
✅ Dimensionality Reduction Techniques:
📌 Principal Component Analysis (PCA)
Unsupervised technique

Projects data to 2D while retaining maximum variance

Helps visualize class separation

📌 Linear Discriminant Analysis (LDA)
Supervised technique

Maximizes class separability in lower-dimensional space

📚 Libraries Used:
python
Copy
Edit
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
📊 Output:
2D plots of PCA and LDA-transformed Iris dataset using matplotlib

💻 How to Run
Install required dependencies:

bash
Copy
Edit
pip install numpy pandas scikit-learn matplotlib seaborn
Run the Python script or Jupyter notebook to explore the complete pipeline.

📂 Project Structure
Copy
Edit
.
├── heart.csv
├── iris_pca_lda.py
├── eda_analysis.py
├── README.md
👨‍💻 Author
Created with ❤️ using Python, NumPy, Scikit-Learn, and Matplotlib.
