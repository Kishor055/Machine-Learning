# 🧹 Data Preprocessing in Machine Learning

## 📌 Overview

**Data Preprocessing** is a fundamental step in the Machine Learning lifecycle that transforms raw data into a clean, consistent, and model-ready format.

Real-world datasets often contain missing values, duplicate records, inconsistent categories, outliers, different feature scales, and categorical variables that machine learning algorithms cannot directly process.

This section provides a structured, hands-on guide to data preprocessing using Python, NumPy, Pandas, and Scikit-learn. It covers essential techniques for preparing high-quality datasets and building reliable machine learning workflows.

## 🎯 Learning Objectives

- Understand the importance of data preprocessing in machine learning.
- Identify and handle missing, duplicate, and invalid data.
- Detect and manage outliers and inconsistent values.
- Encode categorical features into numerical representations.
- Apply feature scaling and mathematical transformations.
- Perform imputation and feature engineering.
- Build preprocessing workflows using `Pipeline` and `ColumnTransformer`.
- Prevent data leakage during model training and evaluation.
- Prepare reproducible datasets for machine learning algorithms.

## 📚 Topics Covered

### 1. Data Cleaning

Learn how to identify and correct problems in raw datasets before model training.

| No. | Topic | Description |
|---|---|---|
| 01 | [Introduction to Data Cleaning](./01-Data-Cleaning/) | Understand the fundamentals of cleaning raw datasets. |
| 02 | [Missing Data](./02-Missing-Data/) | Identify and handle missing observations. |
| 03 | [Duplicate Data](./03-Duplicate-Data/) | Detect and remove duplicate records. |
| 04 | [Inconsistent Values](./04-Inconsistent-Values/) | Standardize inconsistent labels and formats. |
| 05 | [Outliers](./05-Outliers/) | Detect and handle unusual observations. |
| 06 | [Invalid Data](./06-Invalid-Data/) | Identify invalid values and incorrect data types. |
| 07 | [Data Leakage](./07-Data-Leakage/) | Understand and prevent information leakage during model development. |

### 2. Data Preprocessing Techniques

Explore transformations that convert cleaned data into a format suitable for machine learning algorithms.

| No. | Topic | Description |
|---|---|---|
| 01 | [Feature Scaling](./01-Feature-Scaling/) | Understand why features with different scales may require normalization or standardization. |
| 02 | [Normalization](./02-Normalization/) | Scale feature values using normalization techniques. |
| 03 | [Standardization](./03-Standardization/) | Transform features using their mean and standard deviation. |
| 04 | [Encoding](./04-Encoding/) | Convert categorical features into numerical representations. |
| 05 | [Label Encoding](./05-Label-Encoding/) | Represent categories using integer labels. |
| 06 | [One-Hot Encoding](./06-One-Hot-Encoding/) | Represent nominal categories using binary indicator features. |
| 07 | [Ordinal Encoding](./07-Ordinal-Encoding/) | Encode categories with a meaningful order. |
| 08 | [Imputation](./08-Imputation/) | Estimate and replace missing values. |
| 09 | [Transformation](./09-Transformation/) | Apply mathematical transformations to features. |
| 10 | [Pipelines](./10-Pipelines/) | Combine preprocessing and modeling into reusable workflows. |
| 11 | [ColumnTransformer](./11-ColumnTransformer/) | Apply different transformations to selected columns. |

> **Note:** The links above follow the topic naming convention used in this learning guide. Ensure each linked directory exists in the repository and adjust any paths to match your actual folder structure.

## 🛠️ Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical computing and array operations |
| Pandas | Data manipulation and cleaning |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, encoding, imputation, and pipelines |
| Jupyter Notebook | Interactive experimentation and documentation |

## ⚙️ Installation

Install the required Python libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

## 🔄 Data Preprocessing Workflow

A typical preprocessing workflow follows these steps:

```text
Raw Dataset
     |
     v
Data Inspection
     |
     v
Data Cleaning
     |
     +--> Handle Missing Values
     +--> Remove Duplicates
     +--> Correct Invalid Values
     +--> Investigate Outliers
     |
     v
Train-Test Split
     |
     v
Feature Preprocessing
     |
     +--> Imputation
     +--> Categorical Encoding
     +--> Feature Scaling
     +--> Feature Transformation
     |
     v
Preprocessing Pipeline
     |
     v
Machine Learning Model
     |
     v
Model Evaluation
```

**Important:** Split the dataset before fitting transformations that learn parameters from data. Fit these transformations on the training set and apply the learned transformations to validation and test data. This helps prevent data leakage.

## 💻 Example: Basic Data Preprocessing

The following example demonstrates missing-value imputation, categorical encoding, feature scaling, and integration with a machine learning model.

```python
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create a sample dataset
data = pd.DataFrame({
    "Age": [22, 25, None, 35, 40, 28, 45, 32],
    "Salary": [25000, 30000, 45000, None, 60000, 35000, 70000, 48000],
    "City": [
        "Pune", "Mumbai", "Pune", "Delhi",
        "Mumbai", "Delhi", "Pune", "Mumbai"
    ],
    "Purchased": [0, 0, 1, 1, 1, 0, 1, 1]
})

# Separate features and target
X = data.drop(columns="Purchased")
y = data["Purchased"]

# Identify feature types
numerical_features = ["Age", "Salary"]
categorical_features = ["City"]

# Numerical preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine transformations
preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Create the complete pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression())
])

# Split before fitting
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# Train and evaluate
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

This small dataset is intended to demonstrate the workflow, not to establish reliable predictive performance. Use sufficiently large datasets and suitable cross-validation for meaningful evaluation.

## 📈 Why Is Data Preprocessing Important?

- **Improved data quality:** Addresses missing, invalid, and inconsistent observations.
- **Better model compatibility:** Converts features into appropriate numerical formats.
- **Consistent feature representation:** Applies repeatable transformations to new data.
- **Reliable evaluation:** Helps avoid data leakage when used correctly.
- **Improved workflow efficiency:** Automates repeated preprocessing tasks.
- **Reproducibility:** Makes preprocessing decisions easier to document and reproduce.

## 🧪 Practice Exercises

### Beginner

- Load a CSV dataset using Pandas.
- Identify missing values and duplicate records.
- Convert columns to appropriate data types.
- Apply standardization and normalization.
- Encode categorical variables.

### Intermediate

- Compare different missing-value imputation strategies.
- Detect outliers using the IQR method.
- Compare `OneHotEncoder` and `OrdinalEncoder`.
- Apply transformations to skewed numerical features.
- Use `ColumnTransformer` for mixed feature types.

### Advanced

- Build an end-to-end preprocessing pipeline.
- Integrate preprocessing with a classification or regression model.
- Use cross-validation with a preprocessing pipeline.
- Demonstrate how preprocessing leakage affects evaluation.
- Save and reload a fitted pipeline using `joblib`.

## 📖 References

- [Scikit-learn: Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Scikit-learn: Imputation of Missing Values](https://scikit-learn.org/stable/modules/impute.html)
- [Scikit-learn: Pipelines and Composite Estimators](https://scikit-learn.org/stable/modules/compose.html)
- [Scikit-learn: Common Pitfalls and Data Leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)

## 🎯 Learning Outcomes

After completing this section, you should be able to:

- Inspect and clean real-world datasets.
- Handle missing values, duplicates, outliers, and inconsistent data.
- Encode categorical features and scale numerical features.
- Select appropriate transformations for different feature types.
- Build reusable preprocessing pipelines.
- Prepare data for regression, classification, and other machine learning tasks.
- Apply preprocessing techniques without leaking information from evaluation data.

---

**Next Step:** Continue to the [Model Evaluation](../../10-Model-Evaluation/) section to learn how to measure model performance using appropriate evaluation metrics.

**Repository:** [Machine Learning — GitHub](https://github.com/Kishor055/Machine-Learning)
