# 🤖 Scikit-Learn for Machine Learning

> **A practical, production-oriented guide to Machine Learning with Python and Scikit-Learn — from preprocessing and model training to evaluation, pipelines, and model selection.**

Scikit-Learn is one of the most widely used Python libraries for classical Machine Learning. It provides consistent APIs for **data preprocessing, supervised learning, unsupervised learning, model evaluation, feature selection, dimensionality reduction, pipelines, and hyperparameter tuning**.

This section builds on the Python, NumPy, Pandas, Matplotlib, and Seaborn foundations developed earlier in this repository and progresses toward complete end-to-end Machine Learning workflows.

---

## 📚 Table of Contents

* [What is Scikit-Learn?](#-what-is-scikit-learn)
* [Why Scikit-Learn?](#-why-scikit-learn)
* [Installation](#-installation)
* [Import Convention](#-import-convention)
* [Machine Learning Workflow](#-machine-learning-workflow)
* [Core Scikit-Learn API](#-core-scikit-learn-api)
* [Datasets](#-datasets)
* [Data Preprocessing](#-data-preprocessing)
* [Feature Engineering](#-feature-engineering)
* [Supervised Learning](#-supervised-learning)
* [Regression](#-regression)
* [Classification](#-classification)
* [Unsupervised Learning](#-unsupervised-learning)
* [Clustering](#-clustering)
* [Dimensionality Reduction](#-dimensionality-reduction)
* [Feature Selection](#-feature-selection)
* [Model Evaluation](#-model-evaluation)
* [Cross-Validation](#-cross-validation)
* [Hyperparameter Tuning](#-hyperparameter-tuning)
* [Pipelines](#-pipelines)
* [Ensemble Learning](#-ensemble-learning)
* [Model Persistence](#-model-persistence)
* [Common Mistakes](#-common-mistakes)
* [Recommended Project Structure](#-recommended-project-structure)
* [Learning Roadmap](#-learning-roadmap)
* [Quick Reference](#-quick-reference)
* [Projects](#-projects)
* [Best Practices](#-best-practices)
* [Practice Problems](#-practice-problems)
* [Key Takeaways](#-key-takeaways)
* [Next Step](#-next-step)

---

# 🧠 What is Scikit-Learn?

**Scikit-Learn** (`sklearn`) is an open-source Machine Learning library for Python.

It provides implementations of many classical Machine Learning algorithms and supporting utilities.

### Major areas include:

```text
Scikit-Learn
│
├── Preprocessing
│   ├── Scaling
│   ├── Encoding
│   ├── Imputation
│   └── Feature Transformation
│
├── Supervised Learning
│   ├── Regression
│   └── Classification
│
├── Unsupervised Learning
│   ├── Clustering
│   └── Dimensionality Reduction
│
├── Model Selection
│   ├── Train/Test Split
│   ├── Cross Validation
│   └── Hyperparameter Tuning
│
├── Evaluation
│   ├── Regression Metrics
│   ├── Classification Metrics
│   └── Clustering Metrics
│
├── Feature Selection
│   ├── Filter Methods
│   ├── Wrapper Methods
│   └── Embedded Methods
│
└── Pipelines
    ├── Preprocessing
    ├── Model Training
    └── Evaluation
```

---

# 🚀 Why Scikit-Learn?

Scikit-Learn provides a consistent interface across many Machine Learning algorithms.

For example:

```python
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

The same general pattern can be used with many different estimators.

### Advantages

* Simple and consistent API
* Large collection of classical ML algorithms
* Excellent preprocessing utilities
* Cross-validation support
* Hyperparameter optimization
* Pipeline support
* Feature selection
* Dimensionality reduction
* Evaluation metrics
* Dataset utilities
* Strong ecosystem integration
* Excellent documentation
* Suitable for experimentation and production-oriented workflows

---

# 📦 Installation

Install Scikit-Learn with pip:

```bash
pip install scikit-learn
```

Recommended complete environment:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Verify installation:

```python
import sklearn

print(sklearn.__version__)
```

---

# 🐍 Import Convention

Common imports:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
```

Example:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```

---

# 🔄 Machine Learning Workflow

A typical Machine Learning project follows:

```text
                 ┌─────────────────┐
                 │ Collect Dataset │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Understand Data │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Clean Data      │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Feature Engineer│
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Train/Test Split│
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Preprocessing   │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Train Model     │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Validate Model  │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Tune Parameters │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Final Evaluation│
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Save / Deploy   │
                 └─────────────────┘
```

---

# 🔑 Core Scikit-Learn API

Most Scikit-Learn estimators follow a predictable API.

## `fit()`

Train a model:

```python
model.fit(X_train, y_train)
```

---

## `predict()`

Generate predictions:

```python
predictions = model.predict(X_test)
```

---

## `predict_proba()`

Generate class probabilities for supported classifiers:

```python
probabilities = model.predict_proba(X_test)
```

---

## `transform()`

Transform data:

```python
X_scaled = scaler.transform(X)
```

---

## `fit_transform()`

Fit a transformer and transform the data:

```python
X_train_scaled = scaler.fit_transform(X_train)
```

---

## `score()`

Many estimators provide a default scoring method:

```python
score = model.score(X_test, y_test)
```

Always verify what `score()` means for the particular estimator rather than assuming the same metric applies everywhere.

---

# 🗃️ Datasets

Scikit-Learn provides several datasets for learning and experimentation.

## Built-in Toy Datasets

Examples include:

```python
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_diabetes,
)
```

Example:

```python
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

print(X.shape)
print(y.shape)
```

---

## Synthetic Datasets

Scikit-Learn can generate synthetic datasets.

```python
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=500,
    n_features=5,
    noise=10,
    random_state=42,
)
```

Classification:

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=500,
    n_features=10,
    random_state=42,
)
```

Clustering:

```python
from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=500,
    centers=3,
    random_state=42,
)
```

---

# 🧹 Data Preprocessing

Good preprocessing is essential for Machine Learning.

Scikit-Learn provides many preprocessing utilities.

---

## Standardization

Standardization transforms features approximately to:

```text
mean = 0
standard deviation = 1
```

Using:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Important

Fit the scaler using training data:

```python
scaler.fit(X_train)
```

Then transform both:

```python
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Do **not** learn scaling parameters from the test set.

---

# 📏 Min-Max Scaling

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Typically maps features into a specified range, commonly `[0, 1]`.

---

# 🛡️ Robust Scaling

Useful when outliers are important:

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

# 🧩 Handling Missing Values

Use imputers:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
```

Common strategies:

```text
mean
median
most_frequent
constant
```

---

# 🔤 Categorical Encoding

Machine Learning algorithms generally require numerical representations of categorical variables.

## One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore"
)

X_encoded = encoder.fit_transform(
    X_categorical
)
```

---

## Label Encoding

For target labels:

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)
```

Be careful about using label encoding for input features because integer codes can introduce unintended ordering.

---

# 🧱 ColumnTransformer

Different columns often require different preprocessing.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_columns,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns,
        ),
    ]
)
```

This becomes especially useful inside a Pipeline.

---

# 🛠️ Feature Engineering

Feature engineering transforms raw variables into useful model inputs.

Examples:

```python
df["income_per_year"] = (
    df["income"] / df["experience"].clip(lower=1)
)
```

Polynomial features:

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(
    degree=2,
    include_bias=False,
)

X_poly = poly.fit_transform(X)
```

---

# 🎯 Supervised Learning

Supervised learning uses labeled data.

```text
Input Features X
       ↓
     Model
       ↓
Prediction ŷ
       ↓
Compare with y
```

Two major tasks:

```text
Supervised Learning
│
├── Regression
│   └── Predict numerical values
│
└── Classification
    └── Predict classes
```

---

# 📈 Regression

Regression predicts continuous numerical values.

Examples:

```text
House Price
Salary
Temperature
Sales
Demand
Revenue
```

---

## Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(
    X_train,
    y_train,
)

predictions = model.predict(
    X_test
)
```

---

## Ridge Regression

Ridge adds L2 regularization.

```python
from sklearn.linear_model import Ridge

model = Ridge(
    alpha=1.0
)

model.fit(
    X_train,
    y_train,
)
```

---

## Lasso Regression

Lasso uses L1 regularization.

```python
from sklearn.linear_model import Lasso

model = Lasso(
    alpha=0.01
)

model.fit(
    X_train,
    y_train,
)
```

Lasso can shrink some coefficients to zero, which can be useful for feature selection in suitable settings.

---

## Elastic Net

Combines L1 and L2 regularization.

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5,
)
```

---

# 🧠 Classification

Classification predicts discrete classes.

Examples:

```text
Spam / Not Spam
Fraud / Not Fraud
Disease / No Disease
Customer Churn / No Churn
Class A / Class B / Class C
```

---

## Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train,
)

predictions = model.predict(
    X_test
)
```

Probabilities:

```python
probabilities = model.predict_proba(
    X_test
)
```

---

## K-Nearest Neighbors

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=5
)

model.fit(
    X_train,
    y_train,
)
```

---

## Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train,
)
```

---

## Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)
```

---

## Support Vector Machine

```python
from sklearn.svm import SVC

model = SVC(
    kernel="rbf"
)

model.fit(
    X_train,
    y_train,
)
```

---

# 🌳 Decision Trees

Decision trees recursively split data based on feature conditions.

Conceptually:

```text
             Feature ≤ threshold?
                 /       \
               Yes        No
               /           \
          Decision       Decision
             /               \
          Class A           Class B
```

Advantages:

* Easy to visualize
* Captures nonlinear relationships
* Handles interactions
* Minimal preprocessing for many tree-based models

Potential limitations:

* Can overfit
* Sensitive to training data
* A single tree may have high variance

---

# 🌲 Ensemble Learning

Ensemble methods combine multiple models.

Examples:

```text
Random Forest
Gradient Boosting
HistGradientBoosting
Voting
Stacking
Bagging
AdaBoost
```

---

## Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)
```

---

## Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    random_state=42
)

model.fit(
    X_train,
    y_train,
)
```

---

# 🔵 Unsupervised Learning

Unsupervised learning works without target labels.

```text
Dataset
   ↓
Algorithm
   ↓
Discover Structure
```

Common tasks:

* Clustering
* Dimensionality reduction
* Anomaly detection
* Representation analysis

---

# 🔵 Clustering

## K-Means

```python
from sklearn.cluster import KMeans

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init="auto",
)

labels = model.fit_predict(X)
```

---

## DBSCAN

```python
from sklearn.cluster import DBSCAN

model = DBSCAN(
    eps=0.5,
    min_samples=5,
)

labels = model.fit_predict(X)
```

---

## Agglomerative Clustering

```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(
    n_clusters=3
)

labels = model.fit_predict(X)
```

---

# 📉 Dimensionality Reduction

Dimensionality reduction transforms high-dimensional data into fewer dimensions.

---

## PCA

Principal Component Analysis:

```python
from sklearn.decomposition import PCA

pca = PCA(
    n_components=2
)

X_reduced = pca.fit_transform(X)
```

Inspect explained variance:

```python
print(
    pca.explained_variance_ratio_
)
```

---

## Why PCA?

PCA can be useful for:

* Visualization
* Compression
* Noise reduction
* Feature transformation
* Exploring high-dimensional datasets

PCA components are transformed combinations of the original features and should be interpreted carefully.

---

# 🎯 Feature Selection

Feature selection identifies useful input variables.

---

## Variance Threshold

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(
    threshold=0.0
)

X_selected = selector.fit_transform(X)
```

---

## SelectKBest

```python
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
)

selector = SelectKBest(
    score_func=f_classif,
    k=5,
)

X_selected = selector.fit_transform(
    X_train,
    y_train,
)
```

---

## Recursive Feature Elimination

```python
from sklearn.feature_selection import RFE

selector = RFE(
    estimator=model,
    n_features_to_select=5,
)

X_selected = selector.fit_transform(
    X_train,
    y_train,
)
```

---

# 📊 Model Evaluation

Training a model is only one part of Machine Learning.

You must evaluate its behavior on appropriate unseen data.

---

# 📈 Regression Metrics

## Mean Absolute Error

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(
    y_test,
    predictions,
)
```

Formula:

```text
MAE = mean(|y - ŷ|)
```

Lower values indicate smaller average absolute errors, measured in the target's units.

---

## Mean Squared Error

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(
    y_test,
    predictions,
)
```

---

## Root Mean Squared Error

```python
rmse = mean_squared_error(
    y_test,
    predictions,
    squared=False,
)
```

Depending on your installed Scikit-Learn version, you may also compute:

```python
rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions,
    )
)
```

---

## R² Score

```python
from sklearn.metrics import r2_score

r2 = r2_score(
    y_test,
    predictions,
)
```

R² describes the proportion of target variance accounted for by the model relative to a mean-baseline formulation.

---

# 🎯 Classification Metrics

## Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    predictions,
)
```

---

## Precision

```python
from sklearn.metrics import precision_score

precision = precision_score(
    y_test,
    predictions,
)
```

---

## Recall

```python
from sklearn.metrics import recall_score

recall = recall_score(
    y_test,
    predictions,
)
```

---

## F1 Score

```python
from sklearn.metrics import f1_score

f1 = f1_score(
    y_test,
    predictions,
)
```

---

## Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(
    y_test,
    predictions,
)

print(matrix)
```

Conceptually:

```text
                  Predicted
                Positive Negative

Actual Positive    TP       FN

Actual Negative    FP       TN
```

---

## Classification Report

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        predictions,
    )
)
```

---

# 🔁 Cross-Validation

A single train/test split may not provide enough information about model performance.

Cross-validation repeatedly splits the available training data into training and validation portions.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
)

print(scores)
print(scores.mean())
```

---

## K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

---

## Stratified K-Fold

Useful for classification when preserving class proportions across folds is important.

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

---

# 🔧 Hyperparameter Tuning

Hyperparameters are configuration values selected before or during model fitting rather than learned as ordinary model coefficients.

---

## GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

parameter_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameter_grid,
    cv=5,
    scoring="accuracy",
)

grid_search.fit(
    X_train,
    y_train,
)

print(
    grid_search.best_params_
)
```

---

## RandomizedSearchCV

Useful when the search space is large.

```python
from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=parameter_grid,
    n_iter=10,
    cv=5,
    random_state=42,
)

search.fit(
    X_train,
    y_train,
)
```

---

# 🔗 Pipelines

Pipelines combine preprocessing and modeling into a single workflow.

Example:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline(
    [
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            ),
        ),
    ]
)

pipeline.fit(
    X_train,
    y_train,
)

predictions = pipeline.predict(
    X_test
)
```

---

# 🛡️ Why Pipelines Matter

Pipelines help:

* Keep preprocessing consistent
* Reduce repeated code
* Prevent many forms of preprocessing leakage
* Integrate preprocessing with cross-validation
* Simplify deployment
* Make workflows easier to reproduce

A strong pattern is:

```text
Raw Data
   ↓
Pipeline
   ├── Imputation
   ├── Encoding
   ├── Scaling
   ├── Feature Engineering
   └── Model
   ↓
Prediction
```

---

# 🧱 ColumnTransformer + Pipeline

Production-style preprocessing often combines:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

numeric_pipeline = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)

categorical_pipeline = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    [
        (
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ),
    ]
)
```

Then:

```python
model_pipeline = Pipeline(
    [
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            ),
        ),
    ]
)
```

---

# 🚨 Data Leakage

Data leakage occurs when information unavailable at prediction time improperly influences model training.

A common mistake:

```python
scaler.fit_transform(X)
```

before splitting the data.

Better:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Even better for many workflows:

```python
Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ]
)
```

---

# 🌲 Ensemble Learning

Ensemble methods combine multiple estimators.

Important Scikit-Learn ensemble algorithms include:

```text
Random Forest
Extra Trees
Gradient Boosting
HistGradientBoosting
AdaBoost
Voting
Stacking
Bagging
```

Example:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)
```

---

# 💾 Model Persistence

A trained model can be serialized for later use.

## Joblib

```python
import joblib

joblib.dump(
    model,
    "model.joblib",
)
```

Load:

```python
model = joblib.load(
    "model.joblib"
)
```

### Important

Only load serialized model files from trusted sources. Model-loading mechanisms can execute code depending on the serialization format and environment.

For production systems, also record:

```text
Python version
Scikit-Learn version
Training data version
Feature definitions
Preprocessing configuration
Model parameters
Evaluation metrics
```

# 🗺️ Learning Roadmap

Follow this progression:

```text
Python
  ↓
NumPy
  ↓
Pandas
  ↓
Matplotlib
  ↓
Seaborn
  ↓
Scikit-Learn Basics
  ↓
Data Preprocessing
  ↓
Regression
  ↓
Classification
  ↓
Model Evaluation
  ↓
Cross Validation
  ↓
Feature Engineering
  ↓
Pipelines
  ↓
Hyperparameter Tuning
  ↓
Ensemble Learning
  ↓
Unsupervised Learning
  ↓
End-to-End ML Projects
```

---

# 📋 Quick Reference

| Task                          | Scikit-Learn Tool        |
| ----------------------------- | ------------------------ |
| Train/Test Split              | `train_test_split`       |
| Standardization               | `StandardScaler`         |
| Min-Max Scaling               | `MinMaxScaler`           |
| Robust Scaling                | `RobustScaler`           |
| Missing Values                | `SimpleImputer`          |
| One-Hot Encoding              | `OneHotEncoder`          |
| Column-Specific Processing    | `ColumnTransformer`      |
| Pipeline                      | `Pipeline`               |
| Linear Regression             | `LinearRegression`       |
| Ridge                         | `Ridge`                  |
| Lasso                         | `Lasso`                  |
| Logistic Regression           | `LogisticRegression`     |
| KNN                           | `KNeighborsClassifier`   |
| Decision Tree                 | `DecisionTreeClassifier` |
| Random Forest                 | `RandomForestClassifier` |
| SVM                           | `SVC`                    |
| K-Means                       | `KMeans`                 |
| DBSCAN                        | `DBSCAN`                 |
| PCA                           | `PCA`                    |
| Feature Selection             | `SelectKBest`            |
| Recursive Feature Elimination | `RFE`                    |
| Cross Validation              | `cross_val_score`        |
| Grid Search                   | `GridSearchCV`           |
| Random Search                 | `RandomizedSearchCV`     |
| Classification Metrics        | `sklearn.metrics`        |
| Model Persistence             | `joblib`                 |

---

# 🧪 Practice Problems

## Beginner

1. Load the Iris dataset.
2. Display its shape.
3. Separate `X` and `y`.
4. Split the data into training and testing sets.
5. Train Logistic Regression.
6. Calculate accuracy.

---

## Intermediate

1. Load the Wine dataset.
2. Standardize the features.
3. Train:

   * Logistic Regression
   * KNN
   * Random Forest
4. Compare their evaluation metrics.
5. Use cross-validation.
6. Visualize the confusion matrix.

---

## Advanced

Build a complete classification pipeline:

```text
CSV
 ↓
Pandas
 ↓
Missing Value Handling
 ↓
Train/Test Split
 ↓
ColumnTransformer
 ↓
Scaling
 ↓
One-Hot Encoding
 ↓
Model
 ↓
Cross Validation
 ↓
GridSearchCV
 ↓
Evaluation
 ↓
Model Persistence
```

---

# 🏗️ End-to-End ML Template

A clean Scikit-Learn workflow often looks like:

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Load data
df = pd.read_csv("data.csv")


# Separate features and target
X = df.drop(
    columns=["target"]
)

y = df["target"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# Build pipeline
pipeline = Pipeline(
    [
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            ),
        ),
    ]
)


# Train
pipeline.fit(
    X_train,
    y_train,
)


# Predict
predictions = pipeline.predict(
    X_test
)


# Evaluate
print(
    classification_report(
        y_test,
        predictions,
    )
)
```

---

# ⚠️ Common Mistakes

### 1. Scaling Before Train/Test Split

❌ Avoid:

```python
X_scaled = scaler.fit_transform(X)

X_train, X_test = train_test_split(
    X_scaled
)
```

Prefer fitting preprocessing only on training data, or use a Pipeline.

---

### 2. Fitting the Model on Test Data

❌:

```python
model.fit(
    X_test,
    y_test,
)
```

The test set should generally remain unseen until final evaluation.

---

### 3. Using the Wrong Metric

Accuracy alone may not describe performance adequately for every classification problem.

Consider:

```text
Precision
Recall
F1
ROC-AUC
PR-AUC
Confusion Matrix
```

depending on the task and error costs.

---

### 4. Ignoring Class Imbalance

Inspect:

```python
y.value_counts(
    normalize=True
)
```

and select evaluation metrics appropriate to the problem.

---

### 5. Overfitting

A model can perform strongly on training data while generalizing poorly.

Use:

```text
Validation
Cross-Validation
Regularization
Appropriate Model Complexity
```

---

### 6. Tuning on the Test Set

Do not repeatedly use the final test set to make modeling decisions.

Use validation/cross-validation for model development and reserve the test set for final assessment.

---

### 7. Ignoring Data Leakage

Always ask:

> "Could this information realistically be available at prediction time?"

If not, it should not influence training.

---

### 8. Treating All Models the Same

Different algorithms have different:

* Assumptions
* Hyperparameters
* Sensitivities
* Computational requirements
* Interpretability characteristics

---

# 🧠 Scikit-Learn Design Pattern

One of the most important concepts is the estimator API.

```python
model = SomeEstimator()

model.fit(
    X_train,
    y_train,
)

predictions = model.predict(
    X_test
)
```

Transformers follow:

```python
transformer.fit(
    X_train
)

X_train_transformed = transformer.transform(
    X_train
)

X_test_transformed = transformer.transform(
    X_test
)
```

Pipeline combines these operations:

```python
pipeline.fit(
    X_train,
    y_train,
)

predictions = pipeline.predict(
    X_test
)
```

This consistent interface is one of the core reasons Scikit-Learn is useful for building reusable ML workflows.

---

# 📊 Scikit-Learn Ecosystem

Scikit-Learn works naturally with:

```text
NumPy
   ↓
Numerical Computing

Pandas
   ↓
Data Manipulation

Matplotlib
   ↓
Visualization

Seaborn
   ↓
Statistical Visualization

Scikit-Learn
   ↓
Classical Machine Learning

Joblib
   ↓
Model Persistence
```

For deep learning, specialized frameworks such as PyTorch and TensorFlow are typically used instead of Scikit-Learn's classical estimators.

---

# 🧩 Suggested Repository Progression

This repository section should progress approximately as:

```text
05-Scikit-Learn
│
├── 01-Introduction
│
├── 02-Data-Preprocessing
│
├── 03-Regression
│
├── 04-Classification
│
├── 05-Clustering
│
├── 06-Dimensionality-Reduction
│
├── 07-Feature-Selection
│
├── 08-Model-Evaluation
│
├── 09-Hyperparameter-Tuning
│
├── 10-Pipelines
│
├── 11-Ensemble-Learning
│
└── 12-Projects
```

---

# 💡 Project Ideas

### 🏠 1. House Price Prediction

Use:

```text
Pandas
↓
Data Cleaning
↓
Feature Engineering
↓
ColumnTransformer
↓
Regression
↓
Cross Validation
↓
Hyperparameter Tuning
```

---

### 👥 2. Customer Churn Prediction

Use:

```text
Categorical Encoding
+
Numerical Scaling
+
Logistic Regression
+
Random Forest
+
Classification Metrics
```

---

### 🛡️ 3. Fraud Detection

Focus on:

```text
Class Imbalance
Precision
Recall
F1
PR-AUC
Confusion Matrix
```

---

### 🌸 4. Iris Classification

Compare:

```text
Logistic Regression
KNN
Decision Tree
Random Forest
SVM
```

---

### 🧬 5. Customer Segmentation

Use:

```text
Scaling
↓
K-Means
↓
Cluster Analysis
↓
PCA
↓
Visualization
```

---

# 🏆 Best Practices

### Data

* Understand the dataset before modeling.
* Validate data types.
* Check missing values.
* Investigate duplicates and invalid observations.
* Understand the target variable.
* Check class distribution.

### Preprocessing

* Fit preprocessing on training data.
* Use Pipelines where appropriate.
* Use `ColumnTransformer` for heterogeneous datasets.
* Keep transformations reproducible.

### Modeling

* Establish a baseline.
* Compare multiple appropriate models.
* Use cross-validation during model development.
* Tune hyperparameters systematically.
* Avoid unnecessary model complexity.

### Evaluation

* Select metrics based on the task.
* Keep the final test set separate.
* Inspect error patterns.
* Consider uncertainty and dataset limitations.

### Production

* Save preprocessing and model logic together when appropriate.
* Version datasets and models.
* Record package versions.
* Validate inputs.
* Monitor model behavior after deployment.

---

# 🧭 From Notebook to Production

A professional ML workflow should evolve from:

```text
Notebook Experiment
        ↓
Reusable Python Functions
        ↓
Pipeline
        ↓
Validation
        ↓
Configuration
        ↓
Model Artifact
        ↓
API / Application
        ↓
Monitoring
```

A model is only one component of a production Machine Learning system.

---

# 📈 Learning Objectives

After completing this section, you should be able to:

* Understand Scikit-Learn's estimator API.
* Prepare numerical and categorical data.
* Handle missing values.
* Scale numerical features.
* Encode categorical features.
* Build regression models.
* Build classification models.
* Apply clustering algorithms.
* Perform dimensionality reduction.
* Select useful features.
* Evaluate ML models.
* Apply cross-validation.
* Tune hyperparameters.
* Build preprocessing/model pipelines.
* Reduce preprocessing leakage.
* Compare model behavior.
* Save trained models.
* Structure reusable ML projects.

---

# 🧠 Key Takeaways

```text
1. Scikit-Learn provides a consistent ML API.

2. Separate training and evaluation data.

3. Fit preprocessing using training data.

4. Pipelines help make preprocessing reproducible.

5. Different tasks require different models and metrics.

6. Cross-validation provides more robust model-development
   estimates than relying on a single split.

7. Hyperparameter tuning should use validation procedures,
   not repeated peeking at the final test set.

8. Data leakage can produce misleading evaluation results.

9. A good ML workflow combines data preparation,
   modeling, validation, and evaluation.

10. Production ML requires more than simply saving a model.
```

---

# 🔗 Prerequisites

Before starting this section, it is recommended to complete:

```text
01-Python
      ↓
02-NumPy
      ↓
03-Pandas
      ↓
04-Matplotlib
      ↓
05-Seaborn
      ↓
06-Scikit-Learn
```

Recommended foundational knowledge:

* Python functions
* NumPy arrays
* Pandas DataFrames
* Data cleaning
* Data visualization
* Basic statistics
* Basic probability
* Linear algebra fundamentals

---

# 🚀 Next Step

After completing Scikit-Learn, continue toward:

```text
Machine Learning
      ↓
Advanced ML
      ↓
Feature Engineering
      ↓
Model Optimization
      ↓
End-to-End Projects
      ↓
MLOps
      ↓
Deep Learning
      ↓
Generative AI
```

---

## 👨‍💻 Author

**Kishor Patil**

Machine Learning • Artificial Intelligence • Python • Data Science

GitHub:

`https://github.com/Kishor055`

---

## 🤝 Contributing

Contributions are welcome.

If you find an issue or have an improvement:

1. Fork the repository.
2. Create a feature branch.
3. Add or improve the example.
4. Test the code.
5. Commit your changes.
6. Open a Pull Request.

Example:

```bash
git clone https://github.com/Kishor055/Machine-Learning.git

cd Machine-Learning

git checkout -b feature/scikit-learn-improvement
```

---

## ⭐ Support

If this repository helps you learn Machine Learning, consider giving it a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational purposes. Refer to the repository's license file for the applicable terms.

---

> **Learn the fundamentals → Build models → Validate properly → Engineer reliable pipelines → Create real-world Machine Learning systems.**
