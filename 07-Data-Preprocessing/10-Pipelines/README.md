# 🔗 Machine Learning Pipelines

## 📌 Overview

A **Machine Learning Pipeline** is a sequence of data processing and modeling steps organized into a single workflow. It automates the process of transforming raw data into predictions while ensuring that preprocessing operations are applied consistently during training and testing.

Pipelines help prevent data leakage, simplify model development, improve reproducibility, and make machine learning workflows easier to maintain.

In Scikit-learn, pipelines are commonly created using `Pipeline` and `ColumnTransformer`.

## 🎯 Learning Objectives

- Understand the concept and importance of ML pipelines.
- Build end-to-end machine learning workflows.
- Combine multiple preprocessing steps into a single pipeline.
- Apply different transformations to numerical and categorical features.
- Integrate preprocessing with machine learning models.
- Prevent data leakage during model training and evaluation.
- Perform hyperparameter tuning using pipelines.
- Save and reuse trained pipelines.

## 📚 Table of Contents

1. Introduction to ML Pipelines
2. Why Use Pipelines?
3. Important Scikit-learn Components
4. Basic Pipeline Example
5. Numerical and Categorical Preprocessing
6. Complete ML Pipeline
7. Pipelines and Data Leakage
8. Hyperparameter Tuning
9. Saving and Loading Pipelines
10. Advantages and Limitations
11. Best Practices
12. Applications
13. Practice Exercises
14. References

---

## 1. Introduction to ML Pipelines

A typical machine learning workflow involves several steps:

1. Load the dataset.
2. Separate features and target.
3. Split data into training and testing sets.
4. Handle missing values.
5. Encode categorical variables.
6. Scale numerical features when necessary.
7. Train a machine learning model.
8. Evaluate model performance.
9. Use the trained model to make predictions.

Without a pipeline, these operations must be managed separately. A pipeline combines compatible steps into a single reusable object.

### Workflow

```text
Raw Dataset
     |
     v
Train-Test Split
     |
     v
Data Preprocessing
     |
     +--> Missing Value Imputation
     |
     +--> Encoding
     |
     +--> Feature Scaling
     |
     v
Machine Learning Model
     |
     v
Predictions
     |
     v
Model Evaluation
```

## 2. Why Use Pipelines?

| Feature | Benefit |
|---|---|
| Automation | Combines multiple operations into one workflow |
| Consistency | Applies the same fitted transformations to new data |
| Data leakage prevention | Keeps preprocessing fitted on training data |
| Reusability | Allows the complete workflow to be reused |
| Maintainability | Makes code easier to organize and update |
| Hyperparameter tuning | Supports tuning preprocessing and model parameters together |
| Deployment | Packages preprocessing and prediction into one object |

## 3. Important Scikit-learn Components

### `Pipeline`

Connects multiple transformations and an optional final estimator in sequence.

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

Each intermediate step must implement `fit` and `transform`. The final estimator generally implements `fit` and `predict`.

### `ColumnTransformer`

Applies different preprocessing operations to different columns.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer([
    ("numerical", StandardScaler(), ["Age", "Salary"]),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), ["City"])
])
```

### `make_pipeline`

Creates a pipeline without requiring manually assigned step names.

```python
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

pipeline = make_pipeline(
    SimpleImputer(strategy="mean"),
    StandardScaler(),
    LinearRegression()
)
```

Scikit-learn automatically generates the step names.

## 4. Basic Pipeline Example

This example demonstrates missing-value imputation, feature scaling, and model training using a numerical dataset.

```python
import numpy as np

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create a sample regression dataset
X, y = make_regression(
    n_samples=200,
    n_features=4,
    noise=10,
    random_state=42
)

# Introduce missing values
X[0, 0] = np.nan
X[5, 2] = np.nan

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create the pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
```

**What happens here?**

- `SimpleImputer` fills missing numerical values using training-set means.
- `StandardScaler` standardizes the input features.
- `LinearRegression` learns the relationship between features and the target.
- `predict()` automatically applies the fitted preprocessing steps before generating predictions.

## 5. Numerical and Categorical Preprocessing

Real-world datasets often contain both numerical and categorical features. `ColumnTransformer` lets us process them independently.

For example:

- Numerical columns: `Age`, `Salary`
- Categorical columns: `City`, `Department`
- Target column: `Purchased`

```python
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

data = pd.DataFrame({
    "Age": [22, 25, 30, 35, 40, 28, 45, 32],
    "Salary": [25000, 30000, 45000, 50000, 60000, 35000, 70000, 48000],
    "City": [
        "Pune", "Mumbai", "Pune", "Delhi",
        "Mumbai", "Delhi", "Pune", "Mumbai"
    ],
    "Purchased": [0, 0, 1, 1, 1, 0, 1, 1]
})

X = data.drop(columns="Purchased")
y = data["Purchased"]

numerical_features = ["Age", "Salary"]
categorical_features = ["City"]

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression())
])

model_pipeline.fit(X, y)

predictions = model_pipeline.predict(X)

print("Predictions:", predictions)
```

**Important:** This small dataset demonstrates pipeline construction only. For reliable evaluation, split the dataset before fitting the pipeline and evaluate on unseen test data.

## 6. Complete Machine Learning Pipeline

The following example demonstrates a complete classification workflow with separate training and testing data.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset
dataset = load_breast_cancer(as_frame=True)

X = dataset.data
y = dataset.target

# Split data before preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Build the pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

The pipeline learns preprocessing parameters from the training data and applies those learned parameters to the test data.

## 7. Pipelines and Data Leakage

**Data leakage** occurs when information that would not legitimately be available during model training influences training or evaluation.

### Incorrect approach

```python
# Avoid fitting preprocessing on the complete dataset
X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)
```

The scaler learns its statistics from both training and test features before the split.

### Correct approach

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
```

Place preprocessing inside the pipeline and split the data before fitting. This ensures that learned preprocessing parameters come from the training partition.

## 8. Hyperparameter Tuning with Pipelines

Pipelines work with `GridSearchCV` and `RandomizedSearchCV`, enabling preprocessing and model parameters to be tuned together.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000))
])

param_grid = {
    "imputer__strategy": ["mean", "median"],
    "model__C": [0.1, 1, 10],
    "model__solver": ["lbfgs"]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

best_pipeline = grid_search.best_estimator_
print("Test Accuracy:", best_pipeline.score(X_test, y_test))
```

The double underscore (`__`) separates a pipeline step name from one of its parameters.

For example, `model__C` refers to the `C` parameter of the step named `model`.

## 9. Saving and Loading Pipelines

A fitted pipeline can be saved and reused for inference without manually repeating preprocessing.

```python
import joblib

# Save the fitted pipeline
joblib.dump(pipeline, "ml_pipeline.joblib")

# Load the saved pipeline
loaded_pipeline = joblib.load("ml_pipeline.joblib")

# Predict using new data
predictions = loaded_pipeline.predict(X_test)

print(predictions)
```

**Security note:** Only load joblib or pickle files from trusted sources, because loading them can execute arbitrary code. Keep the required Python and Scikit-learn versions compatible.

## 10. Advantages and Limitations

### Advantages

- Automates preprocessing and model training.
- Reduces duplicated code.
- Improves consistency between training and inference.
- Helps prevent preprocessing leakage.
- Supports cross-validation and hyperparameter optimization.
- Makes model deployment easier.

### Limitations

- Complex workflows can require nested pipelines.
- Debugging transformations may require inspecting individual steps.
- Custom transformations must follow Scikit-learn's estimator conventions.
- Incorrect pipeline design can still produce leakage or invalid evaluation.
- Some transformations may increase memory usage or feature dimensionality.

## 11. Best Practices

1. Split data before fitting learned preprocessing steps.
2. Keep preprocessing inside the pipeline.
3. Use `ColumnTransformer` for mixed data types.
4. Select transformations based on the model and data distribution.
5. Use cross-validation for model selection.
6. Keep the test set separate until final evaluation.
7. Handle unseen categories using appropriate encoding settings.
8. Validate input column names, types, and order at inference time.
9. Save the fitted pipeline rather than only the final estimator.
10. Track package versions and document the training dataset.
11. Avoid including identifiers or target-derived features that cause leakage.
12. Evaluate the complete workflow on representative, unseen data.

## 12. Applications

Machine learning pipelines are widely used in:

- Customer churn prediction
- Fraud detection
- Credit risk assessment
- House price prediction
- Medical classification
- Customer segmentation
- Recommendation systems
- Natural language processing
- Production ML inference systems

## 13. Practice Exercises

### Beginner

- Create a pipeline containing `SimpleImputer` and `LinearRegression`.
- Apply `StandardScaler` before a classification model.
- Train and evaluate a pipeline on a built-in Scikit-learn dataset.

### Intermediate

- Build a `ColumnTransformer` for numerical and categorical columns.
- Compare a pipeline using `StandardScaler` with one using `MinMaxScaler`.
- Use `GridSearchCV` to tune model parameters.

### Advanced

- Create a custom transformer using `BaseEstimator` and `TransformerMixin`.
- Build nested pipelines for mixed data types.
- Save a complete pipeline and create a prediction script.
- Evaluate a pipeline using cross-validation and multiple scoring metrics.

## 14. Requirements

Install the required libraries:

```bash
pip install numpy pandas scikit-learn joblib
```

## 15. References

- [Scikit-learn: Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [Scikit-learn: ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Scikit-learn: Compose and Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [Scikit-learn: Common Pitfalls and Data Leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [Scikit-learn: GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

---

## ✅ Key Takeaways

- A pipeline combines preprocessing and modeling into one workflow.
- `Pipeline` chains sequential transformations and an estimator.
- `ColumnTransformer` handles different feature types independently.
- Fitting preprocessing only on training data helps prevent leakage.
- Pipelines simplify cross-validation, hyperparameter tuning, deployment, and reproducibility.

**Next topic:** Model Evaluation and Validation — learn how to measure model performance and assess generalization on unseen data.
