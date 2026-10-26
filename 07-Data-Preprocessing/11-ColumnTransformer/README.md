# 🔄 ColumnTransformer in Machine Learning

## 📌 Overview

`ColumnTransformer` is a Scikit-learn utility used to apply different preprocessing transformations to different columns of a dataset.

In real-world machine learning projects, datasets often contain multiple types of features, such as numerical values, categorical variables, and text data. These features usually require different preprocessing techniques.

For example:
- Numerical features may require missing-value imputation and scaling.
- Categorical features may require imputation and one-hot encoding.
- Text features may require vectorization.

`ColumnTransformer` allows these transformations to be combined into a single preprocessing object that can be integrated with a machine learning pipeline.

## 🎯 Learning Objectives

- Understand the purpose of `ColumnTransformer`.
- Learn how to select columns for preprocessing.
- Apply different transformations to numerical and categorical data.
- Handle missing values in multiple feature types.
- Combine `ColumnTransformer` with `Pipeline`.
- Prevent data leakage during preprocessing.
- Integrate preprocessing with machine learning models.
- Understand output feature names and transformed data.

## 📚 Table of Contents

1. Introduction to ColumnTransformer
2. Why Use ColumnTransformer?
3. Syntax and Parameters
4. Basic Example
5. Numerical Feature Transformation
6. Categorical Feature Transformation
7. Handling Missing Values
8. Complete Machine Learning Workflow
9. ColumnTransformer with Pipeline
10. Accessing Transformed Features
11. Remainder Parameter
12. Handling Unknown Categories
13. Common Mistakes
14. Best Practices
15. Practice Exercises
16. Requirements
17. References

---

## 1. Introduction to ColumnTransformer

A dataset may contain features that require different preprocessing operations.

Consider the following dataset:

| Age | Salary | City | Purchased |
|---:|---:|---|---:|
| 22 | 25000 | Pune | 0 |
| 25 | 30000 | Mumbai | 0 |
| 30 | 45000 | Pune | 1 |
| 35 | 50000 | Delhi | 1 |

Here:

- `Age` and `Salary` are numerical features.
- `City` is a categorical feature.
- `Purchased` is the target variable.

Applying the same transformation to every feature would not be appropriate. `ColumnTransformer` allows numerical and categorical columns to be processed separately.

### Workflow

```text
              Input Dataset
                    |
                    v
             ColumnTransformer
                    |
          +---------+---------+
          |                   |
          v                   v
   Numerical Features   Categorical Features
          |                   |
          v                   v
    Missing Values       Missing Values
          |                   |
          v                   v
    Feature Scaling      One-Hot Encoding
          |                   |
          +---------+---------+
                    |
                    v
          Transformed Features
                    |
                    v
          Machine Learning Model
                    |
                    v
                Prediction
```

## 2. Why Use ColumnTransformer?

| Feature | Benefit |
|---|---|
| Column selection | Selects specific columns for transformation |
| Multiple transformations | Applies different operations to different feature groups |
| Missing-value handling | Supports separate imputation strategies |
| Encoding | Converts categorical features into numerical representations |
| Pipeline integration | Works with Scikit-learn pipelines |
| Reusability | Reuses fitted transformations on new data |
| Data leakage prevention | Learns preprocessing parameters from training data when correctly fitted |

## 3. Syntax and Parameters

### Basic Syntax

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("name", transformer, columns)
    ],
    remainder="drop"
)
```

### Important Parameters

| Parameter | Description |
|---|---|
| `transformers` | List of named transformations and their selected columns |
| `remainder` | Determines how unspecified columns are handled |
| `sparse_threshold` | Controls whether combined output is sparse |
| `n_jobs` | Number of jobs used for parallel transformation |
| `transformer_weights` | Applies weights to the outputs of individual transformers |
| `verbose_feature_names_out` | Controls prefixes in generated feature names |

Each transformer entry contains three elements:

1. A unique name.
2. A transformer object or the string `"drop"` or `"passthrough"`.
3. The column names, indices, or selector used to identify the columns.

## 4. Basic Example

```python
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data = pd.DataFrame({
    "Age": [22, 25, 30, 35],
    "Salary": [25000, 30000, 45000, 50000],
    "City": ["Pune", "Mumbai", "Pune", "Delhi"]
})

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", StandardScaler(), ["Age", "Salary"]),
        ("categorical", OneHotEncoder(), ["City"])
    ]
)

transformed_data = preprocessor.fit_transform(data)

print(transformed_data)
print("Output shape:", transformed_data.shape)
```

### Explanation

- `StandardScaler()` standardizes the numerical features.
- `OneHotEncoder()` converts the city categories into binary features.
- `fit_transform()` learns the required transformation parameters and transforms the supplied data.
- The output is a numerical array or a sparse matrix, depending on the configured transformers and output settings.

For real model evaluation, fit the preprocessor on training data only.

## 5. Numerical Feature Transformation

Numerical data often requires imputation, scaling, or other mathematical transformations.

Common techniques include:

- `SimpleImputer`
- `StandardScaler`
- `MinMaxScaler`
- `RobustScaler`
- `PowerTransformer`

### Example: Scaling Numerical Features

```python
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = pd.DataFrame({
    "Age": [20, 25, 30, 35, 40],
    "Salary": [20000, 30000, 40000, 50000, 60000]
})

preprocessor = ColumnTransformer(
    transformers=[
        ("age_scaler", StandardScaler(), ["Age"]),
        ("salary_scaler", MinMaxScaler(), ["Salary"])
    ]
)

result = preprocessor.fit_transform(data)

print(result)
```

Different scalers can be assigned to different numerical features based on the characteristics of the dataset.

## 6. Categorical Feature Transformation

Categorical features contain labels rather than continuous numerical measurements.

Common encoding techniques include:

- `OneHotEncoder`
- `OrdinalEncoder`
- Custom categorical transformers

### Example: One-Hot Encoding

```python
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

data = pd.DataFrame({
    "City": ["Pune", "Mumbai", "Delhi", "Pune"],
    "Department": ["IT", "HR", "Finance", "IT"]
})

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical_encoder",
            OneHotEncoder(handle_unknown="ignore"),
            ["City", "Department"]
        )
    ]
)

result = preprocessor.fit_transform(data)

print(result)
```

`OneHotEncoder` creates separate indicator features for the observed categories. Using `handle_unknown="ignore"` allows unseen categories during transformation without raising an error.

## 7. Handling Missing Values

Missing values can occur in both numerical and categorical columns. Different imputation strategies can be applied to each group.

```python
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

data = pd.DataFrame({
    "Age": [22, None, 30, 35, 40],
    "Salary": [25000, 30000, None, 50000, 60000],
    "City": ["Pune", "Mumbai", None, "Delhi", "Pune"]
})

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
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

transformed_data = preprocessor.fit_transform(data)

print(transformed_data)
```

### Explanation

- Numerical missing values are replaced with the median of each column.
- Numerical features are standardized.
- Categorical missing values are replaced with the most frequent category.
- Categorical values are one-hot encoded.

For production and evaluation workflows, fit this preprocessor only on the training partition.

## 8. Complete Machine Learning Workflow

This example combines numerical preprocessing, categorical preprocessing, train-test splitting, model training, and evaluation.

```python
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create a sample dataset
data = pd.DataFrame({
    "Age": [22, 25, 30, 35, 40, 28, 45, 32, 26, 38],
    "Salary": [
        25000, 30000, 45000, 50000, 60000,
        35000, 70000, 48000, 32000, 55000
    ],
    "City": [
        "Pune", "Mumbai", "Pune", "Delhi", "Mumbai",
        "Delhi", "Pune", "Mumbai", "Delhi", "Pune"
    ],
    "Purchased": [0, 0, 1, 1, 1, 0, 1, 1, 0, 1]
})

# Separate features and target
X = data.drop(columns=["Purchased"])
y = data["Purchased"]

# Identify feature groups
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

# Complete model pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

# Split before fitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train the pipeline
model_pipeline.fit(X_train, y_train)

# Predict
y_pred = model_pipeline.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
```

**Note:** This small dataset demonstrates implementation, not reliable model performance. Use a sufficiently large representative dataset and appropriate cross-validation for meaningful evaluation.

## 9. ColumnTransformer with Pipeline

`ColumnTransformer` and `Pipeline` serve complementary purposes.

| Component | Main responsibility |
|---|---|
| `ColumnTransformer` | Applies transformations to selected columns |
| `Pipeline` | Chains sequential transformations and an estimator |
| `SimpleImputer` | Fills missing values |
| `StandardScaler` | Standardizes numerical features |
| `OneHotEncoder` | Encodes categorical features |
| `LogisticRegression` | Performs classification |

A typical structure is:

```python
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])
```

This makes it possible to train, validate, tune, save, and reuse the complete workflow as one object.

## 10. Accessing Transformed Features

When using pandas DataFrames, you can obtain the output feature names from a fitted `ColumnTransformer`.

```python
feature_names = preprocessor.get_feature_names_out()

print(feature_names)
```

You can also configure the transformer to return pandas output in supported Scikit-learn versions:

```python
preprocessor.set_output(transform="pandas")

transformed_df = preprocessor.fit_transform(data)

print(transformed_df.head())
print(transformed_df.columns)
```

Some transformers or configurations may still produce sparse output or impose limitations on pandas output. Verify compatibility with the transformations you use.

## 11. Understanding the `remainder` Parameter

The `remainder` parameter controls what happens to columns not explicitly listed in `transformers`.

### `remainder="drop"`

Unspecified columns are discarded. This is the default behavior.

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), ["Age"])
    ],
    remainder="drop"
)
```

### `remainder="passthrough"`

Unspecified columns are retained without additional transformation.

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), ["Age"])
    ],
    remainder="passthrough"
)
```

### Custom estimator

A transformer can also be supplied as the `remainder` value to process remaining columns, subject to Scikit-learn's column-selection and ordering requirements.

## 12. Handling Unknown Categories

A model may encounter a category that was not present during training.

For example, the training dataset may contain `Pune` and `Mumbai`, while future data contains `Bengaluru`.

Using `handle_unknown="ignore"` helps prevent transformation errors:

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore")
```

The unseen category is represented without activating any of the learned category indicators for that feature.

Always consider how unseen categories should be handled for your particular application.

## 13. Common Mistakes

| Mistake | Solution |
|---|---|
| Applying the same transformation to every column | Group columns by data type and requirements |
| Fitting preprocessing on all data before splitting | Split first and fit within the training pipeline |
| Forgetting categorical encoding | Use an appropriate encoder |
| Leaving missing values unhandled | Add suitable imputation steps |
| Accidentally dropping important columns | Review the `remainder` setting |
| Encountering unknown categories | Configure an appropriate unknown-category strategy |
| Mixing feature and target columns | Exclude the target from preprocessing inputs |
| Assuming output is always a DataFrame | Check output type and sparse settings |
| Evaluating on training data only | Use held-out test data or appropriate cross-validation |

## 14. Best Practices

1. Use descriptive names for transformers.
2. Separate numerical and categorical features explicitly.
3. Keep learned preprocessing inside a pipeline.
4. Split data before fitting preprocessing.
5. Choose transformations based on data characteristics.
6. Use `handle_unknown="ignore"` when suitable for categorical inference.
7. Inspect transformed feature names and output dimensions.
8. Validate input columns and data types before inference.
9. Use cross-validation for model selection.
10. Keep the test set independent until final evaluation.
11. Document feature-selection decisions and preprocessing assumptions.
12. Save the fitted pipeline for consistent inference.

## 15. Practice Exercises

### Beginner

- Apply `StandardScaler` to numerical columns and `OneHotEncoder` to categorical columns.
- Use `ColumnTransformer` to transform two numerical and one categorical feature.
- Experiment with `remainder="drop"` and `remainder="passthrough"`.

### Intermediate

- Handle missing values in numerical and categorical columns.
- Build a preprocessing pipeline for a real-world CSV dataset.
- Display transformed feature names and dimensions.
- Compare `StandardScaler` and `MinMaxScaler`.

### Advanced

- Build an end-to-end pipeline with `ColumnTransformer` and `GridSearchCV`.
- Apply different encoders to different categorical feature groups.
- Save and load the complete fitted pipeline using `joblib`.
- Evaluate the workflow using cross-validation and multiple metrics.

## 16. Requirements

Install the required libraries:

```bash
pip install numpy pandas scikit-learn joblib
```

## 17. References

- [Scikit-learn: ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Scikit-learn: Pipelines and Composite Estimators](https://scikit-learn.org/stable/modules/compose.html)
- [Scikit-learn: OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [Scikit-learn: SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
- [Scikit-learn: Common Pitfalls and Data Leakage](https://scikit-learn.org/stable/common_pitfalls.html)

---

## ✅ Key Takeaways

- `ColumnTransformer` applies different preprocessing operations to different columns.
- Numerical and categorical features can be processed independently.
- `Pipeline` combines preprocessing with machine learning estimators.
- Splitting data before fitting prevents leakage from the held-out test set.
- The `remainder` parameter determines how unspecified columns are handled.
- A fitted preprocessing pipeline ensures consistent transformations during training and inference.

**Next topic:** Feature Selection — learn how to identify relevant features and reduce unnecessary input dimensions in machine learning.
