# 📊 Standardization in Machine Learning

> **Standardization = Center Features Around Zero → Scale by Standard Deviation → Improve Numerical Consistency**

Standardization is a feature transformation technique that adjusts numerical features so that they have a mean of approximately `0` and a standard deviation of approximately `1`.

It is commonly used in Machine Learning algorithms that are sensitive to feature magnitudes, including Logistic Regression, Support Vector Machines, K-Nearest Neighbors, Ridge Regression, Lasso Regression, Neural Networks, and Principal Component Analysis.

---

# 🌱 GROW → INSPECT → STANDARDIZE → TRAIN → EVALUATE 🚀

```
                    📊 Raw Dataset
                          │
                          ▼
                   🔍 Inspect Features
                          │
                          ▼
                   ✂️ Train-Test Split
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
          Training Data        Testing Data
                │                   │
                ▼                   │
       Calculate Mean & Std          │
                │                   │
                ▼                   │
       Fit StandardScaler            │
                │                   │
                ├── Transform Train │
                │                   ▼
                └────────────── Transform Test
                          │
                          ▼
                    🤖 Train Model
                          │
                          ▼
                    📈 Evaluate
```

## 📚 Table of Contents

1. What Is Standardization?
2. Why Standardization Is Important
3. Mathematical Formula
4. Step-by-Step Calculation
5. Understanding Z-Scores
6. Standardization vs Normalization
7. Standardization vs Min-Max Scaling
8. StandardScaler in Scikit-Learn
9. Manual Standardization with Python
10. Standardization with NumPy
11. Standardization with Pandas
12. Standardizing Multiple Features
13. Train-Test Split and Standardization
14. Preventing Data Leakage
15. Standardization with Pipelines
16. Standardization with Cross-Validation
17. Standardization and Outliers
18. Standardization and Missing Values
19. Standardization and Categorical Features
20. Standardization for Different Algorithms
21. Complete Classification Example
22. Complete Regression Example
23. Visualizing Standardization
24. Common Mistakes
25. Best Practices
26. Mini Projects
27. Exercises
28. Checklist
29. Project Structure
30. Key Takeaways
31. Next Step

---

# 🎯 What Is Standardization?

Standardization transforms a numerical feature by subtracting its mean and dividing by its standard deviation.

The resulting feature is expressed in terms of how far each observation is from the mean, measured in standard deviations.

### Example

Before standardization:

| Student | Study Hours | Annual Family Income |
| ------- | ----------- | -------------------- |
| A       | 2           | 200,000              |
| B       | 4           | 300,000              |
| C       | 6           | 400,000              |
| D       | 8           | 500,000              |
| E       | 10          | 600,000              |

The features have different numerical magnitudes.

After standardization, each feature is transformed independently using statistics learned from the data.

```
Before:
Study Hours → 2, 4, 6, 8, 10
Income      → 200000, 300000, 400000, 500000, 600000

After:
Study Hours → standardized values
Income      → standardized values
```

The exact transformed values depend on the mean and standard deviation used.

**Important:** Standardization does not make all features identical. It changes their centers and scales while preserving their relative ordering when the scale is positive.

---

# 🤔 Why Standardization Is Important

Consider a dataset with:

```
Age        → 18 to 80
Salary     → 20,000 to 500,000
Experience → 0 to 40
```

Some algorithms are affected by the relative magnitudes of these features.

For example, distance-based algorithms may assign excessive influence to Salary because its numerical differences are much larger.

Standardization helps put the features on comparable scales.

### Benefits

- Reduces differences in feature magnitude.
- Can improve optimization stability.
- Often improves convergence for gradient-based methods.
- Helps regularized models treat feature coefficients more comparably.
- Makes distance calculations less dominated by large numerical ranges.
- Is an important preprocessing step for many PCA workflows.

Standardization is not guaranteed to improve every model. Its effectiveness depends on the algorithm and dataset.

---

# 🧮 Mathematical Formula

For a feature (x), the standardized value is:

[\
z = \frac{x-\mu}{\sigma}\
]

Where:

- (x) = original observation.
- (\mu) = mean of the feature.
- (\sigma) = standard deviation of the feature.
- (z) = standardized value, or z-score.

For a dataset with multiple features, standardization is performed separately for each feature.

For feature (j):

[\
z\_{ij} = \frac{x\_{ij}-\mu_j}{\sigma_j}\
]

Here, (i) identifies an observation and (j) identifies a feature.

In machine learning, the mean and standard deviation must be learned from the **training data** and reused when transforming validation, test, and production data.

---

# ✏️ Step-by-Step Calculation

Suppose a feature contains:

```
values = [10, 20, 30, 40, 50]
```

## Step 1: Calculate the mean

[\
\mu = \frac{10+20+30+40+50}{5}\
]

[\
\mu = 30\
]

## Step 2: Calculate the population standard deviation

For these five values:

[\
\sigma =\
\sqrt{\
\frac{\sum\_{i=1}^{n}(x_i-\mu)^2}{n}\
}\
]

The squared deviations are:

```
(10 - 30)² = 400
(20 - 30)² = 100
(30 - 30)² = 0
(40 - 30)² = 100
(50 - 30)² = 400
```

Therefore:

[\
\sigma = \sqrt{\frac{1000}{5}}\
]

[\
\sigma = \sqrt{200} \approx 14.1421\
]

## Step 3: Standardize each value

For the value `10`:

[\
z = \frac{10-30}{14.1421}\
]

[\
z \approx -1.4142\
]

For the value `30`:

[\
z = \frac{30-30}{14.1421}=0\
]

For the value `50`:

[\
z = \frac{50-30}{14.1421}\
]

[\
z \approx 1.4142\
]

The standardized values are approximately:

```
Original:      10       20       30       40       50

Standardized: -1.4142  -0.7071   0.0000   0.7071   1.4142
```

This example uses the population standard deviation, matching the default variance convention of Scikit-Learn's `StandardScaler`.

---

# 📍 Understanding Z-Scores

A z-score represents the distance of an observation from the mean, measured in standard deviations.

| Z-Score | Interpretation                         |
| ------- | -------------------------------------- |
| `0`     | Equal to the mean                      |
| `1`     | One standard deviation above the mean  |
| `-1`    | One standard deviation below the mean  |
| `2`     | Two standard deviations above the mean |
| `-2`    | Two standard deviations below the mean |

For example:

```
Original value = 80
Mean           = 60
Standard deviation = 10
```

Then:

[\
z = \frac{80-60}{10}=2\
]

The value is two standard deviations above the mean.

Explore how changing the observation, mean, and standard deviation affects the z-score.

\<LearningViz type_id="STANDARD_SCORE_Z" initial_values={{"x":80,"mu":60,"sigma":10}} />

**Note:** A z-score does not automatically indicate that an observation is an outlier. Outlier identification requires additional statistical and domain context.

---

# ⚖️ Standardization vs Normalization

These terms are sometimes used interchangeably, but they refer to different transformations in common machine learning terminology.

| Property       | Standardization                                | L2 Normalization                       |
| -------------- | ---------------------------------------------- | -------------------------------------- |
| Main objective | Center and scale each feature                  | Scale each observation vector          |
| Operation      | Subtract mean and divide by standard deviation | Divide each row by its L2 norm         |
| Mean centered  | Approximately zero per feature on fitted data  | Not necessarily                        |
| Unit variance  | Approximately one for nonconstant features     | No                                     |
| Typical tool   | `StandardScaler`                               | `Normalizer(norm="l2")`                |
| Common use     | Regression, classification, PCA                | Vector similarity, some text workflows |

Example:

```
from sklearn.preprocessing import Normalizer, StandardScaler

standardizer = StandardScaler()
normalizer = Normalizer(norm="l2")
```

Use the transformation that matches the data and the algorithm.

---

# 📏 Standardization vs Min-Max Scaling

| Property            | Standardization               | Min-Max Scaling                       |
| ------------------- | ----------------------------- | ------------------------------------- |
| Formula             | ((x-\mu)/\sigma)              | ((x-x\_{\min})/(x\_{\max}-x\_{\min})) |
| Center              | Mean around zero              | Depends on the chosen range           |
| Scale               | Standard deviation around one | Fixed training range                  |
| Output bounds       | Unbounded                     | Commonly 0 to 1 on training data      |
| Outlier sensitivity | Sensitive                     | Sensitive                             |
| Scikit-Learn tool   | `StandardScaler`              | `MinMaxScaler`                        |

Choose based on the algorithm, feature distribution, and modeling objective—not simply on which output looks better.

---

# 🛠️ StandardScaler in Scikit-Learn

`StandardScaler` is the standard Scikit-Learn transformer for feature-wise standardization.

## Installation

```
python -m pip install numpy pandas scikit-learn
```

## Basic implementation

```
from sklearn.preprocessing import StandardScaler

X = [
    [10, 1000],
    [20, 2000],
    [30, 3000],
    [40, 4000],
]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(X_scaled)
```

### Understanding the methods

- `fit(X_train)`: learns each feature's mean and scale.
- `transform(X)`: applies the learned transformation.
- `fit_transform(X_train)`: learns the statistics and transforms the training data in one step.

Inspect the learned statistics:

```
print("Means:", scaler.mean_)
print("Scales:", scaler.scale_)
```

`scale_` stores the standard deviations used for scaling, with a safe scale of `1` for constant features.

**Important:** Do not call `fit_transform()` independently on the test dataset.

---

# 🐍 Manual Standardization with Python

Implementing standardization manually is useful for understanding the mathematics.

```
"""
Manual Standardization
----------------------
Demonstrates z-score standardization using
the population standard deviation.
"""

import math


def standardize(values):
    """Return standardized values, mean, and population std."""
    if not values:
        raise ValueError("values must not be empty")

    mean = sum(values) / len(values)

    variance = sum(
        (value - mean) ** 2
        for value in values
    ) / len(values)

    std = math.sqrt(variance)

    if std == 0:
        # A constant feature has no variation.
        return [0.0 for _ in values], mean, std

    standardized = [
        (value - mean) / std
        for value in values
    ]

    return standardized, mean, std


def main():
    values = [10, 20, 30, 40, 50]

    result, mean, std = standardize(values)

    print("Original:", values)
    print(f"Mean: {mean:.4f}")
    print(f"Population std: {std:.4f}")
    print("Standardized:", [round(x, 4) for x in result])


if __name__ == "__main__":
    main()
```

This educational function handles a constant feature without dividing by zero. Scikit-Learn also handles constant features safely.

---

# 🔢 Standardization with NumPy

NumPy provides convenient numerical operations.

```
import numpy as np

values = np.array([10, 20, 30, 40, 50], dtype=float)

mean = np.mean(values)
std = np.std(values)

if std == 0:
    standardized = np.zeros_like(values)
else:
    standardized = (values - mean) / std

print("Mean:", mean)
print("Standard deviation:", std)
print("Standardized:", standardized)
```

`np.std()` uses `ddof=0` by default, which calculates the population standard deviation.

For a sample standard deviation, NumPy supports `ddof=1`. This is different from the default convention used by `StandardScaler`.

---

# 🐼 Standardization with Pandas

Pandas can standardize numerical columns using column-wise operations.

```
import pandas as pd

df = pd.DataFrame({
    "age": [20, 25, 30, 35, 40],
    "salary": [20000, 30000, 40000, 50000, 60000],
})

means = df.mean()
stds = df.std(ddof=0)

scaled_df = (df - means) / stds

print(scaled_df)
```

Using `ddof=0` makes the calculation consistent with the population-standard-deviation convention used by `StandardScaler`.

This is suitable for learning. For an ML workflow, learn the statistics from training data only and reuse them for other datasets.

---

# 📊 Standardizing Multiple Features

Suppose a dataset contains:

```
age
salary
experience
hours_studied
```

Standardize all numerical features:

```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Each feature receives its own mean and scale.

For DataFrame inputs, the transformed output is usually a NumPy array. If you need DataFrame column labels, preserve them explicitly:

```
import pandas as pd

X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns,
    index=X_train.index,
)

X_test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns,
    index=X_test.index,
)
```

---

# ✂️ Train-Test Split and Standardization

The correct sequence is:

1. Separate the features and target.
2. Split the dataset.
3. Fit the scaler on the training features.
4. Transform the training features.
5. Transform the test features with the same scaler.
6. Train the model and evaluate it.

Example:

```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The test data must not contribute to the mean or standard deviation used for training.

---

# 🚨 Preventing Data Leakage

Data leakage happens when information from outside the training process improperly influences model fitting.

## Incorrect workflow

```
Entire Dataset
      ↓
Fit StandardScaler
      ↓
Transform Entire Dataset
      ↓
Train-Test Split
```

The scaler has already learned statistics from the test data.

## Correct workflow

```
Entire Dataset
      ↓
Train-Test Split
      │
      ├── Training Features
      │       ↓
      │   Fit StandardScaler
      │       ↓
      │   Transform Training
      │
      └── Test Features
              ↓
          Transform with
          the fitted scaler
```

**Rule:** Fit preprocessing transformations on training data only. Reuse the fitted transformation for validation, testing, and future inference.

---

# 🔗 Standardization with Pipelines

A Scikit-Learn pipeline combines standardization and modeling.

```
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
```

Benefits:

- Reduces the risk of preprocessing leakage.
- Applies consistent transformations.
- Simplifies cross-validation.
- Supports reproducible model development.
- Makes deployment easier because the fitted preprocessing is part of the model workflow.

When a pipeline is fitted on `X_train`, the scaler learns statistics from the training features before the model is fitted.

---

# 🔁 Standardization with Cross-Validation

For cross-validation, use a pipeline so that each fold fits its own scaler using only that fold's training portion.

```
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])

scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
)

print("Fold scores:", scores)
print(f"Mean accuracy: {scores.mean():.4f}")
```

Keep the final test set separate from cross-validation and model selection.

For classification tasks with imbalanced classes, consider stratified cross-validation where appropriate.

---

# 📉 Standardization and Outliers

StandardScaler uses the mean and standard deviation, which can be influenced by extreme observations.

Example:

```
10
11
12
13
1000
```

The extreme value `1000` can shift the mean and increase the standard deviation.

As a result, the other values may become clustered near one another after standardization.

### What should you do?

1. Inspect the observation.
2. Check for data-entry or measurement errors.
3. Determine whether the value is legitimate.
4. Consider domain-specific treatment if justified.
5. Compare StandardScaler with RobustScaler when appropriate.

Do not automatically remove outliers simply because they affect scaling.

---

# 🧩 Standardization and Missing Values

StandardScaler is not a general-purpose missing-value imputer.

A robust workflow usually handles missing numerical values before scaling.

For example:

```
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
```

Fit this pipeline on training data only.

The median is learned from the training data and then reused for subsequent transformations.

For datasets containing both numerical and categorical columns, use a `ColumnTransformer` with separate preprocessing pipelines.

---

# 🔤 Standardization and Categorical Features

Standardization is designed for numerical measurements.

Do not blindly standardize arbitrary category labels such as:

```
Red = 1
Blue = 2
Green = 3
```

Those numbers may not represent meaningful distances.

Instead, encode categorical features appropriately. For mixed datasets, use:

- `StandardScaler` for numerical columns.
- `OneHotEncoder` for nominal categorical columns.
- `OrdinalEncoder` for genuinely ordered categories, when suitable.

Example:

```
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

numeric_features = ["age", "salary"]
categorical_features = ["city"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])
```

This creates a leakage-safe preprocessing structure when fitted within the training workflow.

---

# 🤖 Standardization for Different Algorithms

| Algorithm                                | Is Standardization Commonly Useful? | Reason                                                             |
| ---------------------------------------- | ----------------------------------- | ------------------------------------------------------------------ |
| K-Nearest Neighbors                      | Yes                                 | Distance calculations                                              |
| K-Means                                  | Yes                                 | Distance to cluster centers                                        |
| Support Vector Machines                  | Yes                                 | Optimization and distance geometry                                 |
| Logistic Regression                      | Usually                             | Optimization and regularization                                    |
| Ridge Regression                         | Usually                             | Comparable feature penalty                                         |
| Lasso Regression                         | Usually                             | Comparable feature penalty                                         |
| Neural Networks                          | Often                               | Numerical optimization                                             |
| PCA                                      | Often                               | Prevents large-scale features from dominating                      |
| Linear Regression without regularization | Depends                             | Scaling affects coefficient units, but not necessarily predictions |
| Decision Tree                            | Usually unnecessary                 | Threshold-based splits                                             |
| Random Forest                            | Usually unnecessary                 | Tree-based learning                                                |
| Gradient-Boosted Trees                   | Usually unnecessary                 | Tree-based learning                                                |

These are general guidelines rather than absolute rules. Model implementation and dataset characteristics can affect the best choice.

---

# 🧪 Complete Classification Example

This example standardizes features and trains Logistic Regression on the Breast Cancer dataset.

```
"""
Standardization - Classification
---------------------------------
Workflow:
1. Load a dataset
2. Split into training and testing data
3. Fit StandardScaler inside a pipeline
4. Train Logistic Regression
5. Evaluate on unseen test data
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    data = load_breast_cancer()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000)),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()
```

---

# 📈 Complete Regression Example

This example standardizes numerical features before fitting Ridge Regression.

```
"""
Standardization - Regression
----------------------------
Demonstrates a preprocessing pipeline
with Ridge Regression.
"""

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    data = load_diabetes()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")


if __name__ == "__main__":
    main()
```

---

# 📊 Visualizing Standardization

The following example compares a feature before and after standardization.

```
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

values = np.array([10, 20, 30, 40, 50], dtype=float).reshape(-1, 1)

scaler = StandardScaler()
scaled_values = scaler.fit_transform(values).ravel()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(values.ravel(), bins=5, edgecolor="black")
axes[0].set_title("Before Standardization")
axes[0].set_xlabel("Original Value")
axes[0].set_ylabel("Frequency")

axes[1].hist(scaled_values, bins=5, edgecolor="black")
axes[1].set_title("After Standardization")
axes[1].set_xlabel("Standardized Value")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
```

Standardization changes the center and scale of the feature. It does **not** necessarily make a skewed distribution normal.

---

# ❌ Common Mistakes

## 1. Fitting the scaler before splitting

Incorrect:

```
X_scaled = StandardScaler().fit_transform(X)
```

followed by a train-test split.

**Why it is wrong:** Test-set statistics influence the transformation.

## 2. Fitting separate scalers for training and testing

Incorrect:

```
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)
```

**Why it is wrong:** Training and test data are transformed using different learned statistics.

Correct:

```
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## 3. Assuming standardized values must lie between -1 and 1

Standardization has no fixed output bounds. Values may be less than `-1` or greater than `1`.

## 4. Assuming standardization removes outliers

Standardization changes units; it does not eliminate extreme observations.

## 5. Assuming every algorithm requires standardization

Tree-based models generally do not need it.

## 6. Standardizing the target without a reason

Feature standardization normally applies to `X`. Scaling the target `y` is a separate modeling decision and requires consistent inverse transformation when predictions must return to the original units.

## 7. Ignoring constant features

A constant feature has zero variance. StandardScaler handles this safely by using a scale of `1`, but the feature itself contains no variation and may not be useful to the model.

---

# ✅ Best Practices

- Inspect numerical features before selecting a transformation.
- Split data before fitting preprocessing.
- Fit StandardScaler only on training data.
- Reuse the fitted scaler for validation, test, and production data.
- Prefer pipelines for repeatable workflows.
- Use pipelines during cross-validation.
- Investigate outliers before choosing a scaling strategy.
- Handle missing values using an appropriate preprocessing step.
- Encode categorical features with suitable encoders.
- Compare model performance using a valid validation strategy.
- Keep the final test set untouched during model selection.
- Save the fitted preprocessing and model together when deploying.

---

# 🛠️ Mini Projects

## Project 1 — Manual Standardization

Create a numerical list and calculate:

- mean
- population standard deviation
- standardized values
- mean of the transformed values
- standard deviation of the transformed values

Compare your results with `StandardScaler`.

## Project 2 — Standardization vs Min-Max Scaling

Use a dataset containing age, income, and experience.

Compare the output of:

- `StandardScaler`
- `MinMaxScaler`

Explain the difference in their output ranges.

## Project 3 — Logistic Regression

Train Logistic Regression with and without standardization.

Compare accuracy, convergence warnings, and other appropriate metrics. Use the same split for a fair comparison.

## Project 4 — Outlier Experiment

Create a dataset with an extreme value.

Compare the effect on:

- mean
- standard deviation
- standardized values
- robust-scaled values

## Project 5 — Leakage-Safe Pipeline

Build a pipeline with:

```
Training Data
      ↓
Missing-Value Imputation
      ↓
Standardization
      ↓
Classification Model
      ↓
Evaluation
```

Verify that the pipeline is fitted only on training data.

---

# 📝 Exercises

## Beginner

1. What is standardization?
2. What is a z-score?
3. What is the formula for standardization?
4. What do a mean of zero and standard deviation of one mean?
5. What is `StandardScaler`?
6. What does `fit_transform()` do?
7. What is the difference between standardization and Min-Max scaling?

## Intermediate

8. Why is standardization useful for KNN?
9. Why can standardization help Logistic Regression?
10. Why is standardization generally unnecessary for decision trees?
11. How do outliers affect the mean and standard deviation?
12. Why should the scaler be fitted only on training data?
13. What is the difference between population and sample standard deviation?
14. Why does standardization not guarantee values between `-1` and `1`?

## Advanced

15. Why can standardization affect regularized models?
16. How can standardization influence PCA?
17. Why should scaling be inside a cross-validation pipeline?
18. How would you standardize numerical features while encoding categorical features?
19. How should a fitted scaler be reused for production inference?
20. When might standardization be less appropriate than robust scaling?

---

# 📋 Standardization Checklist

- Identify numerical features.
- Inspect feature ranges and distributions.
- Investigate outliers and missing values.
- Decide whether the chosen algorithm benefits from scaling.
- Split the dataset before fitting the scaler.
- Fit `StandardScaler` on training features only.
- Transform training data.
- Transform validation and test data using the same fitted scaler.
- Use a pipeline where possible.
- Check the transformed feature distributions.
- Evaluate on unseen data.
- Keep preprocessing consistent during deployment.

---

# 📁 Project Structure

```
07-Data-Preprocessing/
│
├── README.md
│
├── 01-Train-Test-Split/
│   └── README.md
│
├── 02-Feature-Scaling/
│   └── README.md
│
├── 03-Standardization/
│   ├── README.md
│   ├── manual_standardization.py
│   ├── numpy_standardization.py
│   ├── pandas_standardization.py
│   ├── standard_scaler.py
│   ├── classification_example.py
│   ├── regression_example.py
│   ├── visualization.py
│   └── pipeline_example.py
│
└── ...
```

---

# 🗺️ Learning Roadmap

```
🌱 Data Preprocessing
        │
        ▼
✂️ Train-Test Split
        │
        ▼
📏 Feature Scaling
        │
        ▼
📊 Standardization
        │
        ▼
🔢 Min-Max Scaling
        │
        ▼
🛡️ Robust Scaling
        │
        ▼
🔤 Categorical Encoding
        │
        ▼
🔄 Feature Transformation
        │
        ▼
⚙️ Preprocessing Pipelines
        │
        ▼
🤖 Model Training
```

---

# 💡 Key Takeaways

> **1. Standardization transforms features using their mean and standard deviation.**

> **2. The standardization formula is (z=(x-\mu)/\sigma).**

> **3. Standardized features have approximately zero mean and unit variance when measured using the fitted training statistics and the same variance convention.**

> **4. StandardScaler performs the transformation independently for each feature.**

> **5. Standardization is commonly useful for distance-based, gradient-based, and regularized algorithms.**

> **6. Standardization does not guarantee a normal distribution or a fixed numerical range.**

> **7. Outliers can strongly influence the mean and standard deviation.**

> **8. Fit the scaler on training data only to prevent data leakage.**

> **9. Pipelines make standardization safer and more reproducible.**

> **10. Always choose preprocessing based on the data and the algorithm.**

---

# 🚀 Next Step

Continue with the next topic in your Data Preprocessing module:

## 🔢 Min-Max Scaling

Learn how to transform numerical features into a specified range using:

- Minimum and maximum values
- The Min-Max scaling formula
- `MinMaxScaler`
- Custom output ranges
- Outlier sensitivity
- Train-test leakage prevention
- Scikit-Learn pipelines
- Practical classification and regression examples

```
✂️ Split
   ↓
📏 Feature Scaling
   ↓
📊 Standardization
   ↓
🔢 Min-Max Scaling
   ↓
🛡️ Robust Scaling
   ↓
🔤 Categorical Encoding
   ↓
⚙️ Preprocessing Pipeline
   ↓
🤖 Machine Learning
```

**Standardization is not about changing the meaning of your data. It is about representing numerical features on a more comparable scale so that suitable machine learning algorithms can use them effectively.**

---

## 👨‍💻 Author

**Kishor Patil**

Machine Learning · Python · Data Science · Artificial Intelligence

GitHub: [Kishor055](https://github.com/Kishor055)

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request to improve the examples, explanations, and exercises.

## ⭐ Support

If this repository helps you learn Machine Learning, consider starring it, sharing it with other learners, and contributing improvements.

## 📄 License

This project is intended for educational purposes. Refer to the repository's root license file for the applicable license terms.
