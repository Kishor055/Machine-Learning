Next module: **Feature Scaling**. This README covers the theory, formulas, when to use each scaler, leakage prevention, Scikit-Learn implementations, practical examples, comparisons, exercises, and an end-to-end workflow.

# 📏 Feature Scaling

> **Feature Scaling = Transform Numerical Features → Comparable Scale → Stable Optimization → Better Model Performance**

Feature scaling is a preprocessing technique used to transform numerical features so that they have a comparable range or distribution.

It is especially important for algorithms that depend on:

* distances
* gradients
* magnitudes
* variance
* regularization

Examples include:

* K-Nearest Neighbors
* K-Means
* Support Vector Machines
* Logistic Regression
* Linear Regression with regularization
* Neural Networks
* Principal Component Analysis
* Gradient-based algorithms

---

# 🌱 GROW → INSPECT → SPLIT → SCALE → TRAIN → EVALUATE 🚀

```text
                    📊 Raw Dataset
                          │
                          ▼
                  🔍 Inspect Features
                          │
                          ▼
                  ✂️ Train-Test Split
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        🧠 Training Data          🧪 Test Data
              │                       │
              ▼                       │
       Fit Scaler Only               │
       on Training                   │
              │                       │
              ├──── transform ───────┤
              ▼                       ▼
        Scaled Training        Scaled Testing
              │                       │
              └──────────┬────────────┘
                         ▼
                   🤖 Train Model
                         │
                         ▼
                   📈 Evaluate
```

---

# 📚 Table of Contents

1. [What is Feature Scaling?](#-what-is-feature-scaling)
2. [Why Feature Scaling Matters](#-why-feature-scaling-matters)
3. [When Scaling is Important](#-when-scaling-is-important)
4. [When Scaling is Usually Not Required](#-when-scaling-is-usually-not-required)
5. [The Feature Scale Problem](#-the-feature-scale-problem)
6. [Common Scaling Techniques](#-common-scaling-techniques)
7. [Standardization](#-standardization)
8. [StandardScaler](#-standardscaler)
9. [Min-Max Scaling](#-min-max-scaling)
10. [MinMaxScaler](#-minmaxscaler)
11. [Robust Scaling](#-robust-scaling)
12. [RobustScaler](#-robustscaler)
13. [MaxAbs Scaling](#-maxabs-scaling)
14. [Normalization](#-normalization)
15. [L1 Normalization](#-l1-normalization)
16. [L2 Normalization](#-l2-normalization)
17. [Mean Normalization](#-mean-normalization)
18. [Log Transformation](#-log-transformation)
19. [Power Transformations](#-power-transformations)
20. [Quantile Transformation](#-quantile-transformation)
21. [Standardization vs Normalization](#-standardization-vs-normalization)
22. [Choosing the Right Scaler](#-choosing-the-right-scaler)
23. [Data Leakage](#-data-leakage)
24. [Correct Scaling Workflow](#-correct-scaling-workflow)
25. [Feature Scaling with Pandas](#-feature-scaling-with-pandas)
26. [Feature Scaling with Scikit-Learn](#-feature-scaling-with-scikit-learn)
27. [Scaling Train and Test Data](#-scaling-train-and-test-data)
28. [Scaling Multiple Features](#-scaling-multiple-features)
29. [Scaling with Pipelines](#-scaling-with-pipelines)
30. [Scaling and Cross-Validation](#-scaling-and-cross-validation)
31. [Scaling and Outliers](#-scaling-and-outliers)
32. [Scaling and Categorical Features](#-scaling-and-categorical-features)
33. [Scaling Sparse Data](#-scaling-sparse-data)
34. [Scaling and Machine Learning Algorithms](#-scaling-and-machine-learning-algorithms)
35. [Common Mistakes](#-common-mistakes)
36. [Best Practices](#-best-practices)
37. [Complete Classification Example](#-complete-classification-example)
38. [Complete Regression Example](#-complete-regression-example)
39. [Scaler Comparison](#-scaler-comparison)
40. [Mini Projects](#-mini-projects)
41. [Exercises](#-exercises)
42. [Checklist](#-checklist)
43. [Project Structure](#-project-structure)
44. [Key Takeaways](#-key-takeaways)
45. [Next Step](#-next-step)

---

# 🎯 What is Feature Scaling?

Suppose a dataset contains:

```text
Age        → 18 - 80
Income     → 20,000 - 500,000
Experience → 0 - 40
```

The numerical features exist on very different scales.

A distance-based algorithm may give excessive influence to `Income` simply because its numerical values are much larger.

Feature scaling transforms the values into a more comparable scale.

For example:

```text
Before:

Age         → 18 - 80
Income      → 20,000 - 500,000
Experience  → 0 - 40

After Standardization:

Age         → approximately -2 to +2
Income      → approximately -2 to +2
Experience  → approximately -2 to +2
```

The actual ranges depend on the data.

---

# 🤔 Why Feature Scaling Matters

Consider two features:

```text
Age       = 25
Salary    = 800000
```

Calculate Euclidean distance between two observations:

```text
Person A:
Age = 25
Salary = 800000

Person B:
Age = 30
Salary = 810000
```

The differences are:

```text
Age difference     = 5
Salary difference  = 10,000
```

The salary difference dominates the distance.

After scaling:

```text
Age difference     → comparable
Salary difference  → comparable
```

The algorithm can then consider both features more fairly.

---

# ⚙️ When Scaling is Important

Feature scaling is commonly important for algorithms based on distances or optimization.

| Algorithm               | Scaling               |
| ----------------------- | --------------------- |
| KNN                     | ✅ Important           |
| K-Means                 | ✅ Important           |
| SVM                     | ✅ Important           |
| Logistic Regression     | ✅ Usually recommended |
| Linear Regression       | ⚠️ Depends            |
| Ridge                   | ✅ Important           |
| Lasso                   | ✅ Important           |
| Elastic Net             | ✅ Important           |
| Neural Networks         | ✅ Very important      |
| PCA                     | ✅ Important           |
| Hierarchical Clustering | ✅ Important           |
| Decision Tree           | ❌ Usually unnecessary |
| Random Forest           | ❌ Usually unnecessary |
| Gradient Boosting Trees | ❌ Usually unnecessary |

### Important principle

> **Scaling is generally more important for distance-based and gradient-based algorithms than for tree-based algorithms.**

---

# 🚫 When Scaling is Usually Not Required

Tree-based models split features using thresholds.

For example:

```text
Income < 50000
```

Multiplying every income value by 1,000 does not fundamentally change the ordering.

Therefore, algorithms such as:

* Decision Trees
* Random Forest
* Gradient Boosted Trees

usually do not require feature scaling.

However, preprocessing decisions should still be based on the specific model and pipeline.

---

# 📐 The Feature Scale Problem

Imagine:

```text
Feature A:
0 → 10

Feature B:
0 → 1,000,000
```

For a distance-based algorithm:

```text
Distance ≈ sqrt(
    difference_A² +
    difference_B²
)
```

Because `difference_B` can be extremely large, Feature B can dominate the calculation.

Scaling reduces this problem.

---

# 🧮 Common Scaling Techniques

The most useful techniques include:

```text
Standardization
      │
      ├── StandardScaler
      │
      ▼
Min-Max Scaling
      │
      ├── MinMaxScaler
      │
      ▼
Robust Scaling
      │
      ├── RobustScaler
      │
      ▼
Max Absolute Scaling
      │
      ├── MaxAbsScaler
      │
      ▼
Normalization
      │
      ├── L1
      └── L2
```

For heavily skewed data:

```text
Log Transformation
Power Transformation
Quantile Transformation
```

---

# 📊 Standardization

Standardization transforms a feature so that it has:

```text
Mean ≈ 0
Standard Deviation ≈ 1
```

The formula is:

$$
z = \frac{x-\mu}{\sigma}
$$

Where:

* \(x\) = original value
* \(\mu\) = training mean
* \(\sigma\) = training standard deviation
* \(z\) = standardized value

### Example

Suppose:

```text
Mean = 50
Standard deviation = 10
Value = 70
```

Then:

$$
z = \frac{70-50}{10}=2
$$

The standardized value is:

```text
2
```

---

# 🧰 StandardScaler

Scikit-Learn implementation:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Notice the difference:

```text
Training:
fit_transform()

Testing:
transform()
```

Never fit the scaler independently on the test data.

---

# 📉 Min-Max Scaling

Min-Max scaling transforms values into a specified range.

The common default range is:

```text
0 → 1
```

Formula:

$$
x' = \frac{x-x_{min}}{x_{max}-x_{min}}
$$

For example:

```text
Minimum = 10
Maximum = 50
Value = 30
```

Then:

$$
x' = \frac{30-10}{50-10}
$$

$$
x' = 0.5
$$

So:

```text
30 → 0.5
```

---

# 🧰 MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

You can specify another range:

```python
scaler = MinMaxScaler(feature_range=(-1, 1))
```

---

# 🛡️ Robust Scaling

Standardization can be strongly affected by outliers.

Robust scaling uses:

* median
* interquartile range (IQR)

The basic transformation is:

$$
x' = \frac{x-\text{median}}{IQR}
$$

where:

$$
IQR = Q_3-Q_1
$$

This makes Robust Scaling useful when features contain substantial outliers.

---

# 🧰 RobustScaler

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

# 📦 MaxAbs Scaling

MaxAbs scaling divides each feature by its maximum absolute value.

Conceptually:

$$
x' = \frac{x}{\max(|x|)}
$$

It typically maps values into:

```text
[-1, 1]
```

Example:

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

This can be useful when preserving sparsity matters.

---

# 🧍 Normalization

Scaling and normalization are related but different concepts.

**Feature scaling** usually transforms each feature across observations.

**Normalization** often transforms each individual observation so that its vector has a specified norm.

Example:

```text
Observation:

[3, 4]
```

Its L2 norm is:

$$
\sqrt{3^2+4^2}=5
$$

After L2 normalization:

```text
[3/5, 4/5]

→ [0.6, 0.8]
```

---

# 1️⃣ L1 Normalization

L1 normalization divides by the sum of absolute values.

$$
x'_i = \frac{x_i}{\sum_j |x_j|}
$$

Example:

```text
[2, 3, 5]
```

Sum:

```text
10
```

Normalized:

```text
[0.2, 0.3, 0.5]
```

Scikit-Learn:

```python
from sklearn.preprocessing import Normalizer

normalizer = Normalizer(norm="l1")

X_normalized = normalizer.fit_transform(X)
```

---

# 2️⃣ L2 Normalization

L2 normalization divides by the Euclidean norm.

$$
x'_i =
\frac{x_i}
{\sqrt{\sum_j x_j^2}}
$$

Example:

```text
[3, 4]
```

Norm:

```text
5
```

Result:

```text
[0.6, 0.8]
```

Implementation:

```python
from sklearn.preprocessing import Normalizer

normalizer = Normalizer(norm="l2")

X_normalized = normalizer.fit_transform(X)
```

---

# 📊 Mean Normalization

Mean normalization can be represented as:

$$
x' =
\frac{x-\text{mean}(x)}
{\max(x)-\min(x)}
$$

This centers the data around zero while scaling by the feature range.

It is less commonly used as a standard Scikit-Learn preprocessing choice than StandardScaler or MinMaxScaler.

---

# 📈 Log Transformation

Some numerical features are highly right-skewed.

For example:

```text
Income:
20k
25k
30k
35k
40k
500k
```

A log transformation can reduce the influence of extreme magnitudes.

For non-negative data:

```python
import numpy as np

X_log = np.log1p(X)
```

`log1p(x)` calculates:

$$
\log(1+x)
$$

and safely handles zero.

### Important

A log transformation is **not simply another scaler**.

It changes the shape of the feature distribution.

Use it when it makes sense for the feature and problem.

---

# 🔄 Power Transformations

Power transformations can make data more Gaussian-like.

Scikit-Learn provides:

```python
from sklearn.preprocessing import PowerTransformer
```

Example:

```python
transformer = PowerTransformer(method="yeo-johnson")

X_transformed = transformer.fit_transform(X_train)
```

Common methods include:

* Yeo-Johnson
* Box-Cox

### Box-Cox

Box-Cox generally requires strictly positive values.

### Yeo-Johnson

Yeo-Johnson can handle zero and negative values.

---

# 📊 Quantile Transformation

`QuantileTransformer` transforms features based on their quantiles.

Example:

```python
from sklearn.preprocessing import QuantileTransformer

transformer = QuantileTransformer(
    output_distribution="normal",
    random_state=42
)

X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)
```

Possible output distributions include:

```text
uniform
normal
```

Use carefully because this transformation can significantly alter the original feature distribution and interpretation.

---

# ⚖️ Standardization vs Normalization

These terms are often confused.

| Property        | Standardization           | Normalization                 |
| --------------- | ------------------------- | ----------------------------- |
| Main idea       | Center and scale features | Scale individual observations |
| Typical tool    | `StandardScaler`          | `Normalizer`                  |
| Mean-centered   | Usually yes               | Not necessarily               |
| Unit variance   | Approximately yes         | No                            |
| Operates across | Samples for each feature  | Features within each sample   |
| Common use      | General ML preprocessing  | Vector/distance applications  |

### Remember

```text
StandardScaler
→ feature-wise transformation

Normalizer
→ sample-wise transformation
```

---

# 🧭 Choosing the Right Scaler

A practical decision guide:

```text
                    Start
                      │
                      ▼
              Are features numerical?
                      │
                     Yes
                      │
                      ▼
              Are there major outliers?
                 /              \
               Yes              No
                │                │
                ▼                ▼
          RobustScaler     Is bounded range
                           desirable?
                           /          \
                         Yes           No
                          │             │
                          ▼             ▼
                    MinMaxScaler   StandardScaler
```

For sparse data:

```text
Consider MaxAbsScaler
```

For strongly skewed features:

```text
Consider transformation
before or alongside scaling
```

---

# 🚨 Data Leakage

This is one of the most important concepts in feature scaling.

## ❌ Incorrect

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)
```

Why is this problematic?

The scaler calculated:

```text
mean
standard deviation
```

using the complete dataset, including the test set.

Therefore, information from the test set influenced the training transformation.

---

# ✅ Correct Workflow

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The rule is:

> **Fit on training data. Transform training and test data using the same fitted transformer.**

---

# 🔄 Correct Scaling Workflow

```text
Raw Data
   │
   ▼
Separate X and y
   │
   ▼
Train-Test Split
   │
   ├──────────────────┐
   ▼                  ▼
X_train             X_test
   │                  │
   ▼                  │
fit scaler            │
   │                  │
   ▼                  │
transform             │
   │                  │
   │             transform
   │                  │
   └────────┬─────────┘
            ▼
       Scaled Dataset
            │
            ▼
        Train Model
            │
            ▼
      Evaluate on Test
```

---

# 🐼 Feature Scaling with Pandas

For educational purposes, scaling can be implemented manually.

## Standardization

```python
import pandas as pd

df = pd.DataFrame({
    "age": [20, 25, 30, 35, 40],
    "salary": [20000, 30000, 40000, 50000, 60000]
})

numeric_columns = ["age", "salary"]

train_mean = df[numeric_columns].mean()
train_std = df[numeric_columns].std()

scaled = (
    df[numeric_columns] - train_mean
) / train_std

print(scaled)
```

In real ML workflows, use a fitted preprocessing object or pipeline so that the training statistics are consistently reused.

---

# 🤖 Feature Scaling with Scikit-Learn

Scikit-Learn is generally preferred for ML preprocessing.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler stores learned statistics:

```python
print(scaler.mean_)
print(scaler.scale_)
```

This allows the same transformation to be applied to new data.

---

# 🧪 Scaling Train and Test Data

Example:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Notice:

```text
Training → fit + transform
Testing  → transform only
```

---

# 📚 Scaling Multiple Features

Suppose:

```text
X =
[
    [age, income, experience],
    [age, income, experience],
    ...
]
```

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Each feature is scaled independently.

---

# 🧩 Scaling with Pipelines

Pipelines are the safest and cleanest way to combine preprocessing and modeling.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
```

Benefits:

* prevents preprocessing leakage
* keeps transformations consistent
* simplifies cross-validation
* makes deployment easier
* improves reproducibility

---

# 🔁 Scaling and Cross-Validation

Do not manually scale the entire dataset before cross-validation.

❌ Risky:

```python
X_scaled = StandardScaler().fit_transform(X)

cross_val_score(
    model,
    X_scaled,
    y,
    cv=5
)
```

The scaler has seen all observations before the folds are created.

### Better

Use a pipeline:

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5
)

print(scores)
print(scores.mean())
```

Each cross-validation training fold gets its own fitted scaler.

---

# 🚨 Scaling and Outliers

Consider:

```text
10
11
12
13
1000
```

The value `1000` is an extreme observation.

Standardization uses mean and standard deviation, both of which can be affected by extreme values.

Therefore:

```text
Outliers present
      ↓
Consider RobustScaler
```

But do not automatically remove outliers just because a scaler is sensitive to them.

First investigate whether they are:

* data-entry errors
* legitimate observations
* rare but important cases
* measurement problems
* fraud/anomalies

---

# 🔤 Scaling and Categorical Features

Feature scaling is generally applied to **numerical features**.

Categorical features should usually be encoded first.

Example:

```text
Gender
Male
Female
Female
Male
```

One-hot encoding:

```text
Gender_Male
Gender_Female
```

A preprocessing pipeline can handle different column types:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = [
    "age",
    "income"
]

categorical_features = [
    "city"
]

preprocessor = ColumnTransformer([
    (
        "numeric",
        StandardScaler(),
        numeric_features
    ),
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
])
```

This avoids treating category labels as continuous numerical measurements.

---

# 🧮 Scaling Sparse Data

Some datasets contain sparse matrices, especially after one-hot encoding or text vectorization.

For sparse data, be careful with transformations that center the data.

For example, standard centering can destroy sparsity.

`MaxAbsScaler` is often useful when preserving sparsity matters:

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()

X_scaled = scaler.fit_transform(X)
```

The correct preprocessing depends on the representation and model.

---

# 🤖 Scaling and Machine Learning Algorithms

## K-Nearest Neighbors

KNN uses distances.

```text
Scaling → ⭐⭐⭐⭐⭐
```

Without scaling, large-magnitude features can dominate.

---

## K-Means

K-Means uses distances to cluster centers.

```text
Scaling → ⭐⭐⭐⭐⭐
```

Scaling is generally important.

---

## Support Vector Machines

SVM optimization and distance relationships can be affected by feature magnitude.

```text
Scaling → ⭐⭐⭐⭐⭐
```

---

## Logistic Regression

Scaling is generally recommended, especially when:

* regularization is used
* features have very different scales
* optimization speed matters

```text
Scaling → ⭐⭐⭐⭐
```

---

## Neural Networks

Neural networks typically benefit strongly from appropriately scaled numerical inputs.

```text
Scaling → ⭐⭐⭐⭐⭐
```

---

## PCA

PCA depends on variance.

A large-scale feature can dominate the principal components.

```text
Scaling → ⭐⭐⭐⭐⭐
```

---

## Decision Trees

Tree splits depend primarily on ordering and thresholds.

```text
Scaling → ⭐
```

Usually unnecessary.

---

## Random Forest

Random Forest is tree-based.

```text
Scaling → ⭐
```

Usually unnecessary.

---

# ❌ Common Mistakes

## 1. Fitting the scaler on all data

```python
scaler.fit(X)
```

before splitting can cause leakage.

---

## 2. Fitting a separate scaler on the test set

❌ Incorrect:

```python
train_scaler = StandardScaler()
test_scaler = StandardScaler()

X_train = train_scaler.fit_transform(X_train)
X_test = test_scaler.fit_transform(X_test)
```

The model expects the test data to be transformed using the same rules learned from training data.

### Correct

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 3. Scaling the target unnecessarily

Feature scaling normally refers to input features.

Do not automatically scale `y`.

For some regression workflows, target transformation may be useful, but that is a separate modeling decision.

---

## 4. Scaling categorical labels incorrectly

Do not turn:

```text
Red = 1
Blue = 2
Green = 3
```

and then blindly standardize them.

This can introduce artificial numerical relationships.

---

## 5. Using Min-Max Scaling without considering future values

Min-Max scaling depends on observed minimum and maximum values.

New production observations can fall outside the training range.

For example:

```text
Training:
0 → 100

New value:
120
```

The transformed value can be greater than `1`.

That is not necessarily an error.

---

## 6. Assuming StandardScaler always creates values between -1 and 1

It does not.

Standardization produces approximately:

```text
mean = 0
standard deviation = 1
```

but values can be much larger than `1` or `-1`.

---

## 7. Scaling tree-based models without a reason

Scaling tree-based models is usually unnecessary.

Keep preprocessing as simple as the model requires.

---

# ✅ Best Practices

### 1. Split before fitting the scaler

```text
Split → Fit → Transform
```

### 2. Fit only on training data

Never use test data to calculate scaling statistics.

### 3. Transform test data using the fitted scaler

```python
scaler.transform(X_test)
```

### 4. Use pipelines

Pipelines reduce leakage and make workflows reproducible.

### 5. Choose the scaler based on the data

Consider:

* outliers
* bounds
* sparsity
* skewness
* model type

### 6. Inspect distributions before scaling

Scaling does not automatically fix:

* skewness
* outliers
* missing values
* incorrect data

### 7. Keep preprocessing consistent

The same fitted preprocessing must be used for:

```text
Training
Validation
Testing
Production inference
```

---

# 🧪 Complete Classification Example

```python
"""
Feature Scaling - Classification
---------------------------------
Compare a Logistic Regression model
with and without StandardScaler.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # Load dataset
    data = load_breast_cancer()

    X = data.data
    y = data.target

    # Split before preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Build pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(max_iter=5000)
        )
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, predictions)

    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
```

---

# 📉 Complete Regression Example

```python
"""
Feature Scaling - Regression
----------------------------
Demonstrates scaling numerical features
before training a Ridge regression model.
"""

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    data = load_diabetes()

    X = data.data
    y = data.target

    # Split first
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Scaling + Ridge regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")


if __name__ == "__main__":
    main()
```

---

# 🔬 Scaler Comparison

You can compare several preprocessing strategies.

```python
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler
)

scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
    "MaxAbsScaler": MaxAbsScaler()
}

for name, scaler in scalers.items():
    X_scaled = scaler.fit_transform(X_train)

    print(
        f"{name}: "
        f"shape={X_scaled.shape}"
    )
```

The best scaler should not be selected merely because its output "looks nicest."

Evaluate it according to:

* model performance
* validation strategy
* data characteristics
* interpretability
* production requirements

---

# 📊 Quick Comparison

| Technique           | Main Idea                    |       Outlier Sensitivity |   Typical Range |
| ------------------- | ---------------------------- | ------------------------: | --------------: |
| StandardScaler      | Mean = 0, std ≈ 1            |                      High |       Unbounded |
| MinMaxScaler        | Scale to range               |                      High |     Usually 0–1 |
| RobustScaler        | Median + IQR                 |                     Lower |       Unbounded |
| MaxAbsScaler        | Divide by max absolute value |                  Moderate |    Usually -1–1 |
| Normalizer          | Scale each sample vector     |                   Depends | Depends on norm |
| Log Transform       | Reduce skew                  |                  Can help |       Unbounded |
| PowerTransformer    | Adjust distribution          |                   Depends |       Unbounded |
| QuantileTransformer | Quantile mapping             | Can reduce skew influence |    Configurable |

---

# 🧠 Practical Scaler Decision Tree

```text
Start
  │
  ▼
Are features numerical?
  │
  ├── No → Encode first
  │
  └── Yes
       │
       ▼
Are there strong outliers?
       │
       ├── Yes → Consider RobustScaler
       │
       └── No
            │
            ▼
Need a fixed bounded range?
            │
            ├── Yes → MinMaxScaler
            │
            └── No → StandardScaler
```

Additional cases:

```text
Sparse data
    ↓
Consider MaxAbsScaler

Strong skew
    ↓
Consider a suitable transformation

Sample-wise vector scaling
    ↓
Consider Normalizer
```

---

# 🛠️ Mini Projects

## Project 1 — KNN Without Scaling vs Scaling

Train KNN:

```text
1. Without scaling
2. With StandardScaler
3. With MinMaxScaler
4. Compare accuracy
```

Analyze why performance changes.

---

## Project 2 — K-Means Scaling Experiment

Use a dataset containing features with very different ranges.

Compare:

```text
K-Means + raw data
K-Means + StandardScaler
K-Means + MinMaxScaler
```

Compare cluster assignments.

---

## Project 3 — Outlier-Heavy Dataset

Create a dataset with extreme values.

Compare:

```text
StandardScaler
RobustScaler
```

Inspect:

* transformed distributions
* median
* spread
* model performance

---

## Project 4 — PCA Scaling

Perform PCA:

```text
Without scaling
        vs
With StandardScaler
```

Compare the explained variance and principal components.

---

## Project 5 — Complete Preprocessing Pipeline

Build a pipeline containing:

```text
Train-Test Split
       ↓
ColumnTransformer
       ↓
Numerical Scaling
       ↓
Categorical Encoding
       ↓
Model
       ↓
Evaluation
```

---

# 📝 Exercises

## Beginner

1. What is feature scaling?
2. Why is scaling important?
3. What is standardization?
4. What is Min-Max scaling?
5. What does `StandardScaler` do?
6. What does `MinMaxScaler` do?
7. What is the difference between scaling and normalization?

## Intermediate

8. Why can KNN be affected by feature scale?
9. Why is scaling useful for K-Means?
10. Why is scaling generally unnecessary for decision trees?
11. What is RobustScaler?
12. When would you use MinMaxScaler?
13. What is the difference between `fit_transform()` and `transform()`?
14. Why should the scaler be fitted only on training data?
15. Why can outliers affect StandardScaler?

## Advanced

16. Explain feature scaling mathematically.
17. Why does standardization not guarantee values between -1 and 1?
18. Why can MinMaxScaler produce values outside its configured range on new data?
19. Why can scaling before cross-validation cause leakage?
20. How does a pipeline prevent preprocessing leakage?
21. Why can centering be problematic for sparse matrices?
22. When would RobustScaler be preferable to StandardScaler?
23. How can feature scaling affect regularized models?
24. Why can PCA produce different results before and after scaling?
25. How would you design a preprocessing strategy for mixed numerical and categorical data?

---

# 🧪 Practical Experiment

Create a dataset:

```python
import numpy as np

X = np.array([
    [20, 20000],
    [25, 30000],
    [30, 50000],
    [35, 80000],
    [40, 120000]
])
```

Compare:

```python
StandardScaler()
MinMaxScaler()
RobustScaler()
```

Print the resulting arrays.

Then compare:

```text
Mean
Standard deviation
Minimum
Maximum
```

Ask yourself:

> Which scaler is appropriate for this dataset, and why?

---

# 📋 Feature Scaling Checklist

Before training:

* [ ] Identify numerical features
* [ ] Identify categorical features
* [ ] Understand feature ranges
* [ ] Inspect outliers
* [ ] Inspect skewness
* [ ] Identify whether the model needs scaling
* [ ] Split data before fitting transformations
* [ ] Fit scaler only on training data
* [ ] Transform training data
* [ ] Transform validation/test data
* [ ] Use a pipeline when possible
* [ ] Avoid preprocessing leakage
* [ ] Check transformed features
* [ ] Evaluate model performance
* [ ] Keep preprocessing consistent in production

---

# 📁 Project Structure

```text
07-Data-Preprocessing/
│
├── README.md
│
├── 01-Train-Test-Split/
│   └── README.md
│
├── 02-Feature-Scaling/
│   ├── README.md
│   ├── standard_scaling.py
│   ├── minmax_scaling.py
│   ├── robust_scaling.py
│   ├── maxabs_scaling.py
│   ├── normalization.py
│   ├── power_transform.py
│   └── scaling_pipeline.py
│
├── 03-Encoding-Categorical-Data/
├── 04-Feature-Transformation/
├── 05-Feature-Selection/
├── 06-Feature-Engineering/
└── ...
```

---

# 🗺️ Learning Roadmap

```text
              📊 Raw Data
                  │
                  ▼
          ✂️ Train-Test Split
                  │
                  ▼
            📏 Feature Scaling
                  │
          ┌───────┴────────┐
          ▼                ▼
   StandardScaler      RobustScaler
          │                │
          └───────┬────────┘
                  ▼
         🔤 Categorical Encoding
                  │
                  ▼
          🔄 Transformation
                  │
                  ▼
           🎯 Feature Selection
                  │
                  ▼
          🛠️ Feature Engineering
                  │
                  ▼
         ⚙️ Preprocessing Pipeline
                  │
                  ▼
             🤖 ML Model
```

---

# 💡 Key Takeaways

> **1. Feature scaling puts numerical features on comparable scales.**

> **2. Scaling is especially important for distance-based, gradient-based, and regularized algorithms.**

> **3. StandardScaler centers features around zero and scales by their standard deviation.**

> **4. MinMaxScaler maps features to a chosen range, commonly 0–1.**

> **5. RobustScaler uses the median and IQR and can be more resistant to outliers.**

> **6. Normalization is different from feature-wise standardization.**

> **7. Always split the data before fitting a scaler.**

> **8. Use `fit_transform()` on training data and `transform()` on validation/test data.**

> **9. Pipelines help prevent preprocessing leakage and keep transformations consistent.**

> **10. Tree-based models generally do not require feature scaling.**

> **11. Scaling does not automatically solve missing values, outliers, or skewness.**

> **12. The correct scaler depends on the data, model, and deployment requirements.**

---

# 🚀 Next Step

After learning Feature Scaling, the next preprocessing concept is:

## 🔤 Encoding Categorical Data

You will learn how to transform categorical variables into numerical representations using:

* Label Encoding
* Ordinal Encoding
* One-Hot Encoding
* Frequency Encoding
* Target Encoding
* `OneHotEncoder`
* `OrdinalEncoder`
* Handling unseen categories
* High-cardinality categorical features
* Leakage-safe encoding
* Encoding inside `ColumnTransformer` and pipelines

```text
✂️ Split
   ↓
📏 Scale
   ↓
🔤 Encode
   ↓
🔄 Transform
   ↓
🎯 Select
   ↓
🛠️ Engineer
   ↓
🤖 Train
   ↓
📈 Evaluate
```

**Feature scaling is not about making every feature look the same—it is about giving the model an appropriate numerical representation of the data.**
