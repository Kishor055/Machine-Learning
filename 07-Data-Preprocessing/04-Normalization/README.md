# 📊 Normalization in Machine Learning

**Normalization = Rescale → Standardize the Range → Improve Comparability → Build Better ML Pipelines**

Normalization is a data preprocessing technique that transforms numerical features to a common scale. It can help distance-based and gradient-based machine learning algorithms work more effectively when input features have very different magnitudes.

---

## 🌱 GROW → LEARN → PREPROCESS → SCALE → BUILD → EVALUATE 🚀

> **Goal:** Understand normalization, implement it with Python and Scikit-Learn, prevent data leakage, and apply the right scaling technique to real-world machine learning projects.

## 📚 Table of Contents

1. Introduction
2. What Is Normalization?
3. Why Is Normalization Important?
4. Normalization vs Standardization
5. Types of Normalization
6. Min-Max Normalization
7. Mean Normalization
8. Maximum Absolute Scaling
9. Vector Normalization
10. Manual Implementation with Python
11. Normalization Using NumPy
12. Normalization Using Pandas
13. Normalization Using Scikit-Learn
14. Normalize Multiple Features
15. Train-Test Split and Data Leakage
16. Normalization with Pipelines
17. Normalization for Different ML Algorithms
18. Handling Outliers and Constant Features
19. Normalization in Classification
20. Normalization in Regression
21. Normalization in Deep Learning
22. Common Mistakes
23. Best Practices
24. Mini Projects
25. Practice Exercises
26. Quick Revision
27. Project Structure
28. Learning Roadmap
29. Key Takeaways
30. What's Next?

---

## 📌 Introduction

Real-world datasets often contain features with different ranges.

For example:

| Feature       | Example values          | Approximate range |
| ------------- | ----------------------- | ----------------- |
| Age           | 18, 25, 40, 60          | 18–60             |
| Annual income | 25,000; 80,000; 150,000 | 25,000–150,000    |
| Rating        | 1, 3, 5                 | 1–5               |
| Experience    | 0, 3, 10, 20            | 0–20              |

A model that relies on distances or gradient-based optimization may be affected by differences in feature scales.

Normalization transforms numerical values into a more comparable range.

### Example

Before normalization:

```
Age:             18, 25, 40, 60
Annual income:   25000, 80000, 150000, 200000
```

After Min-Max normalization to the range `[0, 1]`:

```
Age:             0.00, 0.17, 0.52, 1.00
Annual income:   0.00, 0.29, 0.71, 1.00
```

The resulting values are easier to compare numerically, although the features still represent different concepts.

---

## 🎯 What Is Normalization?

Normalization is a preprocessing method that rescales numerical data according to a chosen rule.

Different methods use different formulas, so **normalization does not always mean converting values to the range `[0, 1]`**.

Common approaches include:

- Min-Max normalization
- Mean normalization
- Maximum absolute scaling
- Vector normalization, also called unit-norm normalization

### When should you normalize?

Normalization is often useful when:

- Features have significantly different numerical ranges.
- A model calculates distances between observations.
- A model uses gradient-based optimization.
- You want a bounded range for numerical inputs.
- Feature magnitudes should not dominate a calculation simply because of their units.

---

## 💡 Why Is Normalization Important?

### 1. Makes feature ranges comparable

Features measured in different units can be placed on comparable numerical scales.

### 2. Helps distance-based algorithms

Algorithms such as K-Nearest Neighbors and K-Means use distances. Without scaling, a feature with large numerical values may dominate the distance calculation.

### 3. Can improve optimization

For some linear models and neural networks, scaling can improve numerical conditioning and make optimization more efficient.

### 4. Supports regularized models

L1 and L2 regularization penalize model coefficients. Feature scaling can make these penalties more comparable across features.

### 5. Helps numerical stability

Suitable scaling can reduce problems caused by extreme differences in numerical magnitudes.

**Important:** Normalization does not automatically improve every model. Its effect depends on the algorithm, dataset, and selected scaling method.

---

## ⚖️ Normalization vs Standardization

These terms are sometimes used interchangeably, but in machine learning they commonly refer to different transformations.

| Property              | Normalization                         | Standardization                            |
| --------------------- | ------------------------------------- | ------------------------------------------ |
| Common technique      | Min-Max scaling                       | Z-score scaling                            |
| Formula               | ((x-x\_{\min})/(x\_{\max}-x\_{\min})) | ((x-\mu)/\sigma)                           |
| Typical output        | `[0, 1]`                              | Mean near `0`, standard deviation near `1` |
| Fixed output range    | Yes, for values within fitted bounds  | No                                         |
| Sensitive to outliers | Min-Max is sensitive                  | Mean and standard deviation are sensitive  |
| Scikit-Learn tool     | `MinMaxScaler`                        | `StandardScaler`                           |

Example:

```
from sklearn.preprocessing import MinMaxScaler, StandardScaler

X = [[10], [20], [30], [40], [50]]

minmax = MinMaxScaler()
standard = StandardScaler()

print("Min-Max:")
print(minmax.fit_transform(X))

print("Standardization:")
print(standard.fit_transform(X))
```

Use `MinMaxScaler` when a bounded range is useful. Use `StandardScaler` when centering features and scaling their variance is more appropriate.

---

## 🧮 Types of Normalization

| Method                   | Main idea                                                    | Scikit-Learn tool            |
| ------------------------ | ------------------------------------------------------------ | ---------------------------- |
| Min-Max                  | Map features to a selected range                             | `MinMaxScaler`               |
| Mean normalization       | Center around the mean and scale by the range                | Manual or custom transformer |
| Maximum absolute scaling | Divide each feature by its maximum absolute value            | `MaxAbsScaler`               |
| L1 vector normalization  | Make the sum of absolute values equal to `1` for each sample | `Normalizer(norm="l1")`      |
| L2 vector normalization  | Make the Euclidean length equal to `1` for each sample       | `Normalizer(norm="l2")`      |

Choose the method according to the data and the algorithm, rather than applying every method automatically.

---

## 1. Min-Max Normalization

Min-Max scaling transforms a feature into a selected range, commonly `[0, 1]`.

### Formula

For a feature with minimum (x\_{\min}) and maximum (x\_{\max}):

[\
x'=\frac{x-x\_{\min}}{x\_{\max}-x\_{\min}}\
]

To map the values into a general range ([a,b]):

[\
x'=a+\frac{x-x\_{\min}}{x\_{\max}-x\_{\min}}(b-a)\
]

### Manual calculation

Suppose the values are:

```
10, 20, 30, 40, 50
```

For the value `30`:

[\
x'=\frac{30-10}{50-10}\
]

[\
x'=\frac{20}{40}=0.5\
]

The complete result is:

```
Original:     10, 20, 30, 40, 50
Normalized: 0.00, 0.25, 0.50, 0.75, 1.00
```

### Python example

```
def min_max_normalize(values):
    """Scale a list of numbers into the range [0, 1]."""
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)
    feature_range = maximum - minimum

    if feature_range == 0:
        return [0.0 for _ in values]

    return [
        (value - minimum) / feature_range
        for value in values
    ]


def main():
    numbers = [10, 20, 30, 40, 50]

    normalized = min_max_normalize(numbers)

    print("Original:", numbers)
    print("Normalized:", normalized)


if __name__ == "__main__":
    main()
```

**Output:**

```
Original: [10, 20, 30, 40, 50]
Normalized: [0.0, 0.25, 0.5, 0.75, 1.0]
```

The function returns zeros for a constant feature because its range is zero. This is a practical convention for this example; Scikit-Learn handles constant features according to its own implementation.

### Min-Max scaling with Scikit-Learn

```
from sklearn.preprocessing import MinMaxScaler

X = [[10], [20], [30], [40], [50]]

scaler = MinMaxScaler(feature_range=(0, 1))
X_normalized = scaler.fit_transform(X)

print(X_normalized)
```

### Scaling to `[-1, 1]`

```
from sklearn.preprocessing import MinMaxScaler

X = [[10], [20], [30], [40], [50]]

scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

print(X_scaled)
```

### Advantages

- Produces an easy-to-interpret range.
- Preserves the relative ordering of values.
- Useful when bounded inputs are desirable.

### Limitations

- Sensitive to extreme minimum and maximum values.
- New values can fall outside the selected range when they exceed the training bounds.
- Does not make the distribution normal.

---

## 2. Mean Normalization

Mean normalization subtracts the feature mean and divides by the feature range.

### Formula

[\
x'=\frac{x-\mu}{x\_{\max}-x\_{\min}}\
]

Where:

- (x) is the original value.
- (\mu) is the feature mean.
- (x\_{\max}) is the maximum.
- (x\_{\min}) is the minimum.

For `[10, 20, 30, 40, 50]`, the mean is `30` and the range is `40`.

The transformed values are:

```
-0.50, -0.25, 0.00, 0.25, 0.50
```

### Python implementation

```
def mean_normalize(values):
    """Center values around the mean and scale by their range."""
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)
    mean = sum(values) / len(values)
    feature_range = maximum - minimum

    if feature_range == 0:
        return [0.0 for _ in values]

    return [
        (value - mean) / feature_range
        for value in values
    ]


def main():
    numbers = [10, 20, 30, 40, 50]
    print(mean_normalize(numbers))


if __name__ == "__main__":
    main()
```

Mean normalization is less commonly used as a standard Scikit-Learn preprocessing component than Min-Max scaling or standardization. If used in a predictive workflow, implement it as a fitted transformer so its statistics are learned from the training data only.

---

## 3. Maximum Absolute Scaling

Maximum absolute scaling divides each feature by its largest absolute value.

### Formula

[\
x'=\frac{x}{\max(|x|)}\
]

The transformed values typically lie in `[-1, 1]`.

### Example

Original values:

```
-100, -50, 0, 50, 100
```

Normalized values:

```
-1.0, -0.5, 0.0, 0.5, 1.0
```

### Python implementation

```
def max_abs_normalize(values):
    """Scale values by their maximum absolute magnitude."""
    if not values:
        return []

    maximum_absolute = max(abs(value) for value in values)

    if maximum_absolute == 0:
        return [0.0 for _ in values]

    return [
        value / maximum_absolute
        for value in values
    ]


def main():
    numbers = [-100, -50, 0, 50, 100]
    print(max_abs_normalize(numbers))


if __name__ == "__main__":
    main()
```

### Scikit-Learn implementation

```
from sklearn.preprocessing import MaxAbsScaler

X = [
    [-100, 10],
    [-50, 20],
    [0, 30],
    [50, 40],
    [100, 50],
]

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)
```

`MaxAbsScaler` is useful for sparse data because it does not center features, helping preserve zero entries.

---

## 4. Vector Normalization

Vector normalization scales each sample so that its vector length becomes `1` or its sum of absolute values becomes `1`.

Unlike `MinMaxScaler`, which learns a minimum and maximum for each feature, `Normalizer` works across the features of each individual sample.

### L2 normalization

The L2 norm is:

[\
|x|\_2=\sqrt{x_1^2+x_2^2+\cdots+x_n^2}\
]

Each component is transformed as:

[\
x_i'=\frac{x_i}{|x|\_2}\
]

Example:

```
Original vector: [3, 4]

L2 norm = sqrt(3² + 4²)
        = sqrt(25)
        = 5

Normalized vector: [0.6, 0.8]
```

The normalized vector has a length of `1`.

### L1 normalization

The L1 norm is:

[\
|x|*1=\sum*{i=1}^{n}|x_i|\
]

For `[3, 4]`, the L1 norm is `7`, producing:

```
[3/7, 4/7]
```

### Scikit-Learn example

```
from sklearn.preprocessing import Normalizer

X = [
    [3, 4],
    [1, 2],
    [5, 12],
]

l2_normalizer = Normalizer(norm="l2")
X_l2 = l2_normalizer.transform(X)

print("L2 normalized:")
print(X_l2)

l1_normalizer = Normalizer(norm="l1")
X_l1 = l1_normalizer.transform(X)

print("L1 normalized:")
print(X_l1)
```

`Normalizer` is stateless: it normalizes each input row independently and does not learn feature minima, maxima, means, or standard deviations.

### Where is vector normalization useful?

- Text classification using TF-IDF vectors.
- Document similarity.
- Cosine-similarity workflows.
- Some clustering and nearest-neighbor applications.

**Important:** Vector normalization changes the magnitude of each sample. Use it only when the direction or relative proportions of the vector matter more than its original magnitude.

---

## 🐍 Normalization Using NumPy

NumPy provides efficient vectorized operations for numerical arrays.

### Min-Max normalization

```
import numpy as np


def min_max_scale_array(values):
    """Min-Max scale a one-dimensional NumPy array."""
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return values.copy()

    minimum = np.min(values)
    maximum = np.max(values)
    feature_range = maximum - minimum

    if feature_range == 0:
        return np.zeros_like(values)

    return (values - minimum) / feature_range


def main():
    values = np.array([10, 20, 30, 40, 50])

    result = min_max_scale_array(values)

    print("Original:", values)
    print("Normalized:", result)


if __name__ == "__main__":
    main()
```

### Normalize each column in a 2D array

```
import numpy as np

X = np.array(
    [
        [10, 100],
        [20, 200],
        [30, 300],
        [40, 400],
    ],
    dtype=float,
)

minimum = X.min(axis=0)
maximum = X.max(axis=0)
ranges = maximum - minimum

# Avoid division by zero for constant columns.
safe_ranges = np.where(ranges == 0, 1, ranges)

X_normalized = (X - minimum) / safe_ranges

print(X_normalized)
```

Here, `axis=0` calculates statistics column by column.

For real ML workflows, prefer a fitted transformer such as `MinMaxScaler` so the training statistics can be reused consistently on validation, test, and production data.

---

## 🐼 Normalization Using Pandas

Pandas is useful for normalizing selected numerical columns in a DataFrame.

```
import pandas as pd

df = pd.DataFrame(
    {
        "age": [20, 30, 40, 50],
        "salary": [20000, 40000, 60000, 80000],
        "score": [40, 60, 80, 100],
    }
)

columns = ["age", "salary", "score"]

minimum = df[columns].min()
maximum = df[columns].max()
ranges = maximum - minimum

# Replace zero ranges to avoid division by zero.
safe_ranges = ranges.mask(ranges == 0, 1)

df_normalized = df.copy()
df_normalized[columns] = (
    df[columns] - minimum
) / safe_ranges

print(df_normalized)
```

### Important Pandas considerations

- Select numerical columns intentionally.
- Preserve the original DataFrame when you need an audit trail.
- Handle missing values according to your preprocessing strategy.
- Avoid normalizing identifiers, arbitrary category codes, or target labels without a clear reason.
- For predictive ML, fit scaling statistics on training data only.

This DataFrame example demonstrates the calculation. It is not a substitute for a train-fitted preprocessing pipeline in a real model.

---

## 🤖 Normalization Using Scikit-Learn

Scikit-Learn offers reusable preprocessing transformers.

### `MinMaxScaler`

```
from sklearn.preprocessing import MinMaxScaler

X_train = [[10], [20], [30], [40]]
X_test = [[25], [50]]

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training data:")
print(X_train_scaled)

print("Test data:")
print(X_test_scaled)
```

The scaler learns the minimum and maximum from `X_train`. The test value `50` is transformed using those training statistics, so its scaled value can exceed `1`.

### `MaxAbsScaler`

```
from sklearn.preprocessing import MaxAbsScaler

X = [[-10, 100], [0, 200], [10, 300]]

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)
```

### `Normalizer`

```
from sklearn.preprocessing import Normalizer

X = [[3, 4], [5, 12]]

normalizer = Normalizer(norm="l2")
X_normalized = normalizer.transform(X)

print(X_normalized)
```

### Quick comparison

| Transformer      | Learns training statistics? | Operates across     |
| ---------------- | --------------------------- | ------------------- |
| `MinMaxScaler`   | Yes                         | Each feature/column |
| `MaxAbsScaler`   | Yes                         | Each feature/column |
| `StandardScaler` | Yes                         | Each feature/column |
| `Normalizer`     | No                          | Each sample/row     |

---

## 🔢 Normalize Multiple Features

Consider a dataset containing age, salary, and experience.

```
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.DataFrame(
    {
        "age": [22, 28, 35, 45],
        "salary": [25000, 45000, 70000, 120000],
        "experience": [0, 3, 8, 18],
    }
)

features = ["age", "salary", "experience"]

scaler = MinMaxScaler()

scaled_values = scaler.fit_transform(df[features])

df_scaled = df.copy()
df_scaled[features] = scaled_values

print(df_scaled)
```

Each feature gets its own minimum and maximum. This prevents salary's large numerical magnitude from automatically overwhelming age or experience in calculations based on feature scale.

**Do not automatically scale every column.** Dates, identifiers, free-text fields, categorical labels, and target variables need their own treatment.

---

## 🚨 Train-Test Split and Data Leakage

Data leakage occurs when information from outside the training process influences model training or preprocessing in a way that would not be available at prediction time.

A common mistake is fitting the scaler on the entire dataset before splitting it.

### ❌ Incorrect approach

```
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Incorrect for an ordinary held-out evaluation:
X_scaled = scaler.fit_transform(X)

# The scaler has already seen the test-set distribution.
```

### ✅ Correct approach

```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Remember:

- `fit_transform()` learns statistics and transforms the training data.
- `transform()` reuses the learned statistics for validation, test, or new data.
- Do not independently fit a new scaler on the test set.
- Do not use test-set results to choose preprocessing settings repeatedly.

For cross-validation, use a pipeline so the scaler is fitted separately inside each training fold.

---

## 🔗 Normalization with Pipelines

A Scikit-Learn pipeline combines preprocessing and model training into one reproducible workflow.

### Example: Min-Max scaling with Logistic Regression

```
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def main():
    dataset = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )

    model = Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=5000),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
```

Install dependencies if needed:

```
python -m pip install numpy pandas scikit-learn
```

### Why use a pipeline?

- Helps prevent preprocessing leakage.
- Keeps training and inference transformations consistent.
- Works naturally with cross-validation and hyperparameter search.
- Simplifies model deployment.
- Makes the preprocessing steps easier to reproduce.

For most supervised learning workflows, this is safer than manually scaling training and test data in separate code blocks.

---

## 🧠 Normalization for Different ML Algorithms

| Algorithm               | Is scaling usually useful? | Explanation                                           |
| ----------------------- | -------------------------- | ----------------------------------------------------- |
| K-Nearest Neighbors     | Yes                        | Distances depend on feature magnitudes                |
| K-Means                 | Yes                        | Clustering is based on distances                      |
| Support Vector Machines | Usually                    | Scale affects distances and optimization              |
| Logistic Regression     | Often                      | Helps optimization and regularization                 |
| Linear Regression       | Sometimes                  | Useful for numerical conditioning and regularization  |
| Neural Networks         | Often                      | Can support more stable optimization                  |
| PCA                     | Usually                    | Features with large variance can dominate components  |
| Decision Trees          | Usually unnecessary        | Splits generally depend on feature ordering           |
| Random Forest           | Usually unnecessary        | Tree-based splitting is generally scale-insensitive   |
| Gradient-Boosted Trees  | Usually unnecessary        | Most tree-based implementations are scale-insensitive |

These are general guidelines, not universal rules. Check the requirements of the particular implementation and compare results using an appropriate validation strategy.

---

## ⚠️ Handling Outliers and Constant Features

### Outliers and Min-Max scaling

Suppose the values are:

```
10, 20, 30, 40, 1000
```

The extreme value `1000` becomes `1`, while most of the remaining values are compressed close to zero.

Min-Max scaling does not remove or reduce the influence of an outlier; it simply rescales the observed range.

Potential approaches include:

- Investigate whether the extreme value is a data error.
- Use domain-specific validation rules.
- Consider a transformation such as `log1p` for suitable non-negative, highly skewed variables.
- Compare standardization, robust scaling, and Min-Max scaling.
- Apply clipping only when justified by the data and use case.

Do not remove legitimate rare observations solely because they are extreme.

### Constant features

If every value in a feature is `5`, its range is zero. A manual Min-Max formula would divide by zero.

Use a well-tested transformer, such as Scikit-Learn's `MinMaxScaler`, for robust handling. Consider removing features that carry no useful information, subject to domain requirements.

### Missing values

Most preprocessing workflows should address missing values deliberately, often by imputing them before scaling.

For example:

```
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

numeric_preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ]
)
```

Fit this pipeline only on training data, or include it in a larger model pipeline.

---

## 🧪 Normalization in Classification

Normalization is frequently useful for classification algorithms that rely on distances, margins, or gradient-based optimization.

Example applications include:

- Predicting whether a customer will leave a service.
- Classifying medical measurements.
- Detecting spam from numerical text features.
- Identifying categories from sensor measurements.

For classification, split the data first, then use a pipeline that includes the scaler and classifier. Evaluate with appropriate metrics such as accuracy, precision, recall, F1-score, and ROC-AUC when applicable.

Avoid comparing models solely on training accuracy.

---

## 📈 Normalization in Regression

Regression models can also benefit from feature scaling, particularly when regularization is used.

Common examples include:

- House-price prediction.
- Salary prediction.
- Demand forecasting.
- Energy-consumption prediction.

A simple Ridge regression pipeline:

```
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def main():
    dataset = load_diabetes()

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean squared error: {mse:.4f}")
    print(f"R² score: {r2:.4f}")


if __name__ == "__main__":
    main()
```

This example scales the input features, not the target variable. Target scaling is a separate decision and requires inverse-transforming predictions when original units are needed.

---

## 🧠 Normalization in Deep Learning

Neural networks often benefit from appropriately scaled input features. Inputs with vastly different magnitudes can make optimization more difficult.

Common input preprocessing choices include:

- Min-Max scaling for bounded numerical inputs.
- Standardization for features centered around zero.
- Image pixel scaling, such as dividing pixel values by `255.0` for 8-bit images.
- Vector normalization for selected embedding or similarity workflows.

Example for image pixels:

```
import numpy as np

pixels = np.array(
    [0, 64, 128, 192, 255],
    dtype=np.float32,
)

normalized_pixels = pixels / 255.0

print(normalized_pixels)
```

Output:

```
[0.        0.2509804 0.5019608 0.7529412 1.       ]
```

**Do not confuse input normalization with neural-network normalization layers.** Batch Normalization and Layer Normalization are model operations with different purposes and behaviors.

---

## ❌ Common Mistakes

1. Fitting a scaler on the full dataset before splitting.
2. Calling `fit_transform()` separately on training and test data.
3. Assuming all normalized values must always remain in `[0, 1]`.
4. Believing normalization makes a distribution Gaussian.
5. Ignoring extreme values before applying Min-Max scaling.
6. Scaling categorical labels as if they were continuous measurements.
7. Normalizing IDs that have no meaningful numerical relationship.
8. Using row-wise vector normalization when absolute magnitude is important.
9. Assuming scaling always improves tree-based models.
10. Forgetting to save and reuse the fitted preprocessing pipeline at inference time.
11. Applying different transformations during training and production.
12. Choosing a scaling technique using the test set instead of validation data.

---

## ✅ Best Practices

- Explore feature distributions and ranges before scaling.
- Split data before learning preprocessing statistics.
- Use pipelines for cross-validation and production workflows.
- Select only appropriate numerical features.
- Choose the transformation based on the algorithm and data.
- Investigate outliers instead of blindly removing them.
- Handle missing values explicitly.
- Keep feature names and column ordering consistent.
- Compare relevant alternatives using the same validation procedure.
- Preserve the fitted preprocessing steps with the model artifact.
- Document the preprocessing choices for reproducibility.

---

## 🚀 Mini Projects

### Project 1: Student Performance Scaling

**Objective:** Normalize student marks, attendance, and study hours.

Tasks:

- Create a dataset with at least three numerical features.
- Apply Min-Max scaling.
- Compare original and scaled values.
- Plot the distributions before and after scaling.
- Explain whether the transformation changed the ordering of values.

### Project 2: Customer Segmentation

**Objective:** Compare clustering results before and after scaling.

Tasks:

- Use a customer dataset with numerical features.
- Apply K-Means before scaling.
- Apply K-Means after scaling.
- Compare cluster assignments and suitable clustering metrics.
- Explain how feature magnitudes affected the distance calculations.

### Project 3: Classification Pipeline

**Objective:** Build a classification model using a preprocessing pipeline.

Tasks:

- Load a suitable classification dataset.
- Split it into training and test sets.
- Build a pipeline with `MinMaxScaler` and a classifier.
- Evaluate accuracy, precision, recall, and F1-score.
- Compare results against an appropriate baseline.

### Project 4: Scaling Method Comparison

**Objective:** Compare Min-Max scaling, standardization, and maximum absolute scaling.

Tasks:

- Choose a dataset containing multiple numerical features.
- Train the same suitable model with each scaling method.
- Use identical train/test splits and evaluation metrics.
- Compare performance and training behavior.
- Write a short conclusion explaining which method worked best for the selected experiment.

---

## 📝 Practice Exercises

1. Define normalization in machine learning.
2. Write the formula for Min-Max normalization.
3. Normalize `[5, 10, 15, 20, 25]` into `[0, 1]`.
4. Explain the difference between normalization and standardization.
5. What happens when a feature has the same value in every row?
6. Why is Min-Max scaling sensitive to outliers?
7. Explain the difference between `MinMaxScaler` and `Normalizer`.
8. Why should the scaler be fitted only on training data?
9. Name three machine learning algorithms that commonly benefit from feature scaling.
10. Why do decision trees generally not require feature scaling?
11. What happens if a future value is larger than the training maximum?
12. Explain how a Scikit-Learn pipeline helps prevent data leakage.

---

## ⚡ Quick Revision

| Concept                            | Remember                                               |
| ---------------------------------- | ------------------------------------------------------ |
| Normalization                      | Rescales numerical data using a chosen rule            |
| Min-Max formula                    | ((x-x\_{\min})/(x\_{\max}-x\_{\min}))                  |
| Common range                       | `[0, 1]`                                               |
| General range                      | Any valid interval such as `[-1, 1]`                   |
| Mean normalization                 | Centers values around the mean and scales by the range |
| Maximum absolute scaling           | Divides by the largest absolute feature value          |
| Vector normalization               | Scales each sample according to its vector norm        |
| Scikit-Learn Min-Max tool          | `MinMaxScaler`                                         |
| Scikit-Learn maximum absolute tool | `MaxAbsScaler`                                         |
| Scikit-Learn vector tool           | `Normalizer`                                           |
| Preventing leakage                 | Fit on training data; transform other datasets         |
| Production workflow                | Save and reuse the fitted pipeline                     |

---

## 📁 Project Structure

A recommended folder structure for this module:

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
│   └── README.md
│
└── 04-Normalization/
    ├── README.md
    ├── min_max_normalization.py
    ├── mean_normalization.py
    ├── max_abs_normalization.py
    ├── vector_normalization.py
    ├── numpy_normalization.py
    ├── pandas_normalization.py
    ├── sklearn_normalization.py
    └── normalization_pipeline.py
```

---

## 🗺️ Learning Roadmap

Follow this sequence to build your preprocessing skills:

1. Understand data splitting and validation.
2. Learn why feature scaling matters.
3. Understand standardization and z-scores.
4. Master Min-Max normalization.
5. Explore maximum absolute and vector normalization.
6. Practice with NumPy and Pandas.
7. Use Scikit-Learn preprocessing transformers.
8. Build leakage-safe pipelines.
9. Compare scaling techniques through experiments.
10. Apply preprocessing to an end-to-end ML project.

---

## 🎯 Key Takeaways

- Normalization rescales numerical data according to a selected method.
- Min-Max scaling is commonly used to map features into `[0, 1]`.
- Mean normalization and maximum absolute scaling follow different formulas.
- Vector normalization operates on individual samples rather than learning a range for each feature.
- Scaling is especially relevant to distance-based algorithms and many gradient-based models.
- Tree-based algorithms usually do not require scaling.
- Always learn preprocessing statistics from training data only.
- Scikit-Learn pipelines make preprocessing safer and more reproducible.

---

## 🔜 What's Next?

Continue your Machine Learning journey with the next preprocessing topic:

**Feature encoding** — learn how to transform categorical data into numerical representations that machine learning algorithms can use.

Explore the complete repository: [Machine Learning — GitHub](https://github.com/Kishor055/Machine-Learning)

---

## 👨‍💻 Author

**Kishor Patil**

GitHub: [Kishor055](https://github.com/Kishor055)

Explore the repository, practice the examples, and build your own machine learning projects.

## 🤝 Contributing

Contributions are welcome! You can improve explanations, correct mistakes, add practical examples, or suggest new exercises.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Submit a pull request with a clear description.

## ⭐ Support

If you find this repository useful, consider giving it a star on GitHub. Your support helps encourage continued learning and improvement.

## 📄 License

Refer to the repository's root `LICENSE` file for licensing terms.

---

**Keep learning. Keep building. Keep growing. 🚀**
