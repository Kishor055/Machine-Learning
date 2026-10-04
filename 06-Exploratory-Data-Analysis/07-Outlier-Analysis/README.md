
# 🚨 Outlier Analysis

> **Outlier Analysis = Detect → Investigate → Validate → Decide → Treat → Monitor**

![Machine Learning](https://img.shields.io/badge/Domain-Machine%20Learning-blue)
![EDA](https://img.shields.io/badge/Topic-Exploratory%20Data%20Analysis-orange)
![Python](https://img.shields.io/badge/Python-3.x-yellow)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-purple)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-darkgreen)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E)
![Status](https://img.shields.io/badge/Status-Learning%20Module-success)

---

# 🌱 GROW → DETECT → INVESTIGATE → VALIDATE → TREAT → BUILD 🚀

```text
                         📊 DATASET
                            │
                            ▼
                    🔍 DATA INSPECTION
                            │
                            ▼
                    🚨 OUTLIER ANALYSIS
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
            IQR          Z-SCORE       PERCENTILE
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                     📈 VISUALIZATION
                            │
                            ▼
                    🧠 INVESTIGATION
                            │
                 ┌──────────┼──────────┐
                 ▼          ▼          ▼
              VALID      ERROR       UNKNOWN
                 │          │          │
                 ▼          ▼          ▼
               KEEP       FIX       INVESTIGATE
                 │          │          │
                 └──────────┼──────────┘
                            ▼
                  MULTIVARIATE ANALYSIS
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
            ISOLATION FOREST      LOCAL OUTLIER
                  │                   │
                  └─────────┬─────────┘
                            ▼
                    🤖 MACHINE LEARNING
                            │
                            ▼
                     📈 MONITORING
```

---

# 📚 Table of Contents

1. [What Is an Outlier?](#-what-is-an-outlier)
2. [Why Outlier Analysis Matters](#-why-outlier-analysis-matters)
3. [Outlier vs Anomaly](#-outlier-vs-anomaly)
4. [Why Outliers Occur](#-why-outliers-occur)
5. [Types of Outliers](#-types-of-outliers)
6. [Univariate Outliers](#-univariate-outliers)
7. [Bivariate Outliers](#-bivariate-outliers)
8. [Multivariate Outliers](#-multivariate-outliers)
9. [Global vs Local Outliers](#-global-vs-local-outliers)
10. [Outliers vs Data Errors](#-outliers-vs-data-errors)
11. [Why You Should Not Automatically Remove Outliers](#-why-you-should-not-automatically-remove-outliers)
12. [Outlier Detection Workflow](#-outlier-detection-workflow)
13. [Visual Outlier Detection](#-visual-outlier-detection)
14. [Box Plot Method](#-box-plot-method)
15. [IQR Method](#-iqr-method)
16. [Manual IQR Calculation](#-manual-iqr-calculation)
17. [IQR Implementation with Pandas](#-iqr-implementation-with-pandas)
18. [Z-Score Method](#-z-score-method)
19. [Manual Z-Score Calculation](#-manual-z-score-calculation)
20. [Z-Score with SciPy](#-z-score-with-scipy)
21. [Modified Z-Score](#-modified-z-score)
22. [Percentile Method](#-percentile-method)
23. [Winsorization](#-winsorization)
24. [Trimming](#-trimming)
25. [Capping](#-capping)
26. [Transformation](#-transformation)
27. [Missing Value Treatment](#-missing-value-treatment)
28. [Domain-Based Rules](#-domain-based-rules)
29. [Outlier Detection with Scatter Plots](#-outlier-detection-with-scatter-plots)
30. [Multivariate Outlier Detection](#-multivariate-outlier-detection)
31. [Mahalanobis Distance](#-mahalanobis-distance)
32. [Isolation Forest](#-isolation-forest)
33. [Local Outlier Factor](#-local-outlier-factor)
34. [One-Class SVM](#-one-class-svm)
35. [Outlier Detection Comparison](#-outlier-detection-comparison)
36. [Outliers and Machine Learning](#-outliers-and-machine-learning)
37. [Outliers and Different Models](#-outliers-and-different-models)
38. [Outliers and Data Leakage](#-outliers-and-data-leakage)
39. [Outlier Treatment Strategies](#-outlier-treatment-strategies)
40. [How to Choose a Strategy](#-how-to-choose-a-strategy)
41. [Outlier Analysis with Python](#-outlier-analysis-with-python)
42. [Reusable Python Functions](#-reusable-python-functions)
43. [Automated Outlier Report](#-automated-outlier-report)
44. [End-to-End Example](#-end-to-end-example)
45. [Common Mistakes](#-common-mistakes)
46. [Best Practices](#-best-practices)
47. [Mini Projects](#-mini-projects)
48. [Exercises](#-exercises)
49. [Project Structure](#-project-structure)
50. [Outlier Analysis Checklist](#-outlier-analysis-checklist)
51. [EDA Roadmap](#-eda-roadmap)
52. [Key Takeaways](#-key-takeaways)
53. [Next Step](#-next-step)

---

# 🔍 What Is an Outlier?

An **outlier** is an observation that is unusually different from the majority of observations in a dataset.

Example:

```text
Normal values:

20
22
21
24
23
25
22
24

Possible outlier:

250
```

However:

> **An outlier is not automatically an error.**

An unusual observation may represent:

* A legitimate rare event
* A high-value customer
* A genuine medical case
* A major transaction
* A sensor failure
* A data-entry error
* Fraud
* A new population segment

Therefore, outlier analysis should answer:

```text
Is it unusual?
       ↓
Why is it unusual?
       ↓
Is it valid?
       ↓
What should we do?
```

---

# 🎯 Why Outlier Analysis Matters

Outliers can influence:

* Mean
* Variance
* Standard deviation
* Correlation
* Regression coefficients
* Distance-based algorithms
* Clustering
* PCA
* Statistical tests
* Model performance

For example:

```text
Without outlier:

10  11  12  13  14

With outlier:

10  11  12  13  1000
```

The mean changes dramatically.

Therefore:

```text
Outlier Analysis
       ↓
Better Data Understanding
       ↓
Better Modeling Decisions
```

---

# 🚨 Outlier vs Anomaly

The terms are related but not always identical.

### Outlier

An observation that differs substantially from other observations.

### Anomaly

An observation or pattern that is considered unusual within a specific context or expected behavior.

For example:

```text
Transaction of ₹10 lakh
```

may be:

```text
Normal for a corporate account
```

but:

```text
Anomalous for a small personal account
```

Context matters.

---

# ❓ Why Outliers Occur

Outliers can arise from several sources.

## 1. Measurement Error

```text
Temperature = 250°C
```

when the sensor range is:

```text
-50°C to 100°C
```

---

## 2. Data Entry Error

```text
Age = 250
```

instead of:

```text
Age = 25
```

---

## 3. Sensor Failure

A faulty sensor may generate extreme values.

---

## 4. Genuine Rare Event

Examples:

* Extremely high income
* Large financial transaction
* Exceptional exam score
* Major earthquake

---

## 5. Fraud

Fraudulent transactions may appear unusual.

---

## 6. Population Mixing

Different populations may have different distributions.

Example:

```text
Students + Professionals
```

may produce a mixed age distribution.

---

## 7. Sampling Problems

A sample may contain unusual observations because of how it was collected.

---

# 🧩 Types of Outliers

Outliers can be categorized by how they are detected.

```text
Outliers
│
├── Univariate
│
├── Bivariate
│
├── Multivariate
│
├── Global
│
├── Local
│
└── Contextual
```

---

# 1️⃣ Univariate Outliers

A univariate outlier is unusual with respect to one variable.

Example:

```text
Salary:

30k
32k
35k
37k
39k
500k  ← possible outlier
```

Common techniques:

* IQR
* Z-score
* Modified Z-score
* Percentile rules

---

# 2️⃣ Bivariate Outliers

An observation may not be unusual for either variable independently but may be unusual in their relationship.

Example:

```text
Experience → Salary

5 years → ₹50k     normal
10 years → ₹90k    normal
10 years → ₹5k     unusual relationship
```

Scatter plots are useful for detecting such observations.

---

# 3️⃣ Multivariate Outliers

A point may appear normal in every individual feature but unusual when considering several features together.

Example:

```text
Age       = 30
Income    = ₹50k
Expenses  = ₹45k
Savings   = ₹1k
```

Each value may be individually plausible, but the combination may be unusual.

Multivariate methods include:

* Mahalanobis distance
* Isolation Forest
* Local Outlier Factor
* One-Class SVM

---

# 🌍 Global vs Local Outliers

## Global Outlier

An observation is unusual compared with the entire dataset.

```text
Dataset
──────────────────────────
Normal Normal Normal  X
                       ↑
                   Global outlier
```

---

## Local Outlier

An observation is unusual only within its local neighborhood.

```text
Cluster A       Cluster B

•••••            •••••
•••••      X     •••••
•••••            •••••
           ↑
      Local outlier
```

This is why methods such as Local Outlier Factor are useful.

---

# ⚠️ Outliers vs Data Errors

Consider:

```text
Age = 150
```

This is likely a data-quality issue.

But:

```text
Transaction = ₹50,00,000
```

may be completely legitimate.

Therefore:

```text
Unusual ≠ Incorrect
```

The correct workflow is:

```text
Detect
  ↓
Investigate
  ↓
Validate
  ↓
Decide
```

---

# 🛑 Why You Should Not Automatically Remove Outliers

A common beginner mistake is:

```text
Find Outliers
      ↓
Delete Outliers
```

This is dangerous.

Instead:

```text
Find Outliers
      ↓
Understand Cause
      ↓
Validate Observation
      ↓
Choose Treatment
```

Possible actions:

```text
Keep
Fix
Remove
Cap
Transform
Impute
Separate
Flag
```

---

# 🔄 Outlier Detection Workflow

```text
                  RAW DATA
                     │
                     ▼
              DATA VALIDATION
                     │
                     ▼
              VISUAL INSPECTION
                     │
                     ▼
             DETECTION METHODS
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
       IQR         Z-SCORE     PERCENTILE
        │            │            │
        └────────────┼────────────┘
                     ▼
              INVESTIGATE CASES
                     │
              ┌──────┼──────┐
              ▼      ▼      ▼
            VALID  ERROR  UNKNOWN
              │      │      │
              ▼      ▼      ▼
            KEEP    FIX   INVESTIGATE
                     │
                     ▼
              CHOOSE TREATMENT
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        REMOVE      CAP      TRANSFORM
          │          │          │
          └──────────┼──────────┘
                     ▼
                VALIDATE
                     │
                     ▼
             MACHINE LEARNING
```

---

# 📈 Visual Outlier Detection

Visualization should usually be one of the first steps.

Useful plots:

```text
Box Plot
Scatter Plot
Histogram
ECDF
Violin Plot
```

---

# 📦 Box Plot Method

A box plot is one of the easiest ways to identify potential outliers.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df["salary"])

plt.title("Salary Box Plot")
plt.show()
```

Potential outliers appear beyond the whiskers.

---

# 📐 IQR Method

The **Interquartile Range (IQR)** is:

$$
IQR = Q_3 - Q_1
$$

Define:

$$
LowerBound = Q_1 - 1.5(IQR)
$$

$$
UpperBound = Q_3 + 1.5(IQR)
$$

Any value outside these boundaries is considered a **potential outlier** under the IQR rule.

---

# 🧮 Manual IQR Calculation

```python
import numpy as np

data = np.array([
    10, 12, 13, 14, 15,
    16, 17, 18, 20, 100
])

q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = data[
    (data < lower_bound) |
    (data > upper_bound)
]

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower:", lower_bound)
print("Upper:", upper_bound)
print("Outliers:", outliers)
```

---

# 🐼 IQR Implementation with Pandas

```python
def detect_iqr_outliers(series):
    clean = series.dropna()

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (
        (series < lower) |
        (series > upper)
    )

    return mask
```

Usage:

```python
mask = detect_iqr_outliers(
    df["salary"]
)

print(df.loc[mask])
```

---

# 📊 IQR Outlier Count

```python
mask = detect_iqr_outliers(
    df["salary"]
)

print("Outlier count:", mask.sum())
```

Outlier percentage:

```python
percentage = (
    mask.sum() / df["salary"].notna().sum()
) * 100

print(
    f"Outlier percentage: {percentage:.2f}%"
)
```

---

# 📏 Z-Score Method

The z-score measures how many standard deviations a value is from the mean.

$$
z = \frac{x-\mu}{\sigma}
$$

A common rule is:

```text
|z| > 3
```

may indicate a potential outlier.

Example:

```python
import numpy as np

data = np.array([
    10, 12, 13, 14, 15,
    16, 17, 18, 20, 100
])

mean = data.mean()
std = data.std()

z_scores = (
    (data - mean) / std
)

outliers = data[
    np.abs(z_scores) > 3
]

print("Z-scores:", z_scores)
print("Potential outliers:", outliers)
```

---

# 🧮 Manual Z-Score Calculation

For:

```text
x = 100
mean = 20
std = 10
```

we calculate:

$$
z = \frac{100-20}{10}=8
$$

Therefore:

```text
z = 8
```

which is extremely far from the mean under a normal-distribution reference.

---

# 🧪 Z-Score with SciPy

```python
from scipy.stats import zscore

z_scores = zscore(
    df["salary"].dropna()
)

outliers = df["salary"].dropna()[
    abs(z_scores) > 3
]

print(outliers)
```

### Important

The classic z-score method works best when:

* The distribution is reasonably well behaved
* Mean and standard deviation are meaningful
* Extreme skewness is not dominating the statistics

For heavily skewed data, robust methods may be preferable.

---

# 🛡️ Modified Z-Score

The modified z-score uses the **median** and **Median Absolute Deviation (MAD)**.

A common formulation is:

$$
Modified\ Z =
0.6745
\frac{x-\text{Median}}{MAD}
$$

where:

$$
MAD =
Median(|x_i-Median|)
$$

This method is more robust to extreme values than the standard z-score.

Example:

```python
import numpy as np


def modified_z_scores(series):
    clean = series.dropna()

    median = clean.median()

    mad = np.median(
        np.abs(
            clean - median
        )
    )

    if mad == 0:
        return pd.Series(
            np.nan,
            index=clean.index
        )

    return (
        0.6745
        * (clean - median)
        / mad
    )
```

Usage:

```python
scores = modified_z_scores(
    df["salary"]
)

outliers = df.loc[
    scores.abs() > 3.5
]

print(outliers)
```

---

# 🎯 Percentile Method

Sometimes domain knowledge suggests reasonable percentile limits.

Example:

```python
lower = df["income"].quantile(0.01)
upper = df["income"].quantile(0.99)

outliers = df[
    (df["income"] < lower) |
    (df["income"] > upper)
]
```

This identifies observations outside the central 98% of the observed distribution.

Percentile thresholds are especially useful when:

* The variable is strongly skewed
* Extreme tails need controlled treatment
* Business rules support the chosen thresholds

But percentile trimming can remove legitimate rare events.

---

# ✂️ Trimming

**Trimming** means removing observations considered outliers.

Example:

```python
q1 = df["income"].quantile(0.01)
q99 = df["income"].quantile(0.99)

clean_df = df[
    df["income"].between(q1, q99)
].copy()
```

Use trimming carefully.

Before removing observations, ask:

```text
Are they errors?
Are they valid rare cases?
Will removing them bias the dataset?
Does the model require removal?
```

---

# 📌 Capping

Capping replaces extreme values with predefined limits.

Example:

```python
lower = df["income"].quantile(0.01)
upper = df["income"].quantile(0.99)

df["income_capped"] = (
    df["income"]
    .clip(
        lower=lower,
        upper=upper
    )
)
```

Conceptually:

```text
Below lower bound → lower bound
Above upper bound → upper bound
Otherwise         → unchanged
```

Capping preserves the number of observations but modifies extreme values.

---

# 📦 Winsorization

Winsorization is a formal form of tail treatment where extreme observations are replaced by selected percentile values.

Conceptually:

```text
Lowest 1%   → 1st percentile
Highest 1%  → 99th percentile
```

One implementation is available through SciPy:

```python
from scipy.stats.mstats import winsorize

winsorized = winsorize(
    df["income"].dropna(),
    limits=[0.01, 0.01]
)

print(winsorized)
```

Use this only when the statistical or business rationale supports it.

---

# 🔄 Transformation

Instead of deleting an extreme value, a transformation can reduce its influence.

For positive data:

```python
import numpy as np

df["log_income"] = np.log1p(
    df["income"]
)
```

Other transformations include:

```text
Square Root
Log
Box-Cox
Yeo-Johnson
```

Transformations are especially useful for strongly right-skewed variables.

---

# 🕳️ Missing Value Treatment

Sometimes an extreme value is actually an invalid or missing observation.

For example:

```text
Age = -5
```

could be an invalid measurement rather than a legitimate outlier.

Possible workflow:

```python
df.loc[
    df["age"] < 0,
    "age"
] = np.nan
```

Then apply a missing-value strategy appropriate to the dataset.

The important principle is:

```text
Invalid value
     ↓
Correct representation
     ↓
Missing-value treatment
```

rather than blindly treating every unusual value as an ordinary outlier.

---

# 🧠 Domain-Based Rules

Statistical rules should be combined with domain knowledge.

Example:

```python
df.loc[
    ~df["age"].between(0, 120),
    "age"
] = np.nan
```

For product quantities:

```python
df.loc[
    df["quantity"] < 0,
    "quantity"
] = np.nan
```

For a percentage:

```python
invalid = ~df["conversion_rate"].between(
    0,
    1
)
```

Domain rules are often more meaningful than generic statistical thresholds.

---

# 📈 Outlier Detection with Scatter Plots

A scatter plot can reveal unusual relationships.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    data=df,
    x="experience",
    y="salary"
)

plt.title(
    "Experience vs Salary"
)

plt.show()
```

Look for:

* Isolated points
* Unexpected clusters
* Extreme x-values
* Extreme y-values
* Points far from the main pattern

---

# 🔬 Multivariate Outlier Detection

Univariate methods inspect one variable at a time.

But real-world anomalies are often multivariate.

Example:

```text
Age = 25
Income = ₹50,000
Expenses = ₹49,000
```

Each feature may be normal independently, but the combination may be unusual.

Multivariate methods include:

```text
Mahalanobis Distance
Isolation Forest
Local Outlier Factor
One-Class SVM
```

---

# 📐 Mahalanobis Distance

Mahalanobis distance accounts for relationships between variables.

For vector `x`:

$$
D_M(x)=
\sqrt{
(x-\mu)^T
\Sigma^{-1}
(x-\mu)
}
$$

where:

* `μ` = mean vector
* `Σ` = covariance matrix

Unlike ordinary Euclidean distance, Mahalanobis distance considers feature scale and covariance.

Conceptually:

```text
Euclidean Distance
       ↓
Ignores feature relationships

Mahalanobis Distance
       ↓
Accounts for covariance
```

A simple implementation:

```python
import numpy as np


def mahalanobis_distance(X):
    X = np.asarray(X)

    mean = X.mean(axis=0)
    covariance = np.cov(
        X,
        rowvar=False
    )

    inverse_covariance = np.linalg.pinv(
        covariance
    )

    centered = X - mean

    distances = np.sqrt(
        np.einsum(
            "ij,jk,ik->i",
            centered,
            inverse_covariance,
            centered
        )
    )

    return distances
```

---

# 🌲 Isolation Forest

**Isolation Forest** is an unsupervised anomaly-detection algorithm.

The key idea is:

> Anomalous observations are often easier to isolate than normal observations.

Conceptually:

```text
Normal observations
      ↓
Need more splits
      ↓
Harder to isolate

Anomalous observations
      ↓
Need fewer splits
      ↓
Easier to isolate
```

Example:

```python
from sklearn.ensemble import IsolationForest

features = [
    "age",
    "income",
    "expenses"
]

X = df[features].dropna()

model = IsolationForest(
    contamination=0.05,
    random_state=42
)

predictions = model.fit_predict(X)

outliers = X[
    predictions == -1
]

print(outliers)
```

Interpretation:

```text
1  → inlier
-1 → outlier
```

### Important

`contamination` should not be chosen blindly. It represents an assumption about the expected proportion of anomalies and should be validated against the problem.

---

# 📍 Local Outlier Factor

**Local Outlier Factor (LOF)** identifies observations that have substantially lower local density than their neighbors.

Conceptually:

```text
Dense Region

••••••
••••••
••• X •
••••••

X may be normal


Local Sparse Point

••••••

      X

••••••

X may be a local outlier
```

Example:

```python
from sklearn.neighbors import LocalOutlierFactor

X = df[
    ["age", "income"]
].dropna()

model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05
)

predictions = model.fit_predict(X)

outliers = X[
    predictions == -1
]

print(outliers)
```

LOF is particularly useful when local density matters.

---

# 🧠 One-Class SVM

One-Class SVM learns a boundary around the majority of observations.

```python
from sklearn.svm import OneClassSVM

X = df[
    ["age", "income"]
].dropna()

model = OneClassSVM(
    kernel="rbf",
    nu=0.05
)

predictions = model.fit_predict(X)

outliers = X[
    predictions == -1
]

print(outliers)
```

One-Class SVM can be sensitive to:

* Feature scale
* Kernel parameters
* Dataset size
* Noise

Scaling is often important.

---

# ⚖️ Outlier Detection Comparison

| Method           | Type               | Best For                   | Main Limitation                  |
| ---------------- | ------------------ | -------------------------- | -------------------------------- |
| IQR              | Univariate         | Robust numerical detection | Ignores relationships            |
| Z-score          | Univariate         | Approximately normal data  | Sensitive to outliers            |
| Modified Z-score | Univariate         | Robust detection           | MAD can be zero                  |
| Percentile       | Univariate         | Tail control               | Threshold-dependent              |
| Mahalanobis      | Multivariate       | Correlated numerical data  | Covariance assumptions           |
| Isolation Forest | Multivariate       | General anomaly detection  | Parameter-sensitive              |
| LOF              | Multivariate/local | Local anomalies            | Sensitive to neighborhood size   |
| One-Class SVM    | Multivariate       | Complex boundaries         | Can be computationally expensive |

---

# 🤖 Outliers and Machine Learning

Outliers affect different algorithms differently.

## Linear Regression

Potentially sensitive because extreme observations can strongly influence coefficients.

---

## Logistic Regression

Can also be affected by extreme feature values, especially depending on preprocessing and regularization.

---

## K-Nearest Neighbors

Highly sensitive to extreme values because it relies on distances.

---

## K-Means

Sensitive because centroids and distances can be influenced by extreme observations.

---

## PCA

Sensitive because covariance-based calculations can be strongly influenced by extreme values.

---

## Decision Trees

Often more robust to extreme feature values than distance-based methods, although extreme observations can still affect splits and the target.

---

## Random Forest

Generally robust to many feature outliers, but target outliers can still matter in regression.

---

# 🧠 Model Sensitivity Summary

```text
                    Outlier Sensitivity

High
 │
 │  KNN
 │  K-Means
 │  PCA
 │  Linear Models
 │
 │
 │  Logistic Models
 │
 │
 │  Tree-Based Models
 │
Low
```

This is a simplified conceptual guide, not a universal ranking.

Always evaluate the specific dataset and model.

---

# 🔐 Outliers and Data Leakage

Outlier treatment can cause leakage if thresholds are calculated using the entire dataset before splitting.

Incorrect:

```text
Full Dataset
     ↓
Calculate IQR
     ↓
Remove/Capture Outliers
     ↓
Train/Test Split
```

Preferred:

```text
Dataset
   ↓
Train/Test Split
   ↓
Calculate Training Thresholds
   ↓
Apply Training Transformation
   ↓
Apply Same Transformation to Test
```

For machine-learning pipelines, fit preprocessing only on training data.

---

# 🛠️ Outlier Treatment Strategies

There is no single best strategy.

| Strategy  | Use When                                        |
| --------- | ----------------------------------------------- |
| Keep      | Observation is valid and meaningful             |
| Fix       | Data-entry/measurement error                    |
| Remove    | Observation is invalid and cannot be corrected  |
| Cap       | Extreme values are valid but overly influential |
| Transform | Distribution is heavily skewed                  |
| Impute    | Value is invalid/missing and can be estimated   |
| Flag      | Outlier itself contains useful information      |
| Separate  | Rare observations represent a different process |

---

# 🧭 How to Choose a Strategy

```text
                 OUTLIER FOUND
                      │
                      ▼
             Is it a data error?
                │           │
               YES          NO
                │           │
                ▼           ▼
              FIX       Is it valid?
                           │
                     ┌─────┴─────┐
                    YES          NO
                     │            │
                     ▼            ▼
                KEEP / FLAG     REMOVE/FIX
                     │
                     ▼
              Is influence a
               modeling issue?
                     │
              ┌──────┴──────┐
             YES             NO
              │               │
              ▼               ▼
        Transform/Cap       KEEP
```

---

# 🐍 Outlier Analysis with Python

A complete univariate analysis can be written as:

```python
import numpy as np
import pandas as pd


def outlier_summary(series):
    clean = series.dropna()

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (
        (clean < lower) |
        (clean > upper)
    )

    return {
        "count": clean.count(),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": mask.sum(),
        "outlier_percentage": (
            mask.mean() * 100
        )
    }
```

Usage:

```python
summary = outlier_summary(
    df["income"]
)

for key, value in summary.items():
    print(f"{key}: {value}")
```

---

# 🧩 Reusable Python Functions

## IQR Bounds

```python
def iqr_bounds(series):
    clean = series.dropna()

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)

    iqr = q3 - q1

    return (
        q1 - 1.5 * iqr,
        q3 + 1.5 * iqr
    )
```

---

## Get Outliers

```python
def get_iqr_outliers(series):
    lower, upper = iqr_bounds(series)

    return series[
        (series < lower) |
        (series > upper)
    ]
```

---

## Remove IQR Outliers

```python
def remove_iqr_outliers(df, column):
    lower, upper = iqr_bounds(
        df[column]
    )

    return df[
        df[column].between(
            lower,
            upper
        )
    ].copy()
```

---

## Cap Outliers

```python
def cap_outliers(
    df,
    column,
    lower_quantile=0.01,
    upper_quantile=0.99
):
    result = df.copy()

    lower = result[column].quantile(
        lower_quantile
    )

    upper = result[column].quantile(
        upper_quantile
    )

    result[column] = result[column].clip(
        lower=lower,
        upper=upper
    )

    return result
```

---

# 📋 Automated Outlier Report

```python
def create_outlier_report(df):
    numerical = df.select_dtypes(
        include="number"
    )

    rows = []

    for column in numerical.columns:
        series = numerical[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (
            (series < lower) |
            (series > upper)
        )

        rows.append({
            "feature": column,
            "count": len(series),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": mask.sum(),
            "outlier_percentage": (
                mask.mean() * 100
            )
        })

    return pd.DataFrame(rows)
```

Usage:

```python
report = create_outlier_report(
    df
)

print(report)
```

---

# 📊 Visual Outlier Report

```python
import matplotlib.pyplot as plt
import seaborn as sns


def plot_outlier_analysis(
    df,
    column
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    sns.histplot(
        data=df,
        x=column,
        kde=True,
        ax=axes[0]
    )

    axes[0].set_title(
        f"{column} Distribution"
    )

    sns.boxplot(
        x=df[column],
        ax=axes[1]
    )

    axes[1].set_title(
        f"{column} Box Plot"
    )

    plt.tight_layout()
    plt.show()
```

Usage:

```python
plot_outlier_analysis(
    df,
    "income"
)
```

---

# 🚀 End-to-End Example

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_dataset():
    rng = np.random.default_rng(42)

    income = rng.normal(
        loc=60000,
        scale=12000,
        size=200
    )

    # Add intentionally extreme observations
    income = np.concatenate([
        income,
        [180000, 220000, 250000]
    ])

    experience = rng.normal(
        loc=6,
        scale=2,
        size=len(income)
    )

    experience = np.clip(
        experience,
        0,
        None
    )

    return pd.DataFrame({
        "income": income,
        "experience": experience
    })


def detect_iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return (
        (series < lower) |
        (series > upper)
    )


def analyze_outliers(df):
    mask = detect_iqr_outliers(
        df["income"]
    )

    print("\nOutlier Analysis")
    print("=" * 50)

    print(
        "Total observations:",
        len(df)
    )

    print(
        "Potential outliers:",
        mask.sum()
    )

    print(
        "Outlier percentage:",
        f"{mask.mean() * 100:.2f}%"
    )

    print("\nPotential outliers:")
    print(
        df.loc[
            mask
        ]
    )


def visualize(df):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    sns.histplot(
        data=df,
        x="income",
        kde=True,
        ax=axes[0]
    )

    axes[0].set_title(
        "Income Distribution"
    )

    sns.boxplot(
        x=df["income"],
        ax=axes[1]
    )

    axes[1].set_title(
        "Income Box Plot"
    )

    plt.tight_layout()
    plt.show()


def main():
    df = create_dataset()

    print("Dataset:")
    print(df.head())

    analyze_outliers(df)

    visualize(df)


if __name__ == "__main__":
    main()
```

---

# 🔬 Advanced Example — Isolation Forest

For multivariate anomaly detection:

```python
import pandas as pd

from sklearn.ensemble import IsolationForest


def isolation_forest_detection(
    df,
    features,
    contamination=0.05
):
    data = df[features].dropna()

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    predictions = model.fit_predict(
        data
    )

    result = data.copy()

    result["outlier"] = (
        predictions == -1
    )

    return result
```

Usage:

```python
features = [
    "age",
    "income",
    "expenses"
]

result = isolation_forest_detection(
    df,
    features
)

print(
    result[
        result["outlier"]
    ]
)
```

---

# 🔬 Advanced Example — Local Outlier Factor

```python
from sklearn.neighbors import (
    LocalOutlierFactor
)


def lof_detection(
    df,
    features,
    n_neighbors=20,
    contamination=0.05
):
    data = df[features].dropna()

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )

    predictions = model.fit_predict(
        data
    )

    result = data.copy()

    result["outlier"] = (
        predictions == -1
    )

    return result
```

---

# 🧪 Comparing Before and After Treatment

Always validate the impact of treatment.

```python
before = df["income"].describe()

df["income_capped"] = (
    df["income"]
    .clip(
        lower=df["income"].quantile(0.01),
        upper=df["income"].quantile(0.99)
    )
)

after = df["income_capped"].describe()

print("Before:")
print(before)

print("\nAfter:")
print(after)
```

Also compare:

```python
print(
    "Original skew:",
    df["income"].skew()
)

print(
    "Capped skew:",
    df["income_capped"].skew()
)
```

---

# 🧠 Outlier Treatment Decision Matrix

| Situation                            | Recommended Starting Point            |
| ------------------------------------ | ------------------------------------- |
| Invalid measurement                  | Correct or mark as missing            |
| Data-entry error                     | Correct if source is known            |
| Legitimate rare event                | Usually keep                          |
| Strong right skew                    | Consider transformation               |
| Extreme but valid values             | Consider robust models/transformation |
| Need fixed sample size               | Consider capping                      |
| Small number of clearly invalid rows | Removal may be appropriate            |
| Multivariate anomaly                 | Isolation Forest / LOF / Mahalanobis  |
| Business-defined invalid range       | Domain rule                           |
| Unknown cause                        | Investigate before treatment          |

---

# ❌ Common Mistakes

## 1. Removing Every Outlier

Wrong:

```text
Outlier = Bad Data
```

Correct:

```text
Outlier = Observation Requiring Investigation
```

---

## 2. Using Only One Detection Method

Different methods detect different patterns.

Compare:

```text
IQR
Z-score
Visualization
Domain Rules
Multivariate Methods
```

---

## 3. Applying Z-Score to Strongly Skewed Data

Mean and standard deviation can themselves be heavily influenced by extreme values.

Use robust methods when appropriate.

---

## 4. Ignoring Domain Knowledge

A value may be statistically unusual but completely valid.

---

## 5. Removing Outliers Before Understanding Them

Always investigate first.

---

## 6. Calculating Thresholds on the Entire Dataset

For modeling workflows, this can leak information from validation/test data.

---

## 7. Treating Multivariate Problems as Univariate

A point can be normal in each feature but unusual in combination.

---

## 8. Ignoring the Target Variable

Extreme target values can strongly affect regression models.

---

## 9. Applying Blind Winsorization

Capping changes the data. It should have a clear rationale.

---

## 10. Forgetting to Document Treatment

Record:

```text
Method
Threshold
Reason
Number affected
Treatment
Validation result
```

---

# ✅ Best Practices

### 1. Understand the Data First

Know:

* Units
* Valid ranges
* Measurement process
* Business meaning

### 2. Visualize Before Removing

Use:

```text
Histogram
Box Plot
Scatter Plot
ECDF
```

### 3. Use Robust Statistics

Median and IQR are often less sensitive to extreme observations.

### 4. Combine Statistical and Domain Rules

Neither should automatically replace the other.

### 5. Distinguish Error from Rare Event

A rare event can contain valuable information.

### 6. Consider Model Sensitivity

Different models react differently to extreme values.

### 7. Preserve Original Data

Prefer creating transformed columns or copies during exploration.

### 8. Fit Modeling Preprocessing on Training Data

Avoid leakage.

### 9. Validate After Treatment

Compare:

```text
Before
vs
After
```

### 10. Monitor Outliers in Production

New outliers may indicate:

* Data drift
* Sensor problems
* Fraud
* New behavior
* Pipeline errors

---

# 🧪 Mini Projects

## 🏠 Project 1 — House Price Outliers

Analyze:

```text
Price
Area
Bedrooms
Bathrooms
Age
```

Tasks:

* Detect price outliers using IQR.
* Visualize price distribution.
* Investigate unusually large houses.
* Compare before/after log transformation.
* Determine whether extreme prices are valid.

---

## 💳 Project 2 — Fraud Detection

Analyze:

```text
Transaction Amount
Transaction Frequency
Account Age
Location Distance
```

Tasks:

* Detect univariate anomalies.
* Use Isolation Forest.
* Compare global and local anomalies.
* Investigate suspicious transactions.

---

## 🏭 Project 3 — Sensor Anomaly Detection

Analyze:

```text
Temperature
Pressure
Vibration
Voltage
```

Tasks:

* Apply IQR.
* Apply z-score.
* Detect multivariate anomalies.
* Compare Isolation Forest and LOF.

---

## 👨‍💼 Project 4 — Employee Salary Analysis

Analyze:

```text
Age
Experience
Salary
Bonus
Performance
```

Questions:

* Which features contain outliers?
* Are extreme salaries legitimate?
* Does experience explain extreme salary values?
* Does capping improve a regression model?

---

# 📝 Exercises

## 🌱 Beginner

1. Define an outlier.
2. Explain why outliers occur.
3. Create a box plot.
4. Calculate IQR manually.
5. Detect IQR outliers.
6. Calculate z-scores.
7. Explain the difference between an outlier and an error.
8. Create a histogram containing an extreme value.

---

## 🚀 Intermediate

1. Compare IQR and z-score detection.
2. Implement modified z-score.
3. Calculate outlier percentages.
4. Apply percentile-based detection.
5. Compare trimming and capping.
6. Analyze outliers with scatter plots.
7. Apply a log transformation.
8. Build an automated outlier report.
9. Compare distributions before and after treatment.
10. Analyze outliers in a real-world dataset.

---

## 🧠 Advanced

1. Implement Mahalanobis distance.
2. Apply Isolation Forest.
3. Apply Local Outlier Factor.
4. Compare multiple anomaly-detection methods.
5. Study local vs global outliers.
6. Build an outlier detection pipeline.
7. Investigate outlier drift over time.
8. Compare model performance before and after treatment.
9. Analyze multivariate anomalies.
10. Build a production-ready anomaly monitoring workflow.

---

# 📁 Project Structure

```text
06-Exploratory-Data-Analysis/
│
├── README.md
│
├── 01-Univariate-Analysis/
│   └── README.md
│
├── 02-Bivariate-Analysis/
│   └── README.md
│
├── 03-Multivariate-Analysis/
│   └── README.md
│
├── 04-Descriptive-Statistics/
│   └── README.md
│
├── 05-Correlation-Analysis/
│   └── README.md
│
├── 06-Distribution-Analysis/
│   └── README.md
│
├── 07-Outlier-Analysis/
│   ├── README.md
│   ├── iqr-method.py
│   ├── z-score.py
│   ├── modified-z-score.py
│   ├── percentile-method.py
│   ├── outlier-visualization.py
│   ├── mahalanobis-distance.py
│   ├── isolation-forest.py
│   ├── local-outlier-factor.py
│   └── outlier-report.py
│
└── 08-Data-Visualization/
    └── README.md
```

---

# 🔎 Outlier Analysis Checklist

Before making an outlier decision:

```text
☐ Understand the variable
☐ Check valid domain ranges
☐ Inspect missing values
☐ Visualize the distribution
☐ Create box plots
☐ Check scatter plots
☐ Calculate IQR
☐ Consider z-score where appropriate
☐ Consider robust methods
☐ Check multivariate relationships
☐ Investigate the cause
☐ Distinguish errors from rare events
☐ Use domain knowledge
☐ Choose treatment intentionally
☐ Avoid data leakage
☐ Compare before/after distributions
☐ Evaluate model impact
☐ Document the decision
☐ Monitor production behavior
```

---

# 🗺️ EDA Roadmap

```text
06-Exploratory-Data-Analysis/
│
├── 01-Univariate-Analysis
│       ↓
├── 02-Bivariate-Analysis
│       ↓
├── 03-Multivariate-Analysis
│       ↓
├── 04-Descriptive-Statistics
│       ↓
├── 05-Correlation-Analysis
│       ↓
├── 06-Distribution-Analysis
│       ↓
├── 07-Outlier-Analysis
│       ↓
├── 08-Data-Visualization
│       ↓
└── 09-EDA-Projects
```

---

# 🧠 Key Takeaways

```text
📌 An outlier is an unusually different observation.

📌 An outlier is not automatically an error.

📌 Outliers can be caused by errors, rare events, fraud, sensors, or population differences.

📌 Univariate outliers can be detected using IQR, z-score, modified z-score, and percentile methods.

📌 IQR is generally more robust to extreme values than mean/std-based methods.

📌 Z-score is most useful when its assumptions are reasonably appropriate.

📌 Modified z-score uses median and MAD for robust detection.

📌 Multivariate outliers require methods that consider relationships between features.

📌 Mahalanobis distance considers covariance.

📌 Isolation Forest isolates unusual observations efficiently.

📌 LOF detects observations that are unusual relative to their local neighborhood.

📌 Different ML algorithms have different sensitivities to outliers.

📌 Outlier treatment should be driven by context rather than a single threshold.

📌 Valid rare events should not automatically be deleted.

📌 Modeling-related thresholds should be learned from training data only.

📌 Always compare the dataset before and after treatment.

📌 Outlier analysis can also be useful for production monitoring and anomaly detection.
```

---

# 🚀 Next Step

After identifying and handling unusual observations, the next step is to communicate patterns effectively through visualization.

Continue to:

```text
06-Exploratory-Data-Analysis/
└── 08-Data-Visualization/
    └── README.md
```

You will explore:

```text
📊 DATA VISUALIZATION
        ↓
Matplotlib
        ↓
Seaborn
        ↓
Categorical Plots
        ↓
Numerical Plots
        ↓
Distribution Plots
        ↓
Relationship Plots
        ↓
Statistical Plots
        ↓
Subplots
        ↓
Advanced Visualization
        ↓
EDA STORYTELLING
```

---

# 🌟 Final Thought

> **Outlier analysis is not about deleting unusual data. It is about understanding why the data is unusual and making an informed decision.**

A professional workflow is:

```text
          DETECT
             ↓
        INVESTIGATE
             ↓
          VALIDATE
             ↓
           DECIDE
             ↓
           TREAT
             ↓
          VALIDATE
             ↓
          MONITOR
```

That mindset turns outlier handling from a mechanical cleaning step into a reliable part of **data quality, exploratory analysis, and machine learning engineering**.

---

# 👨‍💻 Author

**Kishor Patil**

Machine Learning | Python | Data Science | AI

🔗 GitHub: [Kishor055](https://github.com/Kishor055)

---

# 🤝 Contributing

Contributions are welcome!

If you find:

* 🐛 A mistake
* 💡 An unclear explanation
* 🧩 A missing concept
* 🐍 A Python improvement
* 📊 A better visualization
* 🤖 A useful anomaly-detection technique
* 📝 A useful exercise

feel free to open an issue or submit a pull request.

### Contribution Workflow

```bash
git clone https://github.com/Kishor055/Machine-Learning.git

cd Machine-Learning

git checkout -b feature/outlier-analysis

git add .

git commit -m "Improve outlier analysis module"

git push origin feature/outlier-analysis
```

Then open a Pull Request.

---

# ⭐ Support

If this repository helps you learn Machine Learning:

* ⭐ Star the repository
* 🍴 Fork the repository
* 🐛 Report issues
* 💡 Suggest improvements
* 🤝 Contribute examples
* 📢 Share it with other learners

---

# 📄 License

This project is intended for **educational and learning purposes**.

See the repository license for complete terms.

---

<div align="center">

### 🌱 Learn → Detect → Understand → Validate → Build → Improve 🚀

**Keep learning. Keep experimenting. Keep building.**

</div>

This module now gives the EDA section a strong progression:

**Univariate → Bivariate → Multivariate → Descriptive Statistics → Correlation → Distribution → Outliers → Data Visualization → EDA Projects**.
