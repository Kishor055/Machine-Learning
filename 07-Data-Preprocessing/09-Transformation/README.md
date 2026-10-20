# 🔄 Data Transformation in Python

🌱 **INSPECT → UNDERSTAND → TRANSFORM → VALIDATE → MODEL → EVALUATE**

Welcome to **Data Transformation**, part of the Data Preprocessing module in the Machine Learning learning path. This guide explains how to transform numerical features into representations that are more suitable for data analysis, visualization, statistical modeling, and machine learning.

Real-world datasets frequently contain highly skewed values, extreme differences in scale, nonlinear relationships, and features whose distributions make certain algorithms harder to train effectively. Appropriate transformations can help address these issues while preserving useful information.

This tutorial covers logarithmic transformations, square-root transformations, reciprocal transformations, Box-Cox, Yeo-Johnson, quantile transformations, polynomial features, discretization, cyclical transformations, Scikit-learn pipelines, and leakage-safe machine learning workflows.

## 📚 Table of Contents

1. [Introduction](#-introduction)
2. [What Is Data Transformation?](#-what-is-data-transformation)
3. [Why Transform Data?](#-why-transform-data)
4. [Transformation vs Scaling vs Normalization](#-transformation-vs-scaling-vs-normalization)
5. [Create a Sample Dataset](#-create-a-sample-dataset)
6. [Inspect Feature Distributions](#-inspect-feature-distributions)
7. [Logarithmic Transformation](#-logarithmic-transformation)
8. [Square-Root Transformation](#-square-root-transformation)
9. [Reciprocal Transformation](#-reciprocal-transformation)
10. [Box-Cox Transformation](#-box-cox-transformation)
11. [Yeo-Johnson Transformation](#-yeo-johnson-transformation)
12. [Quantile Transformation](#-quantile-transformation)
13. [Polynomial and Interaction Features](#-polynomial-and-interaction-features)
14. [Discretization and Binning](#-discretization-and-binning)
15. [Cyclical Feature Transformation](#-cyclical-feature-transformation)
16. [Custom Transformations with FunctionTransformer](#-custom-transformations-with-functiontransformer)
17. [Transformation with Scikit-learn Pipelines](#-transformation-with-scikit-learn-pipelines)
18. [Complete Machine Learning Example](#-complete-machine-learning-example)
19. [Inverse Transformations](#-inverse-transformations)
20. [Common Mistakes](#-common-mistakes)
21. [Best Practices](#-best-practices)
22. [Mini Projects](#-mini-projects)
23. [Practice Exercises](#-practice-exercises)
24. [Quick Revision](#-quick-revision)
25. [Project Structure](#-project-structure)
26. [Learning Roadmap](#-learning-roadmap)
27. [References](#-references)

---

## 🎯 Introduction

Consider a dataset containing annual salaries:

```python
import pandas as pd

df = pd.DataFrame({
    "Salary": [
        25000, 30000, 35000, 40000,
        50000, 65000, 80000, 500000
    ]
})

print(df["Salary"].describe())
```

Most salaries are relatively moderate, but one value is much larger than the others. This can produce a strongly right-skewed distribution.

A logarithmic transformation can reduce the numerical distance between small and large positive values:

```python
import numpy as np

df["Salary_Log"] = np.log1p(df["Salary"])

print(df)
```

The transformed column preserves the ordering of the original salaries while compressing their numerical range.

A transformation does not automatically make data better. Its usefulness depends on the feature, the model, and the intended analysis.

### Learning objectives

By completing this guide, you will learn to:

- Understand the purpose of numerical transformations.
- Identify skewness and nonlinear patterns.
- Apply logarithmic, square-root, and reciprocal transformations.
- Use Box-Cox and Yeo-Johnson power transformations.
- Understand quantile transformations and their limitations.
- Create polynomial and interaction features.
- Transform cyclical variables such as hours and months.
- Integrate transformations into Scikit-learn pipelines.
- Prevent preprocessing leakage.
- Evaluate transformations using appropriate validation techniques.

## 🧠 What Is Data Transformation?

**Data transformation** is the process of changing the representation, scale, shape, or structure of data to make it more suitable for analysis or modeling.

For numerical features, a transformation often applies a mathematical function to the observed values.

For example:

\[
x' = \log(1+x)
\]

where:

- \(x\) is the original value.
- \(x'\) is the transformed value.

Other transformations include square roots, power functions, quantile mappings, and polynomial feature construction.

### Example

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "Original": [1, 10, 100, 1000, 10000]
})

df["Log_Transformed"] = np.log10(df["Original"])

print(df)
```

The logarithm changes the spacing between values:

| Original | Base-10 logarithm |
|---:|---:|
| 1 | 0 |
| 10 | 1 |
| 100 | 2 |
| 1,000 | 3 |
| 10,000 | 4 |

Large multiplicative differences become smaller additive differences in the transformed space.

### Common goals

Transformations may be used to:

- Reduce skewness.
- Make variance more stable.
- Represent nonlinear relationships.
- Compress large positive values.
- Reduce the influence of extreme magnitudes.
- Construct useful model features.
- Represent cyclical variables appropriately.
- Make data more suitable for an algorithm's assumptions.

Not every transformation achieves every goal, and some transformations can introduce new limitations.

## 🚀 Why Transform Data?

### 1. Reduce positive skewness

Variables such as income, transaction amounts, and population often have long right tails.

A logarithm may make such distributions more symmetric.

### 2. Improve numerical behavior

Large differences in feature magnitudes can create optimization difficulties for some algorithms. Transformations can sometimes reduce these differences, although dedicated scaling may still be necessary.

### 3. Represent nonlinear relationships

For example, the relationship between advertising expenditure and sales may show diminishing returns. A logarithmic feature can sometimes represent that pattern more appropriately than the raw value.

### 4. Reduce the influence of extreme magnitudes

A logarithmic transformation compresses large positive values, potentially making a feature less dominated by its largest observations.

However, it does not remove incorrect records or guarantee robust model behavior.

### 5. Represent domain relationships

Some physical, economic, and scientific relationships are naturally multiplicative or exponential. A suitable transformation can simplify their representation.

### 6. Construct additional predictive features

Polynomial features and interaction terms can represent nonlinear effects that are not captured by a simple linear model.

**Important:** Use transformations because they make sense for the data or improve validation performance, not because every feature must be transformed.

## ⚖️ Transformation vs Scaling vs Normalization

These concepts are related but have different purposes.

| Technique | Main purpose | Example |
|---|---|---|
| Data transformation | Change the mathematical representation or distribution | `np.log1p(x)` |
| Standardization | Center features and scale them by standard deviation | `StandardScaler()` |
| Min-Max scaling | Map feature values to a specified range | `MinMaxScaler()` |
| Robust scaling | Scale using the median and interquartile range | `RobustScaler()` |
| Vector normalization | Scale individual samples to a chosen vector norm | `Normalizer()` |
| Quantile transformation | Map features using their empirical distribution | `QuantileTransformer()` |

For example, standardization changes the location and scale of a feature, while logarithmic transformation changes the relationship between its original and transformed values.

Some transformations, including Scikit-learn's `PowerTransformer`, can also standardize their output by default. Therefore, understand each estimator's behavior before combining it with another scaler.

## 🧪 Create a Sample Dataset

We will use numerical features containing different distributions.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

df = pd.DataFrame({
    "Income": rng.lognormal(mean=10, sigma=0.8, size=500),
    "Study_Hours": rng.gamma(shape=2.0, scale=2.0, size=500),
    "Temperature": rng.normal(loc=25, scale=5, size=500)
})

print(df.head())
print(df.describe())
```

The sample features have different characteristics:

- `Income` contains positive, right-skewed values.
- `Study_Hours` is nonnegative and may be positively skewed.
- `Temperature` can contain values below and above zero.

Different features may require different transformations. The same mathematical operation should not be applied indiscriminately to every column.

## 🔍 Inspect Feature Distributions

Before transforming a feature, examine its distribution and range.

### 1. Descriptive statistics

```python
print(df.describe())
```

Useful statistics include:

- Mean.
- Median.
- Standard deviation.
- Minimum and maximum.
- Quartiles.
- Interquartile range.

A large gap between the mean and median can indicate skewness, although it is not a complete description of the distribution.

### 2. Calculate skewness

```python
print(df.skew(numeric_only=True))
```

Pandas' skewness calculation helps identify asymmetry.

Positive skewness generally indicates a longer right tail, while negative skewness indicates a longer left tail. The statistic should be interpreted together with visual inspection and domain knowledge.

### 3. Plot a histogram

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.hist(df["Income"], bins=30, edgecolor="black")
plt.title("Income Before Transformation")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
```

### 4. Compare a transformed distribution

```python
plt.figure(figsize=(8, 5))
plt.hist(np.log1p(df["Income"]), bins=30, edgecolor="black")
plt.title("Income After Log Transformation")
plt.xlabel("log(1 + Income)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
```

A transformed distribution should be evaluated rather than assumed to be normal or otherwise ideal.

## 📉 Logarithmic Transformation

A logarithmic transformation applies a logarithm to each value.

Common forms include:

\[
x' = \ln(x)
\]

and

\[
x' = \ln(1+x)
\]

The natural logarithm is represented by \(\ln(x)\).

### When is it useful?

Logarithmic transformations may be useful for:

- Income and salary.
- Property prices.
- Sales and transaction amounts.
- Population counts.
- Highly skewed positive features.
- Features with multiplicative relationships.

### 1. Natural logarithm

```python
import numpy as np

values = np.array([1, 10, 100, 1000])

transformed = np.log(values)

print(transformed)
```

**Constraint:** `np.log(x)` requires positive values for finite real-valued results.

### 2. Log base 10

```python
transformed = np.log10(values)

print(transformed)
```

Base 10 is sometimes useful for data spanning several orders of magnitude.

### 3. `log1p()` transformation

```python
values = np.array([0, 1, 10, 100, 1000])

transformed = np.log1p(values)

print(transformed)
```

`np.log1p(x)` calculates \(\ln(1+x)\). It supports zero and nonnegative values, making it convenient for counts and amounts that may legitimately be zero.

### 4. Apply a logarithm to a Pandas column

```python
df["Income_Log"] = np.log1p(df["Income"])
```

### 5. Compare statistics

```python
comparison = pd.DataFrame({
    "Original": df["Income"],
    "Log_Transformed": np.log1p(df["Income"])
})

print(comparison.describe())
```

### Advantages

- Compresses large positive values.
- Often reduces positive skewness.
- Can simplify some multiplicative relationships.
- Preserves the order of positive observations.

### Limitations

- The interpretation of a unit change becomes nonlinear.
- Ordinary logarithms cannot handle negative inputs as real-valued numbers.
- A logarithm does not eliminate all outlier or data-quality problems.
- It does not guarantee a normal distribution.

**Best practice:** Check the minimum value before applying a logarithm. Never use `log1p()` to conceal negative values that violate the feature's data definition.

## 📐 Square-Root Transformation

The square-root transformation is:

\[
x' = \sqrt{x}
\]

It compresses large nonnegative values, generally less aggressively than a logarithm.

### Example

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "Count": [0, 1, 4, 9, 16, 25, 100]
})

df["Count_Sqrt"] = np.sqrt(df["Count"])

print(df)
```

### When is it useful?

Square-root transformations can be considered for:

- Nonnegative counts.
- Certain positively skewed features.
- Variance-stabilizing transformations in some statistical settings.

### Advantages

- Simple to calculate.
- Supports zero values.
- Can reduce the skewness of some count-like variables.

### Limitations

- Real-valued square roots of negative observations are not defined.
- It may not sufficiently reduce severe skewness.
- It is not a universal solution for count data.

### Example with study hours

```python
df["Study_Hours_Sqrt"] = np.sqrt(df["Study_Hours"])
```

Before applying the operation, validate that the values are nonnegative or handle invalid observations according to the feature's domain rules.

## 🔁 Reciprocal Transformation

The reciprocal transformation is:

\[
x' = \frac{1}{x}
\]

It changes large positive values into small positive values and small positive values into large values.

### Example

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "Distance": [1, 2, 4, 5, 10]
})

df["Distance_Reciprocal"] = 1 / df["Distance"]

print(df)
```

### When is it useful?

A reciprocal transformation may be useful when:

- A relationship is approximately inverse.
- The feature has a meaningful reciprocal interpretation.
- Domain knowledge supports a relationship such as rate per unit distance.

### Important limitations

- Zero causes division by zero.
- Values near zero can become extremely large.
- Negative inputs produce negative reciprocals.
- The transformation can strongly distort the original distribution.

Do not add an arbitrary constant to avoid division by zero without understanding what the modified feature means.

## 🧮 Box-Cox Transformation

The Box-Cox transformation is a family of power transformations that can reduce skewness and make variance more stable.

A common mathematical form is:

\[
x' =
\begin{cases}
\frac{x^\lambda-1}{\lambda}, & \lambda \ne 0 \\[6pt]
\ln(x), & \lambda = 0
\end{cases}
\]

The parameter \(\lambda\) controls the transformation.

Scikit-learn's `PowerTransformer` can estimate a suitable parameter from the training data.

### Constraint

**Box-Cox requires strictly positive input values.**

If the feature contains zero or negative values, validate the data and select a suitable alternative rather than blindly applying Box-Cox.

### Example

```python
import numpy as np
from sklearn.preprocessing import PowerTransformer

values = np.array([
    [1.0],
    [2.0],
    [4.0],
    [10.0],
    [20.0],
    [50.0],
    [100.0]
])

transformer = PowerTransformer(
    method="box-cox",
    standardize=False
)

transformed = transformer.fit_transform(values)

print(transformed)
print("Estimated lambda:", transformer.lambdas_)
```

`standardize=False` prevents the transformer from applying its additional zero-mean, unit-variance scaling stage.

### Advantages

- Estimates a power transformation from observed data.
- Can reduce skewness and stabilize variance.
- Useful for appropriate positive numerical features.

### Limitations

- Requires strictly positive values.
- Assumptions and results depend on the data distribution.
- It may not make every feature approximately normal.
- The estimated parameter must be fitted only on training data during model evaluation.

Use Box-Cox when its input restrictions and statistical behavior fit the feature.

## 🔄 Yeo-Johnson Transformation

Yeo-Johnson is another power transformation. Unlike Box-Cox, it can accommodate zero and negative input values.

It is provided by Scikit-learn's `PowerTransformer`.

### Example

```python
import numpy as np
from sklearn.preprocessing import PowerTransformer

values = np.array([
    [-10.0],
    [-3.0],
    [-1.0],
    [0.0],
    [1.0],
    [3.0],
    [10.0],
    [50.0]
])

transformer = PowerTransformer(
    method="yeo-johnson",
    standardize=True
)

transformed = transformer.fit_transform(values)

print(transformed)
print("Estimated lambda:", transformer.lambdas_)
```

### Box-Cox vs Yeo-Johnson

| Property | Box-Cox | Yeo-Johnson |
|---|---|---|
| Positive inputs | Supported | Supported |
| Zero inputs | Not supported | Supported |
| Negative inputs | Not supported | Supported |
| Power parameter | Estimated from the data | Estimated from the data |
| Standardizes output by default in Scikit-learn | Yes | Yes |

### Why use Yeo-Johnson?

It can be a useful default candidate when a numerical feature includes zero or negative values and a power transformation is appropriate.

However, do not assume it will automatically produce an ideal distribution. Check the transformed data and validate its impact on the model.

## 📊 Quantile Transformation

`QuantileTransformer` changes feature values according to their empirical distribution.

It can map each feature toward either:

- A uniform distribution.
- A normal distribution.

The transformation is nonlinear and is fitted from the observed training values. It can reduce the influence of extreme marginal values, but it may also distort distances and correlations. See the [official Scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html).

### 1. Transform to a uniform distribution

```python
import numpy as np
from sklearn.preprocessing import QuantileTransformer

values = np.array([
    [1.0],
    [2.0],
    [4.0],
    [10.0],
    [20.0],
    [100.0],
    [1000.0]
])

transformer = QuantileTransformer(
    n_quantiles=7,
    output_distribution="uniform",
    random_state=42
)

transformed = transformer.fit_transform(values)

print(transformed)
```

### 2. Transform to a normal distribution

```python
transformer = QuantileTransformer(
    n_quantiles=7,
    output_distribution="normal",
    random_state=42
)

transformed = transformer.fit_transform(values)

print(transformed)
```

### When can it be useful?

Quantile transformation may be considered when:

- A feature has a highly irregular distribution.
- A monotonic rank-based transformation is acceptable.
- Reducing the influence of extreme marginal observations is useful.
- A more uniform or normal-like feature distribution is desired.

### Limitations

- It can alter linear correlations.
- It can distort distances between observations.
- Values outside the fitted range are mapped to output bounds.
- Ties and small sample sizes can reduce useful resolution.
- It may be inappropriate when original numeric distances have important meaning.

Use it inside a fitted pipeline when working with a machine learning model.

## 🧱 Polynomial and Interaction Features

Not every transformation changes the distribution of a feature. Some transformations create additional features so a model can represent nonlinear relationships.

### 1. Polynomial features

A polynomial transformation of a feature \(x\) may create:

\[
x,\quad x^2,\quad x^3
\]

These features can help a linear model represent curved relationships.

Example:

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.array([
    [1],
    [2],
    [3],
    [4]
])

transformer = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_transformed = transformer.fit_transform(X)

print(X_transformed)
```

Output:

```text
[[ 1.  1.]
 [ 2.  4.]
 [ 3.  9.]
 [ 4. 16.]]
```

The output contains the original feature and its square.

### 2. Interaction features

For two variables \(x_1\) and \(x_2\), an interaction feature is:

\[
x_1x_2
\]

An interaction can represent a relationship in which the effect of one variable depends on another.

Example:

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.array([
    [2, 3],
    [4, 5],
    [6, 7]
])

transformer = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_transformed = transformer.fit_transform(X)

print(transformer.get_feature_names_out(["Hours", "Attendance"]))
print(X_transformed)
```

With two input features, degree two, and `include_bias=False`, the generated columns represent the original variables, their squares, and their pairwise interaction.

### Advantages

- Helps linear models represent nonlinear relationships.
- Creates interpretable interaction terms when selected carefully.
- Can improve performance when the true feature relationship is nonlinear.

### Limitations

- The number of features can increase rapidly.
- Higher-degree features may overfit.
- Polynomial features can become strongly correlated.
- Numerical magnitudes can grow significantly.
- Feature construction must be evaluated using validation data.

Keep feature expansion inside a pipeline when selecting or tuning polynomial degrees using cross-validation.

## 📦 Discretization and Binning

Discretization converts a continuous numerical feature into a set of intervals or bins.

For example, age values can be grouped into predefined ranges.

### 1. Manual binning with Pandas

```python
import pandas as pd

df = pd.DataFrame({
    "Age": [18, 22, 28, 35, 42, 55, 67]
})

bins = [0, 18, 30, 45, 60, 120]

labels = [
    "0-18",
    "19-30",
    "31-45",
    "46-60",
    "61+"
]

df["Age_Group"] = pd.cut(
    df["Age"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print(df)
```

The interval definitions and labels are domain-specific. Confirm that they match the intended boundary conventions of your application.

### 2. Equal-width binning

Equal-width bins divide a specified value range into intervals of equal width.

```python
df["Age_Equal_Width"] = pd.cut(
    df["Age"],
    bins=4
)
```

### 3. Quantile-based binning

Quantile-based bins aim to place approximately equal numbers of observations into each bin.

```python
df["Age_Quantile_Bin"] = pd.qcut(
    df["Age"],
    q=4,
    duplicates="drop"
)
```

Tied observations may prevent the requested number of distinct quantile boundaries from being created. `duplicates="drop"` allows repeated boundaries to be removed.

### 4. Scikit-learn discretization

```python
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

X = pd.DataFrame({
    "Income": [20000, 25000, 30000, 50000, 75000, 120000]
})

transformer = KBinsDiscretizer(
    n_bins=3,
    encode="ordinal",
    strategy="quantile"
)

X_binned = transformer.fit_transform(X)

print(X_binned)
```

This creates ordinal bin codes. If the bin boundaries are learned from data, fit the discretizer on training data only.

### Advantages

- Simplifies complicated numerical relationships.
- Can make results easier to communicate.
- May capture threshold effects.
- Can be useful for models that benefit from categorical representations.

### Limitations

- Loses information about differences within each bin.
- Depends on the choice of bin boundaries.
- May create unstable results around boundary values.
- Can reduce useful predictive information.

Binning should reflect domain meaning or demonstrate value during validation.

## 🕒 Cyclical Feature Transformation

Some numerical features repeat periodically.

Examples include:

- Hour of day.
- Day of week.
- Month of year.
- Direction expressed as an angle.

A raw representation creates artificial boundaries. For example, hours `23` and `0` are adjacent in time but numerically far apart.

A sine/cosine transformation can represent the cycle continuously.

### Formula

For a cyclical feature \(x\) with period \(P\):

\[
x_{\sin} = \sin\left(\frac{2\pi x}{P}\right)
\]

\[
x_{\cos} = \cos\left(\frac{2\pi x}{P}\right)
\]

### Example: Hour of day

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "Hour": [0, 6, 12, 18, 23]
})

period = 24

df["Hour_Sin"] = np.sin(
    2 * np.pi * df["Hour"] / period
)

df["Hour_Cos"] = np.cos(
    2 * np.pi * df["Hour"] / period
)

print(df)
```

The pair of sine and cosine features represents the hour as a point on a circle.

### Example: Day of week

For a convention where Monday is `0` and Sunday is `6`:

```python
df["Day_Of_Week"] = [0, 1, 2, 3, 4]

period = 7

df["Day_Sin"] = np.sin(
    2 * np.pi * df["Day_Of_Week"] / period
)

df["Day_Cos"] = np.cos(
    2 * np.pi * df["Day_Of_Week"] / period
)
```

Always use the correct period and category convention for the feature.

## 🛠️ Custom Transformations with FunctionTransformer

Scikit-learn's `FunctionTransformer` wraps a Python function as a transformer, making a custom transformation compatible with preprocessing pipelines.

See the [official FunctionTransformer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html).

### Example: Logarithmic transformation

```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(
    np.log1p,
    validate=True
)

X = np.array([
    [0],
    [1],
    [10],
    [100]
])

X_transformed = log_transformer.transform(X)

print(X_transformed)
```

`FunctionTransformer` can also be used inside a `Pipeline`.

### Example: Custom square transformation

```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer

square_transformer = FunctionTransformer(
    np.square,
    validate=True
)

X = np.array([
    [2],
    [3],
    [4]
])

print(square_transformer.transform(X))
```

### Important considerations

- Validate the input domain of the mathematical function.
- Make sure missing-value imputation occurs at the appropriate point in the pipeline.
- Test the transformation with edge cases, including zero and negative values.
- Use a reproducible function that can also be applied during inference.
- Avoid non-serializable lambdas if the transformer must be saved and reused.

Unlike `PowerTransformer` and `QuantileTransformer`, a simple `FunctionTransformer` around a fixed mathematical function generally does not need to estimate distribution parameters. Its function should still be applied only where its input assumptions are satisfied.

## 🔗 Transformation with Scikit-learn Pipelines

A pipeline combines transformations and a model into one fitted workflow.

This is especially useful when a transformation learns parameters from data, such as:

- Power transformations.
- Quantile transformations.
- Polynomial preprocessing combined with feature selection.
- Imputation followed by transformations.
- Feature scaling and model training.

### Example: Power transformation and regression

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import Ridge

model = Pipeline([
    ("transformation", PowerTransformer(method="yeo-johnson")),
    ("regressor", Ridge(alpha=1.0))
])
```

This example defines a pipeline. Fit it on the training features and target before evaluating it on held-out data.

### Why pipelines matter

A pipeline allows data-dependent transformations to be fitted using training data and then reused on validation, test, and production inputs.

That prevents a common form of **data leakage**, where the evaluation data influences transformation parameters.

When multiple feature types are present, use `ColumnTransformer` to apply different transformations to different columns instead of indiscriminately transforming every feature.

## 🤖 Complete Machine Learning Example

The following end-to-end example uses:

- A synthetic regression dataset.
- A right-skewed income feature.
- A numerical experience feature.
- A nominal department feature.
- A training/test split.
- Numerical imputation.
- Yeo-Johnson transformation.
- One-Hot Encoding.
- Ridge Regression.
- Regression evaluation metrics.

Save this as `transformation.py`.

```python
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer


def create_dataset(
    n_samples: int = 500,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Create a reproducible synthetic regression dataset."""

    rng = np.random.default_rng(random_state)

    income = rng.lognormal(
        mean=10,
        sigma=0.8,
        size=n_samples
    )

    experience = rng.uniform(
        low=0,
        high=20,
        size=n_samples
    )

    department = rng.choice(
        ["IT", "Finance", "HR"],
        size=n_samples
    )

    noise = rng.normal(
        loc=0,
        scale=5,
        size=n_samples
    )

    target = (
        2.0 * np.log1p(income)
        + 1.5 * experience
        + noise
    )

    X = pd.DataFrame({
        "Income": income,
        "Experience": experience,
        "Department": department
    })

    y = pd.Series(target, name="Target")

    # Introduce missing feature values for demonstration.
    income_missing = rng.choice(
        n_samples,
        size=int(n_samples * 0.05),
        replace=False
    )

    department_missing = rng.choice(
        n_samples,
        size=int(n_samples * 0.03),
        replace=False
    )

    X.loc[income_missing, "Income"] = np.nan
    X.loc[department_missing, "Department"] = None

    return X, y


def build_model() -> Pipeline:
    """Build a leakage-safe preprocessing and regression pipeline."""

    numeric_features = ["Income", "Experience"]
    categorical_features = ["Department"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "transformer",
            PowerTransformer(
                method="yeo-johnson",
                standardize=True
            )
        )
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0))
    ])


def main() -> None:
    X, y = create_dataset()

    print("DATASET SHAPE")
    print(X.shape)

    print("\nMISSING VALUES BEFORE PREPROCESSING")
    print(X.isna().sum())

    print("\nINCOME SKEWNESS BEFORE TRANSFORMATION")
    print(X["Income"].skew())

    # Split before fitting any learned preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = build_model()

    # Fit imputation, transformation, encoding, and regression
    # using only training observations.
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nTEST RESULTS")
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
    print("R2:", r2_score(y_test, predictions))

    # Inspect the training-fitted numerical transformer.
    fitted_preprocessor = model.named_steps["preprocessor"]
    fitted_numeric_pipeline = (
        fitted_preprocessor.named_transformers_["numeric"]
    )
    fitted_transformer = (
        fitted_numeric_pipeline.named_steps["transformer"]
    )

    print("\nEstimated power parameters:")
    print(fitted_transformer.lambdas_)


if __name__ == "__main__":
    main()
```

### What this project demonstrates

- Generating reproducible synthetic data.
- Creating a positively skewed income feature.
- Introducing missing numerical and categorical values.
- Splitting data before fitting learned preprocessing.
- Imputing missing values.
- Applying Yeo-Johnson transformation to numerical features.
- Encoding nominal categories separately.
- Training a regression model through a pipeline.
- Evaluating results using MAE, RMSE, and \(R^2\).
- Inspecting learned transformation parameters.

The dataset is synthetic and designed for demonstrating implementation rather than proving that transformation improves predictive accuracy.

To assess whether a transformation helps, compare a justified baseline with transformed alternatives using a consistent validation strategy.

## 🔁 Inverse Transformations

Some transformations support reversing an encoded value back to its original numerical space.

An inverse transformation can be useful when results must be communicated in the original measurement units.

### Example: PowerTransformer

```python
import numpy as np
from sklearn.preprocessing import PowerTransformer

X = np.array([
    [1.0],
    [2.0],
    [5.0],
    [10.0],
    [20.0]
])

transformer = PowerTransformer(
    method="yeo-johnson"
)

X_transformed = transformer.fit_transform(X)

X_recovered = transformer.inverse_transform(X_transformed)

print("Original:")
print(X)

print("\nTransformed:")
print(X_transformed)

print("\nRecovered:")
print(X_recovered)
```

The recovered values should closely match the input values within floating-point precision.

### Example: QuantileTransformer

```python
from sklearn.preprocessing import QuantileTransformer

transformer = QuantileTransformer(
    n_quantiles=5,
    output_distribution="normal",
    random_state=42
)

X_transformed = transformer.fit_transform(X)

X_recovered = transformer.inverse_transform(X_transformed)

print(X_recovered)
```

Quantile transformations use estimated distributions, so inverse transformation may not perfectly recover every original observation, particularly when multiple observations have tied values or quantile interpolation causes loss of detail.

### Logarithmic inverse

For:

\[
x' = \ln(1+x)
\]

the inverse is:

\[
x = e^{x'}-1
\]

Python example:

```python
import numpy as np

original = np.array([0, 1, 10, 100])

transformed = np.log1p(original)

recovered = np.expm1(transformed)

print(recovered)
```

Always apply the inverse corresponding to the original transformation. Do not use an unrelated inverse formula simply because the output is numerical.

## ❌ Common Mistakes

### 1. Applying a logarithm to negative values

`np.log()` and `np.log1p()` have specific mathematical domains.

**Solution:** Check the feature range and choose a suitable transformation, such as Yeo-Johnson, when appropriate.

### 2. Assuming every transformation produces normal data

A transformed distribution may remain skewed or multimodal.

**Solution:** Inspect the result and validate the transformation's usefulness.

### 3. Transforming every numerical feature

Some features already have suitable distributions, and some have a meaningful original scale.

**Solution:** Select transformations based on feature semantics, distribution, and model requirements.

### 4. Using Box-Cox on data containing zero or negative values

Box-Cox requires strictly positive input.

**Solution:** Validate the domain or consider Yeo-Johnson where appropriate.

### 5. Confusing transformation with scaling

A logarithm changes the relationship between values, while standardization primarily changes their location and scale.

**Solution:** Understand each estimator's purpose and output behavior.

### 6. Fitting transformation parameters on the full dataset

Learned transformations can leak information from the test partition.

**Solution:** Split first and fit learned transformations inside a training pipeline.

### 7. Ignoring the effects on correlations

Quantile and other nonlinear transformations can change feature relationships.

**Solution:** Reassess correlations, model interpretation, and validation performance after transforming features.

### 8. Overusing polynomial features

Higher-degree transformations can create many correlated features and encourage overfitting.

**Solution:** Begin with simpler transformations and evaluate feature complexity.

### 9. Binning without considering meaningful boundaries

Arbitrary binning may discard valuable information.

**Solution:** Use domain-informed boundaries or justify learned bins with validation.

### 10. Transforming target values without handling predictions correctly

A transformed regression target requires inverse transformation of predictions when results must be returned in the original target units.

**Solution:** Use the correct inverse operation or a suitable target-transformation wrapper.

## 🌟 Best Practices

- Inspect feature distributions before selecting a transformation.
- Understand the mathematical domain and limitations of every operation.
- Separate data validation from mathematical transformation.
- Keep original columns when traceability is important.
- Use logarithms for suitable positive or nonnegative skewed features.
- Consider Box-Cox only for strictly positive inputs.
- Consider Yeo-Johnson when nonpositive values are present.
- Use quantile transformations only when their nonlinear effects are acceptable.
- Introduce polynomial features only when they have a reasonable modeling justification.
- Treat cyclical features with transformations that preserve their periodic structure.
- Fit learned transformations on training data only.
- Use `Pipeline` and `ColumnTransformer` for repeatable preprocessing.
- Compare transformed and untransformed baselines using consistent validation.
- Check for numerical instability and invalid transformed values.
- Preserve the ability to interpret or recover predictions in their original units.
- Monitor feature distributions after deployment.

## 💡 Mini Projects

### Project 1: Income Distribution Transformation

Create a dataset containing salaries and transaction amounts.

Tasks:

1. Inspect histograms and skewness.
2. Apply `log1p()`.
3. Apply Yeo-Johnson.
4. Compare the resulting distributions.
5. Explain which transformation is suitable for the selected feature.

### Project 2: Student Performance Transformation

Create a dataset with study hours, marks, and attendance.

Tasks:

1. Identify skewed numerical features.
2. Apply a suitable transformation.
3. Compare summary statistics.
4. Build a regression model with and without the transformation.
5. Evaluate both using the same validation strategy.

### Project 3: Nonlinear Regression Features

Create a dataset where a target depends on both linear and nonlinear relationships.

Tasks:

1. Train a linear regression baseline.
2. Add polynomial features.
3. Compare validation performance.
4. Inspect the increase in feature count.
5. Explain the trade-off between flexibility and overfitting.

### Project 4: Cyclical Feature Engineering

Create a dataset containing hour of day and daily demand.

Tasks:

1. Represent the hour as a numerical feature.
2. Create sine and cosine features.
3. Compare both representations in a regression model.
4. Inspect how the transformed representation treats hours near midnight.
5. Evaluate the models using a suitable split.

### Project 5: Compare Power and Quantile Transformers

Create a dataset with different numerical distributions.

Tasks:

1. Inspect the feature distributions.
2. Apply Yeo-Johnson.
3. Apply a quantile transformation.
4. Compare correlations before and after transformation.
5. Evaluate their effects on a model.
6. Document limitations and assumptions.

## 📝 Practice Exercises

1. Apply `np.log()` to a positive numerical array.
2. Compare `np.log()` and `np.log1p()`.
3. Apply a square-root transformation to nonnegative counts.
4. Explain why reciprocal transformations are unstable near zero.
5. Apply Box-Cox to strictly positive data.
6. Apply Yeo-Johnson to a feature containing negative values.
7. Use `QuantileTransformer` with uniform output.
8. Use `QuantileTransformer` with normal output.
9. Generate degree-two polynomial features.
10. Create interaction features between two numerical variables.
11. Convert ages to categorical bins using `pd.cut()`.
12. Use `KBinsDiscretizer` for quantile-based binning.
13. Encode an hour-of-day feature using sine and cosine.
14. Create a custom transformation using `FunctionTransformer`.
15. Build a pipeline that combines transformation, imputation, and regression.
16. Explain why transformation parameters must be fitted without using test data.
17. Compare a baseline model and a transformed-feature model.
18. Demonstrate how to invert a logarithmic or power transformation.

## ⚡ Quick Revision

| Task | Method |
|---|---|
| Natural logarithm | `np.log(x)` |
| Logarithm that supports zero | `np.log1p(x)` |
| Reverse `log1p()` | `np.expm1(x)` |
| Square root | `np.sqrt(x)` |
| Reciprocal | `1 / x`, with zero checks |
| Box-Cox transformation | `PowerTransformer(method="box-cox")` |
| Yeo-Johnson transformation | `PowerTransformer(method="yeo-johnson")` |
| Uniform quantile transformation | `QuantileTransformer(output_distribution="uniform")` |
| Normal quantile transformation | `QuantileTransformer(output_distribution="normal")` |
| Polynomial features | `PolynomialFeatures()` |
| Binning | `pd.cut()` or `pd.qcut()` |
| Scikit-learn binning | `KBinsDiscretizer()` |
| Custom mathematical operation | `FunctionTransformer()` |
| Apply different transformations by column | `ColumnTransformer()` |
| Combine transformations and a model | `Pipeline()` |
| Inspect skewness | `df.skew(numeric_only=True)` |
| Compare distributions | `matplotlib.pyplot.hist()` |

## 📂 Project Structure

Recommended repository structure:

```text
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
├── 04-Normalization/
│   └── README.md
│
├── 05-Label-Encoding/
│   └── README.md
│
├── 06-One-Hot-Encoding/
│   └── README.md
│
├── 07-Ordinal-Encoding/
│   └── README.md
│
├── 08-Imputation/
│   └── README.md
│
└── 09-Transformation/
    ├── README.md
    ├── transformation.py
    ├── log_transformation.py
    ├── power_transformation.py
    ├── quantile_transformation.py
    ├── polynomial_features.py
    ├── discretization.py
    ├── cyclical_transformation.py
    └── transformation_pipeline.py
```

| File | Purpose |
|---|---|
| `README.md` | Theory, methods, examples, and best practices |
| `transformation.py` | End-to-end machine learning example |
| `log_transformation.py` | Logarithmic and square-root transformations |
| `power_transformation.py` | Box-Cox and Yeo-Johnson |
| `quantile_transformation.py` | Quantile-based transformations |
| `polynomial_features.py` | Polynomial and interaction features |
| `discretization.py` | Continuous feature binning |
| `cyclical_transformation.py` | Sine/cosine feature engineering |
| `transformation_pipeline.py` | Leakage-safe transformation workflow |

Create the individual Python files as you expand the topic. The examples in this README can serve as the starting point for each script.

## 🗺️ Learning Roadmap

Follow this progression to develop practical data transformation skills:

1. Inspect numerical features and identify distribution characteristics.
2. Understand transformation, scaling, and normalization.
3. Practice logarithmic and square-root transformations.
4. Learn reciprocal, Box-Cox, and Yeo-Johnson transformations.
5. Explore quantile transformations and their effects on relationships.
6. Create polynomial and interaction features.
7. Learn discretization and cyclical feature engineering.
8. Integrate transformations with imputation and scaling.
9. Build reproducible Scikit-learn pipelines.
10. Compare alternatives using appropriate validation.
11. Monitor feature distributions and transformation behavior in production.

## 🎯 Key Takeaways

- Data transformations change the mathematical representation of features.
- Logarithmic transformations can compress large positive values and reduce skewness.
- Square-root transformations are useful candidates for some nonnegative features.
- Reciprocal transformations require special care around zero.
- Box-Cox requires strictly positive values, while Yeo-Johnson supports negative and zero values.
- Quantile transformations can reshape distributions but may distort distances and correlations.
- Polynomial features can capture nonlinear relationships but may increase overfitting risk.
- Cyclical transformations preserve periodic relationships between values near a cycle's boundaries.
- Learned transformations must be fitted using training data only.
- The effectiveness of a transformation should be evaluated rather than assumed.

## 👨‍💻 Author

**Kishor Patil**

Python | Data Science | Machine Learning | Artificial Intelligence

GitHub: [Kishor055](https://github.com/Kishor055)

Repository: [Machine-Learning](https://github.com/Kishor055/Machine-Learning)

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Add examples, improvements, or corrections.
4. Test your code.
5. Submit a pull request with a clear description.

## ⭐ Support

If this guide helps you learn data preprocessing and machine learning, consider starring the repository on GitHub.

**Keep learning, keep building, and keep growing!** 🌱

## 📖 References

- [Scikit-learn: Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Scikit-learn: PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)
- [Scikit-learn: QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)
- [Scikit-learn: FunctionTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
- [Scikit-learn: PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- [Scikit-learn: KBinsDiscretizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html)
- [Scikit-learn: Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [NumPy: Mathematical functions](https://numpy.org/doc/stable/reference/routines.math.html)

---

*Part of the Machine Learning learning path: Data Collection → Data Cleaning → Exploratory Data Analysis → Data Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment.*
