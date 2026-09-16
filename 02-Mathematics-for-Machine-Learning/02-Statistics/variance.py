"""
Variance for Machine Learning
=============================

File:
02-Mathematics-for-Machine-Learning/02-Statistics/variance.py

Purpose:
Learn how variance measures the dispersion of data around
its mean and how variance is used in statistics, probability,
data analysis, and Machine Learning.

Topics Covered:
1. What is variance?
2. Why variance matters
3. Population variance
4. Sample variance
5. Variance and standard deviation
6. Squared deviations
7. Manual calculation
8. Step-by-step variance
9. Population vs sample variance
10. NumPy implementation
11. Pandas implementation
12. Low vs high variance
13. Effect of outliers
14. Variance and units
15. Translation invariance
16. Scaling behavior
17. Zero variance
18. Z-score relationship
19. Coefficient of variation
20. Covariance relationship
21. Correlation relationship
22. Feature selection
23. Variance threshold
24. Low-variance features
25. Data preprocessing
26. Standardization
27. Data leakage
28. Grouped variance
29. Rolling variance
30. Residual variance
31. Bias-variance tradeoff
32. Statistical interpretation
33. Common mistakes
34. Practical Machine Learning examples

Requirements:
Python 3.x

Optional Libraries:
numpy
pandas
scikit-learn

Author:
Kishor Patil
"""

# =============================================================================

# 1. IMPORTS

# =============================================================================

import math
from typing import Dict, List, Sequence, Tuple

# =============================================================================

# 2. WHAT IS VARIANCE?

# =============================================================================

"""
Variance
--------

Variance measures how far observations are spread around
their mean.

The basic idea is:

```
1. Calculate the mean.
2. Calculate each observation's deviation from the mean.
3. Square each deviation.
4. Calculate the average of the squared deviations.
```

The result is variance.

Why square the deviations?

If we simply averaged:

```
(x - mean)
```

positive and negative deviations would cancel.

For example:

```
Data = [1, 2, 3]

Mean = 2

Deviations:
    -1, 0, +1

Average deviation:
    (-1 + 0 + 1) / 3 = 0
```

That does not describe the spread.

Squaring gives:

```
1, 0, 1
```

Now the average is positive and represents dispersion.
"""

# =============================================================================

# 3. VARIANCE FORMULAS

# =============================================================================

"""
Population Variance
-------------------

When the data represents the complete population:

```
             N
            ---
            \
             /   (xi - μ)²
            ---
            i=1

σ² = ----------------
          N
```

Where:

```
σ² = population variance
xi = observation
μ  = population mean
N  = population size
```

## Sample Variance

When the data represents a sample from a larger population:

```
              n
             ---
             \
              /   (xi - x̄)²
             ---
             i=1

s² = ------------------
            n - 1
```

Where:

```
s² = sample variance
xi = observation
x̄  = sample mean
n  = sample size
```

Important:

The denominator is different:

```
Population -> N
Sample     -> n - 1
```

"""

# =============================================================================

# 4. ALTERNATIVE POPULATION VARIANCE FORMULA

# =============================================================================

"""
Computational Formula
---------------------

Population variance can also be written as:

```
Var(X) = E[X²] - (E[X])²
```

For a finite dataset:

```
variance =
    average(x²) - average(x)²
```

This identity is mathematically useful and appears frequently
in probability, statistics, and Machine Learning.

However, directly using:

```
E[X²] - E[X]²
```

with floating-point numbers can sometimes suffer from
numerical cancellation.

For educational and general-purpose calculations, the
centered form:

```
average((x - mean)²)
```

is often preferable.
"""

# =============================================================================

# 5. POPULATION VARIANCE

# =============================================================================

def population_variance(
data: Sequence[float],
) -> float:
"""
Calculate population variance manually.

```
Parameters
----------
data : Sequence[float]
    Complete population dataset.

Returns
-------
float
    Population variance.

Formula
-------
    σ² = Σ(x - μ)² / N

Raises
------
ValueError
    If data is empty.

Examples
--------
>>> population_variance([2, 4, 6])
2.666666...
"""

if not data:
    raise ValueError(
        "Cannot calculate variance of empty data."
    )

mean = sum(data) / len(data)

squared_deviations = [
    (value - mean) ** 2
    for value in data
]

return sum(squared_deviations) / len(data)
```

# =============================================================================

# 6. SAMPLE VARIANCE

# =============================================================================

def sample_variance(
data: Sequence[float],
) -> float:
"""
Calculate sample variance manually.

```
Parameters
----------
data : Sequence[float]
    Sample dataset.

Returns
-------
float
    Sample variance.

Formula
-------
    s² = Σ(x - x̄)² / (n - 1)

Raises
------
ValueError
    If fewer than two observations are provided.
"""

if len(data) < 2:
    raise ValueError(
        "Sample variance requires at least two observations."
    )

mean = sum(data) / len(data)

squared_deviations = [
    (value - mean) ** 2
    for value in data
]

return sum(squared_deviations) / (len(data) - 1)
```

# =============================================================================

# 7. STANDARD DEVIATION FROM VARIANCE

# =============================================================================

def population_standard_deviation(
data: Sequence[float],
) -> float:
"""
Calculate population standard deviation from variance.

```
Formula:

    σ = sqrt(σ²)
"""

return math.sqrt(
    population_variance(data)
)
```

def sample_standard_deviation(
data: Sequence[float],
) -> float:
"""
Calculate sample standard deviation from sample variance.

```
Formula:

    s = sqrt(s²)
"""

return math.sqrt(
    sample_variance(data)
)
```

# =============================================================================

# 8. STEP-BY-STEP VARIANCE

# =============================================================================

def explain_variance(
data: Sequence[float],
) -> None:
"""
Display the complete variance calculation.

```
Steps:

    1. Calculate mean.
    2. Calculate deviations.
    3. Square deviations.
    4. Sum squared deviations.
    5. Divide by population size.
    6. Obtain population variance.
"""

if not data:
    raise ValueError(
        "Dataset cannot be empty."
    )

mean = sum(data) / len(data)

deviations = [
    value - mean
    for value in data
]

squared_deviations = [
    deviation ** 2
    for deviation in deviations
]

total_squared_deviation = sum(
    squared_deviations
)

variance = (
    total_squared_deviation
    / len(data)
)

print("\nStep-by-Step Variance")
print("=" * 60)

print(f"Data: {list(data)}")
print(f"Mean: {mean}")

print("\nDeviations:")
print(deviations)

print("\nSquared Deviations:")
print(squared_deviations)

print(
    f"\nSum of Squared Deviations: "
    f"{total_squared_deviation}"
)

print(f"Variance: {variance}")
```

# =============================================================================

# 9. POPULATION VS SAMPLE VARIANCE

# =============================================================================

def compare_population_sample(
data: Sequence[float],
) -> None:
"""
Compare population and sample variance.
"""

```
population_var = population_variance(data)

if len(data) >= 2:
    sample_var = sample_variance(data)
else:
    sample_var = float("nan")

print("\nPopulation vs Sample Variance")
print("=" * 60)

print(f"Data: {list(data)}")
print(f"Population Variance: {population_var}")
print(f"Sample Variance    : {sample_var}")
```

# =============================================================================

# 10. LOW VARIANCE

# =============================================================================

def low_variance_example() -> None:
"""
Demonstrate a dataset with relatively low variance.
"""

```
data = [49, 50, 50, 51, 50]

variance = population_variance(data)

print("\nLow Variance")
print("-" * 40)
print(f"Data: {data}")
print(f"Mean: {sum(data) / len(data):.2f}")
print(f"Variance: {variance:.4f}")
```

# =============================================================================

# 11. HIGH VARIANCE

# =============================================================================

def high_variance_example() -> None:
"""
Demonstrate a dataset with relatively high variance.
"""

```
data = [10, 30, 50, 70, 90]

variance = population_variance(data)

print("\nHigh Variance")
print("-" * 40)
print(f"Data: {data}")
print(f"Mean: {sum(data) / len(data):.2f}")
print(f"Variance: {variance:.4f}")
```

# =============================================================================

# 12. SAME MEAN, DIFFERENT VARIANCE

# =============================================================================

def same_mean_different_variance() -> None:
"""
Demonstrate that datasets can have the same mean
but very different variance.
"""

```
dataset_a = [48, 49, 50, 51, 52]
dataset_b = [10, 30, 50, 70, 90]

mean_a = sum(dataset_a) / len(dataset_a)
mean_b = sum(dataset_b) / len(dataset_b)

variance_a = population_variance(dataset_a)
variance_b = population_variance(dataset_b)

print("\nSame Mean, Different Variance")
print("=" * 60)

print(f"Dataset A: {dataset_a}")
print(f"Mean A: {mean_a}")
print(f"Variance A: {variance_a}")

print()

print(f"Dataset B: {dataset_b}")
print(f"Mean B: {mean_b}")
print(f"Variance B: {variance_b}")
```

# =============================================================================

# 13. VARIANCE USING NUMPY

# =============================================================================

def numpy_variance_example() -> None:
"""
Demonstrate variance using NumPy.

```
NumPy:

    np.var(data)

uses:

    ddof=0

by default.

Therefore:

    np.var(data)
        -> population variance

    np.var(data, ddof=1)
        -> sample variance
"""

try:
    import numpy as np
except ImportError:
    print("\nNumPy is not installed.")
    print("Install it using:")
    print("pip install numpy")
    return

data = np.array(
    [10, 20, 30, 40, 50]
)

population_var = np.var(data)
sample_var = np.var(
    data,
    ddof=1,
)

print("\nNumPy Variance")
print("-" * 50)
print(f"Data: {data}")
print(f"Population Variance: {population_var}")
print(f"Sample Variance    : {sample_var}")
```

# =============================================================================

# 14. VARIANCE USING PANDAS

# =============================================================================

def pandas_variance_example() -> None:
"""
Demonstrate variance using Pandas.

```
Pandas Series.var() uses:

    ddof=1

by default.

Therefore, it returns sample variance by default.
"""

try:
    import pandas as pd
except ImportError:
    print("\nPandas is not installed.")
    print("Install it using:")
    print("pip install pandas")
    return

data = pd.Series(
    [10, 20, 30, 40, 50]
)

print("\nPandas Variance")
print("-" * 50)
print(f"Data:\n{data}")

print("\nSample Variance:")
print(data.var())

print("\nPopulation Variance:")
print(data.var(ddof=0))
```

# =============================================================================

# 15. DATAFRAME VARIANCE

# =============================================================================

def dataframe_variance_example() -> None:
"""
Calculate variance for DataFrame columns.
"""

```
try:
    import pandas as pd
except ImportError:
    print("\nPandas is not installed.")
    return

df = pd.DataFrame(
    {
        "Age": [20, 21, 22, 25, 30],
        "Salary": [
            30000,
            32000,
            35000,
            40000,
            50000,
        ],
        "Experience": [
            1,
            2,
            2,
            4,
            7,
        ],
    }
)

print("\nDataFrame")
print(df)

print("\nSample Variance:")
print(df.var())

print("\nPopulation Variance:")
print(df.var(ddof=0))
```

# =============================================================================

# 16. VARIANCE AND STANDARD DEVIATION

# =============================================================================

def variance_vs_standard_deviation() -> None:
"""
Demonstrate the relationship between variance
and standard deviation.

```
Formula:

    Standard Deviation = sqrt(Variance)

    Variance = Standard Deviation²
"""

data = [10, 20, 30, 40, 50]

variance = population_variance(data)
standard_deviation = math.sqrt(variance)

print("\nVariance vs Standard Deviation")
print("=" * 60)

print(f"Data: {data}")
print(f"Variance: {variance}")
print(
    f"Standard Deviation: "
    f"{standard_deviation}"
)
```

# =============================================================================

# 17. EFFECT OF OUTLIERS

# =============================================================================

def outlier_effect_example() -> None:
"""
Demonstrate that variance is highly sensitive to outliers.

```
Because deviations are squared, large deviations receive
disproportionately large weight.
"""

normal_data = [
    10,
    10,
    10,
    10,
    10,
]

outlier_data = [
    10,
    10,
    10,
    10,
    100,
]

normal_variance = population_variance(
    normal_data
)

outlier_variance = population_variance(
    outlier_data
)

print("\nEffect of Outliers")
print("=" * 60)

print(
    f"Without outlier: "
    f"{normal_variance:.4f}"
)

print(
    f"With outlier   : "
    f"{outlier_variance:.4f}"
)
```

# =============================================================================

# 18. TRANSLATION INVARIANCE

# =============================================================================

def translation_invariance_example() -> None:
"""
Demonstrate that adding a constant to every value
does not change variance.

```
If:

    Y = X + c

then:

    Var(Y) = Var(X)
"""

original = [10, 20, 30]

shifted = [
    value + 100
    for value in original
]

original_variance = population_variance(
    original
)

shifted_variance = population_variance(
    shifted
)

print("\nTranslation Invariance")
print("-" * 50)

print(f"Original: {original}")
print(f"Shifted : {shifted}")

print(
    f"Original Variance: "
    f"{original_variance}"
)

print(
    f"Shifted Variance : "
    f"{shifted_variance}"
)
```

# =============================================================================

# 19. SCALING PROPERTY

# =============================================================================

def scaling_property_example() -> None:
"""
Demonstrate the scaling property of variance.

```
If:

    Y = aX

then:

    Var(Y) = a² Var(X)

This is an important difference from standard deviation:

    SD(aX) = |a| SD(X)
"""

data = [10, 20, 30]

scaled_data = [
    2 * value
    for value in data
]

original_variance = population_variance(
    data
)

scaled_variance = population_variance(
    scaled_data
)

print("\nScaling Property")
print("=" * 50)

print(f"Original: {data}")
print(f"Scaled: {scaled_data}")

print(
    f"Original Variance: "
    f"{original_variance}"
)

print(
    f"Scaled Variance: "
    f"{scaled_variance}"
)

print(
    "\nExpected:"
    f" {4 * original_variance}"
)
```

# =============================================================================

# 20. CONSTANT DATA

# =============================================================================

def constant_data_example() -> None:
"""
Demonstrate zero variance.

```
If every observation has the same value:

    [5, 5, 5, 5]

then every deviation from the mean is zero.

Therefore:

    Variance = 0
"""

data = [
    5,
    5,
    5,
    5,
]

variance = population_variance(data)

print("\nConstant Data")
print("-" * 40)
print(f"Data: {data}")
print(f"Variance: {variance}")
```

# =============================================================================

# 21. ZERO-VARIANCE FEATURE

# =============================================================================

def zero_variance_feature_example() -> None:
"""
Demonstrate a constant Machine Learning feature.

```
Example:

    Feature = [1, 1, 1, 1, 1]

The feature has no variability and therefore carries no
information for distinguishing observations based on its
variation.

In practical preprocessing, constant features can often
be removed.
"""

feature = [
    1,
    1,
    1,
    1,
    1,
]

variance = population_variance(feature)

print("\nZero-Variance Feature")
print("=" * 50)
print(f"Feature: {feature}")
print(f"Variance: {variance}")

if variance == 0:
    print(
        "This feature is constant and has no variation."
    )
```

# =============================================================================

# 22. Z-SCORE AND VARIANCE

# =============================================================================

def z_score_example() -> None:
"""
Demonstrate the relationship between variance,
standard deviation, and z-score.

```
Formula:

    z = (x - mean) / standard_deviation

Since:

    standard_deviation = sqrt(variance)

variance indirectly determines the scale used by
standardization.
"""

data = [
    10,
    20,
    30,
    40,
    50,
]

mean = sum(data) / len(data)

variance = population_variance(data)

standard_deviation = math.sqrt(variance)

value = 50

z = (
    value - mean
) / standard_deviation

print("\nVariance and Z-Score")
print("=" * 50)

print(f"Mean: {mean}")
print(f"Variance: {variance}")
print(
    f"Standard Deviation: "
    f"{standard_deviation}"
)
print(f"Value: {value}")
print(f"Z-Score: {z:.4f}")
```

# =============================================================================

# 23. COEFFICIENT OF VARIATION

# =============================================================================

def coefficient_of_variation(
data: Sequence[float],
) -> float:
"""
Calculate the coefficient of variation (CV).

```
Formula:

    CV = standard_deviation / mean

Often expressed as a percentage:

    CV% = (standard_deviation / mean) × 100

CV is useful for comparing relative variability when the
measurement scales differ and the ratio interpretation is
meaningful.

Important:

    CV is not generally appropriate when the mean is zero
    or near zero, and interpretation can be problematic for
    variables that can take negative values.
"""

if not data:
    raise ValueError(
        "Dataset cannot be empty."
    )

mean = sum(data) / len(data)

if mean == 0:
    raise ValueError(
        "Coefficient of variation is undefined when mean is zero."
    )

standard_deviation = population_standard_deviation(
    data
)

return standard_deviation / mean
```

def coefficient_of_variation_example() -> None:
"""
Demonstrate coefficient of variation.
"""

```
data = [
    90,
    100,
    110,
    100,
    100,
]

cv = coefficient_of_variation(data)

print("\nCoefficient of Variation")
print("-" * 50)
print(f"Data: {data}")
print(f"CV: {cv:.4f}")
print(f"CV (%): {cv * 100:.2f}%")
```

# =============================================================================

# 24. VARIANCE AND COVARIANCE

# =============================================================================

def covariance_relationship_example() -> None:
"""
Explain the relationship between variance and covariance.

```
Variance is a special case of covariance:

    Var(X) = Cov(X, X)

Covariance measures how two variables vary together.

Variance measures how one variable varies with itself.
"""

x = [
    1,
    2,
    3,
    4,
    5,
]

mean_x = sum(x) / len(x)

variance = population_variance(x)

covariance_xx = sum(
    (value - mean_x) * (value - mean_x)
    for value in x
) / len(x)

print("\nVariance and Covariance")
print("=" * 50)

print(f"Variance(X): {variance}")
print(f"Covariance(X, X): {covariance_xx}")
```

# =============================================================================

# 25. CORRELATION RELATIONSHIP

# =============================================================================

def correlation_relationship_example() -> None:
"""
Explain the relationship between covariance, variance,
standard deviation, and correlation.

```
Pearson correlation:

    r =
        Cov(X, Y)
        ----------------
        SD(X) × SD(Y)

Variance is therefore part of the scale normalization
used to calculate correlation.
"""

x = [
    1,
    2,
    3,
    4,
    5,
]

y = [
    2,
    4,
    6,
    8,
    10,
]

mean_x = sum(x) / len(x)
mean_y = sum(y) / len(y)

covariance = sum(
    (xi - mean_x) * (yi - mean_y)
    for xi, yi in zip(x, y)
) / len(x)

sd_x = math.sqrt(
    population_variance(x)
)

sd_y = math.sqrt(
    population_variance(y)
)

correlation = (
    covariance
    / (sd_x * sd_y)
)

print("\nVariance and Correlation")
print("=" * 50)

print(f"Covariance: {covariance}")
print(f"SD(X): {sd_x}")
print(f"SD(Y): {sd_y}")
print(f"Correlation: {correlation}")
```

# =============================================================================

# 26. VARIANCE THRESHOLD

# =============================================================================

def variance_threshold(
data: Sequence[float],
threshold: float,
) -> bool:
"""
Determine whether a feature's variance exceeds a threshold.

```
Parameters
----------
data : Sequence[float]
    Feature values.

threshold : float
    Minimum variance required.

Returns
-------
bool
    True if variance is greater than threshold.

This is the basic idea behind variance-threshold feature
selection.

Important:

    The threshold must be chosen with awareness of feature
    scale. Raw variance is not scale-invariant.
"""

if threshold < 0:
    raise ValueError(
        "Variance threshold cannot be negative."
    )

return population_variance(data) > threshold
```

# =============================================================================

# 27. VARIANCE THRESHOLD EXAMPLE

# =============================================================================

def variance_threshold_example() -> None:
"""
Demonstrate variance-based feature filtering.
"""

```
low_variance_feature = [
    10,
    10,
    10,
    11,
    10,
]

high_variance_feature = [
    10,
    30,
    50,
    70,
    90,
]

threshold = 10

print("\nVariance Threshold")
print("=" * 50)

print(
    "Low variance feature:",
    low_variance_feature,
)

print(
    "Variance:",
    population_variance(
        low_variance_feature
    ),
)

print(
    "Pass threshold:",
    variance_threshold(
        low_variance_feature,
        threshold,
    ),
)

print()

print(
    "High variance feature:",
    high_variance_feature,
)

print(
    "Variance:",
    population_variance(
        high_variance_feature
    ),
)

print(
    "Pass threshold:",
    variance_threshold(
        high_variance_feature,
        threshold,
    ),
)
```

# =============================================================================

# 28. SCIKIT-LEARN VARIANCE THRESHOLD

# =============================================================================

def sklearn_variance_threshold_example() -> None:
"""
Demonstrate scikit-learn VarianceThreshold.

```
VarianceThreshold removes features whose variance does not
meet a specified threshold.

Important:

    Raw variance depends on feature units and scale.
    Therefore, threshold selection requires context.
"""

try:
    from sklearn.feature_selection import VarianceThreshold
except ImportError:
    print("\nscikit-learn is not installed.")
    print("Install it using:")
    print("pip install scikit-learn")
    return

data = [
    [1.0, 10.0, 100.0],
    [1.0, 20.0, 100.0],
    [1.0, 30.0, 100.0],
    [1.0, 40.0, 100.0],
    [1.0, 50.0, 100.0],
]

selector = VarianceThreshold(
    threshold=0.0
)

transformed = selector.fit_transform(data)

print("\nScikit-Learn VarianceThreshold")
print("=" * 50)

print("Original feature matrix:")
for row in data:
    print(row)

print("\nFeatures after removing zero-variance columns:")

for row in transformed:
    print(row)
```

# =============================================================================

# 29. GROUPED VARIANCE

# =============================================================================

def grouped_variance_example() -> None:
"""
Calculate variance separately for different groups.
"""

```
try:
    import pandas as pd
except ImportError:
    print("\nPandas is not installed.")
    return

df = pd.DataFrame(
    {
        "Department": [
            "IT",
            "IT",
            "IT",
            "HR",
            "HR",
            "HR",
        ],
        "Salary": [
            50000,
            55000,
            60000,
            40000,
            42000,
            45000,
        ],
    }
)

grouped_variance = (
    df.groupby("Department")["Salary"]
    .var()
)

print("\nGrouped Variance")
print("=" * 50)

print(df)

print("\nSample Variance by Department:")
print(grouped_variance)
```

# =============================================================================

# 30. ROLLING VARIANCE

# =============================================================================

def rolling_variance_example() -> None:
"""
Demonstrate rolling variance for time-series data.
"""

```
try:
    import pandas as pd
except ImportError:
    print("\nPandas is not installed.")
    return

data = pd.Series(
    [
        10,
        12,
        11,
        20,
        19,
        21,
        22,
    ]
)

rolling_variance = (
    data.rolling(window=3)
    .var()
)

print("\nRolling Variance")
print("=" * 50)

print("Data:")
print(data)

print("\nRolling Variance:")
print(rolling_variance)
```

# =============================================================================

# 31. RESIDUAL VARIANCE

# =============================================================================

def residual_variance_example() -> None:
"""
Demonstrate variance of prediction residuals.

```
Residual:

    e = actual - predicted

Residual variance measures the spread of prediction errors
around their mean.
"""

actual = [
    100,
    120,
    130,
    150,
    170,
]

predicted = [
    98,
    125,
    128,
    145,
    175,
]

residuals = [
    actual_value - predicted_value
    for actual_value, predicted_value
    in zip(actual, predicted)
]

variance = population_variance(
    residuals
)

print("\nResidual Variance")
print("=" * 50)

print(f"Actual    : {actual}")
print(f"Predicted : {predicted}")
print(f"Residuals : {residuals}")
print(f"Residual Variance: {variance:.4f}")
```

# =============================================================================

# 32. BIAS-VARIANCE TRADEOFF

# =============================================================================

def bias_variance_example() -> None:
"""
Explain the conceptual role of variance in Machine Learning.

```
In supervised learning, model prediction error is often
discussed using the bias-variance decomposition.

Conceptually:

    Expected prediction error
    =
    Bias²
    + Variance
    + Irreducible noise

Model variance describes how much a learned model's
predictions can change when the training data changes.

This is different from the variance of a raw feature.

High model variance is commonly associated with
overfitting, while high bias is commonly associated
with underfitting.

The decomposition has assumptions and exact forms vary
with the loss function and setup.
"""

print("\nBias-Variance Concept")
print("=" * 50)

print("Prediction Error ≈ Bias² + Variance + Noise")
print()
print("High bias:")
print("  Model may be too simple.")
print()
print("High variance:")
print("  Model may be overly sensitive to training data.")
print()
print("Goal:")
print("  Balance model complexity and generalization.")
```

# =============================================================================

# 33. VARIANCE IN MACHINE LEARNING

# =============================================================================

def machine_learning_application() -> None:
"""
Explain common applications of variance in Machine Learning.
"""

```
applications = [
    "1. Measuring feature variability.",
    "2. Detecting constant features.",
    "3. Removing low-variance features.",
    "4. Understanding data distributions.",
    "5. Standardization and scaling.",
    "6. Statistical feature analysis.",
    "7. Outlier analysis.",
    "8. Residual/error analysis.",
    "9. Bias-variance analysis.",
    "10. Monitoring changes in feature distributions.",
]

print("\nVariance in Machine Learning")
print("=" * 60)

for application in applications:
    print(application)
```

# =============================================================================

# 34. DATA LEAKAGE AND PREPROCESSING

# =============================================================================

def data_leakage_example() -> None:
"""
Explain training-only preprocessing.

```
When variance is used to make a preprocessing decision,
such as feature selection, the decision should generally
be learned from the training data only.

Correct workflow:

    1. Split the dataset.
    2. Fit preprocessing on training data.
    3. Transform training data.
    4. Transform validation/test data using the learned
       preprocessing.

This helps prevent information leakage.
"""

training_feature = [
    10,
    20,
    30,
    40,
]

test_feature = [
    50,
    60,
]

training_variance = population_variance(
    training_feature
)

print("\nTraining-Only Variance Analysis")
print("=" * 60)

print(f"Training feature: {training_feature}")
print(f"Test feature: {test_feature}")
print(
    f"Training variance: "
    f"{training_variance}"
)

print(
    "\nPreprocessing decisions should be fitted "
    "using training data."
)
```

# =============================================================================

# 35. VARIANCE AND UNITS

# =============================================================================

def units_example() -> None:
"""
Explain why variance has squared units.

```
If the original measurement is:

    meters

then:

    variance -> meters²

If the original measurement is:

    kilograms

then:

    variance -> kilograms²

Standard deviation returns to the original units.
"""

print("\nVariance and Units")
print("=" * 50)

print("Original data: meters")
print("Variance: meters²")
print("Standard deviation: meters")

print()

print("Original data: rupees")
print("Variance: rupees²")
print("Standard deviation: rupees")
```

# =============================================================================

# 36. NUMERICAL STABILITY

# =============================================================================

def numerically_stable_variance(
data: Sequence[float],
) -> float:
"""
Calculate variance using a numerically stable two-pass method.

```
First pass:

    Calculate mean.

Second pass:

    Calculate squared deviations from that mean.

This is generally preferable to directly evaluating:

    mean(x²) - mean(x)²

for floating-point data where cancellation can cause loss
of precision.
"""

if not data:
    raise ValueError(
        "Cannot calculate variance of empty data."
    )

mean = sum(data) / len(data)

squared_deviations = sum(
    (value - mean) ** 2
    for value in data
)

return squared_deviations / len(data)
```

# =============================================================================

# 37. NUMERICAL STABILITY EXAMPLE

# =============================================================================

def numerical_stability_example() -> None:
"""
Demonstrate the preferred centered variance calculation.
"""

```
data = [
    1000000.1,
    1000000.2,
    1000000.3,
    1000000.4,
    1000000.5,
]

variance = numerically_stable_variance(data)

print("\nNumerical Stability")
print("=" * 50)

print(f"Data: {data}")
print(f"Variance: {variance:.12f}")
```

# =============================================================================

# 38. COMPLETE VARIANCE ANALYSIS

# =============================================================================

def complete_analysis(
data: Sequence[float],
) -> Dict[str, float]:
"""
Return a complete variance analysis.

```
Returns
-------
Dict[str, float]
    Mean, population variance, population standard deviation,
    and sample statistics when available.
"""

if not data:
    raise ValueError(
        "Dataset cannot be empty."
    )

mean = sum(data) / len(data)

population_var = population_variance(
    data
)

population_sd = math.sqrt(
    population_var
)

result: Dict[str, float] = {
    "mean": mean,
    "population_variance": population_var,
    "population_standard_deviation": population_sd,
}

if len(data) >= 2:
    sample_var = sample_variance(data)

    result["sample_variance"] = sample_var
    result["sample_standard_deviation"] = math.sqrt(
        sample_var
    )

return result
```

# =============================================================================

# 39. COMMON MISTAKES

# =============================================================================

def common_mistakes() -> None:
"""
Display common mistakes when working with variance.
"""

```
mistakes = [
    "1. Confusing variance with standard deviation.",
    "2. Forgetting to square deviations.",
    "3. Using N when sample variance requires n - 1.",
    "4. Using n - 1 when the data represents the complete population.",
    "5. Forgetting that variance has squared units.",
    "6. Ignoring the strong effect of outliers.",
    "7. Treating raw variance as scale-independent.",
    "8. Using variance thresholds without considering feature units.",
    "9. Fitting feature-selection rules on test data.",
    "10. Assuming low variance automatically means a feature is useless.",
    "11. Confusing feature variance with model variance.",
    "12. Ignoring numerical precision in large-valued data.",
]

print("\nCommon Mistakes")
print("=" * 60)

for mistake in mistakes:
    print(mistake)
```

# =============================================================================

# 40. QUICK REFERENCE

# =============================================================================

def quick_reference() -> None:
"""
Display a quick reference for variance.
"""

```
reference = {
    "Population variance": "Σ(x - μ)² / N",
    "Sample variance": "Σ(x - x̄)² / (n - 1)",
    "Standard deviation": "sqrt(variance)",
    "Low variance": "Values are relatively close together",
    "High variance": "Values are more spread out",
    "Zero variance": "All values are identical",
    "Units": "Squared units of original data",
    "Outlier sensitivity": "High",
    "ML use": "Feature analysis and selection",
    "Variance = covariance": "Var(X) = Cov(X, X)",
    "Scaling": "Var(aX) = a² Var(X)",
    "Translation": "Var(X + c) = Var(X)",
}

print("\nVariance Quick Reference")
print("=" * 60)

for key, value in reference.items():
    print(f"{key:<25}: {value}")
```

# =============================================================================

# 41. MAIN FUNCTION

# =============================================================================

def main() -> None:
"""
Run the variance examples.
"""

```
print("=" * 70)
print("VARIANCE IN STATISTICS FOR MACHINE LEARNING")
print("=" * 70)

# Basic calculation
data = [
    2,
    4,
    6,
    8,
    10,
]

print("\nBasic Variance")
print("-" * 40)

print(f"Data: {data}")
print(
    f"Population Variance: "
    f"{population_variance(data)}"
)

print(
    f"Sample Variance: "
    f"{sample_variance(data)}"
)

# Step-by-step
explain_variance(data)

# Population vs sample
compare_population_sample(data)

# Low variance
low_variance_example()

# High variance
high_variance_example()

# Same mean, different variance
same_mean_different_variance()

# NumPy
numpy_variance_example()

# Pandas
pandas_variance_example()

# DataFrame
dataframe_variance_example()

# Variance vs standard deviation
variance_vs_standard_deviation()

# Outliers
outlier_effect_example()

# Translation
translation_invariance_example()

# Scaling
scaling_property_example()

# Constant data
constant_data_example()

# Zero-variance feature
zero_variance_feature_example()

# Z-score
z_score_example()

# Coefficient of variation
coefficient_of_variation_example()

# Covariance
covariance_relationship_example()

# Correlation
correlation_relationship_example()

# Variance threshold
variance_threshold_example()

# Scikit-learn
sklearn_variance_threshold_example()

# Grouped variance
grouped_variance_example()

# Rolling variance
rolling_variance_example()

# Residual variance
residual_variance_example()

# Bias-variance
bias_variance_example()

# ML applications
machine_learning_application()

# Data leakage
data_leakage_example()

# Units
units_example()

# Numerical stability
numerical_stability_example()

# Common mistakes
common_mistakes()

# Quick reference
quick_reference()

# Complete analysis
result = complete_analysis(
    [10, 20, 30, 40, 50]
)

print("\nComplete Variance Analysis")
print("=" * 60)

for key, value in result.items():
    print(f"{key:<35}: {value}")
```

# =============================================================================

# 42. PROGRAM ENTRY POINT

# =============================================================================

if **name** == "**main**":
main()

"""
Key Takeaways
=============

1. Variance measures the average squared deviation from
   the mean.

2. Population variance uses:

   ```
   N
   ```

   in the denominator.

3. Sample variance uses:

   ```
   n - 1
   ```

   in the denominator.

4. Standard deviation is the square root of variance.

5. Variance is always non-negative.

6. Variance is zero when every observation is identical.

7. Variance has squared units.

8. Standard deviation has the same units as the original data.

9. Variance is sensitive to outliers because deviations
   are squared.

10. Adding a constant does not change variance:

    ```
    Var(X + c) = Var(X)
    ```

11. Multiplying by a constant changes variance according to:

    ```
    Var(aX) = a² Var(X)
    ```

12. Variance is a special case of covariance:

    ```
    Var(X) = Cov(X, X)
    ```

13. Variance is used in feature analysis and low-variance
    feature selection.

14. Raw variance depends on feature scale and units.

15. Variance-based preprocessing decisions should generally
    be learned from training data only.

16. Feature variance and model variance are different concepts.

17. The bias-variance tradeoff concerns model behavior,
    not simply the variance of an input feature.

18. The identity:

    ```
    Var(X) = E(X²) - [E(X)]²
    ```

    is mathematically important, but centered calculations
    are often preferable for floating-point numerical stability.
    """
