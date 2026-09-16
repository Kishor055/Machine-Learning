"""
Standard Deviation for Machine Learning
=======================================

File:
02-Mathematics-for-Machine-Learning/02-Statistics/standard-deviation.py

Purpose:
Learn how standard deviation measures the spread or dispersion
of data around its mean and how it is used in statistics,
data analysis, preprocessing, and Machine Learning.

Topics Covered:
1. What is standard deviation?
2. Why standard deviation matters
3. Population standard deviation
4. Sample standard deviation
5. Variance and standard deviation
6. Manual calculation
7. Step-by-step calculation
8. Population vs sample
9. NumPy implementation
10. Pandas implementation
11. Mean and standard deviation
12. Low vs high standard deviation
13. Standard deviation and normal distribution
14. Empirical rule
15. Z-score
16. Outlier detection
17. Standardization
18. Feature scaling in Machine Learning
19. StandardScaler concept
20. Data leakage prevention
21. Standard deviation and units
22. Effect of outliers
23. Translation invariance
24. Scaling behavior
25. Constant data
26. Numerical stability
27. Grouped standard deviation
28. Rolling standard deviation
29. Standard deviation of residuals
30. ML applications
31. Common mistakes
32. Practical examples

Requirements:
Python 3.x

Optional Libraries:
numpy
pandas

Author:
Kishor Patil
"""

# =============================================================================

# 1. IMPORTS

# =============================================================================

import math
from typing import Dict, List, Sequence, Tuple

# =============================================================================

# 2. WHAT IS STANDARD DEVIATION?

# =============================================================================

"""
Standard deviation measures how far observations typically spread
from the mean.

A small standard deviation means:

```
Values are relatively close to the mean.
```

A large standard deviation means:

```
Values are more spread out.
```

Example:

```
Dataset A:
    [49, 50, 51]

Dataset B:
    [10, 50, 90]
```

Both datasets have mean:

```
50
```

But Dataset B has much greater spread.

Therefore:

```
Standard deviation(B) > standard deviation(A)
```

Standard deviation is one of the most important measures of
dispersion in statistics and Machine Learning.
"""

# =============================================================================

# 3. VARIANCE AND STANDARD DEVIATION

# =============================================================================

"""
Variance and standard deviation are closely related.

Variance measures the average squared distance from the mean.

Standard deviation is the square root of variance.

Therefore:

```
Standard Deviation = sqrt(Variance)
```

Why take the square root?

Because variance is expressed in squared units.

For example:

```
Height -> meters

Variance -> meters²

Standard deviation -> meters
```

Standard deviation therefore returns the measure of spread
to the original unit of the data.
"""

# =============================================================================

# 4. POPULATION STANDARD DEVIATION

# =============================================================================

"""
Population standard deviation is used when the dataset represents
the entire population of interest.

Formula:

```
            n
           ---
           \
            /   (xi - μ)²
           ---
           i=1

σ = sqrt( ---------------- )
                n
```

Where:

```
σ  = population standard deviation
xi = individual observation
μ  = population mean
n  = population size
```

Example:

```
Data = [2, 4, 6]

Mean = 4

Deviations:
    2 - 4 = -2
    4 - 4 =  0
    6 - 4 =  2

Squared deviations:
    4, 0, 4

Variance:
    (4 + 0 + 4) / 3
    = 8 / 3

Standard deviation:
    sqrt(8 / 3)
```

"""

def population_standard_deviation(
data: Sequence[float],
) -> float:
"""
Calculate population standard deviation manually.

```
Parameters
----------
data : Sequence[float]
    Complete population dataset.

Returns
-------
float
    Population standard deviation.

Formula
-------
    σ = sqrt( Σ(x - μ)² / N )

Raises
------
ValueError
    If data is empty.

Examples
--------
>>> population_standard_deviation([2, 4, 6])
1.632993...
"""

if not data:
    raise ValueError(
        "Cannot calculate standard deviation of empty data."
    )

mean = sum(data) / len(data)

squared_deviations = [
    (x - mean) ** 2
    for x in data
]

variance = sum(squared_deviations) / len(data)

return math.sqrt(variance)
```

# =============================================================================

# 5. SAMPLE STANDARD DEVIATION

# =============================================================================

"""
Sample standard deviation is used when the dataset represents
a sample taken from a larger population.

Formula:

```
             n
            ---
            \
             /   (xi - x̄)²
            ---
            i=1

s = sqrt( ---------------- )
                 n - 1
```

Where:

```
s  = sample standard deviation
xi = observation
x̄  = sample mean
n  = sample size
```

Why n - 1?

When estimating population variability from a sample, dividing
by n - 1 provides Bessel's correction for the usual unbiased
estimator of population variance under standard assumptions.

The distinction between population and sample calculations
is important in statistical analysis.
"""

def sample_standard_deviation(
data: Sequence[float],
) -> float:
"""
Calculate sample standard deviation manually.

```
Parameters
----------
data : Sequence[float]
    Sample dataset.

Returns
-------
float
    Sample standard deviation.

Raises
------
ValueError
    If fewer than two observations are provided.

Formula
-------
    s = sqrt( Σ(x - x̄)² / (n - 1) )
"""

if len(data) < 2:
    raise ValueError(
        "Sample standard deviation requires at least two observations."
    )

mean = sum(data) / len(data)

squared_deviations = [
    (x - mean) ** 2
    for x in data
]

variance = sum(squared_deviations) / (len(data) - 1)

return math.sqrt(variance)
```

# =============================================================================

# 6. POPULATION VARIANCE

# =============================================================================

def population_variance(
data: Sequence[float],
) -> float:
"""
Calculate population variance.

```
Formula:

    σ² = Σ(x - μ)² / N
"""

if not data:
    raise ValueError("Cannot calculate variance of empty data.")

mean = sum(data) / len(data)

return sum(
    (x - mean) ** 2
    for x in data
) / len(data)
```

# =============================================================================

# 7. SAMPLE VARIANCE

# =============================================================================

def sample_variance(
data: Sequence[float],
) -> float:
"""
Calculate sample variance.

```
Formula:

    s² = Σ(x - x̄)² / (n - 1)
"""

if len(data) < 2:
    raise ValueError(
        "Sample variance requires at least two observations."
    )

mean = sum(data) / len(data)

return sum(
    (x - mean) ** 2
    for x in data
) / (len(data) - 1)
```

# =============================================================================

# 8. STEP-BY-STEP STANDARD DEVIATION

# =============================================================================

def explain_standard_deviation(
data: Sequence[float],
) -> None:
"""
Display the complete population standard deviation calculation.

```
Steps:

    1. Calculate mean
    2. Calculate deviations
    3. Square deviations
    4. Calculate variance
    5. Take square root
"""

if not data:
    raise ValueError("Dataset cannot be empty.")

mean = sum(data) / len(data)

deviations = [
    x - mean
    for x in data
]

squared_deviations = [
    deviation ** 2
    for deviation in deviations
]

variance = sum(squared_deviations) / len(data)

standard_deviation = math.sqrt(variance)

print("\nStep-by-Step Standard Deviation")
print("=" * 60)

print(f"Data: {list(data)}")
print(f"Mean: {mean}")

print("\nDeviations:")
print(deviations)

print("\nSquared Deviations:")
print(squared_deviations)

print(f"\nVariance: {variance}")
print(f"Standard Deviation: {standard_deviation}")
```

# =============================================================================

# 9. COMPARE POPULATION AND SAMPLE

# =============================================================================

def compare_population_sample(
data: Sequence[float],
) -> None:
"""
Compare population and sample standard deviations.
"""

```
population_sd = population_standard_deviation(data)

if len(data) >= 2:
    sample_sd = sample_standard_deviation(data)
else:
    sample_sd = float("nan")

print("\nPopulation vs Sample Standard Deviation")
print("=" * 60)
print(f"Data: {list(data)}")
print(f"Population SD: {population_sd}")
print(f"Sample SD    : {sample_sd}")
```

# =============================================================================

# 10. LOW STANDARD DEVIATION

# =============================================================================

def low_variability_example() -> None:
"""
Demonstrate data with relatively low variability.
"""

```
data = [49, 50, 50, 51, 50]

print("\nLow Variability")
print("-" * 40)
print(f"Data: {data}")
print(f"Mean: {sum(data) / len(data):.2f}")
print(
    f"Standard Deviation: "
    f"{population_standard_deviation(data):.4f}"
)
```

# =============================================================================

# 11. HIGH STANDARD DEVIATION

# =============================================================================

def high_variability_example() -> None:
"""
Demonstrate data with relatively high variability.
"""

```
data = [10, 30, 50, 70, 90]

print("\nHigh Variability")
print("-" * 40)
print(f"Data: {data}")
print(f"Mean: {sum(data) / len(data):.2f}")
print(
    f"Standard Deviation: "
    f"{population_standard_deviation(data):.4f}"
)
```

# =============================================================================

# 12. SAME MEAN, DIFFERENT STANDARD DEVIATION

# =============================================================================

def same_mean_different_spread() -> None:
"""
Demonstrate that two datasets can have the same mean
but different standard deviations.
"""

```
dataset_a = [48, 49, 50, 51, 52]
dataset_b = [10, 30, 50, 70, 90]

mean_a = sum(dataset_a) / len(dataset_a)
mean_b = sum(dataset_b) / len(dataset_b)

sd_a = population_standard_deviation(dataset_a)
sd_b = population_standard_deviation(dataset_b)

print("\nSame Mean, Different Spread")
print("=" * 60)

print(f"Dataset A: {dataset_a}")
print(f"Mean A: {mean_a}")
print(f"SD A: {sd_a}")

print()

print(f"Dataset B: {dataset_b}")
print(f"Mean B: {mean_b}")
print(f"SD B: {sd_b}")
```

# =============================================================================

# 13. STANDARD DEVIATION USING NUMPY

# =============================================================================

def numpy_standard_deviation_example() -> None:
"""
Demonstrate standard deviation using NumPy.

```
Important:

    np.std(data)

uses population-style normalization by default:

    ddof=0

For sample standard deviation:

    np.std(data, ddof=1)
"""

try:
    import numpy as np
except ImportError:
    print("\nNumPy is not installed.")
    print("Install it using:")
    print("pip install numpy")
    return

data = np.array([10, 20, 30, 40, 50])

population_sd = np.std(data)
sample_sd = np.std(data, ddof=1)

print("\nNumPy Standard Deviation")
print("-" * 50)
print(f"Data: {data}")
print(f"Population SD: {population_sd}")
print(f"Sample SD    : {sample_sd}")
```

# =============================================================================

# 14. STANDARD DEVIATION USING PANDAS

# =============================================================================

def pandas_standard_deviation_example() -> None:
"""
Demonstrate standard deviation using Pandas.

```
Important:

    Pandas Series.std()

uses sample standard deviation by default:

    ddof=1
"""

try:
    import pandas as pd
except ImportError:
    print("\nPandas is not installed.")
    print("Install it using:")
    print("pip install pandas")
    return

data = pd.Series([10, 20, 30, 40, 50])

print("\nPandas Standard Deviation")
print("-" * 50)
print(f"Data:\n{data}")

print(f"\nSample SD:")
print(data.std())

print(f"\nPopulation SD:")
print(data.std(ddof=0))
```

# =============================================================================

# 15. STANDARD DEVIATION OF DATAFRAME COLUMNS

# =============================================================================

def dataframe_standard_deviation_example() -> None:
"""
Calculate standard deviation for DataFrame columns.
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
        "Salary": [30000, 32000, 35000, 40000, 50000],
        "Experience": [1, 2, 2, 4, 7],
    }
)

print("\nDataFrame")
print(df)

print("\nSample Standard Deviation:")
print(df.std())

print("\nPopulation Standard Deviation:")
print(df.std(ddof=0))
```

# =============================================================================

# 16. Z-SCORE

# =============================================================================

"""
Z-Score
-------

A z-score tells us how many standard deviations an observation
is away from the mean.

Formula:

```
         x - μ
z = ---------------
          σ
```

For sample-based calculations, the appropriate sample statistic
may be used depending on the context.

Interpretation:

```
z = 0
    Value is at the mean.

z = 1
    Value is one standard deviation above the mean.

z = -2
    Value is two standard deviations below the mean.
```

Z-scores are widely used in:

```
- Standardization
- Outlier analysis
- Statistics
- Feature scaling
- Data analysis
```

"""

def z_score(
value: float,
mean: float,
standard_deviation: float,
) -> float:
"""
Calculate the z-score of a value.

```
Formula:

    z = (x - mean) / standard_deviation

Raises
------
ValueError
    If standard deviation is zero.
"""

if standard_deviation == 0:
    raise ValueError(
        "Z-score is undefined when standard deviation is zero."
    )

return (value - mean) / standard_deviation
```

# =============================================================================

# 17. Z-SCORE EXAMPLE

# =============================================================================

def z_score_example() -> None:
"""
Demonstrate z-score calculation.
"""

```
data = [10, 20, 30, 40, 50]

mean = sum(data) / len(data)

sd = population_standard_deviation(data)

value = 50

z = z_score(value, mean, sd)

print("\nZ-Score Example")
print("-" * 40)
print(f"Data: {data}")
print(f"Mean: {mean}")
print(f"Standard Deviation: {sd}")
print(f"Value: {value}")
print(f"Z-Score: {z:.4f}")
```

# =============================================================================

# 18. Z-SCORES FOR A DATASET

# =============================================================================

def z_scores(
data: Sequence[float],
) -> List[float]:
"""
Calculate z-scores for all observations.

```
Uses population mean and population standard deviation.

Returns
-------
List[float]
    Z-score for each observation.
"""

if not data:
    raise ValueError("Dataset cannot be empty.")

mean = sum(data) / len(data)

sd = population_standard_deviation(data)

if sd == 0:
    return [0.0 for _ in data]

return [
    (value - mean) / sd
    for value in data
]
```

# =============================================================================

# 19. Z-SCORE DATASET EXAMPLE

# =============================================================================

def z_scores_example() -> None:
"""
Display z-scores for a dataset.
"""

```
data = [10, 20, 30, 40, 50]

scores = z_scores(data)

print("\nZ-Scores")
print("-" * 40)

for value, score in zip(data, scores):
    print(f"{value:>5} -> {score:>8.4f}")
```

# =============================================================================

# 20. EMPIRICAL RULE

# =============================================================================

"""
Empirical Rule
--------------

For a distribution that is approximately normal:

```
About 68% of observations
    fall within 1 standard deviation of the mean.

About 95% of observations
    fall within 2 standard deviations.

About 99.7% of observations
    fall within 3 standard deviations.
```

This is commonly written as:

```
μ ± 1σ  -> approximately 68%

μ ± 2σ  -> approximately 95%

μ ± 3σ  -> approximately 99.7%
```

Important:

The empirical rule is appropriate for approximately normal
distributions. It should not be blindly applied to every dataset.
"""

def empirical_rule_example() -> None:
"""
Display the empirical rule ranges.
"""

```
mean = 100
sd = 15

print("\nEmpirical Rule Example")
print("=" * 50)

for k in [1, 2, 3]:
    lower = mean - k * sd
    upper = mean + k * sd

    print(
        f"{k} SD: "
        f"{lower} <= X <= {upper}"
    )
```

# =============================================================================

# 21. OUTLIER DETECTION USING Z-SCORE

# =============================================================================

def detect_z_score_outliers(
data: Sequence[float],
threshold: float = 3.0,
) -> List[Tuple[float, float]]:
"""
Detect observations whose absolute z-score exceeds a threshold.

```
Parameters
----------
data : Sequence[float]
    Numerical dataset.

threshold : float
    Z-score threshold.

Returns
-------
List[Tuple[float, float]]
    Tuples containing:

        (value, z_score)

Notes
-----
A threshold such as 3 is a common heuristic, not a universal
statistical law. It is more naturally motivated when the
distribution is approximately normal.
"""

if not data:
    raise ValueError("Dataset cannot be empty.")

if threshold <= 0:
    raise ValueError("Threshold must be positive.")

scores = z_scores(data)

return [
    (value, score)
    for value, score in zip(data, scores)
    if abs(score) > threshold
]
```

# =============================================================================

# 22. OUTLIER EXAMPLE

# =============================================================================

def outlier_example() -> None:
"""
Demonstrate z-score based outlier detection.
"""

```
data = [
    10,
    11,
    10,
    12,
    11,
    10,
    100,
]

outliers = detect_z_score_outliers(
    data,
    threshold=2.0,
)

print("\nZ-Score Outlier Detection")
print("-" * 50)
print(f"Data: {data}")
print(f"Potential outliers: {outliers}")
```

# =============================================================================

# 23. STANDARDIZATION

# =============================================================================

def standardize(
data: Sequence[float],
) -> List[float]:
"""
Standardize a numerical dataset.

```
Formula:

    z = (x - mean) / standard_deviation

After standardization, a non-constant dataset has:

    Mean approximately 0
    Standard deviation approximately 1

Uses population standard deviation for this educational
implementation.
"""

if not data:
    raise ValueError("Dataset cannot be empty.")

mean = sum(data) / len(data)

sd = population_standard_deviation(data)

if sd == 0:
    return [0.0 for _ in data]

return [
    (value - mean) / sd
    for value in data
]
```

# =============================================================================

# 24. STANDARDIZATION EXAMPLE

# =============================================================================

def standardization_example() -> None:
"""
Demonstrate feature standardization.
"""

```
data = [10, 20, 30, 40, 50]

standardized = standardize(data)

print("\nStandardization")
print("-" * 50)
print(f"Original     : {data}")
print(
    "Standardized : "
    f"{[round(value, 4) for value in standardized]}"
)

print(
    f"Mean after standardization: "
    f"{sum(standardized) / len(standardized):.6f}"
)

print(
    "SD after standardization: "
    f"{population_standard_deviation(standardized):.6f}"
)
```

# =============================================================================

# 25. MIN-MAX SCALING VS STANDARDIZATION

# =============================================================================

def scaling_comparison() -> None:
"""
Compare standardization and Min-Max scaling conceptually.

```
Standardization:

    z = (x - mean) / standard_deviation

Min-Max scaling:

    x_scaled = (x - min) / (max - min)

Standardization centers data around zero and scales according
to standard deviation.

Min-Max scaling maps values to a selected range, commonly [0, 1].

Neither method is universally best; the appropriate choice
depends on the algorithm and data.
"""

data = [10, 20, 30, 40, 50]

standardized = standardize(data)

minimum = min(data)
maximum = max(data)

min_max = [
    (value - minimum) / (maximum - minimum)
    for value in data
]

print("\nScaling Comparison")
print("=" * 50)
print(f"Original      : {data}")
print(
    f"Standardized  : "
    f"{[round(x, 4) for x in standardized]}"
)
print(
    f"Min-Max       : "
    f"{[round(x, 4) for x in min_max]}"
)
```

# =============================================================================

# 26. STANDARD DEVIATION AND OUTLIERS

# =============================================================================

def outlier_effect_example() -> None:
"""
Demonstrate that standard deviation is sensitive to outliers.

```
Compare:

    [10, 10, 10, 10, 10]

with:

    [10, 10, 10, 10, 100]
"""

normal_data = [10, 10, 10, 10, 10]
outlier_data = [10, 10, 10, 10, 100]

print("\nEffect of Outliers")
print("=" * 50)

print(
    f"Without outlier SD: "
    f"{population_standard_deviation(normal_data):.4f}"
)

print(
    f"With outlier SD   : "
    f"{population_standard_deviation(outlier_data):.4f}"
)
```

# =============================================================================

# 27. TRANSLATION INVARIANCE

# =============================================================================

def translation_invariance_example() -> None:
"""
Demonstrate that adding a constant to every observation
does not change standard deviation.

```
Example:

    [10, 20, 30]

Add 100:

    [110, 120, 130]

The spread is unchanged.
"""

original = [10, 20, 30]
shifted = [x + 100 for x in original]

original_sd = population_standard_deviation(original)
shifted_sd = population_standard_deviation(shifted)

print("\nTranslation Invariance")
print("-" * 50)
print(f"Original: {original}")
print(f"Shifted : {shifted}")
print(f"Original SD: {original_sd}")
print(f"Shifted SD : {shifted_sd}")
```

# =============================================================================

# 28. SCALING BEHAVIOR

# =============================================================================

def scaling_behavior_example() -> None:
"""
Demonstrate how standard deviation changes under scaling.

```
If:

    Y = aX + b

then:

    SD(Y) = |a| SD(X)

Adding b does not change spread.

Multiplying by a scales spread by |a|.
"""

data = [10, 20, 30]

multiplied = [2 * x for x in data]

original_sd = population_standard_deviation(data)
multiplied_sd = population_standard_deviation(multiplied)

print("\nScaling Behavior")
print("-" * 50)
print(f"Original: {data}")
print(f"Multiplied by 2: {multiplied}")
print(f"Original SD: {original_sd}")
print(f"New SD     : {multiplied_sd}")
```

# =============================================================================

# 29. CONSTANT DATA

# =============================================================================

def constant_data_example() -> None:
"""
Demonstrate standard deviation for constant data.

```
Example:

    [5, 5, 5, 5]

Every value equals the mean.

Therefore:

    Every deviation = 0
    Variance = 0
    Standard deviation = 0
"""

data = [5, 5, 5, 5]

sd = population_standard_deviation(data)

print("\nConstant Data")
print("-" * 40)
print(f"Data: {data}")
print(f"Standard Deviation: {sd}")
```

# =============================================================================

# 30. STANDARD DEVIATION AND UNITS

# =============================================================================

def units_example() -> None:
"""
Explain the units of variance and standard deviation.
"""

```
print("\nUnits of Measurement")
print("-" * 50)

print("Original measurement: meters")
print("Variance: meters²")
print("Standard deviation: meters")

print("\nOriginal measurement: rupees")
print("Variance: rupees²")
print("Standard deviation: rupees")
```

# =============================================================================

# 31. GROUPED STANDARD DEVIATION

# =============================================================================

def grouped_standard_deviation_example() -> None:
"""
Calculate standard deviation separately for groups using Pandas.
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

grouped_sd = df.groupby("Department")["Salary"].std()

print("\nGrouped Standard Deviation")
print("-" * 50)
print(df)
print("\nStandard Deviation by Department:")
print(grouped_sd)
```

# =============================================================================

# 32. ROLLING STANDARD DEVIATION

# =============================================================================

def rolling_standard_deviation_example() -> None:
"""
Demonstrate rolling standard deviation.

```
Rolling statistics are useful in:

    - Time series analysis
    - Financial analysis
    - Monitoring
    - Sensor data
    - Anomaly detection
"""

try:
    import pandas as pd
except ImportError:
    print("\nPandas is not installed.")
    return

data = pd.Series(
    [10, 12, 11, 20, 19, 21, 22]
)

rolling_sd = data.rolling(window=3).std()

print("\nRolling Standard Deviation")
print("-" * 50)
print("Data:")
print(data)

print("\nRolling SD:")
print(rolling_sd)
```

# =============================================================================

# 33. RESIDUAL STANDARD DEVIATION

# =============================================================================

def residual_standard_deviation_example() -> None:
"""
Demonstrate standard deviation of prediction residuals.

```
Residual:

    residual = actual - predicted

Standard deviation of residuals describes the spread of
prediction errors around their average residual.

This is different from simply calculating a model's MAE or RMSE.
"""

actual = [100, 120, 130, 150, 170]
predicted = [98, 125, 128, 145, 175]

residuals = [
    actual_value - predicted_value
    for actual_value, predicted_value
    in zip(actual, predicted)
]

residual_sd = population_standard_deviation(residuals)

print("\nResidual Standard Deviation")
print("-" * 50)
print(f"Actual    : {actual}")
print(f"Predicted : {predicted}")
print(f"Residuals : {residuals}")
print(f"Residual SD: {residual_sd:.4f}")
```

# =============================================================================

# 34. STANDARD DEVIATION OF RESIDUALS VS RMSE

# =============================================================================

def residual_metrics_example() -> None:
"""
Compare residual standard deviation and RMSE.

```
RMSE:

    sqrt(mean(residual²))

Residual standard deviation depends on whether residuals are
centered and whether population/sample conventions are used.

They should not be treated as interchangeable metrics.
"""

actual = [100, 120, 130, 150]
predicted = [95, 125, 128, 155]

residuals = [
    a - p
    for a, p in zip(actual, predicted)
]

mse = sum(
    error ** 2
    for error in residuals
) / len(residuals)

rmse = math.sqrt(mse)

residual_sd = population_standard_deviation(residuals)

print("\nResidual Metrics")
print("-" * 50)
print(f"Residuals: {residuals}")
print(f"Residual SD: {residual_sd:.4f}")
print(f"RMSE       : {rmse:.4f}")
```

# =============================================================================

# 35. STANDARD DEVIATION IN MACHINE LEARNING

# =============================================================================

def machine_learning_application() -> None:
"""
Explain common ML applications of standard deviation.

```
Standard deviation is useful for:

    1. Feature scaling
    2. Standardization
    3. Outlier analysis
    4. Data quality checks
    5. Distribution analysis
    6. Statistical feature analysis
    7. Error analysis
    8. Monitoring feature drift
    9. Detecting unusually large deviations
    10. Understanding feature variability

Algorithms that can be sensitive to feature scale include
many distance-based and gradient-based methods.

Examples:

    - K-Nearest Neighbors
    - K-Means
    - Support Vector Machines
    - Logistic Regression
    - Linear Regression
    - Neural Networks

Tree-based algorithms generally do not require standardization
merely because features have different units, although
preprocessing decisions should still be based on the broader
pipeline.
"""

print("\nMachine Learning Applications")
print("=" * 50)

applications = [
    "Feature standardization",
    "Outlier analysis",
    "Data preprocessing",
    "Feature variability analysis",
    "Error analysis",
    "Distribution analysis",
    "Data quality monitoring",
    "Feature drift monitoring",
]

for index, application in enumerate(applications, start=1):
    print(f"{index}. {application}")
```

# =============================================================================

# 36. STANDARDIZATION AND DATA LEAKAGE

# =============================================================================

def data_leakage_example() -> None:
"""
Explain the correct workflow for standardization.

```
Correct workflow:

    1. Split data into training and test sets.
    2. Calculate training mean.
    3. Calculate training standard deviation.
    4. Transform training data.
    5. Use the same training statistics to transform test data.

Incorrect workflow:

    Calculate mean and standard deviation using the complete
    dataset before splitting.

The incorrect approach can allow information from the test
set to influence preprocessing.
"""

training_data = [10, 20, 30, 40]
test_data = [50, 60]

training_mean = sum(training_data) / len(training_data)
training_sd = population_standard_deviation(training_data)

transformed_test = [
    (value - training_mean) / training_sd
    for value in test_data
]

print("\nTraining-Only Standardization")
print("=" * 50)
print(f"Training data: {training_data}")
print(f"Test data: {test_data}")
print(f"Training mean: {training_mean}")
print(f"Training SD: {training_sd}")
print(
    "Transformed test data: "
    f"{[round(x, 4) for x in transformed_test]}"
)
```

# =============================================================================

# 37. SCIKIT-LEARN STANDARD SCALER

# =============================================================================

def sklearn_standard_scaler_example() -> None:
"""
Demonstrate scikit-learn's StandardScaler.

```
StandardScaler transforms each feature approximately as:

    z = (x - mean) / standard_deviation

The scaler should normally be fitted only on training data
and then used to transform validation/test data.
"""

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("\nscikit-learn is not installed.")
    print("Install it using:")
    print("pip install scikit-learn")
    return

data = [
    [10, 1000],
    [20, 2000],
    [30, 3000],
    [40, 4000],
    [50, 5000],
]

scaler = StandardScaler()

transformed = scaler.fit_transform(data)

print("\nScikit-Learn StandardScaler")
print("-" * 50)
print("Original:")
for row in data:
    print(row)

print("\nStandardized:")
for row in transformed:
    print(
        [round(value, 4) for value in row]
    )
```

# =============================================================================

# 38. FEATURE WITH ZERO STANDARD DEVIATION

# =============================================================================

def zero_variance_feature_example() -> None:
"""
Demonstrate a constant feature.

```
A feature such as:

    [1, 1, 1, 1, 1]

has:

    Variance = 0
    Standard deviation = 0

Standardization by division with zero standard deviation
is undefined.

In practical ML pipelines, constant features are often
identified and may be removed because they contain no
variation across observations.
"""

feature = [1, 1, 1, 1, 1]

sd = population_standard_deviation(feature)

print("\nZero-Variance Feature")
print("-" * 50)
print(f"Feature: {feature}")
print(f"Standard deviation: {sd}")

if sd == 0:
    print("This feature has no variation.")
```

# =============================================================================

# 39. STANDARD DEVIATION AND FEATURE VARIABILITY

# =============================================================================

def feature_variability_example() -> None:
"""
Compare variability across different features.

```
Note:

Standard deviation depends on units and scale, so comparing
raw standard deviations across features with different units
may not be meaningful by itself.
"""

age = [20, 21, 22, 23, 24]
salary = [30000, 50000, 70000, 90000, 110000]

age_sd = population_standard_deviation(age)
salary_sd = population_standard_deviation(salary)

print("\nFeature Variability")
print("=" * 50)

print(f"Age SD: {age_sd:.2f}")
print(f"Salary SD: {salary_sd:.2f}")

print(
    "\nRaw SD values cannot be directly compared across "
    "different units without considering scale."
)
```

# =============================================================================

# 40. COMMON MISTAKES

# =============================================================================

def common_mistakes() -> None:
"""
Display common standard deviation mistakes.
"""

```
mistakes = [
    "1. Confusing variance with standard deviation.",
    "2. Using n instead of n-1 when a sample statistic is required.",
    "3. Using n-1 when the data represents the complete population.",
    "4. Forgetting that standard deviation has the same units as the data.",
    "5. Assuming every distribution is approximately normal.",
    "6. Treating ±3 standard deviations as a universal outlier rule.",
    "7. Ignoring the effect of extreme outliers.",
    "8. Standardizing before splitting train and test data.",
    "9. Dividing by zero for constant features.",
    "10. Comparing raw standard deviations across incompatible units.",
    "11. Assuming standardization always improves model performance.",
    "12. Confusing standard deviation with standard error.",
]

print("\nCommon Mistakes")
print("=" * 60)

for mistake in mistakes:
    print(mistake)
```

# =============================================================================

# 41. QUICK REFERENCE

# =============================================================================

def quick_reference() -> None:
"""
Display a quick reference for standard deviation.
"""

```
reference = {
    "Population formula": "sqrt(Σ(x - μ)² / N)",
    "Sample formula": "sqrt(Σ(x - x̄)² / (n - 1))",
    "Related measure": "Variance",
    "Units": "Same as original data",
    "Low SD": "Values are relatively close to mean",
    "High SD": "Values are more spread out",
    "Z-score": "(x - mean) / SD",
    "Standardization": "Center near 0 and scale by SD",
    "ML use": "Feature scaling and variability analysis",
    "Sensitive to outliers": "Yes",
}

print("\nStandard Deviation Quick Reference")
print("=" * 60)

for key, value in reference.items():
    print(f"{key:<25}: {value}")
```

# =============================================================================

# 42. COMPLETE STANDARD DEVIATION ANALYSIS

# =============================================================================

def complete_analysis(
data: Sequence[float],
) -> Dict[str, float]:
"""
Return a complete statistical analysis.

```
Returns
-------
Dict[str, float]
    Mean, population variance, population SD,
    sample variance, and sample SD.
"""

if not data:
    raise ValueError("Dataset cannot be empty.")

mean = sum(data) / len(data)

population_var = population_variance(data)
population_sd = math.sqrt(population_var)

result: Dict[str, float] = {
    "mean": mean,
    "population_variance": population_var,
    "population_standard_deviation": population_sd,
}

if len(data) >= 2:
    sample_var = sample_variance(data)
    sample_sd = math.sqrt(sample_var)

    result["sample_variance"] = sample_var
    result["sample_standard_deviation"] = sample_sd

return result
```

# =============================================================================

# 43. MAIN FUNCTION

# =============================================================================

def main() -> None:
"""
Run the standard deviation examples.
"""

```
print("=" * 70)
print("STANDARD DEVIATION IN STATISTICS FOR MACHINE LEARNING")
print("=" * 70)

# Basic calculation
data = [2, 4, 6, 8, 10]

print("\nBasic Calculation")
print("-" * 40)
print(f"Data: {data}")
print(
    "Population SD:",
    population_standard_deviation(data),
)
print(
    "Sample SD:",
    sample_standard_deviation(data),
)

# Step-by-step explanation
explain_standard_deviation(data)

# Population vs sample
compare_population_sample(data)

# Low variability
low_variability_example()

# High variability
high_variability_example()

# Same mean, different spread
same_mean_different_spread()

# NumPy
numpy_standard_deviation_example()

# Pandas
pandas_standard_deviation_example()

# DataFrame
dataframe_standard_deviation_example()

# Z-score
z_score_example()

# Multiple z-scores
z_scores_example()

# Empirical rule
empirical_rule_example()

# Outlier detection
outlier_example()

# Standardization
standardization_example()

# Scaling comparison
scaling_comparison()

# Outlier effect
outlier_effect_example()

# Translation invariance
translation_invariance_example()

# Scaling behavior
scaling_behavior_example()

# Constant data
constant_data_example()

# Units
units_example()

# Grouped standard deviation
grouped_standard_deviation_example()

# Rolling standard deviation
rolling_standard_deviation_example()

# Residual analysis
residual_standard_deviation_example()

# Residual metrics
residual_metrics_example()

# ML applications
machine_learning_application()

# Data leakage
data_leakage_example()

# Scikit-learn
sklearn_standard_scaler_example()

# Zero variance
zero_variance_feature_example()

# Feature variability
feature_variability_example()

# Common mistakes
common_mistakes()

# Quick reference
quick_reference()

# Complete analysis
result = complete_analysis(
    [10, 20, 30, 40, 50]
)

print("\nComplete Analysis")
print("=" * 60)

for key, value in result.items():
    print(f"{key:<35}: {value}")
```

# =============================================================================

# 44. PROGRAM ENTRY POINT

# =============================================================================

if **name** == "**main**":
main()

"""
Key Takeaways
=============

1. Standard deviation measures the spread of observations
   around their mean.

2. Population standard deviation uses:

   ```
   N
   ```

   in the denominator.

3. Sample standard deviation uses:

   ```
   n - 1
   ```

   in the denominator.

4. Variance is the square of standard deviation.

5. Standard deviation has the same units as the original data.

6. A low standard deviation indicates relatively low spread.

7. A high standard deviation indicates relatively high spread.

8. Standard deviation is sensitive to extreme values.

9. Z-score measures how many standard deviations an observation
   is away from the mean.

10. Standardization commonly uses:

    ```
    z = (x - mean) / standard_deviation
    ```

11. Standardization is useful for many Machine Learning
    algorithms that are sensitive to feature scale.

12. The empirical rule applies to approximately normal
    distributions.

13. A constant feature has standard deviation equal to zero.

14. Standardization statistics should generally be learned
    from training data only to avoid data leakage.

15. NumPy uses:

    ```
    ddof=0
    ```

    by default for `np.std()`.

16. Pandas uses:

    ```
    ddof=1
    ```

    by default for `Series.std()`.

17. Standard deviation should not be interpreted without
    considering the distribution, units, and context.
    """
