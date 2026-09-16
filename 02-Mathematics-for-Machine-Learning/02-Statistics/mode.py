"""
Mode for Machine Learning
=========================

File:
02-Mathematics-for-Machine-Learning/02-Statistics/mode.py

Purpose:
Learn how the mode represents the most frequently occurring
value in a dataset and how it is used in statistics,
data preprocessing, and Machine Learning.

Topics Covered:
1. What is mode?
2. Frequency and mode
3. Unique mode
4. Multiple modes
5. No unique mode
6. Manual mode calculation
7. Frequency tables
8. Mode for numerical data
9. Mode for categorical data
10. Mode with strings
11. Mode with negative values
12. Mode with duplicate values
13. NumPy implementation
14. Pandas implementation
15. Mode vs mean vs median
16. Mode and categorical data
17. Mode for missing-value imputation
18. Mode in Machine Learning preprocessing
19. Training-data-only imputation
20. Multimodal distributions
21. Weighted mode
22. Grouped mode
23. Mode in classification
24. Majority class
25. Common mistakes
26. Practical Machine Learning examples

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

from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# =============================================================================

# 2. WHAT IS MODE?

# =============================================================================

"""
Mode
----

The mode is the value that appears most frequently in a dataset.

Example:

```
Data:
    [2, 3, 3, 5, 7]

Frequencies:
    2 -> 1
    3 -> 2
    5 -> 1
    7 -> 1

Mode:
    3
```

The mode is especially useful for categorical data.

Example:

```
Colors:
    ["red", "blue", "red", "green", "red"]

Mode:
    "red"
```

Unlike the mean and median, the mode does not require
numerical data.

Therefore, the mode can be used with:

```
- Numbers
- Strings
- Categories
- Labels
- Boolean values
```

"""

# =============================================================================

# 3. MODE FORMULA

# =============================================================================

"""
There is no arithmetic formula for mode similar to the mean.

Instead:

```
Mode = value with the highest frequency
```

If:

```
frequency(x) = number of times x appears
```

then:

```
mode = argmax frequency(x)
```

Example:

```
Data = [1, 2, 2, 3, 3, 3, 4]

Frequency:
    1 -> 1
    2 -> 2
    3 -> 3
    4 -> 1

Mode = 3
```

"""

# =============================================================================

# 4. BASIC MANUAL MODE

# =============================================================================

def mode(data: Sequence[Any]) -> Any:
"""
Calculate the mode of a dataset.

```
Parameters
----------
data : Sequence[Any]
    Dataset containing values.

Returns
-------
Any
    The most frequently occurring value.

Raises
------
ValueError
    If the dataset is empty or there is no unique mode.

Examples
--------
>>> mode([1, 2, 2, 3])
2

>>> mode(["red", "blue", "red"])
'red'

Notes
-----
If multiple values have the same highest frequency, there is
no unique mode. Use `modes()` if you want all modes.
"""

if not data:
    raise ValueError("Cannot calculate mode of an empty dataset.")

frequencies = Counter(data)

highest_frequency = max(frequencies.values())

mode_values = [
    value
    for value, frequency in frequencies.items()
    if frequency == highest_frequency
]

if len(mode_values) != 1:
    raise ValueError(
        "Dataset does not have a unique mode. "
        f"Multiple modes found: {mode_values}"
    )

return mode_values[0]
```

# =============================================================================

# 5. RETURN ALL MODES

# =============================================================================

def modes(data: Sequence[Any]) -> List[Any]:
"""
Return all modes in a dataset.

```
A dataset may have:

    - One mode       -> Unimodal
    - Two modes      -> Bimodal
    - More than two  -> Multimodal

Parameters
----------
data : Sequence[Any]
    Dataset containing values.

Returns
-------
List[Any]
    All values having the highest frequency.

Examples
--------
>>> modes([1, 2, 2, 3])
[2]

>>> modes([1, 1, 2, 2, 3])
[1, 2]
"""

if not data:
    raise ValueError("Cannot calculate mode of an empty dataset.")

frequencies = Counter(data)

highest_frequency = max(frequencies.values())

return [
    value
    for value, frequency in frequencies.items()
    if frequency == highest_frequency
]
```

# =============================================================================

# 6. FREQUENCY TABLE

# =============================================================================

def frequency_table(data: Sequence[Any]) -> Dict[Any, int]:
"""
Create a frequency table.

```
A frequency table shows how many times each value occurs.

Example:

    Data:
        [10, 20, 20, 30, 30, 30]

    Frequency:
        10 -> 1
        20 -> 2
        30 -> 3

Parameters
----------
data : Sequence[Any]

Returns
-------
Dict[Any, int]
    Mapping from value to frequency.
"""

if not data:
    raise ValueError("Cannot create a frequency table from empty data.")

return dict(Counter(data))
```

# =============================================================================

# 7. DISPLAY FREQUENCY TABLE

# =============================================================================

def display_frequency_table(data: Sequence[Any]) -> None:
"""
Print a simple frequency table.
"""

```
frequencies = frequency_table(data)

print("\nFrequency Table")
print("-" * 30)

for value, frequency in frequencies.items():
    print(f"{value!r:<15} {frequency}")
```

# =============================================================================

# 8. STEP-BY-STEP MODE ANALYSIS

# =============================================================================

def analyze_mode(data: Sequence[Any]) -> None:
"""
Display a complete mode analysis.

```
Shows:

    - Dataset
    - Frequency table
    - Highest frequency
    - All modes
    - Number of modes
"""

if not data:
    raise ValueError("Cannot analyze an empty dataset.")

frequencies = frequency_table(data)
all_modes = modes(data)
highest_frequency = max(frequencies.values())

print("\nMode Analysis")
print("=" * 50)

print(f"Data: {list(data)}")

print("\nFrequency Table:")
for value, frequency in frequencies.items():
    print(f"  {value!r}: {frequency}")

print(f"\nHighest Frequency: {highest_frequency}")
print(f"Mode(s): {all_modes}")
print(f"Number of Modes: {len(all_modes)}")

if len(all_modes) == 1:
    print("Distribution Type: Unimodal")
elif len(all_modes) == 2:
    print("Distribution Type: Bimodal")
else:
    print("Distribution Type: Multimodal")
```

# =============================================================================

# 9. MODE FOR NUMERICAL DATA

# =============================================================================

def numerical_mode_example() -> None:
"""
Demonstrate mode with numerical data.
"""

```
data = [10, 20, 20, 30, 40, 20, 50]

print("\nNumerical Data")
print("-" * 30)
print(f"Data: {data}")
print(f"Mode: {mode(data)}")
```

# =============================================================================

# 10. MODE FOR CATEGORICAL DATA

# =============================================================================

def categorical_mode_example() -> None:
"""
Demonstrate mode with categorical data.

```
Categorical variables are one of the most important
use cases for mode.
"""

data = [
    "Python",
    "Java",
    "Python",
    "C++",
    "Python",
    "Java",
]

print("\nCategorical Data")
print("-" * 30)
print(f"Data: {data}")
print(f"Mode: {mode(data)}")
```

# =============================================================================

# 11. MODE WITH STRINGS

# =============================================================================

def string_mode_example() -> None:
"""
Demonstrate that mode can work with strings.
"""

```
data = [
    "Apple",
    "Banana",
    "Apple",
    "Orange",
    "Apple",
    "Banana",
]

print("\nString Mode")
print("-" * 30)
print(f"Data: {data}")
print(f"Mode: {mode(data)}")
```

# =============================================================================

# 12. MODE WITH BOOLEAN DATA

# =============================================================================

def boolean_mode_example() -> None:
"""
Demonstrate mode with Boolean values.
"""

```
data = [True, False, True, True, False, True]

print("\nBoolean Data")
print("-" * 30)
print(f"Data: {data}")
print(f"Mode: {mode(data)}")
```

# =============================================================================

# 13. MULTIPLE MODES

# =============================================================================

def multiple_modes_example() -> None:
"""
Demonstrate a bimodal dataset.
"""

```
data = [1, 1, 2, 2, 3, 4]

print("\nMultiple Modes")
print("-" * 30)
print(f"Data: {data}")
print(f"Modes: {modes(data)}")
```

# =============================================================================

# 14. MODE AND OUTLIERS

# =============================================================================

def outlier_example() -> None:
"""
Demonstrate that the mode is generally unaffected by an
isolated numerical outlier.

```
Example:

    [10, 10, 10, 20, 20, 1000]

Mode:
    10

The outlier 1000 does not change the most frequent value.
"""

data = [10, 10, 10, 20, 20, 1000]

print("\nMode and Outliers")
print("-" * 30)
print(f"Data: {data}")
print(f"Mode: {mode(data)}")
```

# =============================================================================

# 15. MEAN VS MEDIAN VS MODE

# =============================================================================

def compare_central_tendency(data: Sequence[float]) -> None:
"""
Compare mean, median, and mode.

```
This function uses the standard library for the mean and
median while calculating mode manually.

Parameters
----------
data : Sequence[float]
    Numerical dataset.
"""

if not data:
    raise ValueError("Dataset cannot be empty.")

sorted_data = sorted(data)

# Mean
mean_value = sum(data) / len(data)

# Median
n = len(sorted_data)

if n % 2 == 1:
    median_value = sorted_data[n // 2]
else:
    middle_1 = sorted_data[n // 2 - 1]
    middle_2 = sorted_data[n // 2]
    median_value = (middle_1 + middle_2) / 2

# Mode
mode_values = modes(data)

print("\nCentral Tendency Comparison")
print("=" * 50)
print(f"Data   : {list(data)}")
print(f"Mean   : {mean_value}")
print(f"Median : {median_value}")
print(f"Mode   : {mode_values}")
```

# =============================================================================

# 16. MODE USING NUMPY

# =============================================================================

def numpy_mode_example() -> None:
"""
Demonstrate mode-related operations with NumPy.

```
NumPy itself does not provide a general-purpose `mode()`
function equivalent to SciPy's statistics functionality.

Therefore, frequency counting can be performed using
`numpy.unique(..., return_counts=True)`.
"""

try:
    import numpy as np
except ImportError:
    print("\nNumPy is not installed.")
    print("Install it using:")
    print("pip install numpy")
    return

data = np.array([10, 20, 20, 30, 20, 40])

values, counts = np.unique(data, return_counts=True)

max_count = counts.max()

numpy_modes = values[counts == max_count]

print("\nNumPy Mode")
print("-" * 30)
print(f"Data: {data}")
print(f"Mode(s): {numpy_modes}")
print(f"Frequency: {max_count}")
```

# =============================================================================

# 17. MODE USING PANDAS

# =============================================================================

def pandas_mode_example() -> None:
"""
Demonstrate mode using Pandas.

```
Pandas `Series.mode()` returns all modes.
"""

try:
    import pandas as pd
except ImportError:
    print("\nPandas is not installed.")
    print("Install it using:")
    print("pip install pandas")
    return

data = pd.Series([10, 20, 20, 30, 20, 40])

print("\nPandas Mode")
print("-" * 30)
print(data)
print("\nMode:")
print(data.mode())
```

# =============================================================================

# 18. MODE OF A DATAFRAME COLUMN

# =============================================================================

def dataframe_column_mode_example() -> None:
"""
Calculate the mode of a Pandas DataFrame column.
"""

```
try:
    import pandas as pd
except ImportError:
    print("\nPandas is not installed.")
    return

df = pd.DataFrame(
    {
        "Age": [21, 22, 21, 25, 21, 30],
        "City": [
            "Pune",
            "Mumbai",
            "Pune",
            "Delhi",
            "Pune",
            "Mumbai",
        ],
    }
)

print("\nDataFrame")
print(df)

print("\nAge Mode:")
print(df["Age"].mode().tolist())

print("\nCity Mode:")
print(df["City"].mode().tolist())
```

# =============================================================================

# 19. MODE FOR MISSING-VALUE IMPUTATION

# =============================================================================

def mode_imputation_example() -> None:
"""
Demonstrate mode imputation.

```
Mode imputation replaces missing categorical values with
the most frequently occurring category.

Example:

    ["Red", "Blue", None, "Red", "Green", None]

Mode:
    "Red"

Missing values can therefore be replaced with:

    "Red"

This technique is commonly used for categorical features.
"""

data = ["Red", "Blue", None, "Red", "Green", None, "Red"]

non_missing = [value for value in data if value is not None]

most_common = mode(non_missing)

imputed_data = [
    most_common if value is None else value
    for value in data
]

print("\nMode Imputation")
print("-" * 30)
print(f"Original : {data}")
print(f"Mode     : {most_common}")
print(f"Imputed  : {imputed_data}")
```

# =============================================================================

# 20. TRAINING DATA AND MODE IMPUTATION

# =============================================================================

def training_data_imputation_example() -> None:
"""
Explain the correct Machine Learning workflow for mode imputation.

```
Important:

    The mode should be calculated using the training dataset.

The same learned mode can then be applied to validation
and test datasets.

This prevents information from the validation/test sets
from influencing preprocessing.

This is an example of avoiding data leakage.
"""

training_categories = [
    "Red",
    "Blue",
    "Red",
    "Green",
    "Red",
    "Blue",
]

validation_categories = [
    "Green",
    None,
    "Blue",
    None,
]

learned_mode = mode(training_categories)

transformed_validation = [
    learned_mode if value is None else value
    for value in validation_categories
]

print("\nTraining-Only Mode Imputation")
print("-" * 40)
print(f"Training Mode       : {learned_mode}")
print(f"Validation Original : {validation_categories}")
print(f"Validation Imputed  : {transformed_validation}")
```

# =============================================================================

# 21. MODE IN CLASSIFICATION

# =============================================================================

def majority_class(labels: Sequence[Any]) -> Any:
"""
Return the majority class.

```
In classification problems, the most frequent class is
equivalent to the mode of the target labels.

Example:

    ["spam", "ham", "spam", "spam", "ham"]

Mode:
    "spam"

This concept is also related to a majority-class baseline.
"""

return mode(labels)
```

def classification_example() -> None:
"""
Demonstrate mode as the majority class in classification.
"""

```
labels = [
    "Cat",
    "Dog",
    "Cat",
    "Cat",
    "Dog",
    "Cat",
]

print("\nClassification Example")
print("-" * 30)
print(f"Labels: {labels}")
print(f"Majority Class: {majority_class(labels)}")
```

# =============================================================================

# 22. GROUPED MODE

# =============================================================================

def grouped_mode_example() -> None:
"""
Demonstrate mode calculation for groups.

```
This is useful when exploring categorical data
across different groups.
"""

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
            "Sales",
            "Sales",
            "Sales",
        ],
        "Preferred_Language": [
            "Python",
            "Python",
            "Java",
            "Excel",
            "Excel",
            "Python",
            "Java",
            "Python",
        ],
    }
)

print("\nGrouped Mode")
print("-" * 30)
print(df)

grouped_modes = (
    df.groupby("Department")["Preferred_Language"]
    .agg(lambda series: series.mode().tolist())
)

print("\nMode by Department:")
print(grouped_modes)
```

# =============================================================================

# 23. MODE OF EACH DATAFRAME COLUMN

# =============================================================================

def dataframe_modes_example() -> None:
"""
Calculate the mode for each DataFrame column.
"""

```
try:
    import pandas as pd
except ImportError:
    print("\nPandas is not installed.")
    return

df = pd.DataFrame(
    {
        "Age": [20, 21, 21, 22, 21],
        "City": ["Pune", "Mumbai", "Pune", "Pune", "Mumbai"],
        "Grade": ["A", "B", "A", "A", "B"],
    }
)

print("\nDataFrame")
print(df)

print("\nMode of Each Column:")
print(df.mode())
```

# =============================================================================

# 24. WEIGHTED MODE

# =============================================================================

def weighted_mode(
values: Sequence[Any],
weights: Sequence[float],
) -> Any:
"""
Calculate a weighted mode.

```
Unlike the ordinary mode, weighted mode considers the
total weight associated with each value.

Example:

    values  = ["A", "B", "A", "B"]
    weights = [5, 2, 3, 10]

    A total weight = 8
    B total weight = 12

    Weighted mode = "B"

This is not the standard definition of mode. It is a
useful extension when observations have weights.
"""

if not values:
    raise ValueError("Values cannot be empty.")

if len(values) != len(weights):
    raise ValueError("Values and weights must have equal length.")

if any(weight < 0 for weight in weights):
    raise ValueError("Weights cannot be negative.")

weighted_frequency: Dict[Any, float] = {}

for value, weight in zip(values, weights):
    weighted_frequency[value] = (
        weighted_frequency.get(value, 0) + weight
    )

highest_weight = max(weighted_frequency.values())

weighted_modes = [
    value
    for value, total_weight in weighted_frequency.items()
    if total_weight == highest_weight
]

if len(weighted_modes) != 1:
    raise ValueError(
        f"Weighted mode is not unique: {weighted_modes}"
    )

return weighted_modes[0]
```

# =============================================================================

# 25. WEIGHTED MODE EXAMPLE

# =============================================================================

def weighted_mode_example() -> None:
"""
Demonstrate weighted mode.
"""

```
values = ["A", "B", "A", "B"]
weights = [5, 2, 3, 10]

result = weighted_mode(values, weights)

print("\nWeighted Mode")
print("-" * 30)
print(f"Values : {values}")
print(f"Weights: {weights}")
print(f"Mode   : {result}")
```

# =============================================================================

# 26. MODE AND DATA DISTRIBUTIONS

# =============================================================================

def distribution_examples() -> None:
"""
Demonstrate unimodal, bimodal, and multimodal datasets.
"""

```
datasets = {
    "Unimodal": [1, 2, 2, 3, 4],
    "Bimodal": [1, 1, 2, 2, 3],
    "Multimodal": [1, 1, 2, 2, 3, 3, 4],
}

print("\nDistribution Examples")
print("=" * 50)

for name, data in datasets.items():
    print(f"{name:<12}: {data}")
    print(f"Modes       : {modes(data)}")
    print()
```

# =============================================================================

# 27. MODE AND SKEWNESS

# =============================================================================

def mode_skewness_example() -> None:
"""
Explain the relationship between mode and distribution shape.

```
For some idealized unimodal distributions:

    Symmetric distribution:
        Mean ≈ Median ≈ Mode

    Right-skewed distribution:
        Mode < Median < Mean

    Left-skewed distribution:
        Mean < Median < Mode

These are descriptive relationships, not universal rules.
"""

print("\nMode and Distribution Shape")
print("-" * 40)

print("Symmetric:    Mean ≈ Median ≈ Mode")
print("Right-skewed: Mode < Median < Mean")
print("Left-skewed:  Mean < Median < Mode")
```

# =============================================================================

# 28. MODE AND DATA TYPES

# =============================================================================

def data_type_examples() -> None:
"""
Show examples of data types where mode can be useful.
"""

```
examples = {
    "Integers": [1, 2, 2, 3],
    "Floats": [1.5, 2.5, 2.5, 3.5],
    "Strings": ["A", "B", "A"],
    "Booleans": [True, False, True],
    "Categories": ["Low", "Medium", "Low"],
}

print("\nMode Across Data Types")
print("=" * 50)

for data_type, data in examples.items():
    print(f"{data_type:<12}: {data} -> {modes(data)}")
```

# =============================================================================

# 29. MODE FOR DISCRETE DATA

# =============================================================================

def discrete_data_example() -> None:
"""
Mode is particularly intuitive for discrete values.

```
Example:

    Number of purchases:
        [1, 2, 2, 2, 3, 4]

    Mode:
        2
"""

data = [1, 2, 2, 2, 3, 4]

print("\nDiscrete Data")
print("-" * 30)
print(f"Data: {data}")
print(f"Mode: {mode(data)}")
```

# =============================================================================

# 30. MODE FOR NOMINAL DATA

# =============================================================================

def nominal_data_example() -> None:
"""
Demonstrate mode with nominal categories.

```
Nominal categories have labels but no inherent numerical order.

Example:

    ["Pune", "Mumbai", "Pune", "Delhi"]

Mode:
    "Pune"
"""

cities = ["Pune", "Mumbai", "Pune", "Delhi", "Pune"]

print("\nNominal Data")
print("-" * 30)
print(f"Cities: {cities}")
print(f"Mode  : {mode(cities)}")
```

# =============================================================================

# 31. MODE VS MEAN

# =============================================================================

def mode_vs_mean_example() -> None:
"""
Show why mode can be more appropriate than mean for categories.

```
Example:

    Cities:
        ["Pune", "Mumbai", "Pune"]

Taking an arithmetic mean of city names has no meaning.

The mode, however, is:

    "Pune"
"""

cities = ["Pune", "Mumbai", "Pune"]

print("\nMode vs Mean")
print("-" * 30)
print(f"Categories: {cities}")
print(f"Mode      : {mode(cities)}")
print("Arithmetic mean is not defined for city names.")
```

# =============================================================================

# 32. MODE AS A BASELINE FOR CLASSIFICATION

# =============================================================================

def classification_baseline_example() -> None:
"""
Demonstrate the majority-class baseline concept.

```
If a training dataset contains:

    80% Class A
    20% Class B

Always predicting Class A gives an accuracy of 80% on
data having the same class distribution.

This does not mean the model is useful; it is simply a
baseline for comparison.

In imbalanced classification, metrics such as precision,
recall, F1-score, ROC-AUC, or PR-AUC may provide additional
information depending on the problem.
"""

labels = ["A"] * 8 + ["B"] * 2

majority = mode(labels)

print("\nClassification Baseline")
print("-" * 40)
print(f"Labels: {labels}")
print(f"Majority Class: {majority}")
print(f"Baseline Accuracy on this dataset: {labels.count(majority) / len(labels):.2%}")
```

# =============================================================================

# 33. MODE FOR MISSING NUMERICAL VALUES

# =============================================================================

def numerical_mode_imputation_example() -> None:
"""
Mode can technically be used for numerical missing values,
especially when the numerical variable is discrete.

```
However, for continuous numerical variables, mean or median
is often considered depending on the distribution and
preprocessing strategy.

The choice should be based on the data and the modeling
pipeline rather than applying one method universally.
"""

data = [1, 2, None, 2, 3, None, 2]

non_missing = [value for value in data if value is not None]

learned_mode = mode(non_missing)

imputed = [
    learned_mode if value is None else value
    for value in data
]

print("\nNumerical Mode Imputation")
print("-" * 40)
print(f"Original: {data}")
print(f"Mode: {learned_mode}")
print(f"Imputed: {imputed}")
```

# =============================================================================

# 34. COMMON MISTAKES

# =============================================================================

def common_mistakes() -> None:
"""
Print common mistakes when working with mode.
"""

```
print("\nCommon Mistakes")
print("=" * 50)

mistakes = [
    "1. Assuming every dataset has exactly one mode.",
    "2. Ignoring multiple modes.",
    "3. Calculating mode from an empty dataset.",
    "4. Treating categorical values as numerical values.",
    "5. Using test data to learn an imputation value.",
    "6. Assuming mode is always better than mean or median.",
    "7. Ignoring missing-value representation.",
    "8. Assuming the mode describes the entire distribution.",
    "9. Confusing majority class with model quality.",
    "10. Ignoring class imbalance in classification.",
]

for mistake in mistakes:
    print(mistake)
```

# =============================================================================

# 35. QUICK REFERENCE

# =============================================================================

def quick_reference() -> None:
"""
Print a quick reference for the mode.
"""

```
print("\nMode Quick Reference")
print("=" * 50)

reference = {
    "Definition": "Most frequently occurring value",
    "Works with numbers": "Yes",
    "Works with categories": "Yes",
    "Works with strings": "Yes",
    "Affected by outliers": "Usually not directly",
    "Can have multiple modes": "Yes",
    "One mode": "Unimodal",
    "Two modes": "Bimodal",
    "More than two": "Multimodal",
    "Common ML use": "Categorical missing-value imputation",
    "Classification use": "Majority class / baseline",
}

for key, value in reference.items():
    print(f"{key:<30}: {value}")
```

# =============================================================================

# 36. COMPLETE MODE ANALYSIS

# =============================================================================

def complete_mode_analysis(data: Sequence[Any]) -> Dict[str, Any]:
"""
Return a complete mode analysis as a dictionary.

```
Parameters
----------
data : Sequence[Any]

Returns
-------
Dict[str, Any]
    Analysis containing frequencies, modes, and distribution type.
"""

if not data:
    raise ValueError("Cannot analyze an empty dataset.")

frequencies = frequency_table(data)
all_modes = modes(data)

if len(all_modes) == 1:
    distribution_type = "Unimodal"
elif len(all_modes) == 2:
    distribution_type = "Bimodal"
else:
    distribution_type = "Multimodal"

return {
    "data": list(data),
    "frequency_table": frequencies,
    "modes": all_modes,
    "highest_frequency": max(frequencies.values()),
    "number_of_modes": len(all_modes),
    "distribution_type": distribution_type,
}
```

# =============================================================================

# 37. MAIN FUNCTION

# =============================================================================

def main() -> None:
"""
Run all major mode examples.
"""

```
print("=" * 70)
print("MODE IN STATISTICS FOR MACHINE LEARNING")
print("=" * 70)

# Basic numerical example
data = [1, 2, 2, 3, 4, 2, 5]

analyze_mode(data)

# Numerical data
numerical_mode_example()

# Categorical data
categorical_mode_example()

# Strings
string_mode_example()

# Boolean values
boolean_mode_example()

# Multiple modes
multiple_modes_example()

# Outliers
outlier_example()

# Central tendency
compare_central_tendency(
    [10, 10, 10, 20, 20, 100]
)

# NumPy
numpy_mode_example()

# Pandas
pandas_mode_example()

# DataFrame column mode
dataframe_column_mode_example()

# Missing-value imputation
mode_imputation_example()

# Training-only imputation
training_data_imputation_example()

# Classification
classification_example()

# Grouped mode
grouped_mode_example()

# DataFrame modes
dataframe_modes_example()

# Weighted mode
weighted_mode_example()

# Distribution types
distribution_examples()

# Distribution shape
mode_skewness_example()

# Data types
data_type_examples()

# Discrete data
discrete_data_example()

# Nominal data
nominal_data_example()

# Mode vs mean
mode_vs_mean_example()

# Classification baseline
classification_baseline_example()

# Numerical imputation
numerical_mode_imputation_example()

# Common mistakes
common_mistakes()

# Quick reference
quick_reference()

# Complete analysis
result = complete_mode_analysis(
    ["Python", "Java", "Python", "C++", "Python"]
)

print("\nComplete Analysis Result")
print("=" * 50)

for key, value in result.items():
    print(f"{key:<20}: {value}")
```

# =============================================================================

# 38. PROGRAM ENTRY POINT

# =============================================================================

if **name** == "**main**":
main()
"""

# Key Takeaways

1. The mode is the most frequently occurring value.

2. Mode works with both numerical and categorical data.

3. A dataset can have:
   - One mode
   - Two modes
   - Multiple modes

4. Mode is particularly useful for categorical variables.

5. Mode is commonly used for categorical missing-value
   imputation.

6. In Machine Learning preprocessing, learn the mode from
   the training data and apply that learned value to other
   datasets.

7. The mode of classification labels represents the majority
   class.

8. Mode is different from mean and median.

9. Outliers generally do not directly affect the mode unless
   they occur frequently enough to change the most common value.

10. Always consider the data type and distribution before
    choosing a measure of central tendency.
    """
