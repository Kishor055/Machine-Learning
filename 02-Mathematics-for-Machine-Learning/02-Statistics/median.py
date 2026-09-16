"""
Median for Machine Learning
===========================

File:
    02-Mathematics-for-Machine-Learning/02-Statistics/median.py

Purpose:
    Learn the median as a measure of central tendency and
    understand why it is useful for skewed data, outliers,
    data preprocessing, and Machine Learning.

Topics Covered:
    1. What is the median?
    2. Ordered data
    3. Odd number of observations
    4. Even number of observations
    5. Manual median calculation
    6. Median without modifying original data
    7. Median with negative values
    8. Median with decimal values
    9. Median and outliers
    10. Mean vs median
    11. Median for skewed distributions
    12. Weighted median
    13. Grouped median
    14. Running median
    15. Rolling median
    16. NumPy median
    17. Pandas median
    18. Median imputation
    19. Median normalization
    20. Median Absolute Deviation (MAD)
    21. Robust statistics
    22. Median in Machine Learning
    23. Practical examples

Requirements:
    Python 3.x

Optional:
    numpy
    pandas

Author:
    Kishor Patil
"""


# ============================================================
# 1. WHAT IS THE MEDIAN?
# ============================================================

"""
The median is a measure of central tendency.

It is the middle value of an ordered dataset.

The first step when calculating the median is:

    1. Sort the data.
    2. Find the middle position.

There are two cases.

ODD NUMBER OF OBSERVATIONS
--------------------------

If n is odd:

                    n + 1
    Median position = ---
                       2

Example:

    [10, 20, 30, 40, 50]

    Middle value = 30

    Median = 30


EVEN NUMBER OF OBSERVATIONS
---------------------------

If n is even:

    Median =
        (two middle values) / 2

Example:

    [10, 20, 30, 40]

    Middle values:
        20 and 30

    Median =
        (20 + 30) / 2
        = 25

Important:

    The median is resistant to extreme values.

This makes it especially useful for skewed datasets and
datasets containing outliers.
"""


# ============================================================
# 2. BASIC MEDIAN — ODD NUMBER OF VALUES
# ============================================================

values_odd = [
    10,
    20,
    30,
    40,
    50
]

sorted_values_odd = sorted(
    values_odd
)

median_odd = sorted_values_odd[
    len(sorted_values_odd) // 2
]

print("\n--- Median: Odd Number of Values ---")

print(
    "Original Data:",
    values_odd
)

print(
    "Sorted Data:",
    sorted_values_odd
)

print(
    "Median:",
    median_odd
)


# ============================================================
# 3. BASIC MEDIAN — EVEN NUMBER OF VALUES
# ============================================================

values_even = [
    10,
    20,
    30,
    40
]

sorted_values_even = sorted(
    values_even
)

middle_right = (
    len(sorted_values_even) // 2
)

middle_left = (
    middle_right - 1
)

median_even = (
    sorted_values_even[middle_left]
    + sorted_values_even[middle_right]
) / 2

print("\n--- Median: Even Number of Values ---")

print(
    "Original Data:",
    values_even
)

print(
    "Sorted Data:",
    sorted_values_even
)

print(
    "Middle Values:",
    sorted_values_even[middle_left],
    "and",
    sorted_values_even[middle_right]
)

print(
    "Median:",
    median_even
)


# ============================================================
# 4. MANUAL MEDIAN FUNCTION
# ============================================================

def median(values):
    """
    Calculate the median manually.

    Steps:

        1. Check that the dataset is not empty.
        2. Sort the values.
        3. Find the middle value.
        4. If the number of observations is even,
           average the two middle values.

    Parameters:
        values : list
            Numerical observations.

    Returns:
        float
            Median value.

    Raises:
        ValueError:
            If the dataset is empty.
    """

    if len(values) == 0:
        raise ValueError(
            "Cannot calculate the median of an empty dataset."
        )

    sorted_values = sorted(
        values
    )

    n = len(sorted_values)

    middle = n // 2

    if n % 2 == 1:
        return sorted_values[middle]

    return (
        sorted_values[middle - 1]
        + sorted_values[middle]
    ) / 2


print("\n--- Manual Median Function ---")

data = [
    7,
    2,
    9,
    4,
    5
]

print(
    "Data:",
    data
)

print(
    "Median:",
    median(data)
)


# ============================================================
# 5. MEDIAN DOES NOT REQUIRE ORIGINAL DATA TO BE SORTED
# ============================================================

unsorted_data = [
    50,
    10,
    40,
    20,
    30
]

print("\n--- Unsorted Data ---")

print(
    "Original:",
    unsorted_data
)

print(
    "Median:",
    median(
        unsorted_data
    )
)

"""
The median depends on the ordered position of values,
not on the order in which the observations were originally
stored.

Therefore:

    [50, 10, 40, 20, 30]

and:

    [10, 20, 30, 40, 50]

have the same median.
"""


# ============================================================
# 6. MEDIAN WITHOUT MODIFYING ORIGINAL DATA
# ============================================================

def median_preserve_data(values):
    """
    Calculate the median without modifying the original list.

    sorted(values) creates a sorted copy rather than sorting
    the original list in place.
    """

    if len(values) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    ordered = sorted(
        values
    )

    n = len(ordered)

    if n % 2 == 1:
        return ordered[n // 2]

    return (
        ordered[n // 2 - 1]
        + ordered[n // 2]
    ) / 2


original_data = [
    40,
    10,
    30,
    20
]

original_copy = original_data.copy()

median_value = median_preserve_data(
    original_data
)

print("\n--- Original Data Preservation ---")

print(
    "Original Data:",
    original_data
)

print(
    "Median:",
    median_value
)

print(
    "Original Data Unchanged:",
    original_data == original_copy
)


# ============================================================
# 7. MEDIAN WITH NEGATIVE VALUES
# ============================================================

negative_values = [
    -20,
    -10,
    0,
    10,
    20
]

print("\n--- Median With Negative Values ---")

print(
    "Data:",
    negative_values
)

print(
    "Median:",
    median(
        negative_values
    )
)


# ============================================================
# 8. MEDIAN WITH DECIMAL VALUES
# ============================================================

decimal_values = [
    10.5,
    20.25,
    15.75,
    30.0,
    25.5
]

print("\n--- Median With Decimal Values ---")

print(
    "Data:",
    decimal_values
)

print(
    "Median:",
    median(
        decimal_values
    )
)


# ============================================================
# 9. EMPTY DATASET
# ============================================================

print("\n--- Empty Dataset ---")

try:
    median([])

except ValueError as error:
    print(
        "Error:",
        error
    )


# ============================================================
# 10. MEDIAN STEP BY STEP
# ============================================================

def median_step_by_step(values):
    """
    Display the median calculation step by step.
    """

    if len(values) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    ordered = sorted(
        values
    )

    n = len(ordered)

    print("\n--- Step-by-Step Median ---")

    print(
        "Original Data:",
        values
    )

    print(
        "Sorted Data:",
        ordered
    )

    print(
        "Number of Observations:",
        n
    )

    if n % 2 == 1:

        middle_index = n // 2

        print(
            "Middle Index:",
            middle_index
        )

        print(
            "Middle Value:",
            ordered[middle_index]
        )

        return ordered[middle_index]

    left_index = n // 2 - 1
    right_index = n // 2

    left_value = ordered[left_index]
    right_value = ordered[right_index]

    result = (
        left_value + right_value
    ) / 2

    print(
        "Left Middle Value:",
        left_value
    )

    print(
        "Right Middle Value:",
        right_value
    )

    print(
        "Median:",
        result
    )

    return result


median_step_by_step([
    50,
    10,
    30,
    20,
    40
])


median_step_by_step([
    10,
    20,
    30,
    40
])


# ============================================================
# 11. MEDIAN AND OUTLIERS
# ============================================================

data_without_outlier = [
    10,
    12,
    13,
    15,
    16
]

data_with_outlier = [
    10,
    12,
    13,
    15,
    100
]

median_without_outlier = median(
    data_without_outlier
)

median_with_outlier = median(
    data_with_outlier
)

print("\n--- Effect of Outliers ---")

print(
    "Median without outlier:",
    median_without_outlier
)

print(
    "Median with outlier:",
    median_with_outlier
)

"""
Notice that the extreme value 100 does not dramatically change
the median.

Compare this with the arithmetic mean.

Mean:

    [10, 12, 13, 15, 16]

versus:

    [10, 12, 13, 15, 100]

The mean changes substantially.

The median changes much less.

Therefore:

    Median → robust to extreme values
    Mean   → sensitive to extreme values
"""


# ============================================================
# 12. MEAN VS MEDIAN
# ============================================================

def compare_mean_and_median(values):
    """
    Compare arithmetic mean and median.
    """

    if len(values) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    mean_value = (
        sum(values) / len(values)
    )

    median_value = median(
        values
    )

    return {
        "mean": mean_value,
        "median": median_value,
        "difference": (
            mean_value - median_value
        )
    }


print("\n--- Mean vs Median ---")

skewed_data = [
    10,
    12,
    13,
    15,
    100
]

comparison = compare_mean_and_median(
    skewed_data
)

print(
    "Data:",
    skewed_data
)

print(
    "Mean:",
    comparison["mean"]
)

print(
    "Median:",
    comparison["median"]
)

print(
    "Mean - Median:",
    comparison["difference"]
)


# ============================================================
# 13. SKEWED DISTRIBUTIONS
# ============================================================

"""
The relationship between mean and median can provide useful
information about distribution shape.

Positive/right skew:

    Mean > Median

    A few large values pull the mean toward the right tail.


Negative/left skew:

    Mean < Median

    A few small values pull the mean toward the left tail.


Approximately symmetric distribution:

    Mean ≈ Median

These are useful descriptive clues, but mean-versus-median
alone should not be treated as a complete test of skewness.
"""


# ============================================================
# 14. POSITIVELY SKEWED DATA
# ============================================================

right_skewed = [
    10,
    11,
    12,
    13,
    100
]

right_mean = (
    sum(right_skewed)
    / len(right_skewed)
)

right_median = median(
    right_skewed
)

print("\n--- Positive / Right Skew ---")

print(
    "Data:",
    right_skewed
)

print(
    "Mean:",
    right_mean
)

print(
    "Median:",
    right_median
)


# ============================================================
# 15. NEGATIVELY SKEWED DATA
# ============================================================

left_skewed = [
    1,
    88,
    90,
    92,
    94
]

left_mean = (
    sum(left_skewed)
    / len(left_skewed)
)

left_median = median(
    left_skewed
)

print("\n--- Negative / Left Skew ---")

print(
    "Data:",
    left_skewed
)

print(
    "Mean:",
    left_mean
)

print(
    "Median:",
    left_median
)


# ============================================================
# 16. MEDIAN AND PERCENTILES
# ============================================================

"""
The median is also the 50th percentile.

Therefore:

    Median = P50

It divides ordered observations into two halves.

Similarly:

    Q1 = 25th percentile
    Q2 = 50th percentile = Median
    Q3 = 75th percentile

This connects median to quartiles and the interquartile range.
"""


# ============================================================
# 17. PERCENTILE FUNCTION
# ============================================================

def percentile_nearest_rank(values, percentile):
    """
    Calculate a simple nearest-rank percentile.

    This implementation is intended for learning.

    percentile:
        0 to 100

    Note:
        Statistical libraries may use different interpolation
        or percentile definitions. Always check the method when
        exact compatibility matters.
    """

    if len(values) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    if not 0 <= percentile <= 100:
        raise ValueError(
            "Percentile must be between 0 and 100."
        )

    ordered = sorted(
        values
    )

    rank = (
        percentile / 100
    ) * len(ordered)

    index = max(
        0,
        int(rank + 0.999999) - 1
    )

    index = min(
        index,
        len(ordered) - 1
    )

    return ordered[index]


print("\n--- Percentiles ---")

percentile_data = [
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80
]

print(
    "Data:",
    percentile_data
)

print(
    "P50:",
    percentile_nearest_rank(
        percentile_data,
        50
    )
)

print(
    "P25:",
    percentile_nearest_rank(
        percentile_data,
        25
    )
)

print(
    "P75:",
    percentile_nearest_rank(
        percentile_data,
        75
    )
)


# ============================================================
# 18. QUARTILES
# ============================================================

def quartiles(values):
    """
    Calculate Q1, Q2, and Q3 using the median-of-halves method.

    This is a teaching implementation.

    Different statistical libraries can use different quartile
    interpolation conventions.
    """

    if len(values) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    ordered = sorted(
        values
    )

    n = len(ordered)

    q2 = median(
        ordered
    )

    if n % 2 == 0:

        lower_half = ordered[
            :n // 2
        ]

        upper_half = ordered[
            n // 2:
        ]

    else:

        lower_half = ordered[
            :n // 2
        ]

        upper_half = ordered[
            n // 2 + 1:
        ]

    q1 = median(
        lower_half
    ) if lower_half else q2

    q3 = median(
        upper_half
    ) if upper_half else q2

    return q1, q2, q3


print("\n--- Quartiles ---")

quartile_data = [
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80
]

q1, q2, q3 = quartiles(
    quartile_data
)

print(
    "Q1:",
    q1
)

print(
    "Q2 / Median:",
    q2
)

print(
    "Q3:",
    q3
)


# ============================================================
# 19. INTERQUARTILE RANGE
# ============================================================

def interquartile_range(values):
    """
    Calculate the interquartile range.

    Formula:

        IQR = Q3 - Q1

    IQR measures the spread of the middle 50% of observations.
    """

    q1, _, q3 = quartiles(
        values
    )

    return q3 - q1


print("\n--- Interquartile Range ---")

print(
    "Data:",
    quartile_data
)

print(
    "IQR:",
    interquartile_range(
        quartile_data
    )
)

"""
The IQR is robust to extreme observations.

It is commonly used for:

    - Outlier detection
    - Box plots
    - Robust descriptive statistics
"""


# ============================================================
# 20. OUTLIER DETECTION USING IQR
# ============================================================

def detect_iqr_outliers(values):
    """
    Detect potential outliers using the 1.5 × IQR rule.

    Lower Bound:
        Q1 - 1.5 × IQR

    Upper Bound:
        Q3 + 1.5 × IQR
    """

    q1, _, q3 = quartiles(
        values
    )

    iqr = q3 - q1

    lower_bound = (
        q1 - 1.5 * iqr
    )

    upper_bound = (
        q3 + 1.5 * iqr
    )

    outliers = [
        value
        for value in values
        if (
            value < lower_bound
            or
            value > upper_bound
        )
    ]

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outliers": outliers
    }


print("\n--- IQR Outlier Detection ---")

outlier_dataset = [
    10,
    12,
    13,
    14,
    15,
    16,
    18,
    100
]

outlier_result = detect_iqr_outliers(
    outlier_dataset
)

for key, value in outlier_result.items():
    print(
        f"{key}: {value}"
    )


# ============================================================
# 21. WEIGHTED MEDIAN
# ============================================================

"""
A weighted median is different from a weighted mean.

The weighted median is a value where at least half of the
total weight lies at or below the value and at least half lies
at or above the value.

Steps:

    1. Sort values.
    2. Associate each value with a weight.
    3. Calculate total weight.
    4. Find the point where cumulative weight reaches
       at least half of total weight.

Weighted medians can be useful when observations have
different importance.
"""


# ============================================================
# 22. WEIGHTED MEDIAN FUNCTION
# ============================================================

def weighted_median(values, weights):
    """
    Calculate the weighted median.

    Parameters:
        values : list
            Numerical values.

        weights : list
            Non-negative weights.

    Returns:
        float
            Weighted median.
    """

    if len(values) != len(weights):
        raise ValueError(
            "Values and weights must have equal lengths."
        )

    if len(values) == 0:
        raise ValueError(
            "Values cannot be empty."
        )

    if any(
        weight < 0
        for weight in weights
    ):
        raise ValueError(
            "Weights must be non-negative."
        )

    total_weight = sum(
        weights
    )

    if total_weight == 0:
        raise ValueError(
            "Total weight must be greater than zero."
        )

    pairs = sorted(
        zip(values, weights),
        key=lambda pair: pair[0]
    )

    cumulative_weight = 0

    for value, weight in pairs:

        cumulative_weight += weight

        if cumulative_weight >= (
            total_weight / 2
        ):
            return value

    return pairs[-1][0]


print("\n--- Weighted Median ---")

weighted_values = [
    10,
    20,
    30,
    40
]

weighted_weights = [
    1,
    1,
    6,
    2
]

print(
    "Values:",
    weighted_values
)

print(
    "Weights:",
    weighted_weights
)

print(
    "Weighted Median:",
    weighted_median(
        weighted_values,
        weighted_weights
    )
)


# ============================================================
# 23. RUNNING MEDIAN
# ============================================================

"""
A running median calculates the median after each new
observation arrives.

Example:

    Data stream:

        10
        20
        5
        30

Running medians:

        [10]       → 10
        [10,20]    → 15
        [10,20,5]  → 10
        [10,20,5,30] → 15

Running medians are useful for:

    - Streaming data
    - Sensor monitoring
    - Real-time analytics
    - Robust online statistics
"""


# ============================================================
# 24. RUNNING MEDIAN FUNCTION
# ============================================================

def running_median(values):
    """
    Calculate the median after each new observation.

    This simple implementation sorts the accumulated data
    at every step.

    Complexity:
        O(n² log n) in a straightforward worst-case analysis.

    More advanced implementations use two heaps to achieve
    approximately O(log n) insertion and O(1) median lookup.
    """

    medians = []
    current_values = []

    for value in values:

        current_values.append(
            value
        )

        medians.append(
            median(
                current_values
            )
        )

    return medians


print("\n--- Running Median ---")

stream_data = [
    10,
    20,
    5,
    30,
    15
]

print(
    "Data:",
    stream_data
)

print(
    "Running Medians:",
    running_median(
        stream_data
    )
)


# ============================================================
# 25. ROLLING / MOVING MEDIAN
# ============================================================

"""
A rolling median calculates the median over a fixed-size
sliding window.

Example:

    Data:
        [10, 20, 5, 30, 15]

    Window = 3

    Window 1:
        [10, 20, 5]
        Median = 10

    Window 2:
        [20, 5, 30]
        Median = 20

    Window 3:
        [5, 30, 15]
        Median = 15

Rolling medians are especially useful for noisy time-series
data because they are less sensitive to isolated spikes than
simple moving averages.
"""


# ============================================================
# 26. ROLLING MEDIAN FUNCTION
# ============================================================

def rolling_median(values, window):
    """
    Calculate a rolling median.
    """

    if window <= 0:
        raise ValueError(
            "Window must be greater than zero."
        )

    if window > len(values):
        raise ValueError(
            "Window cannot exceed dataset length."
        )

    return [
        median(
            values[i:i + window]
        )
        for i in range(
            len(values) - window + 1
        )
    ]


print("\n--- Rolling Median ---")

time_series = [
    10,
    20,
    5,
    30,
    15
]

print(
    "Data:",
    time_series
)

print(
    "Rolling Median:",
    rolling_median(
        time_series,
        window=3
    )
)


# ============================================================
# 27. NUMPY MEDIAN
# ============================================================

try:
    import numpy as np

    numpy_data = np.array([
        50,
        10,
        40,
        20,
        30
    ])

    numpy_median = np.median(
        numpy_data
    )

    print("\n--- NumPy Median ---")

    print(
        "Data:",
        numpy_data
    )

    print(
        "Median:",
        numpy_median
    )

except ImportError:
    print(
        "\nNumPy is not installed."
        "\nInstall it using: pip install numpy"
    )


# ============================================================
# 28. NUMPY MEDIAN BY AXIS
# ============================================================

try:
    import numpy as np

    matrix = np.array([
        [10, 20, 30],
        [20, 30, 40],
        [30, 40, 50]
    ])

    print("\n--- NumPy Median by Axis ---")

    print(
        "Matrix:"
    )

    print(matrix)

    print(
        "\nOverall Median:",
        np.median(matrix)
    )

    print(
        "Column Medians:",
        np.median(
            matrix,
            axis=0
        )
    )

    print(
        "Row Medians:",
        np.median(
            matrix,
            axis=1
        )
    )

except ImportError:
    pass


# ============================================================
# 29. PANDAS MEDIAN
# ============================================================

try:
    import pandas as pd

    student_scores = pd.DataFrame({
        "Math": [
            80,
            70,
            95,
            85,
            75
        ],
        "Science": [
            75,
            85,
            90,
            80,
            70
        ],
        "English": [
            90,
            80,
            85,
            95,
            88
        ]
    })

    print("\n--- Pandas Median ---")

    print(
        student_scores
    )

    print(
        "\nColumn Medians:"
    )

    print(
        student_scores.median()
    )

except ImportError:
    print(
        "\nPandas is not installed."
        "\nInstall it using: pip install pandas"
    )


# ============================================================
# 30. MEDIAN BY GROUP
# ============================================================

try:
    import pandas as pd

    student_data = pd.DataFrame({
        "Department": [
            "CS",
            "CS",
            "CS",
            "IT",
            "IT",
            "AI",
            "AI"
        ],
        "Score": [
            80,
            90,
            70,
            65,
            85,
            95,
            88
        ]
    })

    group_medians = (
        student_data
        .groupby("Department")["Score"]
        .median()
    )

    print("\n--- Median by Group ---")

    print(
        student_data
    )

    print(
        "\nMedian Score by Department:"
    )

    print(
        group_medians
    )

except ImportError:
    pass


# ============================================================
# 31. MEDIAN IMPUTATION
# ============================================================

"""
Median imputation replaces missing numerical values with the
median of the available training observations.

Example:

    Age:

        [20, 25, None, 30, 100]

The median is less affected by the extreme value 100 than
the mean.

Advantages:

    - Simple
    - Robust to outliers
    - Useful for skewed numerical variables

Limitations:

    - Reduces variability
    - Can distort distributions
    - Does not model the missingness mechanism
    - May be inappropriate when missingness carries meaning

Machine Learning rule:

    Learn the imputation statistic from training data only.

Do not calculate the median using validation or test data when
building a predictive pipeline, because that can introduce
information leakage.
"""


# ============================================================
# 32. MEDIAN IMPUTATION FUNCTION
# ============================================================

def median_imputation(values):
    """
    Replace None values with the median of available values.

    Teaching implementation.
    """

    valid_values = [
        value
        for value in values
        if value is not None
    ]

    if len(valid_values) == 0:
        raise ValueError(
            "No valid values available for imputation."
        )

    median_value = median(
        valid_values
    )

    return [
        median_value
        if value is None
        else value
        for value in values
    ]


print("\n--- Median Imputation ---")

missing_data = [
    20,
    25,
    None,
    30,
    100
]

print(
    "Original Data:",
    missing_data
)

print(
    "Imputed Data:",
    median_imputation(
        missing_data
    )
)


# ============================================================
# 33. MEAN VS MEDIAN IMPUTATION
# ============================================================

def compare_imputation(values):
    """
    Compare mean and median replacement values.
    """

    valid_values = [
        value
        for value in values
        if value is not None
    ]

    if len(valid_values) == 0:
        raise ValueError(
            "No valid observations available."
        )

    mean_value = (
        sum(valid_values)
        / len(valid_values)
    )

    median_value = median(
        valid_values
    )

    return {
        "mean_imputation_value": mean_value,
        "median_imputation_value": median_value
    }


print("\n--- Mean vs Median Imputation ---")

imputation_data = [
    20,
    25,
    None,
    30,
    100
]

imputation_comparison = (
    compare_imputation(
        imputation_data
    )
)

for key, value in (
    imputation_comparison.items()
):
    print(
        f"{key}: {value}"
    )


# ============================================================
# 34. MEDIAN NORMALIZATION
# ============================================================

"""
Median-based scaling can be useful in robust preprocessing.

One common robust transformation is:

    x' = (x - median(X)) / IQR(X)

This is closely related to robust scaling.

Compared with standardization:

    Standardization:
        uses mean and standard deviation.

    Robust scaling:
        uses median and IQR.

Robust scaling is often useful when numerical features contain
outliers.
"""


# ============================================================
# 35. ROBUST SCALING FUNCTION
# ============================================================

def robust_scale(values):
    """
    Scale data using median and IQR.

    Formula:

        x' = (x - median) / IQR
    """

    if len(values) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    median_value = median(
        values
    )

    iqr = interquartile_range(
        values
    )

    if iqr == 0:
        raise ValueError(
            "Cannot robust-scale data with zero IQR."
        )

    return [
        (value - median_value) / iqr
        for value in values
    ]


print("\n--- Robust Scaling ---")

robust_data = [
    10,
    12,
    13,
    15,
    16,
    100
]

print(
    "Original:",
    robust_data
)

print(
    "Robust Scaled:",
    robust_scale(
        robust_data
    )
)


# ============================================================
# 36. MEDIAN ABSOLUTE DEVIATION (MAD)
# ============================================================

"""
Median Absolute Deviation (MAD) is a robust measure of
dispersion.

Formula:

    MAD = Median(
              |xi - Median(X)|
          )

Steps:

    1. Calculate the median.
    2. Calculate absolute deviations from the median.
    3. Take the median of those deviations.

MAD is less sensitive to extreme values than variance and
standard deviation.
"""


# ============================================================
# 37. MAD FUNCTION
# ============================================================

def median_absolute_deviation(values):
    """
    Calculate Median Absolute Deviation (MAD).
    """

    if len(values) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    median_value = median(
        values
    )

    absolute_deviations = [
        abs(
            value - median_value
        )
        for value in values
    ]

    return median(
        absolute_deviations
    )


print("\n--- Median Absolute Deviation ---")

mad_data = [
    10,
    12,
    13,
    15,
    16,
    100
]

print(
    "Data:",
    mad_data
)

print(
    "Median:",
    median(
        mad_data
    )
)

print(
    "MAD:",
    median_absolute_deviation(
        mad_data
    )
)


# ============================================================
# 38. ROBUST Z-SCORE USING MAD
# ============================================================

"""
A robust z-score can be constructed using the median and MAD.

A commonly used modified z-score is:

             0.6745 × (x - Median)
    M = -----------------------------
                     MAD

The constant 0.6745 makes the score approximately comparable
to a standard normal z-score under normality.

Very large absolute modified z-scores can indicate potential
outliers.

Thresholds should be treated as heuristics rather than
universal laws.
"""


# ============================================================
# 39. MODIFIED Z-SCORE FUNCTION
# ============================================================

def modified_z_scores(values):
    """
    Calculate modified z-scores using median and MAD.
    """

    if len(values) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    median_value = median(
        values
    )

    mad = median_absolute_deviation(
        values
    )

    if mad == 0:
        raise ValueError(
            "Modified z-scores are undefined when MAD is zero."
        )

    return [
        0.6745 * (
            value - median_value
        ) / mad
        for value in values
    ]


print("\n--- Modified Z-Scores ---")

modified_scores_data = [
    10,
    12,
    13,
    15,
    16,
    100
]

print(
    "Data:",
    modified_scores_data
)

print(
    "Modified Z-Scores:",
    modified_z_scores(
        modified_scores_data
    )
)


# ============================================================
# 40. MEDIAN AND MACHINE LEARNING
# ============================================================

"""
The median appears in many Machine Learning workflows.

Applications include:

    1. Exploratory Data Analysis

        Understanding the center of skewed features.


    2. Missing-value handling

        Median imputation for numerical features.


    3. Robust preprocessing

        Median and IQR can be used for robust scaling.


    4. Outlier analysis

        Median and IQR are used in common outlier detection
        techniques.


    5. Robust statistics

        Median-based methods are less sensitive to extreme
        observations.


    6. Evaluation

        Median Absolute Error is useful when robustness to
        outliers is desirable.


    7. Time-series processing

        Rolling medians can reduce the influence of isolated
        spikes.
"""


# ============================================================
# 41. MEDIAN ABSOLUTE ERROR
# ============================================================

"""
Median Absolute Error measures the median of absolute
prediction errors.

Formula:

    MedAE =
        Median(|yi - ŷi|)

It is more robust to extreme errors than MSE and can be useful
when a few very large errors should not dominate the metric.
"""


# ============================================================
# 42. MEDIAN ABSOLUTE ERROR FUNCTION
# ============================================================

def median_absolute_error(
    actual,
    predicted
):
    """
    Calculate Median Absolute Error.
    """

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted values must have equal lengths."
        )

    if len(actual) == 0:
        raise ValueError(
            "Datasets cannot be empty."
        )

    absolute_errors = [
        abs(
            actual_value - predicted_value
        )
        for actual_value, predicted_value
        in zip(
            actual,
            predicted
        )
    ]

    return median(
        absolute_errors
    )


print("\n--- Median Absolute Error ---")

actual = [
    100,
    200,
    300,
    400
]

predicted = [
    110,
    190,
    1000,
    390
]

print(
    "Actual:",
    actual
)

print(
    "Predicted:",
    predicted
)

print(
    "Median Absolute Error:",
    median_absolute_error(
        actual,
        predicted
    )
)


# ============================================================
# 43. MEAN ABSOLUTE ERROR VS MEDIAN ABSOLUTE ERROR
# ============================================================

def mean_absolute_error(
    actual,
    predicted
):
    """
    Calculate Mean Absolute Error.
    """

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted values must have equal lengths."
        )

    errors = [
        abs(
            actual_value - predicted_value
        )
        for actual_value, predicted_value
        in zip(
            actual,
            predicted
        )
    ]

    return sum(errors) / len(errors)


print("\n--- MAE vs Median Absolute Error ---")

print(
    "MAE:",
    mean_absolute_error(
        actual,
        predicted
    )
)

print(
    "Median Absolute Error:",
    median_absolute_error(
        actual,
        predicted
    )
)

"""
MAE:
    Uses the average absolute error.

Median Absolute Error:
    Uses the middle absolute error.

Median Absolute Error can be more robust when the error
distribution contains extreme values.
"""


# ============================================================
# 44. MEDIAN OF BINARY DATA
# ============================================================

"""
For binary data containing only 0 and 1:

    The median indicates which value occupies the middle
    position after sorting.

Example:

    [0, 0, 0, 1, 1]

    Median = 0

If at least half the observations are 1, the median can
become 1 depending on the sample size and exact distribution.
"""

binary_data = [
    0,
    0,
    1,
    1,
    1
]

print("\n--- Median of Binary Data ---")

print(
    "Data:",
    binary_data
)

print(
    "Median:",
    median(
        binary_data
    )
)


# ============================================================
# 45. MEDIAN OF A DATA STREAM
# ============================================================

"""
For very large streaming datasets, repeatedly sorting the
entire dataset is inefficient.

A common algorithmic solution uses two heaps:

    Max Heap:
        Stores lower half.

    Min Heap:
        Stores upper half.

Maintain:

    size(left) ≈ size(right)

Then:

    Odd number of observations:
        Median = top of larger heap.

    Even number:
        Median = average of both heap tops.

This allows efficient online median computation.
"""


# ============================================================
# 46. MEDIAN AND DATA DISTRIBUTION
# ============================================================

"""
Median is especially useful when:

    - Data is skewed.
    - Data contains outliers.
    - The distribution has a long tail.
    - A robust center is required.

Examples:

    House prices
    Salaries
    Income
    Response times
    Transaction values
    Medical measurements

For highly skewed variables such as income, the median can
often describe the typical observation more robustly than
the mean.
"""


# ============================================================
# 47. PRACTICAL HOUSE PRICE EXAMPLE
# ============================================================

house_prices = [
    200000,
    220000,
    250000,
    280000,
    300000,
    1500000
]

house_mean = (
    sum(house_prices)
    / len(house_prices)
)

house_median = median(
    house_prices
)

print("\n--- House Price Example ---")

print(
    "House Prices:",
    house_prices
)

print(
    "Mean Price:",
    house_mean
)

print(
    "Median Price:",
    house_median
)

"""
A very expensive property can pull the mean upward.

The median remains closer to the center of the ordered
observations.

This illustrates why both statistics should be considered
when analyzing skewed real-world data.
"""


# ============================================================
# 48. PRACTICAL SALARY EXAMPLE
# ============================================================

salaries = [
    25000,
    30000,
    32000,
    35000,
    40000,
    250000
]

salary_mean = (
    sum(salaries)
    / len(salaries)
)

salary_median = median(
    salaries
)

print("\n--- Salary Example ---")

print(
    "Salaries:",
    salaries
)

print(
    "Mean Salary:",
    salary_mean
)

print(
    "Median Salary:",
    salary_median
)


# ============================================================
# 49. MEDIAN IN GROUPED ML DATA
# ============================================================

try:
    import pandas as pd

    employee_data = pd.DataFrame({
        "Department": [
            "Engineering",
            "Engineering",
            "Engineering",
            "Marketing",
            "Marketing",
            "Marketing"
        ],
        "Salary": [
            50000,
            60000,
            55000,
            40000,
            45000,
            90000
        ]
    })

    department_median_salary = (
        employee_data
        .groupby("Department")["Salary"]
        .median()
    )

    print("\n--- Grouped Median in Data Analysis ---")

    print(
        employee_data
    )

    print(
        "\nMedian Salary by Department:"
    )

    print(
        department_median_salary
    )

except ImportError:
    pass


# ============================================================
# 50. IMPORTANT PROPERTIES
# ============================================================

"""
Important properties of the median:

1. The median is based on order.

2. The median is resistant to extreme observations.

3. The median divides the ordered dataset into two parts.

4. The median is the 50th percentile.

5. The median does not generally equal the mean.

6. For symmetric distributions:

       Mean ≈ Median

7. For right-skewed distributions:

       Mean > Median

8. For left-skewed distributions:

       Mean < Median

9. The median minimizes the sum of absolute deviations.

       Σ|xi - median|

   is minimized by a median.

10. The median is especially useful for ordinal data because
    it depends on ordering rather than requiring meaningful
    numerical distances.
"""


# ============================================================
# 51. MEDIAN MINIMIZES ABSOLUTE DEVIATION
# ============================================================

candidate_values = [
    8,
    10,
    12,
    14,
    100
]

candidate_median = median(
    candidate_values
)

deviation_from_median = sum(
    abs(
        value - candidate_median
    )
    for value in candidate_values
)

deviation_from_mean = sum(
    abs(
        value -
        (
            sum(candidate_values)
            / len(candidate_values)
        )
    )
    for value in candidate_values
)

print("\n--- Absolute Deviation Property ---")

print(
    "Data:",
    candidate_values
)

print(
    "Median:",
    candidate_median
)

print(
    "Sum of Absolute Deviations from Median:",
    deviation_from_median
)

print(
    "Sum of Absolute Deviations from Mean:",
    deviation_from_mean
)


# ============================================================
# 52. COMMON MISTAKES
# ============================================================

"""
Common mistakes:

1. Forgetting to sort the data.

       Median depends on ordered position.


2. Choosing the wrong middle value for even-sized data.

       Average the two middle observations.


3. Assuming median equals mean.

       They are different measures.


4. Assuming median is always better than mean.

       The appropriate measure depends on the distribution
       and analytical goal.


5. Ignoring missing values.

       Missing-value handling should be explicit.


6. Ignoring different percentile definitions.

       Statistical libraries can use different interpolation
       conventions.


7. Using test-set information to calculate median imputation.

       In ML, learn preprocessing statistics from training
       data only.


8. Treating median as a complete description of a dataset.

       A center measure should be considered together with
       dispersion and distribution shape.


9. Assuming the median cannot be affected by outliers.

       The median is robust, not completely immune to changes
       caused by extreme observations.


10. Confusing rolling median with rolling mean.

        Rolling median uses the middle value.
        Rolling mean uses the arithmetic average.
"""


# ============================================================
# 53. QUICK REFERENCE
# ============================================================

"""
MEDIAN
------

Sort the data.

If n is odd:

                 n + 1
    Position =   -----
                   2


If n is even:

    Median =
        (middle_left + middle_right) / 2


PERCENTILE
----------

    Median = 50th percentile


QUARTILES
---------

    Q1 = 25th percentile
    Q2 = 50th percentile = Median
    Q3 = 75th percentile


INTERQUARTILE RANGE
-------------------

    IQR = Q3 - Q1


IQR OUTLIER RULE
----------------

    Lower Bound = Q1 - 1.5 × IQR

    Upper Bound = Q3 + 1.5 × IQR


MEDIAN ABSOLUTE DEVIATION
-------------------------

    MAD =
        Median(|xi - Median(X)|)


MEDIAN ABSOLUTE ERROR
---------------------

    MedAE =
        Median(|yi - ŷi|)


ROBUST SCALING
--------------

              x - Median(X)
    x' = -----------------------
                 IQR(X)
"""


# ============================================================
# 54. COMPLETE MEDIAN ANALYSIS
# ============================================================

def analyze_median(values):
    """
    Perform a compact median-based statistical analysis.
    """

    if len(values) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    mean_value = (
        sum(values)
        / len(values)
    )

    median_value = median(
        values
    )

    q1, q2, q3 = quartiles(
        values
    )

    iqr = q3 - q1

    mad = median_absolute_deviation(
        values
    )

    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": mean_value,
        "median": median_value,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "iqr": iqr,
        "mad": mad
    }


print("\n--- Complete Median Analysis ---")

analysis = analyze_median([
    10,
    12,
    13,
    15,
    16,
    100
])

for key, value in analysis.items():
    print(
        f"{key}: {value}"
    )


# ============================================================
# 55. MAIN FUNCTION
# ============================================================

def main():
    """
    Main entry point for median examples.
    """

    print("\n" + "=" * 65)
    print("Median for Machine Learning")
    print("=" * 65)

    values = [
        10,
        20,
        30,
        40,
        50
    ]

    print("\nDataset:")
    print(values)

    print(
        "\nMean:",
        sum(values) / len(values)
    )

    print(
        "Median:",
        median(values)
    )

    print(
        "Q1, Q2, Q3:",
        quartiles(values)
    )

    print(
        "IQR:",
        interquartile_range(values)
    )

    print(
        "MAD:",
        median_absolute_deviation(values)
    )

    print(
        "\nRolling Median:",
        rolling_median(
            values,
            window=3
        )
    )

    print("\n" + "=" * 65)
    print("Median examples completed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
```
