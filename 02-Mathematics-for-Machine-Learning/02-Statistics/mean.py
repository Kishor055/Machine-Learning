"""
Mean for Machine Learning
==========================

File:
    02-Mathematics-for-Machine-Learning/02-Statistics/mean.py

Purpose:
    Learn different types of means and understand how they are
    used in Statistics, Data Analysis, and Machine Learning.

Topics Covered:
    1. What is a mean?
    2. Arithmetic mean
    3. Manual calculation
    4. Mean with negative values
    5. Mean with decimals
    6. Mean of an empty dataset
    7. Mean and outliers
    8. Weighted mean
    9. Geometric mean
    10. Harmonic mean
    11. Trimmed mean
    12. Moving mean
    13. NumPy mean
    14. Pandas mean
    15. Mean by groups
    16. Mean imputation
    17. Mean normalization
    18. Mean Absolute Error (MAE)
    19. Mean Squared Error (MSE)
    20. Mean in Machine Learning
    21. Practical examples

Requirements:
    Python 3.x

Optional:
    numpy
    pandas

Author:
    Kishor Patil
"""


# ============================================================
# 1. WHAT IS THE MEAN?
# ============================================================

"""
The arithmetic mean is one of the most common measures of
central tendency.

It represents the average value of a dataset.

Formula:

                 x₁ + x₂ + x₃ + ... + xₙ
    Mean =      ----------------------------
                            n

Equivalent notation:

                       1
    x̄ =              --- Σxᵢ
                       n

Where:

    x̄  = arithmetic mean
    xᵢ = individual observation
    n  = number of observations

Example:

    Data = [10, 20, 30]

    Mean = (10 + 20 + 30) / 3
         = 60 / 3
         = 20
"""


# ============================================================
# 2. BASIC ARITHMETIC MEAN
# ============================================================

numbers = [10, 20, 30, 40, 50]

mean_value = sum(numbers) / len(numbers)

print("\n--- Basic Arithmetic Mean ---")

print("Data:", numbers)
print("Mean:", mean_value)


# ============================================================
# 3. MANUAL MEAN FUNCTION
# ============================================================

def arithmetic_mean(values):
    """
    Calculate the arithmetic mean manually.

    Formula:

        Mean = Σx / n

    Parameters:
        values : list
            Numerical observations.

    Returns:
        float
            Arithmetic mean.

    Raises:
        ValueError:
            If the dataset is empty.
    """

    if len(values) == 0:
        raise ValueError(
            "Cannot calculate the mean of an empty dataset."
        )

    return sum(values) / len(values)


print("\n--- Manual Mean Function ---")

data = [5, 10, 15, 20, 25]

print(
    "Data:",
    data
)

print(
    "Mean:",
    arithmetic_mean(data)
)


# ============================================================
# 4. STEP-BY-STEP MEAN CALCULATION
# ============================================================

def mean_step_by_step(values):
    """
    Display the arithmetic mean calculation step by step.
    """

    if len(values) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    total = sum(values)
    count = len(values)
    mean = total / count

    print("\n--- Step-by-Step Mean ---")

    print("Values:", values)
    print("Sum:", total)
    print("Number of observations:", count)
    print("Mean:", mean)

    return mean


mean_step_by_step(
    [10, 20, 30, 40, 50]
)


# ============================================================
# 5. MEAN WITH NEGATIVE VALUES
# ============================================================

negative_values = [
    -10,
    -5,
    0,
    5,
    10
]

print("\n--- Mean With Negative Values ---")

print(
    "Data:",
    negative_values
)

print(
    "Mean:",
    arithmetic_mean(
        negative_values
    )
)

"""
Negative values are handled normally.

Example:

    (-10 + -5 + 0 + 5 + 10) / 5

    = 0 / 5

    = 0
"""


# ============================================================
# 6. MEAN WITH DECIMAL VALUES
# ============================================================

decimal_values = [
    10.5,
    20.25,
    30.75,
    40.0
]

print("\n--- Mean With Decimal Values ---")

print(
    "Data:",
    decimal_values
)

print(
    "Mean:",
    arithmetic_mean(
        decimal_values
    )
)


# ============================================================
# 7. MEAN OF AN EMPTY DATASET
# ============================================================

print("\n--- Empty Dataset ---")

try:
    arithmetic_mean([])

except ValueError as error:
    print("Error:", error)


# ============================================================
# 8. MEAN AND OUTLIERS
# ============================================================

normal_data = [
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

mean_without_outlier = arithmetic_mean(
    normal_data
)

mean_with_outlier = arithmetic_mean(
    data_with_outlier
)

print("\n--- Effect of Outliers ---")

print(
    "Mean without outlier:",
    mean_without_outlier
)

print(
    "Mean with outlier:",
    mean_with_outlier
)

"""
The arithmetic mean is sensitive to extreme values.

Example:

    [10, 12, 13, 15, 16]

has a mean near the center of the data.

Adding:

    100

substantially increases the mean.

This is why the median can sometimes be more representative
when the dataset contains strong outliers.
"""


# ============================================================
# 9. MEAN AND TOTAL SUM
# ============================================================

values = [
    20,
    30,
    40,
    50
]

mean = arithmetic_mean(values)

reconstructed_sum = mean * len(values)

print("\n--- Mean and Sum ---")

print("Original Sum:", sum(values))
print("Mean:", mean)
print(
    "Mean × Number of Values:",
    reconstructed_sum
)

"""
Important identity:

    Sum = Mean × Number of Observations

Therefore:

    Σx = n × x̄
"""


# ============================================================
# 10. WEIGHTED MEAN
# ============================================================

"""
A weighted mean gives different importance to different
observations.

Formula:

                     Σ(wᵢxᵢ)
    Weighted Mean = ---------
                       Σwᵢ

Where:

    xᵢ = value
    wᵢ = weight

Example:

    Assignment = 80, weight = 20%
    Midterm    = 70, weight = 30%
    Final      = 90, weight = 50%

Weighted mean:

    (80 × 0.20)
    + (70 × 0.30)
    + (90 × 0.50)
"""


# ============================================================
# 11. MANUAL WEIGHTED MEAN FUNCTION
# ============================================================

def weighted_mean(values, weights):
    """
    Calculate the weighted arithmetic mean.

    Formula:

                 Σ(wᵢxᵢ)
        Mean = ----------
                   Σwᵢ

    Parameters:
        values : list
            Numerical observations.

        weights : list
            Importance assigned to each observation.

    Returns:
        float
            Weighted mean.
    """

    if len(values) != len(weights):
        raise ValueError(
            "Values and weights must have the same length."
        )

    if len(values) == 0:
        raise ValueError(
            "Values cannot be empty."
        )

    total_weight = sum(weights)

    if total_weight == 0:
        raise ValueError(
            "The sum of weights cannot be zero."
        )

    return sum(
        value * weight
        for value, weight in zip(
            values,
            weights
        )
    ) / total_weight


print("\n--- Weighted Mean ---")

scores = [
    80,
    70,
    90
]

weights = [
    0.20,
    0.30,
    0.50
]

print("Scores:", scores)
print("Weights:", weights)

print(
    "Weighted Mean:",
    weighted_mean(
        scores,
        weights
    )
)


# ============================================================
# 12. WEIGHTED MEAN IN MACHINE LEARNING
# ============================================================

"""
Weighted averages appear in many ML workflows.

Examples:

    - Weighted loss functions
    - Class weighting
    - Sample weighting
    - Weighted evaluation metrics
    - Ensemble methods
    - Importance-weighted datasets

For example, observations from an underrepresented class
may receive larger weights during model training.
"""


# ============================================================
# 13. GEOMETRIC MEAN
# ============================================================

"""
The geometric mean is useful when values are multiplied
together or represent growth factors.

Formula:

                       n
    Geometric Mean = √(x₁ × x₂ × ... × xₙ)

For positive values.

Example:

    [2, 8]

    Geometric Mean = √(2 × 8)
                   = √16
                   = 4

Geometric mean is commonly useful for:

    - Growth rates
    - Multiplicative changes
    - Ratios
    - Investment returns
"""


# ============================================================
# 14. MANUAL GEOMETRIC MEAN
# ============================================================

def geometric_mean(values):
    """
    Calculate the geometric mean.

    Values must be positive.

    Formula:

                    n
        GM = (Πxᵢ)^(1/n)
    """

    if len(values) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    if any(value <= 0 for value in values):
        raise ValueError(
            "Geometric mean requires positive values."
        )

    product = 1

    for value in values:
        product *= value

    return product ** (
        1 / len(values)
    )


print("\n--- Geometric Mean ---")

growth_factors = [
    2,
    8
]

print(
    "Data:",
    growth_factors
)

print(
    "Geometric Mean:",
    geometric_mean(
        growth_factors
    )
)


# ============================================================
# 15. HARMONIC MEAN
# ============================================================

"""
The harmonic mean is useful for averaging rates and ratios.

Formula:

                       n
    Harmonic Mean = ---------
                    Σ(1/xᵢ)

Example:

    Data = [2, 4]

             2
    HM = -----------
         1/2 + 1/4

       = 2 / 0.75

       ≈ 2.67

The harmonic mean is particularly useful when averaging
rates such as:

    - Speed
    - Throughput
    - Ratios
    - Precision/recall-related measures

The F1-score is the harmonic mean of precision and recall.
"""


# ============================================================
# 16. MANUAL HARMONIC MEAN
# ============================================================

def harmonic_mean(values):
    """
    Calculate the harmonic mean.

    Formula:

                    n
        HM = ----------------
             Σ(1 / xᵢ)

    Values must be non-zero.
    """

    if len(values) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    if any(value == 0 for value in values):
        raise ValueError(
            "Harmonic mean is undefined when a value is zero."
        )

    reciprocal_sum = sum(
        1 / value
        for value in values
    )

    return len(values) / reciprocal_sum


print("\n--- Harmonic Mean ---")

rates = [
    2,
    4
]

print(
    "Data:",
    rates
)

print(
    "Harmonic Mean:",
    harmonic_mean(
        rates
    )
)


# ============================================================
# 17. ARITHMETIC VS GEOMETRIC VS HARMONIC MEAN
# ============================================================

comparison_data = [
    2,
    8
]

print("\n--- Comparing Different Means ---")

print(
    "Arithmetic Mean:",
    arithmetic_mean(
        comparison_data
    )
)

print(
    "Geometric Mean:",
    geometric_mean(
        comparison_data
    )
)

print(
    "Harmonic Mean:",
    harmonic_mean(
        comparison_data
    )
)

"""
For positive values:

    Arithmetic Mean
        ≥
    Geometric Mean
        ≥
    Harmonic Mean

Equality occurs when all values are equal.
"""


# ============================================================
# 18. TRIMMED MEAN
# ============================================================

"""
A trimmed mean removes a specified percentage of the smallest
and largest observations before calculating the mean.

It can reduce the influence of extreme observations.

Example:

    Data:

        [1, 2, 3, 4, 5, 100]

    The value 100 is an extreme observation.

A trimmed mean may remove extreme values before averaging.

Important:
    The trimming percentage must be chosen carefully.
"""


# ============================================================
# 19. MANUAL TRIMMED MEAN
# ============================================================

def trimmed_mean(values, proportion=0.1):
    """
    Calculate a simple symmetric trimmed mean.

    Parameters:
        values : list
            Numerical observations.

        proportion : float
            Fraction removed from each end.

            Example:
                0.10 → remove 10% from each side.

    Returns:
        float
            Trimmed mean.
    """

    if len(values) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    if not 0 <= proportion < 0.5:
        raise ValueError(
            "Proportion must be between 0 and 0.5."
        )

    sorted_values = sorted(values)

    trim_count = int(
        len(values) * proportion
    )

    if trim_count * 2 >= len(values):
        raise ValueError(
            "Too many values would be removed."
        )

    trimmed_values = sorted_values[
        trim_count:
        len(values) - trim_count
    ]

    return arithmetic_mean(
        trimmed_values
    )


print("\n--- Trimmed Mean ---")

outlier_data = [
    1,
    2,
    3,
    4,
    5,
    100
]

print(
    "Original Data:",
    outlier_data
)

print(
    "Arithmetic Mean:",
    arithmetic_mean(
        outlier_data
    )
)

print(
    "Trimmed Mean:",
    trimmed_mean(
        outlier_data,
        proportion=0.16
    )
)


# ============================================================
# 20. MOVING MEAN
# ============================================================

"""
A moving mean, also called a moving average, calculates the
mean over a sliding window.

Example:

    Data:

        [10, 20, 30, 40, 50]

    Window = 3

    First mean:

        (10 + 20 + 30) / 3
        = 20

    Second mean:

        (20 + 30 + 40) / 3
        = 30

    Third mean:

        (30 + 40 + 50) / 3
        = 40

Moving averages are useful for:

    - Time-series analysis
    - Trend detection
    - Noise reduction
    - Financial data
    - Sensor data
"""


# ============================================================
# 21. MOVING MEAN FUNCTION
# ============================================================

def moving_mean(values, window):
    """
    Calculate a simple moving average.

    Parameters:
        values : list
            Numerical observations.

        window : int
            Number of observations in each window.

    Returns:
        list
            Moving mean values.
    """

    if window <= 0:
        raise ValueError(
            "Window must be greater than zero."
        )

    if window > len(values):
        raise ValueError(
            "Window cannot be larger than dataset."
        )

    return [
        arithmetic_mean(
            values[i:i + window]
        )
        for i in range(
            len(values) - window + 1
        )
    ]


print("\n--- Moving Mean ---")

time_series = [
    10,
    20,
    30,
    40,
    50
]

print(
    "Data:",
    time_series
)

print(
    "Moving Mean:",
    moving_mean(
        time_series,
        window=3
    )
)


# ============================================================
# 22. NUMPY MEAN
# ============================================================

try:
    import numpy as np

    values_np = np.array([
        10,
        20,
        30,
        40,
        50
    ])

    numpy_mean = np.mean(
        values_np
    )

    print("\n--- NumPy Mean ---")

    print(
        "Data:",
        values_np
    )

    print(
        "Mean:",
        numpy_mean
    )

except ImportError:
    print(
        "\nNumPy is not installed."
        "\nInstall it using: pip install numpy"
    )


# ============================================================
# 23. NUMPY MEAN BY AXIS
# ============================================================

try:
    import numpy as np

    matrix = np.array([
        [10, 20, 30],
        [20, 30, 40],
        [30, 40, 50]
    ])

    print("\n--- NumPy Mean by Axis ---")

    print("Matrix:")
    print(matrix)

    print(
        "\nOverall Mean:",
        np.mean(matrix)
    )

    print(
        "Column Means:",
        np.mean(
            matrix,
            axis=0
        )
    )

    print(
        "Row Means:",
        np.mean(
            matrix,
            axis=1
        )
    )

except ImportError:
    pass


# ============================================================
# 24. PANDAS MEAN
# ============================================================

try:
    import pandas as pd

    data = pd.DataFrame({
        "Math": [80, 70, 90, 85, 75],
        "Science": [75, 85, 95, 80, 70],
        "English": [90, 80, 85, 95, 88]
    })

    print("\n--- Pandas Mean ---")

    print(data)

    print(
        "\nColumn Means:"
    )

    print(
        data.mean()
    )

except ImportError:
    print(
        "\nPandas is not installed."
        "\nInstall it using: pip install pandas"
    )


# ============================================================
# 25. MEAN BY GROUP
# ============================================================

try:
    import pandas as pd

    student_data = pd.DataFrame({
        "Department": [
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
            85,
            95,
            88
        ]
    })

    group_means = (
        student_data
        .groupby("Department")["Score"]
        .mean()
    )

    print("\n--- Mean by Group ---")

    print(student_data)

    print(
        "\nMean Score by Department:"
    )

    print(group_means)

except ImportError:
    pass


# ============================================================
# 26. MEAN IMPUTATION
# ============================================================

"""
Mean imputation is a simple missing-value strategy.

Example:

    Age:

        [20, 25, NaN, 30, 35]

Calculate the mean of available values:

    Mean = (20 + 25 + 30 + 35) / 4

Then replace the missing value with the mean.

Advantages:

    - Simple
    - Fast
    - Easy to implement

Limitations:

    - Can reduce variance
    - Can distort relationships
    - May not be appropriate for skewed data
    - Ignores uncertainty in the missing value

For production ML pipelines, missing-value strategies should
be selected based on the data and model requirements.
"""


# ============================================================
# 27. MANUAL MEAN IMPUTATION
# ============================================================

def mean_imputation(values):
    """
    Replace None values with the mean of available values.

    This is a teaching implementation.

    For real ML pipelines, use a preprocessing pipeline that
    prevents information leakage between training and test data.
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

    mean = arithmetic_mean(
        valid_values
    )

    return [
        mean if value is None else value
        for value in values
    ]


print("\n--- Mean Imputation ---")

missing_data = [
    20,
    25,
    None,
    30,
    35
]

print(
    "Original:",
    missing_data
)

print(
    "Imputed:",
    mean_imputation(
        missing_data
    )
)


# ============================================================
# 28. MEAN NORMALIZATION
# ============================================================

"""
Mean normalization transforms data using its mean and range.

A common formula is:

              x - mean(x)
    x' = -----------------------
             max(x) - min(x)

This is different from standardization (z-score scaling).

Mean normalization is one possible preprocessing technique,
although the appropriate scaling method depends on the ML
algorithm and dataset.
"""


# ============================================================
# 29. MEAN NORMALIZATION FUNCTION
# ============================================================

def mean_normalization(values):
    """
    Apply mean normalization.

    Formula:

              x - mean(x)
        -------------------------
          max(x) - min(x)
    """

    if len(values) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    minimum = min(values)
    maximum = max(values)
    mean = arithmetic_mean(values)

    data_range = maximum - minimum

    if data_range == 0:
        raise ValueError(
            "Mean normalization is undefined for constant data."
        )

    return [
        (value - mean) / data_range
        for value in values
    ]


print("\n--- Mean Normalization ---")

normalization_data = [
    10,
    20,
    30,
    40,
    50
]

print(
    "Original:",
    normalization_data
)

print(
    "Normalized:",
    mean_normalization(
        normalization_data
    )
)


# ============================================================
# 30. MEAN ABSOLUTE ERROR (MAE)
# ============================================================

"""
The Mean Absolute Error measures the average absolute difference
between actual and predicted values.

Formula:

                    1
    MAE =           --- Σ|yi - ŷi|
                    n

Where:

    yi  = actual value
    ŷi  = predicted value

Example:

    Actual:
        [100, 200, 300]

    Predicted:
        [110, 190, 310]

Errors:

    [10, -10, 10]

Absolute errors:

    [10, 10, 10]

MAE:

    (10 + 10 + 10) / 3
    = 10
"""


# ============================================================
# 31. MANUAL MAE
# ============================================================

def mean_absolute_error(actual, predicted):
    """
    Calculate Mean Absolute Error.
    """

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted values must have equal lengths."
        )

    if len(actual) == 0:
        raise ValueError(
            "Datasets cannot be empty."
        )

    return arithmetic_mean([
        abs(
            actual_value - predicted_value
        )
        for actual_value, predicted_value
        in zip(actual, predicted)
    ])


print("\n--- Mean Absolute Error ---")

actual = [
    100,
    200,
    300
]

predicted = [
    110,
    190,
    310
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
    "MAE:",
    mean_absolute_error(
        actual,
        predicted
    )
)


# ============================================================
# 32. MEAN SQUARED ERROR (MSE)
# ============================================================

"""
Mean Squared Error measures the average squared prediction error.

Formula:

                    1
    MSE =           --- Σ(yi - ŷi)²
                    n

Squaring errors makes large errors contribute more strongly.

MSE is widely used as:

    - Regression evaluation metric
    - Regression loss function
"""


# ============================================================
# 33. MANUAL MSE
# ============================================================

def mean_squared_error(actual, predicted):
    """
    Calculate Mean Squared Error.
    """

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted values must have equal lengths."
        )

    if len(actual) == 0:
        raise ValueError(
            "Datasets cannot be empty."
        )

    squared_errors = [
        (
            actual_value - predicted_value
        ) ** 2
        for actual_value, predicted_value
        in zip(actual, predicted)
    ]

    return arithmetic_mean(
        squared_errors
    )


print("\n--- Mean Squared Error ---")

print(
    "MSE:",
    mean_squared_error(
        actual,
        predicted
    )
)


# ============================================================
# 34. ROOT MEAN SQUARED ERROR
# ============================================================

def root_mean_squared_error(
    actual,
    predicted
):
    """
    Calculate Root Mean Squared Error.

    Formula:

        RMSE = √MSE
    """

    mse = mean_squared_error(
        actual,
        predicted
    )

    return mse ** 0.5


print("\n--- Root Mean Squared Error ---")

print(
    "RMSE:",
    root_mean_squared_error(
        actual,
        predicted
    )
)


# ============================================================
# 35. MEAN IN MACHINE LEARNING
# ============================================================

"""
The mean appears throughout Machine Learning.

Examples:

    1. Data preprocessing

        Mean imputation
        Mean normalization


    2. Feature scaling

        Mean and standard deviation are used in
        standardization.


    3. Model evaluation

        MAE
        MSE
        RMSE


    4. Optimization

        Mean losses are commonly used to aggregate
        errors across training examples.


    5. Statistics

        Sample means estimate population means.


    6. Baseline models

        For some regression problems, predicting the
        training-set mean provides a simple baseline.
"""


# ============================================================
# 36. MEAN AS A REGRESSION BASELINE
# ============================================================

def mean_baseline_predictions(
    training_targets,
    test_size
):
    """
    Create a simple mean-prediction baseline.

    The mean is calculated from training targets only.

    This is important in Machine Learning because using test
    targets to calculate the baseline would leak information
    from the test set.
    """

    if len(training_targets) == 0:
        raise ValueError(
            "Training targets cannot be empty."
        )

    if test_size < 0:
        raise ValueError(
            "Test size cannot be negative."
        )

    mean_target = arithmetic_mean(
        training_targets
    )

    return [
        mean_target
        for _ in range(test_size)
    ]


print("\n--- Mean Regression Baseline ---")

training_targets = [
    100,
    120,
    140,
    160,
    180
]

baseline_predictions = (
    mean_baseline_predictions(
        training_targets,
        test_size=3
    )
)

print(
    "Training Targets:",
    training_targets
)

print(
    "Baseline Predictions:",
    baseline_predictions
)


# ============================================================
# 37. ONLINE / RUNNING MEAN
# ============================================================

"""
For very large datasets, storing every observation may not
always be desirable.

A running mean can be updated incrementally.

If the current mean after n observations is:

    mean_n

and a new value x arrives:

    mean_(n+1) =
        mean_n + (x - mean_n) / (n + 1)

This avoids recomputing the mean from the beginning.

This idea is useful in:

    - Streaming data
    - Online learning
    - Real-time systems
    - Large-scale data processing
"""


# ============================================================
# 38. RUNNING MEAN FUNCTION
# ============================================================

def running_mean(values):
    """
    Calculate running means incrementally.
    """

    if len(values) == 0:
        return []

    means = []

    current_mean = 0.0

    for count, value in enumerate(
        values,
        start=1
    ):
        current_mean += (
            value - current_mean
        ) / count

        means.append(
            current_mean
        )

    return means


print("\n--- Running Mean ---")

stream_data = [
    10,
    20,
    30,
    40,
    50
]

print(
    "Data:",
    stream_data
)

print(
    "Running Means:",
    running_mean(
        stream_data
    )
)


# ============================================================
# 39. MEAN OF BATCHES
# ============================================================

"""
Suppose a dataset is divided into batches.

Batch 1:
    [10, 20, 30]

Batch 2:
    [40, 50]

You cannot generally calculate the overall mean by simply
averaging the two batch means unless the batches have equal
sizes.

Correct approach:

    Overall Mean =
        Total Sum / Total Count

or use a weighted average of batch means.
"""


# ============================================================
# 40. BATCH MEAN EXAMPLE
# ============================================================

batch_1 = [
    10,
    20,
    30
]

batch_2 = [
    40,
    50
]

batch_1_mean = arithmetic_mean(
    batch_1
)

batch_2_mean = arithmetic_mean(
    batch_2
)

incorrect_mean = (
    batch_1_mean +
    batch_2_mean
) / 2

correct_mean = arithmetic_mean(
    batch_1 + batch_2
)

print("\n--- Mean of Batches ---")

print(
    "Batch 1 Mean:",
    batch_1_mean
)

print(
    "Batch 2 Mean:",
    batch_2_mean
)

print(
    "Simple Average of Batch Means:",
    incorrect_mean
)

print(
    "Correct Overall Mean:",
    correct_mean
)

"""
The simple average of batch means is incorrect here because
the batches contain different numbers of observations.

Weighted calculation:

    Overall Mean =
        (Batch1 Mean × Batch1 Size
         + Batch2 Mean × Batch2 Size)
        /
        Total Size
"""


# ============================================================
# 41. MEAN OF BINARY DATA
# ============================================================

"""
For binary data containing only 0 and 1:

    Mean = proportion of 1s

Example:

    [1, 0, 1, 1, 0]

    Mean = 3 / 5
         = 0.6

Therefore, the mean of a binary variable can represent the
fraction or proportion of observations belonging to class 1.
"""

binary_data = [
    1,
    0,
    1,
    1,
    0
]

print("\n--- Mean of Binary Data ---")

print(
    "Binary Data:",
    binary_data
)

print(
    "Mean:",
    arithmetic_mean(
        binary_data
    )
)

print(
    "Percentage of 1s:",
    arithmetic_mean(
        binary_data
    ) * 100,
    "%"
)


# ============================================================
# 42. MEAN AND STANDARDIZATION
# ============================================================

def standardize_using_mean(values):
    """
    Standardize values using mean and population standard
    deviation.

    Formula:

                    x - μ
        z = ---------------------
                       σ

    The mean is subtracted so the transformed data is centered
    around zero.
    """

    if len(values) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    mean = arithmetic_mean(
        values
    )

    variance = arithmetic_mean([
        (value - mean) ** 2
        for value in values
    ])

    standard_deviation = (
        variance ** 0.5
    )

    if standard_deviation == 0:
        raise ValueError(
            "Cannot standardize constant data."
        )

    return [
        (value - mean) /
        standard_deviation
        for value in values
    ]


print("\n--- Mean and Standardization ---")

standardization_data = [
    10,
    20,
    30,
    40,
    50
]

print(
    "Original:",
    standardization_data
)

print(
    "Standardized:",
    standardize_using_mean(
        standardization_data
    )
)


# ============================================================
# 43. IMPORTANT PROPERTIES OF THE MEAN
# ============================================================

"""
Important properties:

1. The sum of deviations from the arithmetic mean is zero.

       Σ(xᵢ - x̄) = 0


2. The arithmetic mean minimizes the sum of squared deviations.

       Σ(xᵢ - x̄)²

   is minimized when x̄ is the arithmetic mean.


3. Linear transformation:

       If Y = aX + b

       then:

       Mean(Y) = a Mean(X) + b


4. Adding the same constant to every observation:

       Mean(X + c)
           = Mean(X) + c


5. Multiplying every observation by a constant:

       Mean(aX)
           = a Mean(X)
"""


# ============================================================
# 44. VERIFY MEAN DEVIATION PROPERTY
# ============================================================

values = [
    10,
    20,
    30,
    40,
    50
]

mean = arithmetic_mean(
    values
)

deviation_sum = sum(
    value - mean
    for value in values
)

print("\n--- Sum of Deviations ---")

print(
    "Mean:",
    mean
)

print(
    "Sum of Deviations:",
    deviation_sum
)


# ============================================================
# 45. COMMON MISTAKES
# ============================================================

"""
Common mistakes:

1. Dividing by the wrong number.

       Mean = Sum / Number of Observations


2. Forgetting negative values.

       Negative numbers must be included normally.


3. Treating every average as an arithmetic mean.

       Some problems require:

           Weighted mean
           Geometric mean
           Harmonic mean
           Median


4. Ignoring outliers.

       Extreme values can strongly affect the mean.


5. Using the wrong type of mean.

       Rates often require harmonic mean.
       Growth factors often require geometric mean.
       Unequal importance requires weighted mean.


6. Calculating training preprocessing statistics using
   the entire dataset.

       In ML, statistics such as the mean should generally
       be learned from the training data and then applied
       to validation/test data.


7. Averaging batch means without considering batch size.

       Use weighted means when batch sizes differ.


8. Assuming mean always represents the "typical" value.

       For skewed distributions, median may be more
       representative.
"""


# ============================================================
# 46. QUICK REFERENCE
# ============================================================

"""
ARITHMETIC MEAN
---------------

                 Σx
    Mean =       ---
                  n


WEIGHTED MEAN
-------------

                 Σwx
    Mean =       ----
                 Σw


GEOMETRIC MEAN
--------------

                       n
    GM =              √Πx


HARMONIC MEAN
-------------

                   n
    HM =        ---------
                Σ(1/x)


MEAN ABSOLUTE ERROR
-------------------

                 1
    MAE =        --- Σ|y - ŷ|
                 n


MEAN SQUARED ERROR
------------------

                 1
    MSE =        --- Σ(y - ŷ)²
                 n


ROOT MEAN SQUARED ERROR
-----------------------

    RMSE = √MSE


RUNNING MEAN
------------

    mean_new =
        mean_old + (x - mean_old) / n
"""


# ============================================================
# 47. COMPLETE STATISTICAL SUMMARY
# ============================================================

def mean_summary(values):
    """
    Generate a compact summary based on the arithmetic mean.
    """

    if len(values) == 0:
        raise ValueError(
            "Dataset cannot be empty."
        )

    mean = arithmetic_mean(
        values
    )

    return {
        "count": len(values),
        "sum": sum(values),
        "mean": mean,
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values)
    }


print("\n--- Mean Summary ---")

summary = mean_summary([
    10,
    20,
    30,
    40,
    50
])

for key, value in summary.items():
    print(
        f"{key}: {value}"
    )


# ============================================================
# 48. MAIN FUNCTION
# ============================================================

def main():
    """
    Main entry point for the mean examples.
    """

    print("\n" + "=" * 65)
    print("Mean for Machine Learning")
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
        "\nArithmetic Mean:",
        arithmetic_mean(values)
    )

    print(
        "Geometric Mean:",
        geometric_mean(values)
    )

    print(
        "Harmonic Mean:",
        harmonic_mean(values)
    )

    print(
        "\nWeighted Mean:",
        weighted_mean(
            [80, 70, 90],
            [0.20, 0.30, 0.50]
        )
    )

    print(
        "\nMoving Mean:",
        moving_mean(
            values,
            window=3
        )
    )

    print("\n" + "=" * 65)
    print("Mean examples completed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
```
