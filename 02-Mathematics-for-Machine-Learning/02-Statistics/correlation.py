https://chatgpt.com/s/w_6aaa55d12a688191b79c4b6eca193b63

```python
"""
Correlation for Machine Learning
=================================

File:
    02-Mathematics-for-Machine-Learning/02-Statistics/correlation.py

Purpose:
    Learn how correlation measures the strength and direction
    of a linear relationship between two variables.

Topics Covered:
    1. What is correlation?
    2. Positive, negative, and zero correlation
    3. Covariance
    4. Pearson correlation coefficient
    5. Manual correlation calculation
    6. NumPy implementation
    7. Pandas implementation
    8. Correlation matrix
    9. Feature correlation in Machine Learning
    10. Multicollinearity
    11. Correlation vs causation
    12. Practical examples

Requirements:
    Python 3.x

Optional:
    numpy
    pandas

Author:
    Kishor Patil
"""


# ============================================================
# 1. WHAT IS CORRELATION?
# ============================================================

"""
Correlation measures the strength and direction of association
between two variables.

Pearson correlation coefficient:

                Cov(X, Y)
    r = -----------------------------
          σX × σY

The value of r lies between -1 and +1.

    +1 → Perfect positive linear relationship
     0 → No linear correlation
    -1 → Perfect negative linear relationship

Important:
    Correlation measures association, not causation.
"""


# ============================================================
# 2. SIMPLE POSITIVE CORRELATION
# ============================================================

study_hours = [1, 2, 3, 4, 5]
exam_scores = [50, 55, 65, 75, 85]

print("\n--- Positive Correlation ---")
print("Study Hours:", study_hours)
print("Exam Scores:", exam_scores)

"""
As study hours increase, exam scores generally increase.

This represents a positive relationship.
"""


# ============================================================
# 3. SIMPLE NEGATIVE CORRELATION
# ============================================================

temperature = [10, 15, 20, 25, 30]
heating_usage = [95, 85, 70, 50, 30]

print("\n--- Negative Correlation ---")
print("Temperature:", temperature)
print("Heating Usage:", heating_usage)

"""
As temperature increases, heating usage decreases.

This represents a negative relationship.
"""


# ============================================================
# 4. NO CLEAR LINEAR CORRELATION
# ============================================================

x_values = [1, 2, 3, 4, 5]
y_values = [8, 2, 9, 1, 7]

print("\n--- No Clear Linear Correlation ---")
print("X:", x_values)
print("Y:", y_values)

"""
The values do not show a clear linear pattern.

Correlation close to zero means little linear association.
It does NOT necessarily mean that the variables are completely
independent or unrelated.
"""


# ============================================================
# 5. COVARIANCE
# ============================================================

def covariance(x, y):
    """
    Calculate population covariance between two variables.

    Formula:

        Cov(X, Y) =
            (1 / n) Σ[(xi - mean_x)(yi - mean_y)]

    Positive covariance:
        Variables tend to move in the same direction.

    Negative covariance:
        Variables tend to move in opposite directions.

    Important:
        Covariance depends on the scale and units of the
        variables.
    """

    if len(x) != len(y):
        raise ValueError(
            "Both datasets must contain the same number of values."
        )

    if len(x) == 0:
        raise ValueError("Datasets cannot be empty.")

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    covariance_value = sum(
        (xi - mean_x) * (yi - mean_y)
        for xi, yi in zip(x, y)
    ) / len(x)

    return covariance_value


print("\n--- Covariance ---")

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

print("X:", x)
print("Y:", y)
print("Covariance:", covariance(x, y))


# ============================================================
# 6. SAMPLE COVARIANCE
# ============================================================

def sample_covariance(x, y):
    """
    Calculate sample covariance.

    Formula:

        s_xy =
            Σ[(xi - x̄)(yi - ȳ)]
            ------------------
                 n - 1
    """

    if len(x) != len(y):
        raise ValueError(
            "Both datasets must contain the same number of values."
        )

    if len(x) < 2:
        raise ValueError(
            "At least two observations are required."
        )

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    return sum(
        (xi - mean_x) * (yi - mean_y)
        for xi, yi in zip(x, y)
    ) / (len(x) - 1)


print("\n--- Sample Covariance ---")
print("Sample Covariance:", sample_covariance(x, y))


# ============================================================
# 7. PEARSON CORRELATION — MANUAL IMPLEMENTATION
# ============================================================

def pearson_correlation(x, y):
    """
    Calculate Pearson correlation coefficient manually.

    Formula:

                Σ[(xi - x̄)(yi - ȳ)]
        r = -----------------------------
            √[Σ(xi-x̄)² × Σ(yi-ȳ)²]

    Returns:
        A value between -1 and +1.
    """

    if len(x) != len(y):
        raise ValueError(
            "Both datasets must contain the same number of values."
        )

    if len(x) < 2:
        raise ValueError(
            "At least two observations are required."
        )

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    numerator = sum(
        (xi - mean_x) * (yi - mean_y)
        for xi, yi in zip(x, y)
    )

    sum_squared_x = sum(
        (xi - mean_x) ** 2
        for xi in x
    )

    sum_squared_y = sum(
        (yi - mean_y) ** 2
        for yi in y
    )

    denominator = (
        sum_squared_x * sum_squared_y
    ) ** 0.5

    if denominator == 0:
        raise ValueError(
            "Correlation is undefined when one variable "
            "has zero variance."
        )

    return numerator / denominator


print("\n--- Pearson Correlation ---")

correlation = pearson_correlation(x, y)

print("X:", x)
print("Y:", y)
print("Pearson Correlation:", correlation)


# ============================================================
# 8. CORRELATION INTERPRETATION
# ============================================================

def interpret_correlation(r):
    """
    Provide a simple interpretation of Pearson correlation.

    Note:
        These labels are heuristic descriptions rather than
        universal statistical thresholds.
    """

    if r > 0:
        direction = "positive"
    elif r < 0:
        direction = "negative"
    else:
        direction = "zero"

    strength = abs(r)

    if strength >= 0.8:
        level = "strong"
    elif strength >= 0.5:
        level = "moderate"
    elif strength >= 0.3:
        level = "weak"
    else:
        level = "very weak"

    return f"{level} {direction} linear relationship"


print("\n--- Correlation Interpretation ---")

print(
    "r =",
    correlation,
    "→",
    interpret_correlation(correlation)
)


# ============================================================
# 9. DIFFERENT CORRELATION PATTERNS
# ============================================================

examples = {
    "Strong Positive": 0.92,
    "Moderate Positive": 0.65,
    "Weak Positive": 0.35,
    "Near Zero": 0.05,
    "Weak Negative": -0.35,
    "Moderate Negative": -0.65,
    "Strong Negative": -0.92,
}

print("\n--- Correlation Examples ---")

for name, value in examples.items():
    print(
        f"{name:20s}: {value:+.2f}"
    )


# ============================================================
# 10. NUMPY IMPLEMENTATION
# ============================================================

try:
    import numpy as np

    x_np = np.array([1, 2, 3, 4, 5])
    y_np = np.array([2, 4, 6, 8, 10])

    correlation_matrix = np.corrcoef(
        x_np,
        y_np
    )

    numpy_correlation = correlation_matrix[0, 1]

    print("\n--- NumPy Correlation ---")
    print("Correlation Matrix:")
    print(correlation_matrix)

    print(
        "Pearson Correlation:",
        numpy_correlation
    )

except ImportError:
    print(
        "\nNumPy is not installed."
        "\nInstall it using: pip install numpy"
    )


# ============================================================
# 11. PANDAS IMPLEMENTATION
# ============================================================

try:
    import pandas as pd

    data = pd.DataFrame({
        "Study_Hours": [1, 2, 3, 4, 5],
        "Exam_Score": [50, 55, 65, 75, 85]
    })

    pandas_correlation = data["Study_Hours"].corr(
        data["Exam_Score"]
    )

    print("\n--- Pandas Correlation ---")
    print(data)

    print(
        "Pearson Correlation:",
        pandas_correlation
    )

except ImportError:
    print(
        "\nPandas is not installed."
        "\nInstall it using: pip install pandas"
    )


# ============================================================
# 12. CORRELATION MATRIX
# ============================================================

try:
    import pandas as pd

    dataset = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40],
        "Experience": [1, 3, 5, 8, 10],
        "Salary": [25000, 35000, 45000, 60000, 75000],
        "Performance": [60, 65, 75, 85, 90]
    })

    correlation_matrix = dataset.corr()

    print("\n--- Correlation Matrix ---")
    print(correlation_matrix)

except ImportError:
    pass


# ============================================================
# 13. CORRELATION WITH MACHINE LEARNING FEATURES
# ============================================================

"""
Suppose we have:

    Features:
        Age
        Experience
        Salary
        Performance

We can examine correlations to understand relationships
between variables before building a model.

For example:

    Experience ↔ Salary

may have a strong positive correlation.

However, correlation alone does not prove that experience
causes salary to increase.
"""

try:
    print("\n--- ML Feature Correlation ---")

    print(
        dataset.corr()["Salary"]
        .sort_values(ascending=False)
    )

except NameError:
    pass


# ============================================================
# 14. IDENTIFY HIGHLY CORRELATED FEATURES
# ============================================================

def highly_correlated_pairs(
    dataframe,
    threshold=0.8
):
    """
    Find pairs of numerical features whose absolute
    Pearson correlation is greater than or equal to
    the specified threshold.
    """

    correlation_matrix = dataframe.corr(
        numeric_only=True
    )

    columns = correlation_matrix.columns
    pairs = []

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):

            feature_a = columns[i]
            feature_b = columns[j]

            correlation_value = correlation_matrix.loc[
                feature_a,
                feature_b
            ]

            if abs(correlation_value) >= threshold:
                pairs.append(
                    (
                        feature_a,
                        feature_b,
                        correlation_value
                    )
                )

    return pairs


try:
    print("\n--- Highly Correlated Features ---")

    pairs = highly_correlated_pairs(
        dataset,
        threshold=0.8
    )

    for feature_a, feature_b, value in pairs:
        print(
            f"{feature_a} ↔ {feature_b}: "
            f"{value:.3f}"
        )

except NameError:
    pass


# ============================================================
# 15. CORRELATION AND MULTICOLLINEARITY
# ============================================================

"""
Multicollinearity occurs when predictor variables in a model
are strongly linearly related.

Example:

    House Size
        ↕
    Number of Rooms

If two features contain highly overlapping information,
some models can experience difficulties interpreting their
individual effects.

Correlation can be used as an initial diagnostic, although
it does not detect every form of multicollinearity.

More advanced analysis can use:

    VIF (Variance Inflation Factor)
"""


# ============================================================
# 16. OUTLIERS AND CORRELATION
# ============================================================

x_outlier = [1, 2, 3, 4, 5, 100]
y_outlier = [2, 4, 6, 8, 10, 12]

correlation_with_outlier = pearson_correlation(
    x_outlier,
    y_outlier
)

print("\n--- Outlier Effect ---")
print(
    "Correlation with outlier:",
    correlation_with_outlier
)

"""
Pearson correlation can be sensitive to outliers.

Therefore, correlation analysis should be combined with
visual inspection and appropriate data-quality checks.
"""


# ============================================================
# 17. CORRELATION IS NOT CAUSATION
# ============================================================

"""
Example:

    Ice Cream Sales ↑
           ↕
    Swimming Activity ↑

These variables may be positively correlated.

A possible third variable is:

    Temperature

    Temperature
       ↙    ↘
    Ice Cream  Swimming

The correlation between two variables does not by itself
establish a causal relationship.
"""


# ============================================================
# 18. PERFECT POSITIVE CORRELATION
# ============================================================

perfect_positive_x = [1, 2, 3, 4, 5]
perfect_positive_y = [2, 4, 6, 8, 10]

print("\n--- Perfect Positive Correlation ---")

print(
    pearson_correlation(
        perfect_positive_x,
        perfect_positive_y
    )
)


# ============================================================
# 19. PERFECT NEGATIVE CORRELATION
# ============================================================

perfect_negative_x = [1, 2, 3, 4, 5]
perfect_negative_y = [10, 8, 6, 4, 2]

print("\n--- Perfect Negative Correlation ---")

print(
    pearson_correlation(
        perfect_negative_x,
        perfect_negative_y
    )
)


# ============================================================
# 20. ZERO VARIANCE CASE
# ============================================================

constant_x = [5, 5, 5, 5, 5]
changing_y = [1, 2, 3, 4, 5]

print("\n--- Zero Variance ---")

try:
    print(
        pearson_correlation(
            constant_x,
            changing_y
        )
    )
except ValueError as error:
    print("Error:", error)


# ============================================================
# 21. CORRELATION IS SYMMETRIC
# ============================================================

x = [1, 2, 3, 4, 5]
y = [10, 20, 30, 40, 50]

r_xy = pearson_correlation(x, y)
r_yx = pearson_correlation(y, x)

print("\n--- Symmetry ---")

print("Correlation(X, Y):", r_xy)
print("Correlation(Y, X):", r_yx)

print(
    "Equal:",
    r_xy == r_yx
)


# ============================================================
# 22. CORRELATION IS SCALE-INVARIANT
# ============================================================

original_x = [1, 2, 3, 4, 5]
scaled_x = [100, 200, 300, 400, 500]

y = [2, 4, 6, 8, 10]

original_correlation = pearson_correlation(
    original_x,
    y
)

scaled_correlation = pearson_correlation(
    scaled_x,
    y
)

print("\n--- Scale Invariance ---")

print(
    "Original correlation:",
    original_correlation
)

print(
    "Scaled correlation:",
    scaled_correlation
)


# ============================================================
# 23. CORRELATION DOES NOT DETECT EVERY RELATIONSHIP
# ============================================================

"""
Pearson correlation measures linear association.

A relationship can be strongly nonlinear while having a
Pearson correlation near zero.

For example:

    Y = X²

can produce a curved relationship.

Therefore:

    correlation ≈ 0

does not necessarily mean:

    no relationship exists.

Always consider visualization and the nature of the data.
"""


# ============================================================
# 24. ML FEATURE SELECTION EXAMPLE
# ============================================================

try:
    import pandas as pd

    ml_data = pd.DataFrame({
        "Feature_A": [10, 20, 30, 40, 50],
        "Feature_B": [11, 21, 31, 41, 51],
        "Feature_C": [50, 10, 40, 20, 30],
        "Target": [100, 200, 300, 400, 500]
    })

    correlation_with_target = (
        ml_data
        .corr()["Target"]
        .sort_values(
            ascending=False
        )
    )

    print("\n--- Correlation With Target ---")
    print(correlation_with_target)

except ImportError:
    pass


# ============================================================
# 25. COMPLETE CORRELATION ANALYSIS FUNCTION
# ============================================================

def analyze_correlation(x, y):
    """
    Perform a complete basic Pearson correlation analysis.

    Returns:
        Dictionary containing:
            - means
            - covariance
            - correlation
            - interpretation
    """

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    cov = covariance(x, y)
    corr = pearson_correlation(x, y)

    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "covariance": cov,
        "correlation": corr,
        "interpretation": interpret_correlation(corr),
    }


print("\n--- Complete Correlation Analysis ---")

analysis = analyze_correlation(
    [1, 2, 3, 4, 5],
    [2, 4, 6, 8, 10]
)

for key, value in analysis.items():
    print(f"{key}: {value}")


# ============================================================
# 26. KEY FORMULAS
# ============================================================

"""
COVARIANCE
----------

Population:

             1
Cov(X,Y) = ─── Σ(xᵢ-x̄)(yᵢ-ȳ)
             n


PEARSON CORRELATION
-------------------

              Cov(X,Y)
r = ─────────────────────────
       σX × σY


Equivalent computational form:

             Σ(xᵢ-x̄)(yᵢ-ȳ)
r = ─────────────────────────────────────
    √[Σ(xᵢ-x̄)² × Σ(yᵢ-ȳ)²]


RANGE
-----

-1 ≤ r ≤ +1
"""


# ============================================================
# 27. MAIN FUNCTION
# ============================================================

def main():
    """
    Main entry point for the correlation examples.
    """

    print("\n" + "=" * 60)
    print("Correlation for Machine Learning")
    print("=" * 60)

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    r = pearson_correlation(x, y)

    print("\nDataset X:", x)
    print("Dataset Y:", y)
    print(f"Pearson Correlation: {r:.4f}")
    print(
        "Interpretation:",
        interpret_correlation(r)
    )

    print("\n" + "=" * 60)
    print("Correlation examples completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
```
