"""
Covariance for Machine Learning
===============================

File:
    02-Mathematics-for-Machine-Learning/02-Statistics/covariance.py

Purpose:
    Learn how covariance measures how two variables change
    together.

Topics Covered:
    1. What is covariance?
    2. Positive covariance
    3. Negative covariance
    4. Near-zero covariance
    5. Mean and deviations
    6. Population covariance
    7. Sample covariance
    8. Manual covariance calculation
    9. NumPy covariance
    10. Pandas covariance
    11. Covariance matrix
    12. Covariance and correlation
    13. Covariance and Machine Learning
    14. Feature relationships
    15. Multicollinearity
    16. Scale dependence
    17. Covariance with standardized data
    18. Practical ML examples

Requirements:
    Python 3.x

Optional:
    numpy
    pandas

Author:
    Kishor Patil
"""


# ============================================================
# 1. WHAT IS COVARIANCE?
# ============================================================

"""
Covariance measures how two variables change together.

Suppose we have two variables:

    X = Study Hours
    Y = Exam Score

If students who study more hours generally score higher,
X and Y have positive covariance.

If one variable tends to increase while the other decreases,
they have negative covariance.

If there is no consistent linear movement together,
covariance may be close to zero.

Interpretation:

    Cov(X, Y) > 0
        Positive relationship

    Cov(X, Y) < 0
        Negative relationship

    Cov(X, Y) ≈ 0
        Little linear co-movement

Important:
    Covariance measures direction of joint variation, but
    its magnitude depends on the units and scale of the data.

For comparing relationship strength across variables,
correlation is usually easier to interpret.
"""


# ============================================================
# 2. POSITIVE COVARIANCE
# ============================================================

study_hours = [1, 2, 3, 4, 5]
exam_scores = [50, 55, 65, 75, 85]

print("\n--- Positive Covariance Example ---")

print("Study Hours:", study_hours)
print("Exam Scores:", exam_scores)

"""
As study hours increase, exam scores generally increase.

Therefore:

    Cov(Study Hours, Exam Scores) > 0
"""


# ============================================================
# 3. NEGATIVE COVARIANCE
# ============================================================

temperature = [10, 15, 20, 25, 30]
heating_usage = [95, 85, 70, 50, 30]

print("\n--- Negative Covariance Example ---")

print("Temperature:", temperature)
print("Heating Usage:", heating_usage)

"""
As temperature increases, heating usage decreases.

Therefore:

    Cov(Temperature, Heating Usage) < 0
"""


# ============================================================
# 4. NEAR-ZERO COVARIANCE
# ============================================================

x_values = [1, 2, 3, 4, 5]
y_values = [7, 2, 8, 1, 6]

print("\n--- Near-Zero Covariance Example ---")

print("X:", x_values)
print("Y:", y_values)

"""
There is no obvious linear co-movement.

The covariance may therefore be relatively close to zero.

Important:
    Covariance near zero does not prove that variables are
    completely independent.
"""


# ============================================================
# 5. MEAN AND DEVIATIONS
# ============================================================

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

mean_x = sum(x) / len(x)
mean_y = sum(y) / len(y)

deviations_x = [
    value - mean_x
    for value in x
]

deviations_y = [
    value - mean_y
    for value in y
]

print("\n--- Means and Deviations ---")

print("X:", x)
print("Mean X:", mean_x)
print("Deviation X:", deviations_x)

print("\nY:", y)
print("Mean Y:", mean_y)
print("Deviation Y:", deviations_y)

"""
Covariance is based on the product of deviations from
their respective means.

For each pair:

    (xi - mean_x) × (yi - mean_y)

If both deviations have the same sign:

    positive × positive = positive
    negative × negative = positive

If deviations have opposite signs:

    positive × negative = negative
    negative × positive = negative

This is the mathematical reason covariance indicates
whether variables tend to move together or in opposite
directions.
"""


# ============================================================
# 6. PRODUCT OF DEVIATIONS
# ============================================================

products_of_deviations = [
    (xi - mean_x) * (yi - mean_y)
    for xi, yi in zip(x, y)
]

print("\n--- Products of Deviations ---")

for xi, yi, product in zip(
    x,
    y,
    products_of_deviations
):
    print(
        f"X={xi}, Y={yi}, "
        f"Deviation Product={product}"
    )

"""
The sum of these products forms the numerator of covariance.

Population covariance:

                    Σ[(xi - x̄)(yi - ȳ)]
    Cov(X,Y) = -----------------------------
                         n

Sample covariance:

                    Σ[(xi - x̄)(yi - ȳ)]
    Sxy        = -----------------------------
                       n - 1
"""


# ============================================================
# 7. POPULATION COVARIANCE
# ============================================================

def population_covariance(x, y):
    """
    Calculate population covariance.

    Formula:

                    Σ[(xi - x̄)(yi - ȳ)]
        Cov(X,Y) = -------------------------
                             n

    Parameters:
        x : list
            First variable.

        y : list
            Second variable.

    Returns:
        float
            Population covariance.

    Raises:
        ValueError:
            If datasets have different lengths or are empty.
    """

    if len(x) != len(y):
        raise ValueError(
            "Both datasets must contain the same number of values."
        )

    if len(x) == 0:
        raise ValueError(
            "Datasets cannot be empty."
        )

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    covariance = sum(
        (xi - mean_x) * (yi - mean_y)
        for xi, yi in zip(x, y)
    ) / len(x)

    return covariance


print("\n--- Population Covariance ---")

population_cov = population_covariance(
    x,
    y
)

print("X:", x)
print("Y:", y)
print("Population Covariance:", population_cov)


# ============================================================
# 8. SAMPLE COVARIANCE
# ============================================================

def sample_covariance(x, y):
    """
    Calculate sample covariance.

    Formula:

                    Σ[(xi - x̄)(yi - ȳ)]
        Sxy = -----------------------------
                         n - 1

    Sample covariance uses n - 1 because the data represents
    a sample from a larger population.

    Parameters:
        x : list
            First variable.

        y : list
            Second variable.

    Returns:
        float
            Sample covariance.
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

    covariance = sum(
        (xi - mean_x) * (yi - mean_y)
        for xi, yi in zip(x, y)
    ) / (len(x) - 1)

    return covariance


print("\n--- Sample Covariance ---")

sample_cov = sample_covariance(
    x,
    y
)

print("Sample Covariance:", sample_cov)


# ============================================================
# 9. POPULATION VS SAMPLE COVARIANCE
# ============================================================

print("\n--- Population vs Sample Covariance ---")

print(
    "Population Covariance:",
    population_covariance(x, y)
)

print(
    "Sample Covariance:",
    sample_covariance(x, y)
)

"""
Difference:

Population covariance:

    Divide by n

Sample covariance:

    Divide by n - 1

Use population covariance when your dataset represents the
entire population being studied.

Use sample covariance when the observations represent a sample
used to estimate properties of a larger population.
"""


# ============================================================
# 10. MANUAL COVARIANCE CALCULATION
# ============================================================

def covariance_step_by_step(x, y):
    """
    Display the covariance calculation step by step.

    Useful for learning and debugging.
    """

    if len(x) != len(y):
        raise ValueError(
            "Datasets must have equal lengths."
        )

    if len(x) == 0:
        raise ValueError(
            "Datasets cannot be empty."
        )

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    print("\nMeans:")
    print("Mean X:", mean_x)
    print("Mean Y:", mean_y)

    print("\nDeviation Products:")

    total = 0

    for xi, yi in zip(x, y):

        deviation_x = xi - mean_x
        deviation_y = yi - mean_y

        product = deviation_x * deviation_y

        total += product

        print(
            f"X={xi:>6} | "
            f"Y={yi:>6} | "
            f"X deviation={deviation_x:>7.2f} | "
            f"Y deviation={deviation_y:>7.2f} | "
            f"Product={product:>8.2f}"
        )

    covariance = total / len(x)

    print("\nSum of Products:", total)
    print("Population Covariance:", covariance)

    return covariance


print("\n--- Step-by-Step Covariance ---")

covariance_step_by_step(
    [1, 2, 3, 4, 5],
    [2, 4, 6, 8, 10]
)


# ============================================================
# 11. COVARIANCE FUNCTION WITH STATISTICS
# ============================================================

def covariance_direction(covariance_value):
    """
    Interpret the direction of covariance.

    This function intentionally focuses on direction rather than
    assigning universal strength thresholds because covariance
    magnitude depends on measurement units.
    """

    if covariance_value > 0:
        return "Positive co-movement"

    if covariance_value < 0:
        return "Negative co-movement"

    return "No linear co-movement"


print("\n--- Covariance Direction ---")

print(
    "Covariance:",
    population_cov
)

print(
    "Interpretation:",
    covariance_direction(population_cov)
)


# ============================================================
# 12. NUMPY COVARIANCE
# ============================================================

try:
    import numpy as np

    x_np = np.array(
        [1, 2, 3, 4, 5]
    )

    y_np = np.array(
        [2, 4, 6, 8, 10]
    )

    covariance_matrix = np.cov(
        x_np,
        y_np
    )

    print("\n--- NumPy Covariance ---")

    print("Covariance Matrix:")
    print(covariance_matrix)

    print(
        "Sample Covariance:",
        covariance_matrix[0, 1]
    )

except ImportError:
    print(
        "\nNumPy is not installed."
        "\nInstall it using: pip install numpy"
    )


# ============================================================
# 13. NUMPY POPULATION COVARIANCE
# ============================================================

try:
    import numpy as np

    population_cov_numpy = np.cov(
        x_np,
        y_np,
        ddof=0
    )[0, 1]

    sample_cov_numpy = np.cov(
        x_np,
        y_np,
        ddof=1
    )[0, 1]

    print("\n--- NumPy Population vs Sample ---")

    print(
        "Population Covariance:",
        population_cov_numpy
    )

    print(
        "Sample Covariance:",
        sample_cov_numpy
    )

except ImportError:
    pass


# ============================================================
# 14. PANDAS COVARIANCE
# ============================================================

try:
    import pandas as pd

    data = pd.DataFrame({
        "Study_Hours": [1, 2, 3, 4, 5],
        "Exam_Score": [50, 55, 65, 75, 85]
    })

    print("\n--- Pandas Covariance ---")

    print(data)

    print(
        "\nCovariance:",
        data["Study_Hours"].cov(
            data["Exam_Score"]
        )
    )

except ImportError:
    print(
        "\nPandas is not installed."
        "\nInstall it using: pip install pandas"
    )


# ============================================================
# 15. COVARIANCE MATRIX
# ============================================================

try:
    import pandas as pd

    dataset = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40],
        "Experience": [1, 3, 5, 8, 10],
        "Salary": [25000, 35000, 45000, 60000, 75000],
        "Performance": [60, 65, 75, 85, 90]
    })

    covariance_matrix = dataset.cov()

    print("\n--- Covariance Matrix ---")

    print(covariance_matrix)

except ImportError:
    pass


# ============================================================
# 16. UNDERSTANDING THE COVARIANCE MATRIX
# ============================================================

"""
For variables:

    X1
    X2
    X3

The covariance matrix looks like:

            X1       X2       X3

    X1     Var(X1)  Cov12   Cov13
    X2     Cov21    Var(X2)  Cov23
    X3     Cov31    Cov32    Var(X3)


Important:

    Diagonal elements:
        Variances

    Off-diagonal elements:
        Covariances

The covariance matrix is symmetric:

    Cov(X,Y) = Cov(Y,X)
"""


# ============================================================
# 17. COVARIANCE MATRIX SYMMETRY
# ============================================================

try:
    print("\n--- Covariance Matrix Symmetry ---")

    covariance_xy = population_covariance(
        [1, 2, 3, 4, 5],
        [2, 4, 6, 8, 10]
    )

    covariance_yx = population_covariance(
        [2, 4, 6, 8, 10],
        [1, 2, 3, 4, 5]
    )

    print(
        "Covariance(X, Y):",
        covariance_xy
    )

    print(
        "Covariance(Y, X):",
        covariance_yx
    )

    print(
        "Equal:",
        covariance_xy == covariance_yx
    )

except Exception as error:
    print("Error:", error)


# ============================================================
# 18. COVARIANCE AND MACHINE LEARNING
# ============================================================

"""
Covariance is useful in Machine Learning for understanding
relationships between numerical variables.

Applications include:

    1. Exploratory Data Analysis
    2. Feature relationship analysis
    3. Multivariate statistics
    4. Principal Component Analysis (PCA)
    5. Dimensionality reduction
    6. Understanding feature dependencies
    7. Covariance matrices in statistical models

Example:

    Features:

        Age
        Experience
        Salary

If:

    Cov(Age, Experience) > 0

then older observations tend to have higher experience
values in the dataset.

However, covariance magnitude depends on units, so it is
not ideal for directly comparing relationships measured on
different scales.
"""


# ============================================================
# 19. FEATURE COVARIANCE IN ML
# ============================================================

try:
    import pandas as pd

    ml_data = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40],
        "Experience": [1, 3, 5, 8, 10],
        "Salary": [25000, 35000, 45000, 60000, 75000]
    })

    covariance_matrix = ml_data.cov()

    print("\n--- ML Feature Covariance ---")

    print(covariance_matrix)

except ImportError:
    pass


# ============================================================
# 20. COVARIANCE AND MULTICOLLINEARITY
# ============================================================

"""
Multicollinearity occurs when predictor variables contain
strong linear relationships with each other.

Example:

    House Size
          ↕
    Number of Rooms

These variables may carry overlapping information.

Covariance can reveal whether variables move together.

However:

    Covariance alone is not usually enough to diagnose
    multicollinearity.

Correlation and VIF (Variance Inflation Factor) are often
used for additional analysis.
"""


# ============================================================
# 21. SCALE DEPENDENCE
# ============================================================

original_x = [1, 2, 3, 4, 5]
scaled_x = [100, 200, 300, 400, 500]

y = [2, 4, 6, 8, 10]

original_covariance = population_covariance(
    original_x,
    y
)

scaled_covariance = population_covariance(
    scaled_x,
    y
)

print("\n--- Scale Dependence ---")

print(
    "Original X:",
    original_x
)

print(
    "Scaled X:",
    scaled_x
)

print(
    "Original Covariance:",
    original_covariance
)

print(
    "Scaled Covariance:",
    scaled_covariance
)

"""
Notice that scaling X changes covariance.

If:

    X' = aX

then:

    Cov(X', Y)
        = a × Cov(X, Y)

Therefore covariance depends on the units and scale.

This is one reason correlation is often preferred when
comparing the strength of relationships.
"""


# ============================================================
# 22. COVARIANCE VS CORRELATION
# ============================================================

"""
COVARIANCE
----------

Measures:
    Direction of joint variation.

Range:
    Unbounded.

Depends on:
    Units and scale.

Formula:

              Cov(X,Y)
    r = ---------------------
           σX × σY


CORRELATION
-----------

Measures:
    Standardized linear relationship.

Range:

    -1 ≤ r ≤ +1

Advantages:

    - Unitless
    - Easier to interpret
    - Easier to compare across variables

Relationship:

    Correlation is standardized covariance.
"""


# ============================================================
# 23. STANDARDIZED DATA
# ============================================================

def standardize(values):
    """
    Standardize values using population standard deviation.

    Formula:

                  x - mean
        z = -----------------------
                    σ

    Standardization converts a variable to approximately
    zero mean and unit variance.
    """

    if len(values) == 0:
        raise ValueError(
            "Values cannot be empty."
        )

    mean = sum(values) / len(values)

    variance = sum(
        (value - mean) ** 2
        for value in values
    ) / len(values)

    standard_deviation = variance ** 0.5

    if standard_deviation == 0:
        raise ValueError(
            "Cannot standardize a constant variable."
        )

    return [
        (value - mean) / standard_deviation
        for value in values
    ]


standardized_x = standardize(
    [1, 2, 3, 4, 5]
)

standardized_y = standardize(
    [2, 4, 6, 8, 10]
)

print("\n--- Standardized Data ---")

print(
    "Standardized X:",
    standardized_x
)

print(
    "Standardized Y:",
    standardized_y
)

print(
    "Covariance of standardized variables:",
    population_covariance(
        standardized_x,
        standardized_y
    )
)

"""
When both variables are standardized using population
standard deviation:

    Cov(Zx, Zy)

is equal to their Pearson correlation.

This demonstrates the relationship:

    Correlation =
        Standardized Covariance
"""


# ============================================================
# 24. PERFECT POSITIVE COVARIANCE EXAMPLE
# ============================================================

perfect_positive_x = [1, 2, 3, 4, 5]
perfect_positive_y = [2, 4, 6, 8, 10]

print("\n--- Positive Linear Relationship ---")

print(
    "Covariance:",
    population_covariance(
        perfect_positive_x,
        perfect_positive_y
    )
)


# ============================================================
# 25. PERFECT NEGATIVE RELATIONSHIP
# ============================================================

perfect_negative_x = [1, 2, 3, 4, 5]
perfect_negative_y = [10, 8, 6, 4, 2]

print("\n--- Negative Linear Relationship ---")

print(
    "Covariance:",
    population_covariance(
        perfect_negative_x,
        perfect_negative_y
    )
)


# ============================================================
# 26. ZERO COVARIANCE WITH CONSTANT VARIABLE
# ============================================================

constant_x = [5, 5, 5, 5, 5]
changing_y = [1, 2, 3, 4, 5]

print("\n--- Constant Variable ---")

print(
    "Covariance:",
    population_covariance(
        constant_x,
        changing_y
    )
)

"""
A constant variable has zero variance.

Therefore:

    X - mean(X) = 0

for every observation.

So covariance with another variable becomes zero.

However, correlation is undefined because its denominator
contains the standard deviation of X, which is zero.
"""


# ============================================================
# 27. COVARIANCE OF A VARIABLE WITH ITSELF
# ============================================================

values = [1, 2, 3, 4, 5]

variance_from_covariance = population_covariance(
    values,
    values
)

print("\n--- Covariance With Itself ---")

print(
    "Cov(X, X):",
    variance_from_covariance
)

"""
Important identity:

    Cov(X, X) = Var(X)

Therefore, the diagonal elements of a covariance matrix
represent variances.
"""


# ============================================================
# 28. COVARIANCE AND OUTLIERS
# ============================================================

x_without_outlier = [1, 2, 3, 4, 5]
y_without_outlier = [2, 4, 6, 8, 10]

x_with_outlier = [1, 2, 3, 4, 5, 100]
y_with_outlier = [2, 4, 6, 8, 10, 12]

cov_without_outlier = population_covariance(
    x_without_outlier,
    y_without_outlier
)

cov_with_outlier = population_covariance(
    x_with_outlier,
    y_with_outlier
)

print("\n--- Outlier Effect ---")

print(
    "Covariance without outlier:",
    cov_without_outlier
)

print(
    "Covariance with outlier:",
    cov_with_outlier
)

"""
Covariance can be strongly affected by outliers because
extreme observations can produce large deviation products.

Therefore, when performing covariance analysis:

    1. Inspect the data.
    2. Check for outliers.
    3. Understand the domain.
    4. Consider robust statistical methods when appropriate.
"""


# ============================================================
# 29. DIFFERENT UNITS
# ============================================================

height_cm = [150, 160, 170, 180, 190]
weight_kg = [50, 58, 65, 75, 85]

height_m = [
    height / 100
    for height in height_cm
]

covariance_cm = population_covariance(
    height_cm,
    weight_kg
)

covariance_m = population_covariance(
    height_m,
    weight_kg
)

print("\n--- Effect of Units ---")

print(
    "Covariance using centimeters:",
    covariance_cm
)

print(
    "Covariance using meters:",
    covariance_m
)

"""
Changing height from centimeters to meters changes the
numerical covariance.

But the underlying relationship has not changed.

This demonstrates again that covariance is unit-dependent.
"""


# ============================================================
# 30. PRACTICAL HOUSE PRICE EXAMPLE
# ============================================================

house_size = [
    800,
    1000,
    1200,
    1500,
    1800
]

house_price = [
    200000,
    250000,
    290000,
    360000,
    430000
]

house_covariance = population_covariance(
    house_size,
    house_price
)

print("\n--- House Price Example ---")

print(
    "House Size:",
    house_size
)

print(
    "House Price:",
    house_price
)

print(
    "Covariance:",
    house_covariance
)

print(
    "Direction:",
    covariance_direction(
        house_covariance
    )
)

"""
In a Machine Learning dataset, covariance can help us
understand how numerical features and targets vary together.

For example:

    House Size ↔ House Price

may show positive covariance.

However, covariance does not establish causation.
Other variables may also influence house prices.
"""


# ============================================================
# 31. COMPLETE COVARIANCE ANALYSIS
# ============================================================

def analyze_covariance(x, y):
    """
    Perform a complete basic covariance analysis.

    Returns:
        Dictionary containing:

            mean_x
            mean_y
            population_covariance
            sample_covariance
            direction
    """

    if len(x) != len(y):
        raise ValueError(
            "Datasets must have equal lengths."
        )

    if len(x) < 2:
        raise ValueError(
            "At least two observations are required."
        )

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    population_cov = population_covariance(
        x,
        y
    )

    sample_cov = sample_covariance(
        x,
        y
    )

    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "population_covariance": population_cov,
        "sample_covariance": sample_cov,
        "direction": covariance_direction(
            population_cov
        )
    }


print("\n--- Complete Covariance Analysis ---")

analysis = analyze_covariance(
    [1, 2, 3, 4, 5],
    [2, 4, 6, 8, 10]
)

for key, value in analysis.items():
    print(
        f"{key}: {value}"
    )


# ============================================================
# 32. COVARIANCE FORMULAS
# ============================================================

"""
POPULATION COVARIANCE
---------------------

                    Σ[(xi - x̄)(yi - ȳ)]
    Cov(X,Y) = -----------------------------
                         n


SAMPLE COVARIANCE
-----------------

                    Σ[(xi - x̄)(yi - ȳ)]
    Sxy        = -----------------------------
                       n - 1


VARIANCE
--------

    Var(X) = Cov(X, X)


CORRELATION
-----------

              Cov(X,Y)
    r = ---------------------
           σX × σY


STANDARDIZED VARIABLES
----------------------

    If X and Y are standardized:

        Cov(ZX, ZY) = Correlation(X, Y)
"""


# ============================================================
# 33. IMPORTANT PROPERTIES
# ============================================================

"""
Important covariance properties:

1. Symmetry

       Cov(X,Y) = Cov(Y,X)


2. Self-covariance

       Cov(X,X) = Var(X)


3. Constant variable

       Cov(c,Y) = 0


4. Scaling

       Cov(aX,Y) = a Cov(X,Y)


5. Scaling both variables

       Cov(aX,bY)
           = ab Cov(X,Y)


6. Addition

       Cov(X + Z, Y)
           = Cov(X,Y) + Cov(Z,Y)


7. Independence

       If X and Y are independent and their covariance exists:

           Cov(X,Y) = 0

   However, zero covariance does not generally imply
   independence.
"""


# ============================================================
# 34. COMMON MISTAKES
# ============================================================

"""
Common mistakes:

1. Assuming covariance has a fixed range.

       WRONG:
           -1 ≤ covariance ≤ 1

       CORRECT:
           Covariance is unbounded.


2. Confusing covariance with correlation.

       Covariance depends on units.
       Correlation is standardized.


3. Assuming zero covariance means independence.

       Zero covariance only indicates no linear
       co-movement.


4. Ignoring units.

       Changing meters to centimeters changes covariance.


5. Ignoring outliers.

       Extreme observations can strongly affect covariance.


6. Using sample and population formulas interchangeably.

       Population → divide by n
       Sample     → divide by n - 1


7. Assuming covariance proves causation.

       Statistical association does not automatically
       establish cause and effect.


8. Using covariance magnitude to compare unrelated units.

       A large covariance is not automatically a stronger
       relationship than a smaller covariance measured on
       different scales.
"""


# ============================================================
# 35. MAIN FUNCTION
# ============================================================

def main():
    """
    Main entry point for covariance examples.
    """

    print("\n" + "=" * 65)
    print("Covariance for Machine Learning")
    print("=" * 65)

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    population_cov = population_covariance(
        x,
        y
    )

    sample_cov = sample_covariance(
        x,
        y
    )

    print("\nDataset X:")
    print(x)

    print("\nDataset Y:")
    print(y)

    print(
        "\nPopulation Covariance:",
        population_cov
    )

    print(
        "Sample Covariance:",
        sample_cov
    )

    print(
        "Direction:",
        covariance_direction(
            population_cov
        )
    )

    print("\n" + "=" * 65)
    print("Covariance examples completed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
```
