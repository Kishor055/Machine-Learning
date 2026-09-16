```python
"""
Arithmetic for Machine Learning
================================

File:
    02-Mathematics-for-Machine-Learning/01-Arithmetic/examples.py

Purpose:
    Demonstrate fundamental arithmetic concepts used in Machine Learning.

Topics Covered:
    1. Number systems
    2. Basic arithmetic operations
    3. Arithmetic properties
    4. Fractions and decimals
    5. Percentages
    6. Ratios and proportions
    7. Powers and roots
    8. Absolute value
    9. Rounding and approximation
    10. Arithmetic mean
    11. Weighted mean
    12. Percentage change
    13. Prediction error
    14. Absolute error
    15. Squared error
    16. Mean Squared Error (MSE)
    17. Euclidean distance
    18. Floating-point precision

Requirements:
    Python 3.x

Author:
    Kishor Patil
"""


# ============================================================
# 1. NUMBER SYSTEMS
# ============================================================

# Natural numbers
natural_numbers = {1, 2, 3, 4, 5}

# Whole numbers
whole_numbers = {0, 1, 2, 3, 4, 5}

# Integers
integers = {-3, -2, -1, 0, 1, 2, 3}

# Rational numbers can be represented as fractions.
rational_number = 3 / 4

# Real numbers include rational and irrational numbers.
real_number = 3.1415926535

print("Natural Numbers:", natural_numbers)
print("Whole Numbers:", whole_numbers)
print("Integers:", integers)
print("Rational Number:", rational_number)
print("Real Number:", real_number)


# ============================================================
# 2. BASIC ARITHMETIC OPERATIONS
# ============================================================

a = 20
b = 6

addition = a + b
subtraction = a - b
multiplication = a * b
division = a / b

# Floor division returns the integer quotient.
floor_division = a // b

# Modulo returns the remainder.
remainder = a % b

# Exponentiation
power = a ** 2

print("\n--- Basic Arithmetic ---")
print("Addition:", addition)
print("Subtraction:", subtraction)
print("Multiplication:", multiplication)
print("Division:", division)
print("Floor Division:", floor_division)
print("Remainder:", remainder)
print("Power:", power)


# ============================================================
# 3. ARITHMETIC PROPERTIES
# ============================================================

x = 10
y = 5
z = 2

# Commutative property
# a + b = b + a
print("\n--- Commutative Property ---")
print(x + y == y + x)

# Associative property
# (a + b) + c = a + (b + c)
print("\n--- Associative Property ---")
print((x + y) + z == x + (y + z))

# Distributive property
# a(b + c) = ab + ac
print("\n--- Distributive Property ---")
print(x * (y + z) == (x * y) + (x * z))

# Identity property
# a + 0 = a
# a * 1 = a
print("\n--- Identity Property ---")
print(x + 0 == x)
print(x * 1 == x)


# ============================================================
# 4. ORDER OF OPERATIONS
# ============================================================

# Python follows the standard mathematical order of operations.

result = 10 + 5 * 2
result_with_parentheses = (10 + 5) * 2

print("\n--- Order of Operations ---")
print("10 + 5 * 2 =", result)
print("(10 + 5) * 2 =", result_with_parentheses)


# ============================================================
# 5. FRACTIONS AND DECIMALS
# ============================================================

numerator = 3
denominator = 4

fraction = numerator / denominator

print("\n--- Fractions ---")
print("3/4 =", fraction)

# Fraction can be converted to a percentage.
percentage = fraction * 100

print("3/4 as percentage =", percentage, "%")


# ============================================================
# 6. PERCENTAGES
# ============================================================

value = 80
percentage_value = 25

result = value * percentage_value / 100

print("\n--- Percentage ---")
print(f"{percentage_value}% of {value} =", result)


# ============================================================
# 7. RATIOS AND PROPORTIONS
# ============================================================

boys = 30
girls = 20

ratio = boys / girls

print("\n--- Ratio ---")
print("Boys : Girls =", boys, ":", girls)
print("Ratio =", ratio)


# Example:
# If 5 notebooks cost ₹100,
# how much will 8 notebooks cost?

notebooks_1 = 5
cost_1 = 100
notebooks_2 = 8

cost_2 = (cost_1 / notebooks_1) * notebooks_2

print("\n--- Proportion ---")
print("Cost of 8 notebooks =", cost_2)


# ============================================================
# 8. POWERS AND EXPONENTS
# ============================================================

base = 2
exponent = 5

power = base ** exponent

print("\n--- Powers ---")
print(f"{base}^{exponent} =", power)

# Common powers used in ML:
# x² is frequently used in squared error.
# x³, x⁴, etc. may appear in mathematical transformations.


# ============================================================
# 9. ROOTS
# ============================================================

import math

number = 25

square_root = math.sqrt(number)

print("\n--- Roots ---")
print("Square root of 25 =", square_root)


# ============================================================
# 10. ABSOLUTE VALUE
# ============================================================

negative_value = -15
positive_value = 15

print("\n--- Absolute Value ---")
print("|-15| =", abs(negative_value))
print("|15| =", abs(positive_value))


# ============================================================
# 11. ROUNDING AND APPROXIMATION
# ============================================================

value = 3.14159265359

print("\n--- Rounding ---")
print("Original:", value)
print("Rounded to 2 decimals:", round(value, 2))
print("Rounded to 4 decimals:", round(value, 4))


# ============================================================
# 12. SCIENTIFIC NOTATION
# ============================================================

large_number = 1_000_000
small_number = 0.000001

print("\n--- Scientific Notation ---")
print("Large number:", f"{large_number:e}")
print("Small number:", f"{small_number:e}")


# ============================================================
# 13. ARITHMETIC MEAN
# ============================================================

values = [10, 20, 30, 40, 50]

mean = sum(values) / len(values)

print("\n--- Arithmetic Mean ---")
print("Values:", values)
print("Mean:", mean)


# ============================================================
# 14. WEIGHTED MEAN
# ============================================================

scores = [80, 90, 70]
weights = [0.2, 0.5, 0.3]

weighted_mean = sum(
    score * weight
    for score, weight in zip(scores, weights)
)

print("\n--- Weighted Mean ---")
print("Scores:", scores)
print("Weights:", weights)
print("Weighted Mean:", weighted_mean)


# ============================================================
# 15. PERCENTAGE CHANGE
# ============================================================

old_value = 100
new_value = 125

percentage_change = (
    (new_value - old_value) / old_value
) * 100

print("\n--- Percentage Change ---")
print("Old Value:", old_value)
print("New Value:", new_value)
print("Percentage Change:", percentage_change, "%")


# ============================================================
# 16. ML EXAMPLE: PREDICTION ERROR
# ============================================================

actual_value = 100
predicted_value = 90

error = actual_value - predicted_value

print("\n--- Prediction Error ---")
print("Actual:", actual_value)
print("Predicted:", predicted_value)
print("Error:", error)


# ============================================================
# 17. ML EXAMPLE: ABSOLUTE ERROR
# ============================================================

absolute_error = abs(actual_value - predicted_value)

print("\n--- Absolute Error ---")
print("Absolute Error:", absolute_error)


# ============================================================
# 18. ML EXAMPLE: SQUARED ERROR
# ============================================================

squared_error = (actual_value - predicted_value) ** 2

print("\n--- Squared Error ---")
print("Squared Error:", squared_error)


# ============================================================
# 19. ML EXAMPLE: MEAN SQUARED ERROR (MSE)
# ============================================================

actual_values = [100, 200, 300, 400]
predicted_values = [90, 210, 280, 420]

squared_errors = [
    (actual - predicted) ** 2
    for actual, predicted
    in zip(actual_values, predicted_values)
]

mse = sum(squared_errors) / len(squared_errors)

print("\n--- Mean Squared Error ---")
print("Actual Values:", actual_values)
print("Predicted Values:", predicted_values)
print("Squared Errors:", squared_errors)
print("MSE:", mse)


# ============================================================
# 20. ML EXAMPLE: ROOT MEAN SQUARED ERROR (RMSE)
# ============================================================

rmse = math.sqrt(mse)

print("\n--- Root Mean Squared Error ---")
print("RMSE:", rmse)


# ============================================================
# 21. ML EXAMPLE: EUCLIDEAN DISTANCE
# ============================================================

# Two points in a 2-dimensional space:
#
# Point A = (x1, y1)
# Point B = (x2, y2)
#
# Distance:
#
# d = sqrt((x2 - x1)^2 + (y2 - y1)^2)

point_a = (2, 3)
point_b = (5, 7)

distance = math.sqrt(
    (point_b[0] - point_a[0]) ** 2
    + (point_b[1] - point_a[1]) ** 2
)

print("\n--- Euclidean Distance ---")
print("Point A:", point_a)
print("Point B:", point_b)
print("Distance:", distance)


# ============================================================
# 22. ML EXAMPLE: NORMALIZED VALUE
# ============================================================

# Min-Max normalization:
#
# x_normalized = (x - min) / (max - min)

x = 75
minimum = 50
maximum = 100

normalized_x = (x - minimum) / (maximum - minimum)

print("\n--- Min-Max Normalization ---")
print("Original Value:", x)
print("Normalized Value:", normalized_x)


# ============================================================
# 23. ML EXAMPLE: PERCENTAGE ACCURACY
# ============================================================

correct_predictions = 92
total_predictions = 100

accuracy = (
    correct_predictions / total_predictions
) * 100

print("\n--- Model Accuracy ---")
print("Correct Predictions:", correct_predictions)
print("Total Predictions:", total_predictions)
print("Accuracy:", accuracy, "%")


# ============================================================
# 24. FLOATING-POINT PRECISION
# ============================================================

print("\n--- Floating-Point Precision ---")

result = 0.1 + 0.2

print("0.1 + 0.2 =", result)
print("0.1 + 0.2 == 0.3:", result == 0.3)

# Floating-point numbers are represented approximately
# in binary, so some decimal values cannot be represented
# exactly.


# ============================================================
# 25. HANDLING FLOATING-POINT COMPARISON
# ============================================================

value_1 = 0.1 + 0.2
value_2 = 0.3

print("\n--- Floating-Point Comparison ---")

print(
    "Approximately equal:",
    math.isclose(value_1, value_2)
)


# ============================================================
# 26. DIVISION BY ZERO
# ============================================================

print("\n--- Division by Zero ---")

try:
    result = 10 / 0
except ZeroDivisionError:
    print("Error: Division by zero is undefined.")


# ============================================================
# 27. PRACTICAL ML CALCULATION
# ============================================================

# Suppose a model predicts house prices.

actual_prices = [250000, 300000, 350000]
predicted_prices = [245000, 310000, 340000]

errors = [
    actual - predicted
    for actual, predicted
    in zip(actual_prices, predicted_prices)
]

absolute_errors = [
    abs(error)
    for error in errors
]

squared_errors = [
    error ** 2
    for error in errors
]

mae = sum(absolute_errors) / len(absolute_errors)
mse = sum(squared_errors) / len(squared_errors)
rmse = math.sqrt(mse)

print("\n--- House Price Prediction ---")
print("Actual Prices:", actual_prices)
print("Predicted Prices:", predicted_prices)
print("Errors:", errors)
print("Absolute Errors:", absolute_errors)
print("Squared Errors:", squared_errors)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)


# ============================================================
# 28. REUSABLE ARITHMETIC FUNCTIONS
# ============================================================

def calculate_mean(values):
    """Calculate the arithmetic mean of a collection of values."""
    if not values:
        raise ValueError("Values cannot be empty.")

    return sum(values) / len(values)


def calculate_percentage(value, percentage):
    """Calculate a percentage of a value."""
    return value * percentage / 100


def calculate_percentage_change(old_value, new_value):
    """Calculate percentage change between two values."""
    if old_value == 0:
        raise ValueError("Old value cannot be zero.")

    return ((new_value - old_value) / old_value) * 100


def calculate_mse(actual, predicted):
    """Calculate Mean Squared Error."""
    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted values must have the same length."
        )

    squared_errors = [
        (a - p) ** 2
        for a, p in zip(actual, predicted)
    ]

    return sum(squared_errors) / len(squared_errors)


def calculate_euclidean_distance(point_a, point_b):
    """Calculate Euclidean distance between two points."""
    if len(point_a) != len(point_b):
        raise ValueError(
            "Points must have the same number of dimensions."
        )

    squared_distance = sum(
        (a - b) ** 2
        for a, b in zip(point_a, point_b)
    )

    return math.sqrt(squared_distance)


# ============================================================
# 29. TESTING THE FUNCTIONS
# ============================================================

print("\n--- Reusable Functions ---")

data = [10, 20, 30, 40]

print("Mean:", calculate_mean(data))

print(
    "20% of 500:",
    calculate_percentage(500, 20)
)

print(
    "Percentage Change:",
    calculate_percentage_change(100, 120),
    "%"
)

print(
    "MSE:",
    calculate_mse(
        [10, 20, 30],
        [12, 18, 29]
    )
)

print(
    "Euclidean Distance:",
    calculate_euclidean_distance(
        [1, 2],
        [4, 6]
    )
)


# ============================================================
# 30. MAIN ENTRY POINT
# ============================================================

def main():
    """
    Entry point for the Arithmetic examples.

    This section can be expanded as the mathematics
    curriculum grows.
    """
    print("\n" + "=" * 60)
    print("Arithmetic for Machine Learning")
    print("=" * 60)
    print("Examples completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
```
