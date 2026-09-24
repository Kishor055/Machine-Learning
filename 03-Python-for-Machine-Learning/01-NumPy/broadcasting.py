"""
NumPy Broadcasting
==================

Broadcasting is one of the most important concepts in NumPy.

It allows NumPy to perform arithmetic operations on arrays
with different but compatible shapes.

Broadcasting is heavily used in:
- Machine Learning
- Data Preprocessing
- Feature Scaling
- Neural Networks
- Linear Algebra

Author: Kishor Kakde Patil
"""


import numpy as np


# ============================================================
# 1. Basic Broadcasting
# ============================================================

print("=" * 60)
print("1. BASIC BROADCASTING")
print("=" * 60)

arr = np.array([10, 20, 30, 40])

result = arr + 5

print("Original Array:", arr)
print("After Adding 5:", result)

"""
Output:

Original Array: [10 20 30 40]
After Adding 5: [15 25 35 45]

Explanation:
The scalar value 5 is automatically applied to every element.

Conceptually:

[10, 20, 30, 40]
       +
[ 5,  5,  5,  5]
       =
[15, 25, 35, 45]
"""


# ============================================================
# 2. Broadcasting with Subtraction
# ============================================================

print("\n" + "=" * 60)
print("2. BROADCASTING WITH SUBTRACTION")
print("=" * 60)

arr = np.array([10, 20, 30, 40])

result = arr - 10

print("Original Array:", arr)
print("After Subtracting 10:", result)


# ============================================================
# 3. Broadcasting with Multiplication
# ============================================================

print("\n" + "=" * 60)
print("3. BROADCASTING WITH MULTIPLICATION")
print("=" * 60)

arr = np.array([10, 20, 30, 40])

result = arr * 2

print("Original Array:", arr)
print("After Multiplication:", result)


# ============================================================
# 4. Broadcasting with Division
# ============================================================

print("\n" + "=" * 60)
print("4. BROADCASTING WITH DIVISION")
print("=" * 60)

arr = np.array([10, 20, 30, 40])

result = arr / 10

print("Original Array:", arr)
print("After Division:", result)


# ============================================================
# 5. Broadcasting with 2D Array
# ============================================================

print("\n" + "=" * 60)
print("5. BROADCASTING WITH 2D ARRAY")
print("=" * 60)

matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

result = matrix + 10

print("Original Matrix:")
print(matrix)

print("\nAfter Adding 10:")
print(result)

"""
Output:

[[10 20 30]
 [40 50 60]
 [70 80 90]]

+

10

=

[[20 30 40]
 [50 60 70]
 [80 90 100]]
"""


# ============================================================
# 6. Broadcasting with a Row Vector
# ============================================================

print("\n" + "=" * 60)
print("6. ROW VECTOR BROADCASTING")
print("=" * 60)

matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

row = np.array([1, 2, 3])

result = matrix + row

print("Matrix:")
print(matrix)

print("\nRow Vector:")
print(row)

print("\nResult:")
print(result)

"""
The row vector:

[1, 2, 3]

is broadcast across every row.

Result:

[[11 22 33]
 [41 52 63]
 [71 82 93]]
"""


# ============================================================
# 7. Broadcasting with a Column Vector
# ============================================================

print("\n" + "=" * 60)
print("7. COLUMN VECTOR BROADCASTING")
print("=" * 60)

matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

column = np.array([
    [1],
    [2],
    [3]
])

result = matrix + column

print("Matrix:")
print(matrix)

print("\nColumn Vector:")
print(column)

print("\nResult:")
print(result)

"""
The column vector:

[[1],
 [2],
 [3]]

is broadcast across every column.

Result:

[[11 21 31]
 [42 52 62]
 [73 83 93]]
"""


# ============================================================
# 8. Broadcasting with Different Shapes
# ============================================================

print("\n" + "=" * 60)
print("8. DIFFERENT SHAPES")
print("=" * 60)

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

B = np.array([10, 20, 30])

result = A + B

print("A:")
print(A)

print("\nB:")
print(B)

print("\nA + B:")
print(result)

"""
Shapes:

A → (2, 3)
B → (3,)

NumPy treats B as:

(1, 3)

Then:

(2, 3)
(1, 3)

Compatible → Broadcasting works.
"""


# ============================================================
# 9. Broadcasting Rules
# ============================================================

print("\n" + "=" * 60)
print("9. BROADCASTING RULES")
print("=" * 60)

"""
NumPy compares array dimensions from RIGHT to LEFT.

Two dimensions are compatible when:

1. They are equal
OR
2. One of them is 1
OR
3. One dimension does not exist

Example:

Shape A = (3, 4)
Shape B = (4,)

Comparison:

    (3, 4)
    (   4)

Right side:
4 == 4 → Compatible

Therefore broadcasting works.

"""


# ============================================================
# 10. Compatible Shapes
# ============================================================

print("\n" + "=" * 60)
print("10. COMPATIBLE SHAPES")
print("=" * 60)

A = np.zeros((3, 4))
B = np.ones((4,))

print("Shape A:", A.shape)
print("Shape B:", B.shape)

result = A + B

print("Result Shape:", result.shape)


# ============================================================
# 11. Broadcasting with Scalar
# ============================================================

print("\n" + "=" * 60)
print("11. SCALAR BROADCASTING")
print("=" * 60)

X = np.array([
    [1, 2],
    [3, 4]
])

scalar = 10

result = X + scalar

print("X:")
print(X)

print("\nResult:")
print(result)


# ============================================================
# 12. Machine Learning Example
# Feature Centering
# ============================================================

print("\n" + "=" * 60)
print("12. MACHINE LEARNING - FEATURE CENTERING")
print("=" * 60)

"""
Suppose we have a dataset:

Rows    → Samples
Columns → Features

Features:
1. Age
2. Salary
3. Experience
"""

X = np.array([
    [20, 30000, 1],
    [25, 40000, 3],
    [30, 50000, 5],
    [35, 60000, 7]
])

print("Original Dataset:")
print(X)

# Calculate mean of every feature
feature_mean = np.mean(X, axis=0)

print("\nFeature Mean:")
print(feature_mean)

# Broadcasting subtracts feature_mean
# from every row.

X_centered = X - feature_mean

print("\nCentered Dataset:")
print(X_centered)

"""
Mathematically:

X_centered = X - mean

The mean has shape:

(3,)

while X has shape:

(4, 3)

NumPy broadcasts the mean across all 4 samples.
"""


# ============================================================
# 13. Machine Learning Example
# Min-Max Normalization
# ============================================================

print("\n" + "=" * 60)
print("13. MACHINE LEARNING - NORMALIZATION")
print("=" * 60)

X = np.array([
    [10, 100],
    [20, 200],
    [30, 300],
    [40, 400]
], dtype=float)

X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0)

X_normalized = (X - X_min) / (X_max - X_min)

print("Original Data:")
print(X)

print("\nMinimum:")
print(X_min)

print("\nMaximum:")
print(X_max)

print("\nNormalized Data:")
print(X_normalized)

"""
Formula:

X_normalized =
(X - X_min) / (X_max - X_min)

Broadcasting allows X_min and X_max
to operate column-wise.
"""


# ============================================================
# 14. Machine Learning Example
# Standardization
# ============================================================

print("\n" + "=" * 60)
print("14. MACHINE LEARNING - STANDARDIZATION")
print("=" * 60)

X = np.array([
    [10, 100],
    [20, 200],
    [30, 300],
    [40, 400]
], dtype=float)

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_standardized = (X - mean) / std

print("Mean:")
print(mean)

print("\nStandard Deviation:")
print(std)

print("\nStandardized Data:")
print(X_standardized)

"""
Formula:

Z = (X - μ) / σ

Broadcasting applies μ and σ
to each corresponding feature column.
"""


# ============================================================
# 15. Neural Network Example
# ============================================================

print("\n" + "=" * 60)
print("15. NEURAL NETWORK - BIAS BROADCASTING")
print("=" * 60)

"""
In a neural network:

Z = XW + b

Suppose:

XW produces:

(samples, neurons)

Bias b has:

(neurons,)

NumPy broadcasts b across all samples.
"""

Z = np.array([
    [1.5, 2.5, 3.5],
    [4.5, 5.5, 6.5],
    [7.5, 8.5, 9.5]
])

bias = np.array([0.5, 1.0, 1.5])

output = Z + bias

print("Z:")
print(Z)

print("\nBias:")
print(bias)

print("\nZ + Bias:")
print(output)

"""
Result:

[[2.0, 3.5, 5.0],
 [5.0, 6.5, 8.0],
 [8.0, 9.5, 11.0]]

The bias vector is automatically applied
to every sample.
"""


# ============================================================
# 16. Broadcasting Error Example
# ============================================================

print("\n" + "=" * 60)
print("16. INCOMPATIBLE SHAPES")
print("=" * 60)

"""
The following shapes are incompatible:

A → (2, 3)
B → (2,)

Comparison:

(2, 3)
(2)

Rightmost dimensions:

3 != 2

Therefore broadcasting fails.
"""

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

B = np.array([10, 20])

try:
    result = A + B
    print(result)
except ValueError as error:
    print("Broadcasting Error:")
    print(error)


# ============================================================
# 17. Fixing the Shape
# ============================================================

print("\n" + "=" * 60)
print("17. FIXING SHAPE")
print("=" * 60)

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

B = np.array([
    [10],
    [20]
])

result = A + B

print("A:")
print(A)

print("\nB:")
print(B)

print("\nResult:")
print(result)

"""
B has shape:

(2, 1)

A has shape:

(2, 3)

Comparison:

(2, 3)
(2, 1)

Both dimensions are compatible.

Result:

[[11, 12, 13],
 [24, 25, 26]]
"""


# ============================================================
# 18. Broadcasting vs Loop
# ============================================================

print("\n" + "=" * 60)
print("18. BROADCASTING VS LOOP")
print("=" * 60)

arr = np.array([10, 20, 30, 40, 50])

# Loop approach
loop_result = []

for value in arr:
    loop_result.append(value + 10)

# Broadcasting approach
numpy_result = arr + 10

print("Loop Result:")
print(loop_result)

print("\nNumPy Result:")
print(numpy_result)

"""
Broadcasting avoids manually writing loops
for many numerical operations.

This is one reason NumPy is useful for
large numerical datasets.
"""


# ============================================================
# 19. Important Broadcasting Summary
# ============================================================

print("\n" + "=" * 60)
print("19. SUMMARY")
print("=" * 60)

print("""
Broadcasting allows NumPy to operate on
arrays with compatible shapes.

Important rules:

1. Compare dimensions from right to left.
2. Dimensions are compatible if:
   - They are equal, OR
   - One dimension is 1, OR
   - One dimension is missing.

Common ML applications:

- Feature centering
- Normalization
- Standardization
- Neural network bias
- Vectorized calculations
- Data preprocessing
- Matrix operations
""")