"""
NumPy Arrays
============

NumPy is a Python library used for numerical computing.

An ndarray (N-dimensional array) is NumPy's core data structure.

Topics:
1. Creating arrays
2. 1D arrays
3. 2D arrays
4. 3D arrays
5. Array dimensions
6. Shape
7. Size
8. Data type
9. Indexing
10. Slicing
"""

import numpy as np


# ============================================================
# 1. Creating a NumPy Array
# ============================================================

arr = np.array([10, 20, 30, 40, 50])

print("Array:")
print(arr)


# ============================================================
# 2. 1D Array
# ============================================================

one_d = np.array([1, 2, 3, 4, 5])

print("\n1D Array:")
print(one_d)


# ============================================================
# 3. 2D Array
# ============================================================

two_d = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("\n2D Array:")
print(two_d)


# ============================================================
# 4. 3D Array
# ============================================================

three_d = np.array([
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
])

print("\n3D Array:")
print(three_d)


# ============================================================
# 5. Number of Dimensions
# ============================================================

print("\nNumber of Dimensions:")
print(two_d.ndim)


# ============================================================
# 6. Shape
# ============================================================

print("\nShape:")
print(two_d.shape)


# ============================================================
# 7. Size
# ============================================================

print("\nNumber of Elements:")
print(two_d.size)


# ============================================================
# 8. Data Type
# ============================================================

print("\nData Type:")
print(two_d.dtype)


# ============================================================
# 9. Indexing
# ============================================================

arr = np.array([10, 20, 30, 40, 50])

print("\nIndexing:")
print("First element:", arr[0])
print("Third element:", arr[2])
print("Last element:", arr[-1])


# ============================================================
# 10. 2D Array Indexing
# ============================================================

matrix = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

print("\n2D Indexing:")
print(matrix[0, 1])
print(matrix[1, 2])


# ============================================================
# 11. Slicing
# ============================================================

arr = np.array([10, 20, 30, 40, 50])

print("\nSlicing:")
print(arr[1:4])
print(arr[:3])
print(arr[2:])
print(arr[::-1])


# ============================================================
# 12. 2D Slicing
# ============================================================

matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print("\n2D Slicing:")
print(matrix[0:2, 1:3])


# ============================================================
# 13. Array Data Type
# ============================================================

integer_array = np.array([1, 2, 3])

float_array = np.array([1.5, 2.5, 3.5])

print("\nInteger Array:")
print(integer_array)

print("\nFloat Array:")
print(float_array)


# ============================================================
# 14. Convert Data Type
# ============================================================

arr = np.array([1.2, 2.5, 3.8])

integer_arr = arr.astype(int)

print("\nOriginal:")
print(arr)

print("\nConverted to Integer:")
print(integer_arr)


# ============================================================
# 15. Machine Learning Example
# ============================================================

# Rows = Students
# Columns = Features
#
# Features:
# [Age, Marks, Attendance]

student_data = np.array([
    [20, 85, 90],
    [21, 78, 85],
    [22, 92, 95],
    [20, 88, 92]
])

print("\nMachine Learning Dataset:")
print(student_data)

print("\nDataset Shape:")
print(student_data.shape)

print("\nNumber of Students:")
print(student_data.shape[0])

print("\nNumber of Features:")
print(student_data.shape[1])