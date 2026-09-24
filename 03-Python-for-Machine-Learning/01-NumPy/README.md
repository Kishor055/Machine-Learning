NumPy for Machine Learning 🧮🤖

A complete NumPy learning guide for Machine Learning, covering the most important concepts, operations, mathematical functions, linear algebra, preprocessing, and practical examples.

«Goal: Build a strong NumPy foundation required for Machine Learning, Data Science, Deep Learning, and AI.»

---

📌 Table of Contents

- "What is NumPy?" (#-what-is-numpy)
- "Why NumPy for Machine Learning?" (#-why-numpy-for-machine-learning)
- "Installation" (#-installation)
- "Import NumPy" (#-import-numpy)
- "NumPy Array" (#-numpy-array)
- "Array Attributes" (#-array-attributes)
- "Array Creation" (#-array-creation-methods)
- "Data Types" (#-data-types)
- "Indexing and Slicing" (#-indexing-and-slicing)
- "Reshaping Arrays" (#-reshaping-arrays)
- "Flatten and Ravel" (#-flatten-and-ravel)
- "Joining Arrays" (#-joining-arrays)
- "Splitting Arrays" (#-splitting-arrays)
- "Copy vs View" (#-copy-vs-view)
- "Mathematical Operations" (#-mathematical-operations)
- "Statistical Functions" (#-statistical-functions)
- "Aggregation Functions" (#-aggregation-functions)
- "Broadcasting" (#-broadcasting)
- "Boolean Indexing" (#-boolean-indexing)
- "Sorting and Searching" (#-sorting-and-searching)
- "Random Numbers" (#-random-numbers)
- "Probability and Sampling" (#-probability-and-sampling)
- "Linear Algebra" (#-linear-algebra)
- "Matrix Operations" (#-matrix-operations)
- "Transpose" (#-transpose)
- "Dot Product" (#-dot-product)
- "Inverse and Determinant" (#-inverse-and-determinant)
- "Eigenvalues and Eigenvectors" (#-eigenvalues-and-eigenvectors)
- "Normalization" (#-normalization)
- "Standardization" (#-standardization)
- "Handling Missing Values" (#-handling-missing-values)
- "One-Hot Encoding with NumPy" (#-one-hot-encoding-with-numpy)
- "NumPy in Machine Learning" (#-numpy-in-machine-learning)
- "ML Example: Linear Regression" (#-ml-example-linear-regression)
- "ML Example: Gradient Descent" (#-ml-example-gradient-descent)
- "Performance" (#-why-numpy-is-fast)
- "Important Functions Cheat Sheet" (#-important-functions-cheat-sheet)
- "Practice Questions" (#-practice-questions)

---

🔹 What is NumPy?

NumPy stands for Numerical Python.

It is a Python library used for:

- Numerical computing
- Array manipulation
- Mathematical operations
- Matrix operations
- Statistics
- Linear algebra
- Random number generation
- Scientific computing
- Machine Learning data processing

NumPy provides a powerful multidimensional array called:

numpy.ndarray

Example

import numpy as np

arr = np.array([10, 20, 30, 40])

print(arr)

Output:

[10 20 30 40]

---

🔹 Why NumPy for Machine Learning?

Machine Learning works heavily with numerical data.

For example:

Student Dataset

Age    Marks    Attendance
20      85         90
21      78         85
22      92         95

This data can be represented as:

import numpy as np

data = np.array([
    [20, 85, 90],
    [21, 78, 85],
    [22, 92, 95]
])

NumPy provides efficient operations for:

- Feature matrices
- Target vectors
- Mathematical calculations
- Matrix multiplication
- Feature scaling
- Random initialization
- Statistical analysis
- ML algorithm implementation

---

🔹 Installation

Install NumPy using pip:

pip install numpy

Verify installation:

import numpy as np

print(np.__version__)

---

🔹 Import NumPy

The standard convention is:

import numpy as np

Now instead of writing:

numpy.array()

we can write:

np.array()

---

🔹 NumPy Array

An array is a collection of elements stored in a structured format.

1D Array

arr = np.array([1, 2, 3, 4, 5])

print(arr)

Output:

[1 2 3 4 5]

2D Array

arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(arr)

Output:

[[1 2 3]
 [4 5 6]]

3D Array

arr = np.array([
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
])

---

🔹 Array Attributes

Important attributes:

arr.shape
arr.ndim
arr.size
arr.dtype

Example:

arr = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

print(arr.shape)
print(arr.ndim)
print(arr.size)
print(arr.dtype)

Output:

(2, 3)
2
6
int64

Meaning

Attribute| Meaning
"shape"| Dimensions of array
"ndim"| Number of dimensions
"size"| Total number of elements
"dtype"| Data type

---

🔹 Array Creation Methods

"np.array()"

arr = np.array([1, 2, 3])

"np.zeros()"

Creates an array filled with zeros.

arr = np.zeros((3, 4))

print(arr)

"np.ones()"

arr = np.ones((2, 3))

"np.full()"

arr = np.full((2, 3), 7)

Output:

[[7 7 7]
 [7 7 7]]

"np.arange()"

Creates evenly spaced values.

arr = np.arange(0, 10, 2)

print(arr)

Output:

[0 2 4 6 8]

"np.linspace()"

Creates a specified number of equally spaced values.

arr = np.linspace(0, 1, 5)

print(arr)

Output:

[0.   0.25 0.5  0.75 1.  ]

---

🔹 Data Types

NumPy supports several numerical data types:

int
float
bool
complex

Example:

arr = np.array([1, 2, 3], dtype=float)

print(arr)

Output:

[1. 2. 3.]

Convert data type:

arr = np.array([1.5, 2.5, 3.5])

new_arr = arr.astype(int)

print(new_arr)

---

🔹 Indexing and Slicing

Indexing

Python indexing starts from "0".

arr = np.array([10, 20, 30, 40])

print(arr[0])
print(arr[2])

Output:

10
30

Negative Indexing

print(arr[-1])

Output:

40

Slicing

print(arr[1:4])

Output:

[20 30 40]

---

🔹 2D Array Indexing

arr = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

print(arr[0, 1])

Output:

20

Syntax:

array[row, column]

---

🔹 Reshaping Arrays

Reshaping changes the structure without changing the data.

arr = np.arange(1, 7)

new_arr = arr.reshape(2, 3)

print(new_arr)

Output:

[[1 2 3]
 [4 5 6]]

Important Rule

The total number of elements must remain the same.

For example:

6 elements

2 × 3 = 6
3 × 2 = 6
1 × 6 = 6

But:

2 × 4 = 8 ❌

---

🔹 Flatten and Ravel

Convert multidimensional array into 1D.

"flatten()"

arr = np.array([
    [1, 2],
    [3, 4]
])

flat = arr.flatten()

print(flat)

"ravel()"

flat = arr.ravel()

Difference

- "flatten()" generally returns a copy.
- "ravel()" generally returns a view when possible.

---

🔹 Joining Arrays

Concatenate

a = np.array([1, 2])
b = np.array([3, 4])

result = np.concatenate((a, b))

print(result)

Output:

[1 2 3 4]

Vertical Stack

a = np.array([1, 2])
b = np.array([3, 4])

result = np.vstack((a, b))

Output:

[[1 2]
 [3 4]]

Horizontal Stack

result = np.hstack((a, b))

---

🔹 Splitting Arrays

arr = np.array([1, 2, 3, 4, 5, 6])

result = np.array_split(arr, 3)

print(result)

---

🔹 Copy vs View

This is important when working with datasets.

Copy

Creates an independent array.

a = np.array([1, 2, 3])

b = a.copy()

b[0] = 100

print(a)
print(b)

"a" remains unchanged.

View

Shares the underlying data.

a = np.array([1, 2, 3])

b = a.view()

b[0] = 100

print(a)

The original array can also change.

---

🔹 Mathematical Operations

NumPy supports vectorized mathematical operations.

a = np.array([10, 20, 30])
b = np.array([2, 4, 5])

Addition

a + b

Subtraction

a - b

Multiplication

a * b

Division

a / b

Power

a ** 2

---

🔹 Universal Functions

NumPy provides mathematical functions called ufuncs.

np.sqrt()
np.exp()
np.log()
np.sin()
np.cos()
np.abs()

Example:

arr = np.array([1, 4, 9, 16])

print(np.sqrt(arr))

Output:

[1. 2. 3. 4.]

---

🔹 Statistical Functions

Statistics are extremely important in Machine Learning.

Mean

arr = np.array([10, 20, 30, 40, 50])

print(np.mean(arr))

Output:

30.0

Median

np.median(arr)

Standard Deviation

np.std(arr)

Variance

np.var(arr)

Minimum

np.min(arr)

Maximum

np.max(arr)

---

🔹 Aggregation Functions

Common aggregation functions:

np.sum()
np.mean()
np.min()
np.max()
np.std()
np.var()

Example:

data = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

print(np.sum(data))

Output:

210

---

🔹 Axis

Understanding "axis" is extremely important for Machine Learning.

Consider:

data = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

"axis=0"

Operation is performed column-wise.

np.mean(data, axis=0)

Output:

[25. 35. 45.]

"axis=1"

Operation is performed row-wise.

np.mean(data, axis=1)

Output:

[20. 50.]

Remember

axis=0 → down the rows → column result

axis=1 → across columns → row result

---

🔹 Broadcasting

Broadcasting allows NumPy to perform operations on arrays of different shapes when their dimensions are compatible.

Example:

arr = np.array([10, 20, 30])

result = arr + 5

print(result)

Output:

[15 25 35]

The value "5" is effectively applied to every element.

ML Example

Suppose we have three features:

X = np.array([
    [10, 20, 30],
    [20, 30, 40],
    [30, 40, 50]
])

Subtract feature means:

mean = np.mean(X, axis=0)

X_centered = X - mean

Broadcasting automatically applies the mean vector to every row.

---

🔹 Boolean Indexing

Boolean indexing is useful for filtering data.

arr = np.array([10, 20, 30, 40, 50])

result = arr[arr > 25]

print(result)

Output:

[30 40 50]

Multiple Conditions

result = arr[(arr > 20) & (arr < 50)]

Output:

[30 40]

Use:

&

for AND.

Use:

|

for OR.

---

🔹 Sorting and Searching

Sort

arr = np.array([50, 10, 40, 20, 30])

print(np.sort(arr))

Output:

[10 20 30 40 50]

Argmax

Returns the index of the maximum value.

arr = np.array([10, 50, 30])

print(np.argmax(arr))

Output:

1

Argmin

np.argmin(arr)

---

🔹 Random Numbers

Random numbers are heavily used in Machine Learning.

np.random.seed(42)

A seed makes random results reproducible.

Random Float

np.random.rand(5)

Random Integer

np.random.randint(1, 100, size=5)

Random Normal Distribution

np.random.randn(5)

Random Matrix

X = np.random.randn(100, 5)

This can represent:

100 samples
5 features

---

🔹 Probability and Sampling

Generate random samples:

data = np.array([10, 20, 30, 40, 50])

sample = np.random.choice(data, size=3)

print(sample)

Useful for:

- Sampling
- Simulations
- ML experiments
- Train/test experiments
- Random initialization

---

🔹 Linear Algebra

Linear algebra is one of the most important mathematical foundations of Machine Learning.

NumPy provides:

np.linalg

Important operations include:

np.linalg.dot
np.linalg.inv
np.linalg.det
np.linalg.eig
np.linalg.norm

---

🔹 Matrix Operations

Create matrices:

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

Element-wise Multiplication

A * B

Output:

[[ 5 12]
 [21 32]]

Matrix Multiplication

Use:

A @ B

Output:

[[19 22]
 [43 50]]

This distinction is very important in Machine Learning.

---

🔹 Transpose

Transpose changes rows into columns.

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(A.T)

Output:

[[1 4]
 [2 5]
 [3 6]]

---

🔹 Dot Product

For vectors:

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = np.dot(a, b)

print(result)

Calculation:

1×4 + 2×5 + 3×6
= 4 + 10 + 18
= 32

---

🔹 Inverse and Determinant

Matrix:

A = np.array([
    [1, 2],
    [3, 4]
])

Determinant

np.linalg.det(A)

Inverse

np.linalg.inv(A)

A matrix must be non-singular to have a regular inverse.

---

🔹 Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors are important in:

- PCA
- Dimensionality Reduction
- Linear Algebra
- Machine Learning

Example:

A = np.array([
    [2, 0],
    [0, 3]
])

values, vectors = np.linalg.eig(A)

print(values)
print(vectors)

---

🔹 Vector Norm

Norm measures the magnitude of a vector.

x = np.array([3, 4])

norm = np.linalg.norm(x)

print(norm)

Output:

5.0

Because:

√(3² + 4²) = 5

---

🔹 Normalization

Normalization commonly scales values into a range such as "[0, 1]".

Formula:

X_normalized = (X - X_min) / (X_max - X_min)

Example:

X = np.array([10, 20, 30, 40, 50])

X_min = np.min(X)
X_max = np.max(X)

X_norm = (X - X_min) / (X_max - X_min)

print(X_norm)

Output:

[0.   0.25 0.5  0.75 1.  ]

Why?

Useful when features have different scales.

Example:

Age        → 18–60
Salary     → 20,000–2,00,000
Experience → 0–30

Scaling can help many ML algorithms.

---

🔹 Standardization

Standardization transforms data using:

Z = (X - μ) / σ

Where:

- "μ" = mean
- "σ" = standard deviation

NumPy example:

X = np.array([10, 20, 30, 40, 50])

mean = np.mean(X)
std = np.std(X)

X_standardized = (X - mean) / std

print(X_standardized)

The resulting data has approximately:

Mean = 0
Standard Deviation = 1

---

🔹 Handling Missing Values

NumPy represents missing numerical values using:

np.nan

Example:

data = np.array([10, 20, np.nan, 40, 50])

Check missing values:

np.isnan(data)

Count missing values:

np.sum(np.isnan(data))

Calculate mean while ignoring NaN:

np.nanmean(data)

Other useful functions:

np.nanmin()
np.nanmax()
np.nanmedian()
np.nansum()

---

🔹 One-Hot Encoding with NumPy

Suppose:

Classes:

Cat
Dog
Bird

Convert them to:

Cat  → [1, 0, 0]
Dog  → [0, 1, 0]
Bird → [0, 0, 1]

Example:

labels = np.array([0, 1, 2, 1])

one_hot = np.eye(3)[labels]

print(one_hot)

Output:

[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 1. 0.]]

This concept is commonly used in classification tasks.

---

🤖 NumPy in Machine Learning

A typical ML dataset can be represented as:

X = np.array([
    [20, 80],
    [21, 75],
    [22, 90],
    [23, 85]
])

y = np.array([0, 0, 1, 1])

Where:

X → Features
y → Target

Shape:

print(X.shape)
print(y.shape)

Output:

(4, 2)
(4,)

Meaning:

4 samples
2 features

---

🔹 Feature Matrix

Machine Learning usually represents data as:

X =

Sample 1 → [Feature1, Feature2, Feature3]
Sample 2 → [Feature1, Feature2, Feature3]
Sample 3 → [Feature1, Feature2, Feature3]

Example:

X = np.array([
    [25, 50000, 2],
    [30, 65000, 4],
    [35, 80000, 7]
])

Here:

Rows    → Samples
Columns → Features

---

🔹 Target Vector

y = np.array([
    0,
    1,
    1
])

The target contains the value we want the model to predict.

---

🔹 Train/Test Split Concept

NumPy can be used to shuffle data before splitting.

np.random.seed(42)

indices = np.random.permutation(len(X))

X = X[indices]
y = y[indices]

Then:

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

For production ML projects, libraries such as scikit-learn generally provide more robust train/test splitting utilities.

---

🔹 Machine Learning Example: Linear Regression

Linear regression equation:

y = Xw + b

Where:

X → Input features
w → Weights
b → Bias
y → Prediction

Example:

import numpy as np

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([
    2,
    4,
    6,
    8,
    10
])

The relationship is:

y = 2x

Weight:

w = np.array([2])

b = 0

predictions = X @ w + b

print(predictions)

Output:

[ 2  4  6  8 10]

---

🔹 Mean Squared Error

MSE is commonly used as a regression loss function.

Formula:

MSE = 1/n Σ(y - ŷ)²

NumPy implementation:

y_true = np.array([2, 4, 6])
y_pred = np.array([2, 5, 5])

mse = np.mean((y_true - y_pred) ** 2)

print(mse)

Output:

0.666...

---

🔹 Gradient Descent with NumPy

Gradient Descent is a fundamental optimization algorithm used to train many ML models.

Basic idea:

New Weight = Old Weight - Learning Rate × Gradient

Example:

import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

w = 0.0
learning_rate = 0.01

for epoch in range(1000):

    predictions = w * X

    error = predictions - y

    gradient = np.mean(error * X)

    w = w - learning_rate * gradient

print("Weight:", w)

The learned weight approaches:

w ≈ 2

This demonstrates how NumPy can be used to implement ML algorithms from scratch.

---

🔹 Vectorization

Vectorization means performing operations on complete arrays instead of using Python loops.

Without vectorization

result = []

for x in X:
    result.append(x * 2)

With NumPy

result = X * 2

NumPy vectorization is generally much more efficient for numerical operations.

---

🔹 Why NumPy is Fast

NumPy is fast because:

- Arrays use efficient memory layouts.
- Numerical operations are implemented in optimized compiled code.
- Vectorized operations reduce Python-level loops.
- Broadcasting avoids unnecessary manual iteration.
- It provides optimized mathematical and linear algebra routines.

---

🔹 NumPy vs Python List

Python List

a = [1, 2, 3]
b = [4, 5, 6]

NumPy

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)

Output:

[5 7 9]

NumPy is specifically designed for numerical array computation.

---

🔹 NumPy in the Data Science Ecosystem

NumPy is a foundation for many Python data and ML libraries.

              NumPy
                │
        ┌───────┼────────┐
        ↓       ↓        ↓
     Pandas   SciPy   Matplotlib
        │
        ↓
 Scikit-Learn
        │
        ↓
 Machine Learning

NumPy concepts are also important when working with:

- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- TensorFlow
- PyTorch
- OpenCV

---

📚 Important NumPy Functions Cheat Sheet

Function| Purpose
"np.array()"| Create array
"np.zeros()"| Create zeros
"np.ones()"| Create ones
"np.full()"| Fill with value
"np.arange()"| Range of values
"np.linspace()"| Evenly spaced values
"np.reshape()"| Change shape
"np.flatten()"| Convert to 1D copy
"np.ravel()"| Convert to 1D
"np.concatenate()"| Join arrays
"np.vstack()"| Vertical stacking
"np.hstack()"| Horizontal stacking
"np.mean()"| Mean
"np.median()"| Median
"np.std()"| Standard deviation
"np.var()"| Variance
"np.sum()"| Sum
"np.min()"| Minimum
"np.max()"| Maximum
"np.argmin()"| Index of minimum
"np.argmax()"| Index of maximum
"np.sort()"| Sort array
"np.unique()"| Unique values
"np.where()"| Conditional selection
"np.isnan()"| Detect NaN
"np.random.rand()"| Random values
"np.random.randn()"| Normal distribution
"np.random.randint()"| Random integers
"np.random.choice()"| Random sampling
"np.dot()"| Dot product
"np.linalg.inv()"| Matrix inverse
"np.linalg.det()"| Determinant
"np.linalg.eig()"| Eigenvalues/eigenvectors
"np.linalg.norm()"| Vector/matrix norm

---

🧠 Most Important Concepts for Machine Learning

If your goal is Machine Learning, focus especially on these NumPy topics:

⭐ Level 1 — Fundamentals

- Arrays
- Dimensions
- Shape
- Indexing
- Slicing
- Data types
- Reshaping

⭐ Level 2 — Numerical Operations

- Vectorization
- Broadcasting
- Mathematical functions
- Aggregation
- Axis
- Boolean indexing

⭐ Level 3 — Statistics

- Mean
- Median
- Variance
- Standard deviation
- Min/Max
- Percentile

⭐ Level 4 — Linear Algebra

- Vectors
- Matrices
- Matrix multiplication
- Dot product
- Transpose
- Norm
- Inverse
- Determinant
- Eigenvalues
- Eigenvectors

⭐ Level 5 — ML Applications

- Feature matrix
- Target vector
- Normalization
- Standardization
- Missing values
- Random initialization
- Data shuffling
- MSE
- Gradient Descent

---

🎯 Practice Questions

Beginner

1. Create a NumPy array containing numbers from 1 to 20.
2. Find the shape and dimensions of an array.
3. Find the mean, median and standard deviation.
4. Reverse a NumPy array.
5. Extract even numbers from an array.
6. Convert a 1D array into a 3×3 matrix.
7. Find the maximum and minimum values.
8. Count unique values.

Intermediate

9. Explain broadcasting with an example.
10. Calculate column-wise mean using "axis=0".
11. Calculate row-wise mean using "axis=1".
12. Normalize an array between 0 and 1.
13. Standardize an array.
14. Handle "NaN" values.
15. Perform matrix multiplication.
16. Calculate the determinant and inverse of a matrix.

Machine Learning

17. Create a feature matrix "X".
18. Create a target vector "y".
19. Implement MSE using NumPy.
20. Implement linear regression prediction.
21. Implement gradient descent from scratch.
22. Implement Min-Max normalization.
23. Implement standardization.
24. Implement one-hot encoding.
25. Implement a simple linear regression model without scikit-learn.

---

🚀 Mini Project: Linear Regression from Scratch

Create a simple ML model using only NumPy.

Dataset

import numpy as np

X = np.array([1, 2, 3, 4, 5])
y