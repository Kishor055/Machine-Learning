# 🔢 Arithmetic

> **Mathematics for Machine Learning — Chapter 01**

Arithmetic is the most fundamental layer of mathematics used in Machine Learning. Every algorithm, regardless of how advanced it appears, ultimately performs numerical computations.

Operations such as **addition, subtraction, multiplication, division, powers, roots, ratios, percentages, and averages** form the foundation for more advanced mathematical topics such as **algebra, linear algebra, calculus, probability, statistics, and optimization**.

This chapter develops arithmetic from first principles and connects each concept to practical **Machine Learning and Data Science** applications.

---

## 📌 Learning Objectives

By the end of this chapter, you should be able to:

* Understand the mathematical concept of numbers.
* Classify numbers into different number systems.
* Perform arithmetic operations confidently.
* Apply the correct order of operations.
* Work with fractions and decimals.
* Convert between fractions, decimals, and percentages.
* Understand ratios and proportions.
* Work with powers, exponents, and roots.
* Understand absolute values and distances.
* Calculate arithmetic and weighted means.
* Calculate rates and percentage changes.
* Understand approximation and numerical precision.
* Understand floating-point representation.
* Recognize how arithmetic appears inside Machine Learning algorithms.
* Implement arithmetic operations using Python.

---

# 📚 Table of Contents

1. [Why Arithmetic Matters in Machine Learning](#1-why-arithmetic-matters-in-machine-learning)
2. [What Is Arithmetic?](#2-what-is-arithmetic)
3. [Numbers and Number Systems](#3-numbers-and-number-systems)
4. [The Number Line](#4-the-number-line)
5. [Basic Arithmetic Operations](#5-basic-arithmetic-operations)
6. [Properties of Arithmetic](#6-properties-of-arithmetic)
7. [Order of Operations](#7-order-of-operations)
8. [Fractions](#8-fractions)
9. [Decimals](#9-decimals)
10. [Percentages](#10-percentages)
11. [Ratios and Proportions](#11-ratios-and-proportions)
12. [Powers and Exponents](#12-powers-and-exponents)
13. [Roots and Radicals](#13-roots-and-radicals)
14. [Absolute Value](#14-absolute-value)
15. [Negative Numbers](#15-negative-numbers)
16. [Rounding and Approximation](#16-rounding-and-approximation)
17. [Scientific Notation](#17-scientific-notation)
18. [Significant Figures](#18-significant-figures)
19. [Arithmetic Mean](#19-arithmetic-mean)
20. [Weighted Mean](#20-weighted-mean)
21. [Rates and Ratios of Change](#21-rates-and-ratios-of-change)
22. [Percentage Change](#22-percentage-change)
23. [Arithmetic and Measurement](#23-arithmetic-and-measurement)
24. [Numerical Precision](#24-numerical-precision)
25. [Floating-Point Arithmetic](#25-floating-point-arithmetic)
26. [Arithmetic in Machine Learning](#26-arithmetic-in-machine-learning)
27. [Worked ML Examples](#27-worked-ml-examples)
28. [Python Implementation](#28-python-implementation)
29. [Common Mistakes](#29-common-mistakes)
30. [Practice Problems](#30-practice-problems)
31. [Quick Reference](#31-quick-reference)
32. [Key Takeaways](#32-key-takeaways)
33. [What's Next?](#33-whats-next)

---

# 1. Why Arithmetic Matters in Machine Learning

Machine Learning is fundamentally a mathematical discipline.

When a model receives data, it does not "understand" the data in the human sense. Numerical algorithms transform the input into mathematical representations and perform calculations on those representations.

A simplified Machine Learning pipeline looks like:

```text
Raw Data
   │
   ▼
Numerical Representation
   │
   ▼
Arithmetic Operations
   │
   ▼
Feature Transformation
   │
   ▼
Mathematical Model
   │
   ▼
Prediction
   │
   ▼
Error / Loss Calculation
   │
   ▼
Optimization
```

For example, a simple prediction model might calculate:

$$
\hat{y}=wx+b
$$

This seemingly simple equation already requires:

* Multiplication
* Addition
* Signed numbers
* Decimal values

More advanced models perform millions or billions of similar operations.

Therefore:

> **Understanding arithmetic means understanding the numerical building blocks underneath Machine Learning.**

---

# 2. What Is Arithmetic?

**Arithmetic** is the branch of mathematics concerned primarily with numbers and operations performed on numbers.

The fundamental operations are:

| Operation      |     Symbol |         Example |
| -------------- | ---------: | --------------: |
| Addition       |      \(+\) |       \(5+3=8\) |
| Subtraction    |      \(-\) |       \(5-3=2\) |
| Multiplication | \(\times\) | \(5\times3=15\) |
| Division       |   \(\div\) |    \(6\div3=2\) |

Arithmetic also includes concepts such as:

* Fractions
* Decimals
* Percentages
* Ratios
* Proportions
* Powers
* Roots
* Absolute values
* Approximation
* Numerical representation

These concepts later become components of more advanced mathematics.

---

# 3. Numbers and Number Systems

A **number** is a mathematical object used to represent quantity, magnitude, position, or relationships.

Different types of numbers are organized into number systems.

---

## 3.1 Natural Numbers

Natural numbers are generally used for counting.

$$
1,2,3,4,5,\ldots
$$

Some conventions also include zero:

$$
0,1,2,3,\ldots
$$

### Examples

```text
1
7
100
1000
```

Natural numbers are commonly used for:

* Counting observations
* Number of samples
* Number of features
* Number of iterations

---

## 3.2 Whole Numbers

Whole numbers contain zero and all positive integers:

$$
0,1,2,3,4,\ldots
$$

---

## 3.3 Integers

Integers include negative numbers, zero, and positive numbers:

Yes — this is the standard mathematical notation for the **set of integers**.

### Explanation

$$
\mathbb{Z}
=
\{\ldots,-3,-2,-1,0,1,2,3,\ldots\}
$$

Here:

* **\(\mathbb{Z}\)** → the symbol used to represent the set of **integers**.
* **\(\{\}\)** → denotes a **set**, i.e. a collection of elements.
* **\(\ldots\)** → means the pattern continues indefinitely in both directions.
* **Negative integers** → \(-1,-2,-3,\ldots\)
* **Zero** → \(0\)
* **Positive integers** → \(1,2,3,\ldots\)

So, in words:

> **The set of integers contains all negative whole numbers, zero, and all positive whole numbers.**

### Number-line representation

```text
←────────────── Integers ──────────────→

...  -5  -4  -3  -2  -1   0   1   2   3   4   5  ...
                  Negative       Positive
```

### Important distinction

Integers **do not include fractional or decimal values** such as:

$$
\frac12,\quad 2.5,\quad -3.14
$$

Those are real numbers, but they are not integers.

### Number-system hierarchy

A useful way to remember the relationship is:

$$
\mathbb{N}\subseteq\mathbb{W}\subseteq\mathbb{Z}\subseteq\mathbb{Q}\subseteq\mathbb{R}
$$

where:

* \(\mathbb{N}\) = Natural numbers
* \(\mathbb{W}\) = Whole numbers
* \(\mathbb{Z}\) = Integers
* \(\mathbb{Q}\) = Rational numbers
* \(\mathbb{R}\) = Real numbers

For your README, this is a good professional definition:

> **Integers (\(\mathbb{Z}\))** are the set of positive and negative whole numbers, including zero. They can be written as \(\mathbb{Z}=\{\ldots,-3,-2,-1,0,1,2,3,\ldots\}\). Integers do not contain fractional or decimal values.

Examples:

```text
-100
-5
0
7
50
```

Negative values are common in Machine Learning.

For example, a model's prediction error can be negative:

$$
Error = Actual-Predicted
$$

---

## 3.4 Rational Numbers

A rational number can be written as:

$$
\frac{p}{q}
$$

where:

$$
p,q\in\mathbb{Z},\qquad q\neq0
$$

Examples:

$$
\frac12,\quad
\frac34,\quad
-\frac57,\quad
2
$$

Every integer is rational because:

$$
5=\frac51
$$

---

## 3.5 Irrational Numbers

An irrational number cannot be represented as the ratio of two integers.

Examples:

$$
\sqrt2
$$

$$
\pi
$$

Their decimal representations are infinite and non-repeating.

For example:

$$
\sqrt2\approx1.41421356\ldots
$$

---

## 3.6 Real Numbers

Real numbers include both rational and irrational numbers.

The set of real numbers is represented by:

$$
\mathbb{R}
$$

Machine Learning frequently operates on real-valued quantities such as:

```text
0.25
-3.14
1.5
0.00001
100.75
```

---

## 3.7 Number-System Relationship

The common hierarchy is:

```text
Complex Numbers
       │
       └── Real Numbers
              │
              ├── Rational Numbers
              │      │
              │      └── Integers
              │             │
              │             └── Whole Numbers
              │                    │
              │                    └── Natural Numbers
              │
              └── Irrational Numbers
```

The deeper number systems will become important when studying more advanced mathematics.

---

# 4. The Number Line

A number line provides a visual representation of numbers.

```text
←───────────────|───────────────→
      Negative  0     Positive

   -3   -2   -1   0   1   2   3
```

Numbers increase as we move right.

Numbers decrease as we move left.

---

## Distance on a Number Line

The distance between two numbers can be expressed using absolute value.

For numbers \(a\) and \(b\):

$$
Distance=|a-b|
$$

Example:

$$
a=3,\qquad b=-2
$$

Then:

$$
|3-(-2)|=|5|=5
$$

This idea becomes important when studying:

* Distance metrics
* K-Nearest Neighbors
* Clustering
* Optimization
* Error functions

---

# 5. Basic Arithmetic Operations

## 5.1 Addition

Addition combines quantities.

$$
a+b
$$

Example:

$$
8+5=13
$$

### ML Example

Suppose a dataset contains three groups:

$$
100+200+300
$$

Total:

$$
600
$$

---

## 5.2 Subtraction

Subtraction measures difference.

$$
a-b
$$

Example:

$$
20-8=12
$$

### ML Example

Prediction error:

$$
e=y-\hat{y}
$$

If:

$$
y=100
$$

and:

$$
\hat{y}=92
$$

then:

$$
e=100-92=8
$$

---

## 5.3 Multiplication

Multiplication is repeated addition and scaling.

$$
a\times b
$$

Example:

$$
6\times4=24
$$

In Machine Learning, multiplication is fundamental.

A linear model uses:

$$
wx
$$

where \(w\) is a weight and \(x\) is a feature.

---

## 5.4 Division

Division determines how one quantity is distributed relative to another.

$$
a\div b=\frac{a}{b}
$$

Example:

$$
20\div5=4
$$

Division by zero is undefined:

$$
\frac{x}{0}
$$

---

# 6. Properties of Arithmetic

Understanding arithmetic properties helps simplify mathematical expressions and recognize patterns.

---

## 6.1 Commutative Property

Order does not matter for addition and multiplication.

### Addition

$$
a+b=b+a
$$

Example:

$$
3+5=5+3
$$

### Multiplication

$$
ab=ba
$$

Example:

$$
3\times5=5\times3
$$

### Important

Subtraction and division are **not generally commutative**.

$$
5-3\neq3-5
$$

$$
10\div2\neq2\div10
$$

---

## 6.2 Associative Property

Grouping does not matter for addition and multiplication.

### Addition

$$
(a+b)+c=a+(b+c)
$$

### Multiplication

$$
(ab)c=a(bc)
$$

---

## 6.3 Distributive Property

Multiplication distributes over addition:

$$
a(b+c)=ab+ac
$$

Example:

$$
3(4+5)
$$

$$
=3(9)
$$

$$
=27
$$

Also:

$$
3(4)+3(5)=12+15=27
$$

This property becomes extremely important in algebra and linear algebra.

---

## 6.4 Identity Elements

For addition:

$$
a+0=a
$$

Therefore, zero is the additive identity.

For multiplication:

$$
a\times1=a
$$

Therefore, one is the multiplicative identity.

---

# 7. Order of Operations

Mathematical expressions can contain multiple operations.

A standard convention determines the order in which they are evaluated.

A common mnemonic is:

> **PEMDAS**

```text
P → Parentheses
E → Exponents
M → Multiplication
D → Division
A → Addition
S → Subtraction
```

Example:

$$
2+3\times4
$$

Multiplication occurs first:

$$
2+12=14
$$

Therefore:

$$
\boxed{14}
$$

---

## Parentheses

Consider:

$$
(2+3)\times4
$$

First:

$$
2+3=5
$$

Then:

$$
5\times4=20
$$

Therefore:

$$
(2+3)\times4=20
$$

while:

$$
2+3\times4=14
$$

Parentheses can therefore completely change a result.

---

# 8. Fractions

A fraction represents a ratio of two quantities.

$$
\frac{a}{b}
$$

where:

* \(a\) = numerator
* \(b\) = denominator
* \(b\neq0\)

Example:

$$
\frac34
$$

means three parts out of four.

---

## Equivalent Fractions

Fractions can have different representations but the same value.

$$
\frac12=\frac24=\frac36
$$

---

## Adding Fractions

For:

$$
\frac12+\frac14
$$

Convert to a common denominator:

$$
\frac12=\frac24
$$

Therefore:

$$
\frac24+\frac14=\frac34
$$

---

## Multiplying Fractions

$$
\frac23\times\frac45
$$

$$
=\frac{2\times4}{3\times5}
$$

$$
=\frac8{15}
$$

---

## Dividing Fractions

To divide by a fraction, multiply by its reciprocal:

$$
\frac23\div\frac45
$$

$$
=\frac23\times\frac54
$$

$$
=\frac56
$$

---

## ML Connection

Fractions appear in:

* Probabilities
* Ratios
* Averages
* Normalization
* Statistical estimators
* Loss functions

For example:

$$
Mean=\frac{\text{Sum of Values}}{\text{Number of Values}}
$$

---

# 9. Decimals

Decimals represent numbers using a base-10 positional system.

Example:

$$
12.345
$$

can be expanded as:

$$
12+\frac3{10}+\frac4{100}+\frac5{1000}
$$

Therefore:

$$
12.345
$$

contains:

* Ones
* Tenths
* Hundredths
* Thousandths

---

## Fraction ↔ Decimal

Examples:

$$
\frac12=0.5
$$

$$
\frac14=0.25
$$

$$
\frac34=0.75
$$

---

## Why Decimals Matter in ML

Machine Learning datasets frequently contain continuous numerical values:

```text
Height = 172.5
Weight = 68.4
Temperature = 36.7
Probability = 0.82
Learning Rate = 0.001
```

These values are usually represented computationally using floating-point numbers.

---

# 10. Percentages

A percentage means "per hundred."

$$
x\%=\frac{x}{100}
$$

Examples:

$$
50\%=0.5
$$

$$
25\%=0.25
$$

$$
5\%=0.05
$$

---

## Percentage of a Quantity

To calculate 20% of 500:

$$
\frac{20}{100}\times500
$$

$$
=100
$$

---

## ML Example — Dataset Split

Suppose there are 10,000 observations.

If 80% are used for training:

$$
10000\times0.8=8000
$$

Therefore:

```text
Training samples = 8,000
Remaining samples = 2,000
```

---

# 11. Ratios and Proportions

A ratio compares two quantities.

For example:

$$
2:3
$$

means two units of one quantity for every three units of another.

Ratios can also be written as:

$$
\frac23
$$

---

## Proportion

A proportion states that two ratios are equivalent:

$$
\frac ab=\frac cd
$$

Example:

$$
\frac24=\frac36
$$

Both equal:

$$
0.5
$$

---

## ML Applications

Ratios appear in:

* Train/test splits
* Class distributions
* Feature relationships
* Sampling
* Evaluation metrics
* Probabilities

---

# 12. Powers and Exponents

An exponent indicates repeated multiplication.

$$
a^n
$$

means:

$$
\underbrace{a\times a\times\cdots\times a}_{n\text{ times}}
$$

Example:

$$
2^4=2\times2\times2\times2=16
$$

---

## Exponent Rules

### Product Rule

$$
a^m a^n=a^{m+n}
$$

---

### Quotient Rule

$$
\frac{a^m}{a^n}=a^{m-n}
$$

for \(a\neq0\).

---

### Power Rule

$$
(a^m)^n=a^{mn}
$$

---

### Zero Exponent

For \(a\neq0\):

$$
a^0=1
$$

---

### Negative Exponent

$$
a^{-n}=\frac1{a^n}
$$

Example:

$$
2^{-3}=\frac18
$$

---

## ML Connection

Exponentials appear extensively in:

* Logistic regression
* Softmax
* Neural networks
* Probability distributions
* Loss functions
* Optimization

For example, the logistic sigmoid function is:

$$
\sigma(x)=\frac1{1+e^{-x}}
$$

A strong understanding of exponents is therefore essential before studying these concepts.

---

# 13. Roots and Radicals

Roots reverse the operation of powers.

The square root:

$$
\sqrt{x}
$$

asks:

> Which number multiplied by itself produces \(x\)?

Example:

$$
\sqrt{25}=5
$$

because:

$$
5^2=25
$$

---

## Common Roots

$$
\sqrt4=2
$$

$$
\sqrt9=3
$$

$$
\sqrt{16}=4
$$

$$
\sqrt{100}=10
$$

---

## ML Connection — Euclidean Distance

For two points:

$$
A=(x_1,y_1)
$$

and:

$$
B=(x_2,y_2)
$$

Euclidean distance is:

$$
d=
\sqrt{
(x_2-x_1)^2+
(y_2-y_1)^2
}
$$

This formula appears in:

* KNN
* Clustering
* Similarity calculations
* Geometry
* Computer vision
* Data analysis

---

# 14. Absolute Value

Absolute value represents the distance of a number from zero.

It is written as:

$$
|x|
$$

Examples:

$$
|5|=5
$$

$$
|-5|=5
$$

$$
|0|=0
$$

A useful definition is:

$$
|x|=
\begin{cases}
x & x\ge0\\
-x & x<0
\end{cases}
$$

---

## ML Connection — Absolute Error

Absolute prediction error:

$$
AE=|y-\hat y|
$$

Example:

$$
y=100
$$

$$
\hat y=92
$$

Then:

$$
AE=|100-92|=8
$$

The **Mean Absolute Error (MAE)** extends this idea across multiple observations.

---

# 15. Negative Numbers

Negative numbers represent quantities below zero or values in the opposite direction on a number line.

Examples:

$$
-1,-5,-10
$$

---

## Addition

$$
(-5)+(-3)=-8
$$

For different signs:

$$
(-5)+3=-2
$$

---

## Multiplication

Two negative numbers produce a positive result:

$$
(-5)(-3)=15
$$

A negative and positive number produce a negative result:

$$
(-5)(3)=-15
$$

---

## ML Connection

Negative numbers naturally occur in:

* Model weights
* Biases
* Prediction errors
* Gradients
* Feature values
* Correlations

---

# 16. Rounding and Approximation

Many mathematical quantities cannot be represented exactly using a finite number of decimal digits.

For example:

$$
\pi=3.14159265358979\ldots
$$

We may approximate it as:

$$
3.14
$$

---

## Rounding Rule

When rounding to a particular decimal position:

```text
0–4 → retain the current digit
5–9 → increase the current digit
```

Example:

$$
3.146\rightarrow3.15
$$

while:

$$
3.143\rightarrow3.14
$$

---

## Why Rounding Matters in ML

Rounding can introduce numerical error.

Suppose:

$$
x=1.23456789
$$

If we immediately round to:

$$
1.23
$$

and repeatedly use that approximation, errors can accumulate.

### General Principle

> **Keep sufficient numerical precision during computation and round primarily when presenting results.**

---

# 17. Scientific Notation

Scientific notation represents very large or very small numbers compactly.

General form:

$$
a\times10^n
$$

where:

$$
1\le |a|<10
$$

---

## Examples

$$
3,000,000=3\times10^6
$$

$$
0.000004=4\times10^{-6}
$$

---

## ML Examples

Learning rate:

$$
0.0001=1\times10^{-4}
$$

Very small numerical value:

$$
0.000000001=1\times10^{-9}
$$

Scientific notation is useful for understanding:

* Learning rates
* Numerical errors
* Small probabilities
* Large datasets
* Model parameters

---

# 18. Significant Figures

Significant figures indicate the meaningful digits in a numerical value.

For example:

$$
12.34
$$

contains four significant digits.

Significant figures are particularly important when working with:

* Measurements
* Scientific experiments
* Approximate calculations
* Numerical reporting

In Machine Learning, computational precision is generally governed by the numeric data type rather than manually counting significant figures, but the underlying idea of meaningful numerical precision remains important.

---

# 19. Arithmetic Mean

The arithmetic mean is one of the most important concepts connecting arithmetic with statistics and Machine Learning.

For observations:

$$
x_1,x_2,\ldots,x_n
$$

the arithmetic mean is:

$$
\bar{x}
=
\frac{x_1+x_2+\cdots+x_n}{n}
$$

Using summation notation:

$$
\boxed{
\bar{x}=
\frac1n
\sum_{i=1}^{n}x_i
}
$$

---

## Example

Given:

$$
10,20,30,40,50
$$

Sum:

$$
150
$$

Number of observations:

$$
5
$$

Therefore:

$$
\bar{x}=\frac{150}{5}=30
$$

---

## ML Applications

The mean appears in:

* Feature preprocessing
* Mean imputation
* Standardization
* Loss functions
* Statistics
* Data analysis

For example, Mean Squared Error is:

$$
MSE=
\frac1n
\sum_{i=1}^{n}
(y_i-\hat y_i)^2
$$

---

# 20. Weighted Mean

Sometimes different observations have different levels of importance.

A weighted mean is:

$$
\bar{x}_w=
\frac{\sum_{i=1}^{n}w_ix_i}
{\sum_{i=1}^{n}w_i}
$$

where:

* \(x_i\) = value
* \(w_i\) = weight

---

## Example

Suppose:

```text
Score     Weight
80           2
90           3
```

Then:

$$
\bar{x}_w=
\frac{80(2)+90(3)}{2+3}
$$

$$
=
\frac{160+270}{5}
$$

$$
=86
$$

Weighted averages are useful whenever observations do not contribute equally.

---

# 21. Rates and Ratios of Change

A rate measures how one quantity changes relative to another.

General form:

$$
Rate=
\frac{\text{Change in Quantity}}
{\text{Change in Reference Quantity}}
$$

Examples:

* kilometers per hour
* samples per second
* loss per epoch
* dollars per unit

---

## ML Example

Suppose model loss changes from:

$$
0.80
$$

to:

$$
0.60
$$

over 2 epochs.

Change:

$$
0.60-0.80=-0.20
$$

Average change per epoch:

$$
\frac{-0.20}{2}=-0.10
$$

The negative sign indicates that the loss decreased.

---

# 22. Percentage Change

Percentage change measures relative change.

$$
\boxed{
Percentage\ Change=
\frac{New-Old}{Old}\times100
}
$$

---

## Example

Old value:

$$
200
$$

New value:

$$
150
$$

Therefore:

$$
\frac{150-200}{200}\times100
$$

$$
=-25\%
$$

The negative value indicates a decrease.

---

# 23. Arithmetic and Measurement

Arithmetic is also used to represent physical measurements.

Examples:

```text
Height     → centimeters
Weight     → kilograms
Distance   → kilometers
Time       → seconds
Temperature → °C
```

---

## Unit Consistency

Suppose one feature is represented in meters and another in centimeters.

```text
1 meter = 100 centimeters
```

If units are mixed without conversion, numerical calculations can become misleading.

This becomes particularly important in algorithms based on numerical magnitude or distance.

---

## Feature Scaling

Suppose a dataset contains:

```text
Age      → 18–80
Income   → 20,000–2,000,000
```

The two features have very different numerical scales.

Some ML algorithms can be sensitive to such scale differences.

Common scaling techniques include:

* Min-Max Scaling
* Standardization
* Robust Scaling

These topics will be studied in the **Data Preprocessing** section.

---

# 24. Numerical Precision

Mathematical real numbers can contain infinitely many decimal digits.

Computers, however, have finite memory.

Therefore, computers must represent most real numbers using approximations.

This creates the concept of:

> **Numerical precision**

For example:

$$
\frac13=0.333333333\ldots
$$

A computer cannot store infinitely many digits.

It stores an approximation.

---

## Precision vs Accuracy

These concepts are related but different.

### Precision

Describes the level of numerical detail or consistency of representation.

### Accuracy

Describes how close a value is to the true or reference value.

In scientific computing and Machine Learning, distinguishing these concepts is important.

---

# 25. Floating-Point Arithmetic

Most programming languages use floating-point representations for decimal numerical values.

Python example:

```python
x = 0.1
y = 0.2

print(x + y)
```

You may see:

```text
0.30000000000000004
```

instead of exactly:

```text
0.3
```

This does **not** mean Python's arithmetic is broken.

The issue comes from representing decimal fractions using a binary floating-point system.

Many decimal values cannot be represented exactly in finite binary form.

---

## Floating-Point Comparison

Instead of blindly checking:

```python
a == b
```

for approximate numerical calculations, a tolerance can be appropriate.

Python provides:

```python
import math

math.isclose(0.1 + 0.2, 0.3)
```

This concept becomes increasingly important when implementing:

* Numerical algorithms
* Optimization
* Neural networks
* Scientific computing

---

# 26. Arithmetic in Machine Learning

Arithmetic appears throughout the Machine Learning lifecycle.

```text
                 MACHINE LEARNING
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
          Features             Labels
             │                   │
             └─────────┬─────────┘
                       ▼
                Preprocessing
                       │
                       ▼
              Mathematical Model
                       │
                       ▼
                  Prediction
                       │
                       ▼
                 Error / Loss
                       │
                       ▼
                 Optimization
                       │
                       ▼
                   Evaluation
```

Every stage involves numerical operations.

---

# 27. Worked ML Examples

## Example 1 — Prediction Error

Actual value:

$$
y=50
$$

Predicted value:

$$
\hat y=45
$$

Error:

$$
e=y-\hat y
$$

Therefore:

$$
e=50-45=5
$$

---

## Example 2 — Absolute Error

$$
AE=|y-\hat y|
$$

Therefore:

$$
AE=|50-45|
$$

$$
AE=5
$$

---

## Example 3 — Squared Error

$$
SE=(y-\hat y)^2
$$

$$
SE=(50-45)^2
$$

$$
SE=25
$$

---

## Example 4 — Mean Squared Error

Suppose:

```text
Actual:     10, 20, 30
Predicted:  12, 18, 25
```

Errors:

$$
10-12=-2
$$

$$
20-18=2
$$

$$
30-25=5
$$

Squared errors:

$$
(-2)^2=4
$$

$$
2^2=4
$$

$$
5^2=25
$$

Therefore:

$$
MSE=
\frac{4+4+25}{3}
$$

$$
MSE=11
$$

---

## Example 5 — Euclidean Distance

Consider two points:

$$
A=(2,3)
$$

$$
B=(5,7)
$$

Distance:

$$
d=
\sqrt{(5-2)^2+(7-3)^2}
$$

$$
=
\sqrt{3^2+4^2}
$$

$$
=\sqrt{9+16}
$$

$$
=\sqrt{25}
$$

$$
=5
$$

This basic arithmetic becomes the foundation for distance-based algorithms.

---

# 28. Python Implementation

Python provides several built-in arithmetic operators.

```python
a = 10
b = 3

print(a + b)   # Addition
print(a - b)   # Subtraction
print(a * b)   # Multiplication
print(a / b)   # Division
print(a // b)  # Floor division
print(a % b)   # Modulo
print(a ** b)  # Exponentiation
```

---

## Operator Reference

| Operator | Meaning        |  Example | Result |
| -------- | -------------- | -------: | -----: |
| `+`      | Addition       |  `5 + 2` |    `7` |
| `-`      | Subtraction    |  `5 - 2` |    `3` |
| `*`      | Multiplication |  `5 * 2` |   `10` |
| `/`      | Division       |  `5 / 2` |  `2.5` |
| `//`     | Floor Division | `5 // 2` |    `2` |
| `%`      | Modulo         |  `5 % 2` |    `1` |
| `**`     | Exponentiation | `5 ** 2` |   `25` |

---

## Arithmetic Mean

```python
values = [10, 20, 30, 40, 50]

mean = sum(values) / len(values)

print(mean)
```

Output:

```text
30.0
```

---

## Percentage Change

```python
old_value = 200
new_value = 150

percentage_change = (
    (new_value - old_value) / old_value
) * 100

print(percentage_change)
```

Output:

```text
-25.0
```

---

## Euclidean Distance

```python
import math

x1, y1 = 2, 3
x2, y2 = 5, 7

distance = math.sqrt(
    (x2 - x1) ** 2 +
    (y2 - y1) ** 2
)

print(distance)
```

Output:

```text
5.0
```

---

# 29. Common Mistakes

## ❌ Mistake 1 — Ignoring Operator Precedence

Incorrect:

$$
2+3\times4=20
$$

Correct:

$$
2+3\times4=14
$$

---

## ❌ Mistake 2 — Dividing by Zero

$$
\frac{x}{0}
$$

is undefined.

---

## ❌ Mistake 3 — Confusing Percentage and Decimal

Remember:

$$
25\%=0.25
$$

not:

$$
25
$$

---

## ❌ Mistake 4 — Ignoring Parentheses

These are different:

$$
-3^2=-9
$$

and:

$$
(-3)^2=9
$$

---

## ❌ Mistake 5 — Rounding Too Early

Avoid repeatedly rounding intermediate values unless necessary.

---

## ❌ Mistake 6 — Assuming Floating-Point Values Are Exact

Decimal values stored by computers may be approximations.

---

## ❌ Mistake 7 — Ignoring Units

Always ensure measurements use compatible units before performing calculations.

---

# 30. Practice Problems

## 🟢 Beginner

### 1. Addition

Calculate:

$$
25+37
$$

### 2. Subtraction

Calculate:

$$
100-47
$$

### 3. Multiplication

Calculate:

$$
12\times8
$$

### 4. Division

Calculate:

$$
144\div12
$$

### 5. Exponent

Calculate:

$$
2^5
$$

---

## 🟡 Intermediate

### 6. Order of Operations

Evaluate:

$$
5+3\times4-2
$$

### 7. Fractions

Calculate:

$$
\frac12+\frac34
$$

### 8. Percentage

Find 15% of 800.

### 9. Mean

Calculate the mean:

$$
10,20,30,40,50
$$

### 10. Percentage Change

Calculate the percentage change from:

$$
500\rightarrow425
$$

---

## 🔴 Machine Learning

### 11. Prediction Error

Actual:

$$
100
$$

Prediction:

$$
92
$$

Calculate:

* Error
* Absolute Error
* Squared Error

---

### 12. Mean Absolute Error

Given:

```text
Actual:     10, 20, 30
Predicted:  12, 18, 25
```

Calculate:

$$
MAE=
\frac1n\sum_{i=1}^{n}|y_i-\hat y_i|
$$

---

### 13. Euclidean Distance

Find the distance between:

$$
A=(2,3)
$$

and:

$$
B=(5,7)
$$

using:

$$
d=
\sqrt{
(x_2-x_1)^2+
(y_2-y_1)^2
}
$$

---

### 14. Weighted Mean

Calculate the weighted mean:

```text
Value     Weight
70           2
80           3
90           5
```

Use:

$$
\bar{x}_w=
\frac{\sum wx}{\sum w}
$$

---

### 15. Dataset Split

A dataset contains 50,000 observations.

If:

* 70% → Training
* 15% → Validation
* 15% → Testing

Calculate the number of observations in each set.

---

# 31. Quick Reference

| Concept            | Formula                          |              |   |
| ------------------ | -------------------------------- | ------------ | - |
| Addition           | \(a+b\)                          |              |   |
| Subtraction        | \(a-b\)                          |              |   |
| Multiplication     | \(ab\)                           |              |   |
| Division           | \(\frac ab\)                     |              |   |
| Percentage         | \(\frac{x}{100}\)                |              |   |
| Percentage Change  | \(\frac{New-Old}{Old}\times100\) |              |   |
| Mean               | \(\frac{\sum x_i}{n}\)           |              |   |
| Weighted Mean      | \(\frac{\sum w_ix_i}{\sum w_i}\) |              |   |
| Absolute Value     | (                                | x            | ) |
| Power              | \(a^n\)                          |              |   |
| Square Root        | \(\sqrt{x}\)                     |              |   |
| Euclidean Distance | \(\sqrt{\sum(x_i-y_i)^2}\)       |              |   |
| Absolute Error     | (                                | y-\hat y     | ) |
| Squared Error      | \((y-\hat y)^2\)                 |              |   |
| MSE                | \(\frac1n\sum(y_i-\hat y_i)^2\)  |              |   |
| MAE                | (\frac1n\sum                     | y_i-\hat y_i | ) |

---

# 32. Key Takeaways

Arithmetic is not merely elementary mathematics that must be completed before studying Machine Learning.

It is continuously present inside ML algorithms.

You should now understand:

* [x] Number systems
* [x] Natural numbers
* [x] Whole numbers
* [x] Integers
* [x] Rational numbers
* [x] Irrational numbers
* [x] Real numbers
* [x] Number lines
* [x] Arithmetic operations
* [x] Arithmetic properties
* [x] Operator precedence
* [x] Fractions
* [x] Decimals
* [x] Percentages
* [x] Ratios
* [x] Proportions
* [x] Powers
* [x] Exponents
* [x] Roots
* [x] Absolute values
* [x] Negative numbers
* [x] Rounding
* [x] Approximation
* [x] Scientific notation
* [x] Significant figures
* [x] Arithmetic mean
* [x] Weighted mean
* [x] Percentage change
* [x] Numerical precision
* [x] Floating-point arithmetic
* [x] ML-oriented arithmetic

---

# 33. What's Next?

Arithmetic provides the numerical foundation for the mathematical concepts that follow.

The recommended learning progression is:

```text
01-Arithmetic
      │
      ▼
02-Algebra
      │
      ▼
03-Functions
      │
      ▼
04-Geometry
      │
      ▼
05-Linear Algebra
      │
      ▼
06-Calculus
      │
      ▼
07-Probability
      │
      ▼
08-Statistics
      │
      ▼
09-Optimization
      │
      ▼
Machine Learning
```

The next topic, **Algebra**, will introduce variables, mathematical expressions, equations, inequalities, and transformations.

These concepts will allow us to move from calculating individual numbers to describing **relationships between quantities**.

---

# 🧠 Final Perspective

A Machine Learning model is ultimately a mathematical system.

At the lowest level, it performs numerical operations:

$$
+,\quad -,\quad \times,\quad \div
$$

These operations combine into:

$$
\text{Arithmetic}
$$

which leads to:

$$
\text{Algebra}
$$

then:

$$
\text{Linear Algebra}
$$

$$
\text{Calculus}
$$

$$
\text{Probability}
$$

$$
\text{Statistics}
$$

$$
\text{Optimization}
$$

and finally:

$$
\boxed{\text{Machine Learning}}
$$

Understanding these foundations makes advanced ML mathematics much easier to learn, implement, debug, and reason about.

> **Strong Machine Learning starts with strong mathematical foundations.**
