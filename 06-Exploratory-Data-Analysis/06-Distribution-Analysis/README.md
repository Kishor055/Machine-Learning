
# 📊 Distribution Analysis

> **Distribution Analysis = Understand Shape → Measure Spread → Detect Patterns → Identify Skewness → Find Outliers → Prepare Better Features**

![Machine Learning](https://img.shields.io/badge/Domain-Machine%20Learning-blue)
![EDA](https://img.shields.io/badge/Topic-Exploratory%20Data%20Analysis-orange)
![Python](https://img.shields.io/badge/Python-3.x-yellow)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-purple)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-darkgreen)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-teal)
![Status](https://img.shields.io/badge/Status-Learning%20Module-success)

---

# 🌱 GROW → EXPLORE → DISTRIBUTE → VISUALIZE → INTERPRET → TRANSFORM → BUILD 🚀

```text
                         📊 DATA
                            │
                            ▼
                  🔍 INSPECT VARIABLES
                            │
                            ▼
                  📈 DISTRIBUTION ANALYSIS
                            │
            ┌───────────────┼────────────────┐
            ▼               ▼                ▼
        NUMERICAL       CATEGORICAL       DATE/TIME
            │               │                │
            ▼               ▼                ▼
       HISTOGRAM        FREQUENCY        TIME PATTERNS
            │
            ▼
       DISTRIBUTION SHAPE
            │
      ┌─────┼─────┬─────────────┐
      ▼     ▼     ▼             ▼
    CENTER SPREAD SKEWNESS     KURTOSIS
      │     │     │             │
      └─────┴─────┴─────────────┘
                    │
                    ▼
             OUTLIER ANALYSIS
                    │
                    ▼
             TRANSFORMATION
                    │
                    ▼
          FEATURE ENGINEERING
                    │
                    ▼
             MACHINE LEARNING
```

---

# 📚 Table of Contents

1. [What Is Distribution Analysis?](#-what-is-distribution-analysis)
2. [Why Distribution Analysis Matters](#-why-distribution-analysis-matters)
3. [Distribution in Machine Learning](#-distribution-in-machine-learning)
4. [Types of Data Distributions](#-types-of-data-distributions)
5. [Numerical Distributions](#-numerical-distributions)
6. [Categorical Distributions](#-categorical-distributions)
7. [Discrete vs Continuous Distributions](#-discrete-vs-continuous-distributions)
8. [Frequency Distribution](#-frequency-distribution)
9. [Relative Frequency](#-relative-frequency)
10. [Probability Distribution](#-probability-distribution)
11. [PMF](#-probability-mass-function-pmf)
12. [PDF](#-probability-density-function-pdf)
13. [CDF](#-cumulative-distribution-function-cdf)
14. [Empirical CDF](#-empirical-cdf)
15. [Histogram](#-histogram)
16. [Choosing Histogram Bins](#-choosing-histogram-bins)
17. [Density Plot](#-density-plot)
18. [KDE](#-kernel-density-estimation)
19. [Box Plot](#-box-plot)
20. [Violin Plot](#-violin-plot)
21. [ECDF](#-ecdf)
22. [Normal Distribution](#-normal-distribution)
23. [Standard Normal Distribution](#-standard-normal-distribution)
24. [68–95–99.7 Rule](#-6895997-rule)
25. [Uniform Distribution](#-uniform-distribution)
26. [Binomial Distribution](#-binomial-distribution)
27. [Poisson Distribution](#-poisson-distribution)
28. [Exponential Distribution](#-exponential-distribution)
29. [Right-Skewed Distribution](#-right-skewed-distribution)
30. [Left-Skewed Distribution](#-left-skewed-distribution)
31. [Symmetric Distribution](#-symmetric-distribution)
32. [Skewness](#-skewness)
33. [Kurtosis](#-kurtosis)
34. [Mean, Median, and Mode](#-mean-median-and-mode)
35. [Outliers and Distributions](#-outliers-and-distributions)
36. [Distribution and Missing Values](#-distribution-and-missing-values)
37. [Distribution Comparison](#-distribution-comparison)
38. [Q-Q Plot](#-q-q-plot)
39. [Normality Testing](#-normality-testing)
40. [Log Transformation](#-log-transformation)
41. [Square-Root Transformation](#-square-root-transformation)
42. [Box-Cox Transformation](#-box-cox-transformation)
43. [Yeo-Johnson Transformation](#-yeo-johnson-transformation)
44. [Standardization vs Distribution Transformation](#-standardization-vs-distribution-transformation)
45. [Distribution Shift](#-distribution-shift)
46. [Train/Test Distribution](#-traintest-distribution)
47. [Data Leakage](#-data-leakage)
48. [Distribution Analysis with Python](#-distribution-analysis-with-python)
49. [Distribution Analysis with Pandas](#-distribution-analysis-with-pandas)
50. [Distribution Visualization](#-distribution-visualization)
51. [Reusable Python Functions](#-reusable-python-functions)
52. [Automated Distribution Report](#-automated-distribution-report)
53. [End-to-End Example](#-end-to-end-example)
54. [Common Mistakes](#-common-mistakes)
55. [Best Practices](#-best-practices)
56. [Mini Projects](#-mini-projects)
57. [Exercises](#-exercises)
58. [Project Structure](#-project-structure)
59. [Distribution Analysis Checklist](#-distribution-analysis-checklist)
60. [EDA Roadmap](#-eda-roadmap)
61. [Key Takeaways](#-key-takeaways)
62. [Next Step](#-next-step)

---

# 🔍 What Is Distribution Analysis?

**Distribution Analysis** is the process of studying how values of a variable are spread across a dataset.

It helps us understand:

* Where values are concentrated
* How widely values are spread
* Whether data is symmetric
* Whether data is skewed
* Whether extreme values exist
* Whether multiple groups or peaks exist
* Whether a variable approximately follows a known distribution
* Whether transformation may be useful

For example, consider employee salaries:

```text
Salary
│
│              ███
│              ███
│          ███████
│      ███████████
│  ███████████████
└────────────────────────
          Salary
```

The distribution gives us information that a simple average cannot provide.

---

# 🎯 Why Distribution Analysis Matters

Two datasets can have the same mean but completely different distributions.

```text
Dataset A:
10, 20, 30, 40, 50

Dataset B:
1, 2, 3, 4, 140
```

Both may produce similar summary statistics in some situations, but their distributions are very different.

Distribution analysis reveals:

```text
                 DATA
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
    CENTER       SPREAD       SHAPE
       │           │           │
      Mean        Range      Skewness
      Median      IQR        Kurtosis
      Mode        Std        Peaks
       │           │           │
       └───────────┼───────────┘
                   ▼
             DATA UNDERSTANDING
```

---

# 🤖 Distribution Analysis in Machine Learning

Distribution analysis is important before training models.

It can reveal:

### 1. Skewed Features

```text
Income
Sales
House Price
Population
Transaction Amount
```

These variables often have long tails.

### 2. Outliers

Extreme observations can affect some models and statistical techniques.

### 3. Different Feature Scales

```text
Age       → 18–80
Income    → 10,000–500,000
```

### 4. Transformation Opportunities

For example:

```text
Original
   ↓
Log Transformation
   ↓
Less Skewed Feature
```

### 5. Distribution Shift

Training and production data may follow different distributions.

---

# 📦 Types of Data Distributions

A distribution describes how observations occur across possible values.

Common examples include:

* Normal
* Uniform
* Binomial
* Poisson
* Exponential
* Bernoulli
* Log-normal
* Geometric
* Negative binomial

Not every real-world dataset follows a standard theoretical distribution.

---

# 🔢 Numerical Distributions

Numerical distributions describe the behavior of quantitative variables.

Examples:

```text
Age
Salary
Height
Weight
Temperature
House Price
Sales
```

Useful tools:

```text
Histogram
KDE
Box Plot
Violin Plot
ECDF
Q-Q Plot
```

---

# 🏷️ Categorical Distributions

Categorical distributions describe how frequently categories occur.

Example:

```python
import pandas as pd

df = pd.DataFrame({
    "department": [
        "IT",
        "HR",
        "IT",
        "Finance",
        "IT",
        "HR"
    ]
})

print(df["department"].value_counts())
```

Output:

```text
IT          3
HR          2
Finance     1
```

Percentage distribution:

```python
print(
    df["department"]
    .value_counts(normalize=True)
    .mul(100)
)
```

---

# 🔢 Discrete vs Continuous Distributions

## Discrete

A discrete variable takes countable values.

Examples:

```text
Number of children
Number of purchases
Number of defects
Number of calls
```

Example:

```text
0, 1, 2, 3, 4, ...
```

---

## Continuous

A continuous variable can take values within a range.

Examples:

```text
Height
Weight
Temperature
Time
Distance
```

For example:

```text
175.1
175.12
175.123
175.1234
```

---

# 📊 Frequency Distribution

A frequency distribution counts how many observations fall into each category or interval.

Example:

```python
import pandas as pd

scores = [45, 52, 52, 60, 65, 65, 65, 72, 80, 90]

df = pd.DataFrame({
    "score": scores
})

print(df["score"].value_counts().sort_index())
```

For continuous data, values are often grouped into bins.

```python
bins = [0, 20, 40, 60, 80, 100]

groups = pd.cut(
    df["score"],
    bins=bins,
    include_lowest=True
)

print(groups.value_counts().sort_index())
```

---

# 📈 Relative Frequency

Relative frequency tells us the proportion of observations in each group.

$$
Relative\ Frequency =
\frac{Frequency}{Total\ Observations}
$$

Example:

```python
frequency = df["score"].value_counts()

relative_frequency = (
    frequency / len(df)
)

print(relative_frequency)
```

Percentage:

```python
percentage = (
    df["score"]
    .value_counts(normalize=True)
    .mul(100)
)

print(percentage)
```

---

# 🎲 Probability Distribution

A probability distribution describes the probability associated with possible values of a random variable.

For a discrete variable:

```text
P(X = x)
```

For example:

```text
Number of heads in coin flips
```

For continuous variables, probability is represented using density rather than individual point probabilities.

---

# 🎯 Probability Mass Function — PMF

A **PMF** applies to discrete random variables.

Example:

```text
Number of defective products
Number of customers
Number of successes
```

The PMF gives:

$$
P(X=x)
$$

The probabilities satisfy:

$$
\sum_x P(X=x)=1
$$

---

# 📐 Probability Density Function — PDF

A **PDF** describes the density of a continuous random variable.

Important:

> The value of a PDF is not itself the probability of an exact continuous value.

Probability is obtained from the **area under the density curve over an interval**.

For a valid PDF:

$$
f(x) \geq 0
$$

and:

$$
\int_{-\infty}^{\infty}f(x)\,dx=1
$$

---

# 📈 Cumulative Distribution Function — CDF

The CDF gives the probability that a random variable is less than or equal to a value.

$$
F(x)=P(X\leq x)
$$

Example:

```text
F(50) = probability that X ≤ 50
```

The CDF always moves from:

```text
0 → 1
```

and never decreases.

---

# 📊 Empirical CDF

An empirical CDF is calculated directly from observed data.

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    10, 15, 20, 20, 25,
    30, 35, 40, 45, 50
])

x = np.sort(data)
y = np.arange(1, len(x) + 1) / len(x)

plt.step(x, y, where="post")

plt.xlabel("Value")
plt.ylabel("ECDF")
plt.title("Empirical CDF")

plt.tight_layout()
plt.show()
```

ECDFs are useful because they show the complete cumulative distribution without requiring histogram bins.

---

# 📊 Histogram

A histogram divides numerical data into intervals called **bins**.

```python
import matplotlib.pyplot as plt

plt.hist(
    data,
    bins=10
)

plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram")

plt.show()
```

A histogram helps identify:

* Center
* Spread
* Skewness
* Peaks
* Gaps
* Outliers
* Approximate distribution shape

---

# 🧮 Choosing Histogram Bins

The number of bins influences the visualization.

Too few bins:

```text
Under-smoothed
↓
Important structure may disappear
```

Too many bins:

```text
Over-detailed
↓
Random noise may dominate
```

Common approaches include:

### Sturges' Rule

$$
k=1+\log_2(n)
$$

### Square-Root Rule

$$
k\approx\sqrt{n}
$$

### Freedman-Diaconis Rule

$$
h =
2\frac{IQR(x)}{\sqrt[3]{n}}
$$

where `h` is the bin width.

In practice, experiment with reasonable bin choices and inspect the resulting visualization.

---

# 📈 Density Plot

A density plot provides a smooth representation of a numerical distribution.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.kdeplot(
    data=data,
    fill=True
)

plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Density Plot")

plt.show()
```

Density plots are useful for comparing distributions across groups.

---

# 🌊 Kernel Density Estimation — KDE

**Kernel Density Estimation** estimates the underlying probability density from observed data.

Conceptually:

```text
Observations
     ↓
Place smooth kernels
     ↓
Combine kernels
     ↓
Estimated density
```

A common KDE formulation is:

$$
\hat f_h(x)
=
\frac{1}{nh}
\sum_{i=1}^{n}
K\left(
\frac{x-x_i}{h}
\right)
$$

where:

* `n` = number of observations
* `h` = bandwidth
* `K` = kernel function
* `xi` = observed values

The bandwidth controls smoothness.

Too small:

```text
Very wiggly curve
```

Too large:

```text
Over-smoothed curve
```

---

# 📦 Box Plot

A box plot summarizes:

* Minimum/non-outlier lower value
* Q1
* Median
* Q3
* Maximum/non-outlier upper value
* Potential outliers

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(
    x=data
)

plt.title("Box Plot")
plt.show()
```

The interquartile range is:

$$
IQR=Q_3-Q_1
$$

A common outlier rule is:

$$
Lower=Q_1-1.5(IQR)
$$

$$
Upper=Q_3+1.5(IQR)
$$

Observations outside these boundaries are **potential outliers**, not automatically errors.

---

# 🎻 Violin Plot

A violin plot combines distribution information with a box-plot-style summary.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.violinplot(
    y=data
)

plt.title("Violin Plot")
plt.show()
```

Violin plots are particularly useful when comparing distributions across groups.

---

# 📈 ECDF

An **Empirical Cumulative Distribution Function** shows the proportion of observations less than or equal to each value.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.ecdfplot(
    x=data
)

plt.title("Empirical CDF")
plt.show()
```

ECDFs are useful for comparing:

```text
Group A
vs
Group B
```

without choosing histogram bins.

---

# 🔔 Normal Distribution

The normal distribution is one of the most important theoretical distributions in statistics.

It is characterized by:

* Bell-shaped curve
* Symmetry
* Mean = Median = Mode
* Defined by mean `μ`
* Defined by standard deviation `σ`

Probability density:

$$
f(x)=
\frac{1}{\sigma\sqrt{2\pi}}
e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

Shape:

```text
             ▲
           ███
         ███████
       ███████████
     ███████████████
   ███████████████████
──────────────────────────
             μ
```

---

# 📏 Standard Normal Distribution

A standard normal distribution has:

$$
\mu=0
$$

and:

$$
\sigma=1
$$

A value can be standardized using a z-score:

$$
z=\frac{x-\mu}{\sigma}
$$

Example:

```python
def z_score(x, mean, std):
    return (x - mean) / std


score = z_score(
    x=85,
    mean=70,
    std=10
)

print(score)
```

Output:

```text
1.5
```

This means the value is 1.5 standard deviations above the mean.

---

# 📐 68–95–99.7 Rule

For a normal distribution:

```text
Within ±1σ → approximately 68%
Within ±2σ → approximately 95%
Within ±3σ → approximately 99.7%
```

Visualization:

```text
              Normal Distribution

                    μ
                    │
          ┌─────────┼─────────┐
          │         │         │
        -1σ        +1σ
          │         │
        ~68%

      -2σ             +2σ
      └────── ~95% ──────┘

    -3σ                     +3σ
    └──────── ~99.7% ────────┘
```

This rule applies specifically to normal distributions.

---

# 🟦 Uniform Distribution

In a continuous uniform distribution, values within an interval have equal density.

Example:

```text
       ┌───────────────┐
       │               │
       │               │
───────┘               └───────
       a               b
```

A simple simulation:

```python
import numpy as np

samples = np.random.default_rng(42).uniform(
    low=0,
    high=10,
    size=1000
)
```

---

# 🎯 Binomial Distribution

The binomial distribution models the number of successes in a fixed number of independent trials.

Conditions include:

* Fixed number of trials
* Two possible outcomes
* Constant probability of success
* Independent trials

Example:

```python
import numpy as np

rng = np.random.default_rng(42)

samples = rng.binomial(
    n=10,
    p=0.5,
    size=1000
)
```

Examples:

```text
Number of successful sales
Number of defective products
Number of heads
```

---

# ☎️ Poisson Distribution

The Poisson distribution models counts of events occurring within a fixed interval when a rate parameter is appropriate.

Examples:

```text
Calls per minute
Customers per hour
Defects per meter
Requests per second
```

Simulation:

```python
import numpy as np

rng = np.random.default_rng(42)

samples = rng.poisson(
    lam=5,
    size=1000
)
```

---

# ⏱️ Exponential Distribution

The exponential distribution is commonly used to model waiting times between events in a Poisson-process setting.

Example:

```python
import numpy as np

rng = np.random.default_rng(42)

samples = rng.exponential(
    scale=2,
    size=1000
)
```

Applications can include:

```text
Waiting time
Time between events
Reliability analysis
Service systems
```

---

# ↗️ Right-Skewed Distribution

A right-skewed distribution has a longer tail toward larger values.

```text
Frequency
│
│ ███████
│ █████████
│ ████████
│ █████
│ ███
│ ██
│ █
└────────────────────────►
                    Long right tail
```

Often:

```text
Mean > Median
```

Examples can include:

* Income
* House prices
* Transaction amounts
* Population sizes

---

# ↙️ Left-Skewed Distribution

A left-skewed distribution has a longer tail toward smaller values.

```text
Frequency
│              ███████
│            █████████
│           ██████████
│             █████
│               ██
│                █
└────────────────────────►
 Long left tail
```

Often:

```text
Mean < Median
```

---

# ⚖️ Symmetric Distribution

A symmetric distribution has approximately equal shape on both sides of its center.

```text
Frequency
│
│          ███
│        ███████
│      ███████████
│    ███████████████
└────────────────────────►
             Center
```

A normal distribution is an important example.

---

# 📐 Skewness

Skewness measures asymmetry in a distribution.

Conceptually:

```text
Negative Skew     Symmetric      Positive Skew

     ███             ███              ███
   ███████         ███████          ███████
 █████████       █████████        ███████
───────────     ───────────      ───────────────►
```

Pandas:

```python
skewness = df["income"].skew()

print(skewness)
```

General interpretation:

```text
Skewness < 0  → Left-skewed
Skewness ≈ 0  → Approximately symmetric
Skewness > 0  → Right-skewed
```

The exact magnitude should be interpreted in context.

---

# 📊 Kurtosis

Kurtosis describes aspects of the tails and extremity of a distribution.

Pandas uses **Fisher's definition** by default:

```python
kurtosis = df["income"].kurt()

print(kurtosis)
```

Under this convention:

```text
Normal distribution → approximately 0
```

Higher positive values generally indicate heavier tails relative to a normal reference, while negative values indicate lighter tails.

Do not interpret kurtosis simply as "peak height"; tail behavior is important.

---

# 📍 Mean, Median, and Mode

Distribution shape influences the relationship between measures of center.

## Symmetric Distribution

```text
Mean ≈ Median ≈ Mode
```

## Right-Skewed Distribution

Often:

```text
Mode < Median < Mean
```

## Left-Skewed Distribution

Often:

```text
Mean < Median < Mode
```

These are useful rules of thumb, not universal laws.

---

# 🚨 Outliers and Distributions

Outliers can influence:

* Mean
* Standard deviation
* Skewness
* Kurtosis
* Correlation
* Model performance

Example:

```python
import pandas as pd

df = pd.DataFrame({
    "salary": [
        30000,
        32000,
        35000,
        36000,
        38000,
        40000,
        500000
    ]
})

print("Mean:", df["salary"].mean())
print("Median:", df["salary"].median())
print("Skewness:", df["salary"].skew())
```

The extreme salary can pull the mean upward and create strong right skewness.

---

# 🕳️ Distribution and Missing Values

Missing values can alter the observed distribution.

Always inspect:

```python
missing = df["salary"].isna().sum()

print("Missing:", missing)
print(
    "Missing %:",
    df["salary"].isna().mean() * 100
)
```

Before analyzing a distribution, ask:

```text
Are missing values present?
        ↓
Why are they missing?
        ↓
Could missingness bias the distribution?
```

Do not automatically impute values before understanding the missingness mechanism.

---

# ⚖️ Distribution Comparison

Comparing distributions between groups can reveal important patterns.

Example:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(
    data=df,
    x="salary",
    hue="department",
    kde=True
)

plt.title("Salary Distribution by Department")
plt.tight_layout()
plt.show()
```

You can compare:

* Center
* Spread
* Skewness
* Outliers
* Multiple modes
* Overlap

---

# 📐 Q-Q Plot

A **Q-Q plot** compares observed quantiles against theoretical quantiles.

It is commonly used to assess whether data approximately follows a specified distribution, often the normal distribution.

```python
import matplotlib.pyplot as plt
from scipy import stats

stats.probplot(
    df["salary"].dropna(),
    dist="norm",
    plot=plt
)

plt.title("Normal Q-Q Plot")
plt.tight_layout()
plt.show()
```

Interpretation:

```text
Points close to a straight line
        ↓
Approximately consistent with
the reference distribution
```

Large systematic deviations may indicate:

* Skewness
* Heavy tails
* Light tails
* Other distributional differences

A Q-Q plot is a diagnostic, not absolute proof of normality.

---

# 🧪 Normality Testing

Several statistical tests can assess normality.

One common test is the Shapiro-Wilk test.

```python
from scipy.stats import shapiro

sample = df["salary"].dropna()

statistic, p_value = shapiro(sample)

print("Statistic:", statistic)
print("p-value:", p_value)
```

A common interpretation framework is:

```text
H₀: Data follows the specified normality assumption

Small p-value
    ↓
Evidence against normality
```

However, statistical tests can become extremely sensitive with large datasets.

Therefore combine:

```text
Histogram
+
Q-Q Plot
+
Skewness
+
Domain Knowledge
+
Statistical Test
```

---

# 🔄 Log Transformation

Log transformation can reduce right skewness in positive-valued variables.

A common safe form is:

$$
x'=\log(1+x)
$$

Python:

```python
import numpy as np

df["log_income"] = np.log1p(
    df["income"]
)
```

Compare:

```python
print(
    "Original skew:",
    df["income"].skew()
)

print(
    "Transformed skew:",
    df["log_income"].skew()
)
```

Log transformation is often useful for variables such as:

```text
Income
Sales
Population
Transaction Amount
House Price
```

Do not apply it blindly. The transformation should make sense for the variable and modeling objective.

---

# √ Square-Root Transformation

Square-root transformation is another option, particularly for non-negative count-like data.

$$
x'=\sqrt{x}
$$

Example:

```python
import numpy as np

df["sqrt_count"] = np.sqrt(
    df["count"]
)
```

---

# 📦 Box-Cox Transformation

Box-Cox transformation can find a power transformation parameter for **strictly positive** data.

Using SciPy:

```python
from scipy.stats import boxcox

positive_values = df["income"].dropna()

transformed, lambda_value = boxcox(
    positive_values
)

print("Lambda:", lambda_value)
```

Important:

> Box-Cox requires strictly positive values.

For data containing zero or negative values, consider another approach such as Yeo-Johnson.

---

# 🔄 Yeo-Johnson Transformation

Yeo-Johnson can handle zero and negative values.

Using scikit-learn:

```python
from sklearn.preprocessing import PowerTransformer

transformer = PowerTransformer(
    method="yeo-johnson"
)

df["transformed_income"] = (
    transformer.fit_transform(
        df[["income"]]
    )
)
```

For machine learning, fit the transformer on training data and apply it to validation/test data to avoid leakage.

---

# ⚖️ Standardization vs Distribution Transformation

These are different concepts.

## Standardization

Transforms scale:

$$
z=\frac{x-\mu}{\sigma}
$$

Typical result:

```text
Mean ≈ 0
Standard deviation ≈ 1
```

But standardization does **not necessarily make a skewed distribution normal**.

---

## Distribution Transformation

Examples:

```text
Log
Square Root
Box-Cox
Yeo-Johnson
```

These aim to modify distribution shape.

Therefore:

```text
Scaling ≠ Normalization of Shape
```

---

# 🔄 Distribution Shift

**Distribution shift** occurs when the distribution of data changes between environments.

Example:

```text
Training Data
     ↓
Income Distribution
     │
     │
     ▼
Production Data
     ↓
Different Income Distribution
```

Possible causes:

* Economic changes
* User behavior changes
* Seasonal changes
* Sensor changes
* Data collection changes
* Population changes

Monitoring distributions is important in production ML systems.

---

# 🧪 Train/Test Distribution

Before modeling, compare important feature distributions.

```python
import matplotlib.pyplot as plt

plt.hist(
    X_train["income"],
    bins=30,
    alpha=0.5,
    label="Train"
)

plt.hist(
    X_test["income"],
    bins=30,
    alpha=0.5,
    label="Test"
)

plt.legend()
plt.title("Train vs Test Distribution")
plt.show()
```

Significant differences may indicate:

* Sampling problems
* Distribution shift
* Data leakage
* Temporal changes
* Small sample effects

---

# 🔐 Data Leakage

Never use test data to determine transformation parameters.

Incorrect:

```text
Full Dataset
    ↓
Fit Transformation
    ↓
Train/Test Split
```

Preferred:

```text
Dataset
   ↓
Train/Test Split
   ↓
Fit Transformation on Train
   ↓
Transform Train
   ↓
Transform Test
```

Example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)
```

The same principle applies to:

* Power transformations
* Imputation
* Feature selection
* Scaling
* Encoding
* Dimensionality reduction

---

# 🐍 Distribution Analysis with Python

A simple reusable workflow:

```python
import numpy as np
import pandas as pd


def analyze_distribution(series):
    clean = series.dropna()

    return {
        "count": clean.count(),
        "mean": clean.mean(),
        "median": clean.median(),
        "std": clean.std(),
        "min": clean.min(),
        "max": clean.max(),
        "skewness": clean.skew(),
        "kurtosis": clean.kurt(),
        "q1": clean.quantile(0.25),
        "q3": clean.quantile(0.75),
        "iqr": (
            clean.quantile(0.75)
            - clean.quantile(0.25)
        )
    }
```

Usage:

```python
report = analyze_distribution(
    df["income"]
)

for key, value in report.items():
    print(f"{key}: {value}")
```

---

# 🐼 Distribution Analysis with Pandas

Start with:

```python
print(df["income"].describe())
```

Output includes:

```text
count
mean
std
min
25%
50%
75%
max
```

Additional statistics:

```python
print("Skewness:", df["income"].skew())
print("Kurtosis:", df["income"].kurt())
```

Quantiles:

```python
print(
    df["income"].quantile(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )
)
```

---

# 📊 Distribution Visualization

A strong distribution analysis should use multiple visualizations.

## Histogram

```python
sns.histplot(
    data=df,
    x="income",
    kde=True
)

plt.show()
```

## Box Plot

```python
sns.boxplot(
    x=df["income"]
)

plt.show()
```

## ECDF

```python
sns.ecdfplot(
    x=df["income"]
)

plt.show()
```

## Q-Q Plot

```python
from scipy import stats

stats.probplot(
    df["income"].dropna(),
    dist="norm",
    plot=plt
)

plt.show()
```

A practical workflow is:

```text
Histogram
    +
Box Plot
    +
ECDF
    +
Q-Q Plot
    +
Numerical Statistics
```

---

# 🧩 Reusable Python Functions

## Distribution Summary

```python
def distribution_summary(
    df,
    column
):
    series = df[column].dropna()

    return {
        "count": len(series),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "variance": series.var(),
        "min": series.min(),
        "q1": series.quantile(0.25),
        "q3": series.quantile(0.75),
        "max": series.max(),
        "skewness": series.skew(),
        "kurtosis": series.kurt()
    }
```

---

## Detect Potential Outliers

```python
def detect_iqr_outliers(series):
    clean = series.dropna()

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return clean[
        (clean < lower) |
        (clean > upper)
    ]
```

Usage:

```python
outliers = detect_iqr_outliers(
    df["income"]
)

print(outliers)
```

---

# 📋 Automated Distribution Report

```python
import pandas as pd


def create_distribution_report(df):
    numerical = df.select_dtypes(
        include="number"
    )

    rows = []

    for column in numerical.columns:
        series = numerical[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_count = (
            (series < lower) |
            (series > upper)
        ).sum()

        rows.append({
            "feature": column,
            "count": series.count(),
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "q1": q1,
            "q3": q3,
            "max": series.max(),
            "skewness": series.skew(),
            "kurtosis": series.kurt(),
            "outlier_count": outlier_count
        })

    return pd.DataFrame(rows)
```

Usage:

```python
report = create_distribution_report(df)

print(report)
```

---

# 🚀 End-to-End Example

The following example demonstrates a practical distribution-analysis workflow.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_dataset():
    rng = np.random.default_rng(42)

    income = rng.lognormal(
        mean=10,
        sigma=0.45,
        size=500
    )

    age = rng.normal(
        loc=35,
        scale=10,
        size=500
    )

    age = np.clip(
        age,
        18,
        80
    )

    department = rng.choice(
        ["IT", "HR", "Finance"],
        size=500,
        p=[0.5, 0.2, 0.3]
    )

    return pd.DataFrame({
        "income": income,
        "age": age,
        "department": department
    })


def print_summary(df):
    print("\nNumerical Summary")
    print("=" * 50)

    print(
        df[
            ["income", "age"]
        ].describe()
    )

    print("\nSkewness")
    print(
        df[
            ["income", "age"]
        ].skew()
    )

    print("\nKurtosis")
    print(
        df[
            ["income", "age"]
        ].kurt()
    )


def plot_income_distribution(df):
    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=df,
        x="income",
        kde=True
    )

    plt.title(
        "Income Distribution"
    )

    plt.xlabel("Income")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_boxplot(df):
    plt.figure(figsize=(9, 4))

    sns.boxplot(
        x=df["income"]
    )

    plt.title(
        "Income Box Plot"
    )

    plt.tight_layout()
    plt.show()


def plot_ecdf(df):
    plt.figure(figsize=(9, 5))

    sns.ecdfplot(
        data=df,
        x="income"
    )

    plt.title(
        "Income ECDF"
    )

    plt.tight_layout()
    plt.show()


def plot_department_distributions(df):
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="income",
        hue="department",
        kde=True,
        element="step",
        stat="density",
        common_norm=False
    )

    plt.title(
        "Income Distribution by Department"
    )

    plt.tight_layout()
    plt.show()


def main():
    df = create_dataset()

    print("Dataset:")
    print(df.head())

    print_summary(df)

    plot_income_distribution(df)
    plot_boxplot(df)
    plot_ecdf(df)
    plot_department_distributions(df)


if __name__ == "__main__":
    main()
```

---

# 🧠 Distribution Interpretation Framework

When analyzing a numerical feature, ask:

```text
                    VARIABLE
                       │
                       ▼
                 Missing Values?
                       │
                       ▼
                Central Tendency
                       │
                       ▼
                     Spread
                       │
                       ▼
                     Shape
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Symmetric      Skewed      Multimodal
          │            │            │
          ▼            ▼            ▼
       Normal?     Transformation?  Groups?
          │            │            │
          └────────────┼────────────┘
                       ▼
                    Outliers
                       │
                       ▼
                 Visualization
                       │
                       ▼
               Modeling Decision
```

---

# 🔬 Distribution Analysis Decision Guide

```text
START
  │
  ▼
Is the variable numerical?
  │
  ├── NO ──► Frequency / Proportion Analysis
  │
  ▼
 YES
  │
  ▼
Inspect Missing Values
  │
  ▼
Create Histogram
  │
  ▼
Create Box Plot
  │
  ▼
Check Shape
  │
  ├── Symmetric
  │
  ├── Right-Skewed
  │
  ├── Left-Skewed
  │
  └── Multimodal
  │
  ▼
Check Outliers
  │
  ▼
Check Q-Q Plot if Relevant
  │
  ▼
Consider Transformation
  │
  ▼
Validate on Training Data
  │
  ▼
Machine Learning
```

---

# ❌ Common Mistakes

## 1. Assuming Every Variable Should Be Normally Distributed

Machine learning does not universally require normally distributed features.

Model assumptions differ by algorithm.

---

## 2. Using Too Many Histogram Bins

Excessive bins can make random noise appear meaningful.

---

## 3. Using Too Few Bins

Too few bins can hide important structure.

---

## 4. Treating Outliers as Automatically Wrong

An outlier may be:

* A valid observation
* A rare event
* A measurement error
* A data-entry error

Investigate before removing it.

---

## 5. Automatically Applying Log Transformation

A transformation should have a purpose.

Ask:

```text
Why am I transforming this variable?
```

---

## 6. Confusing Scaling with Normalization of Shape

Standardization changes scale.

It does not necessarily make the data normally distributed.

---

## 7. Ignoring Multiple Modes

A bimodal distribution can indicate:

```text
Different populations
Different customer segments
Different processes
Different subgroups
```

---

## 8. Ignoring Missing Values

Missingness can distort the observed distribution.

---

## 9. Fitting Transformations on the Test Set

This can cause data leakage.

Fit preprocessing using training data only.

---

## 10. Relying on One Visualization

Use several complementary tools:

```text
Histogram
+
Box Plot
+
ECDF
+
Q-Q Plot
+
Numerical Statistics
```

---

# ✅ Best Practices

### 1. Start with Data Understanding

Know what the variable represents.

### 2. Inspect Missing Values

Understand why values are missing.

### 3. Use Multiple Visualizations

No single plot tells the complete story.

### 4. Check Both Center and Shape

Mean and median alone are insufficient.

### 5. Investigate Skewness

Strong skewness may affect some statistical procedures and models.

### 6. Investigate Outliers

Determine whether they are errors or legitimate observations.

### 7. Compare Groups

A global distribution may hide important subgroup differences.

### 8. Use Appropriate Transformations

Choose transformations based on the data and modeling objective.

### 9. Avoid Data Leakage

Fit transformations on training data.

### 10. Validate After Transformation

Check whether the transformation actually improved the intended property.

---

# 🧪 Mini Projects

## 🏠 Project 1 — House Price Distribution

Analyze:

```text
Price
Area
Bedrooms
Bathrooms
Age
```

Tasks:

* Create histograms.
* Compare mean and median.
* Calculate skewness.
* Detect outliers.
* Try a log transformation.
* Compare before and after distributions.

---

## 💰 Project 2 — Income Distribution

Analyze:

```text
Salary
Age
Experience
Education
Department
```

Questions:

* Is salary normally distributed?
* Is salary right-skewed?
* Which department has the highest spread?
* Does log transformation reduce skewness?

---

## 🛒 Project 3 — E-Commerce Transactions

Analyze:

```text
Order Value
Quantity
Discount
Shipping Time
Customer Rating
```

Tasks:

* Analyze numerical distributions.
* Detect extreme transactions.
* Compare customer segments.
* Analyze order-value skewness.

---

## 🎓 Project 4 — Student Performance

Analyze:

```text
Study Hours
Attendance
Assignment Score
Exam Score
Sleep Hours
```

Questions:

* Which variables are approximately normal?
* Which variables are skewed?
* Are there unusual observations?
* Which features require transformation?

---

# 📝 Exercises

## 🌱 Beginner

1. Define a probability distribution.
2. Explain the difference between discrete and continuous data.
3. Create a histogram.
4. Create a box plot.
5. Calculate mean, median, and mode.
6. Calculate skewness.
7. Explain the 68–95–99.7 rule.
8. Identify a right-skewed distribution.

---

## 🚀 Intermediate

1. Compare histogram and KDE.
2. Build an ECDF.
3. Create a Q-Q plot.
4. Compare two group distributions.
5. Detect IQR-based outliers.
6. Calculate kurtosis.
7. Compare mean and median for skewed data.
8. Apply log transformation.
9. Compare the distribution before and after transformation.
10. Analyze a real-world dataset.

---

## 🧠 Advanced

1. Implement a histogram without Pandas plotting.
2. Implement an empirical CDF manually.
3. Implement an IQR-based outlier detector.
4. Compare multiple theoretical distributions.
5. Study bandwidth selection for KDE.
6. Analyze distribution shift between datasets.
7. Compare train/test distributions.
8. Build an automated distribution report.
9. Apply Yeo-Johnson transformation using a pipeline.
10. Investigate distribution drift in production data.

---

# 📁 Project Structure

```text
06-Exploratory-Data-Analysis/
│
├── README.md
│
├── 01-Univariate-Analysis/
│   └── README.md
│
├── 02-Bivariate-Analysis/
│   └── README.md
│
├── 03-Multivariate-Analysis/
│   └── README.md
│
├── 04-Descriptive-Statistics/
│   └── README.md
│
├── 05-Correlation-Analysis/
│   └── README.md
│
├── 06-Distribution-Analysis/
│   ├── README.md
│   ├── histogram.py
│   ├── frequency-distribution.py
│   ├── kde.py
│   ├── boxplot.py
│   ├── ecdf.py
│   ├── qq-plot.py
│   ├── skewness.py
│   ├── kurtosis.py
│   ├── distribution-transformations.py
│   └── distribution-report.py
│
└── 07-Outlier-Analysis/
    └── README.md
```

---

# 🔎 Distribution Analysis Checklist

Before completing your analysis:

```text
☐ Identify variable type
☐ Inspect missing values
☐ Calculate descriptive statistics
☐ Calculate quantiles
☐ Plot histogram
☐ Inspect density/KDE
☐ Create box plot
☐ Check ECDF
☐ Inspect skewness
☐ Inspect kurtosis
☐ Check for outliers
☐ Investigate multiple modes
☐ Compare relevant groups
☐ Use Q-Q plot when appropriate
☐ Consider distribution transformations
☐ Compare before/after transformation
☐ Check train/test distributions
☐ Avoid data leakage
☐ Document decisions
```

---

# 🗺️ EDA Roadmap

```text
06-Exploratory-Data-Analysis/
│
├── 01-Univariate-Analysis
│       ↓
├── 02-Bivariate-Analysis
│       ↓
├── 03-Multivariate-Analysis
│       ↓
├── 04-Descriptive-Statistics
│       ↓
├── 05-Correlation-Analysis
│       ↓
├── 06-Distribution-Analysis
│       ↓
├── 07-Outlier-Analysis
│       ↓
├── 08-Data-Visualization
│       ↓
└── 09-EDA-Projects
```

---

# 🧠 Key Takeaways

```text
📌 A distribution describes how observations are spread.

📌 Histograms show frequency across intervals.

📌 KDE provides a smooth density estimate.

📌 ECDF shows cumulative proportions directly from observed data.

📌 Box plots summarize center, spread, and potential outliers.

📌 Normal distributions are symmetric and characterized by μ and σ.

📌 Skewness describes asymmetry.

📌 Kurtosis provides information about tail behavior relative to a reference convention.

📌 Mean, median, and mode can behave differently under skewness.

📌 Outliers can substantially affect statistical summaries.

📌 Not every feature needs to be normally distributed.

📌 Log, Box-Cox, and Yeo-Johnson transformations can modify distribution shape.

📌 Standardization changes scale, not necessarily distribution shape.

📌 Train/test distribution comparison can reveal sampling problems and distribution shift.

📌 Transformations must be fitted using training data to avoid leakage.

📌 Distribution analysis should combine statistics, visualization, and domain knowledge.
```

---

# 🚀 Next Step

After understanding how variables are distributed, the next step is to investigate unusual observations.

Continue to:

```text
06-Exploratory-Data-Analysis/
└── 07-Outlier-Analysis/
    └── README.md
```

You will explore:

```text
OUTLIERS
   ↓
Why Outliers Occur
   ↓
IQR Method
   ↓
Z-Score Method
   ↓
Modified Z-Score
   ↓
Percentile Method
   ↓
Isolation Forest
   ↓
Local Outlier Factor
   ↓
Visualization
   ↓
Treatment Strategies
   ↓
Machine Learning
```

---

# 🌟 Final Thought

> **A dataset is more than its average. Its distribution tells you where the data lives, how it spreads, how it behaves, and where the unusual observations are hiding.**

A strong distribution analysis combines:

```text
📊 Descriptive Statistics
        +
📈 Visualization
        +
📐 Statistical Concepts
        +
🔍 Outlier Detection
        +
🔄 Transformation
        +
🧠 Domain Knowledge
        +
🤖 Machine Learning
```

That is how raw numerical columns become meaningful information for EDA and machine learning.

---

# 👨‍💻 Author

**Kishor Patil**

Machine Learning | Python | Data Science | AI

🔗 GitHub: [Kishor055](https://github.com/Kishor055)

---

# 🤝 Contributing

Contributions are welcome!

If you find:

* 🐛 A mistake
* 💡 An unclear explanation
* 🧩 A missing concept
* 🐍 A Python improvement
* 📊 A better visualization
* 📝 A useful exercise

feel free to open an issue or submit a pull request.

### Contribution Workflow

```bash
git clone https://github.com/Kishor055/Machine-Learning.git

cd Machine-Learning

git checkout -b feature/distribution-analysis

git add .

git commit -m "Improve distribution analysis module"

git push origin feature/distribution-analysis
```

Then open a Pull Request.

---

# ⭐ Support

If this repository helps you learn Machine Learning:

* ⭐ Star the repository
* 🍴 Fork the repository
* 🐛 Report issues
* 💡 Suggest improvements
* 🤝 Contribute examples
* 📢 Share it with other learners

---

# 📄 License

This project is intended for **educational and learning purposes**.

See the repository license for complete terms.

---

<div align="center">

### 🌱 Learn → Analyze → Visualize → Experiment → Build → Improve 🚀

**Keep learning. Keep experimenting. Keep building.**

</div>
