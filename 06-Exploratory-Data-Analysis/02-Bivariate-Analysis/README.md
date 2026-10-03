# 📈 Bivariate Analysis

> **Bivariate Analysis = Understanding the Relationship Between Two Variables**

Bivariate analysis is the second major stage of **Exploratory Data Analysis (EDA)**. While univariate analysis studies one variable at a time, bivariate analysis focuses on **two variables together** to understand relationships, associations, dependencies, differences, and trends.

```text
                    🌱 EDA
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
   UNIVARIATE     BIVARIATE    MULTIVARIATE
     Analysis      Analysis      Analysis
        │            │            │
        ↓            ↓            ↓
   One Variable   Two Variables  3+ Variables
```

# 🌱 GROW → COMPARE → CONNECT → VISUALIZE → INTERPRET → DISCOVER → BUILD 🚀

Bivariate analysis helps answer questions such as:

* Does study time affect exam scores?
* Is salary related to experience?
* Does price increase with house size?
* Is there a relationship between age and spending?
* Do different categories have different distributions?
* Are two numerical variables correlated?
* Is one categorical variable associated with another?
* Does one variable appear to influence another?

---

## 📚 Table of Contents

1. [What Is Bivariate Analysis?](#-what-is-bivariate-analysis)
2. [Why Bivariate Analysis Matters](#-why-bivariate-analysis-matters)
3. [Univariate vs Bivariate vs Multivariate](#-univariate-vs-bivariate-vs-multivariate)
4. [Types of Bivariate Relationships](#-types-of-bivariate-relationships)
5. [Types of Variable Combinations](#-types-of-variable-combinations)
6. [Numerical + Numerical](#-numerical--numerical)
7. [Categorical + Numerical](#-categorical--numerical)
8. [Categorical + Categorical](#-categorical--categorical)
9. [Date-Time + Numerical](#-date-time--numerical)
10. [Scatter Plot](#-scatter-plot)
11. [Line Plot](#-line-plot)
12. [Correlation](#-correlation)
13. [Covariance](#-covariance)
14. [Pearson Correlation](#-pearson-correlation)
15. [Spearman Correlation](#-spearman-correlation)
16. [Correlation Matrix](#-correlation-matrix)
17. [Heatmap](#-heatmap)
18. [Regression Line](#-regression-line)
19. [Grouped Statistics](#-grouped-statistics)
20. [Box Plot](#-box-plot)
21. [Violin Plot](#-violin-plot)
22. [Bar Plot](#-bar-plot)
23. [Count Plot](#-count-plot)
24. [Contingency Tables](#-contingency-tables)
25. [Cross Tabulation](#-cross-tabulation)
26. [Chi-Square Test](#-chi-square-test)
27. [Numerical Relationship Analysis](#-numerical-relationship-analysis)
28. [Categorical Relationship Analysis](#-categorical-relationship-analysis)
29. [Outliers in Bivariate Analysis](#-outliers-in-bivariate-analysis)
30. [Missing Values](#-missing-values)
31. [Data Transformation](#-data-transformation)
32. [Non-Linear Relationships](#-non-linear-relationships)
33. [Correlation Does Not Mean Causation](#-correlation-does-not-mean-causation)
34. [Pandas Implementation](#-pandas-implementation)
35. [Matplotlib Implementation](#-matplotlib-implementation)
36. [Seaborn Implementation](#-seaborn-implementation)
37. [Reusable Functions](#-reusable-functions)
38. [Automated Bivariate Analysis](#-automated-bivariate-analysis)
39. [Feature Selection](#-feature-selection)
40. [Machine Learning Applications](#-machine-learning-applications)
41. [Common Mistakes](#-common-mistakes)
42. [Best Practices](#-best-practices)
43. [Mini Projects](#-mini-projects)
44. [Exercises](#-exercises)
45. [Project Structure](#-project-structure)
46. [End-to-End Example](#-end-to-end-example)
47. [Bivariate Analysis Checklist](#-bivariate-analysis-checklist)
48. [Decision Guide](#-decision-guide)
49. [EDA Roadmap](#-eda-roadmap)
50. [Key Takeaways](#-key-takeaways)
51. [Next Step](#-next-step)
52. [Author](#-author)
53. [Contributing](#-contributing)
54. [Support](#-support)
55. [License](#-license)

---

# 🔎 What Is Bivariate Analysis?

**Bivariate analysis** is the process of analyzing **two variables simultaneously** to understand their relationship.

For example:

```text
Experience ───────────────→ Salary
     X                         Y
```

We may want to determine whether employees with more experience tend to earn higher salaries.

Another example:

```text
House Size ───────────────→ House Price
    X                           Y
```

The goal is not simply to calculate a number.

The goal is to understand:

```text
Relationship
     ↓
Direction
     ↓
Strength
     ↓
Pattern
     ↓
Outliers
     ↓
Possible Explanation
```

---

# 🎯 Why Bivariate Analysis Matters

Bivariate analysis helps us discover relationships that are invisible when variables are examined independently.

For example:

```text
Univariate:

Age
25
31
42
29
35

Salary
40K
50K
65K
45K
70K
```

We can understand the distributions separately.

But:

```text
Age ─────────────── Salary
25                   40K
31                   50K
42                   65K
29                   45K
35                   70K
```

allows us to ask:

> Does salary tend to increase as age increases?

This is a bivariate question.

---

# 🔄 Univariate vs Bivariate vs Multivariate

| Analysis     | Variables | Main Goal                                        |
| ------------ | --------: | ------------------------------------------------ |
| Univariate   |         1 | Understand one variable                          |
| Bivariate    |         2 | Understand relationship between two variables    |
| Multivariate |        3+ | Understand interactions among multiple variables |

Example:

```text
Univariate
    ↓
Analyze Salary

Bivariate
    ↓
Experience ↔ Salary

Multivariate
    ↓
Experience + Education + Age → Salary
```

---

# 🔗 Types of Bivariate Relationships

Two variables can have different types of relationships.

## 1. Positive Relationship

As X increases, Y tends to increase.

```text
X ↑
Y ↑
```

Example:

```text
Experience ↑ → Salary ↑
```

---

## 2. Negative Relationship

As X increases, Y tends to decrease.

```text
X ↑
Y ↓
```

Example:

```text
Price ↑ → Demand ↓
```

---

## 3. No Clear Relationship

Changes in X do not show a clear pattern in Y.

```text
X ↑
Y ↔
```

---

## 4. Non-Linear Relationship

The relationship exists but does not follow a straight line.

Example:

```text
Study Time → Performance

Low study time:
Performance increases

Very high study time:
Performance may plateau
```

---

# 🧩 Types of Variable Combinations

The appropriate bivariate technique depends on the types of variables.

| Variable 1  | Variable 2  | Common Techniques                         |
| ----------- | ----------- | ----------------------------------------- |
| Numerical   | Numerical   | Scatter plot, correlation                 |
| Categorical | Numerical   | Box plot, violin plot, grouped statistics |
| Categorical | Categorical | Cross-tabulation, chi-square              |
| Date-Time   | Numerical   | Line plot, trend analysis                 |
| Date-Time   | Categorical | Time-based category counts                |

---

# 🔢 Numerical + Numerical

Examples:

```text
Age ↔ Salary
Height ↔ Weight
Experience ↔ Salary
Advertising Spend ↔ Sales
Study Hours ↔ Exam Score
```

Useful techniques:

* Scatter plot
* Correlation
* Covariance
* Regression line
* Pearson correlation
* Spearman correlation

---

# 🏷️ Categorical + Numerical

Examples:

```text
Department ↔ Salary
Education Level ↔ Income
City ↔ House Price
Product Category ↔ Sales
Gender ↔ Spending
```

Useful techniques:

* Grouped mean
* Grouped median
* Box plot
* Violin plot
* Bar plot
* Strip plot

---

# 🏷️ Categorical + Categorical

Examples:

```text
Gender ↔ Purchased
Education ↔ Promotion
Department ↔ Attrition
Payment Method ↔ Product Category
```

Useful techniques:

* Cross-tabulation
* Contingency tables
* Stacked bar charts
* Count plots
* Chi-square test

---

# 📅 Date-Time + Numerical

Examples:

```text
Date ↔ Sales
Date ↔ Temperature
Month ↔ Revenue
Time ↔ Website Traffic
```

Useful techniques:

* Line plot
* Rolling statistics
* Trend analysis
* Seasonal comparison

---

# 📊 Scatter Plot

A scatter plot is one of the most important tools for numerical-numerical bivariate analysis.

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "experience": [1, 2, 3, 4, 5, 6, 7],
    "salary": [35, 40, 45, 52, 60, 68, 75]
})

plt.scatter(df["experience"], df["salary"])

plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")

plt.show()
```

### Interpretation

If points move upward from left to right:

```text
Positive relationship
```

If they move downward:

```text
Negative relationship
```

If points are randomly distributed:

```text
Weak or no linear relationship
```

---

# 📈 Line Plot

Line plots are especially useful when one variable represents time.

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "month": ["Jan", "Feb", "Mar", "Apr", "May"],
    "sales": [100, 120, 115, 145, 170]
})

plt.plot(df["month"], df["sales"], marker="o")

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Monthly Sales")

plt.show()
```

A line plot can reveal:

* Trends
* Growth
* Decline
* Seasonality
* Sudden changes
* Possible anomalies

---

# 📐 Correlation

Correlation measures the **strength and direction of association** between two numerical variables.

A common correlation coefficient ranges from:

```text
-1 ───────── 0 ───────── +1
```

| Correlation | Interpretation                       |
| ----------: | ------------------------------------ |
|          +1 | Perfect positive linear relationship |
|        +0.7 | Strong positive relationship         |
|        +0.3 | Weak/moderate positive relationship  |
|           0 | No linear relationship               |
|        -0.3 | Weak/moderate negative relationship  |
|        -0.7 | Strong negative relationship         |
|          -1 | Perfect negative linear relationship |

These are practical guidelines, not universal scientific thresholds.

---

# 🧮 Covariance

Covariance indicates whether two variables tend to move together.

Conceptually:

```text
Positive covariance
→ variables tend to increase together

Negative covariance
→ one tends to increase while the other decreases
```

Formula:

$$
Cov(X,Y)=\frac{\sum_{i=1}^{n}(X_i-\bar X)(Y_i-\bar Y)}{n-1}
$$

However, covariance depends on the units of the variables, making direct comparison difficult.

Correlation standardizes the relationship.

---

# 📊 Pearson Correlation

Pearson correlation measures **linear association** between two numerical variables.

Formula:

$$
r =
\frac{Cov(X,Y)}
{\sigma_X\sigma_Y}
$$

In Pandas:

```python
correlation = df["experience"].corr(df["salary"])

print("Correlation:", correlation)
```

Or:

```python
correlation_matrix = df.corr(numeric_only=True)

print(correlation_matrix)
```

---

# 🏆 Spearman Correlation

Spearman correlation measures association based on **ranks**.

It can be useful when:

* Data is ordinal
* The relationship is monotonic but not linear
* Outliers affect Pearson correlation
* Variables are not normally distributed

```python
spearman = df["experience"].corr(
    df["salary"],
    method="spearman"
)

print("Spearman:", spearman)
```

---

# 🔥 Pearson vs Spearman

| Feature                           | Pearson            | Spearman              |
| --------------------------------- | ------------------ | --------------------- |
| Measures                          | Linear association | Monotonic association |
| Based on                          | Actual values      | Ranks                 |
| Outlier sensitivity               | Higher             | Generally lower       |
| Non-linear monotonic relationship | May miss           | Can detect            |
| Ordinal data                      | Usually unsuitable | Suitable              |

---

# 🧮 Correlation Matrix

A correlation matrix shows pairwise correlations between numerical variables.

```python
correlation_matrix = df.corr(numeric_only=True)

print(correlation_matrix)
```

Example:

```text
             age  income  spending
age          1.00   0.72      0.31
income       0.72   1.00      0.64
spending     0.31   0.64      1.00
```

The diagonal is always:

```text
1.0
```

because every variable is perfectly correlated with itself.

---

# 🔥 Heatmap

A heatmap makes the correlation matrix easier to interpret visually.

```python
import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = df.corr(numeric_only=True)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()
```

A heatmap can quickly identify:

```text
Strong Positive Relationships
Strong Negative Relationships
Weak Relationships
Potentially Redundant Features
```

---

# 📈 Regression Line

A regression line can help visualize a linear trend.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.regplot(
    data=df,
    x="experience",
    y="salary"
)

plt.title("Experience vs Salary")
plt.show()
```

The line summarizes the average linear trend.

Important:

> A regression line does not prove causation.

---

# 📊 Grouped Statistics

When analyzing a categorical variable against a numerical variable, grouped statistics are extremely useful.

```python
grouped = df.groupby("department")["salary"].agg(
    ["count", "mean", "median", "min", "max"]
)

print(grouped)
```

Example:

```text
             count   mean   median
Engineering      4  72.5     70.0
Marketing        3  61.3     60.0
HR               2  55.0     55.0
```

This allows us to compare distributions across categories.

---

# 📦 Box Plot

Box plots are excellent for comparing a numerical variable across categories.

```python
sns.boxplot(
    data=df,
    x="department",
    y="salary"
)

plt.title("Salary by Department")
plt.show()
```

A box plot displays:

* Median
* Quartiles
* IQR
* Potential outliers
* Distribution spread

---

# 🎻 Violin Plot

A violin plot combines distribution shape with summary information.

```python
sns.violinplot(
    data=df,
    x="department",
    y="salary"
)

plt.title("Salary Distribution by Department")
plt.show()
```

Useful when comparing:

```text
Category
   ↓
Distribution
   ↓
Shape + Spread + Center
```

---

# 📊 Bar Plot

Bar plots are useful for comparing aggregated numerical values across categories.

```python
sns.barplot(
    data=df,
    x="department",
    y="salary",
    estimator="mean"
)

plt.title("Average Salary by Department")
plt.show()
```

Always remember:

> A bar plot usually represents an aggregation, not every individual observation.

---

# 🔢 Count Plot

Count plots are useful for categorical-categorical analysis.

```python
sns.countplot(
    data=df,
    x="department",
    hue="gender"
)

plt.title("Department Distribution by Gender")
plt.show()
```

This allows us to compare category frequencies.

---

# 🧾 Contingency Tables

A contingency table summarizes the relationship between two categorical variables.

Example:

```python
table = pd.crosstab(
    df["department"],
    df["gender"]
)

print(table)
```

Output might look like:

```text
gender        Female  Male
department
Engineering       20    35
HR                25    15
Marketing         18    22
```

---

# 🔄 Cross Tabulation

`pd.crosstab()` is one of the most useful Pandas functions for categorical relationships.

```python
pd.crosstab(
    df["education"],
    df["promotion"]
)
```

Percentages can also be calculated:

```python
pd.crosstab(
    df["education"],
    df["promotion"],
    normalize="index"
)
```

This answers:

> Within each education category, what percentage belongs to each promotion category?

---

# 🧪 Chi-Square Test

For two categorical variables, the **chi-square test of independence** can help determine whether the observed association is statistically distinguishable from what would be expected under independence.

```python
from scipy.stats import chi2_contingency

table = pd.crosstab(
    df["department"],
    df["promotion"]
)

chi2, p_value, degrees_of_freedom, expected = chi2_contingency(table)

print("Chi-square:", chi2)
print("p-value:", p_value)
print("Degrees of freedom:", degrees_of_freedom)
```

A commonly used significance threshold is:

```text
α = 0.05
```

But statistical significance should be interpreted together with:

* Effect size
* Sample size
* Study design
* Practical importance
* Multiple testing
* Data quality

---

# 🔢 Numerical Relationship Analysis

For numerical variables, a practical workflow is:

```text
Scatter Plot
     ↓
Correlation
     ↓
Distribution Check
     ↓
Outlier Check
     ↓
Trend / Regression
     ↓
Interpretation
```

Example:

```python
x = df["experience"]
y = df["salary"]

print("Pearson:", x.corr(y))
print("Spearman:", x.corr(y, method="spearman"))
```

---

# 🏷️ Categorical Relationship Analysis

For categorical variables:

```text
Frequency
   ↓
Cross Tabulation
   ↓
Percentage Comparison
   ↓
Visualization
   ↓
Statistical Test
```

Example:

```python
table = pd.crosstab(
    df["department"],
    df["promotion"],
    normalize="index"
)

print(table)
```

---

# 🚨 Outliers in Bivariate Analysis

Outliers can strongly affect relationships.

Example:

```text
Normal observations:

X → Y
X → Y
X → Y
X → Y

Extreme observation:

X → VERY LARGE Y
```

This can change:

* Correlation
* Regression slope
* Mean
* Variance
* Model coefficients

Always inspect scatter plots before interpreting correlation.

---

# 🕳️ Missing Values

Missing values can affect bivariate analysis.

Check missingness first:

```python
print(df[["experience", "salary"]].isna().sum())
```

For correlation, Pandas generally computes using available paired observations.

However, do not blindly delete missing rows.

Consider:

* Why values are missing
* How many values are missing
* Whether missingness is systematic
* Whether imputation is appropriate

---

# 🔄 Data Transformation

Some relationships become easier to understand after transformation.

For heavily right-skewed variables:

```python
import numpy as np

df["log_income"] = np.log1p(df["income"])
```

Then compare:

```python
sns.scatterplot(
    data=df,
    x="experience",
    y="log_income"
)

plt.show()
```

Transformation may reveal patterns hidden by extreme scale differences.

---

# 🌀 Non-Linear Relationships

Correlation does not capture every type of relationship.

Consider:

```text
        Y
        │       ●
        │    ●     ●
        │  ●         ●
        │ ●           ●
        └──────────────── X
```

There is a strong curved relationship, but Pearson correlation may be close to zero.

Therefore:

> Always visualize numerical relationships instead of relying only on correlation coefficients.

Useful tools include:

* Scatter plots
* Regression plots
* Polynomial features
* Transformations
* Spearman correlation
* Domain-specific analysis

---

# ⚠️ Correlation Does Not Mean Causation

This is one of the most important rules in data analysis.

Suppose:

```text
Ice Cream Sales ↑
      ↕
Swimming Accidents ↑
```

A correlation does not mean:

```text
Ice Cream → Swimming Accidents
```

A third variable may influence both:

```text
        Temperature
        ↙        ↘
Ice Cream      Swimming
Sales          Activity
```

Possible explanations for observed association include:

* Confounding variables
* Selection effects
* Reverse causation
* Measurement issues
* Coincidence

Bivariate EDA can identify patterns. It does not, by itself, establish causality.

---

# 🐼 Pandas Implementation

A practical Pandas workflow:

```python
import pandas as pd

df = pd.read_csv("data.csv")

# Numerical relationship
correlation = df["age"].corr(df["income"])

print("Age-Income Correlation:", correlation)

# Grouped statistics
grouped = df.groupby("department")["salary"].agg(
    ["count", "mean", "median", "std"]
)

print(grouped)

# Categorical relationship
table = pd.crosstab(
    df["department"],
    df["gender"]
)

print(table)
```

---

# 🎨 Matplotlib Implementation

Matplotlib provides low-level control over visualizations.

```python
import matplotlib.pyplot as plt

plt.scatter(
    df["experience"],
    df["salary"]
)

plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")

plt.show()
```

For categorical comparisons:

```python
grouped = df.groupby("department")["salary"].mean()

plt.bar(
    grouped.index,
    grouped.values
)

plt.xlabel("Department")
plt.ylabel("Average Salary")
plt.title("Average Salary by Department")

plt.xticks(rotation=45)
plt.show()
```

---

# 🎨 Seaborn Implementation

Seaborn provides convenient statistical visualizations.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    data=df,
    x="experience",
    y="salary"
)

plt.title("Experience vs Salary")
plt.show()
```

Correlation heatmap:

```python
correlation_matrix = df.corr(numeric_only=True)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f"
)

plt.show()
```

---

# 🧰 Reusable Functions

Reusable functions make EDA code cleaner.

## Numerical Relationship

```python
def numerical_relationship(df, x, y):
    result = {
        "pearson": df[x].corr(df[y], method="pearson"),
        "spearman": df[x].corr(df[y], method="spearman")
    }

    return result
```

Usage:

```python
result = numerical_relationship(
    df,
    "experience",
    "salary"
)

print(result)
```

---

## Group Summary

```python
def grouped_summary(df, category, numeric):
    return df.groupby(category)[numeric].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
```

Usage:

```python
summary = grouped_summary(
    df,
    "department",
    "salary"
)

print(summary)
```

---

## Categorical Relationship

```python
def categorical_relationship(df, column_a, column_b):
    return pd.crosstab(
        df[column_a],
        df[column_b]
    )
```

Usage:

```python
table = categorical_relationship(
    df,
    "department",
    "promotion"
)

print(table)
```

---

# 🤖 Automated Bivariate Analysis

A reusable numerical analysis function:

```python
def analyze_numeric_relationship(df, x, y):
    subset = df[[x, y]].dropna()

    return {
        "x": x,
        "y": y,
        "observations": len(subset),
        "pearson": subset[x].corr(
            subset[y],
            method="pearson"
        ),
        "spearman": subset[x].corr(
            subset[y],
            method="spearman"
        )
    }
```

Example:

```python
result = analyze_numeric_relationship(
    df,
    "experience",
    "salary"
)

for key, value in result.items():
    print(f"{key}: {value}")
```

---

# 🎯 Feature Selection

Bivariate analysis can provide useful information for feature selection.

Suppose a dataset contains:

```text
age
income
experience
education
distance
random_id
```

You may investigate:

```text
Feature ↔ Target
```

For example:

```python
for column in ["age", "income", "experience", "distance"]:
    correlation = df[column].corr(df["target"])
    print(column, correlation)
```

However:

> Feature selection should not be based only on pairwise correlation.

A feature may have weak individual correlation with the target but still be useful because of:

* Non-linear relationships
* Interactions
* Conditional relationships
* Category effects
* Multivariate structure

---

# 🤖 Machine Learning Applications

Bivariate analysis supports many stages of ML.

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Univariate Analysis
     ↓
Bivariate Analysis
     ↓
Multivariate Analysis
     ↓
Feature Engineering
     ↓
Feature Selection
     ↓
Modeling
```

It can help identify:

### 1. Potentially useful features

```text
Feature ↔ Target
```

### 2. Redundant variables

```text
Feature A ↔ Feature B
```

Very strong relationships between predictors may indicate redundancy, although correlation alone is not sufficient to diagnose multicollinearity.

### 3. Data quality problems

```text
Impossible relationships
Extreme observations
Unexpected categories
```

### 4. Potential leakage

```text
Feature ↔ Target
```

A very strong relationship may require investigation for target leakage.

---

# ⚠️ Common Mistakes

## ❌ Mistake 1: Looking Only at Correlation

A correlation coefficient does not describe every possible relationship.

### Better:

```text
Correlation + Visualization + Context
```

---

## ❌ Mistake 2: Assuming Correlation Means Causation

Correlation describes association.

It does not automatically establish cause and effect.

---

## ❌ Mistake 3: Ignoring Outliers

One extreme observation can substantially change Pearson correlation.

---

## ❌ Mistake 4: Using the Wrong Plot

For example:

```text
Numerical + Numerical
→ Scatter Plot

Categorical + Numerical
→ Box Plot / Violin Plot

Categorical + Categorical
→ Count Plot / Cross Tabulation
```

---

## ❌ Mistake 5: Comparing Means Only

Two groups can have the same mean but completely different distributions.

Use:

* Median
* Quartiles
* Standard deviation
* Box plots
* Violin plots

---

## ❌ Mistake 6: Ignoring Sample Size

A relationship observed in a tiny dataset may be unstable.

Always consider:

```text
Effect
+
Sample Size
+
Uncertainty
+
Context
```

---

## ❌ Mistake 7: Treating Statistical Significance as Practical Importance

A statistically detectable association may be too small to matter practically.

---

## ❌ Mistake 8: Using Bivariate Analysis as the Final Analysis

Real-world datasets contain many interacting variables.

Bivariate analysis is a step toward:

```text
Multivariate Analysis
```

---

# ✅ Best Practices

### 1. Start With Data Types

```text
Numerical?
Categorical?
Date-Time?
Ordinal?
```

### 2. Visualize Before Interpreting

```text
Plot → Inspect → Calculate → Interpret
```

### 3. Check Missing Values

```python
df[[x, y]].isna().sum()
```

### 4. Check Outliers

Especially for numerical relationships.

### 5. Use Appropriate Statistics

```text
Pearson
Spearman
Grouped statistics
Chi-square
```

### 6. Consider Non-Linearity

Do not assume every relationship is linear.

### 7. Consider Confounders

A third variable may explain the observed relationship.

### 8. Separate Association From Causation

Always use precise language.

Prefer:

> "X is associated with Y."

instead of:

> "X causes Y."

unless the study design and evidence justify a causal claim.

---

# 🚀 Mini Projects

## 🏠 Project 1 — House Price Relationships

Dataset:

```text
house_size
bedrooms
age
location
price
```

Analyze:

```text
House Size ↔ Price
Bedrooms ↔ Price
Age ↔ Price
Location ↔ Price
```

Visualizations:

* Scatter plots
* Box plots
* Correlation matrix
* Heatmap

---

## 💼 Project 2 — Employee Analysis

Dataset:

```text
age
experience
department
education
salary
promotion
```

Analyze:

```text
Experience ↔ Salary
Department ↔ Salary
Education ↔ Promotion
Department ↔ Promotion
```

---

## 🛒 Project 3 — E-Commerce Analysis

Dataset:

```text
age
gender
product_category
purchase_amount
session_time
```

Analyze:

```text
Age ↔ Purchase Amount
Session Time ↔ Purchase Amount
Gender ↔ Purchase Amount
Category ↔ Purchase Amount
```

---

## 📚 Project 4 — Student Performance

Dataset:

```text
study_hours
attendance
sleep_hours
previous_score
final_score
```

Analyze:

```text
Study Hours ↔ Final Score
Attendance ↔ Final Score
Sleep Hours ↔ Final Score
Previous Score ↔ Final Score
```

---

# 🧠 Exercises

## 🟢 Beginner

1. Create a scatter plot between two numerical variables.
2. Calculate Pearson correlation.
3. Calculate Spearman correlation.
4. Create a correlation matrix.
5. Create a heatmap.
6. Calculate grouped means.
7. Create a box plot.
8. Create a cross-tabulation.
9. Create a count plot.
10. Identify positive and negative relationships.

---

## 🟡 Intermediate

1. Compare Pearson and Spearman correlation.
2. Investigate the effect of an outlier on correlation.
3. Compare two groups using box plots.
4. Analyze a categorical-categorical relationship.
5. Perform a chi-square test.
6. Analyze a date-time and numerical relationship.
7. Investigate a possible non-linear relationship.
8. Create an automated bivariate report.
9. Compare mean and median relationships.
10. Investigate potential feature redundancy.

---

## 🔴 Advanced

1. Investigate Simpson's paradox using a real or simulated dataset.
2. Compare correlation before and after transformation.
3. Study the effect of outliers on regression.
4. Investigate confounding variables.
5. Compare Pearson, Spearman, and Kendall correlation.
6. Analyze interaction effects.
7. Study multicollinearity after bivariate exploration.
8. Build an automated EDA pipeline.
9. Investigate potential target leakage.
10. Compare bivariate findings with multivariate model results.

---

# 📁 Project Structure

```text
06-Exploratory-Data-Analysis/
│
├── README.md
│
├── 01-Univariate-Analysis/
│   ├── README.md
│   ├── numerical-analysis.py
│   ├── categorical-analysis.py
│   ├── distribution-analysis.py
│   ├── outlier-analysis.py
│   └── univariate-report.py
│
├── 02-Bivariate-Analysis/
│   ├── README.md
│   ├── numerical-relationships.py
│   ├── correlation-analysis.py
│   ├── categorical-relationships.py
│   ├── grouped-analysis.py
│   ├── contingency-analysis.py
│   └── bivariate-report.py
│
├── 03-Multivariate-Analysis/
│
├── 04-Data-Visualization/
│
└── 05-EDA-Projects/
```

---

# 💻 End-to-End Example

The following example demonstrates a complete bivariate workflow.

```python
"""
Bivariate Analysis
------------------
Analyze relationships between two variables.

Requirements:
    pip install pandas matplotlib seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_dataset():
    """Create a small demonstration dataset."""

    return pd.DataFrame({
        "experience": [1, 2, 3, 4, 5, 6, 7, 8],
        "salary": [35, 40, 45, 50, 58, 65, 72, 80],
        "department": [
            "HR",
            "HR",
            "IT",
            "IT",
            "IT",
            "Sales",
            "Sales",
            "Sales"
        ],
        "promotion": [
            "No",
            "No",
            "Yes",
            "No",
            "Yes",
            "Yes",
            "Yes",
            "Yes"
        ]
    })


def numerical_analysis(df):
    """Analyze the relationship between experience and salary."""

    x = df["experience"]
    y = df["salary"]

    pearson = x.corr(y, method="pearson")
    spearman = x.corr(y, method="spearman")

    print("Numerical Relationship")
    print("-" * 30)
    print(f"Pearson correlation : {pearson:.4f}")
    print(f"Spearman correlation: {spearman:.4f}")


def categorical_analysis(df):
    """Analyze categorical relationships."""

    table = pd.crosstab(
        df["department"],
        df["promotion"]
    )

    print("\nDepartment vs Promotion")
    print("-" * 30)
    print(table)


def grouped_analysis(df):
    """Analyze salary by department."""

    summary = df.groupby("department")["salary"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )

    print("\nSalary by Department")
    print("-" * 30)
    print(summary)


def visualize_numerical(df):
    """Create a scatter plot."""

    sns.scatterplot(
        data=df,
        x="experience",
        y="salary"
    )

    plt.title("Experience vs Salary")
    plt.xlabel("Experience")
    plt.ylabel("Salary")
    plt.tight_layout()
    plt.show()


def visualize_categorical(df):
    """Create a box plot."""

    sns.boxplot(
        data=df,
        x="department",
        y="salary"
    )

    plt.title("Salary by Department")
    plt.xlabel("Department")
    plt.ylabel("Salary")
    plt.tight_layout()
    plt.show()


def main():
    """Run the complete bivariate analysis."""

    df = create_dataset()

    print("Dataset")
    print("-" * 30)
    print(df)

    numerical_analysis(df)
    categorical_analysis(df)
    grouped_analysis(df)

    visualize_numerical(df)
    visualize_categorical(df)


if __name__ == "__main__":
    main()
```

---

# 🔬 Bivariate Analysis Workflow

A strong practical workflow looks like this:

```text
                    DATASET
                       │
                       ↓
                Identify Variables
                       │
                       ↓
              Check Data Types
                       │
             ┌─────────┼─────────┐
             ↓         ↓         ↓
         Num + Num  Cat + Num  Cat + Cat
             │         │         │
             ↓         ↓         ↓
         Scatter     Boxplot    Crosstab
             │         │         │
             ↓         ↓         ↓
       Correlation  Group Stats Chi-Square
             │         │         │
             └─────────┼─────────┘
                       ↓
                Check Outliers
                       ↓
               Check Missing Data
                       ↓
              Interpret Relationship
                       ↓
             Investigate Context
                       ↓
              Multivariate Analysis
```

---

# 📋 Bivariate Analysis Checklist

Before completing your analysis:

### 🔍 Data Understanding

* [ ] Identify the two variables.
* [ ] Check their data types.
* [ ] Understand their meaning.
* [ ] Check missing values.
* [ ] Check unique values.
* [ ] Check outliers.

### 📊 Numerical + Numerical

* [ ] Scatter plot
* [ ] Pearson correlation
* [ ] Spearman correlation
* [ ] Distribution check
* [ ] Outlier investigation
* [ ] Non-linearity check

### 🏷️ Categorical + Numerical

* [ ] Grouped statistics
* [ ] Mean
* [ ] Median
* [ ] Standard deviation
* [ ] Box plot
* [ ] Violin plot

### 🏷️ Categorical + Categorical

* [ ] Cross-tabulation
* [ ] Percentages
* [ ] Count plot
* [ ] Chi-square test when appropriate
* [ ] Effect size/context

### 🧠 Interpretation

* [ ] Describe the relationship accurately.
* [ ] Do not confuse correlation with causation.
* [ ] Consider confounders.
* [ ] Consider sample size.
* [ ] Check whether the relationship is practically important.
* [ ] Investigate surprising patterns.

---

# 🧭 Decision Guide

Use this quick reference:

```text
                What are your variables?
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
       Num + Num      Cat + Num      Cat + Cat
          │              │              │
          ↓              ↓              ↓
      Scatter         Box Plot       Crosstab
          │              │              │
          ↓              ↓              ↓
    Correlation      Group Stats     Chi-Square
          │              │              │
          ↓              ↓              ↓
     Regression      Violin Plot    Count Plot
```

For time-based data:

```text
Date-Time + Numerical
        ↓
    Line Plot
        ↓
    Trend Analysis
```

---

# 🗺️ EDA Roadmap

This section fits into the broader EDA journey:

```text
06-Exploratory-Data-Analysis/
│
├── 01-Univariate-Analysis/
│       ↓
│   One Variable
│
├── 02-Bivariate-Analysis/
│       ↓
│   Two Variables
│
├── 03-Multivariate-Analysis/
│       ↓
│   Multiple Variables
│
├── 04-Data-Visualization/
│       ↓
│   Communicate Insights
│
└── 05-EDA-Projects/
        ↓
    Real-World Analysis
```

### 🌱 GROW WITH EDA

```text
🌱 GROW
  ↓
📖 LEARN
  ↓
🔍 INSPECT
  ↓
📊 ANALYZE
  ↓
🔗 CONNECT
  ↓
📈 VISUALIZE
  ↓
🧠 INTERPRET
  ↓
🤖 MODEL
  ↓
🚀 BUILD
```

---

# 🎯 Key Takeaways

> **Bivariate analysis is about understanding relationships between two variables.**

Remember:

1. **Choose the technique based on variable types.**
2. **Use scatter plots for numerical relationships.**
3. **Use box/violin plots for categorical-numerical comparisons.**
4. **Use cross-tabulation for categorical relationships.**
5. **Correlation measures association, not causation.**
6. **Pearson focuses on linear association.**
7. **Spearman works with ranked/monotonic relationships.**
8. **Always investigate outliers.**
9. **Do not rely on a single statistic.**
10. **Consider sample size and practical importance.**
11. **Look for non-linear relationships.**
12. **Bivariate analysis prepares the foundation for multivariate analysis.**

---

# 🚀 Next Step

After learning how two variables interact, the next challenge is understanding **multiple variables simultaneously**.

Continue with:

```text
06-Exploratory-Data-Analysis/
└── 03-Multivariate-Analysis/
```

You will learn:

* Multiple-variable relationships
* Pair plots
* Multivariate correlation
* Feature interactions
* Multicollinearity
* PCA visualization
* Dimensionality reduction
* Grouped multivariate analysis
* Advanced visualization
* Feature interactions for ML

```text
ONE VARIABLE
     ↓
UNIVARIATE
     ↓
TWO VARIABLES
     ↓
BIVARIATE
     ↓
MULTIPLE VARIABLES
     ↓
MULTIVARIATE
     ↓
MACHINE LEARNING 🚀
```

---

# 👨‍💻 Author

**Kishor Patil**

Machine Learning | Python | Data Science | AI

🔗 GitHub:
[https://github.com/Kishor055](https://github.com/Kishor055)

---

# 🤝 Contributing

Contributions are welcome!

You can contribute by:

* Improving explanations
* Fixing errors
* Adding practical examples
* Adding datasets
* Improving visualizations
* Adding exercises
* Adding real-world EDA projects

### Contribution Workflow

```bash
git clone https://github.com/Kishor055/Machine-Learning.git

cd Machine-Learning

git checkout -b feature/improve-bivariate-analysis
```

Make your changes, test the examples, and create a pull request.

---

# ⭐ Support

If this repository helps you learn Machine Learning:

* ⭐ Star the repository
* 🍴 Fork the repository
* 🐛 Report issues
* 💡 Suggest improvements
* 🤝 Contribute examples

Every contribution helps make the learning path better.

---

# 📄 License

This project is intended for educational and learning purposes.

Please check the repository's license file for the applicable terms.

---

<div align="center">

### 🌱 GROW → ANALYZE → CONNECT → UNDERSTAND → BUILD 🚀

**Learn the relationship. Discover the pattern. Build better models.**

</div>

This README is structured to bridge **`01-Univariate-Analysis` → `02-Bivariate-Analysis` → `03-Multivariate-Analysis`** while keeping the examples directly applicable to Machine Learning workflows.
