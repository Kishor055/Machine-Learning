# 📊 Outlier Detection and Handling in Python

🌱 **GROW → INSPECT → DETECT → INVESTIGATE → HANDLE → VALIDATE**

Welcome to the **Outliers** topic in the Data Cleaning module of the Machine Learning learning path. This guide explains how to identify unusual observations, investigate their causes, and decide whether to retain, correct, transform, cap, or remove them using Python.

Outliers can occur because of data entry errors, measurement failures, unusual but valid events, or naturally extreme values. Correctly handling outliers improves data quality while helping prevent the loss of valuable information.

## 📚 Table of Contents

1. [Introduction](#-introduction)
2. [What Are Outliers?](#-what-are-outliers)
3. [Types of Outliers](#-types-of-outliers)
4. [Why Outlier Handling Matters](#-why-outlier-handling-matters)
5. [Create a Sample Dataset](#-create-a-sample-dataset)
6. [Explore and Visualize Outliers](#-explore-and-visualize-outliers)
7. [Detect Outliers Using the IQR Method](#-detect-outliers-using-the-iqr-method)
8. [Detect Outliers Using Z-Scores](#-detect-outliers-using-z-scores)
9. [Detect Outliers Using the Modified Z-Score](#-detect-outliers-using-the-modified-z-score)
10. [Detect Outliers Using Domain Rules](#-detect-outliers-using-domain-rules)
11. [Multivariate Outlier Detection](#-multivariate-outlier-detection)
12. [Methods for Handling Outliers](#-methods-for-handling-outliers)
13. [Winsorization and Capping](#-winsorization-and-capping)
14. [Log and Other Transformations](#-log-and-other-transformations)
15. [Compare Data Before and After Handling](#-compare-data-before-and-after-handling)
16. [Complete Outlier Detection Program](#-complete-outlier-detection-program)
17. [Outliers in Machine Learning](#-outliers-in-machine-learning)
18. [Common Mistakes](#-common-mistakes)
19. [Best Practices](#-best-practices)
20. [Mini Projects](#-mini-projects)
21. [Practice Exercises](#-practice-exercises)
22. [Quick Revision](#-quick-revision)
23. [Project Structure](#-project-structure)
24. [Learning Roadmap](#-learning-roadmap)

---

## 🎯 Introduction

Real-world datasets often contain observations that differ significantly from most other observations.

For example, a student marks dataset might contain:

```python
marks = [65, 72, 78, 80, 75, 82, 77, 79, 95, 500]
```

Most marks fall between 65 and 95, while `500` is suspicious if the maximum possible mark is 100.

However, an unusually high salary, an expensive property, or an exceptional test result is not necessarily an error. It may be a genuine observation.

Therefore, outlier handling should begin with **detection and investigation**, not automatic deletion.

### Learning objectives

By the end of this guide, you will be able to:

- Understand different kinds of outliers.
- Detect unusual observations using visual and statistical techniques.
- Apply the IQR, Z-score, and Modified Z-score methods.
- Distinguish statistical anomalies from invalid values.
- Explore multivariate outlier detection.
- Choose suitable strategies for handling outliers.
- Build reusable outlier detection functions.
- Evaluate the effect of outliers on machine learning models.
- Avoid data leakage during preprocessing.

## 🧠 What Are Outliers?

An **outlier** is an observation that differs substantially from the general pattern of a dataset or from what is expected under a particular model or domain rule.

Consider these values:

```text
10, 12, 11, 13, 12, 14, 11, 12, 150
```

The value `150` is much larger than the others and may be an outlier.

Whether it is an error depends on the context. It might be a typing mistake, a measurement error, or a genuine exceptional observation.

### Outlier versus invalid value

| Concept | Example | Interpretation |
|---|---|---|
| Statistical outlier | Salary of ₹10,00,000 when most salaries are around ₹40,000–₹60,000 | Unusual, but possibly valid |
| Invalid value | Age of `-10` | Violates a logical constraint |
| Data entry error | Marks of `500` when the maximum is 100 | Likely incorrect |
| Rare but valid value | A very expensive house in a housing dataset | Potentially meaningful |
| Anomalous combination | Ordinary age paired with an unexpectedly extreme spending amount | May require multivariate analysis |

**Important:** A statistical rule identifies unusual observations, not the reason they are unusual.

## 🧩 Types of Outliers

### 1. Univariate outliers

An observation is unusual with respect to one variable.

Example:

```text
Salary = [30000, 32000, 35000, 33000, 34000, 500000]
```

The value `500000` may be unusual compared with the other salary values.

### 2. Multivariate outliers

An observation appears unusual when multiple variables are considered together.

For example, an adult's height and weight may each fall within common ranges individually, but their combination may be unusual for a particular population.

Multivariate detection considers relationships among features rather than evaluating each column separately.

### 3. Global outliers

An observation is unusual across the entire dataset.

Example: an extremely high transaction amount compared with all other transactions.

### 4. Contextual outliers

An observation is unusual only in a particular context.

Example: an electricity reading of 100 units may be ordinary for one building but unusually high for another building with very different usage patterns.

### 5. Collective anomalies

A group or sequence of observations is unusual even when individual values may appear normal.

Example: a rapid series of otherwise ordinary financial transactions may indicate unusual activity.

These distinctions are important because a simple IQR rule may identify individual extremes but fail to detect unusual patterns involving time, groups, or multiple variables.

## 🚀 Why Outlier Handling Matters

Outliers may affect:

- Mean and standard deviation.
- Correlation and regression coefficients.
- Linear regression errors.
- Distance-based methods such as K-nearest neighbors.
- Clustering algorithms such as K-means.
- Feature scaling and visualization.
- Model performance and interpretability.

For example:

```python
import numpy as np

normal_values = np.array([10, 12, 11, 13, 12, 14])
values_with_outlier = np.array([10, 12, 11, 13, 12, 14, 150])

print("Mean without extreme value:", normal_values.mean())
print("Mean with extreme value:", values_with_outlier.mean())
```

The extreme value substantially changes the average.

However, outliers can also contain important business information. In fraud detection, equipment monitoring, financial analysis, and medical research, anomalies may be precisely what the analysis is intended to discover.

The objective is to handle outliers appropriately, not eliminate them automatically.

## 🧪 Create a Sample Dataset

Start with a small employee dataset containing an extreme salary and a suspicious age.

```python
import pandas as pd

df = pd.DataFrame({
    "Employee": [
        "Amit", "Priya", "Rahul", "Neha", "Kiran",
        "Meera", "Arjun", "Sneha", "Rohan", "Anaya"
    ],
    "Age": [24, 28, 31, 29, 35, 27, 32, 30, 26, 250],
    "Salary": [
        35000, 42000, 48000, 45000, 52000,
        40000, 49000, 46000, 43000, 500000
    ],
    "Experience": [1, 3, 6, 4, 9, 2, 7, 5, 2, 4]
})

print(df)
print(df.describe())
```

Potential problems include:

- Age `250`, which violates a reasonable human-age constraint.
- Salary `500000`, which may be an outlier but could still be valid.
- A combination of age and experience that may require further investigation.

The domain rule for age is stronger than a statistical outlier rule. The salary requires additional context before any deletion or correction.

## 🔍 Explore and Visualize Outliers

Visual inspection helps reveal unusual values and distribution patterns.

### 1. Inspect descriptive statistics

```python
print(df[["Age", "Salary", "Experience"]].describe())
```

Look for values such as extreme minimums or maximums and compare them with domain expectations.

### 2. Use a box plot

A box plot displays the median, quartiles, spread, and observations that fall beyond the plotted whiskers.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.boxplot(df["Salary"].dropna())
plt.title("Salary Box Plot")
plt.ylabel("Salary")
plt.show()
```

Box-plot whiskers commonly follow an IQR-based convention. Points plotted beyond the whiskers are potential outliers, not automatically invalid data.

### 3. Use a histogram

```python
plt.figure(figsize=(8, 5))
plt.hist(df["Salary"].dropna(), bins=10, edgecolor="black")
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()
```

A histogram can reveal skewed distributions, multiple peaks, and isolated extreme observations.

### 4. Compare linear and logarithmic scales

Extreme observations can make the rest of a distribution difficult to inspect.

```python
plt.figure(figsize=(8, 5))
plt.hist(df["Salary"].dropna(), bins=10, edgecolor="black")
plt.yscale("log")
plt.title("Salary Distribution with Logarithmic Frequency Axis")
plt.xlabel("Salary")
plt.ylabel("Frequency (log scale)")
plt.show()
```

This example changes the frequency axis. If positive values span a wide range, a logarithmic value axis can also help, provided all displayed values are positive.

### 5. Review suspicious records

```python
print(df.loc[df["Age"] > 120])
print(df.loc[df["Salary"] > 100000])
```

These are initial review filters for this example, not universal outlier thresholds.

---

## 📐 Detect Outliers Using the IQR Method

The **Interquartile Range (IQR)** method identifies values that fall unusually far below or above the middle 50% of observations.

It is often useful for skewed data because it relies on quartiles rather than the mean and standard deviation.

### Formula

The interquartile range is:

\[
IQR = Q_3 - Q_1
\]

Where:

- \(Q_1\) is the 25th percentile.
- \(Q_3\) is the 75th percentile.

Using the conventional multiplier of 1.5:

\[
\text{Lower Fence} = Q_1 - 1.5 \times IQR
\]

\[
\text{Upper Fence} = Q_3 + 1.5 \times IQR
\]

An observation below the lower fence or above the upper fence is flagged as a potential outlier.

The multiplier can be adjusted for specific use cases, but the choice should be justified.

### Step 1: Calculate the quartiles

```python
q1 = df["Salary"].quantile(0.25)
q3 = df["Salary"].quantile(0.75)

iqr = q3 - q1

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
```

### Step 2: Calculate the fences

```python
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

print("Lower fence:", lower_fence)
print("Upper fence:", upper_fence)
```

### Step 3: Identify outliers

```python
outlier_mask = (
    (df["Salary"] < lower_fence)
    | (df["Salary"] > upper_fence)
)

outliers = df.loc[outlier_mask]

print(outliers)
```

### Create a reusable IQR function

```python
import pandas as pd


def detect_iqr_outliers(
    series: pd.Series,
    multiplier: float = 1.5
) -> pd.Series:
    """Return a Boolean mask identifying IQR outliers."""

    numeric = pd.to_numeric(series, errors="coerce")

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - multiplier * iqr
    upper_fence = q3 + multiplier * iqr

    return (
        numeric.lt(lower_fence)
        | numeric.gt(upper_fence)
    ).fillna(False).astype(bool)


salary_outliers = detect_iqr_outliers(df["Salary"])

print(df.loc[salary_outliers])
```

This function:

- Converts values to numeric form where possible.
- Leaves invalid conversions unflagged so they can be inspected separately.
- Preserves the Series index in its Boolean result.
- Does not modify or delete any records.

**Limitation:** The IQR method can flag valid extreme observations. When the IQR is zero or the sample is very small, review the results carefully.

## 📏 Detect Outliers Using Z-Scores

A **Z-score** measures how far an observation lies from the mean in standard deviation units.

### Formula

\[
z = \frac{x - \mu}{\sigma}
\]

Where:

- \(x\) is the observation.
- \(\mu\) is the mean.
- \(\sigma\) is the population standard deviation used in this calculation.

A common exploratory rule flags values whose absolute Z-score exceeds 3.

\[
|z| > 3
\]

This is a heuristic, not a universal rule. Its interpretation is strongest for distributions that are approximately normal and can be misleading for highly skewed data.

### Example

```python
salary = pd.to_numeric(df["Salary"], errors="coerce")

mean = salary.mean()
std = salary.std(ddof=0)

if std > 0:
    z_scores = (salary - mean) / std
    outlier_mask = z_scores.abs() > 3
else:
    z_scores = pd.Series(float("nan"), index=salary.index)
    outlier_mask = pd.Series(False, index=salary.index)

print(df.loc[outlier_mask])
```

### Reusable Z-score function

```python
def detect_zscore_outliers(
    series: pd.Series,
    threshold: float = 3.0
) -> pd.Series:
    """Return a Boolean mask for observations beyond a Z-score threshold."""

    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=0)

    if numeric.notna().sum() == 0 or pd.isna(std) or std == 0:
        return pd.Series(False, index=series.index, dtype=bool)

    z_scores = (numeric - numeric.mean()) / std

    return z_scores.abs().gt(threshold).fillna(False).astype(bool)


zscore_mask = detect_zscore_outliers(df["Salary"])
print(df.loc[zscore_mask])
```

### Advantages

- Easy to understand and calculate.
- Useful for approximately normal numerical data.
- Produces a standardized measure across comparable variables.

### Limitations

- The mean and standard deviation are sensitive to extreme values.
- Skewed data can generate misleading results.
- A small dataset may not provide a reliable estimate of the distribution.

---

## 🛡️ Detect Outliers Using the Modified Z-Score

The **Modified Z-score** uses the median and Median Absolute Deviation (MAD). These statistics are more robust to extreme values than the mean and standard deviation.

### Formula

First calculate the median:

\[
m = \operatorname{median}(x)
\]

Then calculate the median absolute deviation:

\[
MAD = \operatorname{median}(|x_i - m|)
\]

The Modified Z-score is:

\[
M_i = \frac{0.6745(x_i - m)}{MAD}
\]

A common exploratory threshold is:

\[
|M_i| > 3.5
\]

The threshold is a convention, not a guarantee that an observation is wrong.

### Example

```python
import numpy as np
import pandas as pd


def detect_modified_zscore_outliers(
    series: pd.Series,
    threshold: float = 3.5
) -> pd.Series:
    """Detect potential outliers using the Modified Z-score."""

    numeric = pd.to_numeric(series, errors="coerce")

    median = numeric.median()
    mad = (numeric - median).abs().median()

    if numeric.notna().sum() == 0 or pd.isna(mad) or mad == 0:
        return pd.Series(False, index=series.index, dtype=bool)

    modified_z = 0.6745 * (numeric - median) / mad

    return modified_z.abs().gt(threshold).fillna(False).astype(bool)


modified_z_mask = detect_modified_zscore_outliers(df["Salary"])

print(df.loc[modified_z_mask])
```

### When is it useful?

Consider the Modified Z-score when:

- A distribution contains a few very extreme observations.
- The mean is being distorted by unusually large values.
- A robust, median-based detection method is preferable.

**Limitation:** If the MAD is zero, this implementation returns an all-false mask. That does not prove that no outliers exist. In such cases, inspect the distribution, use domain rules, or choose another detection method.

---

## 📋 Detect Outliers Using Domain Rules

Statistical methods should not replace basic data validation.

For example, a negative age or attendance above 100% might violate the definition of a field.

```python
invalid_age = (df["Age"] < 0) | (df["Age"] > 120)

print("Invalid age records:")
print(df.loc[invalid_age])
```

Check attendance against its expected range:

```python
attendance = pd.Series([85, 92, 105, -5, 78])

invalid_attendance = ~attendance.between(0, 100)

print(attendance[invalid_attendance])
```

### Domain validation versus statistical detection

| Method | Question it answers |
|---|---|
| Domain validation | Does this value violate a known rule? |
| IQR | Is this observation unusually far from the central 50%? |
| Z-score | How far is this observation from the mean in standard deviation units? |
| Modified Z-score | How far is this observation from the median, using MAD as a robust scale? |
| Multivariate method | Is this combination of features unusual? |

A value can pass statistical detection and still violate a business rule. It can also be an outlier statistically while remaining perfectly valid.

---

## 🔎 Multivariate Outlier Detection

Univariate methods examine one feature at a time. They may miss unusual combinations of otherwise ordinary values.

For example, age and annual salary might each fall within common ranges, while their combination is unusual for a specific business context.

One option is **Isolation Forest**, an anomaly-detection algorithm that isolates unusual observations using randomly selected features and split points.

### Example using Scikit-Learn

```python
import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.DataFrame({
    "Age": [24, 28, 31, 29, 35, 27, 32, 30, 26, 60],
    "Salary": [
        35000, 42000, 48000, 45000, 52000,
        40000, 49000, 46000, 43000, 350000
    ],
    "Experience": [1, 3, 6, 4, 9, 2, 7, 5, 2, 35]
})

features = df[["Age", "Salary", "Experience"]]

detector = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42
)

predictions = detector.fit_predict(features)

df["Anomaly_Flag"] = predictions == -1

print(df)
print("\nPotential multivariate anomalies:")
print(df.loc[df["Anomaly_Flag"]])
```

In Isolation Forest predictions:

- `1` indicates an observation treated as normal.
- `-1` indicates a potential anomaly.

The `contamination` parameter affects the expected proportion of observations flagged when a numeric value is supplied. `"auto"` uses the estimator's default decision rule. Neither setting guarantees that the flagged records are errors.

### Important limitations

- The algorithm flags anomalies relative to the supplied features and data distribution.
- Results depend on the data, selected features, and model parameters.
- Anomaly flags require investigation before records are removed.
- Missing and nonnumeric values should be handled before fitting.
- For model evaluation, avoid using held-out test data to learn anomaly-detection rules that influence the training dataset.

For time-series data, consider seasonal patterns, trends, and temporal context. A value that is normal in one period may be abnormal in another.

---

## 🧹 Methods for Handling Outliers

After identifying potential outliers, choose a treatment based on the cause and analysis objective.

### 1. Correct data errors

If the source confirms that a value is incorrect, correct it using a reliable source or documented rule.

Example:

```python
df.loc[df["Employee"] == "Amit", "Age"] = 24
```

This example is illustrative: only make such a correction if the correct age is independently verified.

### 2. Remove invalid observations

Remove records only when a clear rule or reliable investigation justifies doing so.

```python
valid_df = df.loc[df["Age"].between(0, 120)].copy()
```

This removes rows that violate the example's age constraints. It should not be used as a universal outlier-removal rule.

### 3. Remove statistically flagged observations

If removal is justified by the project requirements, use an explicit mask.

```python
salary_outliers = detect_iqr_outliers(df["Salary"])

review_df = df.loc[salary_outliers].copy()
retained_df = df.loc[~salary_outliers].copy()
```

Inspect `review_df` before treating its contents as errors. A statistically unusual record can still be valid and important.

### 4. Cap extreme values

Capping limits values to selected lower and upper thresholds while retaining the records.

```python
lower_limit = df["Salary"].quantile(0.01)
upper_limit = df["Salary"].quantile(0.99)

df["Salary_Capped"] = df["Salary"].clip(
    lower=lower_limit,
    upper=upper_limit
)
```

The 1st and 99th percentiles are examples. Choose thresholds based on the application and estimate them from training data when building a machine learning model.

### 5. Transform the distribution

A logarithmic transform can reduce right skew in positive variables.

```python
import numpy as np

df["Salary_Log"] = np.log1p(df["Salary"])
```

`np.log1p(x)` computes \(\log(1+x)\). It is suitable for nonnegative values and may reduce the influence of very large positive numbers.

A transformation changes the scale and distribution; it does not remove an observation. Log transformations can also be unsuitable when values have particular domain meanings or include negative numbers.

### 6. Retain the observations

If the values are valid and meaningful, keeping them may be the best choice.

Examples include:

- High-value financial transactions in fraud detection.
- Rare medical observations.
- Unusually high property prices in luxury markets.
- Exceptional performance measurements.

### Strategy comparison

| Strategy | When to consider it | Main risk |
|---|---|---|
| Correct | A confirmed data error exists | Incorrectly replacing a valid observation |
| Remove | Records are invalid or exclusion is justified | Losing important information |
| Cap | Extremes distort a model or metric | Hiding meaningful magnitude |
| Transform | A skewed distribution needs a different scale | Reduced interpretability |
| Robust methods | Model is sensitive to extreme values | May not solve data-quality issues |
| Retain | Values are valid or anomalies matter | Some algorithms may be sensitive to extremes |

---

## ✂️ Winsorization and Capping

**Winsorization** replaces values outside chosen cutoffs with the cutoff values rather than removing the observations.

For example, if the lower and upper limits are 10 and 100:

```text
Original:     5, 12, 20, 40, 150
Winsorized:  10, 12, 20, 40, 100
```

### Manual quantile capping

```python
lower_limit = df["Salary"].quantile(0.05)
upper_limit = df["Salary"].quantile(0.95)

df["Salary_Winsorized"] = df["Salary"].clip(
    lower=lower_limit,
    upper=upper_limit
)
```

The 5th and 95th percentiles are illustrative thresholds, not universal settings.

### Important considerations

- Keep the original feature if traceability is important.
- Document the capping thresholds.
- Validate that capping makes sense for the feature.
- Estimate learned thresholds from the training data only.
- Evaluate how capping affects downstream performance and interpretation.

For some applications, clipping values to domain-defined limits is more meaningful than choosing quantile cutoffs.

---

## 🔄 Log and Other Transformations

Skewed features can make some models and visualizations harder to interpret.

A logarithmic transform is often considered for nonnegative, right-skewed data:

```python
import numpy as np

df["Salary_Log"] = np.log1p(df["Salary"])
```

Other approaches include square-root transformations and power transformations. The right choice depends on the distribution, the feature's meaning, and the model.

### Compare distributions

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df["Salary"].dropna(), bins=10, edgecolor="black")
axes[0].set_title("Original Salary")
axes[0].set_xlabel("Salary")

log_salary = np.log1p(df.loc[df["Salary"] >= 0, "Salary"])
axes[1].hist(log_salary.dropna(), bins=10, edgecolor="black")
axes[1].set_title("Log-Transformed Salary")
axes[1].set_xlabel("log(1 + Salary)")

plt.tight_layout()
plt.show()
```

Use transformations carefully. The log scale can reduce skew, but it does not guarantee a normal distribution and may not be necessary for all algorithms.

---

## 📊 Compare Data Before and After Handling

Measure the impact of any transformation instead of assuming it improved the dataset.

```python
comparison = pd.DataFrame({
    "Before": df["Salary"],
    "After_Capping": df["Salary"].clip(
        lower=df["Salary"].quantile(0.05),
        upper=df["Salary"].quantile(0.95)
    )
})

print(comparison.describe())
```

Compare the mean and median:

```python
print("Original mean:", df["Salary"].mean())
print("Original median:", df["Salary"].median())

capped_salary = comparison["After_Capping"]

print("Capped mean:", capped_salary.mean())
print("Capped median:", capped_salary.median())
```

Useful checks include:

- Number of observations flagged.
- Number of records removed, if any.
- Missing values introduced.
- Mean, median, range, and spread before and after.
- Whether legitimate extreme observations were affected.
- Model metrics on appropriate validation data.

Outlier handling should be considered successful only when it improves the suitability of the data for the intended task without creating unacceptable distortions.

## 🛠️ Complete Outlier Detection Program

Save this example as `outliers.py`. It creates a sample dataset, detects potential outliers, reports them, applies domain validation, and exports a reviewable file.

```python
import pandas as pd


def detect_iqr_outliers(
    series: pd.Series,
    multiplier: float = 1.5
) -> pd.Series:
    """Identify potential outliers using the IQR rule."""

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() == 0:
        return pd.Series(False, index=series.index, dtype=bool)

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - multiplier * iqr
    upper_fence = q3 + multiplier * iqr

    return (
        numeric.lt(lower_fence)
        | numeric.gt(upper_fence)
    ).fillna(False).astype(bool)


def build_outlier_report(
    df: pd.DataFrame,
    columns: list[str]
) -> pd.DataFrame:
    """Return a report with outlier flags for selected numeric columns."""

    report = df.copy()

    for column in columns:
        if column not in report.columns:
            raise ValueError(f"Column not found: {column}")

        report[f"{column}_IQR_Outlier"] = detect_iqr_outliers(
            report[column]
        )

    return report


def main() -> None:
    data = pd.DataFrame({
        "Employee": [
            "Amit", "Priya", "Rahul", "Neha", "Kiran",
            "Meera", "Arjun", "Sneha", "Rohan", "Anaya"
        ],
        "Age": [24, 28, 31, 29, 35, 27, 32, 30, 26, 250],
        "Salary": [
            35000, 42000, 48000, 45000, 52000,
            40000, 49000, 46000, 43000, 500000
        ],
        "Experience": [1, 3, 6, 4, 9, 2, 7, 5, 2, 4]
    })

    numeric_columns = ["Age", "Salary", "Experience"]

    print("ORIGINAL DATA")
    print(data)

    print("\nDESCRIPTIVE STATISTICS")
    print(data[numeric_columns].describe())

    # Detect potential statistical outliers.
    report = build_outlier_report(data, numeric_columns)

    print("\nOUTLIER REPORT")
    flag_columns = [
        f"{column}_IQR_Outlier"
        for column in numeric_columns
    ]
    print(report[["Employee", *flag_columns]])

    for column in numeric_columns:
        flag_column = f"{column}_IQR_Outlier"
        flagged = report.loc[report[flag_column]]

        print(f"\nPotential outliers in {column}:")
        print(flagged[["Employee", column]])

    # Apply an illustrative domain rule for human age.
    invalid_age = ~data["Age"].between(0, 120)

    print("\nAGES OUTSIDE THE ALLOWED RANGE")
    print(data.loc[invalid_age, ["Employee", "Age"]])

    # Save the complete review report without deleting records.
    report.to_csv("outlier_report.csv", index=False)

    print("\nSaved outlier_report.csv")


if __name__ == "__main__":
    main()
```

### What this program demonstrates

- Creating a reproducible sample dataset.
- Calculating potential outliers using the IQR method.
- Creating a separate flag for every selected numerical feature.
- Inspecting flagged records instead of automatically deleting them.
- Applying a domain-specific age check.
- Exporting the original observations with outlier flags.

The result is an **outlier review report**, not a guarantee that flagged values are errors. Investigate the observations and define the appropriate treatment for the intended application.

---

## 🤖 Outliers in Machine Learning

Outliers affect algorithms differently. Their impact depends on the model, the data distribution, feature scaling, and the objective.

| Algorithm or approach | Potential effect of outliers |
|---|---|
| Linear regression | Extreme observations can strongly influence fitted coefficients |
| Logistic regression | Extreme features can affect the fitted decision boundary |
| K-nearest neighbors | Extreme feature values can influence distances |
| K-means clustering | Extreme points can pull cluster centers |
| Decision trees | Often less sensitive to feature magnitude, but unusual observations can still affect splits |
| Tree ensembles | May be more robust to some extreme feature values, but are not immune to bad data |
| Neural networks | Extreme values can destabilize optimization or dominate feature scales |
| Anomaly detection | Unusual observations may be the main signal of interest |

### Prevent data leakage

When using outlier handling as part of a machine learning workflow:

1. Split the data into training and evaluation partitions.
2. Estimate statistical thresholds from the training data.
3. Apply the same thresholds or fitted preprocessing rules to validation, test, and inference data when appropriate.
4. Do not use evaluation-set outcomes to decide which training observations to remove.
5. Compare candidate preprocessing strategies using validation data.
6. Keep legitimate anomaly cases when they are relevant to the prediction task.

A deterministic business rule, such as rejecting an impossible negative age, can usually be implemented consistently as a data validation rule. A threshold inferred from the sample distribution, such as a training-set percentile, should be estimated without using held-out data.

### Example: Use Robust Scaling

Robust scaling uses the median and interquartile range rather than the mean and standard deviation.

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

scaled_values = scaler.fit_transform(
    df[["Age", "Salary", "Experience"]]
)

print(scaled_values)
```

This snippet demonstrates the scaler. In an evaluated machine learning workflow, fit it on the training features only and apply the fitted scaler to the remaining partitions.

Robust scaling reduces the effect of extreme feature magnitudes on the scale, but it does **not** remove outliers or repair invalid records.

### Choose the treatment for the objective

For a salary prediction model, it may be appropriate to investigate invalid salaries, apply a justified transformation, or use a robust model.

For fraud detection, unusually large transactions may be central to the problem, so removing them could destroy the signal.

Always connect the handling strategy to the prediction objective and the meaning of the observations.

## ❌ Common Mistakes

1. **Deleting every flagged value.** Statistical outliers may be genuine observations.
2. **Using one threshold for every feature.** Distributions and business rules differ.
3. **Ignoring domain knowledge.** Valid value ranges can be more informative than statistical rules.
4. **Using Z-scores for highly skewed data without investigation.** The mean and standard deviation are sensitive to extremes.
5. **Assuming every outlier is a data entry error.** Some represent rare but valuable events.
6. **Fitting thresholds using all data before evaluation.** This can leak information from held-out data.
7. **Ignoring multivariate relationships.** Individual features may appear normal while their combination is unusual.
8. **Capping values without preserving evidence.** The original value may be important for auditing.
9. **Forgetting missing values and invalid numeric conversions.** These need separate checks.
10. **Treating anomaly detection as proof of fraud or error.** A flagged observation requires additional evidence.

## 🌟 Best Practices

- Explore distributions before selecting a detection method.
- Preserve the original dataset and maintain an audit trail.
- Distinguish impossible values from statistically unusual ones.
- Use domain rules wherever reliable constraints exist.
- Compare IQR and Z-score methods when appropriate.
- Consider robust statistical methods for skewed distributions.
- Inspect flagged observations in their business context.
- Document thresholds and reasons for each treatment.
- Avoid removing important rare events.
- Calculate learned thresholds from training data only.
- Compare evaluation metrics before and after preprocessing.
- Validate the resulting data and model behavior.
- Monitor newly arriving data for shifts in outlier frequency and distribution.

## 💡 Mini Projects

### Project 1: Student Marks Outlier Detection

Create a student dataset with marks, attendance, and study hours.

**Tasks:**

1. Calculate descriptive statistics.
2. Detect potential outliers with IQR.
3. Compare IQR with Z-score detection.
4. Check marks and attendance against their known ranges.
5. Visualize the distributions.
6. Produce a report of flagged observations.

### Project 2: Salary Data Analysis

Use an employee salary dataset.

**Tasks:**

1. Visualize the salary distribution.
2. Calculate IQR fences.
3. Investigate extreme salary values.
4. Compare the original data with a capped version.
5. Examine the effect on the mean and median.
6. Explain which observations should be retained.

### Project 3: E-commerce Transaction Analysis

Create a transaction dataset containing transaction amount, item count, and customer frequency.

**Tasks:**

1. Detect unusually large transaction amounts.
2. Identify impossible or invalid values.
3. Explore multivariate anomaly detection.
4. Compare suspicious records with ordinary observations.
5. Explain why flagged records must be reviewed before removal.

### Project 4: Outlier Handling for Regression

Build a regression dataset that contains a few extreme feature values.

**Tasks:**

1. Establish a train/test split.
2. Compare a baseline model with a model using justified preprocessing.
3. Evaluate both on held-out test data.
4. Document threshold calculations.
5. Explain the impact of outlier handling on accuracy and interpretation.

## 📝 Practice Exercises

1. Create a numerical Series containing one extreme observation.
2. Calculate \(Q_1\), \(Q_3\), and IQR manually.
3. Detect outliers using the 1.5 × IQR rule.
4. Calculate Z-scores and flag observations beyond a threshold.
5. Explain the difference between Z-score and Modified Z-score methods.
6. Write a reusable IQR detection function.
7. Detect ages outside an allowed range using domain rules.
8. Cap values at selected percentiles and compare the summary statistics.
9. Apply a log transformation to positive skewed data.
10. Explain how removing valid anomalies could harm a fraud-detection model.
11. Build an outlier report with flags for multiple numeric columns.
12. Describe how to prevent leakage when outlier thresholds are estimated from training data.

## ⚡ Quick Revision

| Task | Method |
|---|---|
| Inspect distributions | `series.describe()` |
| Calculate first quartile | `series.quantile(0.25)` |
| Calculate third quartile | `series.quantile(0.75)` |
| Calculate IQR | `Q3 - Q1` |
| Detect IQR outliers | Compare values against IQR fences |
| Calculate Z-scores | `(x - mean) / standard_deviation` |
| Calculate Modified Z-scores | Median and MAD |
| Apply domain validation | Boolean conditions and `.between()` |
| Cap values | `series.clip(lower, upper)` |
| Log-transform nonnegative values | `numpy.log1p(series)` |
| Robust scaling | `sklearn.preprocessing.RobustScaler` |
| Multivariate anomaly detection | Isolation Forest and similar methods |
| Export outlier report | `df.to_csv("outlier_report.csv", index=False)` |

## 📂 Project Structure

Recommended repository structure:

```text
05-Data-Cleaning/
│
├── README.md
│
├── 01-Missing-Values/
│   └── README.md
│
├── 02-Duplicates/
│   └── README.md
│
├── 03-Incorrect-Data-Types/
│   └── README.md
│
├── 04-Inconsistent-Values/
│   └── README.md
│
└── 05-Outliers/
    ├── README.md
    ├── outliers.py
    ├── employee_data.csv
    └── outlier_report.csv
```

| File | Purpose |
|---|---|
| `README.md` | Concepts, methods, examples, and workflow |
| `outliers.py` | Detection, validation, and reporting code |
| `employee_data.csv` | Sample input dataset |
| `outlier_report.csv` | Original records with outlier flags |

Generated output files are optional. Include them when they help demonstrate reproducible results. Do not commit confidential or sensitive personal information to a public repository.

## 🗺️ Learning Roadmap

Follow this sequence to build practical outlier-handling skills:

1. Understand the dataset and its domain constraints.
2. Inspect summary statistics and visualize distributions.
3. Detect potential outliers with IQR.
4. Learn Z-score and Modified Z-score methods.
5. Distinguish invalid data from valid extreme observations.
6. Explore multivariate and contextual anomalies.
7. Evaluate correction, removal, capping, transformation, and retention strategies.
8. Build reusable detection and reporting functions.
9. Integrate justified preprocessing into a machine learning pipeline.
10. Evaluate the impact and monitor data quality over time.

## 🎯 Key Takeaways

- An outlier is an unusual observation, not necessarily an error.
- The IQR method detects observations beyond quartile-based fences.
- Z-scores measure distance from the mean in standard deviation units.
- Modified Z-scores use the median and MAD for more robust detection.
- Domain rules are essential for identifying invalid values.
- Multivariate techniques can reveal unusual combinations of features.
- Outliers can be corrected, removed, capped, transformed, or retained according to the use case.
- Learned preprocessing thresholds must be handled carefully to avoid data leakage.
- Every outlier treatment should be validated and documented.

---

## 👨‍💻 Author

**Kishor Patil**

Python | Data Science | Machine Learning | Artificial Intelligence

GitHub: [Kishor055](https://github.com/Kishor055)

Repository: [Machine-Learning](https://github.com/Kishor055/Machine-Learning)

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Add improvements, examples, or corrections.
4. Test the code.
5. Submit a pull request with a clear description.

## ⭐ Support

If this guide helps you learn data cleaning and machine learning, consider starring the repository on GitHub.

**Keep learning, keep building, and keep growing!** 🌱

---

*Part of the Machine Learning learning path: Data Collection → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Model Training → Evaluation → Deployment.*
