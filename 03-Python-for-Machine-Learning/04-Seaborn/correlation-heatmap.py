A professional beginner-to-advanced tutorial for understanding,
calculating, visualizing, and interpreting feature correlations
using Pandas, NumPy, Matplotlib, and Seaborn.

Topics covered
Correlation fundamentals
Pearson correlation
Spearman correlation
Pandas correlation matrices
Basic heatmaps
Annotated heatmaps
Heatmap formatting
Color maps
Centering around zero
Lower-triangle heatmaps
Upper-triangle heatmaps
Masking the diagonal
Large correlation matrices
Feature-target correlation
Multicollinearity
Highly correlated feature pairs
Duplicate/redundant features
Correlation and feature selection
Classification datasets
Regression datasets
Correlation limitations
Missing values
Constant features
Numeric-only feature selection
Reusable heatmap functions
Professional ML correlation workflow
Saving high-resolution figures
Requirements

pip install numpy pandas matplotlib seaborn scikit-learn

Author

Kishor Patil
"""

=============================================================================
01. IMPORTS
=============================================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import (
load_diabetes,
load_wine,
make_regression
)

=============================================================================
02. GLOBAL CONFIGURATION
=============================================================================

sns.set_theme(
style="whitegrid",
context="notebook"
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

def section(title: str) -> None:
"""Print a formatted section heading."""
print("\n" + "=" * 80)
print(title)
print("=" * 80)

=============================================================================
03. CREATE A SAMPLE ML DATASET
=============================================================================

def create_sample_dataset(
random_state: int = 42
) -> pd.DataFrame:
"""
Create a synthetic dataset containing correlated ML features.

Parameters
----------
random_state:
    Seed used for reproducibility.

Returns
-------
pd.DataFrame
    Synthetic numerical dataset.
"""

rng = np.random.default_rng(random_state)

age = rng.normal(
    loc=35,
    scale=10,
    size=300
)

experience = (
    age * 0.65
    + rng.normal(0, 3, 300)
)

salary = (
    experience * 4500
    + age * 1000
    + rng.normal(0, 12000, 300)
)

spending = (
    salary * 0.015
    + rng.normal(0, 150, 300)
)

satisfaction = (
    3.5
    + rng.normal(0, 0.6, 300)
)

performance = (
    experience * 0.8
    + satisfaction * 8
    + rng.normal(0, 8, 300)
)

return pd.DataFrame({
    "age": age,
    "experience": experience,
    "salary": salary,
    "spending": spending,
    "satisfaction": satisfaction,
    "performance": performance,
})
=============================================================================
04. CORRELATION FUNDAMENTALS
=============================================================================

def explain_correlation() -> None:
"""
Print a conceptual overview of correlation.

Pearson correlation ranges from -1 to +1.

+1  -> perfect positive linear relationship
 0  -> no linear relationship
-1  -> perfect negative linear relationship
"""

section("04 - Correlation Fundamentals")

print(
    """

Correlation measures the strength and direction of association
between two variables.

Typical interpretation:

+1.00  Perfect positive linear association
+0.70  Strong positive association
+0.40  Moderate positive association
 0.00  No linear association
-0.40  Moderate negative association
-0.70  Strong negative association
-1.00  Perfect negative linear association

Important:
Correlation does NOT prove causation.
"""
)

=============================================================================
05. PEARSON CORRELATION
=============================================================================

def pearson_example(df: pd.DataFrame) -> None:
"""
Calculate Pearson correlation.

Pearson correlation measures linear association between
two numerical variables.
"""

section("05 - Pearson Correlation")

correlation = df["age"].corr(
    df["experience"],
    method="pearson"
)

print(
    f"Pearson correlation between age and experience: "
    f"{correlation:.4f}"
)
=============================================================================
06. SPEARMAN CORRELATION
=============================================================================

def spearman_example(df: pd.DataFrame) -> None:
"""
Calculate Spearman rank correlation.

Spearman correlation evaluates monotonic relationships and
is based on ranked values rather than raw values.
"""

section("06 - Spearman Correlation")

correlation = df["salary"].corr(
    df["experience"],
    method="spearman"
)

print(
    f"Spearman correlation between salary and experience: "
    f"{correlation:.4f}"
)
=============================================================================
07. FULL PEARSON CORRELATION MATRIX
=============================================================================

def pearson_correlation_matrix(
df: pd.DataFrame
) -> pd.DataFrame:
"""Calculate and display the complete Pearson matrix."""

section("07 - Pearson Correlation Matrix")

correlation = df.corr(
    method="pearson",
    numeric_only=True
)

print("\nPearson correlation matrix:")
print(correlation.round(3))

return correlation
=============================================================================
08. SPEARMAN CORRELATION MATRIX
=============================================================================

def spearman_correlation_matrix(
df: pd.DataFrame
) -> pd.DataFrame:
"""Calculate and display the complete Spearman matrix."""

section("08 - Spearman Correlation Matrix")

correlation = df.corr(
    method="spearman",
    numeric_only=True
)

print("\nSpearman correlation matrix:")
print(correlation.round(3))

return correlation
=============================================================================
09. BASIC HEATMAP
=============================================================================

def basic_heatmap(
correlation: pd.DataFrame
) -> None:
"""Create a basic correlation heatmap."""

section("09 - Basic Correlation Heatmap")

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    correlation
)

plt.title(
    "Correlation Heatmap"
)

plt.tight_layout()
plt.show()
=============================================================================
10. ANNOTATED HEATMAP
=============================================================================

def annotated_heatmap(
correlation: pd.DataFrame
) -> None:
"""
Display correlation coefficients directly inside cells.
"""

section("10 - Annotated Correlation Heatmap")

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title(
    "Annotated Correlation Matrix"
)

plt.tight_layout()
plt.show()
=============================================================================
11. PROFESSIONAL HEATMAP
=============================================================================

def professional_heatmap(
correlation: pd.DataFrame
) -> None:
"""Create a presentation-ready correlation heatmap."""

section("11 - Professional Heatmap")

plt.figure(
    figsize=(12, 9)
)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    square=True,
    cbar_kws={
        "label": "Correlation"
    }
)

plt.title(
    "Feature Correlation Matrix",
    fontsize=16
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(
    rotation=0
)

plt.tight_layout()
plt.show()
=============================================================================
12. DIVERGING COLOR MAP
=============================================================================

def diverging_colormap(
correlation: pd.DataFrame
) -> None:
"""
Use a diverging color map centered at zero.

This is especially useful because positive and negative
correlations need to be visually distinguished.
"""

section("12 - Diverging Color Map")

plt.figure(
    figsize=(11, 8)
)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1
)

plt.title(
    "Correlation: -1 to +1"
)

plt.tight_layout()
plt.show()
=============================================================================
13. LOWER TRIANGLE HEATMAP
=============================================================================

def lower_triangle_heatmap(
correlation: pd.DataFrame
) -> None:
"""
Display only the lower triangle.

Correlation matrices are symmetric, so showing both halves
often duplicates information.
"""

section("13 - Lower Triangle Heatmap")

mask = np.triu(
    np.ones_like(
        correlation,
        dtype=bool
    )
)

plt.figure(
    figsize=(11, 9)
)

sns.heatmap(
    correlation,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True
)

plt.title(
    "Lower Triangle Correlation Heatmap"
)

plt.tight_layout()
plt.show()
=============================================================================
14. UPPER TRIANGLE HEATMAP
=============================================================================

def upper_triangle_heatmap(
correlation: pd.DataFrame
) -> None:
"""Display only the upper triangle."""

section("14 - Upper Triangle Heatmap")

mask = np.tril(
    np.ones_like(
        correlation,
        dtype=bool
    )
)

plt.figure(
    figsize=(11, 9)
)

sns.heatmap(
    correlation,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True
)

plt.title(
    "Upper Triangle Correlation Heatmap"
)

plt.tight_layout()
plt.show()
=============================================================================
15. HIDE DIAGONAL
=============================================================================

def hide_diagonal_heatmap(
correlation: pd.DataFrame
) -> None:
"""
Hide the diagonal because each feature is perfectly correlated
with itself.
"""

section("15 - Heatmap Without Diagonal")

mask = np.eye(
    len(correlation),
    dtype=bool
)

plt.figure(
    figsize=(11, 9)
)

sns.heatmap(
    correlation,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1
)

plt.title(
    "Correlation Heatmap Without Diagonal"
)

plt.tight_layout()
plt.show()
=============================================================================
16. P-VALUE NOTE
=============================================================================

def correlation_interpretation() -> None:
"""
Explain why correlation coefficients should not be interpreted
in isolation.
"""

section("16 - Correlation Interpretation")

print(
    """

When interpreting correlation:

Check the direction.
Check the magnitude.
Inspect the underlying scatter plot.
Consider sample size.
Investigate outliers.
Consider nonlinear relationships.
Consider domain knowledge.
Do not interpret correlation as causation.

Example:

A correlation of +0.85 means the variables have a strong
positive linear association in the observed data.

It does not prove that changing one variable causes the other
to change.
"""
)

=============================================================================
17. SCATTER PLOT + CORRELATION
=============================================================================

def scatter_with_correlation(
df: pd.DataFrame,
x: str,
y: str
) -> None:
"""
Visualize two variables together with their Pearson correlation.

A heatmap summarizes many relationships, while a scatter plot
helps inspect the actual shape of one relationship.
"""

section(
    f"17 - Scatter Plot: {x} vs {y}"
)

correlation = df[x].corr(
    df[y],
    method="pearson"
)

plt.figure(
    figsize=(9, 6)
)

sns.regplot(
    data=df,
    x=x,
    y=y,
    scatter_kws={
        "alpha": 0.6
    },
    line_kws={
        "linewidth": 2
    }
)

plt.title(
    f"{x.title()} vs {y.title()} "
    f"(Pearson r = {correlation:.2f})"
)

plt.tight_layout()
plt.show()
=============================================================================
18. TARGET CORRELATION
=============================================================================

def target_correlation(
df: pd.DataFrame,
target: str
) -> pd.Series:
"""
Calculate correlations between numerical features and a target.

Parameters
----------
df:
    Input DataFrame.

target:
    Numerical target column.

Returns
-------
pd.Series
    Feature-target correlations sorted by absolute magnitude.
"""

section("18 - Feature vs Target Correlation")

if target not in df.columns:
    raise ValueError(
        f"Target column '{target}' was not found."
    )

numeric_df = df.select_dtypes(
    include="number"
)

if target not in numeric_df.columns:
    raise ValueError(
        "Target must be numerical for Pearson correlation."
    )

correlations = (
    numeric_df
    .corr()[target]
    .drop(target)
    .sort_values(
        key=lambda values: values.abs(),
        ascending=False
    )
)

print("\nFeature-target correlations:")
print(correlations.round(4))

return correlations
=============================================================================
19. FEATURE-TARGET HEATMAP
=============================================================================

def feature_target_heatmap(
df: pd.DataFrame,
target: str
) -> None:
"""Visualize feature-target correlations."""

section("19 - Feature-Target Heatmap")

numeric_df = df.select_dtypes(
    include="number"
)

if target not in numeric_df.columns:
    raise ValueError(
        f"'{target}' must be a numerical column."
    )

correlations = (
    numeric_df
    .corr()[[target]]
    .sort_values(
        by=target,
        ascending=False
    )
)

plt.figure(
    figsize=(6, 9)
)

sns.heatmap(
    correlations,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1
)

plt.title(
    f"Feature Correlation with {target.title()}"
)

plt.tight_layout()
plt.show()
=============================================================================
20. FIND HIGHLY CORRELATED PAIRS
=============================================================================

def find_highly_correlated_pairs(
correlation: pd.DataFrame,
threshold: float = 0.80
) -> pd.DataFrame:
"""
Find unique feature pairs whose absolute correlation exceeds
a specified threshold.

Parameters
----------
correlation:
    Correlation matrix.

threshold:
    Absolute correlation threshold.

Returns
-------
pd.DataFrame
    Feature pairs and correlation values.
"""

section(
    f"20 - Highly Correlated Pairs "
    f"(Threshold = {threshold:.2f})"
)

if not 0 < threshold <= 1:
    raise ValueError(
        "threshold must be between 0 and 1."
    )

pairs = []

columns = correlation.columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):

        value = correlation.iloc[i, j]

        if abs(value) >= threshold:
            pairs.append({
                "feature_1": columns[i],
                "feature_2": columns[j],
                "correlation": value,
                "absolute_correlation": abs(value)
            })

result = pd.DataFrame(pairs)

if result.empty:
    print(
        "No feature pairs exceeded the threshold."
    )
    return result

result = result.sort_values(
    "absolute_correlation",
    ascending=False
).reset_index(drop=True)

print("\nHighly correlated feature pairs:")
print(result.round(4))

return result
=============================================================================
21. MULTICOLLINEARITY ANALYSIS
=============================================================================

def multicollinearity_analysis(
correlation: pd.DataFrame,
threshold: float = 0.80
) -> None:
"""
Explain potential multicollinearity using feature correlations.

High pairwise correlation can be a warning sign for redundant
information, particularly in some linear models.
"""

section("21 - Multicollinearity Analysis")

pairs = find_highly_correlated_pairs(
    correlation,
    threshold=threshold
)

if pairs.empty:
    print(
        "\nNo high pairwise correlations were detected."
    )
else:
    print(
        """

Potential multicollinearity detected.

Important:
High correlation does not automatically mean that a feature
must be removed.

Feature-selection decisions should consider:

Model type
Domain meaning
Interpretability
Validation performance
Regularization
Feature importance
Data collection process
"""
)
=============================================================================
22. DROP REDUNDANT FEATURES EXAMPLE
=============================================================================

def identify_redundant_features(
correlation: pd.DataFrame,
threshold: float = 0.90
) -> list[str]:
"""
Identify a simple candidate list of redundant features.

This is a heuristic only. It should not automatically be used
as a final feature-selection decision.
"""

section(
    f"22 - Candidate Redundant Features "
    f"(Threshold = {threshold:.2f})"
)

upper = correlation.where(
    np.triu(
        np.ones(
            correlation.shape,
            dtype=bool
        ),
        k=1
    )
)

to_drop = [
    column
    for column in upper.columns
    if any(
        abs(upper[column]) > threshold
    )
]

print(
    "\nCandidate features based on the simple heuristic:"
)

print(to_drop)

print(
    "\nThese are candidates only. "
    "Do not automatically remove them without validation."
)

return to_drop
=============================================================================
23. DIABETES REGRESSION DATASET
=============================================================================

def diabetes_dataset_heatmap() -> None:
"""
Analyze feature correlations in the scikit-learn diabetes
regression dataset.
"""

section("23 - Diabetes Regression Dataset")

dataset = load_diabetes(
    as_frame=True
)

df = dataset.frame

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

correlation = df.corr(
    numeric_only=True
)

plt.figure(
    figsize=(11, 9)
)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True
)

plt.title(
    "Diabetes Dataset Correlation Matrix"
)

plt.tight_layout()
plt.show()
=============================================================================
24. WINE DATASET
=============================================================================

def wine_dataset_heatmap() -> None:
"""
Analyze correlations in the Wine dataset.

The target class is excluded from correlation analysis here
because it is categorical.
"""

section("24 - Wine Dataset")

dataset = load_wine(
    as_frame=True
)

df = dataset.frame

feature_columns = [
    column
    for column in df.columns
    if column != "target"
]

correlation = df[
    feature_columns
].corr()

plt.figure(
    figsize=(14, 11)
)

mask = np.triu(
    np.ones_like(
        correlation,
        dtype=bool
    )
)

sns.heatmap(
    correlation,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True
)

plt.title(
    "Wine Feature Correlation Matrix"
)

plt.tight_layout()
plt.show()
=============================================================================
25. REGRESSION DATASET
=============================================================================

def regression_dataset_heatmap() -> None:
"""
Create a synthetic regression dataset and visualize its
feature/target correlations.
"""

section("25 - Synthetic Regression Dataset")

x, y = make_regression(
    n_samples=300,
    n_features=6,
    n_informative=4,
    noise=15,
    random_state=42
)

feature_names = [
    f"feature_{i + 1}"
    for i in range(x.shape[1])
]

df = pd.DataFrame(
    x,
    columns=feature_names
)

df["target"] = y

correlation = df.corr()

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1
)

plt.title(
    "Synthetic Regression Correlation Matrix"
)

plt.tight_layout()
plt.show()
=============================================================================
26. MISSING VALUES
=============================================================================

def correlation_with_missing_values() -> None:
"""
Demonstrate how Pandas handles missing values during
correlation calculation.
"""

section("26 - Correlation with Missing Values")

df = pd.DataFrame({
    "feature_a": [
        10, 20, 30, 40, 50
    ],
    "feature_b": [
        12, 18, np.nan, 41, 55
    ],
    "feature_c": [
        8, 17, 32, np.nan, 52
    ]
})

print("\nDataset:")
print(df)

correlation = df.corr()

print("\nCorrelation matrix:")
print(correlation.round(3))

print(
    """

Pandas generally calculates pairwise correlations using
available non-missing observations.

Always inspect missingness and sample counts before drawing
strong conclusions from correlations.
"""
)

=============================================================================
27. CONSTANT FEATURES
=============================================================================

def constant_feature_example() -> None:
"""
Demonstrate a constant feature.

A feature with no variance cannot provide meaningful
correlation information.
"""

section("27 - Constant Feature")

df = pd.DataFrame({
    "feature_a": [1, 2, 3, 4, 5],
    "feature_b": [2, 4, 6, 8, 10],
    "constant_feature": [7, 7, 7, 7, 7]
})

print("\nStandard deviations:")
print(df.std())

print("\nCorrelation matrix:")
print(df.corr())
=============================================================================
28. CORRELATION OF ABSOLUTE VALUES
=============================================================================

def absolute_correlation_heatmap(
correlation: pd.DataFrame
) -> None:
"""
Display absolute correlation magnitude.

This removes direction and focuses only on strength.
"""

section("28 - Absolute Correlation Heatmap")

absolute_correlation = correlation.abs()

plt.figure(
    figsize=(11, 9)
)

sns.heatmap(
    absolute_correlation,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    vmin=0,
    vmax=1,
    square=True
)

plt.title(
    "Absolute Correlation Magnitude"
)

plt.tight_layout()
plt.show()
=============================================================================
29. REUSABLE CORRELATION HEATMAP
=============================================================================

def plot_correlation_heatmap(
df: pd.DataFrame,
method: str = "pearson",
figsize: tuple[int, int] = (12, 9),
title: str = "Correlation Heatmap",
mask_upper: bool = True,
annot: bool = True,
decimals: int = 2,
) -> pd.DataFrame:
"""
Create a reusable correlation heatmap.

Parameters
----------
df:
    Input DataFrame.

method:
    Correlation method: "pearson", "spearman", or "kendall".

figsize:
    Matplotlib figure size.

title:
    Chart title.

mask_upper:
    Hide the upper triangle.

annot:
    Display correlation values.

decimals:
    Number of decimal places.

Returns
-------
pd.DataFrame
    Calculated correlation matrix.
"""

valid_methods = {
    "pearson",
    "spearman",
    "kendall"
}

if method not in valid_methods:
    raise ValueError(
        f"method must be one of {valid_methods}"
    )

numeric_df = df.select_dtypes(
    include="number"
)

if numeric_df.empty:
    raise ValueError(
        "No numerical columns found."
    )

correlation = numeric_df.corr(
    method=method
)

mask = None

if mask_upper:
    mask = np.triu(
        np.ones_like(
            correlation,
            dtype=bool
        )
    )

plt.figure(
    figsize=figsize
)

sns.heatmap(
    correlation,
    mask=mask,
    annot=annot,
    fmt=f".{decimals}f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={
        "label": "Correlation"
    }
)

plt.title(
    f"{title} ({method.title()})"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(
    rotation=0
)

plt.tight_layout()
plt.show()

return correlation
=============================================================================
30. FEATURE SELECTION WORKFLOW
=============================================================================

def feature_selection_workflow(
df: pd.DataFrame,
target: str
) -> None:
"""
Demonstrate a correlation-based feature investigation workflow.

Correlation is used as an exploratory signal, not as an
automatic feature-selection algorithm.
"""

section("30 - Feature Selection Workflow")

numeric_df = df.select_dtypes(
    include="number"
)

if target not in numeric_df.columns:
    raise ValueError(
        f"'{target}' must be numerical."
    )

correlation = numeric_df.corr()

target_corr = (
    correlation[target]
    .drop(target)
    .sort_values(
        key=lambda values: values.abs(),
        ascending=False
    )
)

print("\nFeatures ordered by absolute target correlation:")
print(target_corr.round(4))

print(
    """

Next steps should include:

Investigate strong correlations.
Inspect scatter plots.
Check missing values.
Check outliers.
Consider domain meaning.
Check multicollinearity.
Train candidate models.
Validate on held-out data.

Do not select features solely because their correlation is high.
"""
)

=============================================================================
31. SAVE HEATMAP
=============================================================================

def save_correlation_heatmap(
correlation: pd.DataFrame,
filename: str = "correlation-heatmap.png"
) -> None:
"""
Save a high-resolution correlation heatmap.

Parameters
----------
correlation:
    Correlation matrix.

filename:
    Output image path.
"""

section("31 - Save Correlation Heatmap")

mask = np.triu(
    np.ones_like(
        correlation,
        dtype=bool
    )
)

fig, ax = plt.subplots(
    figsize=(12, 10)
)

sns.heatmap(
    correlation,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    ax=ax
)

ax.set_title(
    "Feature Correlation Heatmap"
)

fig.tight_layout()

fig.savefig(
    filename,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(
    f"Saved correlation heatmap to: {filename}"
)
=============================================================================
32. MAIN
=============================================================================

def main() -> None:
"""Run the correlation heatmap tutorial."""

section("SEABORN CORRELATION HEATMAP")

# -------------------------------------------------------------------------
# Create data
# -------------------------------------------------------------------------

df = create_sample_dataset()

print("\nDataset preview:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

# -------------------------------------------------------------------------
# Correlation fundamentals
# -------------------------------------------------------------------------

explain_correlation()

pearson_example(df)
spearman_example(df)

# -------------------------------------------------------------------------
# Correlation matrices
# -------------------------------------------------------------------------

pearson = pearson_correlation_matrix(df)

spearman_correlation_matrix(df)

# -------------------------------------------------------------------------
# Heatmap variations
# -------------------------------------------------------------------------

basic_heatmap(pearson)
annotated_heatmap(pearson)
professional_heatmap(pearson)
diverging_colormap(pearson)

lower_triangle_heatmap(pearson)
upper_triangle_heatmap(pearson)
hide_diagonal_heatmap(pearson)

# -------------------------------------------------------------------------
# Interpretation
# -------------------------------------------------------------------------

correlation_interpretation()

scatter_with_correlation(
    df,
    x="experience",
    y="salary"
)

# -------------------------------------------------------------------------
# Feature-target analysis
# -------------------------------------------------------------------------

target_correlation(
    df,
    target="performance"
)

feature_target_heatmap(
    df,
    target="performance"
)

# -------------------------------------------------------------------------
# Multicollinearity
# -------------------------------------------------------------------------

find_highly_correlated_pairs(
    pearson,
    threshold=0.80
)

multicollinearity_analysis(
    pearson,
    threshold=0.80
)

identify_redundant_features(
    pearson,
    threshold=0.90
)

# -------------------------------------------------------------------------
# Real datasets
# -------------------------------------------------------------------------

diabetes_dataset_heatmap()
wine_dataset_heatmap()
regression_dataset_heatmap()

# -------------------------------------------------------------------------
# Data-quality cases
# -------------------------------------------------------------------------

correlation_with_missing_values()
constant_feature_example()

# -------------------------------------------------------------------------
# Absolute correlation
# -------------------------------------------------------------------------

absolute_correlation_heatmap(
    pearson
)

# -------------------------------------------------------------------------
# Reusable function
# -------------------------------------------------------------------------

reusable_df = df.copy()

plot_correlation_heatmap(
    reusable_df,
    method="pearson",
    title="Reusable Feature Correlation"
)

# -------------------------------------------------------------------------
# Feature-selection workflow
# -------------------------------------------------------------------------

feature_selection_workflow(
    df,
    target="performance"
)

# -------------------------------------------------------------------------
# Save example
# -------------------------------------------------------------------------

save_correlation_heatmap(
    pearson,
    filename="correlation-heatmap.png"
)

print(
    "\nCorrelation heatmap tutorial completed successfully."
)
=============================================================================
33. ENTRY POINT
=============================================================================

if name == "main":
main()

=============================================================================
QUICK REFERENCE
=============================================================================


Pearson correlation:


df.corr(method="pearson")


Spearman correlation:


df.corr(method="spearman")


Basic heatmap:


sns.heatmap(
df.corr(numeric_only=True)
)


Annotated heatmap:


sns.heatmap(
df.corr(numeric_only=True),
annot=True,
fmt=".2f"
)


Professional heatmap:


sns.heatmap(
correlation,
annot=True,
fmt=".2f",
cmap="coolwarm",
center=0,
vmin=-1,
vmax=1
)


Lower triangle:


mask = np.triu(
np.ones_like(correlation, dtype=bool)
)


sns.heatmap(
correlation,
mask=mask,
annot=True
)


Feature-target correlation:


df.corr(numeric_only=True)["target"]


Sort by absolute correlation:


correlation["target"].sort_values(
key=lambda values: values.abs(),
ascending=False
)


Find highly correlated pairs:


for i in range(len(columns)):
for j in range(i + 1, len(columns)):
...


Save:


plt.savefig(
"correlation-heatmap.png",
dpi=300,
bbox_inches="tight"
)


=============================================================================
PROFESSIONAL CHECKLIST
=============================================================================


Before interpreting a correlation heatmap:


[ ] Select appropriate numerical variables.
[ ] Check missing values.
[ ] Check constant features.
[ ] Decide whether Pearson or Spearman is appropriate.
[ ] Inspect strong correlations with scatter plots.
[ ] Investigate potential outliers.
[ ] Check for multicollinearity.
[ ] Consider domain knowledge.
[ ] Remember that correlation is not causation.
[ ] Avoid making feature-selection decisions from correlation alone.
[ ] Fit preprocessing only on training data in ML pipelines.
[ ] Validate feature-selection decisions on held-out data.


=============================================================================
KEY TAKEAWAYS
=============================================================================


1. A correlation matrix summarizes pairwise numerical relationships.
2. Seaborn heatmaps make correlation matrices easy to interpret.
3. Pearson measures linear association.
4. Spearman measures rank-based monotonic association.
5. Correlation values range from -1 to +1.
6. Positive and negative relationships should be interpreted separately.
7. Heatmaps are useful for exploratory data analysis.
8. High feature-feature correlation may indicate redundant information.
9. Correlation does not prove causation.
10. Strong correlations should be investigated with appropriate plots.
11. Feature selection should use validation and domain knowledge.
12. Correlation analysis is one part of a broader ML preprocessing workflow.


=============================================================================
END OF FILE
=============================================================================
