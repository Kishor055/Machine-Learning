A professional beginner-to-advanced tutorial on categorical data
visualization using Seaborn.

Topics covered
Seaborn setup
Example dataset
Count plots
Bar plots
Horizontal bar plots
Grouped bar plots
Stacked categorical analysis
Point plots
Box plots
Violin plots
Strip plots
Swarm plots
Boxen plots
Catplot
Hue-based analysis
Category ordering
Error bars
Custom styling
Pandas integration
Machine Learning class analysis
Class imbalance visualization
Feature vs target analysis
Categorical EDA workflow
Reusable plotting functions
Saving figures
Professional best practices
Requirements

pip install numpy pandas matplotlib seaborn

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

=============================================================================
02. GLOBAL CONFIGURATION
=============================================================================

sns.set_theme(
style="whitegrid",
context="notebook"
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

def section(title: str) -> None:
"""Print a formatted section heading."""
print("\n" + "=" * 80)
print(title)
print("=" * 80)

=============================================================================
03. CREATE AN EXAMPLE DATASET
=============================================================================

def create_sample_dataset() -> pd.DataFrame:
"""
Create a realistic categorical dataset for visualization examples.

Returns
-------
pd.DataFrame
    Sample customer/order dataset.
"""

rng = np.random.default_rng(42)

n = 200

departments = rng.choice(
    ["Technology", "Finance", "Marketing", "HR"],
    size=n,
    p=[0.35, 0.25, 0.25, 0.15]
)

genders = rng.choice(
    ["Female", "Male", "Other"],
    size=n,
    p=[0.46, 0.50, 0.04]
)

cities = rng.choice(
    ["Mumbai", "Pune", "Delhi", "Bengaluru"],
    size=n
)

membership = rng.choice(
    ["Basic", "Standard", "Premium"],
    size=n,
    p=[0.40, 0.38, 0.22]
)

education = rng.choice(
    ["Bachelor", "Master", "PhD"],
    size=n,
    p=[0.55, 0.35, 0.10]
)

age = rng.integers(20, 61, size=n)

salary = np.round(
    rng.normal(65000, 18000, size=n),
    2
)

salary = np.clip(salary, 25000, 150000)

experience = rng.integers(0, 21, size=n)

purchase_amount = np.round(
    rng.normal(4500, 1800, size=n),
    2
)

purchase_amount = np.clip(
    purchase_amount,
    500,
    15000
)

purchased = rng.choice(
    ["No", "Yes"],
    size=n,
    p=[0.62, 0.38]
)

rating = np.round(
    rng.uniform(1, 5, size=n),
    1
)

return pd.DataFrame({
    "department": departments,
    "gender": genders,
    "city": cities,
    "membership": membership,
    "education": education,
    "age": age,
    "salary": salary,
    "experience": experience,
    "purchase_amount": purchase_amount,
    "purchased": purchased,
    "rating": rating,
})
=============================================================================
04. BASIC COUNT PLOT
=============================================================================

def basic_count_plot(df: pd.DataFrame) -> None:
"""Display the frequency of each department."""

section("04 - Basic Count Plot")

plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="department"
)

plt.title("Number of Employees by Department")
plt.xlabel("Department")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
=============================================================================
05. COUNT PLOT WITH HUE
=============================================================================

def count_plot_with_hue(df: pd.DataFrame) -> None:
"""Compare department counts across genders."""

section("05 - Count Plot with Hue")

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="department",
    hue="gender"
)

plt.title("Department Distribution by Gender")
plt.xlabel("Department")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
=============================================================================
06. HORIZONTAL COUNT PLOT
=============================================================================

def horizontal_count_plot(df: pd.DataFrame) -> None:
"""Display categories horizontally."""

section("06 - Horizontal Count Plot")

plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    y="department"
)

plt.title("Employees by Department")
plt.xlabel("Count")
plt.ylabel("Department")

plt.tight_layout()
plt.show()
=============================================================================
07. BAR PLOT
=============================================================================

def basic_bar_plot(df: pd.DataFrame) -> None:
"""
Display the average salary by department.

Unlike countplot(), barplot() typically displays an estimated
numerical statistic for each category.
"""

section("07 - Bar Plot")

plt.figure(figsize=(9, 6))

sns.barplot(
    data=df,
    x="department",
    y="salary"
)

plt.title("Average Salary by Department")
plt.xlabel("Department")
plt.ylabel("Average Salary")

plt.tight_layout()
plt.show()
=============================================================================
08. BAR PLOT WITH HUE
=============================================================================

def grouped_bar_plot(df: pd.DataFrame) -> None:
"""Compare average salary by department and education level."""

section("08 - Grouped Bar Plot")

plt.figure(figsize=(11, 6))

sns.barplot(
    data=df,
    x="department",
    y="salary",
    hue="education"
)

plt.title("Average Salary by Department and Education")
plt.xlabel("Department")
plt.ylabel("Average Salary")

plt.tight_layout()
plt.show()
=============================================================================
09. HORIZONTAL BAR PLOT
=============================================================================

def horizontal_bar_plot(df: pd.DataFrame) -> None:
"""Create a horizontal categorical comparison."""

section("09 - Horizontal Bar Plot")

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    y="department",
    x="purchase_amount",
    errorbar=None
)

plt.title("Average Purchase Amount by Department")
plt.xlabel("Average Purchase Amount")
plt.ylabel("Department")

plt.tight_layout()
plt.show()
=============================================================================
10. CATEGORY ORDERING
=============================================================================

def ordered_categories(df: pd.DataFrame) -> None:
"""Demonstrate explicit category ordering."""

section("10 - Ordered Categories")

order = [
    "Technology",
    "Finance",
    "Marketing",
    "HR"
]

plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="department",
    order=order
)

plt.title("Departments in a Custom Order")
plt.xlabel("Department")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
=============================================================================
11. ORDER BY FREQUENCY
=============================================================================

def order_by_frequency(df: pd.DataFrame) -> None:
"""Order categories according to their frequency."""

section("11 - Order Categories by Frequency")

order = (
    df["department"]
    .value_counts()
    .index
)

plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="department",
    order=order
)

plt.title("Departments Ordered by Frequency")
plt.xlabel("Department")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
=============================================================================
12. POINT PLOT
=============================================================================

def point_plot(df: pd.DataFrame) -> None:
"""Compare average purchase amounts across departments."""

section("12 - Point Plot")

plt.figure(figsize=(10, 6))

sns.pointplot(
    data=df,
    x="department",
    y="purchase_amount"
)

plt.title("Average Purchase Amount by Department")
plt.xlabel("Department")
plt.ylabel("Purchase Amount")

plt.tight_layout()
plt.show()
=============================================================================
13. POINT PLOT WITH HUE
=============================================================================

def point_plot_with_hue(df: pd.DataFrame) -> None:
"""Compare purchase amounts by department and membership."""

section("13 - Point Plot with Hue")

plt.figure(figsize=(11, 6))

sns.pointplot(
    data=df,
    x="department",
    y="purchase_amount",
    hue="membership"
)

plt.title("Purchase Amount by Department and Membership")
plt.xlabel("Department")
plt.ylabel("Purchase Amount")

plt.tight_layout()
plt.show()
=============================================================================
14. BOX PLOT
=============================================================================

def box_plot(df: pd.DataFrame) -> None:
"""
Display salary distributions across departments.

Box plots are particularly useful for comparing distributions
and identifying potential outliers.
"""

section("14 - Box Plot")

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="department",
    y="salary"
)

plt.title("Salary Distribution by Department")
plt.xlabel("Department")
plt.ylabel("Salary")

plt.tight_layout()
plt.show()
=============================================================================
15. BOX PLOT WITH HUE
=============================================================================

def box_plot_with_hue(df: pd.DataFrame) -> None:
"""Compare salary distributions by department and gender."""

section("15 - Box Plot with Hue")

plt.figure(figsize=(11, 6))

sns.boxplot(
    data=df,
    x="department",
    y="salary",
    hue="gender"
)

plt.title("Salary Distribution by Department and Gender")
plt.xlabel("Department")
plt.ylabel("Salary")

plt.tight_layout()
plt.show()
=============================================================================
16. VIOLIN PLOT
=============================================================================

def violin_plot(df: pd.DataFrame) -> None:
"""
Display the full distribution shape of salary by department.
"""

section("16 - Violin Plot")

plt.figure(figsize=(10, 6))

sns.violinplot(
    data=df,
    x="department",
    y="salary"
)

plt.title("Salary Distribution by Department")
plt.xlabel("Department")
plt.ylabel("Salary")

plt.tight_layout()
plt.show()
=============================================================================
17. VIOLIN PLOT WITH HUE
=============================================================================

def violin_plot_with_hue(df: pd.DataFrame) -> None:
"""Compare salary distributions across department and gender."""

section("17 - Violin Plot with Hue")

plt.figure(figsize=(11, 6))

sns.violinplot(
    data=df,
    x="department",
    y="salary",
    hue="gender",
    split=False
)

plt.title("Salary Distribution by Department and Gender")
plt.xlabel("Department")
plt.ylabel("Salary")

plt.tight_layout()
plt.show()
=============================================================================
18. STRIP PLOT
=============================================================================

def strip_plot(df: pd.DataFrame) -> None:
"""
Display individual observations within categories.
"""

section("18 - Strip Plot")

plt.figure(figsize=(10, 6))

sns.stripplot(
    data=df,
    x="department",
    y="salary",
    jitter=True
)

plt.title("Individual Salary Observations")
plt.xlabel("Department")
plt.ylabel("Salary")

plt.tight_layout()
plt.show()
=============================================================================
19. STRIP PLOT WITH HUE
=============================================================================

def strip_plot_with_hue(df: pd.DataFrame) -> None:
"""Display individual observations grouped by gender."""

section("19 - Strip Plot with Hue")

plt.figure(figsize=(11, 6))

sns.stripplot(
    data=df,
    x="department",
    y="salary",
    hue="gender",
    jitter=True
)

plt.title("Individual Salaries by Department and Gender")
plt.xlabel("Department")
plt.ylabel("Salary")

plt.tight_layout()
plt.show()
=============================================================================
20. SWARM PLOT
=============================================================================

def swarm_plot(df: pd.DataFrame) -> None:
"""
Display individual observations with reduced overlap.

Swarm plots are most appropriate for small or moderate datasets.
"""

section("20 - Swarm Plot")

sample = df.sample(
    n=min(100, len(df)),
    random_state=42
)

plt.figure(figsize=(10, 6))

sns.swarmplot(
    data=sample,
    x="department",
    y="rating"
)

plt.title("Employee Ratings by Department")
plt.xlabel("Department")
plt.ylabel("Rating")

plt.tight_layout()
plt.show()
=============================================================================
21. BOXEN PLOT
=============================================================================

def boxen_plot(df: pd.DataFrame) -> None:
"""
Boxen plots provide additional distribution detail and can
be useful for larger datasets.
"""

section("21 - Boxen Plot")

plt.figure(figsize=(10, 6))

sns.boxenplot(
    data=df,
    x="department",
    y="salary"
)

plt.title("Detailed Salary Distribution by Department")
plt.xlabel("Department")
plt.ylabel("Salary")

plt.tight_layout()
plt.show()
=============================================================================
22. CATPLOT - BOX
=============================================================================

def catplot_box(df: pd.DataFrame) -> None:
"""Use Seaborn's figure-level categorical interface."""

section("22 - Catplot with Box Plot")

sns.catplot(
    data=df,
    x="department",
    y="salary",
    kind="box",
    height=6,
    aspect=1.5
)

plt.title("Salary Distribution by Department")

plt.show()
=============================================================================
23. CATPLOT - VIOLIN
=============================================================================

def catplot_violin(df: pd.DataFrame) -> None:
"""Create a violin plot using catplot."""

section("23 - Catplot with Violin Plot")

sns.catplot(
    data=df,
    x="department",
    y="salary",
    kind="violin",
    height=6,
    aspect=1.5
)

plt.title("Salary Distribution by Department")

plt.show()
=============================================================================
24. CATPLOT - BAR
=============================================================================

def catplot_bar(df: pd.DataFrame) -> None:
"""Create a categorical bar plot using catplot."""

section("24 - Catplot with Bar Plot")

sns.catplot(
    data=df,
    x="department",
    y="purchase_amount",
    kind="bar",
    height=6,
    aspect=1.5
)

plt.title("Average Purchase Amount by Department")

plt.show()
=============================================================================
25. CATPLOT WITH COLUMNS
=============================================================================

def catplot_faceting(df: pd.DataFrame) -> None:
"""
Split a categorical plot into multiple panels.

This is useful when a third categorical variable needs
independent visual comparison.
"""

section("25 - Catplot with Faceting")

sns.catplot(
    data=df,
    x="department",
    y="salary",
    col="gender",
    kind="box",
    height=5,
    aspect=0.9
)

plt.show()
=============================================================================
26. CATEGORY VS CATEGORY
=============================================================================

def category_vs_category(df: pd.DataFrame) -> None:
"""
Analyze relationships between two categorical variables.
"""

section("26 - Category vs Category")

table = pd.crosstab(
    df["department"],
    df["membership"]
)

print("\nCross-tabulation:")
print(table)

table.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Department vs Membership")
plt.xlabel("Department")
plt.ylabel("Count")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()
=============================================================================
27. NORMALIZED CATEGORICAL ANALYSIS
=============================================================================

def normalized_category_analysis(df: pd.DataFrame) -> None:
"""
Compare proportions instead of raw category counts.
"""

section("27 - Normalized Category Analysis")

proportions = pd.crosstab(
    df["department"],
    df["membership"],
    normalize="index"
) * 100

print("\nPercentage distribution:")
print(proportions.round(2))

proportions.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

plt.title("Membership Distribution Within Each Department")
plt.xlabel("Department")
plt.ylabel("Percentage")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()
=============================================================================
28. STACKED CATEGORY VISUALIZATION
=============================================================================

def stacked_category_plot(df: pd.DataFrame) -> None:
"""Create a 100% stacked categorical visualization."""

section("28 - 100% Stacked Category Plot")

proportions = pd.crosstab(
    df["department"],
    df["education"],
    normalize="index"
) * 100

ax = proportions.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6)
)

ax.set_title("Education Composition by Department")
ax.set_xlabel("Department")
ax.set_ylabel("Percentage")
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=0
)

plt.tight_layout()
plt.show()
=============================================================================
29. MACHINE LEARNING CLASS DISTRIBUTION
=============================================================================

def ml_class_distribution(df: pd.DataFrame) -> None:
"""
Visualize the distribution of a classification target.

Class distribution is an important first check for classification
problems because heavily unequal classes may affect model
training and evaluation.
"""

section("29 - ML Class Distribution")

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="purchased"
)

plt.title("Target Class Distribution")
plt.xlabel("Purchased")
plt.ylabel("Number of Samples")

plt.tight_layout()
plt.show()

print("\nClass counts:")
print(df["purchased"].value_counts())

print("\nClass proportions:")
print(
    df["purchased"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)
=============================================================================
30. CLASS DISTRIBUTION WITH ANNOTATIONS
=============================================================================

def annotated_class_distribution(df: pd.DataFrame) -> None:
"""Create an annotated target distribution chart."""

section("30 - Annotated Class Distribution")

counts = (
    df["purchased"]
    .value_counts()
    .rename_axis("purchased")
    .reset_index(name="count")
)

plt.figure(figsize=(8, 6))

ax = sns.barplot(
    data=counts,
    x="purchased",
    y="count"
)

ax.bar_label(
    ax.containers[0],
    fmt="%d",
    padding=3
)

plt.title("Classification Target Distribution")
plt.xlabel("Purchased")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
=============================================================================
31. FEATURE VS CLASS
=============================================================================

def feature_vs_class(df: pd.DataFrame) -> None:
"""
Compare a numerical feature across classification classes.
"""

section("31 - Feature vs Classification Target")

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="purchased",
    y="purchase_amount"
)

plt.title("Purchase Amount by Target Class")
plt.xlabel("Purchased")
plt.ylabel("Purchase Amount")

plt.tight_layout()
plt.show()
=============================================================================
32. MULTIPLE FEATURES VS CLASS
=============================================================================

def multiple_features_vs_class(df: pd.DataFrame) -> None:
"""Compare multiple numerical features against the target."""

section("32 - Multiple Features vs Target")

numeric_features = [
    "age",
    "salary",
    "experience",
    "purchase_amount",
    "rating"
]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 9)
)

axes = axes.flatten()

for ax, feature in zip(
    axes,
    numeric_features
):
    sns.boxplot(
        data=df,
        x="purchased",
        y=feature,
        ax=ax
    )

    ax.set_title(f"{feature.title()} vs Purchased")
    ax.set_xlabel("Purchased")

# Hide unused subplot.
for ax in axes[len(numeric_features):]:
    ax.set_visible(False)

fig.suptitle(
    "Numerical Features by Classification Target",
    fontsize=16
)

fig.tight_layout()
plt.show()
=============================================================================
33. CATEGORY VS TARGET
=============================================================================

def category_vs_target(df: pd.DataFrame) -> None:
"""
Analyze the relationship between a categorical feature
and the classification target.
"""

section("33 - Category vs Target")

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="department",
    hue="purchased"
)

plt.title("Purchase Outcome by Department")
plt.xlabel("Department")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
=============================================================================
34. TARGET RATE BY CATEGORY
=============================================================================

def target_rate_by_category(df: pd.DataFrame) -> None:
"""
Calculate and visualize the percentage of positive target
outcomes within each department.
"""

section("34 - Target Rate by Category")

target_rate = (
    df.assign(
        purchased_binary=(
            df["purchased"] == "Yes"
        ).astype(int)
    )
    .groupby("department", as_index=False)
    ["purchased_binary"]
    .mean()
)

target_rate["purchase_rate_percent"] = (
    target_rate["purchased_binary"] * 100
)

print("\nPurchase rate by department:")
print(
    target_rate[
        ["department", "purchase_rate_percent"]
    ].round(2)
)

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=target_rate,
    x="department",
    y="purchase_rate_percent"
)

ax.bar_label(
    ax.containers[0],
    fmt="%.1f%%",
    padding=3
)

plt.title("Purchase Rate by Department")
plt.xlabel("Department")
plt.ylabel("Purchase Rate (%)")

plt.tight_layout()
plt.show()
=============================================================================
35. CATEGORY VS NUMERICAL FEATURE
=============================================================================

def category_numerical_analysis(df: pd.DataFrame) -> None:
"""Compare salary distributions by education level."""

section("35 - Category vs Numerical Feature")

education_order = [
    "Bachelor",
    "Master",
    "PhD"
]

plt.figure(figsize=(9, 6))

sns.violinplot(
    data=df,
    x="education",
    y="salary",
    order=education_order
)

plt.title("Salary Distribution by Education")
plt.xlabel("Education")
plt.ylabel("Salary")

plt.tight_layout()
plt.show()
=============================================================================
36. MULTI-CATEGORY ANALYSIS
=============================================================================

def multi_category_analysis(df: pd.DataFrame) -> None:
"""
Compare multiple categorical dimensions in one visualization.
"""

section("36 - Multi-Category Analysis")

plt.figure(figsize=(12, 6))

sns.barplot(
    data=df,
    x="city",
    y="purchase_amount",
    hue="membership"
)

plt.title(
    "Average Purchase Amount by City and Membership"
)

plt.xlabel("City")
plt.ylabel("Average Purchase Amount")

plt.tight_layout()
plt.show()
=============================================================================
37. CUSTOM PALETTE
=============================================================================

def custom_palette_example(df: pd.DataFrame) -> None:
"""Demonstrate palette customization."""

section("37 - Custom Palette")

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="department",
    y="salary",
    hue="department",
    palette="viridis",
    legend=False
)

plt.title("Average Salary by Department")
plt.xlabel("Department")
plt.ylabel("Average Salary")

plt.tight_layout()
plt.show()
=============================================================================
38. COLORBLIND-FRIENDLY PALETTE
=============================================================================

def colorblind_palette_example(df: pd.DataFrame) -> None:
"""
Demonstrate a palette designed to improve accessibility.
"""

section("38 - Colorblind-Friendly Palette")

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="department",
    hue="gender",
    palette="colorblind"
)

plt.title("Department Distribution by Gender")
plt.xlabel("Department")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
=============================================================================
39. PROFESSIONAL CATEGORY DASHBOARD
=============================================================================

def categorical_dashboard(df: pd.DataFrame) -> None:
"""
Build a compact categorical EDA dashboard.

This combines several categorical analyses into one figure.
"""

section("39 - Categorical EDA Dashboard")

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 10)
)

# -------------------------------------------------------------------------
# Chart 1: Department frequency
# -------------------------------------------------------------------------

sns.countplot(
    data=df,
    x="department",
    ax=axes[0, 0]
)

axes[0, 0].set_title(
    "Department Distribution"
)

axes[0, 0].tick_params(
    axis="x",
    rotation=15
)

# -------------------------------------------------------------------------
# Chart 2: Membership distribution
# -------------------------------------------------------------------------

sns.countplot(
    data=df,
    x="membership",
    ax=axes[0, 1]
)

axes[0, 1].set_title(
    "Membership Distribution"
)

# -------------------------------------------------------------------------
# Chart 3: Salary by department
# -------------------------------------------------------------------------

sns.boxplot(
    data=df,
    x="department",
    y="salary",
    ax=axes[1, 0]
)

axes[1, 0].set_title(
    "Salary Distribution by Department"
)

axes[1, 0].tick_params(
    axis="x",
    rotation=15
)

# -------------------------------------------------------------------------
# Chart 4: Target distribution
# -------------------------------------------------------------------------

sns.countplot(
    data=df,
    x="purchased",
    ax=axes[1, 1]
)

axes[1, 1].set_title(
    "Target Distribution"
)

fig.suptitle(
    "Categorical Exploratory Data Analysis Dashboard",
    fontsize=17
)

fig.tight_layout()
plt.show()
=============================================================================
40. REUSABLE COUNT PLOT FUNCTION
=============================================================================

def plot_category_count(
df: pd.DataFrame,
column: str,
title: str | None = None,
order: list[str] | None = None,
) -> None:
"""
Reusable function for categorical frequency visualization.

Parameters
----------
df:
    Input DataFrame.

column:
    Categorical column to visualize.

title:
    Optional chart title.

order:
    Optional category ordering.
"""

if column not in df.columns:
    raise ValueError(
        f"Column '{column}' does not exist in the DataFrame."
    )

plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x=column,
    order=order
)

plt.title(
    title or f"Distribution of {column}"
)

plt.xlabel(column.replace("_", " ").title())
plt.ylabel("Count")

plt.tight_layout()
plt.show()
=============================================================================
41. REUSABLE CATEGORICAL COMPARISON FUNCTION
=============================================================================

def plot_category_comparison(
df: pd.DataFrame,
category: str,
value: str,
hue: str | None = None,
title: str | None = None,
) -> None:
"""
Reusable categorical-vs-numerical visualization.

Parameters
----------
df:
    Input DataFrame.

category:
    Categorical variable.

value:
    Numerical variable.

hue:
    Optional grouping variable.

title:
    Optional title.
"""

required_columns = [
    category,
    value
]

if hue is not None:
    required_columns.append(hue)

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x=category,
    y=value,
    hue=hue
)

plt.title(
    title
    or f"{value.title()} by {category.title()}"
)

plt.xlabel(
    category.replace("_", " ").title()
)

plt.ylabel(
    value.replace("_", " ").title()
)

plt.tight_layout()
plt.show()
=============================================================================
42. DATA QUALITY CHECKS
=============================================================================

def categorical_data_quality(df: pd.DataFrame) -> None:
"""Print useful checks for categorical variables."""

section("42 - Categorical Data Quality Checks")

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

print("\nCategorical columns:")
print(list(categorical_columns))

for column in categorical_columns:
    print("\n" + "-" * 60)
    print(f"Column: {column}")

    print("Unique values:")
    print(df[column].nunique())

    print("Missing values:")
    print(df[column].isna().sum())

    print("Value counts:")
    print(df[column].value_counts(dropna=False).head(10))
=============================================================================
43. MAIN
=============================================================================

def main() -> None:
"""
Run the categorical visualization tutorial.

All examples are kept as individual functions so learners can
execute only the sections they currently need.
"""

section("SEABORN CATEGORICAL PLOTS")

df = create_sample_dataset()

print("\nDataset preview:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

# -------------------------------------------------------------------------
# Basic categorical plots
# -------------------------------------------------------------------------

basic_count_plot(df)
count_plot_with_hue(df)
horizontal_count_plot(df)

basic_bar_plot(df)
grouped_bar_plot(df)
horizontal_bar_plot(df)

ordered_categories(df)
order_by_frequency(df)

# -------------------------------------------------------------------------
# Statistical categorical plots
# -------------------------------------------------------------------------

point_plot(df)
point_plot_with_hue(df)

box_plot(df)
box_plot_with_hue(df)

violin_plot(df)
violin_plot_with_hue(df)

strip_plot(df)
strip_plot_with_hue(df)

swarm_plot(df)
boxen_plot(df)

# -------------------------------------------------------------------------
# Figure-level categorical plots
# -------------------------------------------------------------------------

catplot_box(df)
catplot_violin(df)
catplot_bar(df)
catplot_faceting(df)

# -------------------------------------------------------------------------
# Categorical analysis
# -------------------------------------------------------------------------

category_vs_category(df)
normalized_category_analysis(df)
stacked_category_plot(df)

# -------------------------------------------------------------------------
# Machine Learning visualization
# -------------------------------------------------------------------------

ml_class_distribution(df)
annotated_class_distribution(df)

feature_vs_class(df)
multiple_features_vs_class(df)

category_vs_target(df)
target_rate_by_category(df)

category_numerical_analysis(df)
multi_category_analysis(df)

# -------------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------------

custom_palette_example(df)
colorblind_palette_example(df)

# -------------------------------------------------------------------------
# Dashboard
# -------------------------------------------------------------------------

categorical_dashboard(df)

# -------------------------------------------------------------------------
# Reusable functions
# -------------------------------------------------------------------------

plot_category_count(
    df,
    column="city",
    title="Customer Distribution by City"
)

plot_category_comparison(
    df,
    category="membership",
    value="purchase_amount",
    title="Purchase Amount by Membership"
)

# -------------------------------------------------------------------------
# Data quality
# -------------------------------------------------------------------------

categorical_data_quality(df)

print("\nCategorical visualization tutorial completed.")
=============================================================================
44. ENTRY POINT
=============================================================================

if name == "main":
main()

=============================================================================
QUICK REFERENCE
=============================================================================


Count categories:
sns.countplot(data=df, x="category")


Compare numerical values by category:
sns.barplot(data=df, x="category", y="value")


Group categories:
sns.barplot(data=df, x="category", y="value", hue="group")


Distribution:
sns.boxplot(data=df, x="category", y="value")


Distribution shape:
sns.violinplot(data=df, x="category", y="value")


Individual observations:
sns.stripplot(data=df, x="category", y="value")


Reduced overlap:
sns.swarmplot(data=df, x="category", y="value")


Detailed distribution:
sns.boxenplot(data=df, x="category", y="value")


Figure-level categorical plot:
sns.catplot(data=df, x="category", y="value", kind="box")


Group by another variable:
sns.catplot(
data=df,
x="category",
y="value",
hue="group",
kind="box"
)


Category frequency:
df["category"].value_counts()


Category proportions:
df["category"].value_counts(normalize=True)


Cross-tabulation:
pd.crosstab(df["category"], df["group"])


Normalized cross-tabulation:
pd.crosstab(
df["category"],
df["group"],
normalize="index"
)


Save:
plt.savefig(
"categorical_plot.png",
dpi=300,
bbox_inches="tight"
)


=============================================================================
KEY TAKEAWAYS
=============================================================================


1. Use countplot() for categorical frequencies.
2. Use barplot() for numerical statistics by category.
3. Use boxplot() to compare distributions and potential outliers.
4. Use violinplot() to understand distribution shapes.
5. Use stripplot() to display individual observations.
6. Use swarmplot() when individual observations should remain visible.
7. Use pointplot() for category-level estimates.
8. Use catplot() for figure-level categorical analysis.
9. Use hue to compare subgroups.
10. Use explicit category ordering when order matters.
11. Use cross-tabulation for category-vs-category analysis.
12. Inspect target class distribution before classification.
13. Investigate class imbalance instead of assuming it is a problem.
14. Combine Seaborn with Matplotlib for professional customization.
15. Keep categorical plots focused, readable, and tied to an EDA question.


=============================================================================
END OF FILE
=============================================================================
