# 🧠 Multivariate Analysis

> **Multivariate Analysis = Understanding Multiple Variables Together**

Multivariate analysis is the next level of **Exploratory Data Analysis (EDA)** after univariate and bivariate analysis.

While:

```text
Univariate  → One variable
Bivariate   → Two variables
Multivariate → Three or more variables
```

multivariate analysis examines how **multiple variables interact, relate, influence patterns, and contribute to the structure of a dataset**.

# 🌱 GROW → CONNECT → EXPLORE → COMPARE → REDUCE → INTERPRET → BUILD 🚀

```text
                    🌱 DATASET
                        │
                        ↓
                ┌───────────────┐
                │   MULTIVARIATE │
                │    ANALYSIS    │
                └───────┬───────┘
                        │
       ┌────────────────┼────────────────┐
       ↓                ↓                ↓
   Relationships     Distributions    Interactions
       │                │                │
       ↓                ↓                ↓
  Correlation       Pair Plots       Group Analysis
       │                │                │
       └────────────────┼────────────────┘
                        ↓
                Dimensionality
                   Reduction
                        ↓
                     PCA
                        ↓
                Feature Insights
                        ↓
                  Machine Learning
```

---

# 📚 Table of Contents

1. [What Is Multivariate Analysis?](#-what-is-multivariate-analysis)
2. [Why Multivariate Analysis Matters](#-why-multivariate-analysis-matters)
3. [Univariate vs Bivariate vs Multivariate](#-univariate-vs-bivariate-vs-multivariate)
4. [Types of Multivariate Analysis](#-types-of-multivariate-analysis)
5. [Types of Variables](#-types-of-variables)
6. [Multivariate Relationships](#-multivariate-relationships)
7. [Correlation Matrix](#-correlation-matrix)
8. [Correlation Heatmap](#-correlation-heatmap)
9. [Pair Plot](#-pair-plot)
10. [Scatter Matrix](#-scatter-matrix)
11. [Grouped Multivariate Analysis](#-grouped-multivariate-analysis)
12. [Multiple Numerical Variables](#-multiple-numerical-variables)
13. [Categorical Variables](#-categorical-variables)
14. [Numerical + Categorical + Numerical](#-numerical--categorical--numerical)
15. [Feature Interactions](#-feature-interactions)
16. [Interaction Effects](#-interaction-effects)
17. [Multicollinearity](#-multicollinearity)
18. [Variance Inflation Factor](#-variance-inflation-factor)
19. [Dimensionality](#-dimensionality)
20. [Curse of Dimensionality](#-curse-of-dimensionality)
21. [Dimensionality Reduction](#-dimensionality-reduction)
22. [Principal Component Analysis](#-principal-component-analysis)
23. [PCA Intuition](#-pca-intuition)
24. [PCA Implementation](#-pca-implementation)
25. [Explained Variance](#-explained-variance)
26. [Standardization Before PCA](#-standardization-before-pca)
27. [PCA Visualization](#-pca-visualization)
28. [Categorical Encoding](#-categorical-encoding)
29. [Missing Values](#-missing-values)
30. [Outliers](#-outliers)
31. [Scaling](#-scaling)
32. [Feature Selection](#-feature-selection)
33. [Target Relationships](#-target-relationships)
34. [Multivariate Visualization](#-multivariate-visualization)
35. [3D Scatter Plot](#-3d-scatter-plot)
36. [Faceted Visualization](#-faceted-visualization)
37. [Parallel Coordinates](#-parallel-coordinates)
38. [Automated Multivariate Analysis](#-automated-multivariate-analysis)
39. [Machine Learning Applications](#-machine-learning-applications)
40. [Common Mistakes](#-common-mistakes)
41. [Best Practices](#-best-practices)
42. [Mini Projects](#-mini-projects)
43. [Exercises](#-exercises)
44. [Project Structure](#-project-structure)
45. [End-to-End Example](#-end-to-end-example)
46. [Multivariate Analysis Checklist](#-multivariate-analysis-checklist)
47. [Decision Guide](#-decision-guide)
48. [EDA Roadmap](#-eda-roadmap)
49. [Key Takeaways](#-key-takeaways)
50. [Next Step](#-next-step)
51. [Author](#-author)
52. [Contributing](#-contributing)
53. [Support](#-support)
54. [License](#-license)

---

# 🔎 What Is Multivariate Analysis?

**Multivariate analysis** studies multiple variables simultaneously to understand:

* Relationships
* Interactions
* Patterns
* Dependencies
* Groups
* Hidden structure
* Feature redundancy
* Dimensionality
* Contributions to a target

Example:

```text
Age
 │
 ├──────────────┐
 │              │
Experience ─── Salary
 │              │
Education ──────┘
```

Instead of asking only:

```text
Experience → Salary
```

we can ask:

```text
Age
Experience
Education
Location
Performance
       ↓
    Salary
```

This provides a more realistic view of complex datasets.

---

# 🎯 Why Multivariate Analysis Matters

Real-world problems rarely depend on only one variable.

For example, house prices may depend on:

```text
Size
Bedrooms
Location
Age
Bathrooms
Parking
Neighborhood
Distance from city
```

Analyzing each variable independently is useful, but insufficient.

Multivariate analysis allows us to investigate:

```text
Size + Location + Bedrooms + Age
                    ↓
                House Price
```

---

# 🔄 Univariate vs Bivariate vs Multivariate

| Analysis     | Variables | Main Question                        |
| ------------ | --------: | ------------------------------------ |
| Univariate   |         1 | What does this variable look like?   |
| Bivariate    |         2 | How are these two variables related? |
| Multivariate |        3+ | How do several variables interact?   |

### Example

```text
UNIVARIATE

Salary
  ↓
Distribution
```

```text
BIVARIATE

Experience ↔ Salary
```

```text
MULTIVARIATE

Age + Experience + Education + Location
                  ↓
                Salary
```

---

# 🧩 Types of Multivariate Analysis

Common approaches include:

### Exploratory techniques

* Correlation matrices
* Pair plots
* Scatter matrices
* Grouped analysis
* Faceted plots
* 3D visualization
* Parallel coordinates

### Statistical techniques

* Multiple regression
* MANOVA
* Cluster analysis
* Factor analysis
* Principal Component Analysis
* Canonical correlation

### Machine Learning techniques

* Feature selection
* Dimensionality reduction
* Clustering
* Representation learning
* Feature interaction analysis

This section focuses primarily on **EDA-oriented multivariate analysis**.

---

# 🧱 Types of Variables

A multivariate dataset can contain:

### Numerical

```text
age
salary
height
income
temperature
```

### Categorical

```text
gender
department
city
education
product_category
```

### Ordinal

```text
low
medium
high
```

### Date-Time

```text
2026-01-01
2026-01-02
2026-01-03
```

### Boolean

```text
True
False
```

The type of variables determines which analysis techniques are appropriate.

---

# 🔗 Multivariate Relationships

Suppose we have:

```text
Age
Experience
Education
Salary
```

Possible relationships include:

```text
Age ↔ Salary
Experience ↔ Salary
Education ↔ Salary
Age ↔ Experience
```

But we can also investigate:

```text
Age + Experience → Salary
```

and:

```text
Education × Experience → Salary
```

This introduces the idea of **interactions**.

---

# 📊 Correlation Matrix

A correlation matrix summarizes pairwise relationships among numerical variables.

```python
import pandas as pd

correlation_matrix = df.corr(numeric_only=True)

print(correlation_matrix)
```

Example:

```text
             age  experience  income  score
age          1.00    0.82      0.70    0.35
experience   0.82    1.00      0.78    0.42
income       0.70    0.78      1.00    0.60
score        0.35    0.42      0.60    1.00
```

This can reveal:

* Strong relationships
* Weak relationships
* Potential redundancy
* Possible multicollinearity

But pairwise correlation does not fully describe multivariate relationships.

---

# 🔥 Correlation Heatmap

A heatmap provides a visual overview.

```python
import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = df.corr(numeric_only=True)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f"
)

plt.title("Multivariate Correlation Heatmap")
plt.tight_layout()
plt.show()
```

A useful workflow is:

```text
Correlation Matrix
       ↓
Heatmap
       ↓
Identify Interesting Relationships
       ↓
Investigate Individually
```

---

# 🔬 Pair Plot

A pair plot displays relationships among multiple numerical variables.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(
    df[
        [
            "age",
            "experience",
            "income",
            "score"
        ]
    ]
)

plt.show()
```

A pair plot usually contains:

```text
Diagonal
→ Individual distributions

Off-diagonal
→ Pairwise relationships
```

---

# 🎨 Pair Plot With Categories

A categorical variable can be used to distinguish groups.

```python
sns.pairplot(
    df,
    vars=[
        "age",
        "experience",
        "income",
        "score"
    ],
    hue="department"
)

plt.show()
```

This can reveal:

* Group separation
* Clusters
* Different trends
* Potential outliers
* Category-specific relationships

---

# 🧮 Scatter Matrix

Pandas provides a convenient scatter matrix.

```python
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

scatter_matrix(
    df[
        [
            "age",
            "experience",
            "income"
        ]
    ],
    figsize=(10, 10)
)

plt.tight_layout()
plt.show()
```

Scatter matrices are useful for quickly inspecting many pairwise relationships.

---

# 👥 Grouped Multivariate Analysis

Grouping allows us to investigate how relationships change across categories.

Example:

```python
grouped = (
    df.groupby("department")
    [["age", "experience", "salary"]]
    .mean()
)

print(grouped)
```

Another example:

```python
grouped = (
    df.groupby(["department", "education"])["salary"]
    .mean()
)

print(grouped)
```

This allows questions such as:

> Does the relationship between education and salary differ across departments?

---

# 🔢 Multiple Numerical Variables

Suppose:

```text
age
experience
income
hours_worked
performance
```

We can calculate:

```python
numeric_columns = [
    "age",
    "experience",
    "income",
    "hours_worked",
    "performance"
]

correlation = df[numeric_columns].corr()

print(correlation)
```

Then visualize:

```python
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.show()
```

---

# 🏷️ Categorical Variables

Categorical variables can also be incorporated into multivariate analysis.

Example:

```text
Department
Education
Gender
Salary
Performance
```

We can group by multiple categories:

```python
summary = (
    df.groupby(
        ["department", "education"]
    )["salary"]
    .agg(
        ["count", "mean", "median"]
    )
)

print(summary)
```

---

# 🔢 Numerical + Categorical + Numerical

This is a common real-world pattern.

Example:

```text
Experience
     +
Department
     ↓
Salary
```

Visualization:

```python
sns.scatterplot(
    data=df,
    x="experience",
    y="salary",
    hue="department",
    style="education"
)

plt.title("Experience vs Salary by Department")
plt.show()
```

Here:

* X-axis = Experience
* Y-axis = Salary
* Color = Department
* Marker style = Education

One visualization can represent several variables.

---

# 🔀 Feature Interactions

A feature interaction occurs when the effect of one variable depends on another variable.

Example:

```text
Experience → Salary
```

may not be identical for every education level.

Instead:

```text
Experience + Education → Salary
```

For example:

```text
Experience
    │
    ├── Bachelor's → Salary Trend A
    │
    ├── Master's   → Salary Trend B
    │
    └── PhD        → Salary Trend C
```

This is a more advanced form of analysis.

---

# ⚡ Interaction Effects

In a regression-style model, an interaction can be represented as:

$$
Y = \beta_0 + \beta_1X_1 + \beta_2X_2 +
\beta_3(X_1X_2) + \epsilon
$$

Where:

* \(X_1\) = first feature
* \(X_2\) = second feature
* \(X_1X_2\) = interaction term
* \(Y\) = target

Example:

```text
Salary =
Base
+ Experience Effect
+ Education Effect
+ Experience × Education Effect
```

The interaction term asks:

> Does the effect of experience change depending on education?

---

# 🚨 Multicollinearity

Multicollinearity occurs when predictor variables contain substantial overlapping information.

Example:

```text
Age
   ↕
Years of Experience
```

These variables may be strongly related.

Another example:

```text
Annual Income
Monthly Income
```

These are almost directly related.

High multicollinearity can make regression coefficients:

* Unstable
* Difficult to interpret
* Sensitive to small data changes

It can also make individual coefficient interpretation less reliable.

---

# 📐 Variance Inflation Factor

**Variance Inflation Factor (VIF)** is commonly used to diagnose multicollinearity among predictors.

Conceptually:

$$
VIF_j = \frac{1}{1-R_j^2}
$$

where \(R_j^2\) is obtained by regressing predictor \(j\) on the other predictors.

Implementation:

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df[
    [
        "age",
        "experience",
        "income"
    ]
].dropna()

vif = pd.DataFrame()

vif["feature"] = X.columns
vif["VIF"] = [
    variance_inflation_factor(
        X.values,
        i
    )
    for i in range(X.shape[1])
]

print(vif)
```

Common rules of thumb sometimes use thresholds such as 5 or 10, but these are **diagnostic guidelines, not universal laws**. Domain context and modeling goals matter.

---

# 📏 Dimensionality

Dimensionality refers to the number of features or variables in a dataset.

Example:

```text
2 Features
→ 2D

3 Features
→ 3D

100 Features
→ High-dimensional data
```

Machine Learning datasets can easily contain:

```text
10
100
1,000
10,000+
```

features.

This creates additional challenges.

---

# ⚠️ Curse of Dimensionality

As the number of dimensions increases:

* Data becomes sparse
* Distance-based methods become harder to interpret
* Visualization becomes difficult
* Computation can increase
* Models may require more data

Conceptually:

```text
Low Dimensions
      ↓
Easy to Visualize
      ↓
More Dimensions
      ↓
Harder to Explore
      ↓
High Dimensions
      ↓
Need Dimensionality Reduction
```

---

# 📉 Dimensionality Reduction

Dimensionality reduction transforms many features into a smaller number of representations.

Common techniques include:

* PCA
* Truncated SVD
* t-SNE
* UMAP

For classical EDA and ML preprocessing, **PCA** is one of the most important techniques to understand.

---

# 🧠 Principal Component Analysis

**Principal Component Analysis (PCA)** transforms correlated numerical variables into a new set of orthogonal components called **principal components**.

Conceptually:

```text
Original Features

X1 ─┐
X2 ─┤
X3 ─┤
X4 ─┤
X5 ─┘
     ↓
    PCA
     ↓
PC1 ─────────────
PC2 ─────────────
PC3 ─────────────
```

Instead of working with:

```text
5 original dimensions
```

we may represent much of the variation using:

```text
2 or 3 principal components
```

---

# 🧭 PCA Intuition

Imagine a cloud of points:

```text
       •
     •
   •
 •
```

PCA searches for directions that capture large amounts of variation.

The first principal component:

```text
PC1
───────────────→
```

captures the greatest variance.

The second component is orthogonal to the first:

```text
       ↑ PC2
       │
       │
───────┼────────→ PC1
```

---

# 💻 PCA Implementation

First, select numerical features.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

features = [
    "age",
    "experience",
    "income",
    "performance"
]

X = df[features].dropna()
```

Standardize:

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Apply PCA:

```python
pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(X_scaled)

print(X_pca.shape)
```

---

# 📊 Explained Variance

PCA provides the proportion of variance explained by each component.

```python
print(pca.explained_variance_ratio_)
```

Example:

```text
[0.61, 0.24]
```

This means:

```text
PC1 → 61%
PC2 → 24%
```

Together:

```text
61% + 24% = 85%
```

of the variance is represented by the two selected components for this transformed dataset.

Cumulative variance:

```python
cumulative_variance = (
    pca.explained_variance_ratio_.cumsum()
)

print(cumulative_variance)
```

---

# ⚖️ Standardization Before PCA

PCA is sensitive to feature scale.

Suppose:

```text
Age         → 20–80
Income      → 20,000–500,000
Experience  → 0–40
```

Income has a much larger numerical scale.

Without scaling, it may dominate the variance structure.

Therefore:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Then:

```python
pca.fit_transform(X_scaled)
```

### Important

Do not automatically standardize every dataset without thinking about the measurement meaning and modeling objective. For standard PCA on differently scaled physical quantities, standardization is commonly appropriate.

---

# 📈 PCA Visualization

After reducing the dataset to two components:

```python
import matplotlib.pyplot as plt

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1]
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Projection")

plt.show()
```

If labels exist:

```python
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df.loc[X.index, "target"]
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection by Target")
plt.colorbar(label="Target")

plt.show()
```

This can reveal:

* Clusters
* Separation
* Overlap
* Outliers
* Hidden structure

---

# 🧩 PCA Loadings

PCA components can help us understand how original variables contribute to each component.

```python
loadings = pd.DataFrame(
    pca.components_.T,
    index=features,
    columns=["PC1", "PC2"]
)

print(loadings)
```

Example:

```text
             PC1     PC2
age          0.42    0.11
experience   0.51    0.08
income       0.57   -0.10
performance  0.31    0.78
```

Large absolute loadings indicate stronger contribution to that component, but interpretation should consider the complete PCA setup.

---

# 🏷️ Categorical Encoding

PCA operates on numerical feature matrices.

Categorical variables must be encoded before PCA.

For example:

```python
encoded = pd.get_dummies(
    df,
    columns=["department", "education"],
    dtype=int
)
```

However, for a production ML pipeline, prefer a consistent preprocessing pipeline using tools such as:

```text
ColumnTransformer
+
OneHotEncoder
+
StandardScaler
+
PCA
```

This avoids inconsistent preprocessing between training and inference.

---

# 🕳️ Missing Values

Most scikit-learn PCA implementations do not accept ordinary missing values directly.

Therefore:

```text
Raw Data
   ↓
Missing Value Handling
   ↓
Feature Selection
   ↓
Scaling
   ↓
PCA
```

Example:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    strategy="median"
)

X_imputed = imputer.fit_transform(X)
```

Then:

```python
X_scaled = scaler.fit_transform(X_imputed)
X_pca = pca.fit_transform(X_scaled)
```

For ML workflows, fit these preprocessing steps only on the training data.

---

# 🚨 Outliers

Outliers can strongly affect:

* Correlation
* Covariance
* PCA
* Regression
* Clustering

Before PCA, investigate extreme observations.

Possible approaches include:

* Domain validation
* Robust transformations
* Carefully justified removal
* Robust scaling
* Separate analysis

Never remove outliers automatically just because they are unusual.

---

# ⚖️ Scaling

Common scaling approaches include:

### StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

### MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
```

### RobustScaler

Useful when extreme values are important and robust scaling is appropriate:

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_scaled = scaler.fit_transform(X)
```

Choose scaling based on the downstream method and data characteristics.

---

# 🎯 Feature Selection

Multivariate analysis can help identify:

```text
Useful Features
Redundant Features
Highly Correlated Features
Low-Variance Features
Potential Leakage
Interactions
```

However, EDA should not be the only basis for feature selection.

For supervised ML, feature selection must be performed without allowing validation/test information to leak into training decisions.

---

# 🎯 Target Relationships

Suppose:

```text
Target = House Price
```

Potential predictors:

```text
Size
Bedrooms
Bathrooms
Location
Age
Parking
```

A useful workflow:

```text
Each Feature ↔ Target
        ↓
Pairwise Relationships
        ↓
Feature ↔ Feature
        ↓
Redundancy
        ↓
Interactions
        ↓
Multivariate Model
```

---

# 🎨 Multivariate Visualization

Multivariate visualization allows multiple variables to be represented in one chart.

For example:

```python
sns.scatterplot(
    data=df,
    x="experience",
    y="salary",
    hue="department",
    size="performance"
)

plt.show()
```

Here:

```text
X position → Experience
Y position → Salary
Color      → Department
Size       → Performance
```

---

# 🧊 3D Scatter Plot

For three numerical variables:

```python
import matplotlib.pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(
    projection="3d"
)

ax.scatter(
    df["age"],
    df["experience"],
    df["salary"]
)

ax.set_xlabel("Age")
ax.set_ylabel("Experience")
ax.set_zlabel("Salary")

plt.title("3D Multivariate Relationship")

plt.show()
```

3D plots can be useful for exploration, but they can become difficult to read. Often, 2D projections or small multiples communicate patterns more clearly.

---

# 🧩 Faceted Visualization

Faceting creates separate plots for categories.

```python
g = sns.FacetGrid(
    df,
    col="department"
)

g.map_dataframe(
    sns.scatterplot,
    x="experience",
    y="salary"
)

g.set_axis_labels(
    "Experience",
    "Salary"
)

plt.show()
```

This allows us to compare:

```text
Department A
Department B
Department C
```

using the same relationship.

---

# 📐 Parallel Coordinates

Parallel coordinates visualize multiple numerical variables across observations.

```python
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt

columns = [
    "department",
    "age",
    "experience",
    "salary"
]

parallel_coordinates(
    df[columns],
    "department"
)

plt.title("Parallel Coordinates")
plt.xticks(rotation=45)
plt.show()
```

This can help identify:

* Patterns
* Clusters
* Similar observations
* Group differences
* Outliers

---

# 🤖 Automated Multivariate Analysis

A reusable numerical analysis function:

```python
def multivariate_summary(df):
    numeric_df = df.select_dtypes(
        include="number"
    )

    return {
        "shape": numeric_df.shape,
        "columns": numeric_df.columns.tolist(),
        "missing_values": (
            numeric_df.isna().sum().to_dict()
        ),
        "correlation": numeric_df.corr()
    }
```

Usage:

```python
summary = multivariate_summary(df)

print("Shape:", summary["shape"])
print("Columns:", summary["columns"])
print(summary["correlation"])
```

---

# 🤖 Machine Learning Applications

Multivariate analysis plays an important role throughout ML.

```text
DATA
 ↓
EDA
 ↓
BIVARIATE
 ↓
MULTIVARIATE
 ↓
FEATURE ENGINEERING
 ↓
FEATURE SELECTION
 ↓
DIMENSIONALITY REDUCTION
 ↓
MODEL TRAINING
 ↓
EVALUATION
```

Applications include:

### Regression

```text
X1 + X2 + X3 + X4
        ↓
        Y
```

### Classification

```text
Features
   ↓
Class Prediction
```

### Clustering

```text
Multiple Features
       ↓
Similarity
       ↓
Clusters
```

### PCA

```text
Many Features
      ↓
Principal Components
      ↓
Lower-Dimensional Representation
```

---

# ⚠️ Common Mistakes

## ❌ Mistake 1: Using Only Pairwise Correlations

Pairwise correlation does not capture all multivariate structure.

---

## ❌ Mistake 2: Ignoring Feature Scale

Methods such as PCA are sensitive to feature scale.

---

## ❌ Mistake 3: Applying PCA Before Handling Missing Values

Standard PCA generally requires a complete numerical matrix.

---

## ❌ Mistake 4: Applying PCA to Raw Categories

Categorical data must first be represented numerically using an appropriate encoding strategy.

---

## ❌ Mistake 5: Assuming PCA Components Are Original Features

Principal components are combinations of original variables.

```text
PC1 ≠ Original Feature
```

---

## ❌ Mistake 6: Treating PCA as Feature Selection

PCA creates transformed components.

Feature selection chooses original features.

These are different approaches.

---

## ❌ Mistake 7: Ignoring Multicollinearity

Strongly related predictors can make model interpretation difficult.

---

## ❌ Mistake 8: Using Too Many Variables in One Plot

A visualization with too many encodings can become unreadable.

Prefer:

```text
Simple Plot
+
Faceting
+
Multiple Focused Views
```

---

## ❌ Mistake 9: Data Leakage

Do not fit preprocessing steps such as:

* Imputation
* Scaling
* PCA
* Feature selection

using the entire dataset before train/test splitting.

Correct:

```text
Training Data
     ↓
Fit Preprocessor
     ↓
Transform Training Data
     ↓
Transform Validation/Test Data
```

---

# ✅ Best Practices

### 1. Start Simple

```text
Univariate
    ↓
Bivariate
    ↓
Multivariate
```

### 2. Understand the Dataset First

Know:

* Feature meanings
* Data types
* Units
* Missing values
* Measurement process

### 3. Visualize Relationships

Use:

* Pair plots
* Heatmaps
* Scatter plots
* Facets
* Grouped plots

### 4. Check Feature Redundancy

Investigate strongly related predictors.

### 5. Check Outliers

Extreme observations can influence multivariate methods.

### 6. Scale When Required

Especially before:

* PCA
* KNN
* K-Means
* SVM
* Many distance-based algorithms

### 7. Avoid Leakage

Fit transformations only on training data in predictive workflows.

### 8. Keep Interpretability in Mind

A technically sophisticated visualization is not necessarily a useful one.

---

# 🚀 Mini Projects

## 🏠 Project 1 — House Price Multivariate Analysis

Variables:

```text
price
size
bedrooms
bathrooms
age
parking
location
```

Tasks:

* Correlation matrix
* Pair plot
* Feature relationships
* Grouped analysis
* PCA experiment
* Multicollinearity investigation

---

## 💼 Project 2 — Employee Salary Analysis

Variables:

```text
age
experience
education
department
performance
salary
```

Questions:

```text
Age + Experience → Salary?
Education + Experience → Salary?
Department + Performance → Salary?
```

---

## 🛒 Project 3 — Customer Behavior

Variables:

```text
age
income
visits
session_time
purchase_amount
category
```

Analyze:

* Customer segments
* Correlation
* Feature interactions
* PCA
* Group differences

---

## 📚 Project 4 — Student Performance

Variables:

```text
study_hours
attendance
sleep_hours
previous_score
internet_usage
final_score
```

Investigate:

```text
Multiple Features
       ↓
Final Score
```

---

# 🧠 Exercises

## 🟢 Beginner

1. Create a correlation matrix.
2. Create a correlation heatmap.
3. Create a pair plot.
4. Create a scatter matrix.
5. Group data using two categorical variables.
6. Create a multivariate scatter plot.
7. Create a 3D scatter plot.
8. Identify strongly correlated features.
9. Calculate grouped statistics.
10. Explain three relationships discovered.

---

## 🟡 Intermediate

1. Analyze multicollinearity.
2. Calculate VIF.
3. Compare grouped relationships.
4. Create faceted plots.
5. Analyze interactions between variables.
6. Apply standardization.
7. Perform PCA.
8. Calculate explained variance.
9. Visualize PCA components.
10. Interpret PCA loadings.

---

## 🔴 Advanced

1. Build a complete multivariate EDA pipeline.
2. Compare PCA with feature selection.
3. Investigate multicollinearity before and after feature selection.
4. Analyze interaction terms in a regression model.
5. Investigate nonlinear multivariate relationships.
6. Compare multiple dimensionality reduction techniques.
7. Analyze clustering after dimensionality reduction.
8. Build a leakage-safe preprocessing pipeline.
9. Study Simpson's paradox using grouped data.
10. Create an automated multivariate EDA report.

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
│   ├── README.md
│   ├── correlation-matrix.py
│   ├── pair-plot.py
│   ├── multivariate-visualization.py
│   ├── multicollinearity.py
│   ├── pca-analysis.py
│   └── multivariate-report.py
│
├── 04-Data-Visualization/
│
└── 05-EDA-Projects/
```

---

# 💻 End-to-End Example

```python
"""
Multivariate Analysis
---------------------
Explore relationships among multiple variables.

Requirements:
    pip install pandas matplotlib seaborn scikit-learn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def create_dataset():
    """Create a demonstration dataset."""

    return pd.DataFrame({
        "age": [22, 25, 28, 32, 35, 38, 42, 45],
        "experience": [1, 3, 5, 7, 10, 12, 15, 18],
        "income": [30, 35, 42, 50, 60, 70, 85, 100],
        "performance": [65, 70, 75, 78, 82, 85, 88, 92],
        "department": [
            "HR",
            "HR",
            "IT",
            "IT",
            "IT",
            "Sales",
            "Sales",
            "Sales"
        ]
    })


def correlation_analysis(df):
    """Calculate and display numerical correlations."""

    numeric_columns = [
        "age",
        "experience",
        "income",
        "performance"
    ]

    correlation = df[numeric_columns].corr()

    print("\nCorrelation Matrix")
    print("-" * 40)
    print(correlation)

    return correlation


def visualize_correlation(correlation):
    """Display a correlation heatmap."""

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def pair_plot(df):
    """Display pairwise numerical relationships."""

    sns.pairplot(
        df,
        vars=[
            "age",
            "experience",
            "income",
            "performance"
        ],
        hue="department"
    )

    plt.show()


def perform_pca(df):
    """Perform PCA on standardized numerical features."""

    features = [
        "age",
        "experience",
        "income",
        "performance"
    ]

    X = df[features]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    pca = PCA(
        n_components=2
    )

    X_pca = pca.fit_transform(X_scaled)

    print("\nExplained Variance")
    print("-" * 40)

    for index, variance in enumerate(
        pca.explained_variance_ratio_,
        start=1
    ):
        print(
            f"PC{index}: "
            f"{variance:.4f}"
        )

    return X_pca, pca


def visualize_pca(X_pca, df):
    """Visualize the PCA projection."""

    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1]
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection")

    plt.tight_layout()
    plt.show()


def main():
    """Run the complete multivariate analysis."""

    df = create_dataset()

    print("Dataset")
    print("-" * 40)
    print(df)

    correlation = correlation_analysis(df)

    visualize_correlation(
        correlation
    )

    pair_plot(df)

    X_pca, pca = perform_pca(df)

    visualize_pca(
        X_pca,
        df
    )


if __name__ == "__main__":
    main()
```

---

# 🔬 Multivariate Analysis Workflow

A professional workflow:

```text
                         DATASET
                            │
                            ↓
                    Understand Variables
                            │
                            ↓
                  Check Data Types
                            │
                            ↓
                    Check Missing Data
                            │
                            ↓
                     Check Outliers
                            │
                            ↓
                  Select Relevant Features
                            │
              ┌─────────────┼─────────────┐
              ↓             ↓             ↓
         Correlation     Pair Plot     Group Analysis
              │             │             │
              └─────────────┼─────────────┘
                            ↓
                    Check Interactions
                            ↓
                  Check Multicollinearity
                            ↓
                  Consider Dimensionality
                            ↓
                          PCA
                            ↓
                  Interpret Structure
                            ↓
                 Feature Engineering
                            ↓
                    Machine Learning
```

---

# 📋 Multivariate Analysis Checklist

## 🔍 Dataset

* [ ] Understand feature meanings.
* [ ] Identify numerical variables.
* [ ] Identify categorical variables.
* [ ] Identify date-time variables.
* [ ] Check missing values.
* [ ] Check duplicate records.
* [ ] Check outliers.

## 📊 Relationships

* [ ] Calculate correlation matrix.
* [ ] Create heatmap.
* [ ] Create pair plot.
* [ ] Investigate important relationships.
* [ ] Analyze grouped patterns.
* [ ] Look for interactions.

## ⚠️ Modeling Concerns

* [ ] Check multicollinearity.
* [ ] Check feature redundancy.
* [ ] Check target leakage.
* [ ] Consider scaling.
* [ ] Consider dimensionality reduction.

## 🧠 PCA

* [ ] Handle missing values.
* [ ] Select numerical features.
* [ ] Standardize when appropriate.
* [ ] Fit PCA on training data in predictive workflows.
* [ ] Check explained variance.
* [ ] Analyze loadings.
* [ ] Visualize components.

---

# 🧭 Decision Guide

```text
              Multiple Variables
                     │
                     ↓
             What do you need?
                     │
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
 Relationships   Visualization   Reduction
       │             │             │
       ↓             ↓             ↓
 Correlation     Pair Plot        PCA
 Heatmap         Faceting         SVD
 Grouping        3D Plot          UMAP*
 Interactions    Parallel Coord.  t-SNE*
```

* These methods are commonly used for dimensionality reduction/visualization but have different goals and interpretation considerations.

---

# 🗺️ EDA Roadmap

You have now progressed through the core relationship-analysis stages:

```text
06-Exploratory-Data-Analysis/
│
├── 01-Univariate-Analysis
│       ↓
│   One Variable
│
├── 02-Bivariate-Analysis
│       ↓
│   Two Variables
│
├── 03-Multivariate-Analysis
│       ↓
│   Multiple Variables
│
├── 04-Data-Visualization
│       ↓
│   Communicate Insights
│
└── 05-EDA-Projects
        ↓
    Real-World EDA
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
🧠 UNDERSTAND
  ↓
📉 REDUCE
  ↓
🎯 SELECT
  ↓
🤖 MODEL
  ↓
🚀 BUILD
```

---

# 🎯 Key Takeaways

> **Multivariate analysis helps us understand complex datasets where multiple variables interact simultaneously.**

Remember:

1. **Real-world problems usually involve multiple variables.**
2. **Correlation matrices provide a useful first overview.**
3. **Pair plots reveal pairwise patterns across multiple variables.**
4. **Grouped analysis can reveal differences hidden in aggregate data.**
5. **Interactions can change the effect of one feature depending on another.**
6. **Multicollinearity can make model coefficients difficult to interpret.**
7. **VIF is one diagnostic for multicollinearity.**
8. **PCA transforms features into principal components.**
9. **Scaling is often important before PCA.**
10. **PCA is dimensionality reduction, not ordinary feature selection.**
11. **Outliers can strongly influence multivariate methods.**
12. **Missing values must be handled before methods that cannot accept them.**
13. **Preprocessing must be leakage-safe in predictive ML workflows.**
14. **Visualization and statistical analysis should complement each other.**
15. **Multivariate EDA prepares the dataset for feature engineering and modeling.**

---

# 🚀 Next Step

After understanding relationships among multiple variables, the next stage is learning how to **communicate those discoveries visually and effectively**.

Continue with:

```text
06-Exploratory-Data-Analysis/
└── 04-Data-Visualization/
```

You will work with:

* Matplotlib
* Seaborn
* Distribution plots
* Relationship plots
* Categorical plots
* Statistical plots
* Advanced visualizations
* Custom dashboards
* EDA storytelling
* Visualization best practices

The journey becomes:

```text
🌱 DATA
  ↓
🔍 INSPECT
  ↓
📊 UNIVARIATE
  ↓
🔗 BIVARIATE
  ↓
🧠 MULTIVARIATE
  ↓
🎨 VISUALIZE
  ↓
💡 DISCOVER
  ↓
🤖 MODEL
  ↓
🚀 BUILD
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
* Fixing technical errors
* Adding datasets
* Adding visualization examples
* Improving PCA demonstrations
* Adding exercises
* Adding real-world EDA projects
* Improving documentation

### Contribution Workflow

```bash
git clone https://github.com/Kishor055/Machine-Learning.git

cd Machine-Learning

git checkout -b feature/improve-multivariate-analysis
```

Make your changes, test the examples, and submit a pull request.

---

# ⭐ Support

If this repository helps you learn Machine Learning:

* ⭐ Star the repository
* 🍴 Fork the repository
* 🐛 Report issues
* 💡 Suggest improvements
* 🤝 Contribute examples

Your support helps improve this learning resource.

---

# 📄 License

This project is intended for educational and learning purposes.

Please check the repository's license file for the applicable terms.

---

<div align="center">

### 🌱 GROW → CONNECT → EXPLORE → UNDERSTAND → BUILD 🚀

**From individual features to complex relationships — understand the data before building the model.**

</div>
