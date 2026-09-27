This module provides a practical, beginner-to-advanced introduction to
datasets in Scikit-Learn.

Topics covered
Understanding the Scikit-Learn dataset API
Loading built-in classification datasets
Loading built-in regression datasets
Exploring dataset structure
Converting Bunch objects to Pandas DataFrames
Separating features and targets
Working with target names
Loading datasets with return_X_y
Generating synthetic datasets
Binary and multiclass classification datasets
Regression datasets
Clustering datasets
Non-linear datasets
Controlling random_state
Train/test splitting
Stratified splitting for classification
Dataset validation
Feature and target inspection
Practical ML dataset preparation
Reusable dataset helper functions
Common mistakes and best practices
Requirements

pip install numpy pandas scikit-learn matplotlib

Run

python datasets.py

Author

Kishor Patil
"""

=============================================================================
1. IMPORTS
=============================================================================

from future import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.datasets import (
load_breast_cancer,
load_diabetes,
load_iris,
load_wine,
make_blobs,
make_classification,
make_moons,
make_regression,
)
from sklearn.model_selection import train_test_split

=============================================================================
2. HELPER FUNCTIONS
=============================================================================

def section(title: str) -> None:
"""Print a formatted section heading."""
print("\n" + "=" * 80)
print(title)
print("=" * 80)

def print_dataset_summary(
dataset: Any,
name: str,
) -> None:
"""
Print a compact summary of a Scikit-Learn dataset.

Parameters
----------
dataset:
    A Scikit-Learn Bunch dataset.
name:
    Human-readable dataset name.
"""
print(f"\n{name}")
print("-" * len(name))

print(f"Type:              {type(dataset).__name__}")
print(f"Features shape:    {dataset.data.shape}")
print(f"Target shape:      {dataset.target.shape}")
print(f"Feature count:     {dataset.data.shape[1]}")
print(f"Sample count:      {dataset.data.shape[0]}")

if hasattr(dataset, "feature_names"):
    print(f"Feature names:     {len(dataset.feature_names)}")

if hasattr(dataset, "target_names"):
    print(f"Target names:      {dataset.target_names}")

def validate_supervised_dataset(
X: np.ndarray,
y: np.ndarray,
) -> None:
"""
Perform basic validation on a supervised learning dataset.

This is intentionally lightweight. More advanced validation can be
performed later with preprocessing pipelines and domain-specific rules.
"""
if len(X) != len(y):
    raise ValueError("X and y must contain the same number of samples.")

if X.ndim != 2:
    raise ValueError("X must be a 2-dimensional feature matrix.")

if y.ndim != 1:
    raise ValueError("y must be a 1-dimensional target array.")

if len(X) == 0:
    raise ValueError("Dataset must contain at least one sample.")

if not np.isfinite(X).all():
    raise ValueError("X contains NaN or infinite values.")

if not np.isfinite(y).all():
    raise ValueError("y contains NaN or infinite values.")

print("Dataset validation: PASSED")
=============================================================================
3. UNDERSTANDING THE DATASET API
=============================================================================

def demonstrate_dataset_api() -> None:
"""Explain the common structure of Scikit-Learn datasets."""

section("3. Understanding the Scikit-Learn Dataset API")

dataset = load_iris()

print("Scikit-Learn built-in datasets commonly provide:")
print()
print("dataset.data          -> feature matrix X")
print("dataset.target        -> target vector y")
print("dataset.feature_names -> names of input features")
print("dataset.target_names  -> human-readable class names")
print("dataset.DESCR         -> dataset documentation")
print("dataset.keys()        -> available dataset fields")
print()

print("Available fields:")
print(dataset.keys())
=============================================================================
4. IRIS DATASET
=============================================================================

def load_iris_dataset() -> None:
"""
Load and inspect the classic Iris classification dataset.

Iris contains measurements of iris flowers and three target classes.
"""

section("4. Iris Classification Dataset")

iris = load_iris()

print_dataset_summary(iris, "Iris Dataset")

print("\nFeature names:")
for index, feature in enumerate(iris.feature_names, start=1):
    print(f"{index}. {feature}")

print("\nTarget names:")
for index, target in enumerate(iris.target_names):
    print(f"{index}. {target}")

print("\nFirst five feature rows:")
print(iris.data[:5])

print("\nFirst five targets:")
print(iris.target[:5])
=============================================================================
5. WINE DATASET
=============================================================================

def load_wine_dataset() -> None:
"""Load and inspect the Wine classification dataset."""

section("5. Wine Classification Dataset")

wine = load_wine()

print_dataset_summary(wine, "Wine Dataset")

print("\nFeature names:")
print(wine.feature_names)

print("\nUnique target classes:")
print(np.unique(wine.target))

print("\nClass distribution:")
unique, counts = np.unique(wine.target, return_counts=True)

for class_id, count in zip(unique, counts):
    print(f"Class {class_id}: {count} samples")
=============================================================================
6. BREAST CANCER DATASET
=============================================================================

def load_breast_cancer_dataset() -> None:
"""Load and inspect the Breast Cancer Wisconsin dataset."""

section("6. Breast Cancer Classification Dataset")

cancer = load_breast_cancer()

print_dataset_summary(cancer, "Breast Cancer Dataset")

print("\nTarget names:")
print(cancer.target_names)

print("\nFirst five target values:")
print(cancer.target[:5])

print("\nTarget distribution:")

unique, counts = np.unique(cancer.target, return_counts=True)

for class_id, count in zip(unique, counts):
    class_name = cancer.target_names[class_id]
    print(f"{class_id} ({class_name}): {count}")
=============================================================================
7. DIABETES DATASET
=============================================================================

def load_diabetes_dataset() -> None:
"""Load and inspect the Diabetes regression dataset."""

section("7. Diabetes Regression Dataset")

diabetes = load_diabetes()

print_dataset_summary(diabetes, "Diabetes Dataset")

print("\nFeature names:")
print(diabetes.feature_names)

print("\nFirst five feature rows:")
print(diabetes.data[:5])

print("\nFirst five target values:")
print(diabetes.target[:5])

print("\nTarget statistics:")
print(f"Mean:   {diabetes.target.mean():.2f}")
print(f"Std:    {diabetes.target.std():.2f}")
print(f"Min:    {diabetes.target.min():.2f}")
print(f"Max:    {diabetes.target.max():.2f}")
=============================================================================
8. RETURN_X_Y
=============================================================================

def demonstrate_return_x_y() -> None:
"""
Demonstrate the return_X_y=True option.

This is useful when metadata such as feature names is not required.
"""

section("8. Loading X and y Directly")

X, y = load_iris(return_X_y=True)

print(f"X type:   {type(X).__name__}")
print(f"y type:   {type(y).__name__}")
print(f"X shape:   {X.shape}")
print(f"y shape:   {y.shape}")

print("\nFirst sample:")
print("X[0] =", X[0])
print("y[0] =", y[0])
=============================================================================
9. CONVERTING TO PANDAS
=============================================================================

def iris_to_dataframe() -> pd.DataFrame:
"""
Convert the Iris dataset into a Pandas DataFrame.

Returns
-------
pd.DataFrame
    DataFrame containing features and target labels.
"""

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names,
)

df["target"] = iris.target

df["target_name"] = pd.Categorical.from_codes(
    iris.target,
    iris.target_names,
)

return df

def demonstrate_dataframe_conversion() -> None:
"""Convert a Scikit-Learn dataset into a Pandas DataFrame."""

section("9. Converting Scikit-Learn Data to Pandas")

df = iris_to_dataframe()

print("DataFrame shape:")
print(df.shape)

print("\nDataFrame:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nTarget distribution:")
print(df["target_name"].value_counts())
=============================================================================
10. FEATURE / TARGET SEPARATION
=============================================================================

def demonstrate_feature_target_split() -> None:
"""Demonstrate the standard X/y convention."""

section("10. Separating Features and Target")

iris = load_iris()

X = iris.data
y = iris.target

print("X = Features")
print("y = Target")
print()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\nFirst feature row:")
print(X[0])

print("\nFirst target:")
print(y[0])

print("\nImportant:")
print("- X contains input variables.")
print("- y contains the value the model learns to predict.")
print("- Keep X and y aligned by row.")
=============================================================================
11. SYNTHETIC CLASSIFICATION DATA
=============================================================================

def generate_binary_classification() -> tuple[np.ndarray, np.ndarray]:
"""
Generate a synthetic binary classification dataset.

Returns
-------
X, y
    Feature matrix and binary target.
"""

X, y = make_classification(
    n_samples=500,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_repeated=0,
    n_classes=2,
    class_sep=1.2,
    random_state=42,
)

return X, y

def demonstrate_binary_classification() -> None:
"""Generate and inspect a synthetic binary classification dataset."""

section("11. Synthetic Binary Classification Dataset")

X, y = generate_binary_classification()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\nClass distribution:")
unique, counts = np.unique(y, return_counts=True)

for class_id, count in zip(unique, counts):
    print(f"Class {class_id}: {count}")

print("\nFirst five samples:")
print(X[:5])

print("\nFirst five targets:")
print(y[:5])
=============================================================================
12. MULTICLASS CLASSIFICATION
=============================================================================

def demonstrate_multiclass_classification() -> None:
"""Generate a synthetic multiclass classification dataset."""

section("12. Synthetic Multiclass Classification")

X, y = make_classification(
    n_samples=600,
    n_features=8,
    n_informative=5,
    n_redundant=1,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42,
)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

unique, counts = np.unique(y, return_counts=True)

print("\nClass distribution:")

for class_id, count in zip(unique, counts):
    print(f"Class {class_id}: {count}")
=============================================================================
13. REGRESSION DATASET
=============================================================================

def demonstrate_regression_generation() -> None:
"""Generate a synthetic regression dataset."""

section("13. Synthetic Regression Dataset")

X, y = make_regression(
    n_samples=500,
    n_features=6,
    n_informative=4,
    noise=15.0,
    random_state=42,
)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\nFirst five feature rows:")
print(X[:5])

print("\nFirst five target values:")
print(y[:5])

print("\nTarget statistics:")
print(f"Mean: {y.mean():.2f}")
print(f"Std:  {y.std():.2f}")
print(f"Min:  {y.min():.2f}")
print(f"Max:  {y.max():.2f}")
=============================================================================
14. CLUSTERING DATASET
=============================================================================

def demonstrate_blobs() -> None:
"""Generate synthetic data suitable for clustering algorithms."""

section("14. Synthetic Clustering Dataset")

X, y = make_blobs(
    n_samples=500,
    centers=4,
    n_features=2,
    cluster_std=1.2,
    random_state=42,
)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\nGenerated cluster labels:")
print(np.unique(y))

print("\nFirst five samples:")
print(X[:5])

print(
    "\nNote: y represents the generated cluster identity. "
    "Unsupervised clustering algorithms may not use y during training."
)
=============================================================================
15. NON-LINEAR DATASET
=============================================================================

def demonstrate_moons() -> None:
"""
Generate a two-moons dataset.

This dataset is useful for demonstrating why some classification
problems require non-linear decision boundaries.
"""

section("15. Non-Linear Two-Moons Dataset")

X, y = make_moons(
    n_samples=500,
    noise=0.15,
    random_state=42,
)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\nFirst five samples:")
print(X[:5])

print("\nClass distribution:")

unique, counts = np.unique(y, return_counts=True)

for class_id, count in zip(unique, counts):
    print(f"Class {class_id}: {count}")
=============================================================================
16. RANDOM STATE
=============================================================================

def demonstrate_random_state() -> None:
"""
Demonstrate reproducible synthetic dataset generation.

random_state makes experiments reproducible when the same random
generation procedure and parameters are used.
"""

section("16. Reproducibility with random_state")

X1, y1 = make_classification(
    n_samples=100,
    n_features=4,
    random_state=42,
)

X2, y2 = make_classification(
    n_samples=100,
    n_features=4,
    random_state=42,
)

X3, y3 = make_classification(
    n_samples=100,
    n_features=4,
    random_state=7,
)

print("Same random_state:")
print("X1 == X2:", np.array_equal(X1, X2))
print("y1 == y2:", np.array_equal(y1, y2))

print("\nDifferent random_state:")
print("X1 == X3:", np.array_equal(X1, X3))
print("y1 == y3:", np.array_equal(y1, y3))

print(
    "\nBest practice: use a fixed random_state during tutorials, "
    "experiments, and reproducible development."
)
=============================================================================
17. TRAIN / TEST SPLIT
=============================================================================

def demonstrate_train_test_split() -> None:
"""Split a dataset into training and testing subsets."""

section("17. Train/Test Split")

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Original dataset:")
print(f"X: {X.shape}")
print(f"y: {y.shape}")

print("\nTraining dataset:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")

print("\nTesting dataset:")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")
=============================================================================
18. STRATIFIED SPLITTING
=============================================================================

def demonstrate_stratification() -> None:
"""
Demonstrate stratified train/test splitting.

Stratification attempts to preserve class proportions across
training and testing subsets.
"""

section("18. Stratified Train/Test Split")

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Original class distribution:")
print(pd.Series(y).value_counts(normalize=True).sort_index())

print("\nTraining class distribution:")
print(
    pd.Series(y_train)
    .value_counts(normalize=True)
    .sort_index()
)

print("\nTesting class distribution:")
print(
    pd.Series(y_test)
    .value_counts(normalize=True)
    .sort_index()
)
=============================================================================
19. DATASET INSPECTION
=============================================================================

def inspect_dataset(
X: np.ndarray,
y: np.ndarray,
feature_names: list[str] | None = None,
) -> None:
"""
Print useful information about an ML dataset.
"""

section("19. Dataset Inspection")

print(f"Number of samples:  {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Target length:      {len(y)}")

print(f"\nX dtype: {X.dtype}")
print(f"y dtype: {y.dtype}")

print(f"\nX contains NaN:      {np.isnan(X).any()}")
print(f"X contains infinity: {np.isinf(X).any()}")

if feature_names is not None:
    print("\nFeature names:")
    for index, name in enumerate(feature_names):
        print(f"{index}: {name}")

print("\nFeature means:")
print(np.mean(X, axis=0))

print("\nFeature standard deviations:")
print(np.std(X, axis=0))
=============================================================================
20. PRACTICAL CLASSIFICATION DATASET
=============================================================================

def prepare_classification_dataset() -> tuple[
np.ndarray,
np.ndarray,
np.ndarray,
np.ndarray,
]:
"""
Prepare a classification dataset for machine learning.

Returns
-------
X_train, X_test, y_train, y_test
"""

iris = load_iris()

X = iris.data
y = iris.target

validate_supervised_dataset(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

return X_train, X_test, y_train, y_test

def demonstrate_classification_preparation() -> None:
"""Prepare Iris data for a supervised ML workflow."""

section("20. Practical Classification Dataset Preparation")

X_train, X_test, y_train, y_test = prepare_classification_dataset()

print("Prepared Iris classification dataset:")
print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test:  {y_test.shape}")

print("\nReady for:")
print("- preprocessing")
print("- model training")
print("- cross-validation")
print("- hyperparameter tuning")
print("- evaluation")
=============================================================================
21. PRACTICAL REGRESSION DATASET
=============================================================================

def prepare_regression_dataset() -> tuple[
np.ndarray,
np.ndarray,
np.ndarray,
np.ndarray,
]:
"""Prepare the built-in Diabetes dataset for regression."""

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

validate_supervised_dataset(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

return X_train, X_test, y_train, y_test

def demonstrate_regression_preparation() -> None:
"""Prepare a regression dataset for machine learning."""

section("21. Practical Regression Dataset Preparation")

X_train, X_test, y_train, y_test = prepare_regression_dataset()

print("Prepared Diabetes regression dataset:")
print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test:  {y_test.shape}")
=============================================================================
22. DATASET COMPARISON
=============================================================================

def compare_builtin_datasets() -> None:
"""Compare several commonly used built-in Scikit-Learn datasets."""

section("22. Built-in Dataset Comparison")

datasets = {
    "Iris": load_iris(),
    "Wine": load_wine(),
    "Breast Cancer": load_breast_cancer(),
    "Diabetes": load_diabetes(),
}

rows = []

for name, dataset in datasets.items():
    problem_type = (
        "Regression"
        if name == "Diabetes"
        else "Classification"
    )

    rows.append(
        {
            "Dataset": name,
            "Problem": problem_type,
            "Samples": dataset.data.shape[0],
            "Features": dataset.data.shape[1],
            "Target Shape": dataset.target.shape,
        }
    )

comparison = pd.DataFrame(rows)

print(comparison.to_string(index=False))
=============================================================================
23. DATASET TO DATAFRAME HELPER
=============================================================================

def sklearn_dataset_to_dataframe(dataset: Any) -> pd.DataFrame:
"""
Convert a Scikit-Learn Bunch dataset into a DataFrame.

The function automatically uses feature_names when available.
The target is appended as a separate column.

Parameters
----------
dataset:
    Scikit-Learn dataset returned by a load_* function.

Returns
-------
pd.DataFrame
    DataFrame containing features and target.
"""

feature_names = getattr(
    dataset,
    "feature_names",
    None,
)

if feature_names is None:
    feature_names = [
        f"feature_{index}"
        for index in range(dataset.data.shape[1])
    ]

df = pd.DataFrame(
    dataset.data,
    columns=feature_names,
)

df["target"] = dataset.target

return df

def demonstrate_reusable_dataframe_helper() -> None:
"""Demonstrate the reusable dataset-to-DataFrame helper."""

section("23. Reusable Dataset-to-DataFrame Helper")

wine = load_wine()

df = sklearn_dataset_to_dataframe(wine)

print(df.head())
print("\nShape:", df.shape)
=============================================================================
24. DATA LEAKAGE WARNING
=============================================================================

def demonstrate_data_leakage_warning() -> None:
"""
Explain an important dataset-handling rule.

Preprocessing parameters should generally be learned from training data
only. Applying transformations using the complete dataset before splitting
can leak information from the test set into the training process.
"""

section("24. Data Leakage Warning")

print(
    """

Incorrect workflow:

Complete Dataset
      |
      v
Scale / Impute / Select Features
      |
      v
Train/Test Split

Potential problem:
The preprocessing step may use information from the future test set.

Preferred workflow:

Complete Dataset
      |
      v
Train/Test Split
   /       \\
  v         v

Train Test
|
v
Fit preprocessing
|
v
Transform Train and Test
using training-fitted parameters

In Scikit-Learn, Pipeline and ColumnTransformer are commonly used
to make this workflow safer and reproducible.
"""
)

=============================================================================
25. COMPLETE DATASET WORKFLOW
=============================================================================

def complete_dataset_workflow() -> None:
"""
Demonstrate a compact end-to-end dataset preparation workflow.

This function intentionally stops before model training. The goal of
datasets.py is to focus on obtaining, understanding, validating, and
splitting data.
"""

section("25. Complete Dataset Workflow")

iris = load_iris()

# Step 1: Extract features and target.
X = iris.data
y = iris.target

# Step 2: Validate.
validate_supervised_dataset(X, y)

# Step 3: Inspect.
print(f"Samples:  {X.shape[0]}")
print(f"Features: {X.shape[1]}")

# Step 4: Split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Step 5: Confirm shapes.
print("\nAfter train/test split:")
print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test:  {y_test.shape}")

print("\nDataset preparation completed successfully.")
=============================================================================
26. COMMON DATASET GENERATORS
=============================================================================

def dataset_generator_reference() -> None:
"""Print a quick reference for commonly used dataset generators."""

section("26. Dataset Generator Reference")

reference = {
    "load_iris": "Built-in multiclass classification",
    "load_wine": "Built-in multiclass classification",
    "load_breast_cancer": "Built-in binary classification",
    "load_diabetes": "Built-in regression",
    "make_classification": "Synthetic classification",
    "make_regression": "Synthetic regression",
    "make_blobs": "Synthetic clustered data",
    "make_moons": "Synthetic non-linear classification",
}

for function_name, description in reference.items():
    print(f"{function_name:<25} -> {description}")
=============================================================================
27. BEST PRACTICES
=============================================================================

def print_best_practices() -> None:
"""Print important dataset-handling best practices."""

section("27. Dataset Best Practices")

practices = [
    "Keep X and y aligned by sample.",
    "Use train/test splitting before fitting preprocessing.",
    "Use stratify=y for classification when appropriate.",
    "Use random_state for reproducible experiments.",
    "Inspect feature shapes and target shapes before training.",
    "Check for missing and infinite values.",
    "Check class distribution for classification problems.",
    "Understand whether the task is classification, regression, or clustering.",
    "Use Pandas for convenient exploratory inspection.",
    "Use Pipeline for leakage-resistant preprocessing.",
    "Never assume a built-in dataset represents your production data.",
    "Document dataset source, assumptions, and preprocessing decisions.",
]

for number, practice in enumerate(practices, start=1):
    print(f"{number:02d}. {practice}")
=============================================================================
28. MAIN
=============================================================================

def main() -> None:
"""Run the Scikit-Learn datasets tutorial."""

print("=" * 80)
print("SCIKIT-LEARN DATASETS TUTORIAL")
print("=" * 80)

demonstrate_dataset_api()
load_iris_dataset()
load_wine_dataset()
load_breast_cancer_dataset()
load_diabetes_dataset()

demonstrate_return_x_y()
demonstrate_dataframe_conversion()
demonstrate_feature_target_split()

demonstrate_binary_classification()
demonstrate_multiclass_classification()
demonstrate_regression_generation()
demonstrate_blobs()
demonstrate_moons()

demonstrate_random_state()
demonstrate_train_test_split()
demonstrate_stratification()

iris = load_iris()
inspect_dataset(
    iris.data,
    iris.target,
    iris.feature_names,
)

demonstrate_classification_preparation()
demonstrate_regression_preparation()
compare_builtin_datasets()
demonstrate_reusable_dataframe_helper()

demonstrate_data_leakage_warning()
complete_dataset_workflow()

dataset_generator_reference()
print_best_practices()

print("\n" + "=" * 80)
print("DATASETS TUTORIAL COMPLETED")
print("=" * 80)

if name == "main":
main()
