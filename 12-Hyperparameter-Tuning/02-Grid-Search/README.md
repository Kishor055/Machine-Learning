# 02. Grid Search – Hyperparameter Tuning

## 1. Introduction

Grid Search is an automated hyperparameter optimization technique used in Machine Learning to identify the best combination of hyperparameter values for a model.

It works by defining a set of possible values for each hyperparameter and systematically testing every possible combination.

Each combination is evaluated using a performance metric, and the combination that produces the best validation score is selected.

Grid Search is implemented in Scikit-learn using the `GridSearchCV` class.

## 2. What is Grid Search?

Grid Search is an exhaustive search technique that evaluates all possible combinations of specified hyperparameter values.

For example, consider a K-Nearest Neighbors (KNN) algorithm with two hyperparameters:

- `n_neighbors`: [3, 5, 7]
- `weights`: ['uniform', 'distance']

Grid Search tests all possible combinations:

| n_neighbors | weights |
|---|---|
| 3 | uniform |
| 3 | distance |
| 5 | uniform |
| 5 | distance |
| 7 | uniform |
| 7 | distance |

Total combinations:

\[
3 \times 2 = 6
\]

If we use 5-fold cross-validation:

\[
6 \times 5 = 30
\]

Therefore, the model is trained 30 times during the search.

## 3. Objectives of Grid Search

- Find the optimal hyperparameter combination.
- Improve machine learning model performance.
- Automate the hyperparameter tuning process.
- Reduce manual experimentation.
- Evaluate models using cross-validation.
- Improve model generalization.

## 4. Working Principle of Grid Search

Grid Search follows these steps:

1. Select a machine learning algorithm.
2. Identify the hyperparameters that need tuning.
3. Define a range of possible values for each hyperparameter.
4. Generate all possible combinations.
5. Train the model using each combination.
6. Evaluate each combination using cross-validation.
7. Calculate the average validation score.
8. Select the best-performing combination.
9. Train and evaluate the final model.

## 5. Important Parameters of GridSearchCV

Scikit-learn provides the `GridSearchCV` class for implementing Grid Search.

Syntax:

```python
GridSearchCV(
    estimator,
    param_grid,
    scoring=None,
    cv=None,
    refit=True,
    n_jobs=None,
    verbose=0
)
```

| Parameter | Description |
|---|---|
| estimator | Machine Learning model to optimize. |
| param_grid | Dictionary containing hyperparameter values. |
| scoring | Evaluation metric such as accuracy or F1-score. |
| cv | Number of cross-validation folds. |
| refit | Retrains the best model using all supplied training data. |
| n_jobs | Number of parallel jobs. |
| verbose | Controls the amount of output information. |

## 6. Implementation Using Python

We will implement Grid Search using the K-Nearest Neighbors (KNN) classification algorithm and the Iris dataset.

### Step 1: Import Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
```

### Step 2: Load the Dataset

```python
# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Dataset Shape:", X.shape)
print("Target Classes:", np.unique(y))
```

### Step 3: Split the Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
```

### Step 4: Create the KNN Model

```python
knn = KNeighborsClassifier()
```

### Step 5: Define the Hyperparameter Grid

```python
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
```

The search space contains:

\[
5 \times 2 \times 2 = 20
\]

Total combinations = 20.

### Step 6: Apply Grid Search

```python
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Train Grid Search
grid_search.fit(X_train, y_train)
```

### Step 7: Display Best Hyperparameters

```python
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Accuracy:")
print(grid_search.best_score_)
```

### Step 8: Evaluate the Best Model

```python
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Step 9: Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix - Grid Search")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.show()
```

### Step 10: Compare All Hyperparameter Combinations

```python
results = pd.DataFrame(grid_search.cv_results_)

results = results[
    [
        'params',
        'mean_test_score',
        'std_test_score',
        'rank_test_score'
    ]
]

results = results.sort_values(
    by='rank_test_score'
)

print(results.to_string(index=False))
```

### Step 11: Visualize Grid Search Results

```python
results_df = pd.DataFrame(grid_search.cv_results_)

plt.figure(figsize=(10, 5))

plt.plot(
    range(1, len(results_df) + 1),
    results_df['mean_test_score'],
    marker='o'
)

plt.xlabel("Hyperparameter Combination")
plt.ylabel("Mean Cross Validation Accuracy")
plt.title("Grid Search Performance")

plt.grid(True)
plt.show()
```

## 7. Understanding Cross-Validation in Grid Search

Cross-validation is used to evaluate model performance more reliably.

In 5-fold cross-validation:

- The training dataset is divided into five subsets.
- Four subsets are used for training.
- One subset is used for validation.
- The process is repeated five times.
- The average validation score is calculated.

The average score is:

\[
CV_{score} = \frac{1}{K}\sum_{i=1}^{K} Score_i
\]

Where:

- K = Number of folds.
- Score = Validation performance for each fold.

## 8. Advantages of Grid Search

1. **Systematic Search:** Tests every specified hyperparameter combination.
2. **Automated Optimization:** Reduces manual tuning effort.
3. **Cross-Validation:** Provides a more reliable performance estimate.
4. **Easy Implementation:** Available directly through Scikit-learn.
5. **Reproducibility:** Produces consistent results when data splitting and random states are controlled.
6. **Best Combination:** Identifies the highest-scoring configuration within the specified search space.

## 9. Disadvantages of Grid Search

1. **Computationally Expensive:** Tests every specified combination.
2. **Time Consuming:** Requires multiple model training operations.
3. **Large Search Space:** Becomes inefficient when many hyperparameters are involved.
4. **Resource Intensive:** May require significant memory and processing power.
5. **Limited Search Space:** Cannot find better values outside the predefined grid.

## 10. Applications of Grid Search

- Optimizing KNN classification models.
- Tuning Decision Tree hyperparameters.
- Improving Random Forest performance.
- Selecting optimal SVM parameters.
- Optimizing Logistic Regression.
- Improving machine learning classification and regression models.

## 11. Grid Search vs Manual Tuning

| Feature | Manual Tuning | Grid Search |
|---|---|---|
| Process | Manually selects values | Automatically tests combinations |
| Execution | Human-controlled | Systematic |
| Search | Limited experiments | Exhaustive within specified grid |
| Cross-validation | Optional/manual | Supported directly |
| Time | Depends on experiments | Can be expensive |
| Best use | Small experiments | Defined search spaces |

## 12. Best Practices

- Start with a reasonable hyperparameter search space.
- Use cross-validation for model selection.
- Select an appropriate scoring metric.
- Avoid unnecessarily large grids.
- Use `n_jobs=-1` when sufficient computational resources are available.
- Keep the test dataset separate from hyperparameter tuning.
- Consider Randomized Search for large search spaces.
- Record the best parameters and validation scores.

## 13. Key Takeaways

- Grid Search is an automated hyperparameter optimization technique.
- It evaluates all specified hyperparameter combinations.
- `GridSearchCV` is the Scikit-learn implementation.
- Cross-validation helps estimate model performance.
- `best_params_` returns the best hyperparameter combination.
- `best_score_` returns the best mean cross-validation score.
- Grid Search is effective for small and medium-sized search spaces.

## 14. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## 15. Conclusion

Grid Search is an important hyperparameter optimization technique in Machine Learning. It systematically evaluates different combinations of hyperparameters using cross-validation to identify the best-performing model configuration.

Although it can become computationally expensive for large search spaces, it is a reliable and easy-to-implement method for optimizing machine learning models.

---

**Next Topic:** `12-Hyperparameter-Tuning/03-Random-Search/README.md`
