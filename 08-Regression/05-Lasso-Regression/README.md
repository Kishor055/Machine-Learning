# Lasso Regression

Lasso Regression (**Least Absolute Shrinkage and Selection Operator**) is a regularized linear regression algorithm that reduces overfitting by adding an **L1 penalty** to the loss function. Unlike Ridge Regression, Lasso can shrink some feature coefficients exactly to zero, making it useful for automatic feature selection.

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Why Lasso Regression?](#2-why-lasso-regression)
- [3. Mathematical Formula](#3-mathematical-formula)
- [4. How Lasso Regression Works](#4-how-lasso-regression-works)
- [5. Important Parameters](#5-important-parameters)
- [6. Implementation Using Scikit-Learn](#6-implementation-using-scikit-learn)
- [7. Lasso Regression from Scratch](#7-lasso-regression-from-scratch)
- [8. Feature Selection](#8-feature-selection)
- [9. Lasso vs. Ridge vs. Linear Regression](#9-lasso-vs-ridge-vs-linear-regression)
- [10. Advantages](#10-advantages)
- [11. Limitations](#11-limitations)
- [12. Applications](#12-applications)
- [13. Best Practices](#13-best-practices)
- [14. Practice Exercises](#14-practice-exercises)
- [15. Interview Questions](#15-interview-questions)
- [16. References](#16-references)

---

## 1. Introduction

Lasso Regression is a supervised machine learning algorithm used to predict a continuous target variable. It extends ordinary linear regression by penalizing the absolute values of model coefficients.

This penalty helps control model complexity and can eliminate less useful features by setting their coefficients to zero.

### Key characteristics

- Uses L1 regularization.
- Reduces the impact of less important features.
- Can perform automatic feature selection.
- Helps reduce overfitting.
- Works best when the relationship between features and target is reasonably represented by a linear model.
- Requires careful parameter tuning, especially when features are correlated.

## 2. Why Lasso Regression?

Ordinary Linear Regression minimizes prediction error without explicitly penalizing large coefficients. With many features or noisy data, this can result in an unnecessarily complex model.

Lasso addresses this by adding a penalty for the absolute magnitude of the coefficients.

### Example

Suppose a house-price prediction model uses these features:

| Feature | Linear Regression coefficient | Possible Lasso coefficient |
|---|---:|---:|
| Area | 120 | 110 |
| Bedrooms | 15 | 8 |
| Age | -20 | -12 |
| Irrelevant feature | 0.8 | 0 |

The Lasso coefficients above are illustrative. The exact values depend on the dataset, feature scaling, and regularization strength.

Lasso can set the coefficient of an unhelpful feature to zero, effectively removing its direct contribution to the prediction.

## 3. Mathematical Formula

### Linear Regression loss

For \(n\) training samples, ordinary least squares minimizes the mean squared error:

\[
\operatorname{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

where:

- \(y_i\): actual target value
- \(\hat y_i\): predicted target value
- \(n\): number of samples

### Lasso Regression objective

Lasso adds an L1 penalty to the squared-error objective:

\[
\min_{\beta_0,\boldsymbol{\beta}}
\left[
\frac{1}{2n}\sum_{i=1}^{n}
(y_i-\beta_0-\mathbf{x}_i^T\boldsymbol{\beta})^2
+\alpha\sum_{j=1}^{p}|\beta_j|
\right]
\]

where:

- \(\beta_0\): intercept
- \(\beta_j\): coefficient for feature \(j\)
- \(p\): number of features
- \(\alpha\): regularization strength
- \(\sum_{j=1}^{p}|\beta_j|\): L1 penalty

The intercept is generally not penalized by Scikit-Learn's `Lasso` estimator when `fit_intercept=True`.

### Effect of alpha

| Alpha value | Regularization effect | Typical behavior |
|---|---|---|
| `0` | No L1 penalty | Use Linear Regression instead |
| Small | Weak | More coefficients remain nonzero |
| Moderate | Stronger | Some coefficients may become zero |
| Large | Very strong | More coefficients may become zero; underfitting is possible |

In Scikit-Learn, `alpha` controls the strength of regularization. It should be selected using validation or cross-validation.

## 4. How Lasso Regression Works

1. Start with a dataset containing features and a continuous target.
2. Fit a linear model to predict the target.
3. Calculate prediction error.
4. Add the L1 penalty for the magnitude of the coefficients.
5. Optimize the objective to balance prediction accuracy and model simplicity.
6. Some coefficients may become exactly zero.
7. Evaluate the fitted model on unseen data.

### Workflow

```text
Dataset
   |
   v
Data Cleaning
   |
   v
Train-Test Split
   |
   v
Feature Scaling
   |
   v
Lasso Regression
   |
   v
L1 Regularization
   |
   v
Feature Selection
   |
   v
Prediction and Evaluation
```

## 5. Important Parameters

Scikit-Learn implementation:

```python
from sklearn.linear_model import Lasso

model = Lasso(
    alpha=1.0,
    fit_intercept=True,
    max_iter=10000,
    tol=0.0001,
    selection="cyclic",
    random_state=42,
)
```

| Parameter | Description |
|---|---|
| `alpha` | Controls the L1 regularization strength |
| `fit_intercept` | Determines whether to estimate an intercept |
| `max_iter` | Maximum number of optimization iterations |
| `tol` | Tolerance for optimization convergence |
| `selection` | Coefficient update strategy: `cyclic` or `random` |
| `random_state` | Controls randomness when applicable, such as with `selection="random"` |

**Important:** `alpha=0` is not recommended for `Lasso`; use `LinearRegression` for ordinary least squares.

## 6. Implementation Using Scikit-Learn

### Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Example: Lasso Regression

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# 1. Generate a dataset with multiple features
X, y = make_regression(
    n_samples=500,
    n_features=10,
    n_informative=5,
    noise=20,
    random_state=42,
)

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# 3. Scale features and train Lasso
model = make_pipeline(
    StandardScaler(),
    Lasso(alpha=1.0, max_iter=10000),
)

model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)

# 5. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Lasso Regression Evaluation")
print(f"MAE:  {mae:.3f}")
print(f"MSE:  {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")

# 6. Inspect coefficients
lasso = model.named_steps["lasso"]

print("\nFeature Coefficients:")
for index, coefficient in enumerate(lasso.coef_):
    print(f"Feature {index + 1}: {coefficient:.4f}")

print("\nNumber of zero coefficients:",
      np.sum(lasso.coef_ == 0))

# 7. Actual vs. predicted plot
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.7)

lower = min(y_test.min(), y_pred.min())
upper = max(y_test.max(), y_pred.max())

plt.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--",
    color="red",
    label="Perfect prediction",
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Lasso Regression: Actual vs. Predicted")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

### Understanding the output

- **MAE:** Average absolute prediction error.
- **MSE:** Average squared prediction error.
- **RMSE:** Square root of MSE, expressed in the target's units.
- **R²:** Measures performance relative to predicting the target mean; it can be negative on test data.
- **Zero coefficients:** Count of features that Lasso has excluded from its linear prediction.

Because the features are standardized, the fitted coefficients correspond to the scaled features. They are useful for comparing relative coefficient magnitudes within this fitted model.

## 7. Lasso Regression from Scratch

The L1 penalty is not differentiable at zero, so a simple gradient-descent implementation must account for this. One educational approach is to use the subgradient of the absolute-value penalty.

For coefficient \(\beta_j\), the subgradient of \(|\beta_j|\) is:

\[
\partial |\beta_j|=
\begin{cases}
1 & \beta_j>0\\
-1 & \beta_j<0\\
[-1,1] & \beta_j=0
\end{cases}
\]

A common subgradient choice at zero is \(0\). However, ordinary subgradient descent does not necessarily produce exact zeros, so it may not demonstrate Lasso's feature-selection behavior as effectively as a proximal-gradient method.

The following educational implementation uses **proximal gradient descent**, which applies a soft-thresholding operation after each gradient step.

```python
import numpy as np


class LassoRegression:
    def __init__(
        self,
        alpha=0.1,
        learning_rate=0.01,
        iterations=10000,
    ):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0.0

    @staticmethod
    def soft_threshold(values, threshold):
        return (
            np.sign(values)
            * np.maximum(np.abs(values) - threshold, 0.0)
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = float(np.mean(y))

        for _ in range(self.iterations):
            predictions = X @ self.weights + self.bias
            errors = predictions - y

            # Gradient of (1 / (2n)) * squared error
            weight_gradient = (X.T @ errors) / n_samples
            bias_gradient = np.mean(errors)

            # Gradient step for the squared-error component
            candidate_weights = (
                self.weights
                - self.learning_rate * weight_gradient
            )

            # Proximal step for the L1 penalty
            self.weights = self.soft_threshold(
                candidate_weights,
                self.learning_rate * self.alpha,
            )

            # Do not regularize the intercept
            self.bias -= self.learning_rate * bias_gradient

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.weights + self.bias


# Example usage
X = np.array([
    [1, 2, 0],
    [2, 1, 1],
    [3, 4, 0],
    [4, 3, 1],
    [5, 5, 0],
], dtype=float)

y = np.array([5, 6, 11, 12, 15], dtype=float)

model = LassoRegression(
    alpha=0.1,
    learning_rate=0.01,
    iterations=10000,
)

model.fit(X, y)

print("Weights:", model.weights)
print("Bias:", model.bias)
print("Predictions:", model.predict(X))
print("Zero coefficients:", np.sum(model.weights == 0))
```

**Note:** This implementation is for learning purposes. Feature scaling and learning-rate selection are important for convergence. For production use, prefer Scikit-Learn's tested implementation.

## 8. Feature Selection

Lasso performs embedded feature selection during model training by setting some coefficients to zero.

```python
import pandas as pd

feature_names = [f"Feature_{i + 1}" for i in range(X_train.shape[1])]

# Use the fitted Scikit-Learn pipeline from Section 6.
lasso = model.named_steps["lasso"]

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": lasso.coef_,
})

feature_importance["Selected"] = (
    feature_importance["Coefficient"] != 0
)

print(feature_importance.sort_values(
    by="Coefficient",
    key=abs,
    ascending=False,
))
```

A zero coefficient means that the feature makes no direct contribution to this fitted linear model. It does not prove that the feature is universally unimportant. Correlated features can substitute for one another, and different datasets or values of `alpha` may select different features.

## 9. Lasso vs. Ridge vs. Linear Regression

| Property | Linear Regression | Ridge Regression | Lasso Regression |
|---|---|---|---|
| Regularization | None | L2 | L1 |
| Penalty | None | Sum of squared coefficients | Sum of absolute coefficients |
| Coefficients can be exactly zero | Not typically | Not typically | Yes |
| Feature selection | No built-in selection | Usually retains all features | Can select features |
| Correlated features | May be unstable | Often distributes weights across correlated features | May select one or a subset |
| Main parameter | None | `alpha` | `alpha` |
| Potential benefit | Simple baseline | Controls coefficient magnitude | Controls coefficient magnitude and sparsity |

### Lasso vs. Ridge formulas

Ridge adds a squared-coefficient penalty:

\[
\text{Ridge loss}=\text{MSE}+\alpha\sum_j\beta_j^2
\]

Lasso adds an absolute-coefficient penalty:

\[
\text{Lasso loss}=\text{MSE}+\alpha\sum_j|\beta_j|
\]

The exact scaling of the error term can differ by implementation. These formulas illustrate the central difference between L1 and L2 regularization.

## 10. Advantages

- Can reduce overfitting.
- Can produce simpler models with fewer active features.
- Supports automatic feature selection.
- Improves interpretability when many coefficients become zero.
- Works naturally with pipelines and cross-validation.
- Is useful when many candidate features may be irrelevant.

## 11. Limitations

- May arbitrarily select one feature from a group of highly correlated features.
- Can underfit when `alpha` is too large.
- Requires careful scaling when features have different units.
- May be unstable when the training dataset changes substantially.
- Does not automatically capture nonlinear relationships.
- A zero coefficient does not establish that a feature has no predictive value in every setting.

## 12. Applications

Lasso Regression can be useful in:

- House-price prediction
- Financial and economic modeling
- Medical and biological data analysis
- High-dimensional datasets
- Marketing response prediction
- Feature selection before further modeling
- Predictive models with many candidate variables

## 13. Best Practices

1. Split the dataset into training and testing sets before fitting preprocessing steps.
2. Scale features using a pipeline to avoid data leakage.
3. Tune `alpha` with cross-validation.
4. Compare against Linear Regression and Ridge Regression.
5. Evaluate on held-out data using MAE, RMSE, and R².
6. Inspect selected features carefully, especially when predictors are correlated.
7. Consider Elastic Net when both L1 feature selection and L2 stabilization are useful.
8. Keep a final test set separate from model selection.

### Tune alpha with cross-validation

```python
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(max_iter=20000)),
])

param_grid = {
    "lasso__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
}

search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
)

search.fit(X_train, y_train)

print("Best alpha:", search.best_params_["lasso__alpha"])
print("Best CV RMSE:", (-search.best_score_) ** 0.5)
```

## 14. Practice Exercises

### Beginner

- Implement Lasso Regression using Scikit-Learn.
- Evaluate a model using MAE, MSE, RMSE, and R².
- Change `alpha` and observe the coefficient values.
- Count the number of zero coefficients.

### Intermediate

- Compare Lasso and Ridge on the same dataset.
- Use `StandardScaler` inside a pipeline.
- Tune `alpha` with `GridSearchCV`.
- Plot feature coefficients for several alpha values.

### Advanced

- Compare Lasso with Elastic Net on correlated features.
- Examine how feature selection changes across train/test splits.
- Compare model performance as the number of irrelevant features increases.
- Implement proximal-gradient optimization from scratch.
- Investigate how feature scaling affects selected coefficients.

## 15. Interview Questions

**1. What is Lasso Regression?**

A regularized linear regression technique that uses an L1 penalty to control coefficient magnitudes and can set some coefficients to zero.

**2. What does L1 regularization mean?**

It penalizes the sum of the absolute values of the model coefficients.

**3. How does Lasso perform feature selection?**

Its optimization can set coefficients exactly to zero, removing those features' direct contribution to the fitted model.

**4. What happens when alpha increases?**

Regularization becomes stronger, typically shrinking coefficients further and potentially increasing the number of zero coefficients.

**5. Why should features often be scaled?**

Regularization acts on coefficient magnitudes. Features with different units can otherwise receive uneven effective penalties.

**6. What is the main difference between Lasso and Ridge?**

Lasso uses L1 regularization and can set coefficients to zero. Ridge uses L2 regularization and generally shrinks coefficients without eliminating them.

**7. Can Lasso handle multicollinearity?**

It can still fit a model with correlated features, but it may select one feature and suppress others in ways that vary with the data.

**8. What happens when alpha is too large?**

The model may become too simple, lose important signal, and underfit.

**9. Why use cross-validation?**

To estimate how different regularization strengths generalize and select a useful `alpha` without choosing it based solely on the test set.

**10. What is Elastic Net?**

A linear regression method that combines L1 and L2 penalties, often useful when feature selection and correlated predictors are both concerns.

## 16. References

- [Scikit-Learn: Lasso Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [Scikit-Learn: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html#lasso)
- [Scikit-Learn: Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [Scikit-Learn: GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

---

**Next topic:** Elastic Net Regression — combining L1 and L2 regularization.
