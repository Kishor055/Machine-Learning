# 04. Ridge Regression

Ridge Regression is a **regularized linear regression algorithm** that reduces overfitting by adding a penalty to the magnitude of model coefficients. It is especially useful when a dataset contains many correlated features or when ordinary linear regression produces large coefficients.

## 📌 Table of Contents

- [Introduction](#-introduction)
- [Why Ridge Regression?](#-why-ridge-regression)
- [Mathematical Formula](#-mathematical-formula)
- [How Ridge Regression Works](#-how-ridge-regression-works)
- [Ridge vs Linear Regression](#-ridge-regression-vs-linear-regression)
- [Important Hyperparameters](#-important-hyperparameters)
- [Implementation Using Scikit-learn](#-implementation-using-scikit-learn)
- [Implementation from Scratch](#-implementation-from-scratch)
- [Feature Scaling](#-feature-scaling)
- [Advantages and Disadvantages](#-advantages-and-disadvantages)
- [Applications](#-applications)
- [Common Mistakes](#-common-mistakes)
- [Practice Exercises](#-practice-exercises)
- [Interview Questions](#-interview-questions)
- [References](#-references)

## 🎯 Introduction

Ordinary Linear Regression estimates the relationship between input features and a continuous target variable by minimizing the sum of squared prediction errors.

However, when features are highly correlated, the model may produce unstable or excessively large coefficients. This can make predictions sensitive to small changes in the training data.

Ridge Regression addresses this problem by adding an **L2 regularization penalty** to the regression objective.

It shrinks coefficients toward zero but generally does not make them exactly zero.

### Example use cases

- Predicting house prices from correlated property features
- Predicting sales using multiple marketing variables
- Estimating financial or economic trends
- Modeling datasets with many numerical features
- Reducing coefficient instability caused by multicollinearity

## 🤔 Why Ridge Regression?

Consider a dataset used to predict house prices with these features:

- House area
- Number of rooms
- Number of bathrooms
- Property age
- Distance from the city center

Some features may be strongly correlated. For example, larger houses often have more rooms and bathrooms.

Ordinary Linear Regression may assign large positive and negative coefficients to correlated features. Ridge Regression penalizes large coefficients, helping produce a more stable model.

**Main objective:** improve generalization by balancing prediction error and model complexity.

## 🧮 Mathematical Formula

Ridge Regression minimizes the following objective:

\[
J(\boldsymbol{\beta}) =
\sum_{i=1}^{n}(y_i-\hat y_i)^2
+\lambda\sum_{j=1}^{p}\beta_j^2
\]

where:

- \(y_i\): actual target value
- \(\hat y_i\): predicted target value
- \(\beta_j\): coefficient of feature \(j\)
- \(n\): number of observations
- \(p\): number of features
- \(\lambda\): regularization strength

The prediction equation is:

\[
\hat y = \beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_px_p
\]

The intercept \(\beta_0\) is generally not penalized.

### Understanding the regularization parameter

- \(\lambda=0\): equivalent to ordinary least-squares regression.
- Small \(\lambda\): weak coefficient penalty.
- Large \(\lambda\): stronger coefficient shrinkage.
- Excessively large \(\lambda\): may cause underfitting.

In Scikit-learn, the corresponding parameter is called `alpha`.

Note that Scikit-learn's Ridge objective uses a particular scaling convention for the residual sum of squares. The practical interpretation is the same: increasing `alpha` strengthens regularization.

## ⚙️ How Ridge Regression Works

1. Receive the training features and target values.
2. Estimate coefficients that minimize prediction error.
3. Add a penalty for large coefficient values.
4. Find coefficients that balance fit quality and coefficient size.
5. Use the fitted model to predict continuous values for new data.

<escape>
Training Data → Feature Scaling → Ridge Regression → Regularized Coefficients → Predictions → Evaluation
</escape>

## 📊 Ridge Regression vs Linear Regression

| Feature | Linear Regression | Ridge Regression |
|---|---|---|
| Objective | Minimize squared error | Minimize squared error plus L2 penalty |
| Regularization | No | Yes |
| Coefficient shrinkage | No explicit penalty | Yes |
| Handles correlated features | Can have unstable coefficients | Often improves coefficient stability |
| Coefficients exactly zero | Possible naturally | Generally not forced to zero |
| Hyperparameter | No regularization parameter | `alpha` |
| Risk of overfitting | Can be higher in some settings | Often reduced with suitable regularization |

## 🔧 Important Hyperparameters

### `alpha`

Controls the strength of regularization.

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
```

- Smaller `alpha`: coefficients are less constrained.
- Larger `alpha`: coefficients are shrunk more strongly.

The best value depends on the dataset and should be selected using validation or cross-validation.

### `fit_intercept`

Determines whether the model estimates an intercept.

```python
model = Ridge(alpha=1.0, fit_intercept=True)
```

### `max_iter`

Sets the maximum number of iterations for solver methods that use iterative optimization.

### `solver`

Specifies the numerical method used to fit the model. The default is often sufficient; other options can be useful for particular datasets.

## 💻 Implementation Using Scikit-learn

This example uses a synthetic regression dataset and a pipeline to prevent data leakage during feature scaling.

### Code

```python
import numpy as np

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# 1. Create a sample regression dataset
X, y = make_regression(
    n_samples=500,
    n_features=8,
    n_informative=6,
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

# 3. Create a preprocessing and modeling pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge()),
])

# 4. Tune alpha using cross-validation
param_grid = {
    "ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
}

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
)

search.fit(X_train, y_train)

# 5. Select the best model
model = search.best_estimator_

print("Best alpha:", search.best_params_["ridge__alpha"])

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# 8. Display coefficients
ridge_model = model.named_steps["ridge"]

print("\nIntercept:", ridge_model.intercept_)
print("Coefficients:", ridge_model.coef_)
```

### Expected output

The program prints the best `alpha` selected through cross-validation, evaluation metrics, the fitted intercept, and feature coefficients. Exact values depend on the dataset and software version.

## 🧠 Implementation from Scratch

The following implementation uses gradient descent to minimize the Ridge Regression objective. It excludes the intercept from the regularization penalty.

```python
import numpy as np


class RidgeRegression:
    def __init__(
        self,
        learning_rate=0.01,
        epochs=5000,
        alpha=1.0,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            y_pred = X @ self.weights + self.bias
            errors = y_pred - y

            # Mean squared error gradient plus L2 penalty.
            dw = (
                (2 / n_samples) * (X.T @ errors)
                + 2 * self.alpha * self.weights
            )
            db = (2 / n_samples) * np.sum(errors)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.weights + self.bias


# Example
X = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [5, 6],
], dtype=float)

y = np.array([5, 6, 11, 12, 17], dtype=float)

model = RidgeRegression(
    learning_rate=0.001,
    epochs=10000,
    alpha=0.1,
)

model.fit(X, y)

predictions = model.predict(X)

print("Weights:", model.weights)
print("Bias:", model.bias)
print("Predictions:", predictions)
```

**Important:** The from-scratch implementation uses the objective

\[
J(\boldsymbol{\beta}) =
\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
+\alpha\|\boldsymbol{\beta}\|_2^2
\]

This convention differs in scaling from Scikit-learn's implementation. Consequently, the same numerical regularization value does not necessarily produce the same coefficients in both implementations. Feature scaling is also important for stable gradient descent.

## 📏 Feature Scaling

Ridge Regression is sensitive to feature scales because the penalty acts on coefficient magnitudes.

For example, one feature might measure area in thousands of square feet, while another measures age in years. Their coefficient scales can differ substantially.

Standardization is a common solution:

\[
z = \frac{x-\mu}{\sigma}
\]

where \(\mu\) is the training-set mean and \(\sigma\) is the training-set standard deviation.

Use `StandardScaler` inside a pipeline so that scaling parameters are learned from training data only.

## ⚖️ Advantages and Disadvantages

### Advantages

- Helps reduce overfitting.
- Shrinks excessively large coefficients.
- Often improves stability when features are correlated.
- Works well with many numerical features.
- Integrates easily with Scikit-learn pipelines and cross-validation.

### Disadvantages

- Does not generally eliminate irrelevant features completely.
- Requires selecting a regularization strength.
- May underfit when regularization is too strong.
- Requires attention to feature scaling.
- Does not automatically solve nonlinear relationships unless suitable nonlinear features are added.

## 🆚 Ridge vs Lasso vs Elastic Net

| Property | Ridge | Lasso | Elastic Net |
|---|---|---|---|
| Penalty | L2 | L1 | L1 + L2 |
| Shrinks coefficients | Yes | Yes | Yes |
| Can set coefficients to zero | Not generally | Yes | Yes |
| Useful with correlated features | Often | Can select one among correlated features | Often useful |
| Scikit-learn class | `Ridge` | `Lasso` | `ElasticNet` |

## 🌍 Applications

Ridge Regression can be used for:

- House price prediction
- Sales and demand forecasting
- Financial modeling
- Economic trend estimation
- Predicting energy consumption
- Modeling datasets with multicollinearity
- Regression tasks with many numerical predictors

## ⚠️ Common Mistakes

1. **Skipping feature scaling:** coefficients can be penalized unevenly across features.
2. **Using the test set for tuning:** this leads to an overly optimistic evaluation.
3. **Choosing `alpha` arbitrarily:** use cross-validation to compare candidate values.
4. **Assuming Ridge performs feature selection:** Ridge shrinks coefficients but generally does not set them to zero.
5. **Ignoring residuals:** evaluation metrics alone may not reveal systematic prediction errors.
6. **Applying it without checking assumptions:** Ridge regularization does not automatically correct every model misspecification.

## 📝 Practice Exercises

### Beginner

- Fit a Ridge Regression model on a synthetic dataset.
- Compare its predictions with ordinary Linear Regression.
- Print the intercept and coefficients.

### Intermediate

- Try different `alpha` values such as `0.01`, `0.1`, `1`, `10`, and `100`.
- Plot coefficient magnitudes against `alpha`.
- Compare MAE, RMSE, and \(R^2\).

### Advanced

- Use `GridSearchCV` to tune `alpha`.
- Compare Ridge, Lasso, and Elastic Net on the same train/test split.
- Investigate how highly correlated features affect fitted coefficients.
- Evaluate the model using cross-validation and residual analysis.

## 🎤 Interview Questions

**1. What is Ridge Regression?**

Ridge Regression is linear regression with an L2 penalty on the model coefficients to discourage large coefficient values.

**2. Why is Ridge Regression used?**

It can reduce overfitting and stabilize coefficient estimates, especially when predictors are strongly correlated.

**3. What does `alpha` control in Scikit-learn?**

It controls the strength of the L2 regularization penalty.

**4. Does Ridge Regression perform feature selection?**

Not generally. It shrinks coefficients toward zero but typically does not force them to become exactly zero.

**5. Why is feature scaling important?**

Because the regularization penalty depends on coefficient magnitudes, differently scaled features can be affected unevenly.

**6. What happens when `alpha` is too large?**

Coefficients may become excessively small, causing underfitting.

**7. How is Ridge Regression different from Lasso Regression?**

Ridge uses an L2 penalty, while Lasso uses an L1 penalty that can set some coefficients exactly to zero.

**8. Can Ridge Regression handle multicollinearity?**

It often improves coefficient stability in the presence of multicollinearity, although it does not remove the underlying correlation between predictors.

## 📚 References

- [Scikit-learn: Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Scikit-learn: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Scikit-learn: StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [Scikit-learn: GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

---

## 🚀 Key Takeaway

Ridge Regression extends ordinary Linear Regression by penalizing large coefficients. With appropriate feature scaling and cross-validation, it can produce more stable models and improve generalization when ordinary regression is prone to overfitting.

