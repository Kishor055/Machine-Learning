# Polynomial Regression

## 📌 Overview

**Polynomial Regression** is a supervised machine learning technique used to model nonlinear relationships between an input feature and a continuous target variable.

It extends Linear Regression by introducing polynomial features such as \(x^2\), \(x^3\), and higher powers of the input variable. The model remains linear in its coefficients, even though its fitted curve can be nonlinear in the input.

For example, Polynomial Regression can help model relationships such as:

- Growth patterns over time
- Temperature versus energy consumption
- Vehicle speed versus fuel efficiency
- Advertising expenditure versus sales
- Experience versus productivity when growth is nonlinear

## 🎯 Learning Objectives

By completing this section, you will learn:

- Understand the difference between Linear and Polynomial Regression.
- Understand polynomial features and their mathematical representation.
- Transform input features using Scikit-learn.
- Train Polynomial Regression models.
- Select an appropriate polynomial degree.
- Visualize nonlinear fitted curves.
- Evaluate regression performance.
- Understand underfitting, overfitting, and data leakage.

## 🧠 Mathematical Foundation

### 1. Simple Linear Regression

Simple Linear Regression models the target using a straight-line relationship:

\[
\hat{y}=b_0+b_1x
\]

### 2. Polynomial Regression

Polynomial Regression adds powers of the input feature:

\[
\hat{y}=b_0+b_1x+b_2x^2+\cdots+b_dx^d
\]

Where:

- \(\hat{y}\) = predicted target value
- \(x\) = input feature
- \(b_0\) = intercept
- \(b_1,\ldots,b_d\) = learned coefficients
- \(d\) = polynomial degree

The degree controls the complexity of the fitted curve.

| Degree | Model | General behavior |
|---:|---|---|
| 1 | \(b_0+b_1x\) | Straight line |
| 2 | \(b_0+b_1x+b_2x^2\) | Can represent a curved, parabolic relationship |
| 3 | \(b_0+b_1x+b_2x^2+b_3x^3\) | Can represent more complex curves |
| 4 or higher | Includes additional powers | Greater flexibility and overfitting risk |

## 📈 Linear vs. Polynomial Regression

<grid columns={2} gap={3}>
  <grid-item>
    <AsyncImage query="simple linear regression scatter plot straight line fit educational graph" aspectRatio="4:3" width="100%" />
    **Linear Regression**

    <text size="sm">Fits a straight-line relationship between the input and target.</text>
  </grid-item>
  <grid-item>
    <AsyncImage query="polynomial regression scatter plot smooth curved quadratic regression fit educational graph" aspectRatio="4:3" width="100%" />
    **Polynomial Regression**

    <text size="sm">Fits a polynomial curve to capture suitable nonlinear patterns.</text>
  </grid-item>
</grid>

| Feature | Linear Regression | Polynomial Regression |
|---|---|---|
| Relationship modeled | Linear in the input | Can be nonlinear in the input |
| Feature transformation | Not necessarily needed | Adds powers of features |
| Model coefficients | Linear coefficients | Still linear coefficients |
| Complexity | Usually lower | Increases with polynomial degree |
| Overfitting risk | Can underfit nonlinear patterns | Can overfit with excessive degree |

## ⚙️ How Polynomial Regression Works

1. Collect input features and target values.
2. Split the data into training and testing sets.
3. Transform input features into polynomial features.
4. Fit a Linear Regression model to the transformed features.
5. Predict target values for unseen observations.
6. Evaluate errors and inspect the fitted curve.
7. Compare polynomial degrees using validation data.

### Example: Degree 2

Starting with one input feature \(x\), a degree-2 transformation produces:

\[
x\rightarrow [x,x^2]
\]

The model then learns:

\[
\hat{y}=b_0+b_1x+b_2x^2
\]

Polynomial Regression is commonly implemented by combining `PolynomialFeatures` with `LinearRegression`.

## 💻 Implementation Using Scikit-learn

The following example creates a synthetic nonlinear dataset, trains a degree-2 polynomial model, evaluates its performance, and plots the fitted curve.

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# 1. Create a synthetic nonlinear dataset
rng = np.random.default_rng(42)

X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = 3 + 2 * X[:, 0] + 1.5 * X[:, 0] ** 2
y = y + rng.normal(0, 3, size=len(X))

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# 3. Build a degree-2 Polynomial Regression pipeline
model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression(),
)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Polynomial Regression — Degree 2")
print("-" * 40)
print(f"MAE:  {mae:.3f}")
print(f"MSE:  {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.4f}")

# 7. Plot the data and fitted curve
X_curve = np.linspace(
    X.min(), X.max(), 300
).reshape(-1, 1)

y_curve = model.predict(X_curve)

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, label="Training data")
plt.scatter(
    X_test,
    y_test,
    marker="x",
    s=60,
    label="Testing data",
)
plt.plot(
    X_curve,
    y_curve,
    linewidth=2,
    label="Polynomial Regression (degree 2)",
)

plt.xlabel("Input Feature (X)")
plt.ylabel("Target (y)")
plt.title("Polynomial Regression")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Installation

```bash
pip install numpy matplotlib scikit-learn
```

### Run the script

Save the code as `polynomial-regression.py` in this folder, then run:

```bash
python 08-Regression/03-Polynomial-Regression/polynomial-regression.py
```

## 📊 Model Evaluation

Use regression metrics to assess predictions on data that was not used for fitting.

### Mean Absolute Error (MAE)

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

Measures average absolute prediction error.

### Mean Squared Error (MSE)

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

Penalizes larger errors more strongly.

### Root Mean Squared Error (RMSE)

\[
RMSE=\sqrt{MSE}
\]

Expresses error in the same units as the target.

### Coefficient of Determination (\(R^2\))

\[
R^2=1-\frac{\sum_i(y_i-\hat{y}_i)^2}
{\sum_i(y_i-\bar{y})^2}
\]

Measures performance relative to predicting the target mean. A negative test-set \(R^2\) is possible.

## 🎚️ Choosing the Polynomial Degree

A larger polynomial degree is not always better.

- **Degree too low:** The model may underfit the data.
- **Appropriate degree:** The model captures meaningful patterns without unnecessary complexity.
- **Degree too high:** The model may fit noise and generalize poorly.

Use a validation set or cross-validation to compare degrees. Keep the final test set separate until model selection is complete.

Example:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

pipeline = Pipeline([
    (
        "poly",
        PolynomialFeatures(include_bias=False),
    ),
    ("regression", LinearRegression()),
])

search = GridSearchCV(
    estimator=pipeline,
    param_grid={
        "poly__degree": [1, 2, 3, 4, 5],
    },
    cv=5,
    scoring="neg_mean_squared_error",
)

search.fit(X_train, y_train)

print("Best degree:", search.best_params_["poly__degree"])
print("Best cross-validation MSE:", -search.best_score_)
```

This example uses cross-validation on the training data. Evaluate the selected model on `X_test` and `y_test` afterward.

## ⚠️ Important Concepts

### 1. Feature Scaling

Polynomial features can have very different magnitudes. Scaling may improve numerical conditioning, especially for high-degree polynomials or regularized models.

### 2. Overfitting

High-degree polynomials can produce oscillating curves that fit training observations but perform poorly on unseen data.

### 3. Data Leakage

Fit polynomial transformations only on training data. A Scikit-learn `Pipeline` keeps the transformation and model fitting together, including during cross-validation.

### 4. Multiple Input Features

Polynomial transformations can create interaction terms. For example, with two features \(x_1\) and \(x_2\), a degree-2 expansion can include:

\[
x_1,\ x_2,\ x_1^2,\ x_1x_2,\ x_2^2
\]

The number of generated features can grow rapidly as the feature count or degree increases.

## 🚀 Practical Projects

1. **Growth Curve Prediction** — model a nonlinear growth pattern over time.
2. **Fuel Efficiency Analysis** — explore nonlinear relationships between speed and fuel consumption.
3. **Sales Forecasting** — model curved relationships between advertising expenditure and sales.
4. **Temperature Analysis** — model energy usage as a function of temperature.
5. **Degree Comparison Project** — compare polynomial degrees using validation scores and residual plots.

For each project, document the dataset, feature transformations, selected degree, evaluation results, visualizations, and limitations.

## 🎤 Interview Questions

1. What is Polynomial Regression?
2. How does it differ from Simple Linear Regression?
3. Why is Polynomial Regression still considered linear in its coefficients?
4. What does `PolynomialFeatures` do?
5. How does polynomial degree affect model complexity?
6. What is the difference between underfitting and overfitting?
7. How can cross-validation help choose the degree?
8. Why should preprocessing be included in a pipeline?
9. What are polynomial interaction terms?
10. Why can high-degree polynomial models be numerically unstable?
11. Does Polynomial Regression guarantee better predictions than Linear Regression?
12. How would you evaluate a polynomial model on unseen data?

## 📚 Recommended Learning Path

1. [Simple Linear Regression](../01-Linear-Regression/README.md)
2. [Multiple Linear Regression](../02-Multiple-Linear-Regression/README.md)
3. Polynomial feature generation and model fitting
4. Model evaluation and cross-validation
5. Regularization techniques such as Ridge Regression
6. More advanced nonlinear modeling methods

## ✅ Summary

Polynomial Regression extends Linear Regression by introducing powers and, optionally, interactions of the input features. It can capture nonlinear relationships while remaining linear in its learned coefficients.

The key is to select the degree carefully, use proper validation, inspect residuals, and evaluate the final model on unseen data.
