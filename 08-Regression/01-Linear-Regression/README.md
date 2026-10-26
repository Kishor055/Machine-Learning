# 📈 Linear Regression in Machine Learning

## 📌 Overview

**Linear Regression** is a supervised machine learning algorithm used to predict a continuous numerical value by modeling the linear relationship between independent variables (features) and a dependent variable (target).

It is one of the fundamental algorithms in machine learning and serves as a foundation for understanding predictive modeling, parameter estimation, residual analysis, and model evaluation.

Linear Regression estimates the best-fitting line by minimizing the sum of squared differences between actual and predicted values, a method known as **Ordinary Least Squares (OLS)**.

## 🎯 Learning Objectives

- Understand the fundamentals of Linear Regression.
- Learn simple and multiple Linear Regression.
- Understand the mathematical equation of a regression model.
- Learn how slope and intercept influence predictions.
- Understand the Ordinary Least Squares method.
- Implement Linear Regression using Scikit-learn.
- Evaluate regression models using appropriate metrics.
- Analyze residuals and identify model limitations.
- Understand assumptions and common challenges.
- Apply Linear Regression to real-world datasets.

## 📚 Table of Contents

1. Introduction to Linear Regression
2. Types of Linear Regression
3. Mathematical Formula
4. How Linear Regression Works
5. Simple Linear Regression
6. Multiple Linear Regression
7. Implementing Linear Regression from Scratch
8. Linear Regression Using Scikit-learn
9. Model Evaluation Metrics
10. Residual Analysis
11. Assumptions of Linear Regression
12. Advantages and Disadvantages
13. Applications
14. Common Mistakes
15. Practice Exercises
16. Requirements
17. References

---

## 1. Introduction to Linear Regression

Linear Regression establishes a relationship between input features and a continuous target variable.

For example, a model can estimate house prices based on area, predict salaries based on experience, or estimate sales based on advertising expenditure.

### Example Dataset

| Experience (Years) | Salary (₹) |
|---:|---:|
| 1 | 25,000 |
| 2 | 30,000 |
| 3 | 40,000 |
| 4 | 45,000 |
| 5 | 55,000 |

In this example:

- **Independent variable (X):** Years of experience.
- **Dependent variable (y):** Salary.
- **Objective:** Learn a relationship that can predict salary for a given level of experience.

## 2. Types of Linear Regression

### Simple Linear Regression

Uses one independent variable to predict a continuous target.

Example: Predicting salary using years of experience.

### Multiple Linear Regression

Uses two or more independent variables to predict a continuous target.

Example: Predicting house prices using area, bedrooms, location-related features, and property age.

| Feature | Simple Linear Regression | Multiple Linear Regression |
|---|---|---|
| Input features | One | Two or more |
| Model equation | \(y = b_0 + b_1x\) | \(y = b_0 + b_1x_1 + \cdots + b_px_p\) |
| Visualization | Usually a 2D line | A higher-dimensional hyperplane |
| Example | Salary vs. experience | House price vs. multiple features |

## 3. Mathematical Formula

### Simple Linear Regression

The prediction equation is:

\[
\hat{y} = b_0 + b_1x
\]

Where:

- \(\hat{y}\) = predicted target value
- \(x\) = input feature
- \(b_0\) = intercept
- \(b_1\) = slope or coefficient

The intercept represents the model's predicted target when \(x=0\). The slope represents the change in predicted target for a one-unit increase in \(x\).

<LearningViz type_id="LEAST_SQUARE_REGRESSION" />

### Multiple Linear Regression

For multiple input features:

\[
\hat{y} = b_0 + \sum_{j=1}^{p} b_jx_j
\]

Here, \(p\) represents the number of input features and \(b_j\) is the coefficient associated with feature \(x_j\).

### Ordinary Least Squares

The OLS method estimates the coefficients by minimizing the sum of squared residuals:

\[
\min_{b_0,\ldots,b_p}
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

A residual is the difference between an observed value and its predicted value.

\[
e_i = y_i - \hat{y}_i
\]

Squaring residuals prevents positive and negative errors from canceling and penalizes larger errors more heavily.

## 4. How Linear Regression Works

A typical Linear Regression workflow involves the following steps:

1. Collect and inspect the dataset.
2. Identify independent features and the target variable.
3. Handle missing values and prepare the data.
4. Split the dataset into training and testing sets.
5. Fit the regression model using training data.
6. Generate predictions on unseen data.
7. Evaluate the model using regression metrics.
8. Analyze residuals and validate the model assumptions.

```text
          Dataset
              |
              v
       Data Preparation
              |
              v
        Train-Test Split
          /         \
         v           v
    Training Set   Test Set
         |
         v
   Train Regression Model
         |
         v
     Learned Coefficients
         |
         v
    Predict on Test Data
         |
         v
     Model Evaluation
```

## 5. Simple Linear Regression

This example demonstrates how to predict salary from years of experience.

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Input feature: years of experience
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

# Target: salary
y = np.array([25000, 30000, 40000, 45000, 55000])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Generate predictions
y_pred = model.predict(X)

# Display model parameters
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

# Predict salary for 6 years of experience
new_experience = np.array([[6]])
predicted_salary = model.predict(new_experience)

print("Predicted salary:", predicted_salary[0])

# Visualize the fitted line
plt.scatter(X.ravel(), y, label="Actual salary")
plt.plot(X.ravel(), y_pred, label="Regression line")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary (₹)")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
```

**Explanation:**

- `reshape(-1, 1)` converts the input into the two-dimensional shape expected by Scikit-learn.
- `fit()` learns the regression coefficients.
- `intercept_` returns the fitted intercept.
- `coef_` returns the fitted coefficient or coefficients.
- `predict()` estimates the target for new input values.

The prediction for six years of experience is an extrapolation beyond the observed range. Its reliability depends on whether the learned relationship remains valid outside the training data.

## 6. Multiple Linear Regression

Multiple Linear Regression uses several independent variables to estimate one continuous target.

For example, house prices may depend on area, number of bedrooms, and property age.

```python
import pandas as pd

from sklearn.linear_model import LinearRegression

# Sample dataset
data = pd.DataFrame({
    "Area": [1000, 1200, 1500, 1800, 2000],
    "Bedrooms": [2, 2, 3, 3, 4],
    "Age": [10, 8, 5, 4, 2],
    "Price": [30, 35, 45, 55, 65]
})

# Separate features and target
X = data[["Area", "Bedrooms", "Age"]]
y = data["Price"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict the price of a new property
new_house = pd.DataFrame({
    "Area": [1600],
    "Bedrooms": [3],
    "Age": [3]
})

prediction = model.predict(new_house)

print("Predicted house price:", prediction[0])
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
```

The sample prices are illustrative and expressed in arbitrary units. A real application requires a representative dataset and appropriate evaluation.

## 7. Implementing Linear Regression from Scratch

The following implementation calculates the slope and intercept for a simple regression model using NumPy.

```python
import numpy as np

# Dataset
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([25000, 30000, 40000, 45000, 55000], dtype=float)

# Calculate means
x_mean = np.mean(X)
y_mean = np.mean(y)

# Calculate slope
numerator = np.sum((X - x_mean) * (y - y_mean))
denominator = np.sum((X - x_mean) ** 2)

slope = numerator / denominator

# Calculate intercept
intercept = y_mean - slope * x_mean

# Predictions
y_pred = intercept + slope * X

print("Slope:", slope)
print("Intercept:", intercept)
print("Predictions:", y_pred)
```

### Mathematical Calculation

The slope is calculated as:

\[
b_1 =
\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sum_{i=1}^{n}(x_i-\bar{x})^2}
\]

The intercept is:

\[
b_0 = \bar{y} - b_1\bar{x}
\]

These formulas give the OLS solution for simple linear regression when the input feature has nonzero variance.

## 8. Linear Regression Using Scikit-learn

The following example demonstrates a standard training and testing workflow.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load dataset
X, y = fetch_california_housing(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Generate predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
print("R² Score:", r2_score(y_test, y_pred))
```

The California Housing dataset is a real-valued regression dataset available through Scikit-learn. Depending on your environment, downloading it for the first time requires internet access.

## 9. Model Evaluation Metrics

Regression metrics measure the difference between actual and predicted values.

### Mean Absolute Error (MAE)

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

MAE measures the average absolute prediction error. It uses the same units as the target variable.

### Mean Squared Error (MSE)

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

MSE squares errors, penalizing large errors more heavily. Its units are the square of the target's units.

### Root Mean Squared Error (RMSE)

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

RMSE is expressed in the target's original units and is more sensitive to large errors than MAE.

### Coefficient of Determination (\(R^2\))

\[
R^2 = 1 -
\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
{\sum_{i=1}^{n}(y_i-\bar{y})^2}
\]

\(R^2\) compares the model's residual sum of squares against a baseline that predicts the target mean. It can be negative when the model performs worse than that baseline.

| Metric | Interpretation | Preferred Direction |
|---|---|---|
| MAE | Average absolute error | Lower |
| MSE | Average squared error | Lower |
| RMSE | Error magnitude in target units | Lower |
| R² | Performance relative to the mean baseline | Generally higher |

Metric selection should reflect the prediction task, error costs, and target scale.

## 10. Residual Analysis

Residual analysis helps determine whether a fitted regression model is appropriate.

```python
import matplotlib.pyplot as plt

residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
```

### What to Look For

- Random scatter around zero can support the adequacy of a linear mean relationship.
- Curved patterns may indicate nonlinearity.
- A funnel-shaped pattern may indicate nonconstant error variance.
- Extreme residuals may indicate unusual observations or model misspecification.

Residual plots should be interpreted alongside domain knowledge and other diagnostics.

## 11. Assumptions of Linear Regression

1. **Linearity:** The conditional mean of the target is linear in the model's predictors and coefficients.
2. **Independence:** Observations or errors are independent when required by the sampling design.
3. **Homoscedasticity:** The conditional error variance is constant for conventional homoscedasticity-based inference.
4. **No perfect multicollinearity:** Predictors must not be exact linear combinations of one another for unique coefficient identification.
5. **Error normality for classical inference:** Normally distributed errors are useful for exact small-sample classical tests and confidence intervals, but are not required merely to calculate OLS predictions.
6. **Appropriate model specification:** Relevant structure and important nonlinear effects should be represented adequately.

Violations can affect inference, coefficient stability, or predictive performance. Some issues can be addressed through feature engineering, robust methods, or alternative models.

## 12. Advantages and Disadvantages

### Advantages

- Simple and easy to interpret.
- Fast to train on many datasets.
- Provides a strong baseline for regression tasks.
- Supports statistical inference under appropriate assumptions.
- Coefficients can describe feature relationships.
- Works well when the conditional relationship is approximately linear.

### Disadvantages

- Cannot automatically capture complex nonlinear relationships.
- Sensitive to influential observations and outliers.
- Multicollinearity can make coefficients unstable.
- Performance depends on feature quality and model specification.
- Extrapolation beyond the training range can be unreliable.
- Ordinary least squares can be affected by heteroscedasticity and correlated errors.

## 13. Applications

Linear Regression is used in:

- House price prediction
- Salary estimation
- Sales forecasting
- Demand estimation
- Cost prediction
- Business trend analysis
- Energy consumption forecasting
- Economic modeling
- Scientific and statistical analysis

## 14. Common Mistakes

- Using classification accuracy to evaluate a regression model.
- Failing to separate training and testing data.
- Evaluating the model only on training data.
- Ignoring missing values and invalid observations.
- Assuming correlation proves causation.
- Interpreting coefficients without considering feature units and other predictors.
- Ignoring nonlinear patterns in residual plots.
- Extrapolating far beyond the observed data range.
- Comparing error metrics across datasets with different target scales without context.

## 15. Practice Exercises

### Beginner

- Implement Simple Linear Regression using Scikit-learn.
- Calculate slope and intercept using NumPy.
- Predict salary based on years of experience.
- Plot observed values and the fitted regression line.

### Intermediate

- Implement Multiple Linear Regression with at least three features.
- Compare MAE, MSE, RMSE, and \(R^2\).
- Analyze residual plots.
- Compare training and test performance.

### Advanced

- Investigate multicollinearity in a multiple regression model.
- Compare Linear Regression with Ridge and Lasso regression.
- Evaluate a regression model using cross-validation.
- Study how outliers affect OLS coefficients.
- Engineer polynomial features and compare performance with a linear baseline.

## 16. Requirements

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## 17. References

- [Scikit-learn: LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Scikit-learn: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Scikit-learn: Regression Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [Scikit-learn: California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

---

## ✅ Key Takeaways

- Linear Regression predicts continuous numerical values.
- Simple Linear Regression uses one feature, while Multiple Linear Regression uses multiple features.
- Ordinary Least Squares estimates coefficients by minimizing squared residuals.
- MAE, MSE, RMSE, and \(R^2\) measure different aspects of regression performance.
- Residual analysis helps identify model limitations.
- Proper data splitting and evaluation are essential for assessing generalization.

**Next topic:** Multiple Linear Regression — explore how several independent variables jointly predict a continuous target.
