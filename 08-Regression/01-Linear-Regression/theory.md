# Simple Linear Regression — Theory

## 1. Introduction

**Simple Linear Regression** is a supervised machine learning algorithm used to model the relationship between one independent variable and one continuous dependent variable.

It finds the best-fitting straight line through observed data points to estimate or predict an output.

### Real-world applications

- Predicting salary based on years of experience
- Estimating house prices based on area
- Forecasting sales based on advertising expenditure
- Predicting product demand based on price
- Estimating fuel consumption based on distance

## 2. What Is Regression?

Regression is a type of supervised learning in which a model learns the relationship between input features and a continuous target variable.

For example:

| Years of Experience | Salary |
|---:|---:|
| 1 | 25,000 |
| 2 | 30,000 |
| 3 | 35,000 |
| 4 | 40,000 |
| 5 | 45,000 |

Here:

- **Independent variable (\(X\)):** Years of experience
- **Dependent variable (\(y\)):** Salary

The model learns how salary changes as experience increases.

## 3. Mathematical Equation

The equation of Simple Linear Regression is:

\[
\hat{y} = b_0 + b_1x
\]

Where:

- \(\hat{y}\) = predicted output
- \(x\) = input feature
- \(b_0\) = intercept
- \(b_1\) = slope or regression coefficient

### Intercept

The intercept represents the predicted value of \(y\) when \(x=0\).

\[
b_0 = \bar{y} - b_1\bar{x}
\]

### Slope

The slope represents the estimated change in the predicted target for a one-unit increase in the input feature.

\[
b_1 =
\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sum_{i=1}^{n}(x_i-\bar{x})^2}
\]

Where:

- \(n\) = number of observations
- \(\bar{x}\) = mean of the input values
- \(\bar{y}\) = mean of the target values

The slope formula assumes that the input feature has nonzero variance.

## 4. How Simple Linear Regression Works

<AsyncImageGroup query={["simple linear regression scatter plot fitted straight regression line residuals educational diagram","linear regression slope intercept coordinate axes labeled educational diagram"]} aspectRatio="5:4" layout="carousel" />

The algorithm follows these steps:

1. Collect observations containing an input feature and a continuous target.
2. Calculate a line that best represents their relationship.
3. Measure prediction errors between observed and predicted values.
4. Find the coefficients that minimize the sum of squared residuals.
5. Use the fitted equation to predict outputs for new inputs.

### Step 1: Make predictions

For each observation:

\[
\hat{y}_i=b_0+b_1x_i
\]

### Step 2: Calculate residuals

A residual is the difference between an actual value and its predicted value.

\[
e_i=y_i-\hat{y}_i
\]

### Step 3: Minimize squared errors

Ordinary Least Squares (OLS) chooses the coefficients that minimize the residual sum of squares:

\[
RSS=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

Squaring errors prevents positive and negative residuals from canceling and gives greater weight to large errors.

## 5. Worked Example

Suppose the dataset is:

| Experience (\(x\)) | Salary (\(y\)) |
|---:|---:|
| 1 | 25,000 |
| 2 | 30,000 |
| 3 | 35,000 |
| 4 | 40,000 |
| 5 | 45,000 |

The salary increases by 5,000 for every additional year of experience.

Therefore:

\[
b_1=5000
\]

Using the point \(x=1,\ y=25000\):

\[
25000=b_0+5000(1)
\]

\[
b_0=20000
\]

The resulting regression equation is:

\[
\hat{y}=20000+5000x
\]

For 6 years of experience:

\[
\hat{y}=20000+5000(6)
\]

\[
\boxed{\hat{y}=50000}
\]

The predicted salary is 50,000 in the dataset's salary units.

This example is perfectly linear for illustration. Real-world datasets generally contain variation and prediction errors.

## 6. Cost Function

A cost function measures how well a model's predictions match observed values.

A commonly used objective for Linear Regression is the Mean Squared Error:

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

The model aims to minimize this quantity.

For a dataset with an intercept, minimizing MSE gives the same optimal coefficients as minimizing RSS because the two objectives differ only by the constant factor \(1/n\).

### Why squared errors?

- Large errors receive a greater penalty.
- Positive and negative errors do not cancel.
- The objective is mathematically convenient to optimize.

A limitation is that squared errors can be strongly affected by outliers.

## 7. Assumptions of Linear Regression

The standard statistical interpretation of ordinary least-squares regression relies on several assumptions.

### 1. Linearity

The conditional mean of the target is modeled as a linear function of the predictor.

### 2. Independence

Observations, or their errors under the assumed sampling structure, should be independent. This can be violated in time-series and grouped data.

### 3. Zero conditional mean

The error term should have a conditional expectation of zero given the predictor. This is important for unbiased coefficient estimates.

### 4. Constant error variance

Homoscedasticity means that the error variance is constant across predictor values.

### 5. Normality of errors

Normally distributed errors are useful for exact small-sample statistical tests and confidence intervals. Normality is not required merely to calculate OLS coefficients.

### 6. Reliable measurement and suitable data

The variables should be measured meaningfully, and influential data-quality issues should be investigated.

**Important:** Linear Regression can still calculate a fitted line when assumptions are violated, but estimates, statistical inference, or predictions may be less reliable.

## 8. Evaluation Metrics

A regression model should be evaluated on data that was not used to fit it.

### Mean Absolute Error (MAE)

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

MAE measures the average absolute prediction error in the target's units.

### Mean Squared Error (MSE)

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

MSE measures average squared prediction error and has squared target units.

### Root Mean Squared Error (RMSE)

\[
RMSE=\sqrt{MSE}
\]

RMSE is expressed in the same units as the target and penalizes large errors more heavily than MAE.

### Coefficient of Determination (\(R^2\))

\[
R^2=1-
\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
{\sum_{i=1}^{n}(y_i-\bar{y})^2}
\]

\(R^2\) compares the model's squared errors with the errors from predicting the target mean.

- \(R^2=1\): perfect predictions on the evaluated observations.
- \(R^2=0\): the model performs like the target-mean baseline under this metric.
- \(R^2<0\): the model performs worse than that baseline on the evaluated data.

A high \(R^2\) alone does not establish that a model is reliable or that the relationship is causal.

## 9. Simple Linear Regression Using Scikit-learn

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Input feature: years of experience
X = np.array([
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10]
])

# Target: salary
y = np.array([
    25000, 30000, 35000, 40000, 45000,
    50000, 55000, 60000, 65000, 70000
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Initialize and train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test-set values
y_pred = model.predict(X_test)

# Display coefficients
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Evaluate on test data
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Predict for a new observation
new_experience = np.array([[6.5]])
print("Predicted salary:", model.predict(new_experience)[0])
```

### Important implementation detail

Scikit-learn expects `X` to be a two-dimensional feature matrix with shape `(n_samples, n_features)`. The target `y` is typically a one-dimensional array with shape `(n_samples,)`.

## 10. Advantages

- Simple to understand and implement
- Computationally efficient for small and medium-sized datasets
- Easy to interpret through the slope and intercept
- Useful as a baseline model
- Provides a clear mathematical foundation for more advanced regression techniques

## 11. Limitations

- Models a linear relationship in the predictor and target.
- Can be sensitive to influential outliers.
- May perform poorly when important predictors are omitted.
- Extrapolation beyond the observed data range can be unreliable.
- A relationship between variables does not prove causation.
- Real-world data may violate the assumptions used for statistical inference.

## 12. Simple vs. Multiple Linear Regression

| Feature | Simple Linear Regression | Multiple Linear Regression |
|---|---|---|
| Number of input features | One | Two or more |
| Equation | \(\hat{y}=b_0+b_1x\) | \(\hat{y}=b_0+b_1x_1+\cdots+b_px_p\) |
| Interpretation | Effect of one predictor in the model | Each coefficient describes a predictor's effect holding the others fixed |
| Visualization | A line in two dimensions | A plane for two predictors, or a higher-dimensional model |
| Example | Salary from experience | Salary from experience, education, and location |

## 13. Common Mistakes

- Using a one-dimensional feature array instead of a two-dimensional matrix.
- Fitting preprocessing steps before splitting the data.
- Evaluating the model only on training data.
- Interpreting correlation as causation.
- Assuming a high \(R^2\) guarantees accurate future predictions.
- Ignoring residual plots and influential observations.
- Extrapolating far beyond the observed feature range without justification.

## 14. Interview Questions

1. What is Simple Linear Regression?
2. How do you calculate the slope and intercept?
3. What is the Ordinary Least Squares method?
4. What is the difference between a residual and a prediction error?
5. What assumptions are used in Linear Regression?
6. What is the difference between MAE, MSE, and RMSE?
7. Can \(R^2\) be negative?
8. How do outliers affect the fitted line?
9. Is feature scaling necessary for ordinary Simple Linear Regression?
10. What is the difference between Simple and Multiple Linear Regression?
11. Why should a model be evaluated on unseen data?
12. Does Linear Regression prove a causal relationship?

## 15. Summary

Simple Linear Regression models the relationship between one numerical predictor and a continuous target using a straight-line equation.

Key concepts to remember:

- The slope describes the estimated change in the prediction for a one-unit increase in the input.
- The intercept is the model's predicted output at an input value of zero.
- Ordinary Least Squares minimizes the sum of squared residuals.
- Test-set metrics help assess predictive performance on unseen data.
- Residual analysis and knowledge of the data are essential for judging model suitability.

**Next step:** Implement the mathematical equations from scratch in `from-scratch.py`, then compare the results with `sklearn.py` and `simple-linear-regression.py`.
