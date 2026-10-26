# Linear Regression — Practice Exercises

A hands-on practice guide for learning **Simple Linear Regression** using Python, NumPy, Pandas, Matplotlib, and Scikit-learn.

## 🎯 Learning Objectives

By completing these exercises, you will learn to:

- Understand the fundamentals of Linear Regression.
- Calculate slope and intercept.
- Implement Simple Linear Regression from scratch.
- Train models using Scikit-learn.
- Evaluate regression models using appropriate metrics.
- Visualize regression lines and prediction errors.
- Make predictions on unseen data.
- Identify overfitting, underfitting, and data leakage.

## 📚 Prerequisites

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Recommended knowledge:

- Python fundamentals
- NumPy arrays
- Pandas DataFrames
- Basic statistics
- Matplotlib visualization

---

## Level 1: Beginner Exercises

### Exercise 1: Understand the Dataset

Create a dataset containing years of experience and salary.

| Experience (Years) | Salary |
|---:|---:|
| 1 | 25,000 |
| 2 | 30,000 |
| 3 | 35,000 |
| 4 | 40,000 |
| 5 | 45,000 |

**Tasks:**

1. Create the dataset using NumPy.
2. Store experience in `X` and salary in `y`.
3. Print the shape of both arrays.
4. Plot experience against salary.
5. Identify the independent and dependent variables.

### Exercise 2: Understand the Regression Equation

The Simple Linear Regression equation is:

\[
\hat{y} = b_0 + b_1x
\]

Where:

- \(\hat{y}\): predicted output
- \(b_0\): intercept
- \(b_1\): slope
- \(x\): input feature

**Tasks:**

1. Calculate the change in salary between consecutive observations.
2. Determine the slope.
3. Calculate the intercept.
4. Write the regression equation.
5. Predict salary for 6 years of experience.

### Exercise 3: Create a Scatter Plot

**Tasks:**

1. Create a scatter plot using Matplotlib.
2. Label both axes.
3. Add a chart title.
4. Explain whether the relationship appears linear.
5. Identify any observations that do not follow the general trend.

---

## Level 2: Intermediate Exercises

### Exercise 4: Implement Linear Regression from Scratch

Create a file named `exercise_04.py`.

**Tasks:**

1. Calculate the mean of `X` and `y`.
2. Calculate the slope using the least-squares formula.
3. Calculate the intercept.
4. Implement a prediction function.
5. Predict the salary for 7 years of experience.
6. Plot the fitted regression line.

Do not use `sklearn.linear_model.LinearRegression` for this exercise.

### Exercise 5: Train a Scikit-learn Model

Use the following starter code:

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = np.array([[1], [2], [3], [4], [5],
              [6], [7], [8], [9], [10]])

y = np.array([25000, 30000, 35000, 40000, 45000,
              50000, 55000, 60000, 65000, 70000])

# TODO: Split the dataset into training and testing sets.

# TODO: Create a LinearRegression model.

# TODO: Train the model.

# TODO: Predict the test-set values.

# TODO: Print the slope and intercept.
```

**Tasks:**

1. Use an 80/20 train-test split.
2. Set `random_state=42`.
3. Train the model.
4. Print the slope and intercept.
5. Compare actual and predicted values.
6. Predict salary for 6.5 years of experience.

### Exercise 6: Evaluate the Model

Use Scikit-learn's metrics to calculate:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of determination (\(R^2\))

**Tasks:**

1. Calculate all four metrics on the test set.
2. Explain what each metric measures.
3. Compare MAE and RMSE.
4. Explain why a negative \(R^2\) value can occur on a test set.
5. Explain why training-set performance alone is insufficient.

### Exercise 7: Analyze Residuals

Residuals are the differences between actual and predicted values.

\[
e_i = y_i - \hat{y}_i
\]

**Tasks:**

1. Calculate residuals for the test set.
2. Plot residuals against predicted values.
3. Add a horizontal reference line at zero.
4. Look for patterns in the residuals.
5. Explain why random-looking residuals are generally desirable.

---

## Level 3: Advanced Exercises

### Exercise 8: Study the Effect of Outliers

Create a dataset with one unusually large salary value.

**Tasks:**

1. Train a Linear Regression model before adding the outlier.
2. Add the outlier and retrain the model.
3. Compare the slope and intercept.
4. Compare predictions before and after adding the outlier.
5. Explain why ordinary least squares is sensitive to extreme values.

### Exercise 9: Compare Different Train-Test Splits

**Tasks:**

1. Train models using test sizes of 20%, 25%, and 30%.
2. Use the same `random_state` for each experiment.
3. Compare MAE, RMSE, and \(R^2\).
4. Explain why scores may change with different splits.
5. Explain why a single small test set can give an unstable performance estimate.

### Exercise 10: Build a Complete Regression Pipeline

Use Scikit-learn's `Pipeline` to combine preprocessing and model training.

**Tasks:**

1. Create a Pandas DataFrame containing experience and salary.
2. Split the data into training and testing sets.
3. Build a pipeline with an appropriate preprocessing step and `LinearRegression`.
4. Fit the pipeline using training data only.
5. Evaluate it on test data.
6. Explain how pipelines help prevent data leakage.

### Exercise 11: Predict House Prices

Create or obtain a dataset containing one numerical feature, such as house area, and a target, such as house price.

**Tasks:**

1. Load the dataset with Pandas.
2. Inspect missing values and data types.
3. Plot area against price.
4. Train a Simple Linear Regression model.
5. Evaluate the model using a test set.
6. Predict prices for new house areas.
7. Discuss why one feature may be insufficient for accurate real-world house-price prediction.

### Exercise 12: Compare Scratch and Scikit-learn Implementations

**Tasks:**

1. Implement Simple Linear Regression from scratch.
2. Train a Scikit-learn `LinearRegression` model on the same training data.
3. Compare their slopes and intercepts.
4. Compare predictions on the same inputs.
5. Compare evaluation metrics.
6. Explain why small numerical differences may occur.

---

## 🧪 Mini Project: Salary Prediction

**Objective:** Build an end-to-end salary prediction model using years of experience.

### Requirements

- [ ] Create or load a dataset.
- [ ] Inspect and clean the data.
- [ ] Perform exploratory data analysis.
- [ ] Split the dataset into training and testing sets.
- [ ] Train a Simple Linear Regression model.
- [ ] Calculate MAE, MSE, RMSE, and \(R^2\).
- [ ] Visualize the regression line.
- [ ] Analyze residuals.
- [ ] Predict salaries for new experience values.
- [ ] Document findings and limitations in a README.

### Expected project structure

```text
salary-prediction/
├── data/
│   └── salary.csv
├── notebooks/
│   └── exploratory-analysis.ipynb
├── src/
│   └── train.py
├── outputs/
│   └── regression-plot.png
├── README.md
└── requirements.txt
```

### Evaluation rubric

| Criteria | Weight |
|---|---:|
| Data exploration and preparation | 20% |
| Model implementation | 25% |
| Evaluation and residual analysis | 25% |
| Visualization | 15% |
| Documentation and code quality | 15% |

---

## 🎤 Interview Practice Questions

### Conceptual Questions

1. What is Linear Regression?
2. What is the difference between Simple and Multiple Linear Regression?
3. What are slope and intercept?
4. What is the least-squares method?
5. What assumptions does ordinary least-squares Linear Regression make?
6. What is the difference between correlation and regression?
7. What is the difference between MAE, MSE, and RMSE?
8. What does \(R^2\) represent?
9. Can \(R^2\) be negative?
10. Why are outliers influential in Linear Regression?

### Practical Questions

1. Why must `X` usually be two-dimensional in Scikit-learn?
2. Why should data be split before fitting preprocessing steps?
3. Does Simple Linear Regression require feature scaling?
4. How do you visualize a regression line?
5. How do you inspect residuals?
6. How can you detect a nonlinear relationship?
7. How would you handle missing values?
8. Why should you avoid evaluating only on training data?

---

## ✅ Self-Assessment Checklist

- [ ] I understand the regression equation.
- [ ] I can calculate slope and intercept.
- [ ] I can implement Linear Regression from scratch.
- [ ] I can train a Scikit-learn model.
- [ ] I can split data correctly.
- [ ] I can calculate regression evaluation metrics.
- [ ] I can visualize a regression line.
- [ ] I can analyze residuals.
- [ ] I understand the effects of outliers.
- [ ] I can explain the assumptions and limitations of Linear Regression.

## 📌 Key Takeaway

Simple Linear Regression is a foundational supervised learning algorithm that models the relationship between one input feature and a continuous target. Practice both the mathematical implementation and the Scikit-learn workflow to build a strong foundation for Multiple Linear Regression and other regression techniques.
