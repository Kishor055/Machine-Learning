# simple-linear-regression.py

```python
"""
Simple Linear Regression using Scikit-learn

Predict a student's salary based on years of experience.
Demonstrates:
- Loading and preparing data
- Training a Linear Regression model
- Making predictions
- Evaluating model performance
- Visualizing the regression line
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


def main():
    # 1. Create sample dataset
    # X must be a 2D array for Scikit-learn.
    X = np.array([
        [1], [2], [3], [4], [5],
        [6], [7], [8], [9], [10],
        [11], [12], [13], [14], [15],
    ])

    y = np.array([
        25000, 30000, 35000, 40000, 45000,
        50000, 55000, 60000, 65000, 70000,
        75000, 80000, 85000, 90000, 95000,
    ])

    # 2. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # 3. Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Make predictions on the test set
    y_pred = model.predict(X_test)

    # 5. Display the learned model parameters
    print("SIMPLE LINEAR REGRESSION")
    print("-" * 40)
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"Slope:     {model.coef_[0]:.2f}")
    print(
        f"Equation:  Salary = "
        f"{model.intercept_:.2f} + "
        f"{model.coef_[0]:.2f} × Experience"
    )

    # 6. Evaluate model performance on the test set
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nMODEL EVALUATION (TEST SET)")
    print("-" * 40)
    print(f"MAE:  {mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")

    # 7. Compare actual and predicted values
    print("\nACTUAL VS PREDICTED")
    print("-" * 40)

    for actual, predicted in zip(y_test, y_pred):
        print(
            f"Actual: {actual:,.0f} | "
            f"Predicted: {predicted:,.2f}"
        )

    # 8. Predict salary for a new data point
    years_experience = np.array([[6.5]])
    predicted_salary = model.predict(years_experience)[0]

    print("\nNEW PREDICTION")
    print("-" * 40)
    print(f"Experience: {years_experience[0, 0]} years")
    print(f"Predicted salary: {predicted_salary:,.2f}")

    # 9. Visualize the regression line
    X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_plot = model.predict(X_plot)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        X_train.ravel(),
        y_train,
        color="blue",
        label="Training data",
    )
    plt.scatter(
        X_test.ravel(),
        y_test,
        color="green",
        marker="x",
        s=80,
        label="Testing data",
    )
    plt.plot(
        X_plot.ravel(),
        y_plot,
        color="red",
        linewidth=2,
        label="Regression line",
    )

    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.title("Simple Linear Regression: Experience vs Salary")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

## How to run

Install the required libraries:

```bash
pip install numpy matplotlib scikit-learn
```

Run the script from your repository root:

```bash
python 08-Regression/01-Linear-Regression/simple-linear-regression.py
```
