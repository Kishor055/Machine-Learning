# sklearn.py

```python
"""
Simple Linear Regression using Scikit-learn

Topics covered:
1. Creating a dataset
2. Splitting data into training and testing sets
3. Training a Linear Regression model
4. Making predictions
5. Evaluating model performance
6. Predicting new values
7. Visualizing the regression line
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
    # --------------------------------------------------
    # 1. Create the dataset
    # --------------------------------------------------
    X = np.array([
        [1], [2], [3], [4], [5],
        [6], [7], [8], [9], [10],
        [11], [12], [13], [14], [15],
        [16], [17], [18], [19], [20],
    ])

    y = np.array([
        25000, 28000, 32000, 35000, 39000,
        42000, 46000, 49000, 53000, 56000,
        60000, 63000, 67000, 70000, 74000,
        77000, 81000, 84000, 88000, 91000,
    ])

    # --------------------------------------------------
    # 2. Split the dataset
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("Dataset shape:", X.shape)
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    # --------------------------------------------------
    # 3. Initialize the model
    # --------------------------------------------------
    model = LinearRegression()

    # --------------------------------------------------
    # 4. Train the model
    # --------------------------------------------------
    model.fit(X_train, y_train)

    # --------------------------------------------------
    # 5. Display model parameters
    # --------------------------------------------------
    print("\nMODEL PARAMETERS")
    print("-" * 40)
    print(f"Intercept (b0): {model.intercept_:.2f}")
    print(f"Slope (b1): {model.coef_[0]:.2f}")
    print(
        f"Regression equation: y = "
        f"{model.intercept_:.2f} + "
        f"{model.coef_[0]:.2f}x"
    )

    # --------------------------------------------------
    # 6. Predict on test data
    # --------------------------------------------------
    y_pred = model.predict(X_test)

    print("\nTEST PREDICTIONS")
    print("-" * 40)

    for experience, actual, predicted in zip(
        X_test.ravel(), y_test, y_pred
    ):
        print(
            f"Experience: {experience:2d} years | "
            f"Actual: {actual:,.0f} | "
            f"Predicted: {predicted:,.2f}"
        )

    # --------------------------------------------------
    # 7. Evaluate model performance
    # --------------------------------------------------
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nMODEL EVALUATION")
    print("-" * 40)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.4f}")

    # --------------------------------------------------
    # 8. Predict for new data
    # --------------------------------------------------
    new_experience = np.array([
        [2.5],
        [7.5],
        [12.5],
        [18.5],
    ])

    new_predictions = model.predict(new_experience)

    print("\nNEW PREDICTIONS")
    print("-" * 40)

    for experience, salary in zip(
        new_experience.ravel(), new_predictions
    ):
        print(
            f"{experience:.1f} years of experience "
            f"-> Predicted salary: {salary:,.2f}"
        )

    # --------------------------------------------------
    # 9. Visualize the model
    # --------------------------------------------------
    X_line = np.linspace(
        X.min(), X.max(), 200
    ).reshape(-1, 1)

    y_line = model.predict(X_line)

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
        X_line.ravel(),
        y_line,
        color="red",
        linewidth=2,
        label="Regression line",
    )

    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.title("Simple Linear Regression using Scikit-learn")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

## Installation

```bash
pip install numpy matplotlib scikit-learn
```

## Run the script

From the root of your repository:

```bash
python 08-Regression/01-Linear-Regression/sklearn.py
```

## Key concepts

- `LinearRegression()` initializes the model.
- `model.fit(X_train, y_train)` learns the best-fit line from training data.
- `model.predict(X_test)` generates predictions for unseen test samples.
- `model.coef_` provides the slope, while `model.intercept_` provides the intercept.
- Evaluation metrics help measure prediction error and model fit.

**Important:** The dataset is synthetic and intended for practice. Test-set metrics provide a more useful estimate of generalization than metrics calculated on the training data, but results from such a small dataset should not be treated as evidence of real-world salary prediction accuracy. 
