# 12. Hyperparameter Tuning – Manual Tuning

## 1. Introduction

Hyperparameter tuning is the process of finding the optimal values of hyperparameters that improve the performance of a Machine Learning model.

Hyperparameters are configuration settings that are defined before the model training process begins. Unlike model parameters, they are not learned automatically from the training data.

**Manual Hyperparameter Tuning** is a technique in which we manually select different hyperparameter values, train the model, evaluate its performance, and choose the combination that produces the best results.

## 2. What is Manual Tuning?

Manual tuning involves experimenting with different hyperparameter values based on domain knowledge, experience, and model performance.

For example, in a K-Nearest Neighbors (KNN) algorithm, the `n_neighbors` parameter determines how many neighboring data points are considered when making predictions.

We can manually experiment with different values:

- K = 3
- K = 5
- K = 7
- K = 9
- K = 11

After evaluating the model with each value, we select the value that provides the best performance on validation data.

## 3. Objectives of Manual Tuning

- Improve model accuracy and predictive performance.
- Reduce overfitting and underfitting.
- Identify the best hyperparameter combination.
- Understand the behavior of machine learning algorithms.
- Improve model generalization on unseen data.
- Develop an understanding of the relationship between hyperparameters and model performance.

## 4. Parameters vs Hyperparameters

| Parameters | Hyperparameters |
|---|---|
| Learned during model training. | Set before model training. |
| Automatically optimized by algorithms. | Configured manually or through tuning methods. |
| Define the internal model behavior. | Control the learning process. |
| Example: Weights and biases. | Example: Learning rate and tree depth. |

## 5. Common Hyperparameters

Different machine learning algorithms have different hyperparameters.

| Algorithm | Hyperparameter | Description |
|---|---|---|
| Linear Regression | fit_intercept | Whether to calculate an intercept. |
| Logistic Regression | C | Controls regularization strength inversely. |
| KNN | n_neighbors | Number of nearest neighbors. |
| Decision Tree | max_depth | Maximum depth of the tree. |
| Random Forest | n_estimators | Number of decision trees. |
| Random Forest | max_features | Number of features considered for splitting. |
| SVM | C | Regularization parameter. |
| SVM | kernel | Type of kernel function. |
| Gradient Boosting | learning_rate | Contribution of each tree. |
| Neural Network | learning_rate | Controls the size of weight updates. |

## 6. Working Procedure of Manual Tuning

1. Select a machine learning algorithm.
2. Identify important hyperparameters.
3. Choose suitable values for each hyperparameter.
4. Train the model using the selected values.
5. Evaluate model performance using validation data.
6. Compare the evaluation results.
7. Repeat the process with different hyperparameter values.
8. Select the best-performing configuration.
9. Evaluate the final model on the test dataset.

## 7. Implementation Using Python

### Step 1: Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
```

### Step 2: Load Dataset

```python
# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Dataset Shape:", X.shape)
print("Target Classes:", np.unique(y))
```

### Step 3: Split Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Step 4: Define Hyperparameter Values

```python
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

results = []
```

### Step 5: Perform Manual Hyperparameter Tuning

```python
for k in k_values:

    # Create KNN model
    model = KNeighborsClassifier(n_neighbors=k)

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)

    results.append({
        "K Value": k,
        "Accuracy": accuracy
    })

    print(f"K = {k}, Accuracy = {accuracy:.4f}")
```

### Step 6: Compare Results

```python
results_df = pd.DataFrame(results)

print(results_df)

best_result = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\nBest Hyperparameter:")
print(best_result)
```

### Step 7: Visualize Results

```python
plt.figure(figsize=(8, 5))

plt.plot(
    results_df["K Value"],
    results_df["Accuracy"],
    marker="o"
)

plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.title("Manual Hyperparameter Tuning - KNN")

plt.grid(True)
plt.show()
```

### Step 8: Train Final Model

```python
best_k = int(best_result["K Value"])

final_model = KNeighborsClassifier(
    n_neighbors=best_k
)

final_model.fit(X_train, y_train)

final_predictions = final_model.predict(X_test)

final_accuracy = accuracy_score(
    y_test,
    final_predictions
)

print("Best K:", best_k)
print("Final Test Accuracy:", final_accuracy)
```

**Note:** For a rigorous evaluation, use a separate validation set or cross-validation to select hyperparameters. Keep the test set untouched until the final evaluation. The example above demonstrates the basic manual tuning process; selecting K using test accuracy would otherwise introduce test-set leakage.

## 8. Advantages of Manual Hyperparameter Tuning

1. **Simple Implementation:** Easy to understand and implement.
2. **Full Control:** Allows complete control over hyperparameter selection.
3. **No Additional Framework:** Does not require specialized tuning libraries.
4. **Educational Value:** Helps understand how hyperparameters affect model performance.
5. **Flexible:** Values can be adjusted based on observations and experience.

## 9. Disadvantages of Manual Hyperparameter Tuning

1. **Time Consuming:** Requires repeated training and evaluation.
2. **Human Dependency:** Results depend on user experience and knowledge.
3. **Limited Search:** Only a limited number of combinations may be tested.
4. **Not Scalable:** Difficult to tune models with many hyperparameters.
5. **No Guarantee:** May not identify the globally optimal combination.

## 10. Applications

- Classification model optimization.
- Regression model optimization.
- Deep learning model configuration.
- Decision tree optimization.
- Random forest optimization.
- Support Vector Machine optimization.
- Improving predictive performance in real-world applications.

## 11. Manual Tuning vs Automated Tuning

| Feature | Manual Tuning | Automated Tuning |
|---|---|---|
| Selection | Human-based | Algorithm-based |
| Execution | Manually performed | Automatically performed |
| Time | Can be time-consuming | Depends on search strategy |
| Complexity | Simple for few parameters | Suitable for larger searches |
| Control | High user control | Search strategy controls exploration |
| Examples | Manual experimentation | Grid Search, Random Search, Bayesian Optimization |

## 12. Best Practices

- Start with a baseline model.
- Tune the most important hyperparameters first.
- Use a validation dataset.
- Apply cross-validation for reliable performance estimation.
- Avoid tuning too many parameters simultaneously.
- Keep a record of every experiment.
- Monitor overfitting and underfitting.
- Evaluate the final configuration on unseen test data.

## 13. Key Takeaways

- Manual tuning is a basic hyperparameter optimization technique.
- Hyperparameters control the learning behavior of machine learning algorithms.
- Different hyperparameter values can produce different model performance.
- Manual tuning is useful for small search spaces and learning model behavior.
- Validation data should be used for hyperparameter selection.
- Automated techniques are more suitable for complex and large search spaces.

## 14. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## 15. Conclusion

Manual Hyperparameter Tuning is an important technique in Machine Learning that helps improve model performance by experimenting with different hyperparameter values. It provides a simple and understandable approach to model optimization. Although it can become time-consuming for complex models, it is an excellent starting point for understanding hyperparameter optimization before moving toward automated techniques such as Grid Search, Random Search, and Bayesian Optimization.

---

**Next Topic:** `12-Hyperparameter-Tuning/02-Grid-Search/README.md`
