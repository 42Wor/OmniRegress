# Linear Regression Usage Guide

## Overview
The `LinearRegression` class implements ordinary least squares linear regression using the normal equation. It supports:
- Single or multiple features
- Coefficient estimation
- Prediction
- R² scoring

## Importing
```python
from omniregress import LinearRegression
import numpy as np  # Import numpy for examples
````

## Initialization

```python
model = LinearRegression()
```

## Methods

### `fit(X, y)`

Fits the model to training data.

**Parameters:**

* `X`: Feature data (array-like, shape `(n_samples, n_features)` or `(n_samples,)`). 1D input will be treated as `(n_samples, 1)`.
* `y`: Target vector (1D array-like, shape `(n_samples,)`)

**Returns:**
Fitted model instance

### `predict(X)`

Makes predictions using fitted model.

**Parameters:**

* `X`: Input features (array-like, shape `(n_samples, n_features)` or `(n_samples,)`). 1D input will be treated as `(n_samples, 1)`.

**Returns:**
Predicted values (1D NumPy array)

### `score(X, y)`

Calculates R² (coefficient of determination).

**Parameters:**

* `X`: Test features (array-like, processed similarly to `fit`)
* `y`: True target values (array-like, processed similarly to `fit`)

**Returns:**
R² score (float)

## Attributes

* `coefficients`: Feature weights (1D NumPy array or `None` if not fitted)
* `intercept`: Bias term (float or `None` if not fitted)

## Complete Example

```python
import numpy as np
from omniregress import LinearRegression  # Assuming omniregress is in the Python path

# Create sample data where y = 2x + 1
X_single = np.array([1, 2, 3, 4, 5])
y_single = np.array([3, 5, 7, 9, 11])

# Initialize and fit model
model_single = LinearRegression()
model_single.fit(X_single, y_single)

# Show parameters
print("--- Single Feature Example ---")
print(f"Intercept: {model_single.intercept:.2f}")        # Expected: 1.00
if model_single.coefficients is not None:
    print(f"Coefficient: {model_single.coefficients[0]:.2f}")  # Expected: 2.00
else:
    print("Coefficients: None (model not fitted or no features)")

# Make predictions
X_test_single = np.array([6, 7])
predictions_single = model_single.predict(X_test_single)
print(f"Predictions for {X_test_single.ravel()}: {predictions_single}")  # Expected: [13. 15.]

# Calculate score
r2_single = model_single.score(X_single, y_single)
print(f"R² score: {r2_single:.4f}")  # Expected: 1.0000
```

## Handling Multiple Features

```python
# Multiple regression example
# Define a model: y = 3 + 1*x₁ + 2*x₂
X_multi = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [3, 2],
    [3, 3],
    [4, 1]
])
y_multi = np.array([
    6,  # 3 + 1*1 + 2*1
    8,  # 3 + 1*1 + 2*2
    7,  # 3 + 1*2 + 2*1
    9,  # 3 + 1*2 + 2*2
    10, # 3 + 1*3 + 2*2
    12, # 3 + 1*3 + 2*3
    9   # 3 + 1*4 + 2*1
])

model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

print("\n--- Multiple Features Example ---")
print(f"Intercept: {model_multi.intercept:.2f}")  # Expected: 3.00
if model_multi.coefficients is not None:
    print(f"Coefficients: {np.round(model_multi.coefficients, 2)}")  # Expected: [1.00 2.00]
else:
    print("Coefficients: None")

# Predict on new multi-feature data
X_test_multi = np.array([
    [1, 3],  # 3 + 1*1 + 2*3 = 10
    [4, 2]   # 3 + 1*4 + 2*2 = 11
])
predictions_multi = model_multi.predict(X_test_multi)
print(f"Predictions for new multi-feature data: {predictions_multi}")  # Expected: [10. 11.]

# Score for multiple features
r2_multi = model_multi.score(X_multi, y_multi)
print(f"R² score (multi): {r2_multi:.4f}")  # Expected: 1.0000
```

## Notes

* Accepts various array-like inputs (e.g., NumPy arrays, Python lists). Prediction results and coefficients are returned as NumPy arrays.
* The underlying Rust implementation uses direct matrix inversion (Gauss-Jordan elimination) via the normal equation. This method may fail or produce inaccurate results for singular or numerically ill-conditioned matrices (i.e., when features are perfectly collinear or nearly so).
* For very large datasets (e.g., >10,000 samples or many features), the normal equation approach can be computationally intensive due to matrix multiplication and inversion. Consider alternative implementations like those based on gradient descent for such scenarios.
* The Python wrapper handles conversion of 1D array-like inputs for `X` into 2D, and ensures data passed to the Rust core is in the expected `list[list[float]]` or `list[float]` format.

