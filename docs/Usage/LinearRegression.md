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
import numpy as np  # Required for examples
```

## Initialization
```python
model = LinearRegression()
```

## Methods

### `fit(X, y)`
Fits the model to training data.

**Parameters:**
- `X`: Feature data (array-like, shape `(n_samples, n_features)` or `(n_samples,)`)
- `y`: Target vector (1D array-like, shape `(n_samples,)`)

**Returns:**
Fitted model instance

### `predict(X)`
Makes predictions using fitted model.

**Parameters:**
- `X`: Input features (array-like, shape `(n_samples, n_features)` or `(n_samples,)`)

**Returns:**
Predicted values (1D NumPy array)

### `score(X, y)`
Calculates R² (coefficient of determination).

**Parameters:**
- `X`: Test features
- `y`: True target values

**Returns:**
R² score (float)

## Attributes
- `coef_`: Feature weights (1D NumPy array or `None` if not fitted)
- `intercept_`: Bias term (float or `None` if not fitted)

## Complete Example

### Single Feature Example
```python
# Create sample data where y = 2x + 1
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([3, 5, 7, 9, 11])

# Initialize and fit model
model = LinearRegression()
model.fit(X, y)

# Show parameters
print("--- Single Feature Example ---")
print(f"Intercept: {model.intercept_:.2f}")        # Expected: 1.00
print(f"Coefficient: {model.coef_[0]:.2f}")       # Expected: 2.00

# Make predictions
X_test = np.array([6, 7]).reshape(-1, 1)
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")  # Expected: [13. 15.]

# Calculate score
r2 = model.score(X, y)
print(f"R² score: {r2:.4f}")  # Expected: 1.0000
```

### Multiple Features Example
```python
# Define a model: y = 3 + 1*x₁ + 2*x₂
X_multi = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2]
])
y_multi = np.array([6, 8, 7, 9])  # 3 + 1*x1 + 2*x2

model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

print("\n--- Multiple Features Example ---")
print(f"Intercept: {model_multi.intercept_:.2f}")  # Expected: 3.00
print(f"Coefficients: {np.round(model_multi.coef_, 2)}")  # Expected: [1.00 2.00]

# Predict on new data
X_test_multi = np.array([
    [1, 3],  # 3 + 1*1 + 2*3 = 10
    [4, 2]   # 3 + 1*4 + 2*2 = 11
])
predictions_multi = model_multi.predict(X_test_multi)
print(f"Predictions: {predictions_multi}")  # Expected: [10. 11.]
```

## Key Notes

1. **Input Handling**:
   - Automatically converts 1D inputs to 2D column vectors
   - Accepts both NumPy arrays and Python lists

2. **Performance Characteristics**:
   - Uses normal equation (matrix inversion)
   - Best for small-to-medium datasets (<10,000 samples)
   - May be slow for very large feature sets

3. **Numerical Stability**:
   - Includes partial pivoting in matrix inversion
   - Will raise errors for singular matrices

4. **Scikit-learn Compatibility**:
   - Uses `coef_` and `intercept_` naming convention
   - Similar method signatures

## Troubleshooting

**Common Errors:**
- `ValueError: Matrix dimensions mismatch`: Check that X and y have same number of samples
- `ValueError: Matrix is singular`: Features may be perfectly correlated
- `AttributeError`: Use `coef_` and `intercept_` (with underscore) not `coefficients`/`intercept`

For large datasets, consider preprocessing with feature scaling or using gradient descent-based implementations.