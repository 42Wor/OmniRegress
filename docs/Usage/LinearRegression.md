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
```

## Initialization
```python
model = LinearRegression()
```

## Methods
### `fit(X, y)`
Fits the model to training data.

**Parameters:**
- `X`: Feature matrix (2D array-like, shape `(n_samples, n_features)`)
- `y`: Target vector (1D array-like, shape `(n_samples,)`)

**Returns:**  
Fitted model instance

### `predict(X)`
Makes predictions using fitted model.

**Parameters:**
- `X`: Input features (2D array-like, shape `(n_samples, n_features)`)

**Returns:**  
Predicted values (1D array)

### `score(X, y)`
Calculates R² (coefficient of determination).

**Parameters:**
- `X`: Test features
- `y`: True target values

**Returns:**  
R² score (float)

## Attributes
- `coefficients`: Feature weights (1D array)
- `intercept`: Bias term (float)

## Complete Example
```python
import numpy as np
from omniregress import LinearRegression

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 5, 7, 9, 11])  # y = 2x + 1

# Initialize and fit model
model = LinearRegression()
model.fit(X, y)

# Show parameters
print(f"Intercept: {model.intercept:.2f}")        # 1.00
print(f"Coefficient: {model.coefficients[0]:.2f}")  # 2.00

# Make predictions
X_test = np.array([[6], [7]])
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")  # [13. 15.]

# Calculate score
r2 = model.score(X, y)
print(f"R² score: {r2:.4f}")  # 1.0
```

## Handling Multiple Features
```python
# Multiple regression example
X_multi = np.array([[1, 2], [2, 4], [3, 1], [4, 3]])
y_multi = np.array([8, 14, 8, 14])  # y = 2x₁ + 3x₂

model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

print(f"Intercept: {model_multi.intercept:.2f}")           # 0.00
print(f"Coefficients: {model_multi.coefficients.round(2)}")  # [2. 3.]
```

## Notes
- Input data is automatically converted to numpy arrays
- Handles singular matrices with pseudoinverse fallback
- For large datasets (>10k samples), consider gradient-based implementations