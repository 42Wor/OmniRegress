# Polynomial Regression Usage Guide

## Overview
The `PolynomialRegression` class extends linear regression to polynomial relationships. Key features:
- Configurable polynomial degree
- Automatic feature transformation
- Compatible API with LinearRegression

## Importing
```python
from omniregress import PolynomialRegression
```

## Initialization
```python
model = PolynomialRegression(degree=3)  # Default degree=2
```

## Methods
### `fit(X, y)`
Fits polynomial model to data.

**Parameters:**
- `X`: Input features (1D array-like, shape `(n_samples,)`)
- `y`: Target values (1D array-like, shape `(n_samples,)`)

**Returns:**  
Fitted model instance

### `predict(X)`
Makes predictions using fitted model.

**Parameters:**
- `X`: Input features (1D array-like, shape `(n_samples,)`)

**Returns:**  
Predicted values (1D array)

### `score(X, y)`
Calculates R² score.

**Parameters:**
- `X`: Test features
- `y`: True target values

**Returns:**  
R² score (float)

## Attributes
- `degree`: Polynomial degree (int)
- `linear_model`: Underlying LinearRegression instance

## Complete Example
```python
import numpy as np
from omniregress import PolynomialRegression

# Create quadratic data
X = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])  # y = x²

# Initialize and fit model (degree=2)
model = PolynomialRegression(degree=2)
model.fit(X, y)

# Access underlying linear model
print("Polynomial coefficients:")
print(f"Intercept: {model.linear_model.intercept:.2f}")          # 0.00
print(f"Coefficients: {model.linear_model.coefficients.round(2)}")  # [0. 1.] (x² term)

# Make predictions
X_test = np.array([6, 7])
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")  # [36. 49.]

# Calculate score
r2 = model.score(X, y)
print(f"R² score: {r2:.4f}")  # 1.0
```

## Handling Different Degrees
```python
# Cubic relationship
X_cubic = np.array([1, 2, 3, 4])
y_cubic = np.array([1, 8, 27, 64])  # y = x³

# Fit with insufficient degree (underfitting)
model_underfit = PolynomialRegression(degree=2)
model_underfit.fit(X_cubic, y_cubic)
underfit_score = model_underfit.score(X_cubic, y_cubic)
print(f"Underfit R²: {underfit_score:.4f}")  # ~0.90

# Fit with correct degree
model_correct = PolynomialRegression(degree=3)
model_correct.fit(X_cubic, y_cubic)
correct_score = model_correct.score(X_cubic, y_cubic)
print(f"Correct R²: {correct_score:.4f}")  # 1.0

# Fit with excessive degree (overfitting)
model_overfit = PolynomialRegression(degree=5)
model_overfit.fit(X_cubic, y_cubic)
overfit_score = model_overfit.score(X_cubic, y_cubic)
print(f"Overfit R²: {overfit_score:.4f}")  # 1.0 (perfect on train, but may not generalize)
```

## Important Notes
1. **Input Requirements:**
   - X must be 1-dimensional
   - For multi-dimensional polynomial regression, use feature engineering

2. **Degree Selection:**
   - Start with degree=2 and increase incrementally
   - Validate with test set or cross-validation
   - Watch for overfitting with higher degrees

3. **Equation Interpretation:**
   The underlying model represents:
   ```
   y = intercept + coef[0]*x¹ + coef[1]*x² + ... + coef[degree-1]*xⁿ
   ```

4. **Performance:**
   - Efficient for small datasets
   - For high degrees (>10), numerical stability may decrease