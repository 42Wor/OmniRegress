# 🚀 Linear Regression Usage Guide

## 🌟 Overview
The `LinearRegression` class provides a simple yet powerful way to perform ordinary least squares regression using the normal equation. With it, you can:

- 📈 Fit models with single or multiple features
- 🧮 Estimate coefficients and intercept
- 🔮 Make predictions
- 📊 Evaluate model performance with R² scoring

---

## 📦 Importing

```python
from omniregress import LinearRegression
import numpy as np  # Required for examples
```

---

## 🛠️ Initialization

```python
model = LinearRegression()
```

---

## ⚙️ Methods

### `fit(X, y)`
Train the model on your data.

- **Parameters:**
  - `X`: Feature data (array-like, shape `(n_samples, n_features)` or `(n_samples,)`)
  - `y`: Target vector (1D array-like, shape `(n_samples,)`)
- **Returns:** Fitted model instance

---

### `predict(X)`
Generate predictions from the fitted model.

- **Parameters:**
  - `X`: Input features (array-like, shape `(n_samples, n_features)` or `(n_samples,)`)
- **Returns:** Predicted values (1D NumPy array)

---

### `score(X, y)`
Compute the R² (coefficient of determination) score.

- **Parameters:**
  - `X`: Test features
  - `y`: True target values
- **Returns:** R² score (float)

---

## 🏷️ Attributes

- `coefficients`: Feature weights (1D NumPy array or `None` if not fitted)
- `intercept`: Bias term (float or `None` if not fitted)

---

## 👨‍💻 Complete Example

### 🎯 Single Feature Example

```python
from omniregress import LinearRegression

model = LinearRegression()
# Create sample data where y = 2x + 1
X = [1, 2, 3, 4, 5]
y = [3, 5, 7, 9, 11]

# Initialize and fit model
model = LinearRegression()
model.fit(X, y)

# Show parameters
print("--- Single Feature Example ---")
print(f"Intercept: {model.intercept:.2f}")        # Expected: 1.00
print(f"Coefficient: {model.coefficients[0]:.2f}")       # Expected: 2.00

# Make predictions
X_test = [6, 7]
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")  # Expected: [13. 15.]

# Calculate score
r2 = model.score(X, y)
print(f"R² score: {r2:.4f}")  # Expected: 1.0000
```

---

### 🧮 Multiple Features Example

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
print(f"Intercept: {model_multi.intercept:.2f}")  # Expected: 3.00
print(f"Coefficients: {np.round(model_multi.coefficients, 2)}")  # Expected: [1.00 2.00]")

# Predict on new data
X_test_multi = np.array([
    [1, 3],  # 3 + 1*1 + 2*3 = 10
    [4, 2]   # 3 + 1*4 + 2*2 = 11
])
predictions_multi = model_multi.predict(X_test_multi)
print(f"Predictions: {predictions_multi}")  # Expected: [10. 11.]
```

---

## 💡 Key Notes

1. **Input Handling**:
   - Automatically converts 1D inputs to 2D column vectors
   - Accepts both NumPy arrays and Python lists

2. **Performance**:
   - Uses the normal equation (matrix inversion)
   - Best for small-to-medium datasets (<10,000 samples)
   - May be slow for very large feature sets

3. **Numerical Stability**:
   - Includes partial pivoting in matrix inversion
   - Raises errors for singular matrices

4. **Scikit-learn Compatibility**:
   - Uses `coefficients` and `intercept` naming convention
   - Similar method signatures

---

## 🛠️ Troubleshooting

**Common Errors:**
- `ValueError: Matrix dimensions mismatch`: Check that X and y have the same number of samples
- `ValueError: Matrix is singular`: Features may be perfectly correlated

For large datasets, consider preprocessing with feature scaling or using gradient descent-based implementations.

---
