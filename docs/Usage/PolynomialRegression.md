
# 🌟 Polynomial Regression: Quick & Cool Guide

## 🚀 What is Polynomial Regression?
`PolynomialRegression` lets you fit curves, not just lines!  
- **Flexible degree:** Model curves of any shape  
- **Auto feature transformation:** No manual work  
- **Easy API:** Just like `LinearRegression`

---

## 📦 How to Import
```python
from omniregress import PolynomialRegression
```

---

## ⚙️ Getting Started
```python
model = PolynomialRegression(degree=3)  # Try degree=2, 3, 4...
```

---

## 🛠️ Main Methods

### `fit(X, y)`
Train your model on data.

- `X`: 1D array-like, features (e.g., `[1, 2, 3]`)
- `y`: 1D array-like, targets (e.g., `[1, 4, 9]`)
- **Returns:** The fitted model

---

### `predict(X)`
Predict new values.

- `X`: 1D array-like, features
- **Returns:** 1D array, predictions

---

### `score(X, y)`
Get the R² score (how well the model fits).

- `X`: Test features
- `y`: True targets
- **Returns:** R² score (float)

---

## 🏷️ Attributes

- `degree`: The polynomial degree (int)

---

## 💡 Full Example

```python
from omniregress import PolynomialRegression

# Create quadratic data
X = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]  # y = x²

# Initialize and fit model (degree=2)
model = PolynomialRegression(degree=2)
model.fit(X, y)

# Access underlying linear model
print("Polynomial coefficients:")
print(f"Intercept: {model.intercept:.2f}")          # 0.00
print(f"Coefficients: {model.coefficients.round(2)}")  # [0. 1.] (x² term)

# Make predictions
X_test =[6, 7]
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")  # [36. 49.]

# Calculate score
r2 = model.score(X, y)
print(f"R² score: {r2:.4f}")  # 1.0
```

---

## 🔥 Degree Matters

```python
from omniregress import PolynomialRegression

# Cubic data: y = x³
X_cubic = [1, 2, 3, 4]
y_cubic = [1, 8, 27, 64]

# Underfit (degree=2)
model_underfit = PolynomialRegression(degree=2)
model_underfit.fit(X_cubic, y_cubic)
underfit_score = model_underfit.score(X_cubic, y_cubic)
print(f"Underfit R²: {underfit_score:.4f}")

# Correct fit (degree=3)
model_correct = PolynomialRegression(degree=3)
model_correct.fit(X_cubic, y_cubic)
correct_score = model_correct.score(X_cubic, y_cubic)
print(f"Correct R²: {correct_score:.4f}")
```

---

## ⚠️ Tips & Notes

1. **Input:**  
   - `X` must be 1D  
   - For multi-dimensional, use feature engineering

2. **Choosing Degree:**  
   - Start with 2, increase if needed  
   - Use validation to avoid overfitting

3. **Equation:**  
   ```
   y = intercept + coef[0]*x¹ + coef[1]*x² + ... + coef[n-1]*xⁿ
   ```

4. **Performance:**  
   - Great for small datasets  
   - High degrees (>10) may be unstable

---

✨ **PolynomialRegression**: Curve fitting made simple!


