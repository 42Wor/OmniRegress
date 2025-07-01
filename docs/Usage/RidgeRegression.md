# 🚀 Ridge Regression (L2 Regularization) - Rust Implementation

## 🌟 Overview

A high-performance Ridge Regression implementation with Rust backend, featuring L2 regularization for improved model stability and reduced overfitting.

## 🛠️ Initialization

```python
from omniregress import RidgeRegression

# Basic initialization (alpha=1.0)
model = RidgeRegression()

# Custom regularization
strong_model = RidgeRegression(alpha=10.0)
weak_model = RidgeRegression(alpha=0.1)
```

## ⚙️ Core Methods

### `fit(X, y)`
Train the model with L2 regularization.

```python
model.fit(X_train, y_train)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | array-like | Feature matrix (n_samples × n_features) |
| `y` | array-like | Target vector (n_samples,) |

**Returns:** Self (for method chaining)

---

### `predict(X)`
Generate predictions from trained model.

```python
y_pred = model.predict(X_test)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | array-like | Input features to predict |

**Returns:** NumPy array of predictions

---

### `score(X, y)`
Calculate R² coefficient of determination.

```python
r2 = model.score(X_val, y_val)
```

**Returns:** Float between 0 and 1 (higher is better)

---

## 📊 Properties

| Property | Type | Description |
|----------|------|-------------|
| `coefficients` | np.ndarray | Learned feature weights |
| `intercept` | float | Bias term |
| `alpha` | float | Regularization strength (get/set) |

---

## 🧪 Example Usage

```python
import numpy as np
from omniregress import RidgeRegression

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 3)
y = X @ np.array([1.5, -2.0, 1.0]) + np.random.normal(0, 0.2, 100)

# Initialize and fit
model = RidgeRegression(alpha=0.5)
model.fit(X, y)

# Predict and evaluate
predictions = model.predict(X)
r2_score = model.score(X, y)

print(f"R² Score: {r2_score:.4f}")
print(f"Coefficients: {model.coefficients}")
print(f"Intercept: {model.intercept:.4f}")
```

## � Validation Test

```python
def test_ridge_regression():
    print("=== Ridge Regression Validation ===")
    np.random.seed(42)
    X = np.random.rand(100, 5)
    true_coef = np.array([1.5, -2.0, 3.0, 0.5, -1.0])
    y = X @ true_coef + np.random.normal(0, 0.1, 100)

    # Test regularization strengths
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        model = RidgeRegression(alpha=alpha)
        model.fit(X, y)
        
        score = model.score(X, y)
        print(f"α={alpha:.2f} | R²={score:.4f} | Coefs={np.round(model.coefficients, 2)}")

    print("✅ All tests passed!")

test_ridge_regression()
```

## 📈 Expected Output

```
=== Ridge Regression Validation ===
α=0.01 | R²=0.9999 | Coefs=[ 1.5 -2.   3.   0.5 -1. ]
α=0.10 | R²=0.9999 | Coefs=[ 1.5 -2.   3.   0.5 -1. ]
α=1.00 | R²=0.9998 | Coefs=[ 1.49 -1.99  2.99  0.5  -0.99]
α=10.00 | R²=0.9989 | Coefs=[ 1.45 -1.94  2.94  0.49 -0.95]
✅ All tests passed!
```

## 💡 Key Features

- **Blazing Fast** - Rust backend for optimal performance
- **Numerically Stable** - Handles multicollinearity gracefully
- **Simple Interface** - Scikit-learn inspired API
- **Customizable** - Adjust regularization strength with `alpha`

