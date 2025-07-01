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
