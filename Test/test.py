import numpy as np
from omniregress import LinearRegression, PolynomialRegression, LogisticRegression


def test_linear_regression():
    print("\n=== Testing Linear Regression ===")
    # Simple linear relationship y = 2x + 1
    X = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y = np.array([3, 5, 7, 9, 11], dtype=np.float64)

    model = LinearRegression()
    model.fit(X, y)

    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")

    test_X = np.array([6, 7], dtype=np.float64)
    predictions = model.predict(test_X)
    print(f"Predictions for [6, 7]: {predictions}")

    score = model.score(X, y)
    print(f"R² score: {score:.4f}")

    expected = np.array([13., 15.])
    assert np.allclose(predictions, expected), "Linear regression predictions incorrect"
    print("Linear Regression Test Passed!")


def test_polynomial_regression():
    print("\n=== Testing Polynomial Regression ===")
    # Quadratic relationship y = x^2 + 2x + 1
    X = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y = np.array([4, 9, 16, 25, 36], dtype=np.float64)

    model = PolynomialRegression(degree=2)
    model.fit(X, y)

    print(f"Coefficients: {model.coefficients}")
    print(f"Intercept: {model.intercept}")

    test_X = np.array([6, 7], dtype=np.float64)
    predictions = model.predict(test_X)
    print(f"Predictions for [6, 7]: {predictions}")

    score = model.score(X, y)
    print(f"R² score: {score:.4f}")

    expected = np.array([49., 64.])
    assert np.allclose(predictions, expected), "Polynomial regression predictions incorrect"
    print("Polynomial Regression Test Passed!")


def test_logistic_regression():
    print("\n=== Testing Logistic Regression ===")
    # Binary classification (sigmoid curve)
    X = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float64).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)

    model = LogisticRegression(learning_rate=0.1, max_iter=1000)
    model.fit(X, y)

    print(f"Coefficients: {model.coefficients}")
    print(f"Intercept: {model.intercept}")

    test_X = np.array([1.0, 2.0, 3.5], dtype=np.float64).reshape(-1, 1)
    probabilities = model.predict_proba(test_X)
    predictions = model.predict(test_X)

    print(f"Probabilities for [1.0, 2.0, 3.5]: {probabilities}")
    print(f"Predictions for [1.0, 2.0, 3.5] (threshold=0.5): {predictions}")

    expected_probs = np.array([0.0, 1.0, 1.0])  # Approximate
    assert np.allclose(predictions, expected_probs, atol=0.1), "Logistic regression predictions incorrect"
    print("Logistic Regression Test Passed!")


if __name__ == "__main__":
    try:
        test_linear_regression()
        test_polynomial_regression()
        test_logistic_regression()
        print("\nAll tests passed successfully!")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
    except Exception as e:
        print(f"\nError during testing: {e}")