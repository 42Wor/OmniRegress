# import numpy as np
from omniregress import PolynomialRegression


def test_polynomial_regression():
    print("=== Testing Polynomial Regression ===")

    # Generate data for y = x² relationship
    X = [-1.0, -0.5, 0.0, 0.5, 1.0]  # Smaller range for numerical stability
    y = [x * x for x in X]  # y = x²

    # Initialize and fit model
    model = PolynomialRegression(degree=2)
    model.fit(X, y)

    # Test parameters
    print(f"Raw Coefficients: {model.coefficients}")
    print(f"Intercept: {model.intercept}")

    # Test predictions with smaller values first
    test_X = [1.5, -1.5]
    predictions = model.predict(test_X)
    expected = [x * x for x in test_X]
    print(f"Test X: {test_X}")
    print(f"Predictions: {predictions}")
    print(f"Expected: {expected}")
    print(f"Absolute Error: {[abs(p - e) for p, e in zip(predictions, expected)]}")

    assert all(abs(p - e) < 0.1 for p, e in zip(predictions, expected)), "Predictions incorrect"

    print("Polynomial Regression Test Passed!\n")


if __name__ == "__main__":
    test_polynomial_regression()