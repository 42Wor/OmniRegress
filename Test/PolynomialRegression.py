import numpy as np
from omniregress import PolynomialRegression


def test_polynomial_regression():
    print("=== Testing Polynomial Regression ===")

    # Generate data for y = x² relationship
    X = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)  # Smaller range for numerical stability
    y = X * X  # y = x²

    # Initialize and fit model
    model = PolynomialRegression(degree=2)
    model.fit(X, y)

    # Test parameters
    print(f"Raw Coefficients: {model.coefficients}")
    print(f"Intercept: {model.intercept}")

    # Test predictions with smaller values first
    test_X = np.array([1.5, -1.5], dtype=np.float64)
    predictions = model.predict(test_X)
    expected = test_X * test_X

    print(f"Test X: {test_X}")
    print(f"Predictions: {predictions}")
    print(f"Expected: {expected}")
    print(f"Absolute Error: {np.abs(predictions - expected)}")

    assert np.allclose(predictions, expected, atol=0.1), "Predictions incorrect"

    print("Polynomial Regression Test Passed!\n")


if __name__ == "__main__":
    test_polynomial_regression()