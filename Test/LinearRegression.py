import numpy as np
from omniregress import LinearRegression


def test_linear_regression():
    print("=== Testing Linear Regression ===")

    # Simple linear relationship y = 2x + 1
    X = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y = np.array([3, 5, 7, 9, 11], dtype=np.float64)

    # Initialize and fit model
    model = LinearRegression()
    model.fit(X, y)

    # Test coefficients
    print(f"Coefficients: {model.coefficients}")
    print(f"Intercept: {model.intercept}")
    assert np.isclose(model.intercept, 1.0, atol=0.1), "Intercept incorrect"
    assert np.isclose(model.coefficients[0], 2.0, atol=0.1), "Coefficient incorrect"

    # Test predictions
    test_X = np.array([6, 7], dtype=np.float64)
    predictions = model.predict(test_X)
    expected = np.array([13., 15.])
    print(f"Predictions: {predictions}")
    assert np.allclose(predictions, expected, atol=0.1), "Predictions incorrect"

    # Test score
    score = model.score(X, y)
    print(f"R² score: {score:.4f}")
    assert score > 0.99, "Score too low"

    print("Linear Regression Test Passed!\n")


if __name__ == "__main__":
    test_linear_regression()