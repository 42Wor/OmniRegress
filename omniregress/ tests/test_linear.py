import numpy as np
from omniregress import LinearRegression


def test_linear_regression():
    # Create some simple data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])  # Perfect linear relationship

    # Create and fit model
    model = LinearRegression()
    model.fit(X, y)

    # Test predictions
    y_pred = model.predict(X)
    assert np.allclose(y_pred, y), "Predictions should match perfect linear relationship"

    # Test coefficients
    assert np.isclose(model.intercept, 0), "Intercept should be 0"
    assert np.isclose(model.coefficients[0], 2), "Slope should be 2"

    # Test score
    assert np.isclose(model.score(X, y), 1.0), "R-squared should be 1.0 for perfect fit"

    print("All linear regression tests passed!")


if __name__ == "__main__":
    test_linear_regression()