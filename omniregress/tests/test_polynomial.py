import numpy as np
from omniregress import PolynomialRegression


def test_polynomial_regression():
    # Create some quadratic data
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 4, 9, 16, 25])  # Perfect quadratic relationship

    # Create and fit model
    model = PolynomialRegression(degree=2)
    model.fit(X, y)

    # Test predictions
    y_pred = model.predict(X)
    assert np.allclose(y_pred, y), "Predictions should match perfect quadratic relationship"

    # Test score
    assert np.isclose(model.score(X, y), 1.0), "R-squared should be 1.0 for perfect fit"

    print("All polynomial regression tests passed!")


if __name__ == "__main__":
    test_polynomial_regression()