import numpy as np
from omniregress import PolynomialRegression


def test_polynomial_regression():
    # Perfect quadratic relationship
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 4, 9, 16, 25])

    model = PolynomialRegression(degree=2)
    model.fit(X, y)

    # Test predictions
    y_pred = model.predict(X)
    assert np.allclose(y_pred, y)

    # Test score
    assert np.isclose(model.score(X, y), 1.0)

    # Test degree handling
    model = PolynomialRegression(degree=3)
    model.fit(X, y)
    assert model.score(X, y) > 0.99

    print("Polynomial regression tests passed!")


if __name__ == "__main__":
    test_polynomial_regression()