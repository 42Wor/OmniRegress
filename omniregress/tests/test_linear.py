import numpy as np
from omniregress import LinearRegression


def test_linear_regression():
    # Perfect linear relationship
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    model = LinearRegression()
    model.fit(X, y)

    # Test coefficients
    assert np.isclose(model.intercept, 0)
    assert np.isclose(model.coefficients[0], 2)

    # Test predictions
    y_pred = model.predict(X)
    assert np.allclose(y_pred, y)

    # Test score
    assert np.isclose(model.score(X, y), 1.0)

    # Test with noise
    y_noisy = y + np.random.normal(0, 0.1, size=y.shape)
    model.fit(X, y_noisy)
    score = model.score(X, y_noisy)
    assert 0.95 < score < 1.0

    print("Linear regression tests passed!")


if __name__ == "__main__":
    test_linear_regression()