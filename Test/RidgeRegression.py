
import numpy as np
from omniregress import RidgeRegression


def test_ridge_regression():
    print("=== Testing Ridge Regression ===")
    np.random.seed(42)
    X = np.random.rand(100, 5)
    true_coef = np.array([1.5, -2.0, 3.0, 0.5, -1.0])
    y = X @ true_coef + np.random.normal(0, 0.1, 100)

    # Test different alpha values
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        model = RidgeRegression(alpha=alpha)
        model.fit(X, y)

        # Check coefficients shape
        assert model.coefficients.shape == (5,)
        assert isinstance(model.intercept, float)

        # Make predictions
        y_pred = model.predict(X)
        assert y_pred.shape == (100,)

        # Check score
        score = model.score(X, y)
        assert 0 <= score <= 1

        print(f"Alpha: {alpha:.2f}, R²: {score:.4f}, Coefficients:",
              np.round(model.coefficients, 4))

    # Test with intercept
    y_with_intercept = y + 2.5
    model = RidgeRegression(alpha=0.1)
    model.fit(X, y_with_intercept)
    assert abs(model.intercept - 2.5) < 0.5  # Should be close

    print("Ridge Regression Test Passed!\n")


if __name__ == "__main__":
    test_ridge_regression()