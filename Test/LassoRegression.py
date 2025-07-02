import numpy as np
from omniregress import LassoRegression


def test_lasso_regression():
    print("=== Testing Lasso Regression ===")
    np.random.seed(42)

    # Generate synthetic data
    X = np.random.randn(100, 10)
    true_coef = np.array([1.5, -2.0, 0.0, 0.0, 3.0, 0.0, 0.5, 0.0, -1.0, 0.0])  # Sparse coefficients
    y = X @ true_coef + np.random.normal(0, 0.5, 100)

    # Test different alpha values
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        model = LassoRegression(alpha=alpha, max_iter=1000)
        model.fit(X, y)

        # Check coefficients shape
        assert model.coefficients.shape == (10,)
        assert isinstance(model.intercept, float)

        # Make predictions
        y_pred = model.predict(X)
        assert y_pred.shape == (100,)

        # Check score
        score = model.score(X, y)
        assert 0 <= score <= 1

        # Check sparsity
        zero_coefs = np.sum(np.abs(model.coefficients) < 1e-6)
        print(f"Alpha: {alpha:.2f}, R²: {score:.4f}, Zero coefs: {zero_coefs}")
        print("True coefficients:", np.round(true_coef, 4))
        print("Estimated coefficients:", np.round(model.coefficients, 4))
        print()

    # Test with intercept
    y_with_intercept = y + 2.5
    model = LassoRegression(alpha=0.1)
    model.fit(X, y_with_intercept)
    assert abs(model.intercept - 2.5) < 0.5  # Should be close

    print("Lasso Regression Test Passed!\n")


if __name__ == "__main__":
    test_lasso_regression()