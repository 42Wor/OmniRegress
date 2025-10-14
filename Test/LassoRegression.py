import numpy as np
from omniregress import LassoRegression


def test_lasso_regression():
    print("=== Testing Lasso Regression ===")
    np.random.seed(42)
    
    # Create dataset with some irrelevant features
    n_samples, n_features = 100, 10
    X = np.random.randn(n_samples, n_features)
    
    # Only first 3 features are relevant
    true_coef = np.array([1.5, -2.0, 3.0] + [0.0] * (n_features - 3))
    y = X @ true_coef + np.random.normal(0, 0.1, n_samples)

    print("True coefficients (first 3 relevant, rest zero):", true_coef)

    # Test different alpha values
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        model = LassoRegression(alpha=alpha, max_iter=1000, tol=1e-4)
        model.fit(X, y)

        # Check coefficients shape
        assert model.coefficients.shape == (n_features,)
        assert isinstance(model.intercept, float)

        # Make predictions
        y_pred = model.predict(X)
        assert y_pred.shape == (n_samples,)

        # Check score
        score = model.score(X, y)
        assert 0 <= score <= 1

        # Count non-zero coefficients (feature selection)
        non_zero_coef = np.sum(np.abs(model.coefficients) > 1e-6)
        
        print(f"Alpha: {alpha:.2f}, R²: {score:.4f}, Non-zero coefficients: {non_zero_coef}")
        print(f"Estimated coefficients: {np.round(model.coefficients, 4)}")

    # Test with intercept
    y_with_intercept = y + 2.5
    model = LassoRegression(alpha=0.1)
    model.fit(X, y_with_intercept)
    assert abs(model.intercept - 2.5) < 0.5  # Should be close

    # Test feature selection capability
    print("\n=== Testing Feature Selection ===")
    model_strong = LassoRegression(alpha=5.0)
    model_strong.fit(X, y)
    
    strong_non_zero = np.sum(np.abs(model_strong.coefficients) > 1e-6)
    print(f"With strong regularization (alpha=5.0): {strong_non_zero} non-zero coefficients")
    print("Coefficients:", np.round(model_strong.coefficients, 4))

    print("Lasso Regression Test Passed!\n")


def test_lasso_sparse_solution():
    """Test that Lasso produces sparse solutions."""
    print("=== Testing Sparse Solution ===")
    np.random.seed(123)
    
    # Create very sparse true coefficients
    X = np.random.randn(200, 20)
    true_coef = np.zeros(20)
    true_coef[[0, 5, 10]] = [2.0, -1.5, 1.0]  # Only 3 non-zero features
    y = X @ true_coef + np.random.normal(0, 0.2, 200)

    model = LassoRegression(alpha=0.5)
    model.fit(X, y)
    
    estimated_non_zero = np.sum(np.abs(model.coefficients) > 1e-4)
    print(f"True non-zero features: 3")
    print(f"Estimated non-zero features: {estimated_non_zero}")
    print(f"Sparsity ratio: {estimated_non_zero / 20:.2f}")
    
    # Check that irrelevant features are set to zero
    irrelevant_indices = [i for i in range(20) if i not in [0, 5, 10]]
    max_irrelevant_coef = np.max(np.abs(model.coefficients[irrelevant_indices]))
    print(f"Maximum coefficient for irrelevant features: {max_irrelevant_coef:.6f}")
    
    assert max_irrelevant_coef < 0.1, "Lasso should set irrelevant features to near zero"


if __name__ == "__main__":
    test_lasso_regression()
    test_lasso_sparse_solution()