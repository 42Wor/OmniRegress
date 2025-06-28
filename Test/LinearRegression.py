from omniregress import LinearRegression


def test_linear_regression():
    print("=== Testing Linear Regression ===")

    # Simple linear relationship y = 2x + 1
    X = [1, 2, 3, 4, 5]
    y = [3, 5, 7, 9, 11]

    # Initialize and fit model
    model = LinearRegression()
    model.fit(X, y)

    # Test coefficients
    print(f"Coefficients: {model.coefficients}")
    print(f"Intercept: {model.intercept}")
    assert model.intercept
    assert model.coefficients[0]

    # Test predictions
    test_X = [6, 7]
    predictions = model.predict(test_X)
    expected = [13., 15.]
    print(f"Predictions: {predictions}")


    # Test score
    score = model.score(X, y)
    print(f"R² score: {score:.4f}")
    assert score > 0.99, "Score too low"

    print("Linear Regression Test Passed!\n")


if __name__ == "__main__":
    test_linear_regression()