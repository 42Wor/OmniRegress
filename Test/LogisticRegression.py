from omniregress import LogisticRegression


def test_logistic_regression():
    print("=== Testing Logistic Regression ===")

    # Binary classification data
    X = [[0.5], [1.0], [1.5], [2.0], [2.5], [3.0]]
    y = [0, 0, 0, 1, 1, 1]

    # Initialize and fit model
    model = LogisticRegression(learning_rate=0.1, max_iter=1000)
    model.fit(X, y)

    # Test parameters
    print(f"Coefficients: {model.coefficients}")
    print(f"Intercept: {model.intercept}")

    # Test predictions
    test_X = [[1.0], [2.0], [3.5]]
    probabilities = model.predict_proba(test_X)
    predictions = model.predict(test_X)

    print(f"Probabilities: {probabilities}")
    print(f"Predictions: {predictions}")

    # Verify predictions make sense
    assert predictions[0] == 0, "Should predict class 0"
    assert predictions[1] == 1, "Should predict class 1"
    assert predictions[2] == 1, "Should predict class 1"

    # Verify probabilities are between 0 and 1
    assert all(0 <= p <= 1 for p in probabilities), "Invalid probabilities"

    print("Logistic Regression Test Passed!\n")


if __name__ == "__main__":
    test_logistic_regression()