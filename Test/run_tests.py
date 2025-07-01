import LinearRegression
import PolynomialRegression
import LogisticRegression
import RidgeRegression

from omniregress import __version__

def main():
    print(f"Running all tests for omniregress v{__version__}...\n")

    LinearRegression.test_linear_regression()
    PolynomialRegression.test_polynomial_regression()
    LogisticRegression.test_logistic_regression()
    RidgeRegression.test_ridge_regression()

    print("All tests completed!")

if __name__ == "__main__":
    main()