import numpy as np
from .linear_regression import LinearRegression


class PolynomialRegression:
    """
    A polynomial regression implementation using linear regression as a base.
    """

    def __init__(self, degree=2):
        """
        Parameters:
        degree : int, optional (default=2)
            The degree of the polynomial
        """
        self.degree = degree
        self.linear_model = LinearRegression()

    def _create_polynomial_features(self, X):
        """
        Create polynomial features from the input data.

        Parameters:
        X : array-like, shape (n_samples, 1)
            Input data (must be 1-dimensional)

        Returns:
        X_poly : array, shape (n_samples, degree)
            Polynomial features
        """
        X = np.array(X)
        if X.ndim != 1:
            raise ValueError("X must be 1-dimensional for polynomial regression")

        X_poly = np.column_stack([X ** i for i in range(1, self.degree + 1)])
        return X_poly

    def fit(self, X, y):
        """
        Fit the polynomial regression model.

        Parameters:
        X : array-like, shape (n_samples, 1)
            Training data (1-dimensional)
        y : array-like, shape (n_samples,)
            Target values

        Returns:
        self : returns an instance of self.
        """
        X_poly = self._create_polynomial_features(X)
        self.linear_model.fit(X_poly, y)
        return self

    def predict(self, X):
        """
        Predict using the polynomial model.

        Parameters:
        X : array-like, shape (n_samples, 1)
            Samples to predict (1-dimensional)

        Returns:
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        X_poly = self._create_polynomial_features(X)
        return self.linear_model.predict(X_poly)

    def score(self, X, y):
        """
        Calculate the R-squared score.

        Parameters:
        X : array-like, shape (n_samples, 1)
            Test samples (1-dimensional)
        y : array-like, shape (n_samples,)
            True values

        Returns:
        score : float
            R-squared score
        """
        X_poly = self._create_polynomial_features(X)
        return self.linear_model.score(X_poly, y)