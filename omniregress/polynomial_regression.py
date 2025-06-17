import numpy as np
from .linear_regression import LinearRegression


class PolynomialRegression:
    """
    Polynomial Regression implementation

    Attributes:
        degree (int): Polynomial degree
        linear_model (LinearRegression): Underlying linear model

    Example:
        >>> model = PolynomialRegression(degree=3)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, degree=2):
        self.degree = degree
        self.linear_model = LinearRegression()

    def _create_polynomial_features(self, X):
        """
        Generate polynomial features

        Args:
            X (ndarray): Input data of shape (n_samples,)

        Returns:
            ndarray: Polynomial features of shape (n_samples, degree)
        """
        X = np.array(X)
        if X.ndim != 1:
            raise ValueError("Polynomial regression requires 1D input")

        return np.column_stack([X ** i for i in range(1, self.degree + 1)])

    def fit(self, X, y):
        """
        Fit polynomial regression model

        Args:
            X (ndarray): Training data of shape (n_samples,)
            y (ndarray): Target values of shape (n_samples,)

        Returns:
            self: Fitted model instance
        """
        X_poly = self._create_polynomial_features(X)
        self.linear_model.fit(X_poly, y)
        return self

    def predict(self, X):
        """
        Make predictions using fitted model

        Args:
            X (ndarray): Input data of shape (n_samples,)

        Returns:
            ndarray: Predicted values
        """
        X_poly = self._create_polynomial_features(X)
        return self.linear_model.predict(X_poly)

    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination)

        Args:
            X (ndarray): Test samples
            y (ndarray): True values

        Returns:
            float: R² score
        """
        X_poly = self._create_polynomial_features(X)
        return self.linear_model.score(X_poly, y)