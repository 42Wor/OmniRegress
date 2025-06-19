# omniregress/polynomial_regression.py
import numpy as np
# Assuming LinearRegression is correctly imported from the Rust extension
# via omniregress.linear_regression or omniregress.__init__
from .linear_regression import LinearRegression


class PolynomialRegression:
    """
    Polynomial Regression implementation

    Attributes:
        degree (int): Polynomial degree
        linear_model (LinearRegression): Underlying linear model

    Example:
        >>> import numpy as np
        >>> model = PolynomialRegression(degree=3)
        >>> X_train = np.array([1, 2, 3, 4], dtype=np.float64)
        >>> y_train = np.array([1, 8, 27, 64], dtype=np.float64)
        >>> model.fit(X_train, y_train)
        >>> X_test = np.array([5, 6], dtype=np.float64)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, degree=2):
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("Degree must be a positive integer.")
        self.degree = degree
        self.linear_model = LinearRegression()

    def _create_polynomial_features(self, X):
        """
        Generate polynomial features

        Args:
            X (ndarray): Input data of shape (n_samples,) or (n_samples, 1)

        Returns:
            ndarray: Polynomial features of shape (n_samples, degree), dtype=np.float64
        """
        # Ensure X is a numpy array and convert to float64 for Rust compatibility
        X_arr = np.array(X, dtype=np.float64)

        if X_arr.ndim == 2 and X_arr.shape[1] == 1:
            X_arr = X_arr.ravel()  # Convert column vector to 1D array
        elif X_arr.ndim != 1:
            raise ValueError(
                "Input X for PolynomialRegression must be a 1D array or a 2D column vector."
            )

        if X_arr.size == 0:  # Handle empty input array
            return np.empty((0, self.degree), dtype=np.float64)

        # Create polynomial features. X_arr is already float64.
        features = [X_arr ** i for i in range(1, self.degree + 1)]
        return np.column_stack(features)

    def fit(self, X, y):
        """
        Fit polynomial regression model

        Args:
            X (ndarray): Training data, 1D array-like (n_samples,)
            y (ndarray): Target values, 1D array-like (n_samples,)

        Returns:
            self: Fitted model instance
        """
        X_poly = self._create_polynomial_features(X)

        # Ensure y is a numpy array and convert to float64 for Rust compatibility
        y_arr = np.array(y, dtype=np.float64)
        if y_arr.ndim != 1:
            raise ValueError("Target y must be a 1D array.")
        if X_poly.shape[0] != y_arr.shape[0] and X_poly.size > 0:  # Allow empty fit if X_poly is empty
            raise ValueError(
                f"Shape mismatch: X_poly has {X_poly.shape[0]} samples and y has {y_arr.shape[0]} samples.")

        self.linear_model.fit(X_poly, y_arr)
        return self

    def predict(self, X):
        """
        Make predictions using fitted model

        Args:
            X (ndarray): Input data, 1D array-like (n_samples,)

        Returns:
            ndarray: Predicted values
        """
        X_poly = self._create_polynomial_features(X)
        if X_poly.size == 0:  # Handle prediction on empty transformed features
            return np.array([], dtype=np.float64)
        return self.linear_model.predict(X_poly)

    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination)

        Args:
            X (ndarray): Test samples, 1D array-like
            y (ndarray): True values, 1D array-like

        Returns:
            float: R² score
        """
        X_poly = self._create_polynomial_features(X)

        # Ensure y is a numpy array and convert to float64 for Rust compatibility
        y_arr = np.array(y, dtype=np.float64)
        if y_arr.ndim != 1:
            raise ValueError("Target y must be a 1D array.")
        if X_poly.shape[0] != y_arr.shape[0] and X_poly.size > 0:
            raise ValueError(
                f"Shape mismatch: X_poly has {X_poly.shape[0]} samples and y has {y_arr.shape[0]} samples.")

        if X_poly.size == 0 and y_arr.size == 0:  # Score for empty data could be NaN or error, let Rust decide or handle here
            # Depending on desired behavior, could return 0, NaN, or raise error.
            # For now, let it pass to Rust, which might error or handle it.
            # If Rust's score can't handle empty, add: if X_poly.size == 0: return np.nan
            pass
        elif X_poly.size == 0 and y_arr.size != 0:  # Mismatch if one is empty and other is not
            raise ValueError("Cannot score with empty X_poly and non-empty y or vice-versa.")

        return self.linear_model.score(X_poly, y_arr)

    @property
    def coefficients(self):
        """Coefficients of the underlying linear model."""
        return self.linear_model.coef_

    @property
    def intercept(self):
        """Intercept of the underlying linear model."""
        return self.linear_model.intercept_