# omniregress/polynomial_regression.py
import numpy as np
from ._omniregress import RustPolynomialRegression as _RustPolynomialRegression

class PolynomialRegression:
    def __init__(self, degree=2):
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("Degree must be a positive integer.")
        self.degree = degree
        self._rust_model = _RustPolynomialRegression(degree)

    def fit(self, X, y):
        """
        Fit polynomial regression model

        Args:
            X (ndarray): Training data, 1D array-like (n_samples,)
            y (ndarray): Target values, 1D array-like (n_samples,)

        Returns:
            self: Fitted model instance
        """
        # Ensure X is a numpy array and convert to float64 for Rust compatibility
        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)

        if X_arr.ndim != 1:
            raise ValueError("Input X for PolynomialRegression must be a 1D array.")

        if y_arr.ndim != 1:
            raise ValueError("Target y must be a 1D array.")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"Shape mismatch: X has {X_arr.shape[0]} samples and y has {y_arr.shape[0]} samples."
            )

        self._rust_model.fit(X_arr.tolist(), y_arr.tolist())
        return self

    def predict(self, X):
        """
        Make predictions using fitted model

        Args:
            X (ndarray): Input data, 1D array-like (n_samples,)

        Returns:
            ndarray: Predicted values
        """
        X_arr = np.array(X, dtype=np.float64)
        if X_arr.ndim != 1:
            raise ValueError("Input X for prediction must be a 1D array.")

        predictions = self._rust_model.predict(X_arr.tolist())
        return np.array(predictions, dtype=np.float64)

    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination)

        Args:
            X (ndarray): Test samples, 1D array-like
            y (ndarray): True values, 1D array-like

        Returns:
            float: R² score
        """
        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64)

        if X_arr.ndim != 1:
            raise ValueError("Input X for scoring must be a 1D array.")

        if y_arr.ndim != 1:
            raise ValueError("Target y must be a 1D array.")

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"Shape mismatch: X has {X_arr.shape[0]} samples and y has {y_arr.shape[0]} samples."
            )

        return self._rust_model.score(X_arr.tolist(), y_arr.tolist())

    @property
    def coefficients(self):
        """Coefficients of the polynomial features (highest degree first)."""
        coeffs = self._rust_model.coefficients
        return np.array(coeffs, dtype=np.float64) if coeffs is not None else None

    @property
    def intercept(self):
        """Intercept term of the model."""
        intercept = self._rust_model.intercept
        return intercept if intercept is not None else 0.0