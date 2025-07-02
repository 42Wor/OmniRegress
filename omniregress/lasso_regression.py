import numpy as np

try:
    from ._omniregress import RustLassoRegression as _RustLassoRegressionInternal
except ImportError as e:
    raise ImportError(
        "Could not import Rust backend. Please install the compiled package."
    ) from e


class LassoRegression:
    """Lasso Regression with Rust backend implementing coordinate descent."""

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        """
        Parameters:
        -----------
        alpha : float, default=1.0
            Regularization strength; must be a positive float
        max_iter : int, default=1000
            Maximum number of iterations for coordinate descent
        tol : float, default=1e-4
            Tolerance for stopping criteria
        """
        self._rust_model = _RustLassoRegressionInternal(alpha, max_iter, tol)
        self._is_fitted = False

    @property
    def coefficients(self):
        """Coefficients of the features in the decision function."""
        if not self._is_fitted:
            return None
        return np.array(self._rust_model.coefficients)

    @property
    def intercept(self):
        """Independent term in the linear model."""
        if not self._is_fitted:
            return None
        return self._rust_model.intercept

    @property
    def alpha(self):
        """Regularization strength."""
        return self._rust_model.alpha

    @alpha.setter
    def alpha(self, value):
        """Set regularization strength."""
        if value <= 0:
            raise ValueError("alpha must be positive")
        self._rust_model.alpha = value

    @property
    def max_iter(self):
        """Maximum number of iterations."""
        return self._rust_model.max_iter

    @max_iter.setter
    def max_iter(self, value):
        """Set maximum number of iterations."""
        if value <= 0:
            raise ValueError("max_iter must be positive")
        self._rust_model.max_iter = value

    @property
    def tol(self):
        """Tolerance for stopping criteria."""
        return self._rust_model.tol

    @tol.setter
    def tol(self, value):
        """Set tolerance for stopping criteria."""
        if value <= 0:
            raise ValueError("tol must be positive")
        self._rust_model.tol = value

    def fit(self, X, y):
        """Fit Lasso regression model using coordinate descent.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        X_array = np.asarray(X, dtype=np.float64)
        y_array = np.asarray(y, dtype=np.float64)

        if X_array.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if y_array.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must have same number of samples")

        self._rust_model.fit(X_array, y_array)
        self._is_fitted = True
        return self

    def predict(self, X):
        """Make predictions.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        array of shape (n_samples,)
            Predicted values
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        X_array = np.asarray(X, dtype=np.float64)
        if X_array.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if X_array.shape[1] != len(self.coefficients):
            raise ValueError("Number of features doesn't match training data")

        return self._rust_model.predict(X_array)

    def score(self, X, y):
        """Return R² score.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values for X

        Returns:
        --------
        float
            R² score
        """
        y_pred = self.predict(X)
        y_true = np.asarray(y, dtype=np.float64)

        u = ((y_true - y_pred) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()
        return 1.0 - u / v if v != 0 else 1.0