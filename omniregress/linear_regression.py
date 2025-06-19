import numpy as np

try:
    from ._omniregress import _RustLinearRegressionInternal
except ImportError as e:
    raise ImportError(
        "Could not import Rust backend. Please install the compiled package."
    ) from e


class LinearRegression:
    """Linear Regression with Rust backend."""

    def __init__(self):
        self._rust_model = _RustLinearRegressionInternal()
        self._is_fitted = False

    @property
    def coef_(self):
        """Coefficients of the features in the decision function."""
        if not self._is_fitted:
            return None
        return np.array(self._rust_model.coefficients or [])

    @property
    def intercept_(self):
        """Independent term in the linear model."""
        if not self._is_fitted:
            return None
        return self._rust_model.intercept or 0.0

    def _ensure_2d_array(self, X):
        """Convert input to 2D list of floats."""
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                return X.reshape(-1, 1).tolist()
            return X.tolist()
        elif isinstance(X, list):
            if not X or isinstance(X[0], (int, float)):
                return [[float(x)] for x in X]
            return [[float(x) for x in row] for row in X]
        raise TypeError("X must be numpy array or list")

    def _ensure_1d_array(self, y):
        """Convert input to 1D list of floats."""
        if isinstance(y, np.ndarray):
            return y.tolist()
        elif isinstance(y, list):
            return [float(val) for val in y]
        raise TypeError("y must be numpy array or list")

    def fit(self, X, y):
        """Fit linear model."""
        X_processed = self._ensure_2d_array(X)
        y_processed = self._ensure_1d_array(y)

        if len(X_processed) != len(y_processed):
            raise ValueError("X and y must have same number of samples")

        self._rust_model.fit(X_processed, y_processed)
        self._is_fitted = True
        return self

    def predict(self, X):
        """Make predictions."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        X_processed = self._ensure_2d_array(X)
        return np.array(self._rust_model.predict(X_processed))

    def score(self, X, y):
        """Return R² score."""
        y_pred = self.predict(X)
        y_true = np.array(self._ensure_1d_array(y))

        u = ((y_true - y_pred) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()
        return 1.0 - u / v if v != 0 else 1.0