import numpy as np
try:
 from ._omniregress import RustLogisticRegression as _RustLogisticRegressionInternal
except ImportError as e:
    raise ImportError(
        "Could not import Rust backend. Please install the compiled package."
    ) from e
class LogisticRegression:
    """
    Logistic Regression implementation with Rust backend.
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self._rust_model = _RustLogisticRegressionInternal(
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol
        )

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
        """
        Fit logistic regression model.
        """
        X_processed = self._ensure_2d_array(X)
        y_processed = self._ensure_1d_array(y)

        if len(X_processed) != len(y_processed):
            raise ValueError("X and y must have same number of samples")

        self._rust_model.fit(X_processed, y_processed)
        return self

    def predict_proba(self, X):
        """
        Predict probability estimates.
        """
        X_processed = self._ensure_2d_array(X)
        return np.array(self._rust_model.predict_proba(X_processed))

    def predict(self, X, threshold=0.5):
        """
        Predict class labels.
        """
        X_processed = self._ensure_2d_array(X)
        return np.array(self._rust_model.predict(X_processed, threshold))

    @property
    def coefficients(self):
        """Coefficients of the features in the decision function."""
        return np.array(self._rust_model.coefficients or [])

    @property
    def intercept(self):
        """Independent term in the linear model."""
        return self._rust_model.intercept or 0.0