import numpy as np

try:
    from ._omniregress import RustLassoRegression as _RustLassoRegressionInternal
except ImportError as e:
    raise ImportError(
        "Could not import Rust backend. Please install the compiled package."
    ) from e

class LassoRegression:
    """Lasso Regression (L1 regularization) with Rust backend.
    
    Lasso regression performs both regularization and feature selection
    by driving some coefficients to exactly zero.
    """

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        """
        Parameters:
        -----------
        alpha : float, default=1.0
            Regularization strength; must be a positive float.
            Larger values result in more sparse solutions.
        max_iter : int, default=1000
            Maximum number of iterations for coordinate descent.
        tol : float, default=1e-4
            Tolerance for convergence.
        """
        self._rust_model = _RustLassoRegressionInternal(alpha, max_iter, tol)
        self._is_fitted = False

    @property
    def coefficients(self):
        """Coefficients of the features in the decision function.
        
        Note: Lasso may set some coefficients to exactly zero (feature selection).
        """
        if not self._is_fitted:
            return None
        return np.array(self._rust_model.coefficients or [])

    @property
    def intercept(self):
        """Independent term in the linear model."""
        if not self._is_fitted:
            return None
        return self._rust_model.intercept or 0.0

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
        """Convergence tolerance."""
        return self._rust_model.tol

    @tol.setter
    def tol(self, value):
        """Set convergence tolerance."""
        if value <= 0:
            raise ValueError("tol must be positive")
        self._rust_model.tol = value

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
        """Fit lasso regression model using coordinate descent.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        X_processed = self._ensure_2d_array(X)
        y_processed = self._ensure_1d_array(y)

        if len(X_processed) != len(y_processed):
            raise ValueError("X and y must have same number of samples")

        self._rust_model.fit(X_processed, y_processed)
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
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        X_processed = self._ensure_2d_array(X)
        return np.array(self._rust_model.predict(X_processed))

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
        score : float
            R² of self.predict(X) wrt. y
        """
        y_pred = self.predict(X)
        y_true = np.array(self._ensure_1d_array(y))

        u = ((y_true - y_pred) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()
        return 1.0 - u / v if v != 0 else 1.0

    def get_nonzero_coefficients(self, threshold=1e-6):
        """Get indices and values of non-zero coefficients.
        
        Parameters:
        -----------
        threshold : float, default=1e-6
            Minimum absolute value to consider as non-zero
            
        Returns:
        --------
        indices : ndarray
            Indices of non-zero coefficients
        values : ndarray  
            Values of non-zero coefficients
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
            
        coef = self.coefficients
        mask = np.abs(coef) > threshold
        indices = np.where(mask)[0]
        values = coef[mask]
        
        return indices, values

    def sparsity_ratio(self, threshold=1e-6):
        """Compute the ratio of zero coefficients.
        
        Parameters:
        -----------
        threshold : float, default=1e-6
            Minimum absolute value to consider as non-zero
            
        Returns:
        --------
        ratio : float
            Ratio of coefficients that are effectively zero
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
            
        coef = self.coefficients
        zero_count = np.sum(np.abs(coef) <= threshold)
        return zero_count / len(coef)