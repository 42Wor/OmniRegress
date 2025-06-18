# omniregress/linear_regression.py
import numpy as np

try:
    # Import the Rust internal implementation
    from ._omniregress import _RustLinearRegressionInternal
except ImportError as e:
    raise ImportError(
        "Could not import the Rust-based _RustLinearRegressionInternal. "
        "Please ensure that the package was compiled and installed correctly. "
        f"Original error: {e}"
    ) from e


class LinearRegression:
    """
    Linear Regression model implemented with a from-scratch Rust backend.

    This model uses basic Python lists for communication with the Rust core,
    and provides a NumPy-friendly interface.
    """

    def __init__(self):
        self._rust_model = _RustLinearRegressionInternal()
        self._is_fitted = False
        self._coefficients = None
        self._intercept = None

    def _preprocess_x(self, X):
        """Converts X to list of lists of floats and validates."""
        if isinstance(X, np.ndarray):
            if X.ndim == 1:  # Convert 1D array to 2D column vector
                X_conv = X.reshape(-1, 1).astype(float).tolist()
            elif X.ndim == 2:
                X_conv = X.astype(float).tolist()
            else:
                raise ValueError("Input X must be 1D or 2D if NumPy array.")
        elif isinstance(X, list):
            if not X:  # Empty list
                return []
            if not X[0] or isinstance(X[0], (int, float)):  # List of numbers (1D) or empty list
                X_conv = [[float(i)] for i in X]
            elif all(isinstance(row, list) and all(isinstance(i, (int, float)) for i in row) for row in
                     X):  # List of lists of numbers
                X_conv = [[float(i) for i in row] for row in X]
            else:
                raise ValueError("Input X, if list, must be a list of numbers or list of lists of numbers.")
        else:
            raise TypeError("Input X must be a NumPy array or a list.")

        # Validate rectangular shape for non-empty X
        if X_conv:
            # Handle case where X_conv might be list of empty lists if input was e.g. [[]]
            if not X_conv[0] and len(X_conv) > 1 and any(X_conv[i] for i in range(1, len(X_conv))):
                raise ValueError("Input X must be rectangular (all rows/sublists must have the same length).")
            if X_conv[0]:
                first_row_len = len(X_conv[0])
                if not all(len(row) == first_row_len for row in X_conv):
                    raise ValueError("Input X must be rectangular (all rows/sublists must have the same length).")
        return X_conv

    def _preprocess_y(self, y):
        """Converts y to list of floats and validates."""
        if isinstance(y, np.ndarray):
            if y.ndim != 1:
                raise ValueError("Input y must be 1D if NumPy array.")
            y_conv = y.astype(float).tolist()
        elif isinstance(y, list):
            if not all(isinstance(i, (int, float)) for i in y):
                raise ValueError("Input y, if list, must be a list of numbers.")
            y_conv = [float(i) for i in y]
        else:
            raise TypeError("Input y must be a NumPy array or a list.")
        return y_conv

    def fit(self, X, y):
        """
        Fit linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        x_processed = self._preprocess_x(X)
        y_processed = self._preprocess_y(y)

        if not x_processed and not y_processed:
            # Allow fitting on empty data if both are empty.
            # Rust side will handle 0-dimensional matrix operations, which might error or produce specific results.
            # For example, if X is 0x0, X_b^T * X_b might be 0x0. Inversion of 0x0 is problematic.
            # The Rust code currently errors if theta is empty or not of expected length.
            # If x_processed is empty, num_features will be 0. theta should be [intercept].
            # If x_processed is like [[]] (0 features, 1 sample), num_features is 0.
            pass  # Let Rust handle it, it has checks for empty inputs.
        elif len(x_processed) != len(y_processed):
            raise ValueError(f"Number of samples in X ({len(x_processed)}) must match y ({len(y_processed)}).")

        if not x_processed and y_processed:
            raise ValueError("Cannot fit with empty X and non-empty y.")

        # Ensure x_processed is not a list of empty lists if y is not empty
        if y_processed and x_processed and not x_processed[0] and any(row for row in x_processed):
            raise ValueError("X contains empty rows (0 features) but y is not empty or X has multiple such rows.")

        self._rust_model.fit(x_processed, y_processed)

        # CORRECTED ACCESS: Use property names as exposed by PyO3's #[getter]
        raw_coeffs = self._rust_model.coefficients
        self._coefficients = np.array(raw_coeffs) if raw_coeffs is not None else None

        self._intercept = self._rust_model.intercept  # This is already an Option<f64> -> Optional[float]

        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "This LinearRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        x_processed = self._preprocess_x(X)

        # Handle prediction with 0 features if model was fit on 0 features
        if self._coefficients is not None and len(self._coefficients) == 0:  # Model has 0 features (only intercept)
            if x_processed and x_processed[0]:  # Input X has features
                raise ValueError(
                    "Model was fitted on data with 0 features, but predict input X has features."
                )
            # If X also has 0 features (e.g. list of empty lists, or empty list), prediction is just intercept
            # The Rust predict will handle this logic based on its stored coefficients.
        elif self._coefficients is not None and x_processed and x_processed[0]:  # Model and input have features
            num_features_input = len(x_processed[0])
            num_features_model = len(self._coefficients)
            if num_features_input != num_features_model:
                raise ValueError(
                    f"Number of features in X for prediction ({num_features_input}) "
                    f"does not match fitted model ({num_features_model})."
                )
        elif self._coefficients is None and x_processed and x_processed[0]:
            # This case should ideally not happen if _is_fitted is true and fit was successful.
            # It implies coefficients are None but model claims to be fitted.
            raise RuntimeError("Model is fitted but coefficients are missing. This indicates an internal issue.")
        elif not x_processed:  # Predicting on empty X
            return np.array([], dtype=float)

        predictions_list = self._rust_model.predict(x_processed)
        return np.array(predictions_list, dtype=float)

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        y_pred_np = self.predict(X)  # Returns NumPy array

        # Preprocess y to ensure it's a flat list of floats, then convert to NumPy array
        y_true_list = self._preprocess_y(y)
        y_true_np = np.array(y_true_list, dtype=float)

        if len(y_true_np) == 0:
            return 1.0 if len(y_pred_np) == 0 else 0.0  # Or np.nan, or raise error

        if len(y_true_np) != len(y_pred_np):
            # This should ideally be caught by X,y length checks earlier if predict didn't error
            raise ValueError(f"Length of y_true ({len(y_true_np)}) and y_pred ({len(y_pred_np)}) mismatch for scoring.")

        u = ((y_true_np - y_pred_np) ** 2).sum()
        v = ((y_true_np - y_true_np.mean()) ** 2).sum()
        if v == 0:  # Avoid division by zero
            return 1.0 if u == 0 else 0.0  # Perfect fit or constant y_true
        return 1.0 - u / v

    @property
    def coefficients(self):
        """Coefficient of the features in the decision function. NumPy array or None."""
        if not self._is_fitted:
            # To be consistent with scikit-learn, which raises NotFittedError
            # raise RuntimeError("This LinearRegression instance is not fitted yet.")
            return None  # Or raise NotFittedError for stricter scikit-learn compatibility
        return self._coefficients

    @property
    def intercept(self):
        """Independent term in the linear model. Float or None."""
        if not self._is_fitted:
            # raise RuntimeError("This LinearRegression instance is not fitted yet.")
            return None  # Or raise NotFittedError
        return self._intercept