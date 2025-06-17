import numpy as np


class LinearRegression:
    """
    Linear Regression implementation using normal equation

    Attributes:
        coefficients (ndarray): Coefficients for each feature
        intercept (float): Bias term

    Example:
        >>> model = LinearRegression()
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        """
        Fit linear regression model to training data

        Args:
            X (ndarray): Training data of shape (n_samples, n_features)
            y (ndarray): Target values of shape (n_samples,)

        Returns:
            self: Fitted model instance
        """
        X = np.array(X)
        y = np.array(y)

        # Add bias term
        X = np.c_[np.ones(X.shape[0]), X]

        # Normal equation: θ = (XᵀX)⁻¹Xᵀy
        try:
            theta = np.linalg.inv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError:
            # Use pseudoinverse if singular matrix
            theta = np.linalg.pinv(X.T @ X) @ X.T @ y

        self.intercept = theta[0]
        self.coefficients = theta[1:]
        return self

    def predict(self, X):
        """
        Make predictions using fitted model

        Args:
            X (ndarray): Input data of shape (n_samples, n_features)

        Returns:
            ndarray: Predicted values
        """
        if self.coefficients is None:
            raise RuntimeError("Model not fitted. Call fit() first")

        X = np.array(X)
        return self.intercept + X @ self.coefficients

    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination)

        Args:
            X (ndarray): Test samples
            y (ndarray): True values

        Returns:
            float: R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)