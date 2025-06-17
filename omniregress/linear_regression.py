import numpy as np


class LinearRegression:
    """
    A simple implementation of linear regression using the normal equation.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        """
        Fit the linear regression model.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values

        Returns:
        self : returns an instance of self.
        """
        # Add a column of ones for the intercept term
        X = np.array(X)
        y = np.array(y)
        X = np.c_[np.ones(X.shape[0]), X]

        # Calculate coefficients using the normal equation
        theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

        self.intercept = theta[0]
        self.coefficients = theta[1:]

        return self

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Samples to predict

        Returns:
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        if self.coefficients is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        X = np.array(X)
        return self.intercept + np.dot(X, self.coefficients)

    def score(self, X, y):
        """
        Calculate the R-squared score.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True values

        Returns:
        score : float
            R-squared score
        """
        y_pred = self.predict(X)
        y_mean = np.mean(y)
        ss_total = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_total)