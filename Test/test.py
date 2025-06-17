from omniregress import LinearRegression, PolynomialRegression
import numpy as np

# Linear regression example
X_lin = np.array([[1], [2], [3], [4], [5]])
y_lin = np.array([2, 4, 6, 8, 10])

lin_model = LinearRegression()
lin_model.fit(X_lin, y_lin)
print("Linear coefficients:", lin_model.coefficients)
print("Linear intercept:", lin_model.intercept)

# Polynomial regression example
X_poly = np.array([1, 2, 3, 4, 5])
y_poly = np.array([1, 4, 9, 16, 25])

poly_model = PolynomialRegression(degree=2)
poly_model.fit(X_poly, y_poly)
print("Polynomial predictions:", poly_model.predict(X_poly))