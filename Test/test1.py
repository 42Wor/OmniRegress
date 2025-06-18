# Test/test1.py
from omniregress import LinearRegression, PolynomialRegression
import numpy as np

# Linear regression example
print("\n--- Testing Linear Regression ---")
X_lin = np.array([[1], [2], [3], [4], [5.5]], dtype=np.float64)
y_lin = np.array([2.1, 3.9, 6.1, 7.9, 11.2], dtype=np.float64)

lin_model = LinearRegression()
print("Fitting linear model...")
lin_model.fit(X_lin, y_lin)

print("Linear coefficients (from property):", lin_model.coefficients)
print("Linear intercept (from property):", lin_model.intercept)

lin_predictions = lin_model.predict(X_lin)
print("Linear predictions:", lin_predictions)
lin_score = lin_model.score(X_lin, y_lin)
print("Linear R^2 score:", lin_score)

# Test with new data
X_new_lin = np.array([[6], [7]], dtype=np.float64) # Ensure float64
print("Linear predictions for new data:", lin_model.predict(X_new_lin))


# Polynomial regression example
print("\n--- Testing Polynomial Regression ---")
# Using a single feature for simplicity
X_poly_simple = np.array([1, 2, 3, 4, 5], dtype=np.float64) # Ensure float64
y_poly_simple = np.array([2, 7, 16, 29, 48], dtype=np.float64) # y = 2x^2 + x - 1 (approx), ensure float64

poly_model = PolynomialRegression(degree=2)
print("Fitting polynomial model (degree=2)...")
poly_model.fit(X_poly_simple, y_poly_simple)

print("Polynomial coefficients (from internal linear model):", poly_model.coefficients)
print("Polynomial intercept (from internal linear model):", poly_model.intercept)

poly_predictions = poly_model.predict(X_poly_simple)
print("Polynomial predictions for training data:", poly_predictions)

# Test with new data for polynomial
X_new_poly = np.array([6, 7], dtype=np.float64) # Ensure float64
print("Polynomial predictions for new data [6, 7]:", poly_model.predict(X_new_poly))

# Example with X as a 2D column vector for PolynomialRegression
X_poly_col = np.array([[1], [2], [3], [4], [5]], dtype=np.float64) # Ensure float64
y_poly_col = np.array([2, 7, 16, 29, 48], dtype=np.float64) # Ensure float64
poly_model_col = PolynomialRegression(degree=2)
print("Fitting polynomial model (degree=2) with column vector X...")
poly_model_col.fit(X_poly_col, y_poly_col)
poly_predictions_col = poly_model_col.predict(X_poly_col)
print(f"Polynomial predictions for column data ({X_poly_col.shape}):", poly_predictions_col)
print(f"Score for column data: {poly_model_col.score(X_poly_col, y_poly_col)}")


print("\n--- Testing Edge Cases ---")
# Empty data for fit
# For LinearRegression, X must be 2D, y must be 1D.
# Rust's ndarray might error on 0-sized dimensions for some operations (e.g. SVD, inv).
# The TypeError for dtype should be resolved, but other runtime errors from ndarray-linalg are possible.
X_empty_lin = np.empty((0, 1), dtype=np.float64) # 0 samples, 1 feature
y_empty_lin = np.array([], dtype=np.float64)

try:
    print("Fitting LinearRegression with empty data...")
    lin_model_empty = LinearRegression()
    lin_model_empty.fit(X_empty_lin, y_empty_lin)
    print("LinearRegression fit with empty data succeeded (coefficients/intercept might be None or specific values).")
    print(f"Coefficients: {lin_model_empty.coefficients}, Intercept: {lin_model_empty.intercept}")
    # Prediction/score might still fail or produce NaNs if model is ill-defined
    # print(f"Predict empty: {lin_model_empty.predict(X_empty_lin)}")
    # print(f"Score empty: {lin_model_empty.score(X_empty_lin, y_empty_lin)}")
except Exception as e:
    print(f"Caught expected error for LinearRegression empty data fit: {e}")

# Empty data for PolynomialRegression
X_empty_poly = np.array([], dtype=np.float64)
y_empty_poly = np.array([], dtype=np.float64)
try:
    print("Fitting PolynomialRegression with empty data...")
    poly_model_empty = PolynomialRegression(degree=2)
    poly_model_empty.fit(X_empty_poly, y_empty_poly)
    print("PolynomialRegression fit with empty data succeeded.")
    print(f"Coefficients: {poly_model_empty.coefficients}, Intercept: {poly_model_empty.intercept}")
    # Test predict and score with empty data if fit succeeds
    print(f"Predict empty poly: {poly_model_empty.predict(X_empty_poly)}") # Should return empty array
    print(f"Score empty poly: {poly_model_empty.score(X_empty_poly, y_empty_poly)}") # Behavior depends on Rust impl.
except Exception as e:
    print(f"Caught expected error for PolynomialRegression empty data fit: {e}")


# Mismatched shapes for fit (should raise error from Rust or Python checks)
try:
    print("Fitting LinearRegression with mismatched shapes (X (2,1), y (1,))...")
    lin_model_mismatch = LinearRegression()
    X_mismatch = np.array([[1],[2]], dtype=np.float64)
    y_mismatch = np.array([1], dtype=np.float64)
    lin_model_mismatch.fit(X_mismatch, y_mismatch)
except Exception as e:
    print(f"Caught expected error for LinearRegression mismatched shapes: {e}")

try:
    print("Fitting PolynomialRegression with mismatched shapes (X (2,), y (1,))...")
    poly_model_mismatch = PolynomialRegression(degree=2)
    X_mismatch_poly = np.array([1, 2], dtype=np.float64)
    y_mismatch_poly = np.array([1], dtype=np.float64)
    poly_model_mismatch.fit(X_mismatch_poly, y_mismatch_poly)
except Exception as e:
    print(f"Caught expected error for PolynomialRegression mismatched shapes: {e}")


# Predict before fit (should raise error from Rust)
try:
    print("Predicting with LinearRegression before fit...")
    lin_model_unfit = LinearRegression()
    lin_model_unfit.predict(X_lin) # X_lin is already float64
except Exception as e:
    print(f"Caught expected error for LinearRegression predict before fit: {e}")

try:
    print("Predicting with PolynomialRegression before fit...")
    poly_model_unfit = PolynomialRegression(degree=2)
    poly_model_unfit.predict(X_poly_simple) # X_poly_simple is already float64
except Exception as e:
    print(f"Caught expected error for PolynomialRegression predict before fit: {e}")

print("\nTests completed.")