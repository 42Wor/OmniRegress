# Test/test1.py
from omniregress import LinearRegression, PolynomialRegression # Assuming PolynomialRegression is also imported
import numpy as np

# --- Original Tests (kept for continuity) ---
print("\n--- Testing Linear Regression (Single Feature) ---")
X_lin_single = np.array([[1], [2], [3], [4], [5.5]], dtype=np.float64)
y_lin_single = np.array([2.1, 3.9, 6.1, 7.9, 11.2], dtype=np.float64)

lin_model_single = LinearRegression()
print("Fitting single-feature linear model...")
lin_model_single.fit(X_lin_single, y_lin_single)

print("Single-feature Linear coefficients (from property):", lin_model_single.coef_)
print("Single-feature Linear intercept (from property):", lin_model_single.intercept_)

lin_predictions_single = lin_model_single.predict(X_lin_single)
print("Single-feature Linear predictions:", lin_predictions_single)
lin_score_single = lin_model_single.score(X_lin_single, y_lin_single)
print("Single-feature Linear R^2 score:", lin_score_single)

X_new_lin_single = np.array([[6], [7]], dtype=np.float64)
print("Single-feature Linear predictions for new data:", lin_model_single.predict(X_new_lin_single))

# --- Testing Multiple Features for Linear Regression ---
print("\n--- Testing Multiple Features for Linear Regression ---")
# y = 1.0 + 2*x1 + 3*x2
X_multi = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3, 3]], dtype=np.float64)
y_multi = np.array([1.0 + 2*1 + 3*1,   # 6
                    1.0 + 2*1 + 3*2,   # 9
                    1.0 + 2*2 + 3*2,   # 11
                    1.0 + 2*2 + 3*3,   # 14
                    1.0 + 2*3 + 3*3],  # 16
                   dtype=np.float64)   # y_multi = [6, 9, 11, 14, 16]

lin_model_multi = LinearRegression()
print("Fitting multi-feature linear model...")
lin_model_multi.fit(X_multi, y_multi)

print(f"Multi-feature Linear coefficients (expected approx [2, 3]): {lin_model_multi.coef_}")
print(f"Multi-feature Linear intercept (expected approx 1.0): {lin_model_multi.intercept_}")

# Check number of coefficients
if lin_model_multi.coef_ is not None:
    print(f"Number of coefficients: {len(lin_model_multi.coef_)} (expected {X_multi.shape[1]})")
else:
    print("Coefficients are None after multi-feature fit.")


lin_predictions_multi = lin_model_multi.predict(X_multi)
print("Multi-feature Linear predictions (compare with y_multi):", lin_predictions_multi)
lin_score_multi = lin_model_multi.score(X_multi, y_multi)
print("Multi-feature Linear R^2 score (expected close to 1.0):", lin_score_multi)

# Test with new multi-feature data
X_new_multi = np.array([[3, 1], [1, 3]], dtype=np.float64)
# Expected predictions for X_new_multi:
# y1 = 1.0 + 2*3 + 3*1 = 1 + 6 + 3 = 10
# y2 = 1.0 + 2*1 + 3*3 = 1 + 2 + 9 = 12
expected_new_multi_preds = np.array([10, 12], dtype=np.float64)
print("Multi-feature Linear predictions for new data:", lin_model_multi.predict(X_new_multi))
print("Expected new data predictions:", expected_new_multi_preds)


# --- Testing Zero Features for Linear Regression ---
print("\n--- Testing Zero Features for Linear Regression ---")
# Model should only learn the intercept, which should be the mean of y
X_zero_feat = np.empty((5, 0), dtype=np.float64) # 5 samples, 0 features
y_zero_feat = np.array([10, 12, 10, 13, 10], dtype=np.float64)
expected_intercept_zero_feat = np.mean(y_zero_feat) # Should be 11.0

lin_model_zero_feat = LinearRegression()
print("Fitting zero-feature linear model...")
lin_model_zero_feat.fit(X_zero_feat, y_zero_feat)

print(f"Zero-feature Linear coefficients (expected empty or None): {lin_model_zero_feat.coef_}")
print(f"Zero-feature Linear intercept (expected approx {expected_intercept_zero_feat}): {lin_model_zero_feat.intercept_}")

# Check characteristics of zero-feature model
if lin_model_zero_feat.coef_ is not None:
    print(f"Length of coefficients: {len(lin_model_zero_feat.coef_)} (expected 0)")
else:
    print("Coefficients are None for zero-feature model (as expected if it becomes an empty list).")


# Predictions should all be the intercept
zero_feat_predictions = lin_model_zero_feat.predict(X_zero_feat)
print("Zero-feature Predictions (all should be intercept):", zero_feat_predictions)
expected_zero_feat_preds = np.full_like(y_zero_feat, expected_intercept_zero_feat, dtype=np.float64)
print("Expected zero-feature predictions:", expected_zero_feat_preds)

# Test predict with new zero-feature data
X_new_zero_feat = np.empty((2, 0), dtype=np.float64)
print("Zero-feature predictions for new data:", lin_model_zero_feat.predict(X_new_zero_feat))

zero_feat_score = lin_model_zero_feat.score(X_zero_feat, y_zero_feat)
print("Zero-feature R^2 score (expected 0.0, as model explains variance only via mean):", zero_feat_score)
# R^2 for a model predicting only the mean is 0.

# --- Original Polynomial Regression Tests (kept for continuity) ---
print("\n--- Testing Polynomial Regression (Single Feature Input) ---")
X_poly_simple = np.array([1, 2, 3, 4, 5], dtype=np.float64)
y_poly_simple = np.array([2, 7, 16, 29, 48], dtype=np.float64)

poly_model = PolynomialRegression(degree=2)
print("Fitting polynomial model (degree=2)...")
poly_model.fit(X_poly_simple, y_poly_simple)

print("Polynomial coefficients (from internal linear model):", poly_model.coefficients)
print("Polynomial intercept (from internal linear model):", poly_model.intercept)

poly_predictions = poly_model.predict(X_poly_simple)
print("Polynomial predictions for training data:", poly_predictions)

X_new_poly = np.array([6, 7], dtype=np.float64)
print("Polynomial predictions for new data [6, 7]:", poly_model.predict(X_new_poly))

X_poly_col = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
y_poly_col = np.array([2, 7, 16, 29, 48], dtype=np.float64)
poly_model_col = PolynomialRegression(degree=2)
print("Fitting polynomial model (degree=2) with column vector X...")
poly_model_col.fit(X_poly_col, y_poly_col)
poly_predictions_col = poly_model_col.predict(X_poly_col)
print(f"Polynomial predictions for column data ({X_poly_col.shape}):", poly_predictions_col)
print(f"Score for column data: {poly_model_col.score(X_poly_col, y_poly_col)}")


# --- Original Edge Case Tests (kept for continuity) ---
print("\n--- Testing Original Edge Cases ---")
X_empty_lin = np.empty((0, 1), dtype=np.float64)
y_empty_lin = np.array([], dtype=np.float64)
try:
    print("Fitting LinearRegression with empty sample data (0 samples, 1 feature)...")
    lin_model_empty = LinearRegression()
    lin_model_empty.fit(X_empty_lin, y_empty_lin)
    print("LinearRegression fit with empty samples succeeded.")
    print(f"Coefficients: {lin_model_empty.coefficients}, Intercept: {lin_model_empty.intercept}")
    print(f"Predict empty: {lin_model_empty.predict(X_empty_lin)}")
    # Score might be tricky, could be 1.0, 0.0 or NaN, or error.
    # print(f"Score empty: {lin_model_empty.score(X_empty_lin, y_empty_lin)}")
except Exception as e:
    print(f"Caught expected error for LinearRegression empty sample data fit: {e}")

X_empty_poly = np.array([], dtype=np.float64)
y_empty_poly = np.array([], dtype=np.float64)
try:
    print("Fitting PolynomialRegression with empty sample data...")
    poly_model_empty = PolynomialRegression(degree=2)
    poly_model_empty.fit(X_empty_poly, y_empty_poly)
    print("PolynomialRegression fit with empty samples succeeded.")
    print(f"Coefficients: {poly_model_empty.coefficients}, Intercept: {poly_model_empty.intercept}")
    print(f"Predict empty poly: {poly_model_empty.predict(X_empty_poly)}")
    # print(f"Score empty poly: {poly_model_empty.score(X_empty_poly, y_empty_poly)}")
except Exception as e:
    # The Python wrapper for PolynomialRegression might raise before Rust if X is empty and _create_polynomial_features is called.
    # Or Rust might complain about "Input X (x_input) cannot be empty." if empty list is passed.
    print(f"Caught expected error for PolynomialRegression empty sample data fit: {e}")

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
    print(f"Caught error for PolynomialRegression mismatched shapes: {e}")

try:
    print("Predicting with LinearRegression before fit...")
    lin_model_unfit = LinearRegression()
    lin_model_unfit.predict(X_lin_single)
except Exception as e:
    print(f"Caught expected error for LinearRegression predict before fit: {e}")

try:
    print("Predicting with PolynomialRegression before fit...")
    poly_model_unfit = PolynomialRegression(degree=2)
    poly_model_unfit.predict(X_poly_simple)
except Exception as e:
    print(f"Caught expected error for PolynomialRegression predict before fit: {e}")

print("\nAll Tests completed.")