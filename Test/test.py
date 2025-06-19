from omniregress import LinearRegression
import numpy as np

X_np = np.array([[1], [2], [3], [4], [5.5]])
y_np = np.array([2.1, 3.9, 6.1, 7.9, 11.2])

model = LinearRegression()
model.fit(X_np, y_np)
print("Coefficients:", model.coef_)  # Changed from coefficients to coef_
print("Intercept:", model.intercept_)  # Changed from intercept to intercept_
predictions = model.predict(np.array([[6], [7]]))
print("Predictions:", predictions)
print("Score:", model.score(X_np, y_np))

# Test with Python lists
X_list = [[1], [2], [3]]
y_list = [2, 4, 5.9]
model_list = LinearRegression()
model_list.fit(X_list, y_list)
print("\nList Coeffs:", model_list.coef_)
print("List Intercept:", model_list.intercept_)