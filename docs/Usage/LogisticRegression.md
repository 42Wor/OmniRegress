
# 🚀 Logistic Regression: Quick & Cool Guide  

## 🌟 Overview  
The `LogisticRegression` class provides an implementation of logistic regression for binary classification. Key features:  

- 🎯 Binary classification with probability outputs  
- 📊 Sigmoid function for probability estimation  
- ⚙️ Configurable learning rate and iterations  
- 🔍 Coefficient and intercept interpretation  

## 📝 Code Example  

```python
from omniregress import LogisticRegression

def test_logistic_regression():
    print("=== Testing Logistic Regression ===")

    # Binary classification data
    X = [[0.5], [1.0], [1.5], [2.0], [2.5], [3.0]]
    y = [0, 0, 0, 1, 1, 1]

    # Initialize and fit model
    model = LogisticRegression(learning_rate=0.1, max_iter=1000)
    model.fit(X, y)

    # Model parameters
    print(f"Coefficients: {model.coefficients}")
    print(f"Intercept: {model.intercept}")

    # Predictions
    test_X = [[1.0], [2.0], [3.5]]
    probabilities = model.predict_proba(test_X)
    predictions = model.predict(test_X)

    print(f"Probabilities: {probabilities}")
    print(f"Predictions: {predictions}")

    # Verification
    assert predictions[0] == 0, "Should predict class 0"
    assert predictions[1] == 1, "Should predict class 1"
    assert predictions[2] == 1, "Should predict class 1"
    assert all(0 <= p <= 1 for p in probabilities), "Invalid probabilities"

    print("Logistic Regression Test Passed!\n")
```

## 🔑 Key Features  

### 📈 Probability Estimation  
Uses sigmoid function to output probabilities between 0 and 1:  
`probability = 1 / (1 + e^(-(wx + b)))`  

### ⚙️ Hyperparameters  
- `learning_rate`: Controls step size in gradient descent  
- `max_iter`: Maximum number of training iterations  

### 📊 Outputs  
- `predict_proba()`: Returns class probabilities  
- `predict()`: Returns class labels (0/1)  

## 🧠 How It Works  
1. Initializes weights randomly  
2. Computes gradients using cross-entropy loss  
3. Updates weights via gradient descent  
4. Applies sigmoid activation for predictions  

## 🏆 When to Use  
✔️ Binary classification problems  
✔️ Probabilistic interpretation needed  
✔️ Linear decision boundary is sufficient  
