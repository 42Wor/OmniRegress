# omniregress/__init__.py
from .linear_regression import LinearRegression
from .polynomial_regression import PolynomialRegression
from .logistic_regression import LogisticRegression

__version__ = "3.0.0"
__all__ = ['LinearRegression', 'PolynomialRegression', 'LogisticRegression']