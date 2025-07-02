# omniregress/__init__.py
from .linear_regression import LinearRegression
from .polynomial_regression import PolynomialRegression
from .logistic_regression import LogisticRegression
from .ridge_regression import RidgeRegression
from .lasso_regression import LassoRegression

__version__ = "4.0.0"
__all__ = ['LinearRegression',
           'PolynomialRegression',
           'LogisticRegression',
           'RidgeRegression',
           'LassoRegression']