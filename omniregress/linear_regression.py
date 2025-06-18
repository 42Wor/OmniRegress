# omniregress/linear_regression.py
"""
Linear Regression module.
This module provides the LinearRegression class, implemented in Rust for performance.
"""
try:
    # Import the Rust implementation from the .so/.pyd file
    # The compiled Rust extension will be named omniregress_rs and placed
    # within the 'omniregress' package by Maturin.
    from ._omniregress import LinearRegression as _RustLinearRegression

    # Re-export with the desired name
    LinearRegression = _RustLinearRegression

    # Optionally, copy over docstrings or set __module__ if needed,
    # though PyO3 usually handles docstrings well.
    # LinearRegression.__module__ = __name__ # Makes help() show omniregress.linear_regression

except ImportError as e:
    # This error handling is crucial for users if the Rust extension fails to build/import.
    raise ImportError(
        "Could not import the Rust-based LinearRegression. "
        "Please ensure that the package was compiled and installed correctly. "
        f"Original error: {e}"
    ) from e

# You can add any Python-specific enhancements or utility functions related to
# LinearRegression here if necessary.