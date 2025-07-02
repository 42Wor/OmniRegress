use pyo3::prelude::*;
mod matrix;
mod linear_regression;
mod polynomial_regression;
mod logistic_regression;
mod ridge_regression;
mod lasso_regression;

pub use linear_regression::RustLinearRegression;
pub use polynomial_regression::RustPolynomialRegression;
pub use logistic_regression::RustLogisticRegression;
pub use ridge_regression::RustRidgeRegression;
pub use lasso_regression::RustLassoRegression;

#[pymodule]
fn _omniregress(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustLinearRegression>()?;
    m.add_class::<RustPolynomialRegression>()?;
    m.add_class::<RustLogisticRegression>()?;
    m.add_class::<RustRidgeRegression>()?;
    m.add_class::<RustLassoRegression>()?;
    Ok(())
}