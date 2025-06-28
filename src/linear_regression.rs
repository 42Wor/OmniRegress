use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use crate::matrix;

#[pyclass]
pub struct RustLinearRegression {
    coefficients: Option<Vec<f64>>,
    intercept: Option<f64>,
}

#[pymethods]
impl RustLinearRegression {
    #[new]
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: None,
        }
    }

    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        if x.is_empty() || y.is_empty() {
            return Err(PyValueError::new_err("Empty input"));
        }

        if x.len() != y.len() {
            return Err(PyValueError::new_err(format!(
                "X and y must have same length ({} vs {})",
                x.len(),
                y.len()
            )));
        }

        // Add intercept column
        let x_b: Vec<Vec<f64>> = x
            .iter()
            .map(|row| {
                let mut new_row = vec![1.0];
                new_row.extend(row);
                new_row
            })
            .collect();

        let x_t = matrix::transpose(&x_b);
        let xtx = matrix::matmul(&x_t, &x_b)?;
        let xtx_inv = matrix::invert(&xtx)?;

        // Convert y to column vector format
        let y_col: Vec<Vec<f64>> = y.iter().map(|&val| vec![val]).collect();
        let xty = matrix::matmul(&x_t, &y_col)?;
        let theta = matrix::matmul(&xtx_inv, &xty)?;

        // Extract coefficients and intercept
        self.intercept = Some(theta[0][0]);
        self.coefficients = Some(theta.iter().skip(1).map(|row| row[0]).collect());
        Ok(())
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let intercept = self
            .intercept
            .ok_or(PyValueError::new_err("Model not fitted"))?;
        let coef = self
            .coefficients
            .as_ref()
            .ok_or(PyValueError::new_err("Model not fitted"))?;

        if !x.is_empty() && x[0].len() != coef.len() {
            return Err(PyValueError::new_err(format!(
                "Input features don't match model coefficients ({} vs {})",
                x[0].len(),
                coef.len()
            )));
        }

        Ok(x.iter()
            .map(|row| intercept + row.iter().zip(coef).map(|(x, w)| x * w).sum::<f64>())
            .collect())
    }

    #[getter]
    pub fn coefficients(&self) -> Option<Vec<f64>> {
        self.coefficients.clone()
    }

    #[getter]
    pub fn intercept(&self) -> Option<f64> {
        self.intercept
    }
}
