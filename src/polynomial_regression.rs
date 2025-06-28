use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[pyclass]
pub struct RustPolynomialRegression {
    degree: usize,
    linear_model: super::linear_regression::RustLinearRegression,
}

#[pymethods]
impl RustPolynomialRegression {
    #[new]
    pub fn new(degree: usize) -> PyResult<Self> {
        if degree < 1 {
            return Err(PyValueError::new_err("Degree must be at least 1"));
        }
        Ok(Self {
            degree,
            linear_model: super::linear_regression::RustLinearRegression::new(),
        })
    }

    fn create_polynomial_features(&self, x: Vec<f64>) -> Vec<Vec<f64>> {
        x.into_iter()
            .map(|val| (1..=self.degree).map(|d| val.powi(d as i32)).collect())
            .collect()
    }

    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) -> PyResult<()> {
        let x_poly = self.create_polynomial_features(x);
        self.linear_model.fit(x_poly, y)
    }

    pub fn predict(&self, x: Vec<f64>) -> PyResult<Vec<f64>> {
        let x_poly = self.create_polynomial_features(x);
        self.linear_model.predict(x_poly)
    }

    pub fn score(&self, x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
        let x_poly = self.create_polynomial_features(x);
        let y_pred = self.linear_model.predict(x_poly.clone())?;

        let y_mean = y.iter().sum::<f64>() / y.len() as f64;
        let ss_total = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>();
        let ss_res = y
            .iter()
            .zip(y_pred.iter())
            .map(|(&yi, &fi)| (yi - fi).powi(2))
            .sum::<f64>();

        Ok(1.0 - (ss_res / ss_total))
    }

    #[getter]
    pub fn coefficients(&self) -> Option<Vec<f64>> {
        self.linear_model.coefficients()
    }

    #[getter]
    pub fn intercept(&self) -> Option<f64> {
        self.linear_model.intercept()
    }
}