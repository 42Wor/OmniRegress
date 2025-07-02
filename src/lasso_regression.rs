use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use ndarray::{Array1, Axis};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};

#[pyclass]
pub struct RustLassoRegression {
    coefficients: Option<Array1<f64>>,
    intercept: Option<f64>,
    alpha: f64,
    max_iter: usize,
    tol: f64,
}

#[pymethods]
impl RustLassoRegression {
    #[new]
    #[pyo3(signature = (alpha=1.0, max_iter=1000, tol=1e-4))]
    pub fn new(alpha: f64, max_iter: usize, tol: f64) -> Self {
        Self {
            coefficients: None,
            intercept: None,
            alpha,
            max_iter,
            tol,
        }
    }

    pub fn fit(
        &mut self,
        _py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let x_arr = x.as_array();
        let y_arr = y.as_array();

        if x_arr.is_empty() || y_arr.is_empty() {
            return Err(PyValueError::new_err("Empty input"));
        }

        if x_arr.shape()[0] != y_arr.shape()[0] {
            return Err(PyValueError::new_err(format!(
                "X and y must have same length ({} vs {})",
                x_arr.shape()[0],
                y_arr.shape()[0]
            )));
        }

        let n_samples = x_arr.shape()[0] as f64;
        let n_features = x_arr.shape()[1];

        // Center data
        let x_mean = x_arr.mean_axis(Axis(0)).unwrap();
        let y_mean = y_arr.mean().unwrap();

        let x_centered = &x_arr - &x_mean.broadcast(x_arr.raw_dim()).unwrap();
        let y_centered = &y_arr - y_mean;

        let mut coef = Array1::zeros(n_features);
        let intercept = y_mean;
        let mut prev_coef = coef.clone();

        for _ in 0..self.max_iter {
            for j in 0..n_features {
                // Calculate residual without j-th feature
                let mut r_j = y_centered.to_owned();
                for k in 0..n_features {
                    if k != j && coef[k] != 0.0 {
                        let xk = x_centered.column(k).to_owned();
                        r_j = r_j - &(coef[k] * &xk);
                    }
                }

                let x_j = x_centered.column(j).to_owned();
                let temp = x_j.dot(&r_j);
                let rho_j = temp / n_samples;

                let denom = x_j.dot(&x_j) / n_samples;
                if rho_j > self.alpha {
                    coef[j] = (rho_j - self.alpha) / denom;
                } else if rho_j < -self.alpha {
                    coef[j] = (rho_j + self.alpha) / denom;
                } else {
                    coef[j] = 0.0;
                }
            }

            let coef_diff = (&coef - &prev_coef).mapv(f64::abs).sum();
            if coef_diff < self.tol {
                break;
            }
            prev_coef = coef.clone();
        }

        self.coefficients = Some(coef);
        self.intercept = Some(intercept);
        Ok(())
    }

    pub fn predict(&self, py: Python<'_>, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
        let intercept = self
            .intercept
            .ok_or(PyValueError::new_err("Model not fitted"))?;
        let coef = self
            .coefficients
            .as_ref()
            .ok_or(PyValueError::new_err("Model not fitted"))?;

        let x_arr = x.as_array();
        if x_arr.shape()[1] != coef.len() {
            return Err(PyValueError::new_err(format!(
                "Input features don't match model coefficients ({} vs {})",
                x_arr.shape()[1],
                coef.len()
            )));
        }

        let y_pred = x_arr.dot(coef) + intercept;
        Ok(y_pred.into_pyarray_bound(py).into())
    }

    #[getter]
    pub fn coefficients(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        match &self.coefficients {
            Some(coef) => Ok(coef.clone().into_pyarray_bound(py).into()),
            None => Err(PyValueError::new_err("Model not fitted")),
        }
    }

    #[getter]
    pub fn intercept(&self) -> Option<f64> {
        self.intercept
    }

    #[getter]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    #[setter]
    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    #[getter]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    #[setter]
    pub fn set_max_iter(&mut self, max_iter: usize) {
        self.max_iter = max_iter;
    }

    #[getter]
    pub fn tol(&self) -> f64 {
        self.tol
    }

    #[setter]
    pub fn set_tol(&mut self, tol: f64) {
        self.tol = tol;
    }
}