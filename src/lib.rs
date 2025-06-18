use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use ndarray::{Array1, Array2, Axis, s};
use ndarray_linalg::{Inverse, SVD, error::LinalgError};

#[pyclass]
#[derive(Debug)]
pub struct LinearRegression {
    coefficients: Option<Array1<f64>>,
    intercept: Option<f64>,
}

// Helper function to convert ndarray errors to PyErr
fn to_py_err<E: std::fmt::Display>(err: E) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(err.to_string())
}

#[pymethods]
impl LinearRegression {
    #[new]
    pub fn new() -> Self {
        LinearRegression {
            coefficients: None,
            intercept: None,
        }
    }

    pub fn fit<'py>(
        &mut self,
        _py: Python<'py>,
        x: &Bound<'py, PyArray2<f64>>,
        y: &Bound<'py, PyArray1<f64>>,
    ) -> PyResult<()> {
        // Safe because we're just viewing the data
        let x = unsafe { x.as_array() };
        let y = unsafe { y.as_array() };

        // Add intercept term
        let ones = Array2::ones((x.nrows(), 1));
        let x_with_intercept = ndarray::concatenate(Axis(1), &[ones.view(), x.view()])
            .map_err(to_py_err)?;

        // Solve using normal equations or SVD fallback
        let xt = x_with_intercept.t();
        let xtx = xt.dot(&x_with_intercept);

        let theta = match xtx.inv() {
            Ok(xtx_inv) => xtx_inv.dot(&xt).dot(&y),
            Err(_) => {
                // SVD fallback for singular matrices
                let (u, s, vt) = x_with_intercept.svd(true, true).map_err(to_py_err)?;
                let u = u.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("SVD failed to compute U matrix"))?;
                let vt = vt.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("SVD failed to compute Vt matrix"))?;

                let s_inv = Array1::from_iter(s.iter().map(|&val| {
                    if val > 1e-10 { 1.0 / val } else { 0.0 }
                }));
                let pseudo_inv = vt.t().dot(&Array2::from_diag(&s_inv).dot(&u.t()));
                pseudo_inv.dot(&y)
            }
        };

        self.intercept = Some(theta[0]);
        self.coefficients = Some(theta.slice_move(s![1..]));
        Ok(())
    }

    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray2<f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match (&self.coefficients, &self.intercept) {
            (Some(coeff), Some(intercept)) => {
                // Safe because we're just viewing the data
                let x_arr = unsafe { x.as_array() };
                let y_pred = x_arr.dot(coeff) + *intercept;
                Ok(y_pred.to_pyarray_bound(py))
            }
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() first",
            )),
        }
    }

    pub fn score<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray2<f64>>,
        y: &Bound<'py, PyArray1<f64>>,
    ) -> PyResult<f64> {
        let y_pred = self.predict(py, x)?;
        // Safe because we're just viewing the data
        let y_true = unsafe { y.as_array() };
        let y_pred = unsafe { y_pred.as_array() };

        let ss_res = (&y_true - &y_pred).mapv(|x| x.powi(2)).sum();
        let y_mean = y_true.mean().unwrap();
        let ss_tot = y_true.mapv(|x| (x - y_mean).powi(2)).sum();

        Ok(1.0 - (ss_res / ss_tot))
    }

    #[getter]
    pub fn coefficients(&self) -> PyResult<Option<Vec<f64>>> {
        Ok(self.coefficients.as_ref().map(|c| c.to_vec()))
    }

    #[getter]
    pub fn intercept(&self) -> PyResult<Option<f64>> {
        Ok(self.intercept)
    }
}

#[pymodule]
fn _omniregress(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<LinearRegression>()?;
    Ok(())
}