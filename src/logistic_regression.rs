use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use std::f64::consts::E;

#[pyclass]
pub struct RustLogisticRegression {
    coefficients: Option<Vec<f64>>,
    intercept: Option<f64>,
    learning_rate: f64,
    max_iter: usize,
    tol: f64,
}

#[pymethods]
impl RustLogisticRegression {
    #[new]
    #[pyo3(signature = (learning_rate=0.01, max_iter=1000, tol=1e-4))]
    pub fn new(learning_rate: f64, max_iter: usize, tol: f64) -> PyResult<Self> {
        if learning_rate <= 0.0 {
            return Err(PyValueError::new_err("Learning rate must be positive"));
        }
        if max_iter == 0 {
            return Err(PyValueError::new_err("Max iterations must be at least 1"));
        }
        if tol <= 0.0 {
            return Err(PyValueError::new_err("Tolerance must be positive"));
        }

        Ok(Self {
            coefficients: None,
            intercept: None,
            learning_rate,
            max_iter,
            tol,
        })
    }

    fn sigmoid(&self, z: f64) -> f64 {
        1.0 / (1.0 + E.powf(-z))
    }

    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        if x.is_empty() || y.is_empty() {
            return Err(PyValueError::new_err("Empty input"));
        }

        if x.len() != y.len() {
            return Err(PyValueError::new_err("X and y must have same length"));
        }

        let n_features = x[0].len();
        let mut weights = vec![0.0; n_features];
        let mut bias = 0.0;

        for _ in 0..self.max_iter {
            let mut grad_weights = vec![0.0; n_features];
            let mut grad_bias = 0.0;

            for (row, &target) in x.iter().zip(y.iter()) {
                let z = bias + row.iter().zip(weights.iter()).map(|(&x, &w)| x * w).sum::<f64>();
                let prediction = self.sigmoid(z);
                let error = prediction - target;

                grad_bias += error;
                for (i, &feature) in row.iter().enumerate() {
                    grad_weights[i] += error * feature;
                }
            }

            // Update parameters
            let m = x.len() as f64;
            bias -= self.learning_rate * grad_bias / m;
            for i in 0..n_features {
                weights[i] -= self.learning_rate * grad_weights[i] / m;
            }

            // Check convergence
            let grad_norm = grad_bias.powi(2) + grad_weights.iter().map(|&g| g.powi(2)).sum::<f64>();
            if grad_norm.sqrt() < self.tol {
                break;
            }
        }

        self.coefficients = Some(weights);
        self.intercept = Some(bias);
        Ok(())
    }

    pub fn predict_proba(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let intercept = self
            .intercept
            .ok_or(PyValueError::new_err("Model not fitted"))?;
        let coef = self
            .coefficients
            .as_ref()
            .ok_or(PyValueError::new_err("Model not fitted"))?;

        if !x.is_empty() && x[0].len() != coef.len() {
            return Err(PyValueError::new_err("Input features don't match model coefficients"));
        }

        Ok(x
            .iter()
            .map(|row| {
                let z = intercept + row.iter().zip(coef.iter()).map(|(&x, &w)| x * w).sum::<f64>();
                self.sigmoid(z)
            })
            .collect())
    }

    #[pyo3(signature = (x, threshold=None))]
    pub fn predict(&self, x: Vec<Vec<f64>>, threshold: Option<f64>) -> PyResult<Vec<f64>> {
        let probabilities = self.predict_proba(x)?;
        let threshold = threshold.unwrap_or(0.5);
        Ok(probabilities
            .into_iter()
            .map(|p| if p >= threshold { 1.0 } else { 0.0 })
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