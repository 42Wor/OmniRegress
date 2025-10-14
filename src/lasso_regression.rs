use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

#[pyclass]
pub struct RustLassoRegression {
    coefficients: Option<Vec<f64>>,
    intercept: Option<f64>,
    alpha: f64,
    max_iter: usize,
    tol: f64,
}

#[pymethods]
impl RustLassoRegression {
    #[new]
    #[pyo3(signature = (alpha = 1.0, max_iter = 1000, tol = 1e-4))]
    pub fn new(alpha: f64, max_iter: usize, tol: f64) -> Self {
        Self {
            coefficients: None,
            intercept: None,
            alpha,
            max_iter,
            tol,
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

        let n_samples = x.len();
        let n_features = if n_samples > 0 { x[0].len() } else { 0 };

        // Center the data for Lasso (we'll handle intercept separately)
        let y_mean = y.iter().sum::<f64>() / n_samples as f64;
        let y_centered: Vec<f64> = y.iter().map(|&val| val - y_mean).collect();

        // Center and scale X
        let mut x_centered = vec![vec![0.0; n_features]; n_samples];
        let mut x_std = vec![0.0; n_features];
        
        for j in 0..n_features {
            let col_mean: f64 = x.iter().map(|row| row[j]).sum::<f64>() / n_samples as f64;
            let col_std: f64 = (x.iter().map(|row| (row[j] - col_mean).powi(2)).sum::<f64>() / n_samples as f64).sqrt();
            
            x_std[j] = if col_std > 1e-12 { col_std } else { 1.0 };
            
            for i in 0..n_samples {
                x_centered[i][j] = (x[i][j] - col_mean) / x_std[j];
            }
        }

        // Initialize coefficients to zero
        let mut coef = vec![0.0; n_features];
        let mut prev_coef = coef.clone();

        // Coordinate descent algorithm for Lasso
        for iter in 0..self.max_iter {
            for j in 0..n_features {
                // Compute residual without feature j
                let mut r_j = vec![0.0; n_samples];
                for i in 0..n_samples {
                    r_j[i] = y_centered[i];
                    for k in 0..n_features {
                        if k != j {
                            r_j[i] -= x_centered[i][k] * coef[k];
                        }
                    }
                }

                // Compute ρ_j = X_j^T * r_j
                let rho_j: f64 = (0..n_samples)
                    .map(|i| x_centered[i][j] * r_j[i])
                    .sum();

                // Update coefficient using soft thresholding
                if rho_j < -self.alpha {
                    coef[j] = (rho_j + self.alpha) / n_samples as f64;
                } else if rho_j > self.alpha {
                    coef[j] = (rho_j - self.alpha) / n_samples as f64;
                } else {
                    coef[j] = 0.0;
                }
            }

            // Check convergence
            let max_change = coef.iter()
                .zip(&prev_coef)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f64, f64::max);

            if max_change < self.tol {
                break;
            }

            prev_coef = coef.clone();

            if iter == self.max_iter - 1 {
                println!("Lasso regression did not converge after {} iterations", self.max_iter);
            }
        }

        // Rescale coefficients back to original scale
        let mut rescaled_coef = vec![0.0; n_features];
        for j in 0..n_features {
            rescaled_coef[j] = coef[j] / x_std[j];
        }

        // Compute intercept
        let mut x_mean = vec![0.0; n_features];
        for j in 0..n_features {
            x_mean[j] = x.iter().map(|row| row[j]).sum::<f64>() / n_samples as f64;
        }

        let intercept = y_mean - (0..n_features)
            .map(|j| x_mean[j] * rescaled_coef[j])
            .sum::<f64>();

        self.coefficients = Some(rescaled_coef);
        self.intercept = Some(intercept);
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