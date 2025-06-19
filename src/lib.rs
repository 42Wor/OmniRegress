use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

mod matrix_ops {
    use super::*;

    pub fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if matrix.is_empty() { return vec![]; }
        (0..matrix[0].len())
            .map(|i| matrix.iter().map(|row| row[i]).collect())
            .collect()
    }

    pub fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> PyResult<Vec<Vec<f64>>> {
        if a.is_empty() || b.is_empty() { return Ok(vec![]); }
        if a[0].len() != b.len() {
            return Err(PyValueError::new_err(format!(
                "Matrix dimensions mismatch: {}x{} vs {}x{}",
                a.len(), a[0].len(), b.len(), b[0].len()
            )));
        }

        Ok((0..a.len())
            .map(|i| (0..b[0].len())
                .map(|j| a[i].iter().zip(b.iter()).map(|(x, row)| x * row[j]).sum())
                .collect())
            .collect())
    }

    pub fn invert(matrix: &[Vec<f64>]) -> PyResult<Vec<Vec<f64>>> {
        let n = matrix.len();
        if n == 0 { return Ok(vec![]); }

        let mut aug: Vec<Vec<f64>> = matrix.iter()
            .map(|row| {
                let mut r = row.clone();
                r.extend(vec![0.0; n]);
                r
            })
            .collect();

        for i in 0..n {
            aug[i][n + i] = 1.0;

            // Partial pivot
            let pivot_row = (i..n).max_by_key(|&k| aug[k][i].abs().to_bits()).unwrap();
            aug.swap(i, pivot_row);

            let pivot = aug[i][i];
            if pivot.abs() < 1e-12 {
                return Err(PyValueError::new_err("Matrix is singular"));
            }

            for j in i..2*n { aug[i][j] /= pivot; }
            for k in 0..n {
                if k != i {
                    let factor = aug[k][i];
                    for j in i..2*n { aug[k][j] -= factor * aug[i][j]; }
                }
            }
        }

        Ok(aug.iter().map(|row| row[n..].to_vec()).collect())
    }
}

#[pyclass]
struct _RustLinearRegressionInternal {
    coefficients: Option<Vec<f64>>,
    intercept: Option<f64>,
}

#[pymethods]
impl _RustLinearRegressionInternal {
    #[new]
    fn new() -> Self {
        Self { coefficients: None, intercept: None }
    }

    fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<()> {
        if x.is_empty() || y.is_empty() {
            return Err(PyValueError::new_err("Empty input"));
        }

        if x.len() != y.len() {
            return Err(PyValueError::new_err(format!(
                "X and y must have same length ({} vs {})",
                x.len(), y.len()
            )));
        }

        // Add intercept column
        let x_b: Vec<Vec<f64>> = x.iter()
            .map(|row| {
                let mut new_row = vec![1.0];
                new_row.extend(row);
                new_row
            })
            .collect();

        let x_t = matrix_ops::transpose(&x_b);
        let xtx = matrix_ops::matmul(&x_t, &x_b)?;
        let xtx_inv = matrix_ops::invert(&xtx)?;

        // Convert y to column vector format for matrix multiplication
        let y_col: Vec<Vec<f64>> = y.iter().map(|&val| vec![val]).collect();
        let xty = matrix_ops::matmul(&x_t, &y_col)?;
        let theta = matrix_ops::matmul(&xtx_inv, &xty)?;

        // Extract coefficients and intercept
        self.intercept = Some(theta[0][0]);
        self.coefficients = Some(theta.iter().skip(1).map(|row| row[0]).collect());
        Ok(())
    }

    fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        let intercept = self.intercept.ok_or(PyValueError::new_err("Model not fitted"))?;
        let coef = self.coefficients.as_ref().ok_or(PyValueError::new_err("Model not fitted"))?;

        if !x.is_empty() && x[0].len() != coef.len() {
            return Err(PyValueError::new_err(format!(
                "Input features don't match model coefficients ({} vs {})",
                x[0].len(), coef.len()
            )));
        }

        Ok(x.iter()
            .map(|row| intercept + row.iter().zip(coef).map(|(x, w)| x * w).sum::<f64>())
            .collect())
    }

    #[getter]
    fn coefficients(&self) -> Option<Vec<f64>> {
        self.coefficients.clone()
    }

    #[getter]
    fn intercept(&self) -> Option<f64> {
        self.intercept
    }
}

#[pymodule]
fn _omniregress(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<_RustLinearRegressionInternal>()?;
    Ok(())
}