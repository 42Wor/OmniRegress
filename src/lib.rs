// src/lib.rs
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

// --- Matrix Utilities (from scratch) ---
// This module is identical to the one provided in the previous "from scratch" answer
// that used Vec<Vec<f64>> and did not depend on ndarray.
// It includes: transpose, multiply_matrices, multiply_matrix_vector, invert_matrix.
mod matrix_ops {
    // Transposes a matrix (Vec<Vec<f64>>)
    pub fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        if matrix.is_empty() {
            return Vec::new();
        }
        let rows = matrix.len();
        let cols = matrix.get(0).map_or(0, |row| row.len());
        if cols == 0 {
             return vec![vec![0.0; rows]; cols]; // Transpose of Rx0 is 0xR
        }

        let mut transposed = vec![vec![0.0; rows]; cols];
        for i in 0..rows {
            for j in 0..cols {
                if j < matrix[i].len() { // Check bounds for potentially jagged input
                    transposed[j][i] = matrix[i][j];
                } else {
                    // This case implies a jagged array, which should ideally be caught earlier.
                    // For now, let's assume valid rectangular input based on Python wrapper's job.
                    // Or, return an error here. For simplicity, assume rectangular.
                }
            }
        }
        transposed
    }

    // Multiplies two matrices (m1 * m2)
    pub fn multiply_matrices(
        m1: &Vec<Vec<f64>>,
        m2: &Vec<Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>, String> {
        let m1_rows = m1.len();
        let m1_cols = if m1_rows == 0 { 0 } else { m1.get(0).map_or(0, |r| r.len()) };

        let m2_rows = m2.len();
        let m2_cols = if m2_rows == 0 { 0 } else { m2.get(0).map_or(0, |r| r.len()) };

        if m1_cols != m2_rows {
            return Err(format!(
                "Matrix dimensions mismatch for multiplication: M1 is {}x{}, M2 is {}x{}. Inner dimensions ({}, {}) must match.",
                m1_rows, m1_cols, m2_rows, m2_cols, m1_cols, m2_rows
            ));
        }
        if m1_rows == 0 || m2_cols == 0 { // Handle multiplication resulting in empty matrix
            return Ok(vec![vec![0.0; m2_cols]; m1_rows]);
        }

        let mut result = vec![vec![0.0; m2_cols]; m1_rows];
        for i in 0..m1_rows {
            for j in 0..m2_cols {
                let mut sum = 0.0;
                for k in 0..m1_cols {
                    sum += m1[i][k] * m2[k][j];
                }
                result[i][j] = sum;
            }
        }
        Ok(result)
    }

    // Multiplies a matrix by a vector (matrix * vector)
    pub fn multiply_matrix_vector(
        matrix: &Vec<Vec<f64>>,
        vector: &Vec<f64>,
    ) -> Result<Vec<f64>, String> {
        let m_rows = matrix.len();
        let m_cols = if m_rows == 0 { 0 } else { matrix.get(0).map_or(0, |r| r.len()) };
        let v_len = vector.len();

        if m_cols != v_len {
            return Err(format!(
                "Matrix-vector dimensions mismatch: Matrix is {}x{}, Vector has length {}. Inner dimensions ({}, {}) must match.",
                m_rows, m_cols, v_len, m_cols, v_len
            ));
        }
        if m_rows == 0 { // Handle multiplication resulting in empty vector
            return Ok(Vec::new());
        }

        let mut result = vec![0.0; m_rows];
        for i in 0..m_rows {
            let mut sum = 0.0;
            for j in 0..m_cols {
                sum += matrix[i][j] * vector[j];
            }
            result[i] = sum;
        }
        Ok(result)
    }

    // Inverts a square matrix using Gauss-Jordan elimination
    pub fn invert_matrix(matrix: &Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, String> {
        if matrix.is_empty() {
            return Ok(Vec::new());
        }
        let n = matrix.len();
        if n == 0 { return Ok(Vec::new()); }

        for row in matrix.iter() {
            if row.len() != n {
                return Err("Matrix must be square for inversion.".to_string());
            }
        }

        let mut aug = Vec::with_capacity(n);
        for i in 0..n {
            let mut current_row_data = matrix[i].clone();
            let mut identity_part = vec![0.0; n];
            identity_part[i] = 1.0;
            current_row_data.extend(identity_part);
            aug.push(current_row_data);
        }

        for i in 0..n {
            let mut pivot_row_idx = i;
            for k in i + 1..n {
                if aug[k][i].abs() > aug[pivot_row_idx][i].abs() {
                    pivot_row_idx = k;
                }
            }
            aug.swap(i, pivot_row_idx);

            let pivot_val = aug[i][i];
            if pivot_val.abs() < 1e-12 {
                return Err("Matrix is singular or numerically unstable; cannot be inverted.".to_string());
            }

            for j in i..2 * n {
                aug[i][j] /= pivot_val;
            }

            for k in 0..n {
                if k != i {
                    let factor = aug[k][i];
                    for j in i..2 * n {
                        aug[k][j] -= factor * aug[i][j];
                    }
                }
            }
        }

        let mut inverse = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                inverse[i][j] = aug[i][j + n];
            }
        }
        Ok(inverse)
    }
}


fn to_py_err(err_msg: String) -> PyErr {
    PyValueError::new_err(err_msg)
}

#[pyclass(name = "_RustLinearRegressionInternal")] // Renamed to avoid clash with Python wrapper
struct RustLinearRegressionInternal {
    coefficients: Option<Vec<f64>>,
    intercept: Option<f64>,
    fitted: bool,
}

#[pymethods]
impl RustLinearRegressionInternal {
    #[new]
    pub fn new() -> Self {
        RustLinearRegressionInternal {
            coefficients: None,
            intercept: None,
            fitted: false,
        }
    }

    // Accepts Python list of lists of floats, and list of floats
    pub fn fit(
        &mut self,
        x_input: Vec<Vec<f64>>, // PyO3 converts Python list[list[float]]
        y_input: Vec<f64>,      // PyO3 converts Python list[float]
    ) -> PyResult<()> {
        if x_input.is_empty() {
            return Err(to_py_err("Input X (x_input) cannot be empty.".to_string()));
        }
        let num_samples = x_input.len();
        let num_features = x_input[0].len(); // Assume rectangular, checked by Python wrapper

        if y_input.is_empty() {
            return Err(to_py_err("Input y (y_input) cannot be empty.".to_string()));
        }
        if num_samples != y_input.len() {
            return Err(to_py_err(format!(
                "Number of samples in X ({}) must match y ({}).",
                num_samples,
                y_input.len()
            )));
        }
        // Validate all rows in x_input have num_features
        for (i, row) in x_input.iter().enumerate() {
            if row.len() != num_features {
                return Err(to_py_err(format!(
                    "Input X must be rectangular. Row {} has {} features, expected {}.",
                    i, row.len(), num_features
                )));
            }
        }


        let mut x_b = Vec::with_capacity(num_samples);
        for row_vec in x_input.iter() {
            let mut new_row_with_intercept = vec![1.0];
            new_row_with_intercept.extend(row_vec.iter().cloned());
            x_b.push(new_row_with_intercept);
        }

        let x_b_t = matrix_ops::transpose(&x_b);
        let xtx = matrix_ops::multiply_matrices(&x_b_t, &x_b).map_err(to_py_err)?;
        let xtx_inv = matrix_ops::invert_matrix(&xtx).map_err(to_py_err)?;
        let xty = matrix_ops::multiply_matrix_vector(&x_b_t, &y_input).map_err(to_py_err)?;

        let theta = matrix_ops::multiply_matrix_vector(&xtx_inv, &xty).map_err(to_py_err)?;

        if theta.is_empty() || theta.len() != (num_features + 1) {
             return Err(to_py_err(format!(
                "Failed to compute valid coefficients. Expected {} coefficients (incl. intercept), got {}.",
                num_features + 1, theta.len()
            )));
        }

        self.intercept = Some(theta[0]);
        self.coefficients = Some(theta[1..].to_vec());
        self.fitted = true;
        Ok(())
    }

    // Accepts Python list of lists of floats, returns Python list of floats
    pub fn predict(
        &self,
        x_input: Vec<Vec<f64>>, // PyO3 converts Python list[list[float]]
    ) -> PyResult<Vec<f64>> {    // PyO3 converts Rust Vec<f64> to Python list[float]
        if !self.fitted {
            return Err(PyValueError::new_err("Model not fitted. Call fit() first."));
        }
        if x_input.is_empty() {
            return Ok(Vec::new()); // Predict on empty input returns empty list
        }

        let intercept = self.intercept.unwrap();
        let coefficients = self.coefficients.as_ref().unwrap();

        let num_features_model = coefficients.len();
        let num_features_input = x_input[0].len(); // Assume rectangular

        if num_features_input != num_features_model {
            return Err(to_py_err(format!(
                "Number of features in input X for prediction ({}) does not match fitted model ({}).",
                num_features_input, num_features_model
            )));
        }
        // Validate all rows in x_input have num_features_input
        for (i, row) in x_input.iter().enumerate() {
            if row.len() != num_features_input {
                return Err(to_py_err(format!(
                    "Input X for predict must be rectangular. Row {} has {} features, expected {}.",
                    i, row.len(), num_features_input
                )));
            }
        }


        let mut predictions = Vec::with_capacity(x_input.len());
        for row_vec in x_input.iter() {
            let mut prediction = intercept;
            for (i, val) in row_vec.iter().enumerate() {
                prediction += coefficients[i] * val;
            }
            predictions.push(prediction);
        }
        Ok(predictions)
    }

    #[getter]
    pub fn get_coefficients(&self) -> PyResult<Option<Vec<f64>>> { // Renamed to avoid clash if Python wrapper uses 'coefficients' property
        Ok(self.coefficients.clone())
    }

    #[getter]
    pub fn get_intercept(&self) -> PyResult<Option<f64>> { // Renamed
        Ok(self.intercept)
    }
}

#[pymodule]
fn _omniregress(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<RustLinearRegressionInternal>()?;
    Ok(())
}