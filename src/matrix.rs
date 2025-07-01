// matrix.rs
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

pub fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if matrix.is_empty() {
        return vec![];
    }
    (0..matrix[0].len())
        .map(|i| matrix.iter().map(|row| row[i]).collect())
        .collect()
}

pub fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> PyResult<Vec<Vec<f64>>> {
    if a.is_empty() || b.is_empty() {
        return Ok(vec![]);
    }
    if a[0].len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "Matrix dimensions mismatch: {}x{} vs {}x{}",
            a.len(), a[0].len(), b.len(), b[0].len()
        )));
    }

    Ok((0..a.len())
        .map(|i| {
            (0..b[0].len())
                .map(|j| {
                    (0..a[0].len())
                        .map(|k| a[i][k] * b[k][j])
                        .sum()
                })
                .collect()
        })
        .collect())
}

pub fn invert(matrix: &[Vec<f64>]) -> PyResult<Vec<Vec<f64>>> {
    let n = matrix.len();
    if n == 0 {
        return Ok(vec![]);
    }
    let mut aug: Vec<Vec<f64>> = matrix
        .iter()
        .enumerate()
        .map(|(idx, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|i| if i == idx { 1.0 } else { 0.0 }));
            r
        })
        .collect();

    for i in 0..n {
        // Find the pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }
        if aug[max_row][i].abs() < 1e-12 {
            return Err(PyValueError::new_err("Matrix is singular"));
        }
        aug.swap(i, max_row);

        // Normalize the pivot row
        let pivot = aug[i][i];
        for j in 0..2 * n {
            aug[i][j] /= pivot;
        }

        // Eliminate other rows
        for k in 0..n {
            if k != i {
                let factor = aug[k][i];
                for j in 0..2 * n {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    Ok(aug.iter().map(|row| row[n..].to_vec()).collect())
}