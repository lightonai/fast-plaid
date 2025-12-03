use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Converts an `anyhow::Error` into a Python `PyValueError`.
pub fn anyhow_to_pyerr(err: anyhow::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}
