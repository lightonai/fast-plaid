// Local library modules.
pub mod index;
pub mod search;
pub mod utils;

// External crate imports.
use anyhow::anyhow;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use tch::{Device, Kind};

// Standard library imports.
use std::ffi::CString;

// Internal module imports.
use crate::index::create::create_index;
use search::load::load_index;
use search::search::{search_index, QueryResult, SearchParameters};

/// Converts a Rust `anyhow::Error` into a Python-compatible `PyErr`.
///
/// This utility function acts as a bridge between Rust's flexible `anyhow` error
/// handling and the error types expected by the PyO3 framework.
fn anyhow_to_pyerr(err: anyhow::Error) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Dynamically loads the native Torch shared library (e.g., `libtorch.so`).
///
/// This is a workaround to ensure Torch's symbols are available in memory,
/// which can prevent linking errors when `tch-rs` is used within a
/// Python extension module.
fn call_torch(torch_path: String) -> Result<(), anyhow::Error> {
    let torch_path_cstr = CString::new(torch_path)
        .map_err(|e| anyhow!("Failed to create CString for libtorch path: {}", e))?;

    // The library handle is intentionally ignored. We only need to load it
    // into the process memory space.
    let _lib_handle = unsafe { libc::dlopen(torch_path_cstr.as_ptr(), libc::RTLD_LAZY) };
    Ok(())
}

/// Parses a string identifier into a `tch::Device`.
///
/// Supports simple device strings like "cpu", "cuda", and indexed CUDA devices
/// such as "cuda:0".
fn get_device(device: &str) -> Result<Device, PyErr> {
    match device.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda(0)), // Default to the first CUDA device.
        s if s.starts_with("cuda:") => {
            let parts: Vec<&str> = s.split(':').collect();
            if parts.len() == 2 {
                parts[1].parse::<usize>().map(Device::Cuda).map_err(|_| {
                    PyValueError::new_err(format!("Invalid CUDA device index: '{}'", parts[1]))
                })
            } else {
                Err(PyValueError::new_err(
                    "Invalid CUDA device format. Expected 'cuda:N'.",
                ))
            }
        },
        _ => Err(PyValueError::new_err(format!(
            "Unsupported device string: '{}'",
            device
        ))),
    }
}

/// Pre-loads the native Torch library from a specified path.
///
/// Call this function once at the start of a Python script if you encounter
/// linking issues with the Torch library, which can occur in complex deployment
/// environments.
///
/// Args:
///     torch_path (str): The file path to the Torch shared library,
///         e.g., `/path/to/libtorch_cuda.so`.
#[pyfunction]
fn initialize_torch(_py: Python<'_>, torch_path: String) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize Torch: {}", e)))
}

/// Creates and saves a new FastPlaid index to disk.
///
/// This function processes document embeddings, clusters them using the provided
/// centroids, calculates quantization residuals, and serializes the complete
/// index structure to the specified directory.
///
/// Args:
///     index (str): The directory path where the new index will be saved.
///     torch_path (str): Path to the Torch shared library (e.g., `libtorch.so`).
///     device (str): The compute device to use for index creation (e.g., "cpu", "cuda:0").
///     embedding_dim (int): The dimensionality of the embeddings.
///     embeddings (list[torch.Tensor]): A list of 2D tensors, where each tensor
///         is a batch of document embeddings.
///     centroids (torch.Tensor): A 2D tensor of shape `[num_centroids, embedding_dim]`
///         used for vector quantization.
#[pyfunction]
fn create(
    _py: Python<'_>,
    index: String,
    torch_path: String,
    device: String,
    embedding_dim: i64,
    embeddings: Vec<PyTensor>,
    centroids: PyTensor,
) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;

    let device = get_device(&device)?;
    let centroids = centroids.to_device(device).to_kind(Kind::Float);

    create_index(&embeddings, &index, embedding_dim, 4, device, centroids)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create index: {}", e)))
}

/// Loads an index and performs a search in a single, high-level operation.
///
/// This is the primary entry point for searching. It handles loading the index
/// from disk, moving it to the specified device, executing the search with the
/// given queries, and returning the results.
///
/// Args:
///     index (str): Path to the directory containing the pre-built index.
///     torch_path (str): Path to the Torch shared library (e.g., `libtorch.so`).
///     device (str): The compute device for the search (e.g., "cpu", "cuda:0").
///     queries_embeddings (torch.Tensor): A 2D tensor of query embeddings with
///         shape `[num_queries, embedding_dim]`.
///     search_parameters (SearchParameters): A configuration object specifying
///         search behavior, such as `k` and `nprobe`.
///
/// Returns:
///     list[QueryResult]: A list of result objects, each containing the
///     `doc_id` and `score` for a retrieved document.
#[pyfunction]
fn load_and_search(
    _py: Python<'_>,
    index: String,
    torch_path: String,
    device: String,
    queries_embeddings: PyTensor,
    search_parameters: &SearchParameters,
    show_progress: bool,
) -> PyResult<Vec<QueryResult>> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;

    let device = get_device(&device)?;
    let loaded_index = load_index(&index, device).map_err(anyhow_to_pyerr)?;
    let query_tensor = queries_embeddings.to_device(Device::Cpu);

    let results = search_index(
        &query_tensor,
        &loaded_index,
        search_parameters,
        device,
        show_progress,
    )
    .map_err(anyhow_to_pyerr)?;

    Ok(results)
}

/// A high-performance document retrieval toolkit using a ColBERT-style late
/// interaction model, implemented in Rust with Python bindings.
///
/// This module provides functions for creating and searching indexes, along with
/// the necessary data classes `SearchParameters` and `QueryResult` to interact
/// with the library from Python.
#[pymodule]
#[pyo3(name = "fast_plaid_rust")]
fn python_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add data classes required for the Python interface.
    m.add_class::<SearchParameters>()?;
    m.add_class::<QueryResult>()?;

    // Add functions to the Python module.
    m.add_function(wrap_pyfunction!(initialize_torch, m)?)?;
    m.add_function(wrap_pyfunction!(create, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_search, m)?)?;

    Ok(())
}
