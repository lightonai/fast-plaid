// Local library modules.
pub mod index;
pub mod search;
pub mod utils;

// External crate imports.
use anyhow::anyhow;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use tch::Kind;

// Standard library imports.
use std::ffi::CString;

// Conditional imports for cross-platform dynamic library loading.
#[cfg(windows)]
use winapi::um::errhandlingapi::GetLastError;
#[cfg(windows)]
use winapi::um::libloaderapi::LoadLibraryA;

// Internal module imports.
use crate::index::create::create_index;
use crate::index::delete::delete_from_index;
use crate::index::update::update_index;
use search::load::{construct_index, get_device, PyLoadedIndex};
use search::search::{search_many, QueryResult, SearchParameters};
use utils::embeddings::reconstruct_embeddings;
use utils::errors::anyhow_to_pyerr;

/// Dynamically loads the native Torch shared library (libtorch).
///
/// This function is necessary to ensure that the `tch` crate can find and
/// link to the PyTorch library at runtime, especially when distributed
/// via Python packages where the exact path isn't known at compile time.
///
/// It uses `dlopen` on Unix-like systems and `LoadLibraryA` on Windows.
///
/// # Arguments
///
/// * `torch_path` - The absolute path to the `libtorch` shared library
///   (e.g., `libtorch.so`, `libtorch.dylib`, or `torch.dll`).
///
/// # Errors
///
/// Returns an `anyhow::Error` if the library fails to load, providing
/// details from `dlerror` (Unix) or `GetLastError` (Windows).
fn call_torch(torch_path: String) -> Result<(), anyhow::Error> {
    let torch_path_cstr = CString::new(torch_path.clone())
        .map_err(|e| anyhow!("Failed to create CString for libtorch path: {}", e))?;

    #[cfg(unix)]
    {
        let handle = unsafe { libc::dlopen(torch_path_cstr.as_ptr(), libc::RTLD_LAZY) };
        if handle.is_null() {
            return Err(anyhow!(
                "Failed to load Torch library '{}' via dlopen. Check the path and permissions.",
                torch_path
            ));
        }
    }

    #[cfg(windows)]
    {
        let handle = unsafe { LoadLibraryA(torch_path_cstr.as_ptr()) };
        if handle.is_null() {
            let error_code = unsafe { GetLastError() };
            return Err(anyhow!(
                "Failed to load Torch library '{}' via LoadLibraryA. Windows error code: {}",
                torch_path,
                error_code
            ));
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        return Err(anyhow!(
            "Dynamic library loading is not supported on this operating system."
        ));
    }

    Ok(())
}

/// Manually initializes and loads the libtorch shared library.
///
/// This function is called automatically by other functions in this module,
/// but can be called explicitly if needed.
///
/// Args:
///     torch_path (str): The absolute path to the `libtorch` shared library
///         (e.g., `libtorch.so` or `torch.dll`).
///
/// Raises:
///     RuntimeError: If the torch library fails to load.
#[pyfunction]
fn initialize_torch(_py: Python<'_>, torch_path: String) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize Torch: {}", e)))
}

/// Creates and saves a new FastPlaid index.
///
/// This is the low-level Rust implementation called by `FastPlaid.create()`.
/// It's generally recommended to use the `FastPlaid` class wrapper instead.
///
/// This function takes pre-computed centroids and embeddings, quantizes the
/// embeddings using the centroids, and saves the index files to the specified directory.
///
/// Args:
///     index (str): The file path to the directory to save the index.
///     torch_path (str): Path to the `libtorch` shared library.
///     device (str): Device to use for computation (e.g., "cpu", "cuda:0").
///     embedding_dim (int): The dimension of the token embeddings (e.g., 128).
///     nbits (int): Number of bits for product quantization (e.g., 4).
///     embeddings (list[torch.Tensor]): A list of 2D tensors
///         (num_tokens, embedding_dim), one for each document.
///     centroids (torch.Tensor): A 2D tensor of (num_centroids, embedding_dim)
///         pre-computed by K-means on the Python side.
///     batch_size (int): Batch size for processing embeddings during creation.
///     seed (int | None): Optional seed for reproducible index creation.
///
/// Raises:
///     RuntimeError: If index creation fails or `libtorch` fails to load.
#[pyfunction]
fn create(
    _py: Python<'_>,
    index: String,
    torch_path: String,
    device: String,
    embedding_dim: i64,
    nbits: i64,
    embeddings: Vec<PyTensor>,
    centroids: PyTensor,
    batch_size: i64,
    seed: Option<u64>,
) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;

    let device = get_device(&device)?;
    let centroids = centroids.to_device(device).to_kind(Kind::Half);

    let result = create_index(
        &embeddings,
        &index,
        embedding_dim,
        nbits,
        device,
        centroids,
        batch_size,
        seed,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create index: {}", e)));

    result
}

/// Performs a multi-vector search on a loaded index.
///
/// This is the low-level Rust implementation called by `FastPlaid.search()`.
/// It's generally recommended to use the `FastPlaid` class wrapper instead.
///
/// Unlike previous versions, this function does not load the index from disk/cache.
/// It accepts a `PyLoadedIndex` object that is managed by the Python side.
///
/// Args:
///     index (PyLoadedIndex): A reference to the loaded index object containing
///         IVF structures and codebooks.
///     device (str): Device to perform the search on (e.g., "cpu", "cuda:0").
///     queries_embeddings (torch.Tensor): A 3D tensor of query embeddings
///         with shape (num_queries, num_query_tokens, embedding_dim).
///     search_parameters (SearchParameters): A SearchParameters object
///         containing `top_k`, `n_ivf_probe`, etc.
///     show_progress (bool): Whether to display a progress bar during search.
///     subset (list[list[int]] | None): An optional filter to restrict the
///         search. Must be a list of lists, where each inner list contains
///         the document IDs to search for that specific query.
///
/// Returns:
///     list[QueryResult]: A list of `QueryResult` objects, one for each query.
///
/// Raises:
///     RuntimeError: If searching fails.
///     ValueError: If search parameters are invalid.
#[pyfunction]
fn pysearch(
    py: Python<'_>,
    index: &PyLoadedIndex,
    device: String,
    queries_embeddings: PyTensor,
    search_parameters: &SearchParameters,
    show_progress: bool,
    subset: Option<Vec<Vec<i64>>>,
) -> PyResult<Vec<QueryResult>> {
    let device_tch = get_device(&device)?;
    let params = search_parameters.clone();
    let index_inner = &index.inner;

    // Release the GIL to allow for parallel execution in Python threads.
    let results = py
        .allow_threads(move || {
            search_many(
                &queries_embeddings,
                index_inner,
                &params,
                device_tch,
                show_progress,
                subset,
            )
        })
        .map_err(anyhow_to_pyerr)?;

    Ok(results)
}

/// Adds new documents to an existing FastPlaid index.
///
/// This is the low-level Rust implementation called by `FastPlaid.update()`.
/// It's generally recommended to use the `FastPlaid` class wrapper instead.
///
/// This function appends new quantized embeddings to the index files on disk.
/// It uses the provided `index` object (which contains the current centroids
/// and codecs) to quantize the new data without recalculating centroids.
///
/// Args:
///     index_path (str): The file path to the directory where the index is stored.
///         New data will be appended to files in this directory.
///     index (PyLoadedIndex): The loaded index object. This is required to access
///         the existing codebooks/centroids for quantizing the new embeddings.
///     torch_path (str): Path to the `libtorch` shared library.
///     device (str): Device to use for computation (e.g., "cpu", "cuda:0").
///     embeddings (list[torch.Tensor]): A list of 2D tensors
///         (num_tokens, embedding_dim), one for each new document.
///     batch_size (int): Batch size for processing new embeddings.
///     update_threshold_centroids (bool | None): Whether to update the quantization
///         residual threshold based on the new data. Defaults to False.
///
/// Raises:
///     RuntimeError: If updating the index fails or `libtorch` fails to load.
#[pyfunction]
fn update(
    _py: Python<'_>,
    index_path: String,
    index: &PyLoadedIndex,
    torch_path: String,
    device: String,
    embeddings: Vec<PyTensor>,
    batch_size: i64,
    update_threshold_centroids: Option<bool>,
) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;

    let device_tch = get_device(&device)?;

    update_index(
        &embeddings,
        &index_path,
        device_tch,
        batch_size,
        &index.inner,
        update_threshold_centroids.unwrap_or(false),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to update index: {}", e)))?;

    Ok(())
}

/// Deletes documents from an existing FastPlaid index.
///
/// This is the low-level Rust implementation called by `FastPlaid.delete()`.
/// It's generally recommended to use the `FastPlaid` class wrapper instead.
///
/// This function removes the specified document IDs from the index files.
/// The remaining documents are re-indexed to maintain sequential IDs.
///
/// Args:
///     index (str): The file path to the directory of the existing index.
///     torch_path (str): Path to the `libtorch` shared library.
///     device (str): Device to use for the operation (e.g., "cpu", "cuda:0").
///     subset (list[int]): A list of document IDs to delete. IDs correspond
///         to the original insertion order.
///
/// Raises:
///     RuntimeError: If deletion fails or `libtorch` fails to load.
#[pyfunction]
fn delete(
    _py: Python<'_>,
    index: String,
    torch_path: String,
    device: String,
    subset: Vec<i64>,
) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;

    let device = get_device(&device)?;

    let result = delete_from_index(&subset, &index, device)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete from index: {}", e)));

    result
}

#[pymodule]
#[pyo3(name = "fast_plaid_rust")]
fn python_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchParameters>()?;
    m.add_class::<QueryResult>()?;
    m.add_class::<PyLoadedIndex>()?;

    m.add_function(wrap_pyfunction!(initialize_torch, m)?)?;
    m.add_function(wrap_pyfunction!(construct_index, m)?)?;
    m.add_function(wrap_pyfunction!(create, m)?)?;
    m.add_function(wrap_pyfunction!(pysearch, m)?)?;
    m.add_function(wrap_pyfunction!(update, m)?)?;
    m.add_function(wrap_pyfunction!(delete, m)?)?;
    m.add_function(wrap_pyfunction!(reconstruct_embeddings, m)?)?;
    Ok(())
}
